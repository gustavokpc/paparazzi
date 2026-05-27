/*
 * nn_e2e_control.c
 * Paparazzi wrapper for an end-to-end NN controller.
 *
 * IMPORTANT:
 * - This file builds the 19-element input expected by nn_parameters.h:
 *   dx, dy, dz, vx, vy, vz, phi, theta, psi, p, q, r,
 *   Mx_ext, My_ext, Mz_ext, omega1, omega2, omega3, omega4
 * - dx,dy,dz are target-relative: waypoint - current_position. Therefore,
 *   in the NN's reference frame the target is always (0,0,0), exactly like
 *   in the papers/training setup.
 * - nn_control() returns normalized motor commands in [0,1]. This wrapper also
 *   exposes PPRZ and RPM-scaled copies for command laws and telemetry.
 * - Motor commands are applied from the airframe command_laws through
 *   nn_e2e_control_get_motor_pprz(), so arming and motor order stay visible
 *   in the airframe file.
 */

#include "modules/nn_e2e_control/nn_e2e_control.h"
#include "modules/nn_e2e_control/nn_operations.h"

#include "state.h"
#include "generated/airframe.h"
#include "generated/flight_plan.h"
#include "modules/actuators/motor_mixing.h"
#include "modules/nav/waypoints.h"

#include <math.h>
#include <stdbool.h>
#include <stdint.h>

extern int16_t actuators_pprz[] __attribute__((weak));

#ifndef NN_E2E_REACHED_RADIUS_M
#define NN_E2E_REACHED_RADIUS_M 0.6f
#endif

#ifndef NN_E2E_TARGET_ALT_M
#define NN_E2E_TARGET_ALT_M 1.0f
#endif

#ifndef NN_E2E_START_WAYPOINT_INDEX
#define NN_E2E_START_WAYPOINT_INDEX 1
#endif

/* Tune these to match your trained output scaling and your Bebop model limits. */
#ifndef NN_E2E_MIN_RPM
#define NN_E2E_MIN_RPM 5000.0f
#endif

#ifndef NN_E2E_MAX_RPM
#define NN_E2E_MAX_RPM 10000.0f
#endif

#ifndef NN_E2E_MIN_PPRZ
#define NN_E2E_MIN_PPRZ 5000
#endif

#ifndef NN_E2E_MAX_PPRZ
#define NN_E2E_MAX_PPRZ 10000
#endif

#ifndef NN_E2E_MOTOR_0_SRC
#define NN_E2E_MOTOR_0_SRC 0
#endif

#ifndef NN_E2E_MOTOR_1_SRC
#define NN_E2E_MOTOR_1_SRC 1
#endif

#ifndef NN_E2E_MOTOR_2_SRC
#define NN_E2E_MOTOR_2_SRC 2
#endif

#ifndef NN_E2E_MOTOR_3_SRC
#define NN_E2E_MOTOR_3_SRC 3
#endif

#ifndef NN_E2E_MOTOR_0_INVERT
#define NN_E2E_MOTOR_0_INVERT 0
#endif

#ifndef NN_E2E_MOTOR_1_INVERT
#define NN_E2E_MOTOR_1_INVERT 0
#endif

#ifndef NN_E2E_MOTOR_2_INVERT
#define NN_E2E_MOTOR_2_INVERT 0
#endif

#ifndef NN_E2E_MOTOR_3_INVERT
#define NN_E2E_MOTOR_3_INVERT 0
#endif

#ifndef NN_E2E_USE_NED_INPUT
#define NN_E2E_USE_NED_INPUT 1
#endif

typedef struct {
  float x;
  float y;
  float z;
} nn_vec3_t;

typedef struct {
  float phi;
  float theta;
  float psi;
} nn_euler_t;

bool nn_e2e_control_enabled = false;
unsigned int nn_e2e_control_waypoint_index = 0;
float nn_e2e_control_last_norm[4] = {0.f, 0.f, 0.f, 0.f};
float nn_e2e_control_last_rpm[4] = {0.f, 0.f, 0.f, 0.f};
int32_t nn_e2e_control_motor_pprz[4] = {0, 0, 0, 0};
int32_t nn_e2e_control_applied_pprz[4] = {0, 0, 0, 0};
int32_t nn_e2e_control_raw_mean_pprz = 0;
unsigned int nn_e2e_control_periodic_count = 0U;
float nn_e2e_control_reached_radius_m = NN_E2E_REACHED_RADIUS_M;
int32_t nn_e2e_control_output_min_pprz = NN_E2E_MIN_PPRZ;
int32_t nn_e2e_control_output_max_pprz = NN_E2E_MAX_PPRZ;
bool nn_e2e_control_use_ned_input = NN_E2E_USE_NED_INPUT;
uint8_t nn_e2e_control_motor_src[4] = {
  NN_E2E_MOTOR_0_SRC,
  NN_E2E_MOTOR_1_SRC,
  NN_E2E_MOTOR_2_SRC,
  NN_E2E_MOTOR_3_SRC
};
bool nn_e2e_control_motor_invert[4] = {
  NN_E2E_MOTOR_0_INVERT,
  NN_E2E_MOTOR_1_INVERT,
  NN_E2E_MOTOR_2_INVERT,
  NN_E2E_MOTOR_3_INVERT
};
float nn_e2e_control_target[3] = {0.f, 0.f, 0.f};
float nn_e2e_control_error[3] = {0.f, 0.f, 0.f};
float nn_e2e_control_input_error[3] = {0.f, 0.f, 0.f};
float nn_e2e_control_input_velocity[3] = {0.f, 0.f, 0.f};
float nn_e2e_control_dist_to_target = 0.f;

/* Square in local coordinates. Keep it inside the Cyberzoo geofence. */
static const nn_vec3_t nn_square_waypoints[] = {
  { 1.0f,  1.0f, NN_E2E_TARGET_ALT_M},
  {-1.0f,  1.0f, NN_E2E_TARGET_ALT_M},
  {-1.0f, -1.0f, NN_E2E_TARGET_ALT_M},
  { 1.0f, -1.0f, NN_E2E_TARGET_ALT_M},
};
static const unsigned int nn_num_square_waypoints = sizeof(nn_square_waypoints) / sizeof(nn_square_waypoints[0]);

static nn_vec3_t get_square_waypoint(unsigned int index)
{
#if defined(WP_NN_SQ_1) && defined(WP_NN_SQ_2) && defined(WP_NN_SQ_3) && defined(WP_NN_SQ_4)
  static const uint8_t wp_ids[] = {WP_NN_SQ_1, WP_NN_SQ_2, WP_NN_SQ_3, WP_NN_SQ_4};
  const uint8_t wp_id = wp_ids[index % (sizeof(wp_ids) / sizeof(wp_ids[0]))];
  return (nn_vec3_t){waypoint_get_x(wp_id), waypoint_get_y(wp_id), waypoint_get_alt(wp_id)};
#else
  return nn_square_waypoints[index % nn_num_square_waypoints];
#endif
}

__attribute__((weak)) nn_vec3_t nn_e2e_get_position_m(void)
{
  const struct EnuCoor_f *pos = stateGetPositionEnu_f();
  return (nn_vec3_t){pos->x, pos->y, pos->z};
}

__attribute__((weak)) nn_vec3_t nn_e2e_get_velocity_mps(void)
{
  const struct EnuCoor_f *vel = stateGetSpeedEnu_f();
  return (nn_vec3_t){vel->x, vel->y, vel->z};
}

__attribute__((weak)) nn_euler_t nn_e2e_get_attitude_rad(void)
{
  const struct FloatEulers *att = stateGetNedToBodyEulers_f();
  return (nn_euler_t){att->phi, att->theta, att->psi};
}

__attribute__((weak)) nn_vec3_t nn_e2e_get_body_rates_radps(void)
{
  const struct FloatRates *rates = stateGetBodyRates_f();
  return (nn_vec3_t){rates->p, rates->q, rates->r};
}

__attribute__((weak)) nn_vec3_t nn_e2e_get_external_moment_nm(void)
{
  /* If unavailable onboard, keep zeros or estimate them exactly as in training. */
  return (nn_vec3_t){0.0f, 0.0f, 0.0f};
}

__attribute__((weak)) void nn_e2e_get_motor_feedback_rpm(float omega_rpm[4])
{
  /* If you do not have motor RPM feedback, use previous command as a fallback. */
  omega_rpm[0] = nn_e2e_control_last_rpm[0];
  omega_rpm[1] = nn_e2e_control_last_rpm[1];
  omega_rpm[2] = nn_e2e_control_last_rpm[2];
  omega_rpm[3] = nn_e2e_control_last_rpm[3];
}

static float clampf(float x, float lo, float hi)
{
  if (x < lo) return lo;
  if (x > hi) return hi;
  return x;
}

static int32_t normalized_to_pprz(float normalized)
{
  int32_t output_min = nn_e2e_control_output_min_pprz;
  int32_t output_max = nn_e2e_control_output_max_pprz;
  if (output_max < output_min) {
    const int32_t tmp = output_min;
    output_min = output_max;
    output_max = tmp;
  }
  normalized = clampf(normalized, 0.f, 1.f);
  return (int32_t)(output_min + normalized * (float)(output_max - output_min));
}

static int32_t clamp_motor_command(int32_t command)
{
  int32_t output_min = nn_e2e_control_output_min_pprz;
  int32_t output_max = nn_e2e_control_output_max_pprz;
  if (output_max < output_min) {
    const int32_t tmp = output_min;
    output_min = output_max;
    output_max = tmp;
  }
  if (command < output_min) {
    return output_min;
  }
  if (command > output_max) {
    return output_max;
  }
  return command;
}

static nn_vec3_t enu_to_ned_vec(const nn_vec3_t enu)
{
  return (nn_vec3_t){enu.y, enu.x, -enu.z};
}

static unsigned int motor_src_for_servo(uint8_t motor_idx)
{
  return nn_e2e_control_motor_src[motor_idx] % 4U;
}

static bool motor_invert_for_servo(uint8_t motor_idx)
{
  return nn_e2e_control_motor_invert[motor_idx];
}

static int32_t maybe_invert_pprz(int32_t command)
{
  int32_t output_min = nn_e2e_control_output_min_pprz;
  int32_t output_max = nn_e2e_control_output_max_pprz;
  if (output_max < output_min) {
    const int32_t tmp = output_min;
    output_min = output_max;
    output_max = tmp;
  }
  return output_min + output_max - command;
}

static int32_t mapped_raw_motor_pprz(uint8_t motor_idx)
{
  const unsigned int src = motor_src_for_servo(motor_idx);
  int32_t command = nn_e2e_control_motor_pprz[src];
  if (motor_invert_for_servo(motor_idx)) {
    command = maybe_invert_pprz(command);
  }
  return command;
}

static int32_t mean_mapped_raw_motor_pprz(void)
{
  int32_t sum = 0;
  for (uint8_t i = 0U; i < 4U; i++) {
    sum += mapped_raw_motor_pprz(i);
  }
  nn_e2e_control_raw_mean_pprz = sum / 4;
  return nn_e2e_control_raw_mean_pprz;
}

static int32_t direct_nn_motor_pprz(uint8_t motor_idx)
{
  mean_mapped_raw_motor_pprz();
  return clamp_motor_command(mapped_raw_motor_pprz(motor_idx));
}

static float waypoint_dist2(const nn_vec3_t pos, const nn_vec3_t wp)
{
  const float dx = wp.x - pos.x;
  const float dy = wp.y - pos.y;
  const float dz = wp.z - pos.z;
  return dx * dx + dy * dy + dz * dz;
}

static void update_target_debug(const nn_vec3_t pos, const nn_vec3_t wp)
{
  nn_e2e_control_target[0] = wp.x;
  nn_e2e_control_target[1] = wp.y;
  nn_e2e_control_target[2] = wp.z;

  nn_e2e_control_error[0] = wp.x - pos.x;
  nn_e2e_control_error[1] = wp.y - pos.y;
  nn_e2e_control_error[2] = wp.z - pos.z;
  nn_e2e_control_dist_to_target = sqrtf(waypoint_dist2(pos, wp));
}

static void reset_target_debug(void)
{
  for (unsigned int i = 0; i < 3U; i++) {
    nn_e2e_control_target[i] = 0.f;
    nn_e2e_control_error[i] = 0.f;
    nn_e2e_control_input_error[i] = 0.f;
    nn_e2e_control_input_velocity[i] = 0.f;
  }
  nn_e2e_control_dist_to_target = 0.f;
}

static void maybe_advance_waypoint(const nn_vec3_t pos)
{
  const nn_vec3_t wp = get_square_waypoint(nn_e2e_control_waypoint_index);
  const float d2 = waypoint_dist2(pos, wp);
  const float reached_radius = nn_e2e_control_reached_radius_m > 0.f ? nn_e2e_control_reached_radius_m : 0.05f;
  if (d2 < reached_radius * reached_radius) {
    nn_e2e_control_waypoint_index = (nn_e2e_control_waypoint_index + 1U) % nn_num_square_waypoints;
    nn_reset();  /* Optional but usually cleaner when changing target. */
  }
  update_target_debug(pos, get_square_waypoint(nn_e2e_control_waypoint_index));
}

void nn_e2e_control_init(void)
{
  nn_e2e_control_enabled = false;
  nn_e2e_control_waypoint_index = 0U;
  nn_e2e_control_periodic_count = 0U;
  for (unsigned int i = 0; i < 4U; i++) {
    nn_e2e_control_last_norm[i] = 0.f;
    nn_e2e_control_last_rpm[i] = 0.f;
    nn_e2e_control_motor_pprz[i] = 0;
    nn_e2e_control_applied_pprz[i] = 0;
  }
  nn_e2e_control_raw_mean_pprz = 0;
  reset_target_debug();
  nn_reset();
}

void nn_e2e_control_start(void)
{
  nn_e2e_control_enabled = true;
  nn_e2e_control_periodic_count = 0U;
  for (unsigned int i = 0; i < 4U; i++) {
    nn_e2e_control_last_norm[i] = 0.f;
    nn_e2e_control_last_rpm[i] = NN_E2E_MIN_RPM;
    nn_e2e_control_motor_pprz[i] = nn_e2e_control_output_min_pprz;
    nn_e2e_control_applied_pprz[i] = nn_e2e_control_output_min_pprz;
  }
  nn_e2e_control_raw_mean_pprz = 0;
  nn_e2e_control_waypoint_index = NN_E2E_START_WAYPOINT_INDEX % nn_num_square_waypoints;
  update_target_debug(nn_e2e_get_position_m(), get_square_waypoint(nn_e2e_control_waypoint_index));
  nn_reset();
}

void nn_e2e_control_stop(void)
{
  nn_e2e_control_enabled = false;
  for (unsigned int i = 0; i < 4U; i++) {
    nn_e2e_control_last_norm[i] = 0.f;
    nn_e2e_control_last_rpm[i] = 0.f;
    nn_e2e_control_motor_pprz[i] = 0;
    nn_e2e_control_applied_pprz[i] = 0;
  }
  nn_e2e_control_raw_mean_pprz = 0;
  reset_target_debug();
  nn_reset();
}

void nn_e2e_control_periodic(void)
{
  if (!nn_e2e_control_enabled) {
    return;
  }
  nn_e2e_control_periodic_count++;

  const nn_vec3_t pos = nn_e2e_get_position_m();
  maybe_advance_waypoint(pos);

  const nn_vec3_t vel = nn_e2e_get_velocity_mps();
  const nn_vec3_t err = {
    nn_e2e_control_error[0],
    nn_e2e_control_error[1],
    nn_e2e_control_error[2]
  };
  const nn_vec3_t nn_err = nn_e2e_control_use_ned_input ? enu_to_ned_vec(err) : err;
  const nn_vec3_t nn_vel = nn_e2e_control_use_ned_input ? enu_to_ned_vec(vel) : vel;
  nn_e2e_control_input_error[0] = nn_err.x;
  nn_e2e_control_input_error[1] = nn_err.y;
  nn_e2e_control_input_error[2] = nn_err.z;
  nn_e2e_control_input_velocity[0] = nn_vel.x;
  nn_e2e_control_input_velocity[1] = nn_vel.y;
  nn_e2e_control_input_velocity[2] = nn_vel.z;

  const nn_euler_t att = nn_e2e_get_attitude_rad();
  const nn_vec3_t rates = nn_e2e_get_body_rates_radps();
  const nn_vec3_t mext = nn_e2e_get_external_moment_nm();
  float omega[4];
  nn_e2e_get_motor_feedback_rpm(omega);

  float state[NUM_STATES] = {
    nn_err.x, nn_err.y, nn_err.z,
    nn_vel.x, nn_vel.y, nn_vel.z,
    att.phi, att.theta, att.psi,
    rates.x, rates.y, rates.z,
    mext.x, mext.y, mext.z,
    omega[0], omega[1], omega[2], omega[3]
  };

  float normalized_cmd[NUM_CONTROLS];
  nn_control(state, normalized_cmd);

  for (unsigned int i = 0; i < 4U; i++) {
    nn_e2e_control_last_norm[i] = clampf(normalized_cmd[i], 0.f, 1.f);
    nn_e2e_control_last_rpm[i] = NN_E2E_MIN_RPM + nn_e2e_control_last_norm[i] * (NN_E2E_MAX_RPM - NN_E2E_MIN_RPM);
    nn_e2e_control_motor_pprz[i] = normalized_to_pprz(nn_e2e_control_last_norm[i]);
  }
}

int32_t nn_e2e_control_get_motor_pprz(uint8_t motor_idx, int32_t autopilot_pprz)
{
  if (!nn_e2e_control_enabled || motor_idx >= 4U) {
    if (motor_idx < 4U) {
      nn_e2e_control_applied_pprz[motor_idx] = autopilot_pprz;
    }
    return autopilot_pprz;
  }

  const int32_t command = direct_nn_motor_pprz(motor_idx);

  nn_e2e_control_applied_pprz[motor_idx] = command;
#ifdef MOTOR_MIXING_NB_MOTOR
  if (motor_idx < MOTOR_MIXING_NB_MOTOR) {
    motor_mixing.commands[motor_idx] = command;
  }
#endif
  if (actuators_pprz != NULL) {
    actuators_pprz[motor_idx] = (int16_t)command;
  }
  return command;
}
