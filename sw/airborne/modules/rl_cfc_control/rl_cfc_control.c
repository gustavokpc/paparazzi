/*
 * rl_cfc_control.c
 * Paparazzi wrapper for the recurrent PPO CfC baseline controller.
 *
 * IMPORTANT:
 * - This file builds the 20-element raw RL observation expected by
 *   rl_cfc_parameters.h:
 *   pos_G, vel_G, euler_B_to_G, rates_B, motor_state, next_gate_G.
 * - The RL policy returns actions in [-1,1]. In the training environment these
 *   actions command steady-state motor speed in rad/s. The Bebop actuator driver
 *   ultimately sends reference motor speed in RPM, while Paparazzi command laws
 *   pass values through the generic PPRZ actuator scaling.
 * - Motor commands are applied from the airframe command_laws through
 *   rl_cfc_control_get_motor_pprz(); motor mapping constants live here.
 */

#include "modules/rl_cfc_control/rl_cfc_control.h"
#include "modules/rl_cfc_control/rl_cfc_operations.h"

#include "paparazzi.h"
#include "state.h"
#include "generated/airframe.h"
#include "generated/flight_plan.h"
#include "modules/actuators/motor_mixing.h"
#include "modules/nav/waypoints.h"

#ifdef BOARD_BEBOP
#include "boards/bebop/actuators.h"
#endif

#include <math.h>
#include <stdbool.h>
#include <stdint.h>

extern int16_t actuators_pprz[] __attribute__((weak));

#ifndef RL_CFC_REACHED_RADIUS_M
#define RL_CFC_REACHED_RADIUS_M 0.6f
#endif

#ifndef RL_CFC_TARGET_ALT_M
#define RL_CFC_TARGET_ALT_M 1.0f
#endif

#ifndef RL_CFC_START_WAYPOINT_INDEX
#define RL_CFC_START_WAYPOINT_INDEX 0
#endif

/* RL motor model training scale. The policy/env use rad/s, Bebop feedback uses RPM. */
#ifndef RL_CFC_MIN_RADPS
#define RL_CFC_MIN_RADPS 238.49f
#endif

#ifndef RL_CFC_MAX_RADPS
#define RL_CFC_MAX_RADPS 3295.50f
#endif

#define RL_CFC_RADPS_TO_RPM (60.0f / (2.0f * (float)M_PI))
#define RL_CFC_RPM_TO_RADPS ((2.0f * (float)M_PI) / 60.0f)

#ifndef RL_CFC_MIN_RPM
#define RL_CFC_MIN_RPM (RL_CFC_MIN_RADPS * RL_CFC_RADPS_TO_RPM)
#endif

#ifndef RL_CFC_MAX_RPM
#define RL_CFC_MAX_RPM (RL_CFC_MAX_RADPS * RL_CFC_RADPS_TO_RPM)
#endif

#ifndef RL_CFC_MIN_PPRZ
#define RL_CFC_MIN_PPRZ 4500
#endif

#ifndef RL_CFC_MAX_PPRZ
#define RL_CFC_MAX_PPRZ MAX_PPRZ
#endif

/*
 * Training dynamics motor order inferred from:
 * Mx = -W1 -W2 +W3 +W4, My = -W1 +W2 -W3 +W4
 * => W1=back-right, W2=front-right, W3=back-left, W4=front-left.
 * Paparazzi/Bebop servo order here is FL, FR, BR, BL.
 */
#ifndef RL_CFC_MOTOR_0_SRC
#define RL_CFC_MOTOR_0_SRC 0
#endif

#ifndef RL_CFC_MOTOR_1_SRC
#define RL_CFC_MOTOR_1_SRC 1
#endif

#ifndef RL_CFC_MOTOR_2_SRC
#define RL_CFC_MOTOR_2_SRC 2
#endif

#ifndef RL_CFC_MOTOR_3_SRC
#define RL_CFC_MOTOR_3_SRC 3
#endif

#ifndef RL_CFC_MOTOR_0_INVERT
#define RL_CFC_MOTOR_0_INVERT 0
#endif

#ifndef RL_CFC_MOTOR_1_INVERT
#define RL_CFC_MOTOR_1_INVERT 0
#endif

#ifndef RL_CFC_MOTOR_2_INVERT
#define RL_CFC_MOTOR_2_INVERT 0
#endif

#ifndef RL_CFC_MOTOR_3_INVERT
#define RL_CFC_MOTOR_3_INVERT 0
#endif

#ifndef RL_CFC_USE_NED_INPUT
#define RL_CFC_USE_NED_INPUT 1
#endif

#ifndef RL_CFC_MOTOR_STATE_MAX_RADPS
#define RL_CFC_MOTOR_STATE_MAX_RADPS 3000.0f
#endif

#ifndef RL_CFC_ACTION_CURVE_K
#define RL_CFC_ACTION_CURVE_K 0.95f
#endif

#ifndef RL_CFC_TRAIN_K_W
#define RL_CFC_TRAIN_K_W 2.49e-6f
#endif

#ifndef RL_CFC_TRAIN_GRAVITY
#define RL_CFC_TRAIN_GRAVITY 9.81f
#endif

#ifndef RL_CFC_REAL_HOVER_PPRZ
#define RL_CFC_REAL_HOVER_PPRZ 6500.0f
#endif

#ifndef RL_CFC_OBS_ROLL_SIGN
#define RL_CFC_OBS_ROLL_SIGN 1.0f
#endif

#ifndef RL_CFC_OBS_PITCH_SIGN
#define RL_CFC_OBS_PITCH_SIGN 1.0f
#endif

#ifndef RL_CFC_OBS_YAW_SIGN
#define RL_CFC_OBS_YAW_SIGN 1.0f
#endif

#ifndef RL_CFC_OBS_P_SIGN
#define RL_CFC_OBS_P_SIGN 1.0f
#endif

#ifndef RL_CFC_OBS_Q_SIGN
#define RL_CFC_OBS_Q_SIGN 1.0f
#endif

#ifndef RL_CFC_OBS_R_SIGN
#define RL_CFC_OBS_R_SIGN 1.0f
#endif

#ifndef RL_CFC_GATE_0_YAW
#define RL_CFC_GATE_0_YAW 0.0f
#endif

#ifndef RL_CFC_GATE_1_YAW
#define RL_CFC_GATE_1_YAW 0.0f
#endif

#ifndef RL_CFC_GATE_2_YAW
#define RL_CFC_GATE_2_YAW 0.0f
#endif

#ifndef RL_CFC_GATE_3_YAW
#define RL_CFC_GATE_3_YAW 0.0f
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

typedef struct {
  float x;
  float y;
  float z;
} rl_vec3_t;

typedef struct {
  float phi;
  float theta;
  float psi;
} rl_euler_t;

bool rl_cfc_control_enabled = false;
unsigned int rl_cfc_control_waypoint_index = 0;
float rl_cfc_control_last_norm[4] = {0.f, 0.f, 0.f, 0.f};
float rl_cfc_control_last_rpm[4] = {0.f, 0.f, 0.f, 0.f};
int32_t rl_cfc_control_motor_pprz[4] = {0, 0, 0, 0};
int32_t rl_cfc_control_applied_pprz[4] = {0, 0, 0, 0};
int32_t rl_cfc_control_raw_mean_pprz = 0;
unsigned int rl_cfc_control_periodic_count = 0U;
float rl_cfc_control_reached_radius_m = RL_CFC_REACHED_RADIUS_M;
int32_t rl_cfc_control_output_min_pprz = RL_CFC_MIN_PPRZ;
int32_t rl_cfc_control_output_max_pprz = RL_CFC_MAX_PPRZ;
bool rl_cfc_control_use_ned_input = RL_CFC_USE_NED_INPUT;
uint8_t rl_cfc_control_motor_src[4] = {
  RL_CFC_MOTOR_3_SRC,
  RL_CFC_MOTOR_1_SRC,
  RL_CFC_MOTOR_0_SRC,
  RL_CFC_MOTOR_2_SRC
};
bool rl_cfc_control_motor_invert[4] = {
  RL_CFC_MOTOR_0_INVERT,
  RL_CFC_MOTOR_1_INVERT,
  RL_CFC_MOTOR_2_INVERT,
  RL_CFC_MOTOR_3_INVERT
};
float rl_cfc_control_target[3] = {0.f, 0.f, 0.f};
float rl_cfc_control_error[3] = {0.f, 0.f, 0.f};
float rl_cfc_control_input_error[3] = {0.f, 0.f, 0.f};
float rl_cfc_control_input_velocity[3] = {0.f, 0.f, 0.f};
float rl_cfc_control_feedback_rpm[4] = {0.f, 0.f, 0.f, 0.f};
float rl_cfc_control_motor_state[4] = {0.f, 0.f, 0.f, 0.f};
float rl_cfc_control_commanded_radps[4] = {0.f, 0.f, 0.f, 0.f};
float rl_cfc_control_actuator_scale = 1.f;
float rl_cfc_control_raw_action[4] = {0.f, 0.f, 0.f, 0.f};
float rl_cfc_control_action[4] = {0.f, 0.f, 0.f, 0.f};
float rl_cfc_control_policy_raw_action[4] = {0.f, 0.f, 0.f, 0.f};
float rl_cfc_control_obs[20] = {0.f};
float rl_cfc_control_next_gate[4] = {0.f, 0.f, 0.f, 0.f};
float rl_cfc_control_gate_yaw[4] = {
  RL_CFC_GATE_0_YAW,
  RL_CFC_GATE_1_YAW,
  RL_CFC_GATE_2_YAW,
  RL_CFC_GATE_3_YAW
};
float rl_cfc_control_dist_to_target = 0.f;

/* Square in local coordinates. Keep it inside the Cyberzoo geofence. */
static const rl_vec3_t rl_square_waypoints[] = {
  { 1.0f,  1.0f, RL_CFC_TARGET_ALT_M},
  {-1.0f,  1.0f, RL_CFC_TARGET_ALT_M},
  {-1.0f, -1.0f, RL_CFC_TARGET_ALT_M},
  { 1.0f, -1.0f, RL_CFC_TARGET_ALT_M},
};
static const unsigned int rl_num_square_waypoints = sizeof(rl_square_waypoints) / sizeof(rl_square_waypoints[0]);

static rl_vec3_t get_square_waypoint(unsigned int index)
{
#if defined(WP_NN_SQ_1) && defined(WP_NN_SQ_2) && defined(WP_NN_SQ_3) && defined(WP_NN_SQ_4)
  static const uint8_t wp_ids[] = {WP_NN_SQ_1, WP_NN_SQ_2, WP_NN_SQ_3, WP_NN_SQ_4};
  const uint8_t wp_id = wp_ids[index % (sizeof(wp_ids) / sizeof(wp_ids[0]))];
  return (rl_vec3_t){waypoint_get_x(wp_id), waypoint_get_y(wp_id), waypoint_get_alt(wp_id)};
#else
  return rl_square_waypoints[index % rl_num_square_waypoints];
#endif
}

__attribute__((weak)) rl_vec3_t rl_cfc_get_position_m(void)
{
  const struct EnuCoor_f *pos = stateGetPositionEnu_f();
  return (rl_vec3_t){pos->x, pos->y, pos->z};
}

__attribute__((weak)) rl_vec3_t rl_cfc_get_velocity_mps(void)
{
  const struct EnuCoor_f *vel = stateGetSpeedEnu_f();
  return (rl_vec3_t){vel->x, vel->y, vel->z};
}

__attribute__((weak)) rl_euler_t rl_cfc_get_attitude_rad(void)
{
  const struct FloatEulers *att = stateGetNedToBodyEulers_f();
  return (rl_euler_t){att->phi, att->theta, att->psi};
}

__attribute__((weak)) rl_vec3_t rl_cfc_get_body_rates_radps(void)
{
  const struct FloatRates *rates = stateGetBodyRates_f();
  return (rl_vec3_t){rates->p, rates->q, rates->r};
}

__attribute__((weak)) void rl_cfc_get_motor_feedback_rpm(float omega_rpm[4])
{
#ifdef BOARD_BEBOP
  omega_rpm[0] = (float)actuators_bebop.rpm_obs[0];
  omega_rpm[1] = (float)actuators_bebop.rpm_obs[1];
  omega_rpm[2] = (float)actuators_bebop.rpm_obs[2];
  omega_rpm[3] = (float)actuators_bebop.rpm_obs[3];
#else
  omega_rpm[0] = rl_cfc_control_last_rpm[0];
  omega_rpm[1] = rl_cfc_control_last_rpm[1];
  omega_rpm[2] = rl_cfc_control_last_rpm[2];
  omega_rpm[3] = rl_cfc_control_last_rpm[3];
#endif
}

static float clampf(float x, float lo, float hi)
{
  if (x < lo) return lo;
  if (x > hi) return hi;
  return x;
}

static float wrap_pi(float x)
{
  while (x > (float)M_PI) {
    x -= 2.0f * (float)M_PI;
  }
  while (x < -(float)M_PI) {
    x += 2.0f * (float)M_PI;
  }
  return x;
}

static float train_command_u_to_radps(float u);

static float train_hover_radps(void)
{
  return sqrtf(RL_CFC_TRAIN_GRAVITY / (4.0f * RL_CFC_TRAIN_K_W));
}

static float train_radps_to_command_u(float radps)
{
  const float y = clampf((radps - RL_CFC_MIN_RADPS) / (RL_CFC_MAX_RADPS - RL_CFC_MIN_RADPS), 0.f, 1.f);
  const float curved = y * y;
  const float b = 1.f - RL_CFC_ACTION_CURVE_K;
  const float disc = b * b + 4.f * RL_CFC_ACTION_CURVE_K * curved;
  if (RL_CFC_ACTION_CURVE_K <= 0.f) {
    return clampf(curved / b, 0.f, 1.f);
  }
  return clampf((-b + sqrtf(disc > 0.f ? disc : 0.f)) / (2.f * RL_CFC_ACTION_CURVE_K), 0.f, 1.f);
}

static float train_hover_u(void)
{
  return train_radps_to_command_u(train_hover_radps());
}

static float real_hover_u(void)
{
  return clampf(RL_CFC_REAL_HOVER_PPRZ / (float)MAX_PPRZ, 0.f, 1.f);
}

static float actuator_scale(void)
{
  const float train_u = train_hover_u();
  if (train_u <= 0.f) {
    return 1.f;
  }
  return real_hover_u() / train_u;
}

static float train_u_to_real_u(float train_u)
{
  train_u = clampf(train_u, 0.f, 1.f);
  const float th = train_hover_u();
  const float rh = real_hover_u();
  if (train_u <= th) {
    return th > 0.f ? rh * train_u / th : 0.f;
  }
  return rh + (1.f - rh) * (train_u - th) / (1.f - th);
}

static float real_u_to_train_u(float real_u)
{
  real_u = clampf(real_u, 0.f, 1.f);
  const float th = train_hover_u();
  const float rh = real_hover_u();
  if (real_u <= rh) {
    return rh > 0.f ? th * real_u / rh : 0.f;
  }
  return th + (1.f - th) * (real_u - rh) / (1.f - rh);
}

static float train_radps_to_bebop_rpm(float radps)
{
  return train_u_to_real_u(train_radps_to_command_u(radps)) * RL_CFC_MAX_RPM;
}

static float bebop_rpm_to_train_radps(float rpm)
{
  return train_command_u_to_radps(real_u_to_train_u(rpm / RL_CFC_MAX_RPM));
}

static float rpm_to_motor_state(float rpm)
{
  const float radps = bebop_rpm_to_train_radps(rpm);
  return 2.0f * radps / RL_CFC_MOTOR_STATE_MAX_RADPS - 1.0f;
}

static rl_vec3_t gate_frame_vec(const rl_vec3_t vec, float gate_yaw)
{
  const float c = cosf(gate_yaw);
  const float s = sinf(gate_yaw);
  return (rl_vec3_t){vec.x * c + vec.y * s, -vec.x * s + vec.y * c, vec.z};
}

static float action_to_train_command_u(float action)
{
  return 0.5f * (clampf(action, -1.f, 1.f) + 1.f);
}

static float train_command_u_to_radps(float u)
{
  u = clampf(u, 0.f, 1.f);

  const float curved =
      RL_CFC_ACTION_CURVE_K * u * u
      + (1.f - RL_CFC_ACTION_CURVE_K) * u;

  return RL_CFC_MIN_RADPS
      + (RL_CFC_MAX_RADPS - RL_CFC_MIN_RADPS)
      * sqrtf(clampf(curved, 0.f, 1.f));
}

static void get_motor_driver_range(uint8_t motor_idx, int32_t *min, int32_t *neutral, int32_t *max)
{
  switch (motor_idx) {
#ifdef SERVO_TOP_LEFT_IDX
    case SERVO_TOP_LEFT_IDX:
      *min = SERVO_TOP_LEFT_MIN;
      *neutral = SERVO_TOP_LEFT_NEUTRAL;
      *max = SERVO_TOP_LEFT_MAX;
      return;
#endif
#ifdef SERVO_TOP_RIGHT_IDX
    case SERVO_TOP_RIGHT_IDX:
      *min = SERVO_TOP_RIGHT_MIN;
      *neutral = SERVO_TOP_RIGHT_NEUTRAL;
      *max = SERVO_TOP_RIGHT_MAX;
      return;
#endif
#ifdef SERVO_BOTTOM_RIGHT_IDX
    case SERVO_BOTTOM_RIGHT_IDX:
      *min = SERVO_BOTTOM_RIGHT_MIN;
      *neutral = SERVO_BOTTOM_RIGHT_NEUTRAL;
      *max = SERVO_BOTTOM_RIGHT_MAX;
      return;
#endif
#ifdef SERVO_BOTTOM_LEFT_IDX
    case SERVO_BOTTOM_LEFT_IDX:
      *min = SERVO_BOTTOM_LEFT_MIN;
      *neutral = SERVO_BOTTOM_LEFT_NEUTRAL;
      *max = SERVO_BOTTOM_LEFT_MAX;
      return;
#endif
    default:
      *min = 0;
      *neutral = 0;
      *max = (int32_t)(RL_CFC_MAX_RPM + 0.5f);
      return;
  }
}

static int32_t rpm_to_pprz_for_motor(uint8_t motor_idx, float rpm)
{
  int32_t min;
  int32_t neutral;
  int32_t max;
  get_motor_driver_range(motor_idx, &min, &neutral, &max);

  rpm = clampf(rpm, (float)min, (float)max);
  float pprz = 0.f;
  if (rpm >= (float)neutral) {
    const float travel = (float)(max - neutral);
    pprz = travel > 0.f ? (rpm - (float)neutral) * (float)MAX_PPRZ / travel : 0.f;
  } else {
    const float travel = (float)(neutral - min);
    pprz = travel > 0.f ? (rpm - (float)neutral) * (float)MAX_PPRZ / travel : 0.f;
  }

  return TRIM_PPRZ((int32_t)(pprz >= 0.f ? pprz + 0.5f : pprz - 0.5f));
}

static int32_t clamp_motor_command(int32_t command)
{
  int32_t output_min = rl_cfc_control_output_min_pprz;
  int32_t output_max = rl_cfc_control_output_max_pprz;
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

static rl_vec3_t enu_to_ned_vec(const rl_vec3_t enu)
{
  return (rl_vec3_t){enu.y, enu.x, -enu.z};
}

static unsigned int motor_src_for_servo(uint8_t motor_idx)
{
  return rl_cfc_control_motor_src[motor_idx] % 4U;
}

static bool motor_invert_for_servo(uint8_t motor_idx)
{
  return rl_cfc_control_motor_invert[motor_idx];
}

static int32_t maybe_invert_pprz(int32_t command)
{
  int32_t output_min = rl_cfc_control_output_min_pprz;
  int32_t output_max = rl_cfc_control_output_max_pprz;
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
  int32_t command = rpm_to_pprz_for_motor(motor_idx, rl_cfc_control_last_rpm[src]);
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
  rl_cfc_control_raw_mean_pprz = sum / 4;
  return rl_cfc_control_raw_mean_pprz;
}

static int32_t direct_rl_motor_pprz(uint8_t motor_idx)
{
  mean_mapped_raw_motor_pprz();
  return clamp_motor_command(mapped_raw_motor_pprz(motor_idx));
}

static float waypoint_dist2(const rl_vec3_t pos, const rl_vec3_t wp)
{
  const float dx = wp.x - pos.x;
  const float dy = wp.y - pos.y;
  const float dz = wp.z - pos.z;
  return dx * dx + dy * dy + dz * dz;
}

static void update_target_debug(const rl_vec3_t pos, const rl_vec3_t wp)
{
  rl_cfc_control_target[0] = wp.x;
  rl_cfc_control_target[1] = wp.y;
  rl_cfc_control_target[2] = wp.z;

  rl_cfc_control_error[0] = wp.x - pos.x;
  rl_cfc_control_error[1] = wp.y - pos.y;
  rl_cfc_control_error[2] = wp.z - pos.z;
  rl_cfc_control_dist_to_target = sqrtf(waypoint_dist2(pos, wp));
}

static void update_next_gate_debug(unsigned int waypoint_index)
{
  const rl_vec3_t current_wp = get_square_waypoint(waypoint_index);
  const rl_vec3_t next_wp = get_square_waypoint(waypoint_index + 1U);
  rl_vec3_t next_rel = {
    next_wp.x - current_wp.x,
    next_wp.y - current_wp.y,
    next_wp.z - current_wp.z
  };
  const float gate_yaw = rl_cfc_control_gate_yaw[waypoint_index % rl_num_square_waypoints];
  if (rl_cfc_control_use_ned_input) {
    next_rel = enu_to_ned_vec(next_rel);
  }
  next_rel = gate_frame_vec(next_rel, gate_yaw);

  rl_cfc_control_next_gate[0] = next_rel.x;
  rl_cfc_control_next_gate[1] = next_rel.y;
  rl_cfc_control_next_gate[2] = next_rel.z;
  rl_cfc_control_next_gate[3] = wrap_pi(rl_cfc_control_gate_yaw[(waypoint_index + 1U) % rl_num_square_waypoints] - gate_yaw);
}

static void reset_target_debug(void)
{
  for (unsigned int i = 0; i < 3U; i++) {
    rl_cfc_control_target[i] = 0.f;
    rl_cfc_control_error[i] = 0.f;
    rl_cfc_control_input_error[i] = 0.f;
    rl_cfc_control_input_velocity[i] = 0.f;
  }
  for (unsigned int i = 0; i < 4U; i++) {
    rl_cfc_control_motor_state[i] = 0.f;
    rl_cfc_control_raw_action[i] = 0.f;
    rl_cfc_control_action[i] = 0.f;
    rl_cfc_control_policy_raw_action[i] = 0.f;
    rl_cfc_control_next_gate[i] = 0.f;
  }
  for (unsigned int i = 0; i < 20U; i++) {
    rl_cfc_control_obs[i] = 0.f;
  }
  rl_cfc_control_dist_to_target = 0.f;
}

static void maybe_advance_waypoint(const rl_vec3_t pos)
{
  const rl_vec3_t wp = get_square_waypoint(rl_cfc_control_waypoint_index);
  const float d2 = waypoint_dist2(pos, wp);
  const float reached_radius = rl_cfc_control_reached_radius_m > 0.f ? rl_cfc_control_reached_radius_m : 0.05f;
  if (d2 < reached_radius * reached_radius) {
    rl_cfc_control_waypoint_index = (rl_cfc_control_waypoint_index + 1U) % rl_num_square_waypoints;
    // rl_cfc_reset();  /* Optional but usually cleaner when changing target. */
  }
  update_target_debug(pos, get_square_waypoint(rl_cfc_control_waypoint_index));
  update_next_gate_debug(rl_cfc_control_waypoint_index);
}

void rl_cfc_control_init(void)
{
  rl_cfc_control_enabled = false;
  rl_cfc_control_waypoint_index = 0U;
  rl_cfc_control_periodic_count = 0U;
  rl_cfc_control_actuator_scale = actuator_scale();
  for (unsigned int i = 0; i < 4U; i++) {
    rl_cfc_control_last_norm[i] = 0.f;
    rl_cfc_control_last_rpm[i] = 0.f;
    rl_cfc_control_motor_pprz[i] = 0;
    rl_cfc_control_applied_pprz[i] = 0;
    rl_cfc_control_feedback_rpm[i] = 0.f;
    rl_cfc_control_motor_state[i] = 0.f;
    rl_cfc_control_commanded_radps[i] = 0.f;
    rl_cfc_control_raw_action[i] = 0.f;
    rl_cfc_control_action[i] = 0.f;
    rl_cfc_control_policy_raw_action[i] = 0.f;
  }
  rl_cfc_control_raw_mean_pprz = 0;
  reset_target_debug();
  rl_cfc_reset();
}

void rl_cfc_control_start(void)
{
  rl_cfc_control_enabled = true;
  rl_cfc_control_periodic_count = 0U;
  for (unsigned int i = 0; i < 4U; i++) {
    rl_cfc_control_last_norm[i] = 0.f;
    rl_cfc_control_last_rpm[i] = RL_CFC_MIN_RPM;
    rl_cfc_control_motor_pprz[i] = rl_cfc_control_output_min_pprz;
    rl_cfc_control_applied_pprz[i] = rl_cfc_control_output_min_pprz;
    rl_cfc_control_feedback_rpm[i] = RL_CFC_MIN_RPM;
    rl_cfc_control_motor_state[i] = rpm_to_motor_state(RL_CFC_MIN_RPM);
    rl_cfc_control_commanded_radps[i] = RL_CFC_MIN_RADPS;
    rl_cfc_control_policy_raw_action[i] = -1.f;
    rl_cfc_control_raw_action[i] = -1.f;
    rl_cfc_control_action[i] = -1.f;
  }
  rl_cfc_control_raw_mean_pprz = 0;
  rl_cfc_control_waypoint_index = RL_CFC_START_WAYPOINT_INDEX % rl_num_square_waypoints;
  update_target_debug(rl_cfc_get_position_m(), get_square_waypoint(rl_cfc_control_waypoint_index));
  update_next_gate_debug(rl_cfc_control_waypoint_index);
  rl_cfc_reset();
}

void rl_cfc_control_stop(void)
{
  rl_cfc_control_enabled = false;
  for (unsigned int i = 0; i < 4U; i++) {
    rl_cfc_control_last_norm[i] = 0.f;
    rl_cfc_control_last_rpm[i] = 0.f;
    rl_cfc_control_motor_pprz[i] = 0;
    rl_cfc_control_applied_pprz[i] = 0;
    rl_cfc_control_feedback_rpm[i] = 0.f;
    rl_cfc_control_motor_state[i] = 0.f;
    rl_cfc_control_commanded_radps[i] = 0.f;
    rl_cfc_control_policy_raw_action[i] = 0.f;
    rl_cfc_control_raw_action[i] = 0.f;
    rl_cfc_control_action[i] = 0.f;
  }
  rl_cfc_control_raw_mean_pprz = 0;
  reset_target_debug();
  rl_cfc_reset();
}

void rl_cfc_control_periodic(void)
{
  if (!rl_cfc_control_enabled) {
    return;
  }
  rl_cfc_control_periodic_count++;

  const rl_vec3_t pos = rl_cfc_get_position_m();
  maybe_advance_waypoint(pos);

  const rl_vec3_t vel = rl_cfc_get_velocity_mps();
  const rl_vec3_t err = {
    rl_cfc_control_error[0],
    rl_cfc_control_error[1],
    rl_cfc_control_error[2]
  };
  rl_vec3_t pos_gate = {-err.x, -err.y, -err.z};
  rl_vec3_t vel_gate = vel;
  if (rl_cfc_control_use_ned_input) {
    pos_gate = enu_to_ned_vec(pos_gate);
    vel_gate = enu_to_ned_vec(vel_gate);
  }
  const float gate_yaw = rl_cfc_control_gate_yaw[rl_cfc_control_waypoint_index % rl_num_square_waypoints];
  pos_gate = gate_frame_vec(pos_gate, gate_yaw);
  vel_gate = gate_frame_vec(vel_gate, gate_yaw);
  rl_cfc_control_input_error[0] = pos_gate.x;
  rl_cfc_control_input_error[1] = pos_gate.y;
  rl_cfc_control_input_error[2] = pos_gate.z;
  rl_cfc_control_input_velocity[0] = vel_gate.x;
  rl_cfc_control_input_velocity[1] = vel_gate.y;
  rl_cfc_control_input_velocity[2] = vel_gate.z;

  const rl_euler_t att = rl_cfc_get_attitude_rad();
  const rl_vec3_t rates = rl_cfc_get_body_rates_radps();
  float omega[4];
  float policy_omega[4] = {0.f, 0.f, 0.f, 0.f};
  rl_cfc_control_actuator_scale = actuator_scale();
  rl_cfc_get_motor_feedback_rpm(omega);
  for (unsigned int i = 0; i < 4U; i++) {
    rl_cfc_control_feedback_rpm[i] = omega[i];
    policy_omega[motor_src_for_servo(i)] = omega[i];
  }
  for (unsigned int i = 0; i < 4U; i++) {
    rl_cfc_control_motor_state[i] = rpm_to_motor_state(policy_omega[i]);
  }

  float obs[OBS_SIZE] = {
    pos_gate.x, pos_gate.y, pos_gate.z,
    vel_gate.x, vel_gate.y, vel_gate.z,
    RL_CFC_OBS_ROLL_SIGN * att.phi,
    RL_CFC_OBS_PITCH_SIGN * att.theta,
    RL_CFC_OBS_YAW_SIGN * wrap_pi(att.psi - gate_yaw),
    RL_CFC_OBS_P_SIGN * rates.x,
    RL_CFC_OBS_Q_SIGN * rates.y,
    RL_CFC_OBS_R_SIGN * rates.z,
    rl_cfc_control_motor_state[0],
    rl_cfc_control_motor_state[1],
    rl_cfc_control_motor_state[2],
    rl_cfc_control_motor_state[3],
    rl_cfc_control_next_gate[0],
    rl_cfc_control_next_gate[1],
    rl_cfc_control_next_gate[2],
    rl_cfc_control_next_gate[3]
  };
  for (unsigned int i = 0; i < OBS_SIZE; i++) {
    rl_cfc_control_obs[i] = obs[i];
  }

  float action[NUM_CONTROLS];
  rl_cfc_control(obs, action);

  for (unsigned int i = 0; i < 4U; i++) {
    rl_cfc_control_policy_raw_action[i] = action[i];
    rl_cfc_control_raw_action[i] = action[i];
    rl_cfc_control_action[i] = clampf(action[i], -1.f, 1.f);
    const float command_u = action_to_train_command_u(rl_cfc_control_action[i]);
    const float commanded_radps = train_command_u_to_radps(command_u);
    rl_cfc_control_commanded_radps[i] = commanded_radps;
    rl_cfc_control_last_norm[i] = (commanded_radps - RL_CFC_MIN_RADPS) / (RL_CFC_MAX_RADPS - RL_CFC_MIN_RADPS);
    rl_cfc_control_last_rpm[i] = train_radps_to_bebop_rpm(commanded_radps);
  }
  for (unsigned int i = 0; i < 4U; i++) {
    rl_cfc_control_motor_pprz[i] = mapped_raw_motor_pprz(i);
  }
}

int32_t rl_cfc_control_get_motor_pprz(uint8_t motor_idx, int32_t autopilot_pprz)
{
  if (!rl_cfc_control_enabled || motor_idx >= 4U) {
    if (motor_idx < 4U) {
      rl_cfc_control_applied_pprz[motor_idx] = autopilot_pprz;
    }
    return autopilot_pprz;
  }

  const int32_t command = direct_rl_motor_pprz(motor_idx);

  rl_cfc_control_applied_pprz[motor_idx] = command;
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
