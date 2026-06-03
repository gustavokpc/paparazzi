/*
 * rl_cfc_control.c
 * Paparazzi wrapper for the recurrent PPO CfC baseline controller.
 *
 * IMPORTANT:
 * - This file builds the 20-element raw RL observation expected by
 *   rl_cfc_parameters.h:
 *   pos_G, vel_G, euler_B_to_G, rates_B, motor_state, next_gate_G.
 * - The RL policy returns actions in [-1,1]. In the training environment these
 *   actions command steady-state motor speed in rad/s through the same nonlinear
 *   action curve implemented below.
 * - The wrapper converts that trained rad/s command directly to Bebop reference
 *   RPM and applies it with actuators_bebop_set(), matching the working CFC
 *   direct-RPM path.
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

#ifndef RL_CFC_REACHED_RADIUS_M
#define RL_CFC_REACHED_RADIUS_M 1.6f
#endif

#ifndef RL_CFC_TARGET_ALT_M
#define RL_CFC_TARGET_ALT_M 1.5f
#endif

#ifndef RL_CFC_START_WAYPOINT_INDEX
#define RL_CFC_START_WAYPOINT_INDEX 7
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

#ifndef RL_CFC_NPS_MIN_RPM
#define RL_CFC_NPS_MIN_RPM 0.0f
#endif

#ifndef RL_CFC_NPS_MAX_RPM
#define RL_CFC_NPS_MAX_RPM RL_CFC_MAX_RPM
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

#ifndef RL_CFC_TORQUE_SCALE
#define RL_CFC_TORQUE_SCALE 1.0f
#endif

#ifndef RL_CFC_OBS_Z_SIGN
#define RL_CFC_OBS_Z_SIGN 1.0f
#endif

#ifndef RL_CFC_OBS_VZ_SIGN
#define RL_CFC_OBS_VZ_SIGN 1.0f
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
int32_t rl_cfc_control_motor_rpm_cmd[4] = {0, 0, 0, 0};
int32_t rl_cfc_control_applied_rpm_cmd[4] = {0, 0, 0, 0};
int32_t rl_cfc_control_raw_mean_rpm = 0;
unsigned int rl_cfc_control_periodic_count = 0U;
float rl_cfc_control_reached_radius_m = RL_CFC_REACHED_RADIUS_M;
bool rl_cfc_control_use_ned_input = RL_CFC_USE_NED_INPUT;
float rl_cfc_control_target[3] = {0.f, 0.f, 0.f};
float rl_cfc_control_error[3] = {0.f, 0.f, 0.f};
float rl_cfc_control_input_error[3] = {0.f, 0.f, 0.f};
float rl_cfc_control_input_velocity[3] = {0.f, 0.f, 0.f};
float rl_cfc_control_feedback_rpm[4] = {0.f, 0.f, 0.f, 0.f};
float rl_cfc_control_motor_state[4] = {0.f, 0.f, 0.f, 0.f};
float rl_cfc_control_commanded_radps[4] = {0.f, 0.f, 0.f, 0.f};
float rl_cfc_control_torque_scale = RL_CFC_TORQUE_SCALE;
float rl_cfc_control_z_sign = RL_CFC_OBS_Z_SIGN;
float rl_cfc_control_vz_sign = RL_CFC_OBS_VZ_SIGN;
float rl_cfc_control_mean_policy_rpm = 0.f;
float rl_cfc_control_mean_commanded_radps = 0.f;
float rl_cfc_control_raw_action[4] = {0.f, 0.f, 0.f, 0.f};
float rl_cfc_control_action[4] = {0.f, 0.f, 0.f, 0.f};
float rl_cfc_control_policy_raw_action[4] = {0.f, 0.f, 0.f, 0.f};
float rl_cfc_control_obs[20] = {0.f};
float rl_cfc_control_next_gate[4] = {0.f, 0.f, 0.f, 0.f};
float rl_cfc_control_gate_yaw[8] = {
  0.5f * (float)M_PI,
  (float)M_PI,
  0.5f * (float)M_PI,
  0.0f,
  -0.5f * (float)M_PI,
  -(float)M_PI,
  -0.5f * (float)M_PI,
  0.0f
};
float rl_cfc_control_dist_to_target = 0.f;

/*
 * Figure-eight gate centers from the RL environment.
 * RL world uses z-down. This table is stored in Paparazzi ENU as:
 *   enu = {rl_y, rl_x, -rl_z}
 * so enu_to_ned_vec(current - gate) recreates the raw RL world position error.
 */
static const rl_vec3_t rl_figure_eight_waypoints[] = {
  {-1.5f,  1.5f, RL_CFC_TARGET_ALT_M},
  { 0.0f,  0.0f, RL_CFC_TARGET_ALT_M},
  { 1.5f, -1.5f, RL_CFC_TARGET_ALT_M},
  { 3.0f,  0.0f, RL_CFC_TARGET_ALT_M},
  { 1.5f,  1.5f, RL_CFC_TARGET_ALT_M},
  { 0.0f,  0.0f, RL_CFC_TARGET_ALT_M},
  {-1.5f, -1.5f, RL_CFC_TARGET_ALT_M},
  {-3.0f,  0.0f, RL_CFC_TARGET_ALT_M},
};
static const unsigned int rl_num_figure_eight_waypoints =
    sizeof(rl_figure_eight_waypoints) / sizeof(rl_figure_eight_waypoints[0]);

/*
 * Policy motor order: BR, FR, BL, FL.
 * Paparazzi/Bebop servo order: FL, FR, BR, BL.
 */
static const uint8_t rl_motor_src_for_servo[4] = {3U, 1U, 0U, 2U};

static rl_vec3_t get_figure_eight_waypoint(unsigned int index)
{
  return rl_figure_eight_waypoints[index % rl_num_figure_eight_waypoints];
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

static float train_radps_to_rpm(float radps)
{
  return clampf(radps, RL_CFC_MIN_RADPS, RL_CFC_MAX_RADPS) * RL_CFC_RADPS_TO_RPM;
}

static float rpm_to_motor_state(float rpm)
{
  const float radps = rpm * RL_CFC_RPM_TO_RADPS;
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

static rl_vec3_t enu_to_ned_vec(const rl_vec3_t enu)
{
  return (rl_vec3_t){enu.y, enu.x, -enu.z};
}

static unsigned int motor_src_for_servo(uint8_t motor_idx)
{
  return rl_motor_src_for_servo[motor_idx] % 4U;
}

static float mapped_motor_rpm(uint8_t motor_idx)
{
  return rl_cfc_control_last_rpm[motor_src_for_servo(motor_idx)];
}

static float mean_policy_rpm(void)
{
  float sum = 0.f;
  for (uint8_t i = 0U; i < 4U; i++) {
    sum += rl_cfc_control_last_rpm[i];
  }
  return sum * 0.25f;
}

static float scaled_mapped_motor_rpm(uint8_t motor_idx)
{
  const float mean = mean_policy_rpm();
  const float scale = clampf(rl_cfc_control_torque_scale, 0.f, 1.f);
  const float rpm = mapped_motor_rpm(motor_idx);
  return mean + scale * (rpm - mean);
}

static int32_t rpm_to_command(float rpm)
{
  rpm = clampf(rpm, RL_CFC_MIN_RPM, RL_CFC_MAX_RPM);
  return (int32_t)(rpm >= 0.f ? rpm + 0.5f : rpm - 0.5f);
}

static int32_t mean_mapped_raw_motor_rpm(void)
{
  int32_t sum = 0;
  for (uint8_t i = 0U; i < 4U; i++) {
    sum += rpm_to_command(scaled_mapped_motor_rpm(i));
  }
  rl_cfc_control_raw_mean_rpm = sum / 4;
  return rl_cfc_control_raw_mean_rpm;
}

static int32_t rpm_to_nps_command(float rpm)
{
  const float nps_min = RL_CFC_NPS_MIN_RPM;
  const float nps_max = RL_CFC_NPS_MAX_RPM > nps_min ? RL_CFC_NPS_MAX_RPM : RL_CFC_MAX_RPM;
  const float norm = (clampf(rpm, nps_min, nps_max) - nps_min) / (nps_max - nps_min);
  return TRIM_PPRZ((int32_t)(norm * (float)MAX_PPRZ + 0.5f));
}

static void apply_motor_rpm(uint8_t motor_idx, int32_t rpm)
{
#ifndef BOARD_BEBOP
  if (motor_idx < MOTOR_MIXING_NB_MOTOR) {
    motor_mixing.commands[motor_idx] = rpm_to_nps_command((float)rpm);
  }
#else
  actuators_bebop_set(motor_idx, (int16_t)rpm);
#endif
}

static float waypoint_dist2(const rl_vec3_t pos, const rl_vec3_t wp)
{
  const float dx = wp.x - pos.x;
  const float dy = wp.y - pos.y;
  const float dz = wp.z - pos.z;
  return dx * dx + dy * dy + dz * dz;
}

static float waypoint_xy_dist2(const rl_vec3_t a, const rl_vec3_t b)
{
  const float dx = a.x - b.x;
  const float dy = a.y - b.y;
  return dx * dx + dy * dy;
}

static unsigned int choose_initial_waypoint(const rl_vec3_t pos)
{
#if RL_CFC_START_WAYPOINT_INDEX >= 0
  (void)pos;
  return (unsigned int)RL_CFC_START_WAYPOINT_INDEX % rl_num_figure_eight_waypoints;
#else
  unsigned int best_any = 0U;
  unsigned int best_behind = 0U;
  float best_any_d2 = 1.0e9f;
  float best_behind_d2 = 1.0e9f;
  bool found_behind = false;
  const rl_vec3_t rl_pos = enu_to_ned_vec(pos);

  for (unsigned int i = 0U; i < rl_num_figure_eight_waypoints; i++) {
    const rl_vec3_t wp = get_figure_eight_waypoint(i);
    const float d2 = waypoint_xy_dist2(pos, wp);
    if (d2 < best_any_d2) {
      best_any_d2 = d2;
      best_any = i;
    }

    const rl_vec3_t rl_wp = enu_to_ned_vec(wp);
    const float gate_yaw = rl_cfc_control_gate_yaw[i];
    const float projected =
        (rl_pos.x - rl_wp.x) * cosf(gate_yaw)
        + (rl_pos.y - rl_wp.y) * sinf(gate_yaw);
    if (projected < 0.f && d2 < best_behind_d2) {
      best_behind_d2 = d2;
      best_behind = i;
      found_behind = true;
    }
  }

  return found_behind ? best_behind : best_any;
#endif
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
  const rl_vec3_t current_wp = get_figure_eight_waypoint(waypoint_index);
  const rl_vec3_t next_wp = get_figure_eight_waypoint(waypoint_index + 1U);
  rl_vec3_t next_rel = {
    next_wp.x - current_wp.x,
    next_wp.y - current_wp.y,
    next_wp.z - current_wp.z
  };
  const float gate_yaw = rl_cfc_control_gate_yaw[waypoint_index % rl_num_figure_eight_waypoints];
  if (rl_cfc_control_use_ned_input) {
    next_rel = enu_to_ned_vec(next_rel);
  }
  next_rel = gate_frame_vec(next_rel, gate_yaw);

  rl_cfc_control_next_gate[0] = next_rel.x;
  rl_cfc_control_next_gate[1] = next_rel.y;
  rl_cfc_control_next_gate[2] = next_rel.z;
  rl_cfc_control_next_gate[3] =
      wrap_pi(rl_cfc_control_gate_yaw[(waypoint_index + 1U) % rl_num_figure_eight_waypoints] - gate_yaw);
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
  const rl_vec3_t wp = get_figure_eight_waypoint(rl_cfc_control_waypoint_index);
  const float d2 = waypoint_dist2(pos, wp);
  const float reached_radius = rl_cfc_control_reached_radius_m > 0.f ? rl_cfc_control_reached_radius_m : 0.05f;
  if (d2 < reached_radius * reached_radius) {
    rl_cfc_control_waypoint_index = (rl_cfc_control_waypoint_index + 1U) % rl_num_figure_eight_waypoints;
    rl_cfc_reset();
  }
  update_target_debug(pos, get_figure_eight_waypoint(rl_cfc_control_waypoint_index));
  update_next_gate_debug(rl_cfc_control_waypoint_index);
}

void rl_cfc_control_init(void)
{
  rl_cfc_control_enabled = false;
  rl_cfc_control_waypoint_index = 0U;
  rl_cfc_control_periodic_count = 0U;
  for (unsigned int i = 0; i < 4U; i++) {
    rl_cfc_control_last_norm[i] = 0.f;
    rl_cfc_control_last_rpm[i] = 0.f;
    rl_cfc_control_motor_rpm_cmd[i] = 0;
    rl_cfc_control_applied_rpm_cmd[i] = 0;
    rl_cfc_control_feedback_rpm[i] = 0.f;
    rl_cfc_control_motor_state[i] = 0.f;
    rl_cfc_control_commanded_radps[i] = 0.f;
    rl_cfc_control_raw_action[i] = 0.f;
    rl_cfc_control_action[i] = 0.f;
    rl_cfc_control_policy_raw_action[i] = 0.f;
  }
  rl_cfc_control_raw_mean_rpm = 0;
  rl_cfc_control_mean_policy_rpm = 0.f;
  rl_cfc_control_mean_commanded_radps = 0.f;
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
    rl_cfc_control_motor_rpm_cmd[i] = rpm_to_command(RL_CFC_MIN_RPM);
    rl_cfc_control_applied_rpm_cmd[i] = rpm_to_command(RL_CFC_MIN_RPM);
    rl_cfc_control_feedback_rpm[i] = RL_CFC_MIN_RPM;
    rl_cfc_control_motor_state[i] = rpm_to_motor_state(RL_CFC_MIN_RPM);
    rl_cfc_control_commanded_radps[i] = RL_CFC_MIN_RADPS;
    rl_cfc_control_policy_raw_action[i] = -1.f;
    rl_cfc_control_raw_action[i] = -1.f;
    rl_cfc_control_action[i] = -1.f;
  }
  rl_cfc_control_raw_mean_rpm = 0;
  rl_cfc_control_mean_policy_rpm = 0.f;
  rl_cfc_control_mean_commanded_radps = 0.f;
  const rl_vec3_t start_pos = rl_cfc_get_position_m();
  rl_cfc_control_waypoint_index = choose_initial_waypoint(start_pos);
  update_target_debug(start_pos, get_figure_eight_waypoint(rl_cfc_control_waypoint_index));
  update_next_gate_debug(rl_cfc_control_waypoint_index);
  rl_cfc_reset();
}

void rl_cfc_control_stop(void)
{
  rl_cfc_control_enabled = false;
  for (unsigned int i = 0; i < 4U; i++) {
    rl_cfc_control_last_norm[i] = 0.f;
    rl_cfc_control_last_rpm[i] = 0.f;
    rl_cfc_control_motor_rpm_cmd[i] = 0;
    rl_cfc_control_applied_rpm_cmd[i] = 0;
    rl_cfc_control_feedback_rpm[i] = 0.f;
    rl_cfc_control_motor_state[i] = 0.f;
    rl_cfc_control_commanded_radps[i] = 0.f;
    rl_cfc_control_policy_raw_action[i] = 0.f;
    rl_cfc_control_raw_action[i] = 0.f;
    rl_cfc_control_action[i] = 0.f;
  }
  rl_cfc_control_raw_mean_rpm = 0;
  rl_cfc_control_mean_policy_rpm = 0.f;
  rl_cfc_control_mean_commanded_radps = 0.f;
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
  const float gate_yaw = rl_cfc_control_gate_yaw[rl_cfc_control_waypoint_index % rl_num_figure_eight_waypoints];
  pos_gate = gate_frame_vec(pos_gate, gate_yaw);
  vel_gate = gate_frame_vec(vel_gate, gate_yaw);
  pos_gate.z *= rl_cfc_control_z_sign >= 0.f ? 1.f : -1.f;
  vel_gate.z *= rl_cfc_control_vz_sign >= 0.f ? 1.f : -1.f;
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

  float rpm_sum = 0.f;
  float radps_sum = 0.f;
  for (unsigned int i = 0; i < 4U; i++) {
    rl_cfc_control_policy_raw_action[i] = action[i];
    rl_cfc_control_raw_action[i] = action[i];
    rl_cfc_control_action[i] = clampf(action[i], -1.f, 1.f);
    const float command_u = action_to_train_command_u(rl_cfc_control_action[i]);
    const float commanded_radps = train_command_u_to_radps(command_u);
    rl_cfc_control_commanded_radps[i] = commanded_radps;
    rl_cfc_control_last_norm[i] = (commanded_radps - RL_CFC_MIN_RADPS) / (RL_CFC_MAX_RADPS - RL_CFC_MIN_RADPS);
    rl_cfc_control_last_rpm[i] = train_radps_to_rpm(commanded_radps);
    rpm_sum += rl_cfc_control_last_rpm[i];
    radps_sum += commanded_radps;
  }
  rl_cfc_control_mean_policy_rpm = rpm_sum * 0.25f;
  rl_cfc_control_mean_commanded_radps = radps_sum * 0.25f;
  for (unsigned int i = 0; i < 4U; i++) {
    rl_cfc_control_motor_rpm_cmd[i] = rpm_to_command(scaled_mapped_motor_rpm(i));
  }
}

void rl_cfc_control_apply_motor_rpm(bool motors_on)
{
  if (!motors_on || !rl_cfc_control_enabled) {
    return;
  }

  mean_mapped_raw_motor_rpm();
  for (uint8_t i = 0U; i < 4U; i++) {
    const int32_t rpm = rpm_to_command(scaled_mapped_motor_rpm(i));
    rl_cfc_control_applied_rpm_cmd[i] = rpm;
    apply_motor_rpm(i, rpm);
  }
}
