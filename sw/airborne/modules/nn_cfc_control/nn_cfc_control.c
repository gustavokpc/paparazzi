/*
 * nn_cfc_control.c
 * Paparazzi wrapper for an end-to-end NN controller.
 *
 * IMPORTANT:
 * - This file builds the 19-element input expected by nn_cfc_parameters.h:
 *   dx, dy, dz, vx, vy, vz, phi, theta, psi, p, q, r,
 *   Mx_ext, My_ext, Mz_ext, omega1, omega2, omega3, omega4
 * - dx,dy,dz are the controller position errors in the trained body frame.
 *   Paparazzi gives local position/velocity as ENU; this wrapper first maps it
 *   to the model world frame {Y, X, -Z}, then applies the same world->body
 *   rotation used by the Python C-controller simulator.
 * - nn_cfc_control() returns normalized motor commands in [0,1]. The wrapper
 *   keeps the trained RPM scale and applies RPM commands directly on Bebop.
 * - The airframe command_laws keep the normal autopilot actuator path as a
 *   fallback; nn_cfc_control_apply_motor_rpm() overwrites it only when the NN is
 *   enabled.
 */

#include "modules/nn_cfc_control/nn_cfc_control.h"
#include "modules/nn_cfc_control/nn_cfc_operations.h"

#include "paparazzi.h"
#include "state.h"
#include "generated/airframe.h"
#include "generated/flight_plan.h"
#include "generated/modules.h"
#include "modules/nav/waypoints.h"

#ifdef MODULE_LOGGER_FILE_ID
#include "modules/loggers/logger_file.h"
#endif

#ifdef BOARD_BEBOP
#include "boards/bebop/actuators.h"
#endif

#if defined(USE_NPS) && USE_NPS
#include "nps_fdm.h"
#endif

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>

#if PERIODIC_TELEMETRY
#include "modules/datalink/telemetry.h"
#endif

/*
 * Controller configuration.
 * Change the NN_CFC_* values here; conf/modules/nn_cfc_control.xml only exposes
 * debug/telemetry variables so there is a single source of truth.
 */
#ifndef NN_CFC_REACHED_RADIUS_M
#define NN_CFC_REACHED_RADIUS_M 0.1f
#endif

#ifndef NN_CFC_TARGET_ALT_M
#define NN_CFC_TARGET_ALT_M 1.5f
#endif

#ifndef NN_CFC_START_WAYPOINT_INDEX
#define NN_CFC_START_WAYPOINT_INDEX 3
#endif

/* RPM is the NN training scale and the Bebop BLDC command unit. */
#ifndef NN_CFC_MIN_RPM
#define NN_CFC_MIN_RPM 5000.0f
#endif

#ifndef NN_CFC_MAX_RPM
#define NN_CFC_MAX_RPM 10000.0f
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

bool nn_cfc_control_enabled = false;
unsigned int nn_cfc_control_waypoint_index = 0;
float nn_cfc_control_last_norm[4] = {0.f, 0.f, 0.f, 0.f};
float nn_cfc_control_last_rpm[4] = {0.f, 0.f, 0.f, 0.f};
int32_t nn_cfc_control_motor_rpm_cmd[4] = {0, 0, 0, 0};
int32_t nn_cfc_control_applied_rpm_cmd[4] = {0, 0, 0, 0};
int32_t nn_cfc_control_raw_mean_rpm = 0;
unsigned int nn_cfc_control_periodic_count = 0U;
unsigned int nn_cfc_control_waypoint_switch_count = 0U;
float nn_cfc_control_reached_radius_m = NN_CFC_REACHED_RADIUS_M;
float nn_cfc_control_target[3] = {0.f, 0.f, 0.f};
float nn_cfc_control_position[3] = {0.f, 0.f, 0.f};
float nn_cfc_control_error[3] = {0.f, 0.f, 0.f};
float nn_cfc_control_input_error[3] = {0.f, 0.f, 0.f};
float nn_cfc_control_input_velocity[3] = {0.f, 0.f, 0.f};
float nn_cfc_control_external_moment_nm[3] = {0.f, 0.f, 0.f};
float nn_cfc_control_feedback_rpm[4] = {0.f, 0.f, 0.f, 0.f};
float nn_cfc_control_attitude[3] = {0.f, 0.f, 0.f};
float nn_cfc_control_body_rates[3] = {0.f, 0.f, 0.f};
float nn_cfc_control_dist_to_target = 0.f;
float nn_cfc_control_dist_xy_to_target = 0.f;
float nn_cfc_control_abs_z_error = 0.f;
float nn_cfc_control_mean_norm = 0.f;
int32_t nn_cfc_control_mean_rpm_cmd = 0;
/* Debug: network raw inputs and outputs */
float nn_cfc_control_net_output_norm[4] = {0.f, 0.f, 0.f, 0.f};
float nn_cfc_control_net_output_raw[4] = {0.f, 0.f, 0.f, 0.f};
float nn_cfc_control_net_input_state[19] = {0.f};
float nn_cfc_control_net_input_normalized[19] = {0.f};
float nn_cfc_control_log_values[38] = {0.f};

/* Network outputs use Paparazzi's QUAD_X motor order: front-left, front-right, back-right, back-left. */
static const uint8_t nn_cfc_net_to_phys_motor[4] = {
  MOTOR_FRONT_LEFT,
  MOTOR_FRONT_RIGHT,
  MOTOR_BACK_RIGHT,
  MOTOR_BACK_LEFT
};

/* ENU coordinates that become the trained 3x4m NED rectangle after conversion. */
static const nn_vec3_t nn_square_waypoints[] = {
  { 2.0f,  1.5f, NN_CFC_TARGET_ALT_M},
  { 2.0f, -1.5f, NN_CFC_TARGET_ALT_M},
  {-2.0f, -1.5f, NN_CFC_TARGET_ALT_M},
  {-2.0f,  1.5f, NN_CFC_TARGET_ALT_M},
};
static const unsigned int nn_num_square_waypoints = sizeof(nn_square_waypoints) / sizeof(nn_square_waypoints[0]);

static nn_vec3_t get_square_waypoint(unsigned int index)
{
  /*
   * Prefer the flight-plan waypoints when they exist, so changes in the XML
   * mission are reflected here. The static array above is only a fallback for
   * builds/tests that do not define WP_NN_SQ_*.
   */
#if defined(WP_NN_SQ_1) && defined(WP_NN_SQ_2) && defined(WP_NN_SQ_3) && defined(WP_NN_SQ_4)
  static const uint8_t wp_ids[] = {WP_NN_SQ_1, WP_NN_SQ_2, WP_NN_SQ_3, WP_NN_SQ_4};
  const uint8_t wp_id = wp_ids[index % (sizeof(wp_ids) / sizeof(wp_ids[0]))];
  return (nn_vec3_t){waypoint_get_x(wp_id), waypoint_get_y(wp_id), waypoint_get_alt(wp_id)};
#else
  return nn_square_waypoints[index % nn_num_square_waypoints];
#endif
}

__attribute__((weak)) nn_vec3_t nn_cfc_get_position_m(void)
{
  /*
   * Weak wrappers make the module testable: a unit test can override these
   * functions without touching Paparazzi state internals.
   */
  const struct EnuCoor_f *pos = stateGetPositionEnu_f();
  return (nn_vec3_t){pos->x, pos->y, pos->z};
}

__attribute__((weak)) nn_vec3_t nn_cfc_get_velocity_mps(void)
{
  const struct EnuCoor_f *vel = stateGetSpeedEnu_f();
  return (nn_vec3_t){vel->x, vel->y, vel->z};
}

__attribute__((weak)) nn_euler_t nn_cfc_get_attitude_rad(void)
{
  const struct FloatEulers *att = stateGetNedToBodyEulers_f();
  return (nn_euler_t){att->phi, att->theta, att->psi};
}

__attribute__((weak)) nn_vec3_t nn_cfc_get_body_rates_radps(void)
{
  const struct FloatRates *rates = stateGetBodyRates_f();
  return (nn_vec3_t){rates->p, rates->q, rates->r};
}

__attribute__((weak)) nn_vec3_t nn_cfc_get_external_moment_nm(void)
{
  return (nn_vec3_t){
    nn_cfc_control_external_moment_nm[0],
    nn_cfc_control_external_moment_nm[1],
    nn_cfc_control_external_moment_nm[2]
  };
}

__attribute__((weak)) void nn_cfc_get_motor_feedback_rpm(float omega_rpm[4])
{
#ifdef BOARD_BEBOP
  /* Real Bebop builds can feed measured motor RPM back into the network state. */
  for (uint8_t i = 0U; i < 4U; i++) {
    omega_rpm[i] = (float)actuators_bebop.rpm_obs[nn_cfc_net_to_phys_motor[i]];
  }
#else
  /*
   * In NPS/Gazebo, nps_fdm_gazebo exposes the actuator state in fdm.rpm[].
   * Fall back to the previous command until the FDM has produced valid RPMs.
   */
#if defined(USE_NPS) && USE_NPS
  if (fdm.num_engines >= 4U) {
    for (uint8_t i = 0U; i < 4U; i++) {
      const uint8_t phys_idx = nn_cfc_net_to_phys_motor[i];
      omega_rpm[i] = fdm.rpm[phys_idx] > 0.f ? fdm.rpm[phys_idx] : nn_cfc_control_last_rpm[i];
    }
  } else
#endif
  {
    omega_rpm[0] = nn_cfc_control_last_rpm[0];
    omega_rpm[1] = nn_cfc_control_last_rpm[1];
    omega_rpm[2] = nn_cfc_control_last_rpm[2];
    omega_rpm[3] = nn_cfc_control_last_rpm[3];
  }
#endif
}

static float clampf(float x, float lo, float hi)
{
  if (x < lo) return lo;
  if (x > hi) return hi;
  return x;
}

static void clamp_state_to_training_range(float state[NUM_STATES])
{
  /*
   * The exported network normalizes each input using input_norm_min/max. Clamp
   * before inference so a sensor spike does not push the normalized value far
   * outside the distribution used when exporting/training the model.
   */
  for (unsigned int i = 0U; i < NUM_STATES; i++) {
    state[i] = clampf(state[i], input_norm_min[i], input_norm_max[i]);
  }
}

static void normalize_state_for_debug(const float state[NUM_STATES])
{
  /* Mirror the network normalization so telemetry/logs show raw and normalized inputs. */
  for (unsigned int i = 0U; i < NUM_STATES; i++) {
    nn_cfc_control_net_input_normalized[i] =
        (state[i] - input_norm_min[i]) / (input_norm_max[i] - input_norm_min[i] + 1.0e-10f);
  }
}

static void update_log_values(void)
{
  /* DEBUG_VECT carries raw inputs first, then normalized inputs. */
  for (unsigned int i = 0U; i < NUM_STATES; i++) {
    nn_cfc_control_log_values[i] = nn_cfc_control_net_input_state[i];
    nn_cfc_control_log_values[NUM_STATES + i] = nn_cfc_control_net_input_normalized[i];
  }
}

#if PERIODIC_TELEMETRY
static void nn_cfc_control_send_telemetry(struct transport_tx *trans, struct link_device *dev)
{
  char name[] = "NN_CFC_INPUTS";
  pprz_msg_send_DEBUG_VECT(trans, dev, AC_ID,
                           strlen(name), name,
                           38, nn_cfc_control_log_values);
}
#endif

static nn_vec3_t to_network_frame_vec(const nn_vec3_t enu)
{
  /* Paparazzi ENU -> network world frame used by the exported CFC model. */
  return (nn_vec3_t){enu.y, enu.x, -enu.z};
}

static nn_vec3_t rotate_world_to_body_vec(const nn_vec3_t world, const nn_euler_t att)
{
  /*
   * Same transform as utils/quadrotor_sim.py: world_to_body_state().
   * The Python simulator builds R = Rz(psi) * Ry(theta) * Rx(phi), then uses
   * R.T for position error and velocity before passing them to the C export.
   */
  const float cphi = cosf(att.phi);
  const float sphi = sinf(att.phi);
  const float ctheta = cosf(att.theta);
  const float stheta = sinf(att.theta);
  const float cpsi = cosf(att.psi);
  const float spsi = sinf(att.psi);

  const float x1 = cpsi * world.x + spsi * world.y;
  const float y1 = -spsi * world.x + cpsi * world.y;
  const float z1 = world.z;

  const float x2 = ctheta * x1 - stheta * z1;
  const float y2 = y1;
  const float z2 = stheta * x1 + ctheta * z1;

  return (nn_vec3_t){
    x2,
    cphi * y2 + sphi * z2,
    -sphi * y2 + cphi * z2
  };
}

static int32_t rpm_to_command(float rpm)
{
  /* Keep commands inside the RPM range used during training/export. */
  return (int32_t)clampf(rpm, NN_CFC_MIN_RPM, NN_CFC_MAX_RPM);
}

static void update_motor_command_debug(void)
{
  /* Mean of calibrated NN RPMs, useful as a quick hover/throttle sanity check. */
  int32_t sum = 0;
  for (uint8_t i = 0U; i < 4U; i++) {
    sum += rpm_to_command(nn_cfc_control_last_rpm[i]);
  }
  nn_cfc_control_raw_mean_rpm = sum / 4;
}

static void apply_motor_rpm(uint8_t motor_idx, int32_t rpm)
{
#ifndef BOARD_BEBOP
  /*
   * NPS/Gazebo reads nn_cfc_control_motor_rpm_cmd[] directly when
   * NPS_NN_CFC_DIRECT_RPM is enabled in the nps airframe target.
   */
  (void)motor_idx;
  (void)rpm;
#else
  /* Real Bebop path: the actuator driver accepts RPM-like motor references. */
  actuators_bebop_set(motor_idx, (int16_t)rpm);
#endif
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
  /*
   * Keep debug values in Paparazzi ENU coordinates. These are the values the
   * operator sees: target, current position, and target-position error.
   */
  nn_cfc_control_target[0] = wp.x;
  nn_cfc_control_target[1] = wp.y;
  nn_cfc_control_target[2] = wp.z;

  nn_cfc_control_position[0] = pos.x;
  nn_cfc_control_position[1] = pos.y;
  nn_cfc_control_position[2] = pos.z;

  nn_cfc_control_error[0] = wp.x - pos.x;
  nn_cfc_control_error[1] = wp.y - pos.y;
  nn_cfc_control_error[2] = wp.z - pos.z;
  nn_cfc_control_dist_to_target = sqrtf(waypoint_dist2(pos, wp));
  nn_cfc_control_dist_xy_to_target = sqrtf(nn_cfc_control_error[0] * nn_cfc_control_error[0] +
                                           nn_cfc_control_error[1] * nn_cfc_control_error[1]);
  nn_cfc_control_abs_z_error = fabsf(nn_cfc_control_error[2]);
}

static void reset_target_debug(void)
{
  for (unsigned int i = 0; i < 3U; i++) {
    nn_cfc_control_target[i] = 0.f;
    nn_cfc_control_position[i] = 0.f;
    nn_cfc_control_error[i] = 0.f;
    nn_cfc_control_input_error[i] = 0.f;
    nn_cfc_control_input_velocity[i] = 0.f;
  }
  nn_cfc_control_dist_to_target = 0.f;
  nn_cfc_control_dist_xy_to_target = 0.f;
  nn_cfc_control_abs_z_error = 0.f;
}

static void maybe_advance_waypoint(const nn_vec3_t pos)
{
  /*
   * Waypoint switching is separate from the neural control itself. The network
   * always receives an error to the current active waypoint; this function only
   * decides when to move to the next waypoint in the rectangle.
   */
  const nn_vec3_t wp = get_square_waypoint(nn_cfc_control_waypoint_index);
  const float d2 = waypoint_dist2(pos, wp);
  const float reached_radius = nn_cfc_control_reached_radius_m > 0.f ? nn_cfc_control_reached_radius_m : 0.05f;
  if (d2 < reached_radius * reached_radius) {
    nn_cfc_control_waypoint_index = (nn_cfc_control_waypoint_index + 1U) % nn_num_square_waypoints;
    nn_cfc_control_waypoint_switch_count++;
    // nn_cfc_reset(); // dont know
  }
  update_target_debug(pos, get_square_waypoint(nn_cfc_control_waypoint_index));
}

void nn_cfc_control_init(void)
{
  /* Module initialization: clear all exported state and reset CFC memory. */
  nn_cfc_control_enabled = false;
  nn_cfc_control_waypoint_index = 0U;
  nn_cfc_control_periodic_count = 0U;
  nn_cfc_control_waypoint_switch_count = 0U;
  for (unsigned int i = 0; i < 4U; i++) {
    nn_cfc_control_last_norm[i] = 0.f;
    nn_cfc_control_last_rpm[i] = 0.f;
    nn_cfc_control_motor_rpm_cmd[i] = 0;
    nn_cfc_control_applied_rpm_cmd[i] = 0;
    nn_cfc_control_feedback_rpm[i] = 0.f;
    nn_cfc_control_net_output_norm[i] = 0.f;
    nn_cfc_control_net_output_raw[i] = 0.f;
  }
  for (unsigned int i = 0; i < 3U; i++) {
    nn_cfc_control_external_moment_nm[i] = 0.f;
  }
  for (unsigned int i = 0; i < NUM_STATES; i++) {
    nn_cfc_control_net_input_state[i] = 0.f;
    nn_cfc_control_net_input_normalized[i] = 0.f;
  }
  for (unsigned int i = 0; i < 38U; i++) {
    nn_cfc_control_log_values[i] = 0.f;
  }
  nn_cfc_control_raw_mean_rpm = 0;
  reset_target_debug();
  nn_cfc_reset();
#ifdef MODULE_LOGGER_FILE_ID
  logger_file_start();
#endif
#if PERIODIC_TELEMETRY
  register_periodic_telemetry(DefaultPeriodic, PPRZ_MSG_ID_DEBUG_VECT, nn_cfc_control_send_telemetry);
#endif
}

void nn_cfc_control_start(void)
{
  /*
   * Start is called by the flight plan. Resetting the recurrent state here is
   * important because the CFC hidden state should not carry stale information
   * from a previous manual/standby segment.
   */
  nn_cfc_control_enabled = true;
  nn_cfc_control_periodic_count = 0U;
  nn_cfc_control_waypoint_switch_count = 0U;
  for (unsigned int i = 0; i < 4U; i++) {
    nn_cfc_control_last_norm[i] = 0.f;
    nn_cfc_control_last_rpm[i] = NN_CFC_MIN_RPM;
    nn_cfc_control_motor_rpm_cmd[i] = 0;
    nn_cfc_control_applied_rpm_cmd[i] = 0;
    nn_cfc_control_feedback_rpm[i] = NN_CFC_MIN_RPM;
  }
  nn_cfc_control_raw_mean_rpm = 0;
  nn_cfc_control_waypoint_index = NN_CFC_START_WAYPOINT_INDEX % nn_num_square_waypoints;
  update_target_debug(nn_cfc_get_position_m(), get_square_waypoint(nn_cfc_control_waypoint_index));
  nn_cfc_reset();
#ifdef MODULE_LOGGER_FILE_ID
  logger_file_start();
#endif
}

void nn_cfc_control_stop(void)
{
  /* Stop neural control and return future command_law calls to autopilot. */
  nn_cfc_control_enabled = false;
  for (unsigned int i = 0; i < 4U; i++) {
    nn_cfc_control_last_norm[i] = 0.f;
    nn_cfc_control_last_rpm[i] = 0.f;
    nn_cfc_control_motor_rpm_cmd[i] = 0;
    nn_cfc_control_applied_rpm_cmd[i] = 0;
    nn_cfc_control_feedback_rpm[i] = 0.f;
  }
  nn_cfc_control_raw_mean_rpm = 0;
  reset_target_debug();
  nn_cfc_reset();
#ifdef MODULE_LOGGER_FILE_ID
  logger_file_stop();
#endif
}

void nn_cfc_control_periodic(void)
{
  if (!nn_cfc_control_enabled) {
    return;
  }
  nn_cfc_control_periodic_count++;

  /*
   * 1. Update waypoint bookkeeping and compute the current ENU error. This is
   *    navigation/debug state; motor commands are still produced by the NN.
   */
  const nn_vec3_t pos = nn_cfc_get_position_m();
  maybe_advance_waypoint(pos);

  const nn_vec3_t vel = nn_cfc_get_velocity_mps();
  const nn_vec3_t err = {
    nn_cfc_control_error[0],
    nn_cfc_control_error[1],
    nn_cfc_control_error[2]
  };
  const nn_euler_t att = nn_cfc_get_attitude_rad();

  /*
   * 2. Convert position error and velocity into the frame used during training:
   *    Paparazzi ENU -> network world frame -> body frame.
   */
  const nn_vec3_t nn_world_err = to_network_frame_vec(err);
  const nn_vec3_t nn_world_vel = to_network_frame_vec(vel);
  const nn_vec3_t nn_err = rotate_world_to_body_vec(nn_world_err, att);
  const nn_vec3_t nn_vel = rotate_world_to_body_vec(nn_world_vel, att);
  nn_cfc_control_input_error[0] = nn_err.x;
  nn_cfc_control_input_error[1] = nn_err.y;
  nn_cfc_control_input_error[2] = nn_err.z;
  nn_cfc_control_input_velocity[0] = nn_vel.x;
  nn_cfc_control_input_velocity[1] = nn_vel.y;
  nn_cfc_control_input_velocity[2] = nn_vel.z;
  const nn_vec3_t rates = nn_cfc_get_body_rates_radps();
  nn_cfc_control_attitude[0] = att.phi;
  nn_cfc_control_attitude[1] = att.theta;
  nn_cfc_control_attitude[2] = att.psi;
  nn_cfc_control_body_rates[0] = rates.x;
  nn_cfc_control_body_rates[1] = rates.y;
  nn_cfc_control_body_rates[2] = rates.z;
  const nn_vec3_t mext = nn_cfc_get_external_moment_nm();
  float omega[4];
  nn_cfc_get_motor_feedback_rpm(omega);
  for (unsigned int i = 0; i < 4U; i++) {
    nn_cfc_control_feedback_rpm[i] = omega[i];
  }

  /*
   * 3. Assemble the exact 19-value vector expected by nn_cfc_parameters.h.
   *    Changing this order changes the meaning of the NN input.
   */
  float state[NUM_STATES] = {
    nn_err.x, nn_err.y, nn_err.z,
    nn_vel.x, nn_vel.y, nn_vel.z,
    att.phi, att.theta, att.psi,
    rates.x, rates.y, rates.z,
    mext.x, mext.y, mext.z,
    omega[0], omega[1], omega[2], omega[3]
  };
  // clamp_state_to_training_range(state);  Disable to test rob

  /* 4. Save raw/normalized inputs before inference for logger_file and telemetry. */
  for (unsigned int i = 0; i < NUM_STATES; i++) {
    nn_cfc_control_net_input_state[i] = state[i];
  }
  normalize_state_for_debug(state);

  /* 5. Run the exported CFC network. Output is normalized motor command [0,1]. */
  float normalized_cmd[NUM_CONTROLS];
  nn_cfc_control(state, normalized_cmd);

  /* 6. Convert normalized outputs directly to the trained RPM command range. */
  for (unsigned int i = 0; i < 4U; i++) {
    nn_cfc_control_last_norm[i] = clampf(normalized_cmd[i], 0.f, 1.f);  // already clamped
    nn_cfc_control_net_output_norm[i] = normalized_cmd[i];
    nn_cfc_control_net_output_raw[i] = nn_cfc_last_raw_control[i];
    const float trained_rpm =
        NN_CFC_MIN_RPM +
        nn_cfc_control_last_norm[i] * (NN_CFC_MAX_RPM - NN_CFC_MIN_RPM);
    nn_cfc_control_last_rpm[i] = trained_rpm;
    nn_cfc_control_motor_rpm_cmd[nn_cfc_net_to_phys_motor[i]] = rpm_to_command(nn_cfc_control_last_rpm[i]);
  }

  /* 7. Update summary values and telemetry vector used by the normal CSV log. */
  float norm_sum = 0.f;
  int32_t rpm_sum = 0;
  for (unsigned int i = 0; i < 4U; i++) {
    norm_sum += nn_cfc_control_last_norm[i];
    rpm_sum += nn_cfc_control_motor_rpm_cmd[i];
  }
  nn_cfc_control_mean_norm = norm_sum / 4.f;
  nn_cfc_control_mean_rpm_cmd = rpm_sum / 4;
  update_log_values();
}

void nn_cfc_control_apply_motor_rpm(bool motors_on)
{
  /*
   * The normal command_laws run first and keep takeoff/landing fallback alive.
   * When neural control is enabled, overwrite each motor with the NN-derived
   * reference. In NPS the simulator reads the RPM command array directly; on
   * Bebop it is sent to the actuator driver as RPM.
   */
  if (!motors_on || !nn_cfc_control_enabled) {
    return;
  }

  update_motor_command_debug();
  for (uint8_t i = 0U; i < 4U; i++) {
    const uint8_t phys_idx = nn_cfc_net_to_phys_motor[i];
    const int32_t rpm = rpm_to_command(nn_cfc_control_last_rpm[i]);
    nn_cfc_control_applied_rpm_cmd[phys_idx] = rpm;
    apply_motor_rpm(phys_idx, rpm);
  }
}
