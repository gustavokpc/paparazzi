/*
 * Copyright (C) 2014 Freek van Tienen <freek.v.tienen@gmail.com>
 *               2019 Tom van Dijk <tomvand@users.noreply.github.com>
 *
 * This file is part of paparazzi.
 *
 * paparazzi is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2, or (at your option)
 * any later version.
 *
 * paparazzi is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with paparazzi; see the file COPYING.  If not, write to
 * the Free Software Foundation, 59 Temple Place - Suite 330,
 * Boston, MA 02111-1307, USA.
 *
 */

/** @file modules/loggers/logger_file.c
 *  @brief File logger for Linux based autopilots
 */

#include "logger_file.h"

#include <stdio.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#include "std.h"

#include "autopilot.h"
#include "mcu_periph/sys_time.h"
#include "state.h"
#include "generated/airframe.h"
#include "modules/radio_control/radio_control.h"
#ifdef COMMAND_THRUST
#include "firmwares/rotorcraft/stabilization.h"
#else
#include "firmwares/fixedwing/stabilization/stabilization_attitude.h"
#include "firmwares/fixedwing/stabilization/stabilization_adaptive.h"
#endif

#include "generated/modules.h"
#ifdef MODULE_MOTOR_MIXING_ID
#include "modules/actuators/motor_mixing.h"
#endif
#ifdef BOARD_BEBOP
#include "boards/bebop/actuators.h"
#endif
#ifdef MODULE_NN_CFC_CONTROL_ID
#include "modules/nn_cfc_control/nn_cfc_control.h"
static const char *nn_cfc_input_names[19] = {
  "dx", "dy", "dz",
  "vx", "vy", "vz",
  "phi", "theta", "psi",
  "p", "q", "r",
  "Mx_ext", "My_ext", "Mz_ext",
  "omega1", "omega2", "omega3", "omega4"
};
#endif
#ifdef MODULE_RL_CFC_CONTROL_ID
#include "modules/rl_cfc_control/rl_cfc_control.h"
#endif

/** Set the default File logger path to the USB drive */
#ifndef LOGGER_FILE_PATH
#define LOGGER_FILE_PATH /data/video/usb
#endif

/** The file pointer */
static FILE *logger_file = NULL;
static uint32_t logger_file_last_row_us = 0U;


/** Logging functions */

/** Write CSV header
 * Write column names at the top of the CSV file. Make sure that the columns
 * match those in logger_file_write_row! Don't forget the \n at the end of the
 * line.
 * @param file Log file pointer
 */
static void logger_file_write_header(FILE *file) {
  fprintf(file, "time,log_dt_us,");
  fprintf(file, "autopilot_mode,motors_on,autopilot_in_flight,");
  fprintf(file, "rc_status,rc_time_since_last_frame,rc_frame_rate,");
  fprintf(file, "rc_roll,rc_pitch,rc_yaw,rc_throttle,rc_mode,");
  fprintf(file, "pos_x,pos_y,pos_z,");
  fprintf(file, "vel_x,vel_y,vel_z,");
  fprintf(file, "att_phi,att_theta,att_psi,");
  fprintf(file, "rate_p,rate_q,rate_r,");
#ifdef BOARD_BEBOP
  fprintf(file, "rpm_obs_1,rpm_obs_2,rpm_obs_3,rpm_obs_4,");
  fprintf(file, "rpm_ref_1,rpm_ref_2,rpm_ref_3,rpm_ref_4,");
#endif
#if defined(MODULE_MOTOR_MIXING_ID) && MOTOR_MIXING_NB_MOTOR >= 4
  fprintf(file, "mixer_cmd_1,mixer_cmd_2,mixer_cmd_3,mixer_cmd_4,");
#endif
#ifdef INS_EXT_POSE_H
  ins_ext_pos_log_header(file);
#endif
#ifdef MODULE_NN_CFC_CONTROL_ID
  fprintf(file, "nn_enabled,nn_periodic_count,nn_waypoint_index,nn_waypoint_switch_count,");
  fprintf(file, "nn_target_x,nn_target_y,nn_target_z,");
  fprintf(file, "nn_pos_x,nn_pos_y,nn_pos_z,");
  fprintf(file, "nn_err_x,nn_err_y,nn_err_z,");
  fprintf(file, "nn_dist,nn_dist_xy,nn_abs_z_error,");
  fprintf(file, "nn_mext_mode,nn_mext_est_x,nn_mext_est_y,nn_mext_est_z,");
  fprintf(file, "nn_mext_measured_x,nn_mext_measured_y,nn_mext_measured_z,");
  fprintf(file, "nn_mext_modeled_x,nn_mext_modeled_y,nn_mext_modeled_z,");
  fprintf(file, "nn_mext_filtered_p,nn_mext_filtered_q,nn_mext_filtered_r,nn_mext_time_us,");
  for (unsigned int i = 0; i < 19U; i++) {
    fprintf(file, "nn_in_%s_raw,", nn_cfc_input_names[i]);
  }
  for (unsigned int i = 0; i < 19U; i++) {
    fprintf(file, "nn_in_%s_normalized,", nn_cfc_input_names[i]);
  }
  fprintf(file, "network_out1_norm,network_out2_norm,network_out3_norm,network_out4_norm,");
  fprintf(file, "network_out1_raw,network_out2_raw,network_out3_raw,network_out4_raw,");
  fprintf(file, "rpm_cmd1,rpm_cmd2,rpm_cmd3,rpm_cmd4,");
  fprintf(file, "rpm_applied1,rpm_applied2,rpm_applied3,rpm_applied4,");
  fprintf(file, "nn_mean_norm,nn_mean_rpm_cmd,nn_raw_mean_rpm,");
  fprintf(file, "nn_periodic_dt_us,nn_sensor_read_time_us,nn_inference_time_us,nn_total_time_us,");
#endif
#ifdef MODULE_RL_CFC_CONTROL_ID
  fprintf(file, "rl_enabled,rl_periodic_count,rl_waypoint_index,");
  fprintf(file, "rl_target_x,rl_target_y,rl_target_z,");
  fprintf(file, "rl_err_x,rl_err_y,rl_err_z,");
  fprintf(file, "rl_dist,rl_mean_policy_rpm,rl_raw_mean_rpm,");
  fprintf(file, "rl_mext_mode,rl_mext_est_x,rl_mext_est_y,rl_mext_est_z,");
  fprintf(file, "rl_mext_measured_x,rl_mext_measured_y,rl_mext_measured_z,");
  fprintf(file, "rl_mext_modeled_x,rl_mext_modeled_y,rl_mext_modeled_z,");
  fprintf(file, "rl_mext_filtered_p,rl_mext_filtered_q,rl_mext_filtered_r,rl_mext_time_us,");
  for (unsigned int i = 0; i < 20U; i++) {
    fprintf(file, "rl_obs_%u,", i);
  }
  fprintf(file, "rl_policy_raw1,rl_policy_raw2,rl_policy_raw3,rl_policy_raw4,");
  fprintf(file, "rl_action1,rl_action2,rl_action3,rl_action4,");
  fprintf(file, "rl_motor_state1,rl_motor_state2,rl_motor_state3,rl_motor_state4,");
  fprintf(file, "rl_rpm_cmd1,rl_rpm_cmd2,rl_rpm_cmd3,rl_rpm_cmd4,");
  fprintf(file, "rl_rpm_applied1,rl_rpm_applied2,rl_rpm_applied3,rl_rpm_applied4,");
  fprintf(file, "rl_periodic_dt_us,rl_sensor_read_time_us,rl_inference_time_us,rl_total_time_us,");
#endif
#ifdef COMMAND_THRUST
  fprintf(file, "cmd_thrust,cmd_roll,cmd_pitch,cmd_yaw\n");
#else
  fprintf(file, "h_ctl_aileron_setpoint,h_ctl_elevator_setpoint\n");
#endif
}

/** Write CSV row
 * Write values at this timestamp to log file. Make sure that the printf's match
 * the column headers of logger_file_write_header! Don't forget the \n at the
 * end of the line.
 * @param file Log file pointer
 */
static void logger_file_write_row(FILE *file) {
  struct NedCoor_f *pos = stateGetPositionNed_f();
  struct NedCoor_f *vel = stateGetSpeedNed_f();
  struct FloatEulers *att = stateGetNedToBodyEulers_f();
  struct FloatRates *rates = stateGetBodyRates_f();
  const uint32_t row_time_us = get_sys_time_usec();
  const uint32_t log_dt_us = logger_file_last_row_us == 0U ? 0U : row_time_us - logger_file_last_row_us;
  logger_file_last_row_us = row_time_us;
  const pprz_t rc_roll =
#ifdef RADIO_ROLL
      radio_control_get(RADIO_ROLL);
#else
      0;
#endif
  const pprz_t rc_pitch =
#ifdef RADIO_PITCH
      radio_control_get(RADIO_PITCH);
#else
      0;
#endif
  const pprz_t rc_yaw =
#ifdef RADIO_YAW
      radio_control_get(RADIO_YAW);
#else
      0;
#endif
  const pprz_t rc_throttle =
#ifdef RADIO_THROTTLE
      radio_control_get(RADIO_THROTTLE);
#else
      0;
#endif
  const pprz_t rc_mode =
#ifdef RADIO_MODE
      radio_control_get(RADIO_MODE);
#else
      0;
#endif

  fprintf(file, "%f,%u,", get_sys_time_float(), log_dt_us);
  fprintf(file, "%u,%u,%u,",
      autopilot_get_mode(),
      autopilot_get_motors_on() ? 1U : 0U,
      autopilot_in_flight() ? 1U : 0U);
  fprintf(file, "%u,%u,%u,",
      radio_control.status,
      radio_control.time_since_last_frame,
      radio_control.frame_rate);
  fprintf(file, "%d,%d,%d,%d,%d,",
      rc_roll,
      rc_pitch,
      rc_yaw,
      rc_throttle,
      rc_mode);
  fprintf(file, "%f,%f,%f,", pos->x, pos->y, pos->z);
  fprintf(file, "%f,%f,%f,", vel->x, vel->y, vel->z);
  fprintf(file, "%f,%f,%f,", att->phi, att->theta, att->psi);
  fprintf(file, "%f,%f,%f,", rates->p, rates->q, rates->r);
#ifdef BOARD_BEBOP
  fprintf(file, "%d,%d,%d,%d,",actuators_bebop.rpm_obs[0],actuators_bebop.rpm_obs[1],actuators_bebop.rpm_obs[2],actuators_bebop.rpm_obs[3]);
  fprintf(file, "%d,%d,%d,%d,",actuators_bebop.rpm_ref[0],actuators_bebop.rpm_ref[1],actuators_bebop.rpm_ref[2],actuators_bebop.rpm_ref[3]);
#endif
#if defined(MODULE_MOTOR_MIXING_ID) && MOTOR_MIXING_NB_MOTOR >= 4
  fprintf(file, "%d,%d,%d,%d,",
      motor_mixing.commands[0],
      motor_mixing.commands[1],
      motor_mixing.commands[2],
      motor_mixing.commands[3]);
#endif
#ifdef INS_EXT_POSE_H
  ins_ext_pos_log_data(file);
#endif
#ifdef MODULE_NN_CFC_CONTROL_ID
  fprintf(file, "%u,%u,%u,%u,",
      nn_cfc_control_enabled ? 1U : 0U,
      nn_cfc_control_periodic_count,
      nn_cfc_control_waypoint_index,
      nn_cfc_control_waypoint_switch_count);
  fprintf(file, "%f,%f,%f,",
      nn_cfc_control_target[0],
      nn_cfc_control_target[1],
      nn_cfc_control_target[2]);
  fprintf(file, "%f,%f,%f,",
      nn_cfc_control_position[0],
      nn_cfc_control_position[1],
      nn_cfc_control_position[2]);
  fprintf(file, "%f,%f,%f,",
      nn_cfc_control_error[0],
      nn_cfc_control_error[1],
      nn_cfc_control_error[2]);
  fprintf(file, "%f,%f,%f,",
      nn_cfc_control_dist_to_target,
      nn_cfc_control_dist_xy_to_target,
      nn_cfc_control_abs_z_error);
  fprintf(file, "%u,%f,%f,%f,",
      (unsigned int)nn_cfc_control_mext_mode,
      nn_cfc_control_external_moment_nm[0],
      nn_cfc_control_external_moment_nm[1],
      nn_cfc_control_external_moment_nm[2]);
  fprintf(file, "%f,%f,%f,",
      nn_cfc_control_measured_moment_nm[0],
      nn_cfc_control_measured_moment_nm[1],
      nn_cfc_control_measured_moment_nm[2]);
  fprintf(file, "%f,%f,%f,",
      nn_cfc_control_modeled_moment_nm[0],
      nn_cfc_control_modeled_moment_nm[1],
      nn_cfc_control_modeled_moment_nm[2]);
  fprintf(file, "%f,%f,%f,%u,",
      nn_cfc_control_observer_filtered_rates[0],
      nn_cfc_control_observer_filtered_rates[1],
      nn_cfc_control_observer_filtered_rates[2],
      nn_cfc_control_mext_time_us);
  for (unsigned int i = 0; i < 19U; i++) {
    fprintf(file, "%f,", nn_cfc_control_net_input_state[i]);
  }
  for (unsigned int i = 0; i < 19U; i++) {
    fprintf(file, "%f,", nn_cfc_control_net_input_normalized[i]);
  }
  fprintf(file, "%f,%f,%f,%f,",
      nn_cfc_control_net_output_norm[0],
      nn_cfc_control_net_output_norm[1],
      nn_cfc_control_net_output_norm[2],
      nn_cfc_control_net_output_norm[3]);
  fprintf(file, "%f,%f,%f,%f,",
      nn_cfc_control_net_output_raw[0],
      nn_cfc_control_net_output_raw[1],
      nn_cfc_control_net_output_raw[2],
      nn_cfc_control_net_output_raw[3]);
  fprintf(file, "%d,%d,%d,%d,",
      nn_cfc_control_motor_rpm_cmd[0],
      nn_cfc_control_motor_rpm_cmd[1],
      nn_cfc_control_motor_rpm_cmd[2],
      nn_cfc_control_motor_rpm_cmd[3]);
  fprintf(file, "%d,%d,%d,%d,",
      nn_cfc_control_applied_rpm_cmd[0],
      nn_cfc_control_applied_rpm_cmd[1],
      nn_cfc_control_applied_rpm_cmd[2],
      nn_cfc_control_applied_rpm_cmd[3]);
  fprintf(file, "%f,%d,%d,",
      nn_cfc_control_mean_norm,
      nn_cfc_control_mean_rpm_cmd,
      nn_cfc_control_raw_mean_rpm);
  fprintf(file, "%u,%u,%u,%u,",
      nn_cfc_control_periodic_dt_us,
      nn_cfc_control_sensor_read_time_us,
      nn_cfc_control_inference_time_us,
      nn_cfc_control_total_time_us);
#endif
#ifdef MODULE_RL_CFC_CONTROL_ID
  fprintf(file, "%u,%u,%u,",
      rl_cfc_control_enabled ? 1U : 0U,
      rl_cfc_control_periodic_count,
      rl_cfc_control_waypoint_index);
  fprintf(file, "%f,%f,%f,",
      rl_cfc_control_target[0],
      rl_cfc_control_target[1],
      rl_cfc_control_target[2]);
  fprintf(file, "%f,%f,%f,",
      rl_cfc_control_error[0],
      rl_cfc_control_error[1],
      rl_cfc_control_error[2]);
  fprintf(file, "%f,%f,%d,",
      rl_cfc_control_dist_to_target,
      rl_cfc_control_mean_policy_rpm,
      rl_cfc_control_raw_mean_rpm);
  fprintf(file, "%u,%f,%f,%f,",
      (unsigned int)rl_cfc_control_mext_mode,
      rl_cfc_control_external_moment_nm[0],
      rl_cfc_control_external_moment_nm[1],
      rl_cfc_control_external_moment_nm[2]);
  fprintf(file, "%f,%f,%f,",
      rl_cfc_control_measured_moment_nm[0],
      rl_cfc_control_measured_moment_nm[1],
      rl_cfc_control_measured_moment_nm[2]);
  fprintf(file, "%f,%f,%f,",
      rl_cfc_control_modeled_moment_nm[0],
      rl_cfc_control_modeled_moment_nm[1],
      rl_cfc_control_modeled_moment_nm[2]);
  fprintf(file, "%f,%f,%f,%u,",
      rl_cfc_control_observer_filtered_rates[0],
      rl_cfc_control_observer_filtered_rates[1],
      rl_cfc_control_observer_filtered_rates[2],
      rl_cfc_control_mext_time_us);
  for (unsigned int i = 0; i < 20U; i++) {
    fprintf(file, "%f,", rl_cfc_control_obs[i]);
  }
  fprintf(file, "%f,%f,%f,%f,",
      rl_cfc_control_policy_raw_action[0],
      rl_cfc_control_policy_raw_action[1],
      rl_cfc_control_policy_raw_action[2],
      rl_cfc_control_policy_raw_action[3]);
  fprintf(file, "%f,%f,%f,%f,",
      rl_cfc_control_action[0],
      rl_cfc_control_action[1],
      rl_cfc_control_action[2],
      rl_cfc_control_action[3]);
  fprintf(file, "%f,%f,%f,%f,",
      rl_cfc_control_motor_state[0],
      rl_cfc_control_motor_state[1],
      rl_cfc_control_motor_state[2],
      rl_cfc_control_motor_state[3]);
  fprintf(file, "%d,%d,%d,%d,",
      rl_cfc_control_motor_rpm_cmd[0],
      rl_cfc_control_motor_rpm_cmd[1],
      rl_cfc_control_motor_rpm_cmd[2],
      rl_cfc_control_motor_rpm_cmd[3]);
  fprintf(file, "%d,%d,%d,%d,",
      rl_cfc_control_applied_rpm_cmd[0],
      rl_cfc_control_applied_rpm_cmd[1],
      rl_cfc_control_applied_rpm_cmd[2],
      rl_cfc_control_applied_rpm_cmd[3]);
  fprintf(file, "%u,%u,%u,%u,",
      rl_cfc_control_periodic_dt_us,
      rl_cfc_control_sensor_read_time_us,
      rl_cfc_control_inference_time_us,
      rl_cfc_control_total_time_us);
#endif
#ifdef COMMAND_THRUST
  fprintf(file, "%d,%d,%d,%d\n",
      stabilization.cmd[COMMAND_THRUST], stabilization.cmd[COMMAND_ROLL],
      stabilization.cmd[COMMAND_PITCH], stabilization.cmd[COMMAND_YAW]);
#else
  fprintf(file, "%d,%d\n", h_ctl_aileron_setpoint, h_ctl_elevator_setpoint);
#endif
}


/** Start the file logger and open a new file */
void logger_file_start(void)
{
  if (logger_file != NULL) {
    return;
  }

  // Ensure that the module is running when started with this function
  logger_file_logger_file_periodic_status = MODULES_RUN;
  
  // Create output folder if necessary
  if (access(STRINGIFY(LOGGER_FILE_PATH), F_OK)) {
    char save_dir_cmd[256];
    sprintf(save_dir_cmd, "mkdir -p %s", STRINGIFY(LOGGER_FILE_PATH));
    if (system(save_dir_cmd) != 0) {
      printf("[logger_file] Could not create log file directory %s.\n", STRINGIFY(LOGGER_FILE_PATH));
      return;
    }
  }

  // Get current date/time for filename
  char date_time[80];
  time_t now = time(0);
  struct tm  tstruct;
  tstruct = *localtime(&now);
  strftime(date_time, sizeof(date_time), "%Y%m%d-%H%M%S", &tstruct);

  uint32_t counter = 0;
  char filename[512];

  // Check for available files
  sprintf(filename, "%s/%s.csv", STRINGIFY(LOGGER_FILE_PATH), date_time);
  while ((logger_file = fopen(filename, "r"))) {
    fclose(logger_file);

    sprintf(filename, "%s/%s_%05d.csv", STRINGIFY(LOGGER_FILE_PATH), date_time, counter);
    counter++;
  }

  logger_file = fopen(filename, "w");
  if(!logger_file) {
    printf("[logger_file] ERROR opening log file %s!\n", filename);
    return;
  }

  printf("[logger_file] Start logging to %s...\n", filename);

  logger_file_last_row_us = 0U;
  logger_file_write_header(logger_file);
}

/** Stop the logger an nicely close the file */
void logger_file_stop(void)
{
  if (logger_file != NULL) {
    fclose(logger_file);
    logger_file = NULL;
    logger_file_last_row_us = 0U;
  }
}

/** Log the values to a csv file    */
void logger_file_periodic(void)
{
  if (logger_file == NULL) {
    return;
  }
  logger_file_write_row(logger_file);
}
