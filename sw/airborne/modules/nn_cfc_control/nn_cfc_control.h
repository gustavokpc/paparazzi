#ifndef NN_CFC_CONTROL_H
#define NN_CFC_CONTROL_H

#include <stdbool.h>
#include <stdint.h>

extern bool nn_cfc_control_enabled;
extern unsigned int nn_cfc_control_waypoint_index;
extern float nn_cfc_control_last_norm[4];
extern float nn_cfc_control_last_rpm[4];
extern int32_t nn_cfc_control_motor_pprz[4];
extern int32_t nn_cfc_control_applied_pprz[4];
extern int32_t nn_cfc_control_raw_mean_pprz;
extern unsigned int nn_cfc_control_periodic_count;
extern float nn_cfc_control_reached_radius_m;
extern int32_t nn_cfc_control_output_min_pprz;
extern int32_t nn_cfc_control_output_max_pprz;
extern bool nn_cfc_control_swap_xy_input;
extern bool nn_cfc_control_invert_z_input;
extern uint8_t nn_cfc_control_motor_src[4];
extern bool nn_cfc_control_motor_invert[4];
extern float nn_cfc_control_target[3];
extern float nn_cfc_control_position[3];
extern float nn_cfc_control_error[3];
extern float nn_cfc_control_input_error[3];
extern float nn_cfc_control_input_velocity[3];
extern float nn_cfc_control_external_moment_nm[3];
extern float nn_cfc_control_feedback_rpm[4];
extern float nn_cfc_control_dist_to_target;
extern float nn_cfc_control_dist_xy_to_target;
extern float nn_cfc_control_abs_z_error;
extern float nn_cfc_control_mean_norm;
extern int32_t nn_cfc_control_mean_pprz;

void nn_cfc_control_init(void);
void nn_cfc_control_start(void);
void nn_cfc_control_stop(void);
void nn_cfc_control_periodic(void);
int32_t nn_cfc_control_get_motor_pprz(uint8_t motor_idx, int32_t autopilot_pprz);

#endif
