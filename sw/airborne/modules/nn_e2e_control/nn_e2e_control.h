#ifndef NN_E2E_CONTROL_H
#define NN_E2E_CONTROL_H

#include <stdbool.h>
#include <stdint.h>

extern bool nn_e2e_control_enabled;
extern unsigned int nn_e2e_control_waypoint_index;
extern float nn_e2e_control_last_norm[4];
extern float nn_e2e_control_last_rpm[4];
extern int32_t nn_e2e_control_motor_pprz[4];
extern int32_t nn_e2e_control_applied_pprz[4];
extern int32_t nn_e2e_control_raw_mean_pprz;
extern unsigned int nn_e2e_control_periodic_count;
extern float nn_e2e_control_reached_radius_m;
extern int32_t nn_e2e_control_output_min_pprz;
extern int32_t nn_e2e_control_output_max_pprz;
extern bool nn_e2e_control_use_ned_input;
extern uint8_t nn_e2e_control_motor_src[4];
extern bool nn_e2e_control_motor_invert[4];
extern float nn_e2e_control_target[3];
extern float nn_e2e_control_error[3];
extern float nn_e2e_control_input_error[3];
extern float nn_e2e_control_input_velocity[3];
extern float nn_e2e_control_dist_to_target;

void nn_e2e_control_init(void);
void nn_e2e_control_start(void);
void nn_e2e_control_stop(void);
void nn_e2e_control_periodic(void);
int32_t nn_e2e_control_get_motor_pprz(uint8_t motor_idx, int32_t autopilot_pprz);

#endif
