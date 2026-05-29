#ifndef RL_CFC_CONTROL_H
#define RL_CFC_CONTROL_H

#include <stdbool.h>
#include <stdint.h>

extern bool rl_cfc_control_enabled;
extern unsigned int rl_cfc_control_waypoint_index;
extern float rl_cfc_control_last_norm[4];
extern float rl_cfc_control_last_rpm[4];
extern int32_t rl_cfc_control_motor_pprz[4];
extern int32_t rl_cfc_control_applied_pprz[4];
extern int32_t rl_cfc_control_raw_mean_pprz;
extern unsigned int rl_cfc_control_periodic_count;
extern float rl_cfc_control_reached_radius_m;
extern int32_t rl_cfc_control_output_min_pprz;
extern int32_t rl_cfc_control_output_max_pprz;
extern bool rl_cfc_control_use_ned_input;
extern uint8_t rl_cfc_control_motor_src[4];
extern bool rl_cfc_control_motor_invert[4];
extern float rl_cfc_control_target[3];
extern float rl_cfc_control_error[3];
extern float rl_cfc_control_input_error[3];
extern float rl_cfc_control_input_velocity[3];
extern float rl_cfc_control_feedback_rpm[4];
extern float rl_cfc_control_motor_state[4];
extern float rl_cfc_control_commanded_radps[4];
extern float rl_cfc_control_actuator_scale;
extern float rl_cfc_control_raw_action[4];
extern float rl_cfc_control_action[4];
extern float rl_cfc_control_policy_raw_action[4];
extern float rl_cfc_control_obs[20];
extern float rl_cfc_control_next_gate[4];
extern float rl_cfc_control_gate_yaw[4];
extern float rl_cfc_control_dist_to_target;

void rl_cfc_control_init(void);
void rl_cfc_control_start(void);
void rl_cfc_control_stop(void);
void rl_cfc_control_periodic(void);
int32_t rl_cfc_control_get_motor_pprz(uint8_t motor_idx, int32_t autopilot_pprz);

#endif
