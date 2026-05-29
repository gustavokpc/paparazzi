#ifndef RECURRENT_PPO_CFC_BASELINE_RL_CFC_OPERATIONS_H
#define RECURRENT_PPO_CFC_BASELINE_RL_CFC_OPERATIONS_H

#include <math.h>
#include "rl_cfc_parameters.h"

void rl_cfc_reset(void);
void rl_cfc_control(const float *obs, float *action);

#endif /* RECURRENT_PPO_CFC_BASELINE_RL_CFC_OPERATIONS_H */
