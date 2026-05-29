/*
File: rl_cfc_parameters.h
Generated from: Quadcopter3DGatesGym-v0/recurrent_ppo_cfc_baseline/recurrent_ppo_cfc_baseline.zip
Model kind: cfc
Input: raw RL observation vector used by the Stable-Baselines3 policy.
*/
#ifndef RECURRENT_PPO_CFC_BASELINE_RL_CFC_PARAMETERS_H
#define RECURRENT_PPO_CFC_BASELINE_RL_CFC_PARAMETERS_H

#define OBS_SIZE 20
#define NUM_CONTROLS 4
#define HIDDEN_SIZE 64

/* Constants used by deterministic actor inference, flattened in PyTorch row-major order. */
/* mlp_extractor.policy_net.0.weight: (64, 64) */
extern const float mlp_extractor_policy_net_0_weight[4096];
/* mlp_extractor.policy_net.0.bias: (64,) */
extern const float mlp_extractor_policy_net_0_bias[64];
/* mlp_extractor.policy_net.2.weight: (64, 64) */
extern const float mlp_extractor_policy_net_2_weight[4096];
/* mlp_extractor.policy_net.2.bias: (64,) */
extern const float mlp_extractor_policy_net_2_bias[64];
/* action_net.weight: (4, 64) */
extern const float action_net_weight[256];
/* action_net.bias: (4,) */
extern const float action_net_bias[4];
/* lstm_actor.rnn_cell.backbone.0.weight: (128, 84) */
extern const float lstm_actor_rnn_cell_backbone_0_weight[10752];
/* lstm_actor.rnn_cell.backbone.0.bias: (128,) */
extern const float lstm_actor_rnn_cell_backbone_0_bias[128];
/* lstm_actor.rnn_cell.ff1.weight: (64, 128) */
extern const float lstm_actor_rnn_cell_ff1_weight[8192];
/* lstm_actor.rnn_cell.ff1.bias: (64,) */
extern const float lstm_actor_rnn_cell_ff1_bias[64];
/* lstm_actor.rnn_cell.ff2.weight: (64, 128) */
extern const float lstm_actor_rnn_cell_ff2_weight[8192];
/* lstm_actor.rnn_cell.ff2.bias: (64,) */
extern const float lstm_actor_rnn_cell_ff2_bias[64];
/* lstm_actor.rnn_cell.time_a.weight: (64, 128) */
extern const float lstm_actor_rnn_cell_time_a_weight[8192];
/* lstm_actor.rnn_cell.time_a.bias: (64,) */
extern const float lstm_actor_rnn_cell_time_a_bias[64];
/* lstm_actor.rnn_cell.time_b.weight: (64, 128) */
extern const float lstm_actor_rnn_cell_time_b_weight[8192];
/* lstm_actor.rnn_cell.time_b.bias: (64,) */
extern const float lstm_actor_rnn_cell_time_b_bias[64];

#endif /* RECURRENT_PPO_CFC_BASELINE_RL_CFC_PARAMETERS_H */
