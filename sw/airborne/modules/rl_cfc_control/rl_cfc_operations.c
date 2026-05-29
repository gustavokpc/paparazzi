#include "rl_cfc_operations.h"

static float sigmoidf_local(float x) {
    if (x >= 0.0f) {
        float z = expf(-x);
        return 1.0f / (1.0f + z);
    }
    float z = expf(x);
    return z / (1.0f + z);
}

static float lecun_tanhf(float x) {
    return 1.7159f * tanhf(0.666f * x);
}

static void clip_action(float *action) {
    for (int i = 0; i < NUM_CONTROLS; ++i) {
        if (action[i] < -1.0f) action[i] = -1.0f;
        if (action[i] > 1.0f) action[i] = 1.0f;
    }
}

static void matvec(const float * restrict x, float * restrict y, const float * restrict w, const float * restrict b, int in_dim, int out_dim) {
    for (int o = 0; o < out_dim; ++o) {
        float acc = b[o];
        for (int i = 0; i < in_dim; ++i) {
            acc += w[o * in_dim + i] * x[i];
        }
        y[o] = acc;
    }
}

static void tanh_inplace(float *x, int n) {
    for (int i = 0; i < n; ++i) x[i] = tanhf(x[i]);
}


#define RECURRENT_TIMESPAN 0.01f

static float rl_cfc_hidden[HIDDEN_SIZE];

void rl_cfc_reset(void) {
    for (int i = 0; i < HIDDEN_SIZE; ++i) rl_cfc_hidden[i] = 0.0f;
}

static void cfc_step(const float *obs) {
    float cat[OBS_SIZE + HIDDEN_SIZE];
    float backbone[128];
    float ff1[HIDDEN_SIZE];
    float ff2[HIDDEN_SIZE];
    float time_a[HIDDEN_SIZE];
    float time_b[HIDDEN_SIZE];
    for (int i = 0; i < OBS_SIZE; ++i) cat[i] = obs[i];
    for (int i = 0; i < HIDDEN_SIZE; ++i) cat[OBS_SIZE + i] = rl_cfc_hidden[i];
    matvec(cat, backbone, lstm_actor_rnn_cell_backbone_0_weight, lstm_actor_rnn_cell_backbone_0_bias, OBS_SIZE + HIDDEN_SIZE, 128);
    for (int i = 0; i < 128; ++i) backbone[i] = lecun_tanhf(backbone[i]);
    matvec(backbone, ff1, lstm_actor_rnn_cell_ff1_weight, lstm_actor_rnn_cell_ff1_bias, 128, HIDDEN_SIZE);
    matvec(backbone, ff2, lstm_actor_rnn_cell_ff2_weight, lstm_actor_rnn_cell_ff2_bias, 128, HIDDEN_SIZE);
    matvec(backbone, time_a, lstm_actor_rnn_cell_time_a_weight, lstm_actor_rnn_cell_time_a_bias, 128, HIDDEN_SIZE);
    matvec(backbone, time_b, lstm_actor_rnn_cell_time_b_weight, lstm_actor_rnn_cell_time_b_bias, 128, HIDDEN_SIZE);
    for (int i = 0; i < HIDDEN_SIZE; ++i) {
        float h1 = tanhf(ff1[i]);
        float h2 = tanhf(ff2[i]);
        float gate = sigmoidf_local(time_a[i] * RECURRENT_TIMESPAN + time_b[i]);
        rl_cfc_hidden[i] = h1 * (1.0f - gate) + gate * h2;
    }
}

void rl_cfc_control(const float *obs, float *action) {
    float buf_a[64];
    float buf_b[64];
    cfc_step(obs);
    const float *current = rl_cfc_hidden;
    float *next = buf_a;
    matvec(current, next, mlp_extractor_policy_net_0_weight, mlp_extractor_policy_net_0_bias, 64, 64);
    tanh_inplace(next, 64);
    current = next;
    next = (next == buf_a) ? buf_b : buf_a;
    matvec(current, next, mlp_extractor_policy_net_2_weight, mlp_extractor_policy_net_2_bias, 64, 64);
    tanh_inplace(next, 64);
    matvec(next, action, action_net_weight, action_net_bias, HIDDEN_SIZE, NUM_CONTROLS);
    clip_action(action);
}
