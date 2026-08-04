/*
 * External-moment observer for the Bebop2 NN-CfC controller.
 *
 * M_measured = I * dOmega/dt + Omega x (I * Omega)
 * M_external = M_measured - M_modeled
 *
 * Omega is filtered with an 8 Hz, second-order Butterworth low-pass before
 * differentiation, matching the onboard method described by the source paper.
 * M_modeled is the moment part of quadrotor_sim_matlab.py::forces_moments().
 */

#include "modules/nn_cfc_control/nn_cfc_moment_observer.h"

#include "filters/low_pass_filter.h"

#include <math.h>
#include <stdbool.h>
#include <stddef.h>

#define NN_CFC_OBSERVER_SAMPLE_TIME_S 0.01f
#define NN_CFC_OBSERVER_CUTOFF_HZ 8.0f
#define NN_CFC_OBSERVER_Q 0.7071067811865475f
#define NN_CFC_PI 3.14159265358979323846f

#define NN_CFC_IXX 0.001920f
#define NN_CFC_IYY 0.001850f
#define NN_CFC_IZZ 0.003340f

#define NN_CFC_RHO 1.225f
#define NN_CFC_ROTOR_RADIUS_M 0.075f
#define NN_CFC_ARM_X_M 0.088f
#define NN_CFC_ARM_Y_M 0.115f
#define NN_CFC_ROTOR_AREA_M2 (NN_CFC_PI * NN_CFC_ROTOR_RADIUS_M * NN_CFC_ROTOR_RADIUS_M)
#define NN_CFC_REFERENCE_AREA_M2 (4.0f * NN_CFC_ARM_Y_M * NN_CFC_ARM_X_M)
#define NN_CFC_RPM_TO_RAD_S (2.0f * NN_CFC_PI / 60.0f)
#define NN_CFC_ROTOR_YAW_SIGN 1.0f

static const float nn_cfc_ct[16] = {
  1.5608699335679425e-02f, -5.5203712207002993e-02f,
  6.8373739471780537e-01f, -2.2386746543744325e00f,
  3.0487551954496670e00f, -1.5151394628763879e00f,
  -1.4452997585238964e-02f, 4.5664172755604832e-01f,
  -5.2451229104942632e-01f, 2.3311292320225827e-01f,
  -2.5847908175511535e-02f, 4.0081035350267295e-02f,
  -1.1555207086648811e-02f, -2.2312195771809176e-03f,
  -2.2536635249275513e-02f, 3.3603579143813935e-03f
};

static const float nn_cfc_cq[16] = {
  -2.2666283737348665e-03f, -1.1272325229483371e-03f,
  3.6755812531864369e-03f, -1.0084483009293548e-01f,
  2.2598950398682635e-01f, -1.4625704645524937e-01f,
  -3.0488011051989740e-03f, -7.4756001696528445e-03f,
  -1.1112951725382153e-01f, 1.2154218606659822e-01f,
  3.3562635661743183e-03f, 3.6340289707626175e-03f,
  -7.2880066350763410e-03f, 1.1629568968266250e-03f,
  2.5734357204199219e-03f, -6.8056363992066531e-04f
};

static const float nn_cfc_model8[6] = {
  1.1744348774819557e-02f, -4.9783056660455006e-04f,
  1.6876894655524964e-02f, -2.6481727161199176e-02f,
  6.2434913619265037e-02f, -4.5688818395152909e-02f
};

static const float nn_cfc_model9[6] = {
  -8.4625851462271237e-02f, 2.7646286183395224e-01f,
  1.0188252424024791e-01f, -1.9418572213526053e-01f,
  3.6284183800335434e-02f, 1.9786776362575691e-02f
};

static const float nn_cfc_model10[6] = {
  -3.0729246620140552e-02f, 7.5908541159230708e-02f,
  -1.3399020881700217e-02f, -4.5066769504689422e-02f,
  1.5521950307307686e-02f, -5.7219661069848118e-03f
};

/* Network order: front-left, front-right, back-right, back-left. */
static const float nn_cfc_rotor_x[4] = {
  NN_CFC_ARM_X_M, NN_CFC_ARM_X_M, -NN_CFC_ARM_X_M, -NN_CFC_ARM_X_M
};
static const float nn_cfc_rotor_y[4] = {
  -NN_CFC_ARM_Y_M, NN_CFC_ARM_Y_M, NN_CFC_ARM_Y_M, -NN_CFC_ARM_Y_M
};
static const float nn_cfc_sl[4] = {1.0f, -1.0f, -1.0f, 1.0f};
static const float nn_cfc_sm[4] = {1.0f, 1.0f, -1.0f, -1.0f};
static const float nn_cfc_sn[4] = {-1.0f, 1.0f, -1.0f, 1.0f};

static Butterworth2LowPass nn_cfc_rate_filter[3];
static float nn_cfc_previous_filtered_rate[3];
static bool nn_cfc_observer_initialized;

static float nn_cfc_safe_sign(float value)
{
  return value > 0.0f ? 1.0f : (value < 0.0f ? -1.0f : 0.0f);
}

static float nn_cfc_clamp(float value, float minimum, float maximum)
{
  if (value < minimum) {
    return minimum;
  }
  if (value > maximum) {
    return maximum;
  }
  return value;
}

static float nn_cfc_dot16(const float a[16], const float b[16])
{
  float result = 0.0f;
  for (size_t i = 0U; i < 16U; i++) {
    result += a[i] * b[i];
  }
  return result;
}

static float nn_cfc_p32(float x1, float x2, float scale, const float coefficients[6])
{
  const float x1_2 = x1 * x1;
  const float terms[6] = {
    x1,
    x1_2,
    x1 * x2,
    x1_2 * x1,
    x1 * x2 * x2,
    x1_2 * x2
  };
  float result = 0.0f;
  for (size_t i = 0U; i < 6U; i++) {
    result += terms[i] * coefficients[i];
  }
  return scale * result;
}

static void nn_cfc_p52(float alpha, float mu, float terms[16])
{
  const float mu2 = mu * mu;
  const float mu3 = mu2 * mu;
  const float mu4 = mu3 * mu;
  const float alpha2 = alpha * alpha;
  terms[0] = 1.0f;
  terms[1] = mu;
  terms[2] = mu2;
  terms[3] = mu3;
  terms[4] = mu4;
  terms[5] = mu4 * mu;
  terms[6] = mu * alpha;
  terms[7] = mu2 * alpha;
  terms[8] = mu3 * alpha;
  terms[9] = mu4 * alpha;
  terms[10] = mu * alpha2;
  terms[11] = mu2 * alpha2;
  terms[12] = mu3 * alpha2;
  terms[13] = mu * alpha2 * alpha;
  terms[14] = mu2 * alpha2 * alpha;
  terms[15] = mu * alpha2 * alpha2;
}

void nn_cfc_model_moment_nm(const float velocity_body[3],
                            const float rates_body[3],
                            const float rpm[4],
                            float modeled_moment_nm[3])
{
  const float u = velocity_body[0];
  const float v = velocity_body[1];
  const float w = velocity_body[2];
  const float p = rates_body[0];
  const float q = rates_body[1];
  const float r = rates_body[2];
  const float airspeed = sqrtf(u * u + v * v + w * w);
  float u_bar = 0.0f;
  float v_bar = 0.0f;
  float w_bar = 0.0f;
  if (airspeed >= 0.01f) {
    u_bar = u / airspeed;
    v_bar = v / airspeed;
    w_bar = w / airspeed;
  }

  float mx = 0.0f;
  float my = 0.0f;
  float mz = 0.0f;
  for (size_t i = 0U; i < 4U; i++) {
    float omega = rpm[i] * NN_CFC_RPM_TO_RAD_S;
    if (omega <= 1.0f) {
      omega = 0.0f;
    }

    /* velocity at rotor i: v_body + Omega x r_i */
    const float rotor_u = u - r * nn_cfc_rotor_y[i];
    const float rotor_v = v + r * nn_cfc_rotor_x[i];
    const float rotor_w = w + p * nn_cfc_rotor_y[i] - q * nn_cfc_rotor_x[i];
    const float rotor_speed = sqrtf(rotor_u * rotor_u + rotor_v * rotor_v + rotor_w * rotor_w);
    float mu = omega == 0.0f ? 0.0f : rotor_speed / (omega * NN_CFC_ROTOR_RADIUS_M);
    mu = nn_cfc_clamp(mu, 0.0f, 0.6f);
    const float xy_speed = hypotf(rotor_u, rotor_v);
    const float alpha = xy_speed == 0.0f ? 0.0f : atanf(rotor_w / xy_speed);

    float polynomial[16];
    nn_cfc_p52(alpha, mu, polynomial);
    const float ct = nn_cfc_dot16(polynomial, nn_cfc_ct);
    const float cq = nn_cfc_dot16(polynomial, nn_cfc_cq);
    const float dynamic_head = NN_CFC_RHO * omega * omega *
                               NN_CFC_ROTOR_RADIUS_M * NN_CFC_ROTOR_RADIUS_M;
    const float thrust = ct * dynamic_head * NN_CFC_ROTOR_AREA_M2;
    const float x_force = rotor_u * omega * -3.9621480670528505e-05f +
                          nn_cfc_sn[i] * rotor_v * omega * 2.2925907776961882e-05f;
    const float y_force = -nn_cfc_sn[i] * rotor_u * omega * 2.2925907776961882e-05f +
                          rotor_v * omega * -3.9621480670528505e-05f;
    const float l_moment = nn_cfc_sn[i] * rotor_u * omega * -9.6610559042848528e-07f -
                           rotor_v * omega * 4.6444946965533518e-06f;
    const float m_moment = rotor_u * omega * 4.6444946965533518e-06f +
                           nn_cfc_sn[i] * rotor_v * omega * -9.6610559042848528e-07f;
    const float n_moment = NN_CFC_ROTOR_YAW_SIGN * nn_cfc_sn[i] * cq *
                           dynamic_head * NN_CFC_ROTOR_AREA_M2 * NN_CFC_ROTOR_RADIUS_M;

    mx += nn_cfc_sl[i] * NN_CFC_ARM_Y_M * thrust + l_moment;
    my += nn_cfc_sm[i] * NN_CFC_ARM_X_M * thrust + m_moment;
    mz += NN_CFC_ARM_Y_M * nn_cfc_sl[i] * x_force +
          NN_CFC_ARM_X_M * nn_cfc_sm[i] * y_force + n_moment;
  }

  const float qbar_s = airspeed * airspeed * NN_CFC_REFERENCE_AREA_M2 * NN_CFC_RHO / 2.0f;
  mx += nn_cfc_p32(fabsf(v_bar), w_bar,
                   qbar_s * nn_cfc_safe_sign(v_bar), nn_cfc_model8);
  my += nn_cfc_p32(fabsf(u_bar), w_bar,
                   qbar_s * nn_cfc_safe_sign(u_bar), nn_cfc_model9);
  mz += nn_cfc_p32(fabsf(v_bar), fabsf(u_bar),
                   qbar_s * nn_cfc_safe_sign(v_bar) * nn_cfc_safe_sign(u_bar),
                   nn_cfc_model10);

  modeled_moment_nm[0] = mx;
  modeled_moment_nm[1] = my;
  modeled_moment_nm[2] = mz;
}

void nn_cfc_moment_observer_reset(void)
{
  nn_cfc_observer_initialized = false;
  for (size_t i = 0U; i < 3U; i++) {
    nn_cfc_previous_filtered_rate[i] = 0.0f;
  }
}

void nn_cfc_moment_observer_update(const float velocity_body[3],
                                   const float rates_body[3],
                                   const float rpm[4],
                                   float dt,
                                   float measured_moment_nm[3],
                                   float modeled_moment_nm[3],
                                   float external_moment_nm[3],
                                   float filtered_rates_rad_s[3])
{
  if (dt < 0.002f || dt > 0.05f) {
    dt = NN_CFC_OBSERVER_SAMPLE_TIME_S;
  }

  if (!nn_cfc_observer_initialized) {
    const float tau = 1.0f / (2.0f * NN_CFC_PI * NN_CFC_OBSERVER_CUTOFF_HZ);
    for (size_t i = 0U; i < 3U; i++) {
      init_second_order_low_pass(&nn_cfc_rate_filter[i], tau, NN_CFC_OBSERVER_Q,
                                 NN_CFC_OBSERVER_SAMPLE_TIME_S, rates_body[i]);
      nn_cfc_previous_filtered_rate[i] = rates_body[i];
      filtered_rates_rad_s[i] = rates_body[i];
      measured_moment_nm[i] = 0.0f;
      external_moment_nm[i] = 0.0f;
    }
    nn_cfc_model_moment_nm(velocity_body, rates_body, rpm, modeled_moment_nm);
    nn_cfc_observer_initialized = true;
    return;
  }

  float angular_acceleration[3];
  for (size_t i = 0U; i < 3U; i++) {
    filtered_rates_rad_s[i] = update_second_order_low_pass(&nn_cfc_rate_filter[i], rates_body[i]);
    angular_acceleration[i] =
        (filtered_rates_rad_s[i] - nn_cfc_previous_filtered_rate[i]) / dt;
    nn_cfc_previous_filtered_rate[i] = filtered_rates_rad_s[i];
  }

  const float p = filtered_rates_rad_s[0];
  const float q = filtered_rates_rad_s[1];
  const float r = filtered_rates_rad_s[2];
  measured_moment_nm[0] = NN_CFC_IXX * angular_acceleration[0] +
                          (NN_CFC_IZZ - NN_CFC_IYY) * q * r;
  measured_moment_nm[1] = NN_CFC_IYY * angular_acceleration[1] +
                          (NN_CFC_IXX - NN_CFC_IZZ) * p * r;
  measured_moment_nm[2] = NN_CFC_IZZ * angular_acceleration[2] +
                          (NN_CFC_IYY - NN_CFC_IXX) * p * q;

  /* Use the same filtered rates in the aerodynamic model for consistency. */
  nn_cfc_model_moment_nm(velocity_body, filtered_rates_rad_s, rpm, modeled_moment_nm);
  for (size_t i = 0U; i < 3U; i++) {
    external_moment_nm[i] = measured_moment_nm[i] - modeled_moment_nm[i];
  }
}
