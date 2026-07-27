#ifndef BEBOP2_MATLAB_AERO_H
#define BEBOP2_MATLAB_AERO_H

/*
 * Header-only port of
 * LNN_behavioural_cloning_quadrotor/utils/dynamics_models/
 * quadrotor_sim_matlab.py::forces_moments().
 *
 * This file deliberately has no Gazebo or Paparazzi dependencies so the
 * numerical model can be regression-tested against Python.
 */

#include <algorithm>
#include <array>
#include <cmath>

namespace bebop2_matlab_aero {

struct Vec3 {
  double x;
  double y;
  double z;
};

struct Wrench {
  Vec3 force;
  Vec3 moment;
};

constexpr double kRho = 1.225;
constexpr double kRadius = 0.075;
constexpr double kArmX = 0.088;
constexpr double kArmY = 0.115;
constexpr double kArea = 3.14159265358979323846 * kRadius * kRadius;
constexpr double kReferenceArea = 4.0 * kArmY * kArmX;
constexpr double kRpmToRadS = 2.0 * 3.14159265358979323846 / 60.0;

constexpr std::array<double, 16> kCt = {{
  1.5608699335679425e-02, -5.5203712207002993e-02,
  6.8373739471780537e-01, -2.2386746543744325e00,
  3.0487551954496670e00, -1.5151394628763879e00,
  -1.4452997585238964e-02, 4.5664172755604832e-01,
  -5.2451229104942632e-01, 2.3311292320225827e-01,
  -2.5847908175511535e-02, 4.0081035350267295e-02,
  -1.1555207086648811e-02, -2.2312195771809176e-03,
  -2.2536635249275513e-02, 3.3603579143813935e-03
}};

constexpr std::array<double, 16> kCq = {{
  -2.2666283737348665e-03, -1.1272325229483371e-03,
  3.6755812531864369e-03, -1.0084483009293548e-01,
  2.2598950398682635e-01, -1.4625704645524937e-01,
  -3.0488011051989740e-03, -7.4756001696528445e-03,
  -1.1112951725382153e-01, 1.2154218606659822e-01,
  3.3562635661743183e-03, 3.6340289707626175e-03,
  -7.2880066350763410e-03, 1.1629568968266250e-03,
  2.5734357204199219e-03, -6.8056363992066531e-04
}};

constexpr std::array<Vec3, 4> kRotorPosition = {{
  { kArmX, -kArmY, 0.0 },
  { kArmX,  kArmY, 0.0 },
  {-kArmX,  kArmY, 0.0 },
  {-kArmX, -kArmY, 0.0 }
}};
constexpr std::array<double, 4> kSl = {{1.0, -1.0, -1.0, 1.0}};
constexpr std::array<double, 4> kSm = {{1.0, 1.0, -1.0, -1.0}};
/* SIGNR=-1 from the Python model. */
constexpr std::array<double, 4> kSn = {{-1.0, 1.0, -1.0, 1.0}};

inline Vec3 add(const Vec3 &a, const Vec3 &b)
{
  return {a.x + b.x, a.y + b.y, a.z + b.z};
}

inline Vec3 cross(const Vec3 &a, const Vec3 &b)
{
  return {
    a.y * b.z - a.z * b.y,
    a.z * b.x - a.x * b.z,
    a.x * b.y - a.y * b.x
  };
}

inline double norm(const Vec3 &v)
{
  return std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}

inline double safe_sign(double value)
{
  return value > 0.0 ? 1.0 : (value < 0.0 ? -1.0 : 0.0);
}

inline std::array<double, 16> p52(double alpha, double mu)
{
  const double mu2 = mu * mu;
  const double mu3 = mu2 * mu;
  const double mu4 = mu3 * mu;
  const double alpha2 = alpha * alpha;
  return {{
    1.0, mu, mu2, mu3, mu4, mu4 * mu,
    mu * alpha, mu2 * alpha, mu3 * alpha, mu4 * alpha,
    mu * alpha2, mu2 * alpha2, mu3 * alpha2,
    mu * alpha2 * alpha, mu2 * alpha2 * alpha,
    mu * alpha2 * alpha2
  }};
}

template <std::size_t N>
inline double dot(const std::array<double, N> &a,
                  const std::array<double, N> &b)
{
  double result = 0.0;
  for (std::size_t i = 0; i < N; ++i) {
    result += a[i] * b[i];
  }
  return result;
}

inline double p1n2(double x, double scale,
                   const std::array<double, 2> &coeff)
{
  return scale * (x * coeff[0] + x * x * coeff[1]);
}

inline double p32(double x1, double x2, double scale,
                  const std::array<double, 6> &coeff)
{
  const std::array<double, 6> terms = {{
    x1, x1 * x1, x1 * x2, x1 * x1 * x1,
    x1 * x2 * x2, x1 * x1 * x2
  }};
  return scale * dot(terms, coeff);
}

inline Wrench forces_moments(const Vec3 &velocity_body,
                             const Vec3 &rates_body,
                             const std::array<double, 4> &rpm,
                             double rotor_yaw_sign = -1.0)
{
  constexpr std::array<double, 2> kModel1 = {{8.3830000000000005e-01, -2.8536999999999999e00}};
  constexpr std::array<double, 2> kModel2 = {{3.0010201085253096e-02, -8.9189122199398146e-02}};
  constexpr std::array<double, 2> kModel3 = {{-5.0932092901974557e-01, 3.9966451853283669e-01}};
  constexpr double kModel4 = -3.9621480670528505e-05;
  constexpr double kModel5 = 2.2925907776961882e-05;
  constexpr double kModel6 = 4.6444946965533518e-06;
  constexpr double kModel7 = -9.6610559042848528e-07;
  constexpr std::array<double, 6> kModel8 = {{
    1.1744348774819557e-02, -4.9783056660455006e-04,
    1.6876894655524964e-02, -2.6481727161199176e-02,
    6.2434913619265037e-02, -4.5688818395152909e-02
  }};
  constexpr std::array<double, 6> kModel9 = {{
    -8.4625851462271237e-02, 2.7646286183395069e-01,
    1.0188252424024791e-01, -1.941857221941057e-01,
    3.6284183800335434e-02, 1.9786776362575691e-02
  }};
  constexpr std::array<double, 6> kModel10 = {{
    -3.0729246620140552e-02, 7.5908541159230708e-02,
    -1.3399020881700217e-02, -4.5066769504689422e-02,
    1.5521950307307686e-02, -5.7219661069848118e-03
  }};

  const double va = norm(velocity_body);
  double u_bar = 0.0;
  double v_bar = 0.0;
  double w_bar = 0.0;
  if (va >= 0.01) {
    u_bar = velocity_body.x / va;
    v_bar = velocity_body.y / va;
    w_bar = velocity_body.z / va;
  }

  std::array<double, 4> omega = {{0.0, 0.0, 0.0, 0.0}};
  std::array<double, 4> thrust = {{0.0, 0.0, 0.0, 0.0}};
  std::array<double, 4> x_force = {{0.0, 0.0, 0.0, 0.0}};
  std::array<double, 4> y_force = {{0.0, 0.0, 0.0, 0.0}};
  std::array<double, 4> l_moment = {{0.0, 0.0, 0.0, 0.0}};
  std::array<double, 4> m_moment = {{0.0, 0.0, 0.0, 0.0}};
  std::array<double, 4> n_moment = {{0.0, 0.0, 0.0, 0.0}};

  for (std::size_t i = 0; i < 4; ++i) {
    omega[i] = rpm[i] * kRpmToRadS;
    if (omega[i] <= 1.0) {
      omega[i] = 0.0;
    }
    const Vec3 rotor_velocity = add(cross(rates_body, kRotorPosition[i]), velocity_body);
    const double rotor_speed = norm(rotor_velocity);
    double mu = omega[i] == 0.0 ? 0.0 : rotor_speed / (omega[i] * kRadius);
    mu = std::max(0.0, std::min(0.6, mu));
    const double xy_speed = std::hypot(rotor_velocity.x, rotor_velocity.y);
    const double alpha = xy_speed == 0.0 ? 0.0 : std::atan(rotor_velocity.z / xy_speed);
    const double dynamic_head = kRho * omega[i] * omega[i] * kRadius * kRadius;
    const std::array<double, 16> polynomial = p52(alpha, mu);
    const double ct = dot(polynomial, kCt);
    const double cq = dot(polynomial, kCq);

    thrust[i] = ct * dynamic_head * kArea;
    x_force[i] = rotor_velocity.x * omega[i] * kModel4 +
                 kSn[i] * rotor_velocity.y * omega[i] * kModel5;
    y_force[i] = -kSn[i] * rotor_velocity.x * omega[i] * kModel5 +
                 rotor_velocity.y * omega[i] * kModel4;
    l_moment[i] = kSn[i] * rotor_velocity.x * omega[i] * kModel7 -
                  rotor_velocity.y * omega[i] * kModel6;
    m_moment[i] = rotor_velocity.x * omega[i] * kModel6 +
                  kSn[i] * rotor_velocity.y * omega[i] * kModel7;
    n_moment[i] = rotor_yaw_sign * kSn[i] * cq * dynamic_head * kArea * kRadius;
  }

  const double qbar_s = va * va * kReferenceArea * kRho / 2.0;
  const double t0 = -p1n2(std::abs(w_bar), qbar_s * safe_sign(w_bar), kModel1);
  const double x0 = p1n2(std::abs(u_bar), qbar_s * safe_sign(u_bar), kModel2);
  const double y0 = p1n2(std::abs(v_bar), qbar_s * safe_sign(v_bar), kModel3);
  const double l0 = p32(std::abs(v_bar), w_bar, qbar_s * safe_sign(v_bar), kModel8);
  const double m0 = p32(std::abs(u_bar), w_bar, qbar_s * safe_sign(u_bar), kModel9);
  const double n0 = p32(std::abs(v_bar), std::abs(u_bar),
                        qbar_s * safe_sign(v_bar) * safe_sign(u_bar), kModel10);

  Wrench result = {{x0, y0, -t0}, {l0, m0, n0}};
  for (std::size_t i = 0; i < 4; ++i) {
    result.force.x += x_force[i];
    result.force.y += y_force[i];
    result.force.z -= thrust[i];
    result.moment.x += kSl[i] * kArmY * thrust[i] + l_moment[i];
    result.moment.y += kSm[i] * kArmX * thrust[i] + m_moment[i];
    result.moment.z += kArmY * kSl[i] * x_force[i] +
                       kArmX * kSm[i] * y_force[i] + n_moment[i];
  }
  return result;
}

inline double first_order_step(double current, double target,
                               double tau, double dt)
{
  if (tau <= 0.0 || dt <= 0.0) {
    return target;
  }
  return target + (current - target) * std::exp(-dt / tau);
}

} // namespace bebop2_matlab_aero

#endif // BEBOP2_MATLAB_AERO_H
