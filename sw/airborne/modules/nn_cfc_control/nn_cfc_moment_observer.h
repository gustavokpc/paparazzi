#ifndef NN_CFC_MOMENT_OBSERVER_H
#define NN_CFC_MOMENT_OBSERVER_H

/*
 * External-moment observer used by the Bebop2 NN-CfC controller.
 * All vectors use the controller/training body frame and SI units:
 * velocity [m/s], body rates [rad/s], rotor speed [RPM], moment [N m].
 */

void nn_cfc_moment_observer_reset(void);

void nn_cfc_model_moment_nm(const float velocity_body[3],
                            const float rates_body[3],
                            const float rpm[4],
                            float modeled_moment_nm[3]);

void nn_cfc_moment_observer_update(const float velocity_body[3],
                                   const float rates_body[3],
                                   const float rpm[4],
                                   float dt,
                                   float measured_moment_nm[3],
                                   float modeled_moment_nm[3],
                                   float external_moment_nm[3],
                                   float filtered_rates_rad_s[3]);

#endif /* NN_CFC_MOMENT_OBSERVER_H */
