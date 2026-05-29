#ifndef NN_CFC_OPERATIONS_H
#define NN_CFC_OPERATIONS_H

#include <math.h>
#include "nn_cfc_parameters.h"

void nn_cfc_reset(void);
void nn_cfc_control(const float *state, float *control);

#endif
