#ifndef IWT_H
#define IWT_H

#include <stddef.h>
#include <stdbool.h>
#include <ocl/ocl.h>

typedef struct iwt_config
{
    size_t N;
    size_t BATCH_SIZE;
    double D;
    double l0;
    double T;
    double alpha_IWT;
    double beta_IWT;
    double delta_t;

    double DT;
    double ETA;
    double LAMBDA;

    double ETA_Q;
    double LAMBDA_Q;
    double GAMMA_Q;

    double THRESHOLD;
    int MAX_ITER;
} *iwt_config_t;

typedef struct iwt_runtime
{
    double *I;
    double *K;
    double *sumJ;
    double *Q;

    cl_mem I_gpu;
    cl_mem K_gpu;
    cl_mem sumJ_gpu;
    cl_mem Q_gpu;

    struct ocl_core ocl;
} *iwt_runtime_t;

double iwt_pi(void);
double iwt_fundamental_length(void);
double iwt_fundamental_time();
double iwt_alpha_IWT();
double iwt_fractal_dimension(void);
double iwt_I_min();
double iwt_I_max(void);
double iwt_alpha_IWT();
double iwt_beta_IWT();

#endif
