#ifndef IWT_H
#define IWT_H

#include <stddef.h>
#include <stdbool.h>
#include <ocl/ocl.h>

typedef struct iwt_config
{
    size_t N;
    size_t BATCH_SIZE;
    double DT;
    double ETA;
    double LAMBDA;
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

#endif
