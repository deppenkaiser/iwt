#include "init.h"
#include <stdio.h>
#include <math.h>

bool initialize_host_data(const iwt_runtime_t rt, const iwt_config_t cfg)
{
    bool retval = false;

    rt->I = malloc(cfg->N * sizeof(double));
    rt->I_prev = malloc(cfg->N * sizeof(double));
    rt->K = malloc(cfg->N * cfg->N * sizeof(double));
    rt->sumJ = malloc(cfg->N * sizeof(double));
    rt->Q = malloc(cfg->N * sizeof(double));

    if ((rt->I != NULL) && (rt->I_prev != NULL) && (rt->K != NULL) && (rt->sumJ != NULL) && (rt->Q != NULL))
    {
        // Einheitliches Vakuum (keine Fluktuationen)
        for (size_t i = 0; i < cfg->N; i++)
        {
            rt->I[i] = 0.01;
        }

        for (size_t i = 0; i < cfg->N; i++)
        {
            rt->I_prev[i] = rt->I[i];
        }

        for (size_t i = 0; i < cfg->N; i++)
        {
            rt->Q[i] = 0.0;  // Q wird später als Phasenführer implementiert
        }

        int width = (int)sqrt(cfg->N);
        double alpha = 3.0 - cfg->D;

        for (size_t i = 0; i < cfg->N; i++)
        {
            int x_i = i % width;
            int y_i = i / width;
            for (size_t j = 0; j < cfg->N; j++)
            {
                int x_j = j % width;
                int y_j = j / width;
                double dx = x_i - x_j;
                double dy = y_i - y_j;
                double dist = sqrt(dx*dx + dy*dy);
                if (dist < 1.0) dist = 1.0;
                rt->K[i * cfg->N + j] = 1.0 / pow(dist, alpha);
            }
        }

        retval = true;
    }

    return retval;
}

void deinitialize_host_data(const iwt_runtime_t rt)
{
    free(rt->I);
    free(rt->I_prev);
    free(rt->K);
    free(rt->sumJ);
    free(rt->Q);
    rt->I = rt->I_prev = rt->K = rt->sumJ = rt->Q = NULL;
}

bool initialize_gpu_data(const iwt_runtime_t rt, const iwt_config_t cfg)
{
    rt->I_gpu = ocl_create_buffer(&rt->ocl, OCL_BUF_READ_WRITE, cfg->N * sizeof(double), NULL);
    rt->I_prev_gpu = ocl_create_buffer(&rt->ocl, OCL_BUF_READ_WRITE, cfg->N * sizeof(double), NULL);
    rt->K_gpu = ocl_create_buffer(&rt->ocl, OCL_BUF_READ_WRITE, cfg->N * cfg->N * sizeof(double), NULL);
    rt->sumJ_gpu = ocl_create_buffer(&rt->ocl, OCL_BUF_WRITE_ONLY, cfg->N * sizeof(double), NULL);
    rt->Q_gpu = ocl_create_buffer(&rt->ocl, OCL_BUF_READ_WRITE, cfg->N * sizeof(double), NULL);
    return (rt->I_gpu != NULL) && (rt->I_prev_gpu != NULL) && (rt->K_gpu != NULL) && (rt->sumJ_gpu != NULL) && (rt->Q_gpu != NULL);
}

void deinitialize_gpu_data(const iwt_runtime_t rt)
{
    clReleaseMemObject(rt->I_gpu);
    clReleaseMemObject(rt->I_prev_gpu);
    clReleaseMemObject(rt->K_gpu);
    clReleaseMemObject(rt->sumJ_gpu);
    clReleaseMemObject(rt->Q_gpu);
    rt->I_gpu = rt->I_prev_gpu = rt->K_gpu = rt->sumJ_gpu = rt->Q_gpu = NULL;
}