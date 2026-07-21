#include "init.h"
#include "iwt.h"
#include <stdio.h>
#include <math.h>

bool initialize_host_data(const iwt_runtime_t rt, const iwt_config_t cfg)
{
    bool retval = false;

    rt->I = malloc(cfg->N * sizeof(double));
    rt->I_prev = malloc(cfg->N * sizeof(double));
    rt->I_phase = malloc(cfg->N * sizeof(double));
    rt->I_phase_prev = malloc(cfg->N * sizeof(double));
    rt->K = malloc(cfg->N * cfg->N * sizeof(double));
    rt->sumJ = malloc(cfg->N * sizeof(double));
    rt->Q = malloc(cfg->N * sizeof(double));
    rt->R = malloc(cfg->N * sizeof(double));   // NEU
    rt->T = malloc(cfg->N * sizeof(double));   // NEU

    if ((rt->I != NULL) && (rt->I_prev != NULL) && (rt->I_phase != NULL) &&
        (rt->I_phase_prev != NULL) && (rt->K != NULL) && (rt->sumJ != NULL) &&
        (rt->Q != NULL) && (rt->R != NULL) && (rt->T != NULL))
    {
        for (size_t i = 0; i < cfg->N; i++)
        {
            rt->I[i] = iwt_I_min();
            rt->I_prev[i] = iwt_I_min();
            rt->I_phase[i] = 0.0;
            rt->I_phase_prev[i] = 0.0;
            rt->Q[i] = 0.0;
            rt->R[i] = 0.0;   // NEU
            rt->T[i] = 0.0;   // NEU
        }

        double D = cfg->D;
        double alpha = 3.0 - D;

        for (size_t i = 0; i < cfg->N; i++)
        {
            for (size_t j = 0; j < cfg->N; j++)
            {
                double idx_dist = (double)(i > j ? i - j : j - i);
                if (idx_dist < 1.0) idx_dist = 1.0;
                double d_ij = pow(idx_dist, 1.0 / D);
                rt->K[i * cfg->N + j] = 1.0 / pow(d_ij, alpha);
            }
        }

        // Symmetrische Normierung der Kopplungsmatrix
        double* row_sum = malloc(cfg->N * sizeof(double));
        if (row_sum != NULL)
        {
            for (size_t i = 0; i < cfg->N; i++)
            {
                row_sum[i] = 0.0;
                for (size_t j = 0; j < cfg->N; j++)
                {
                    row_sum[i] += rt->K[i * cfg->N + j];
                }
            }

            for (size_t i = 0; i < cfg->N; i++)
            {
                for (size_t j = 0; j < cfg->N; j++)
                {
                    double norm = sqrt(row_sum[i] * row_sum[j]);
                    if (norm > 1e-30)
                    {
                        rt->K[i * cfg->N + j] /= norm;
                    }
                    else
                    {
                        rt->K[i * cfg->N + j] = 0.0;
                    }
                }
            }

            free(row_sum);
        }

        retval = true;
    }

    return retval;
}

bool initialize_gpu_data(const iwt_runtime_t rt, const iwt_config_t cfg)
{
    rt->I_gpu = ocl_create_buffer(&rt->ocl, OCL_BUF_READ_WRITE, cfg->N * sizeof(double), NULL);
    rt->I_prev_gpu = ocl_create_buffer(&rt->ocl, OCL_BUF_READ_WRITE, cfg->N * sizeof(double), NULL);
    rt->I_phase_gpu = ocl_create_buffer(&rt->ocl, OCL_BUF_READ_WRITE, cfg->N * sizeof(double), NULL);
    rt->I_phase_prev_gpu = ocl_create_buffer(&rt->ocl, OCL_BUF_READ_WRITE, cfg->N * sizeof(double), NULL);
    rt->K_gpu = ocl_create_buffer(&rt->ocl, OCL_BUF_READ_WRITE, cfg->N * cfg->N * sizeof(double), NULL);
    rt->sumJ_gpu = ocl_create_buffer(&rt->ocl, OCL_BUF_WRITE_ONLY, cfg->N * sizeof(double), NULL);
    rt->Q_gpu = ocl_create_buffer(&rt->ocl, OCL_BUF_READ_WRITE, cfg->N * sizeof(double), NULL);
    rt->R_gpu = ocl_create_buffer(&rt->ocl, OCL_BUF_READ_WRITE, cfg->N * sizeof(double), NULL);   // NEU
    rt->T_gpu = ocl_create_buffer(&rt->ocl, OCL_BUF_READ_WRITE, cfg->N * sizeof(double), NULL);   // NEU
    
    return (rt->I_gpu != NULL) && (rt->I_prev_gpu != NULL) && (rt->I_phase_gpu != NULL) &&
           (rt->I_phase_prev_gpu != NULL) && (rt->K_gpu != NULL) && (rt->sumJ_gpu != NULL) &&
           (rt->Q_gpu != NULL) && (rt->R_gpu != NULL) && (rt->T_gpu != NULL);
}

void deinitialize_host_data(const iwt_runtime_t rt)
{
    free(rt->I);
    free(rt->I_prev);
    free(rt->I_phase);
    free(rt->I_phase_prev);
    free(rt->K);
    free(rt->sumJ);
    free(rt->Q);
    free(rt->R);   // NEU
    free(rt->T);   // NEU
    rt->I = rt->I_prev = rt->I_phase = rt->I_phase_prev = rt->K = rt->sumJ = rt->Q = rt->R = rt->T = NULL;
}

void deinitialize_gpu_data(const iwt_runtime_t rt)
{
    clReleaseMemObject(rt->I_gpu);
    clReleaseMemObject(rt->I_prev_gpu);
    clReleaseMemObject(rt->I_phase_gpu);
    clReleaseMemObject(rt->I_phase_prev_gpu);
    clReleaseMemObject(rt->K_gpu);
    clReleaseMemObject(rt->sumJ_gpu);
    clReleaseMemObject(rt->Q_gpu);
    clReleaseMemObject(rt->R_gpu);   // NEU
    clReleaseMemObject(rt->T_gpu);   // NEU
    rt->I_gpu = rt->I_prev_gpu = rt->I_phase_gpu = rt->I_phase_prev_gpu = rt->K_gpu = rt->sumJ_gpu = rt->Q_gpu = rt->R_gpu = rt->T_gpu = NULL;
}