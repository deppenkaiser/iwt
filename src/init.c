#include "init.h"
#include "iwt.h"
#include <stdio.h>
#include <math.h>

bool initialize_host_data(const iwt_runtime_t rt, const iwt_config_t cfg)
{
    bool retval = false;

    // === BESTEHENDE ALLOKATIONEN ===
    rt->I = malloc(cfg->N * sizeof(double));
    rt->I_prev = malloc(cfg->N * sizeof(double));
    rt->I_phase = malloc(cfg->N * sizeof(double));
    rt->I_phase_prev = malloc(cfg->N * sizeof(double));
    rt->K = malloc(cfg->N * cfg->N * sizeof(double));
    rt->sumJ = malloc(cfg->N * sizeof(double));
    rt->Q = malloc(cfg->N * sizeof(double));
    rt->R = malloc(cfg->N * sizeof(double));
    rt->T = malloc(cfg->N * sizeof(double));

    // === NEU: ALLOKATIONEN FÜR QUANTENFLUKTUATIONEN ===
    rt->xi_real = malloc(cfg->N * sizeof(double));
    rt->xi_imag = malloc(cfg->N * sizeof(double));
    rt->uncertainty = malloc(cfg->N * sizeof(double));

    if ((rt->I != NULL) && (rt->I_prev != NULL) && (rt->I_phase != NULL) &&
        (rt->I_phase_prev != NULL) && (rt->K != NULL) && (rt->sumJ != NULL) &&
        (rt->Q != NULL) && (rt->R != NULL) && (rt->T != NULL) &&
        (rt->xi_real != NULL) && (rt->xi_imag != NULL) && (rt->uncertainty != NULL))
    {
        // === BESTEHENDE INITIALISIERUNG ===
        for (size_t i = 0; i < cfg->N; i++)
        {
            rt->I[i] = iwt_I_min();
            rt->I_prev[i] = iwt_I_min();
            rt->I_phase[i] = 0.0;
            rt->I_phase_prev[i] = 0.0;
            rt->Q[i] = 0.0;
            rt->R[i] = 0.0;
            rt->T[i] = 0.0;
        }

        // === NEU: INITIALISIERUNG DER FLUKTUATIONSFELDER ===
        // Setze zunächst alle auf 0 (werden später im Simulationslauf mit Zufallswerten gefüllt)
        for (size_t i = 0; i < cfg->N; i++)
        {
            rt->xi_real[i] = 0.0;
            rt->xi_imag[i] = 0.0;
            rt->uncertainty[i] = 0.0;
        }

        // === BESTEHENDE KOPPLUNGSMATRIX ===
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
    // === BESTEHENDE GPU-BUFFER ===
    rt->I_gpu = ocl_create_buffer(&rt->ocl, OCL_BUF_READ_WRITE, cfg->N * sizeof(double), NULL);
    rt->I_prev_gpu = ocl_create_buffer(&rt->ocl, OCL_BUF_READ_WRITE, cfg->N * sizeof(double), NULL);
    rt->I_phase_gpu = ocl_create_buffer(&rt->ocl, OCL_BUF_READ_WRITE, cfg->N * sizeof(double), NULL);
    rt->I_phase_prev_gpu = ocl_create_buffer(&rt->ocl, OCL_BUF_READ_WRITE, cfg->N * sizeof(double), NULL);
    rt->K_gpu = ocl_create_buffer(&rt->ocl, OCL_BUF_READ_WRITE, cfg->N * cfg->N * sizeof(double), NULL);
    rt->sumJ_gpu = ocl_create_buffer(&rt->ocl, OCL_BUF_WRITE_ONLY, cfg->N * sizeof(double), NULL);
    rt->Q_gpu = ocl_create_buffer(&rt->ocl, OCL_BUF_READ_WRITE, cfg->N * sizeof(double), NULL);
    rt->R_gpu = ocl_create_buffer(&rt->ocl, OCL_BUF_READ_WRITE, cfg->N * sizeof(double), NULL);
    rt->T_gpu = ocl_create_buffer(&rt->ocl, OCL_BUF_READ_WRITE, cfg->N * sizeof(double), NULL);

    // === NEU: GPU-BUFFER FÜR QUANTENFLUKTUATIONEN ===
    rt->xi_real_gpu = ocl_create_buffer(&rt->ocl, OCL_BUF_READ_WRITE, cfg->N * sizeof(double), NULL);
    rt->xi_imag_gpu = ocl_create_buffer(&rt->ocl, OCL_BUF_READ_WRITE, cfg->N * sizeof(double), NULL);
    rt->uncertainty_gpu = ocl_create_buffer(&rt->ocl, OCL_BUF_READ_WRITE, cfg->N * sizeof(double), NULL);

    // Prüfung aller GPU-Buffer
    bool all_buffers_valid =
        (rt->I_gpu != NULL) && (rt->I_prev_gpu != NULL) &&
        (rt->I_phase_gpu != NULL) && (rt->I_phase_prev_gpu != NULL) &&
        (rt->K_gpu != NULL) && (rt->sumJ_gpu != NULL) &&
        (rt->Q_gpu != NULL) && (rt->R_gpu != NULL) && (rt->T_gpu != NULL) &&
        (rt->xi_real_gpu != NULL) && (rt->xi_imag_gpu != NULL) && (rt->uncertainty_gpu != NULL);

    return all_buffers_valid;
}

void deinitialize_host_data(const iwt_runtime_t rt)
{
    // === BESTEHENDE FREIGABEN ===
    free(rt->I);
    free(rt->I_prev);
    free(rt->I_phase);
    free(rt->I_phase_prev);
    free(rt->K);
    free(rt->sumJ);
    free(rt->Q);
    free(rt->R);
    free(rt->T);

    // === NEU: FREIGABE DER FLUKTUATIONSFELDER ===
    free(rt->xi_real);
    free(rt->xi_imag);
    free(rt->uncertainty);

    // Alle Zeiger auf NULL setzen
    rt->I = NULL;
    rt->I_prev = NULL;
    rt->I_phase = NULL;
    rt->I_phase_prev = NULL;
    rt->K = NULL;
    rt->sumJ = NULL;
    rt->Q = NULL;
    rt->R = NULL;
    rt->T = NULL;
    rt->xi_real = NULL;
    rt->xi_imag = NULL;
    rt->uncertainty = NULL;
}

void deinitialize_gpu_data(const iwt_runtime_t rt)
{
    // === BESTEHENDE GPU-FREIGABEN ===
    clReleaseMemObject(rt->I_gpu);
    clReleaseMemObject(rt->I_prev_gpu);
    clReleaseMemObject(rt->I_phase_gpu);
    clReleaseMemObject(rt->I_phase_prev_gpu);
    clReleaseMemObject(rt->K_gpu);
    clReleaseMemObject(rt->sumJ_gpu);
    clReleaseMemObject(rt->Q_gpu);
    clReleaseMemObject(rt->R_gpu);
    clReleaseMemObject(rt->T_gpu);

    // === NEU: GPU-FREIGABE DER FLUKTUATIONSFELDER ===
    clReleaseMemObject(rt->xi_real_gpu);
    clReleaseMemObject(rt->xi_imag_gpu);
    clReleaseMemObject(rt->uncertainty_gpu);

    // Alle GPU-Zeiger auf NULL setzen
    rt->I_gpu = NULL;
    rt->I_prev_gpu = NULL;
    rt->I_phase_gpu = NULL;
    rt->I_phase_prev_gpu = NULL;
    rt->K_gpu = NULL;
    rt->sumJ_gpu = NULL;
    rt->Q_gpu = NULL;
    rt->R_gpu = NULL;
    rt->T_gpu = NULL;
    rt->xi_real_gpu = NULL;
    rt->xi_imag_gpu = NULL;
    rt->uncertainty_gpu = NULL;
}