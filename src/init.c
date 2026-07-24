#include "init.h"
#include "iwt.h"
#include <stdio.h>
#include <math.h>

bool initialize_host_data(const iwt_runtime_t rt, const iwt_config_t cfg)
{
    bool retval = false;

    // === IWT-Kernfelder (komplex) ===
    rt->I_real = malloc(cfg->N * sizeof(double));
    rt->I_imag = malloc(cfg->N * sizeof(double));
    rt->I_prev_real = malloc(cfg->N * sizeof(double));
    rt->I_prev_imag = malloc(cfg->N * sizeof(double));
    rt->I_phase = malloc(cfg->N * sizeof(double));
    rt->I_phase_prev = malloc(cfg->N * sizeof(double));

    // === Kopplungsmatrix und Hilfsfelder ===
    rt->K = malloc(cfg->N * cfg->N * sizeof(double));
    rt->sumJ = malloc(cfg->N * sizeof(double));
    rt->Q = malloc(cfg->N * sizeof(double));

    // === Masse und Ladung ===
    rt->mass = malloc(cfg->N * sizeof(double));
    rt->charge = malloc(cfg->N * sizeof(double));

    // === Quantenfluktuationen (Anhang O & P) ===
    rt->xi_real = malloc(cfg->N * sizeof(double));
    rt->xi_imag = malloc(cfg->N * sizeof(double));
    rt->uncertainty = malloc(cfg->N * sizeof(double));

    // === Prüfung aller Allokationen ===
    if ((rt->I_real != NULL) && (rt->I_imag != NULL) &&
        (rt->I_prev_real != NULL) && (rt->I_prev_imag != NULL) &&
        (rt->I_phase != NULL) && (rt->I_phase_prev != NULL) &&
        (rt->K != NULL) && (rt->sumJ != NULL) &&
        (rt->Q != NULL) &&
        (rt->mass != NULL) && (rt->charge != NULL) &&
        (rt->xi_real != NULL) && (rt->xi_imag != NULL) && (rt->uncertainty != NULL))
    {
        // === Initialisierung auf Vakuum (I_real = 0.01, I_imag = 0.0) ===
        for (size_t i = 0; i < cfg->N; i++)
        {
            rt->I_real[i] = 0.01;
            rt->I_imag[i] = 0.0;
            rt->I_prev_real[i] = 0.01;
            rt->I_prev_imag[i] = 0.0;
            rt->I_phase[i] = 0.0;
            rt->I_phase_prev[i] = 0.0;
            rt->Q[i] = 0.0;
            rt->mass[i] = 0.0;
            rt->charge[i] = 0.0;
            rt->xi_real[i] = 0.0;
            rt->xi_imag[i] = 0.0;
            rt->uncertainty[i] = 0.0;
        }

        // === Kopplungsmatrix (fraktal) ===
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

        // === Symmetrische Normierung der Kopplungsmatrix ===
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
    // === IWT-Kernfelder (komplex) ===
    rt->I_real_gpu = ocl_create_buffer(&rt->ocl, OCL_BUF_READ_WRITE, cfg->N * sizeof(double), NULL);
    rt->I_imag_gpu = ocl_create_buffer(&rt->ocl, OCL_BUF_READ_WRITE, cfg->N * sizeof(double), NULL);
    rt->I_prev_real_gpu = ocl_create_buffer(&rt->ocl, OCL_BUF_READ_WRITE, cfg->N * sizeof(double), NULL);
    rt->I_prev_imag_gpu = ocl_create_buffer(&rt->ocl, OCL_BUF_READ_WRITE, cfg->N * sizeof(double), NULL);
    rt->I_phase_gpu = ocl_create_buffer(&rt->ocl, OCL_BUF_READ_WRITE, cfg->N * sizeof(double), NULL);
    rt->I_phase_prev_gpu = ocl_create_buffer(&rt->ocl, OCL_BUF_READ_WRITE, cfg->N * sizeof(double), NULL);

    // === Kopplungsmatrix und Hilfsfelder ===
    rt->K_gpu = ocl_create_buffer(&rt->ocl, OCL_BUF_READ_WRITE, cfg->N * cfg->N * sizeof(double), NULL);
    rt->sumJ_gpu = ocl_create_buffer(&rt->ocl, OCL_BUF_WRITE_ONLY, cfg->N * sizeof(double), NULL);
    rt->Q_gpu = ocl_create_buffer(&rt->ocl, OCL_BUF_READ_WRITE, cfg->N * sizeof(double), NULL);

    // === Masse und Ladung ===
    rt->mass_gpu = ocl_create_buffer(&rt->ocl, OCL_BUF_READ_WRITE, cfg->N * sizeof(double), NULL);
    rt->charge_gpu = ocl_create_buffer(&rt->ocl, OCL_BUF_READ_WRITE, cfg->N * sizeof(double), NULL);

    // === Quantenfluktuationen (Anhang O & P) ===
    rt->xi_real_gpu = ocl_create_buffer(&rt->ocl, OCL_BUF_READ_WRITE, cfg->N * sizeof(double), NULL);
    rt->xi_imag_gpu = ocl_create_buffer(&rt->ocl, OCL_BUF_READ_WRITE, cfg->N * sizeof(double), NULL);
    rt->uncertainty_gpu = ocl_create_buffer(&rt->ocl, OCL_BUF_READ_WRITE, cfg->N * sizeof(double), NULL);

    // === Prüfung aller GPU-Buffer ===
    bool all_buffers_valid =
        (rt->I_real_gpu != NULL) && (rt->I_imag_gpu != NULL) &&
        (rt->I_prev_real_gpu != NULL) && (rt->I_prev_imag_gpu != NULL) &&
        (rt->I_phase_gpu != NULL) && (rt->I_phase_prev_gpu != NULL) &&
        (rt->K_gpu != NULL) && (rt->sumJ_gpu != NULL) &&
        (rt->Q_gpu != NULL) &&
        (rt->mass_gpu != NULL) && (rt->charge_gpu != NULL) &&
        (rt->xi_real_gpu != NULL) && (rt->xi_imag_gpu != NULL) && (rt->uncertainty_gpu != NULL);

    return all_buffers_valid;
}

void deinitialize_host_data(const iwt_runtime_t rt)
{
    // === IWT-Kernfelder (komplex) ===
    free(rt->I_real);
    free(rt->I_imag);
    free(rt->I_prev_real);
    free(rt->I_prev_imag);
    free(rt->I_phase);
    free(rt->I_phase_prev);

    // === Kopplungsmatrix und Hilfsfelder ===
    free(rt->K);
    free(rt->sumJ);
    free(rt->Q);

    // === Masse und Ladung ===
    free(rt->mass);
    free(rt->charge);

    // === Quantenfluktuationen (Anhang O & P) ===
    free(rt->xi_real);
    free(rt->xi_imag);
    free(rt->uncertainty);

    // === Alle Zeiger auf NULL setzen ===
    rt->I_real = NULL;
    rt->I_imag = NULL;
    rt->I_prev_real = NULL;
    rt->I_prev_imag = NULL;
    rt->I_phase = NULL;
    rt->I_phase_prev = NULL;
    rt->K = NULL;
    rt->sumJ = NULL;
    rt->Q = NULL;
    rt->mass = NULL;
    rt->charge = NULL;
    rt->xi_real = NULL;
    rt->xi_imag = NULL;
    rt->uncertainty = NULL;
}

void deinitialize_gpu_data(const iwt_runtime_t rt)
{
    // === IWT-Kernfelder (komplex) ===
    clReleaseMemObject(rt->I_real_gpu);
    clReleaseMemObject(rt->I_imag_gpu);
    clReleaseMemObject(rt->I_prev_real_gpu);
    clReleaseMemObject(rt->I_prev_imag_gpu);
    clReleaseMemObject(rt->I_phase_gpu);
    clReleaseMemObject(rt->I_phase_prev_gpu);

    // === Kopplungsmatrix und Hilfsfelder ===
    clReleaseMemObject(rt->K_gpu);
    clReleaseMemObject(rt->sumJ_gpu);
    clReleaseMemObject(rt->Q_gpu);

    // === Masse und Ladung ===
    clReleaseMemObject(rt->mass_gpu);
    clReleaseMemObject(rt->charge_gpu);

    // === Quantenfluktuationen (Anhang O & P) ===
    clReleaseMemObject(rt->xi_real_gpu);
    clReleaseMemObject(rt->xi_imag_gpu);
    clReleaseMemObject(rt->uncertainty_gpu);

    // === Alle GPU-Zeiger auf NULL setzen ===
    rt->I_real_gpu = NULL;
    rt->I_imag_gpu = NULL;
    rt->I_prev_real_gpu = NULL;
    rt->I_prev_imag_gpu = NULL;
    rt->I_phase_gpu = NULL;
    rt->I_phase_prev_gpu = NULL;
    rt->K_gpu = NULL;
    rt->sumJ_gpu = NULL;
    rt->Q_gpu = NULL;
    rt->mass_gpu = NULL;
    rt->charge_gpu = NULL;
    rt->xi_real_gpu = NULL;
    rt->xi_imag_gpu = NULL;
    rt->uncertainty_gpu = NULL;
}

