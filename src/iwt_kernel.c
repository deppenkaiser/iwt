// ============================================================================
// iwt_kernel.c - IWT Kernel-Funktionen (Entwicklungsversion)
// ============================================================================

#include "iwt_kernel.h"
#include "iwt.h"
#include "iwt_kernel_frozen.h"
#include <api/api.h>
#include <math.h>
#include <ocl/ocl.h>
#include <stdio.h>
#include <string/string.h>

#define ALPHA 1.0
#define BETA 1.0
#define DELTA 1.0
#define GAMMA 1.0

static double compute_mass(const iwt_runtime_t rt, size_t k, size_t N)
{
    double mass = 0.0;
    if (k > 0)
    {
        double dr = rt->I_real[k] - rt->I_real[k - 1];
        double di = rt->I_imag[k] - rt->I_imag[k - 1];
        mass += dr * dr + di * di;
    }
    if (k < N - 1)
    {
        double dr = rt->I_real[k] - rt->I_real[k + 1];
        double di = rt->I_imag[k] - rt->I_imag[k + 1];
        mass += dr * dr + di * di;
    }
    return DELTA * mass;
}

bool run_q_calculation(const iwt_runtime_t rt, const iwt_config_t cfg)
{
    cl_kernel kernel = ocl_get_kernel(&rt->ocl, OCL_KERNEL_IWT_Q);
    if (!kernel) return false;

    double sum_abs_sq = 0.0;
    for (size_t i = 0; i < cfg->N; i++)
    {
        sum_abs_sq += rt->I_real[i] * rt->I_real[i] + rt->I_imag[i] * rt->I_imag[i];
    }

    if (sum_abs_sq < 1e-30)
    {
        for (size_t i = 0; i < cfg->N; i++)
        {
            rt->Q[i] = 1e-6;
        }
        return true;
    }

    int N = (int)cfg->N;
    double hbar = 1.0;
    double m = 1.0;
    double prefactor = -(hbar * hbar) / (2.0 * m);
    double epsilon = 1e-6;
    double Q_min = 1e-6;

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &rt->I_real_gpu);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &rt->I_imag_gpu);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &rt->Q_gpu);
    clSetKernelArg(kernel, 3, sizeof(int), &N);
    clSetKernelArg(kernel, 4, sizeof(double), &sum_abs_sq);
    clSetKernelArg(kernel, 5, sizeof(double), &prefactor);
    clSetKernelArg(kernel, 6, sizeof(double), &epsilon);
    clSetKernelArg(kernel, 7, sizeof(double), &Q_min);

    size_t global = cfg->N;
    size_t local = 64;
    if (local > global) local = global;

    if (!ocl_enqueue_kernel(&rt->ocl, kernel, global, local)) return false;

    clEnqueueReadBuffer(rt->ocl.queue, rt->Q_gpu, CL_TRUE, 0, cfg->N * sizeof(double), rt->Q, 0, NULL, NULL);

    for (size_t i = 0; i < cfg->N; i++)
    {
        if (isnan(rt->Q[i]) || isinf(rt->Q[i]))
        {
            rt->Q[i] = Q_min;
        }
    }

    return true;
}

bool run_flux_calculation(const iwt_runtime_t rt, const iwt_config_t cfg)
{
    cl_kernel kernel = ocl_get_kernel(&rt->ocl, OCL_KERNEL_IWT_FLUX);
    if (!kernel) return false;

    // Daten auf die GPU kopieren
    clEnqueueWriteBuffer(rt->ocl.queue, rt->I_real_gpu, CL_TRUE, 0,
                         cfg->N * sizeof(double), rt->I_real, 0, NULL, NULL);
    clEnqueueWriteBuffer(rt->ocl.queue, rt->I_imag_gpu, CL_TRUE, 0,
                         cfg->N * sizeof(double), rt->I_imag, 0, NULL, NULL);
    clEnqueueWriteBuffer(rt->ocl.queue, rt->Q_gpu, CL_TRUE, 0,
                         cfg->N * sizeof(double), rt->Q, 0, NULL, NULL);
    clEnqueueWriteBuffer(rt->ocl.queue, rt->K_gpu, CL_TRUE, 0,
                         cfg->N * cfg->N * sizeof(double), rt->K, 0, NULL, NULL);

    int N = (int)cfg->N;
    double DT = cfg->DT;

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &rt->I_real_gpu);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &rt->I_imag_gpu);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &rt->Q_gpu);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), &rt->K_gpu);
    clSetKernelArg(kernel, 4, sizeof(cl_mem), &rt->sumJ_gpu);
    clSetKernelArg(kernel, 5, sizeof(int), &N);
    clSetKernelArg(kernel, 6, sizeof(double), &DT);

    size_t global = cfg->N;
    size_t local = 64;
    if (local > global) local = global;

    if (!ocl_enqueue_kernel(&rt->ocl, kernel, global, local)) return false;

    clEnqueueReadBuffer(rt->ocl.queue, rt->sumJ_gpu, CL_TRUE,
                        0, cfg->N * sizeof(double), rt->sumJ, 0, NULL, NULL);
    return true;
}

bool run_update_info(const iwt_runtime_t rt, const iwt_config_t cfg)
{
    cl_kernel kernel = ocl_get_kernel(&rt->ocl, OCL_KERNEL_IWT_UPDATE_INFO);
    if (!kernel) return false;

    int N = (int)cfg->N;
    double DT = cfg->DT;

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &rt->I_real_gpu);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &rt->I_imag_gpu);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &rt->I_phase_gpu);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), &rt->sumJ_gpu);
    clSetKernelArg(kernel, 4, sizeof(cl_mem), &rt->Q_gpu);
    clSetKernelArg(kernel, 5, sizeof(int), &N);
    clSetKernelArg(kernel, 6, sizeof(double), &DT);

    size_t global = cfg->N;
    size_t local = 64;
    if (local > global) local = global;

    if (!ocl_enqueue_kernel(&rt->ocl, kernel, global, local)) return false;

    clEnqueueReadBuffer(rt->ocl.queue, rt->I_real_gpu, CL_TRUE, 0, cfg->N * sizeof(double), rt->I_real, 0, NULL, NULL);
    clEnqueueReadBuffer(rt->ocl.queue, rt->I_imag_gpu, CL_TRUE, 0, cfg->N * sizeof(double), rt->I_imag, 0, NULL, NULL);
    clEnqueueReadBuffer(rt->ocl.queue, rt->I_phase_gpu, CL_TRUE, 0, cfg->N * sizeof(double), rt->I_phase, 0, NULL, NULL);

    return true;
}

bool run_compute_mass_charge(const iwt_runtime_t rt, const iwt_config_t cfg)
{
    cl_kernel kernel = ocl_get_kernel(&rt->ocl, OCL_KERNEL_IWT_MASS_CHARGE);
    if (!kernel) return false;

    clEnqueueWriteBuffer(rt->ocl.queue, rt->I_real_gpu, CL_TRUE, 0, cfg->N * sizeof(double), rt->I_real, 0, NULL, NULL);
    clEnqueueWriteBuffer(rt->ocl.queue, rt->I_imag_gpu, CL_TRUE, 0, cfg->N * sizeof(double), rt->I_imag, 0, NULL, NULL);
    clEnqueueWriteBuffer(rt->ocl.queue, rt->I_phase_gpu, CL_TRUE, 0, cfg->N * sizeof(double), rt->I_phase, 0, NULL, NULL);

    int N = (int)cfg->N;
    double delta = DELTA;

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &rt->I_real_gpu);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &rt->I_imag_gpu);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &rt->I_phase_gpu);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), &rt->mass_gpu);
    clSetKernelArg(kernel, 4, sizeof(cl_mem), &rt->charge_gpu);
    clSetKernelArg(kernel, 5, sizeof(int), &N);
    clSetKernelArg(kernel, 6, sizeof(double), &delta);

    size_t global = cfg->N;
    size_t local = 64;
    if (local > global) local = global;

    if (!ocl_enqueue_kernel(&rt->ocl, kernel, global, local)) return false;

    clEnqueueReadBuffer(rt->ocl.queue, rt->mass_gpu, CL_TRUE, 0, cfg->N * sizeof(double), rt->mass, 0, NULL, NULL);
    clEnqueueReadBuffer(rt->ocl.queue, rt->charge_gpu, CL_TRUE, 0, cfg->N * sizeof(double), rt->charge, 0, NULL, NULL);
    return true;
}

bool run_simulation(const iwt_runtime_t rt, const iwt_config_t cfg)
{
    bool retval = false;

    for (size_t i = 0; i < cfg->N; i++)
    {
        rt->I_real[i] = 0.01;
        rt->I_imag[i] = 0.0;
        rt->I_phase[i] = 0.0;
        rt->I_prev_real[i] = 0.01;
        rt->I_prev_imag[i] = 0.0;
        rt->I_phase_prev[i] = 0.0;
    }

    double sum_abs_sq_initial = 0.0;
    for (size_t i = 0; i < cfg->N; i++)
    {
        sum_abs_sq_initial += rt->I_real[i] * rt->I_real[i] + rt->I_imag[i] * rt->I_imag[i];
    }

    FILE *csv_file = fopen("iwt_timeseries.csv", "w");
    if (csv_file)
    {
        fprintf(csv_file, "iter,sum_I_sq,I_max,sum_mass,sum_E\n");
    }

    string_clear_screen();

    for (int iter = 0; iter < cfg->MAX_ITER; iter++)
    {
        for (size_t i = 0; i < cfg->N; i++)
        {
            rt->I_prev_real[i] = rt->I_real[i];
            rt->I_prev_imag[i] = rt->I_imag[i];
            rt->I_phase_prev[i] = rt->I_phase[i];
        }

        if (!frozen_generate_uncertainty_cpu(rt, cfg)) break;
        if (!frozen_run_apply_fluctuations(rt, cfg)) break;
        if (!frozen_run_apply_redshift_damping(rt, cfg)) break;
        if (!run_flux_calculation(rt, cfg)) break;
        if (!run_q_calculation(rt, cfg)) break;
        if (!run_update_info(rt, cfg)) break;
        if (!run_compute_mass_charge(rt, cfg)) break;

        double max_q = 0.0;
        double sum_abs_sq = 0.0;
        double sum_mass = 0.0;
        double sum_E = 0.0;
        double I_min = 1e30;
        double I_max = -1e30;

        for (size_t i = 0; i < cfg->N; i++)
        {
            double abs_i = sqrt(rt->I_real[i] * rt->I_real[i] + rt->I_imag[i] * rt->I_imag[i] + 1e-30);
            double rho_i = abs_i * abs_i;

            sum_abs_sq += rho_i;
            sum_mass += compute_mass(rt, i, cfg->N);
            sum_E += rho_i;

            if (abs_i < I_min) I_min = abs_i;
            if (abs_i > I_max) I_max = abs_i;

            double abs_q = fabs(rt->Q[i]);
            if (abs_q > max_q) max_q = abs_q;
        }

        if (csv_file)
        {
            fprintf(csv_file, "%d,%.6f,%.6f,%.6f,%.6f\n", iter, sum_abs_sq, I_max, sum_mass, sum_E);
        }

        if (iter % 10 == 0)
        {
            char filename[256];
            snprintf(filename, sizeof(filename), "heatmap_mass_%06d.ppm", iter);
            iwt_save_heatmap_ppm(rt->mass, cfg->N, filename, "mass");
            snprintf(filename, sizeof(filename), "heatmap_charge_%06d.ppm", iter);
            iwt_save_charge_heatmap(rt->charge, cfg->N, filename);
        }

        string_set_cursor_position(1, 1);
        iwt_print_status(rt, cfg, iter, max_q, sum_abs_sq, I_min, I_max, sum_abs_sq);

        fflush(stdout);
        retval = true;
    }

    if (csv_file)
    {
        fclose(csv_file);
        printf("\nZeitreihe gespeichert in: iwt_timeseries.csv\n");
    }

    printf("\n");
    return retval;
}