// ============================================================================
// iwt_kernel.c - IWT Kernel-Funktionen
// Refactored: Reduzierte McCabe-Komplexität, gemeinsame Helper, Pipeline
// ============================================================================

#include "iwt_kernel.h"
#include "iwt.h"
#include "iwt_kernel_frozen.h"
#include "iwt_detect_cluster.h"
#include "iwt_move_cluster.h"
#include <api/api.h>
#include <math.h>
#include <ocl/ocl.h>
#include <stdio.h>
#include <string/string.h>

#define ALPHA 1.0
#define BETA 1.0
#define DELTA 1.0
#define GAMMA 1.0

/* Helper: Workgroup-Größe begrenzen */
static size_t get_local_size(size_t global)
{
    size_t local = 64;
    if (local > global) {
        local = global;
    }
    return local;
}

/* Helper: Kernel-Validierung */
static bool kernel_valid(cl_kernel k)
{
    return k != NULL;
}

/* Helper: Q-Array sanitizen */
static void sanitize_q_array(double *Q, size_t N, double Q_min)
{
    for (size_t i = 0; i < N; i++) {
        if (isnan(Q[i]) || isinf(Q[i])) {
            Q[i] = Q_min;
        }
    }
}

/* Helper: Summe der Betragsquadrate */
static double sum_abs_sq(const iwt_runtime_t rt, const iwt_config_t cfg)
{
    double sum = 0.0;
    for (size_t i = 0; i < cfg->N; i++) {
        sum += rt->I_real[i] * rt->I_real[i] + rt->I_imag[i] * rt->I_imag[i];
    }
    return sum;
}

bool run_q_calculation(const iwt_runtime_t rt, const iwt_config_t cfg)
{
    cl_kernel kernel = ocl_get_kernel(&rt->ocl, OCL_KERNEL_IWT_Q);
    if (!kernel_valid(kernel)) {
        return false;
    }

    double sum_abs_sq_val = sum_abs_sq(rt, cfg);

    if (sum_abs_sq_val < 1e-30) {
        for (size_t i = 0; i < cfg->N; i++) {
            rt->Q[i] = 1e-6;
        }
        return true;
    }

    int N = (int)cfg->N;
    double hbar = cfg->hbar;
    double m = 1.0;
    double beta = cfg->beta;
    double epsilon = 1e-6;
    double Q_min = 1e-6;

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &rt->I_real_gpu);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &rt->I_imag_gpu);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &rt->Q_gpu);
    clSetKernelArg(kernel, 3, sizeof(int), &N);
    clSetKernelArg(kernel, 4, sizeof(double), &sum_abs_sq_val);
    clSetKernelArg(kernel, 5, sizeof(double), &hbar);
    clSetKernelArg(kernel, 6, sizeof(double), &m);
    clSetKernelArg(kernel, 7, sizeof(double), &beta);
    clSetKernelArg(kernel, 8, sizeof(double), &epsilon);
    clSetKernelArg(kernel, 9, sizeof(double), &Q_min);

    size_t global = cfg->N;
    size_t local = get_local_size(global);

    if (!ocl_enqueue_kernel(&rt->ocl, kernel, global, local)) {
        return false;
    }

    clEnqueueReadBuffer(rt->ocl.queue, rt->Q_gpu, CL_TRUE, 0, cfg->N * sizeof(double), rt->Q, 0, NULL, NULL);
    sanitize_q_array(rt->Q, cfg->N, Q_min);
    return true;
}

bool run_flux_calculation(const iwt_runtime_t rt, const iwt_config_t cfg)
{
    cl_kernel kernel = ocl_get_kernel(&rt->ocl, OCL_KERNEL_IWT_FLUX);
    if (!kernel_valid(kernel)) {
        return false;
    }

    clEnqueueWriteBuffer(rt->ocl.queue, rt->I_real_gpu, CL_TRUE, 0, cfg->N * sizeof(double), rt->I_real, 0, NULL, NULL);
    clEnqueueWriteBuffer(rt->ocl.queue, rt->I_imag_gpu, CL_TRUE, 0, cfg->N * sizeof(double), rt->I_imag, 0, NULL, NULL);
    clEnqueueWriteBuffer(rt->ocl.queue, rt->Q_gpu, CL_TRUE, 0, cfg->N * sizeof(double), rt->Q, 0, NULL, NULL);
    clEnqueueWriteBuffer(rt->ocl.queue, rt->K_gpu, CL_TRUE, 0, cfg->N * cfg->N * sizeof(double), rt->K, 0, NULL, NULL);

    int N = (int)cfg->N;
    double DT = cfg->DT;
    double gamma = cfg->gamma;

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &rt->I_real_gpu);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &rt->I_imag_gpu);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &rt->Q_gpu);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), &rt->K_gpu);
    clSetKernelArg(kernel, 4, sizeof(cl_mem), &rt->sumJ_gpu);
    clSetKernelArg(kernel, 5, sizeof(int), &N);
    clSetKernelArg(kernel, 6, sizeof(double), &DT);
    clSetKernelArg(kernel, 7, sizeof(double), &gamma);

    size_t global = cfg->N;
    size_t local = get_local_size(global);

    if (!ocl_enqueue_kernel(&rt->ocl, kernel, global, local)) {
        return false;
    }

    clEnqueueReadBuffer(rt->ocl.queue, rt->sumJ_gpu, CL_TRUE, 0, cfg->N * sizeof(double), rt->sumJ, 0, NULL, NULL);
    return true;
}

bool run_update_info(const iwt_runtime_t rt, const iwt_config_t cfg)
{
    cl_kernel kernel = ocl_get_kernel(&rt->ocl, OCL_KERNEL_IWT_UPDATE_INFO);
    if (!kernel_valid(kernel)) {
        return false;
    }

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
    size_t local = get_local_size(global);

    if (!ocl_enqueue_kernel(&rt->ocl, kernel, global, local)) {
        return false;
    }

    clEnqueueReadBuffer(rt->ocl.queue, rt->I_real_gpu, CL_TRUE, 0, cfg->N * sizeof(double), rt->I_real, 0, NULL, NULL);
    clEnqueueReadBuffer(rt->ocl.queue, rt->I_imag_gpu, CL_TRUE, 0, cfg->N * sizeof(double), rt->I_imag, 0, NULL, NULL);
    clEnqueueReadBuffer(rt->ocl.queue, rt->I_phase_gpu, CL_TRUE, 0, cfg->N * sizeof(double), rt->I_phase, 0, NULL, NULL);
    return true;
}

bool run_compute_mass_charge(const iwt_runtime_t rt, const iwt_config_t cfg)
{
    cl_kernel kernel = ocl_get_kernel(&rt->ocl, OCL_KERNEL_IWT_MASS_CHARGE);
    if (!kernel_valid(kernel)) {
        return false;
    }

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
    size_t local = get_local_size(global);

    if (!ocl_enqueue_kernel(&rt->ocl, kernel, global, local)) {
        return false;
    }

    clEnqueueReadBuffer(rt->ocl.queue, rt->mass_gpu, CL_TRUE, 0, cfg->N * sizeof(double), rt->mass, 0, NULL, NULL);
    clEnqueueReadBuffer(rt->ocl.queue, rt->charge_gpu, CL_TRUE, 0, cfg->N * sizeof(double), rt->charge, 0, NULL, NULL);
    return true;
}

/* Pipeline für Simulationsschritt */
bool run_simulation_step(const iwt_runtime_t rt, const iwt_config_t cfg)
{
    for (size_t i = 0; i < cfg->N; i++) {
        rt->I_prev_real[i] = rt->I_real[i];
        rt->I_prev_imag[i] = rt->I_imag[i];
        rt->I_phase_prev[i] = rt->I_phase[i];
    }

    typedef bool (*step_fn)(const iwt_runtime_t, const iwt_config_t);
    step_fn steps[] = {
        frozen_generate_uncertainty_cpu,
        frozen_run_apply_fluctuations,
        frozen_run_apply_redshift_damping,
        run_flux_calculation,
        run_q_calculation,
        run_update_info,
        run_compute_mass_charge
    };

    for (size_t s = 0; s < sizeof(steps)/sizeof(steps[0]); s++) {
        if (!steps[s](rt, cfg)) {
            return false;
        }
    }

    iwt_detect_clusters(rt, cfg);
    iwt_move_clusters(rt, cfg, cfg->DT);
    return true;
}
