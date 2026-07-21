#include <ocl/ocl.h>
#include <string/string.h>
#include <stdio.h>
#include <math.h>
#include <api/api.h>
#include "iwt.h"

private bool run_flux_calculation(const iwt_runtime_t rt, const iwt_config_t cfg);
private bool run_update_info(const iwt_runtime_t rt, const iwt_config_t cfg);
private double compute_energy(const iwt_runtime_t rt, const iwt_config_t cfg);

private bool run_flux_calculation(const iwt_runtime_t rt, const iwt_config_t cfg)
{
    bool retval = false;
    cl_kernel kernel = ocl_get_kernel(&rt->ocl, OCL_KERNEL_IWT_FLUX);
    
    if (kernel != NULL)
    {
        clEnqueueWriteBuffer(rt->ocl.queue, rt->I_gpu, CL_TRUE, 0,
            cfg->N * sizeof(double), rt->I, 0, NULL, NULL);
        clEnqueueWriteBuffer(rt->ocl.queue, rt->K_gpu, CL_TRUE, 0,
            cfg->N * cfg->N * sizeof(double), rt->K, 0, NULL, NULL);

        int N = (int)cfg->N;
        int start = 0;
        int end = N;
        double DT = cfg->DT;
        double I_max = cfg->I_max;
        double Z_0 = cfg->Z_0;
        double alpha = cfg->alpha_Z;

        clSetKernelArg(kernel, 0, sizeof(cl_mem), &rt->I_gpu);
        clSetKernelArg(kernel, 1, sizeof(cl_mem), &rt->K_gpu);
        clSetKernelArg(kernel, 2, sizeof(cl_mem), &rt->sumJ_gpu);
        clSetKernelArg(kernel, 3, sizeof(cl_mem), &rt->R_gpu);
        clSetKernelArg(kernel, 4, sizeof(cl_mem), &rt->T_gpu);
        clSetKernelArg(kernel, 5, sizeof(int), &N);
        clSetKernelArg(kernel, 6, sizeof(int), &start);
        clSetKernelArg(kernel, 7, sizeof(int), &end);
        clSetKernelArg(kernel, 8, sizeof(double), &DT);
        clSetKernelArg(kernel, 9, sizeof(double), &I_max);
        clSetKernelArg(kernel, 10, sizeof(double), &Z_0);
        clSetKernelArg(kernel, 11, sizeof(double), &alpha);

        size_t global = cfg->N;
        size_t local = 64;
        if (local > global) local = global;

        if (ocl_enqueue_kernel(&rt->ocl, kernel, global, local))
        {
            clEnqueueReadBuffer(rt->ocl.queue, rt->sumJ_gpu, CL_TRUE,
                0, cfg->N * sizeof(double), rt->sumJ, 0, NULL, NULL);
            clEnqueueReadBuffer(rt->ocl.queue, rt->R_gpu, CL_TRUE,
                0, cfg->N * sizeof(double), rt->R, 0, NULL, NULL);
            clEnqueueReadBuffer(rt->ocl.queue, rt->T_gpu, CL_TRUE,
                0, cfg->N * sizeof(double), rt->T, 0, NULL, NULL);
            retval = true;
        }
    }
    
    return retval;
}

private bool run_q_calculation(const iwt_runtime_t rt, const iwt_config_t cfg)
{
    bool retval = false;
    cl_kernel kernel = ocl_get_kernel(&rt->ocl, OCL_KERNEL_IWT_Q);
    
    if (kernel != NULL)
    {
        double sum_abs_sq = 0.0;
        for (size_t i = 0; i < cfg->N; i++)
        {
            sum_abs_sq += rt->I[i] * rt->I[i];
        }

        // Wenn keine Information vorhanden ist, setze Q auf 0 und kehre mit Erfolg zurück
        if (sum_abs_sq < 1e-30)
        {
            for (size_t i = 0; i < cfg->N; i++) rt->Q[i] = 0.0;
            return true;
        }

        int N = (int)cfg->N;
        double hbar = 1.0;
        double m = 1.0;
        double prefactor = -(hbar * hbar) / (2.0 * m);

        clSetKernelArg(kernel, 0, sizeof(cl_mem), &rt->I_gpu);
        clSetKernelArg(kernel, 1, sizeof(cl_mem), &rt->Q_gpu);
        clSetKernelArg(kernel, 2, sizeof(int), &N);
        clSetKernelArg(kernel, 3, sizeof(double), &sum_abs_sq);
        clSetKernelArg(kernel, 4, sizeof(double), &prefactor);

        size_t global = cfg->N;
        size_t local = 64;
        if (local > global) local = global;

        if (ocl_enqueue_kernel(&rt->ocl, kernel, global, local))
        {
            clEnqueueReadBuffer(rt->ocl.queue, rt->Q_gpu, CL_TRUE,
                0, cfg->N * sizeof(double), rt->Q, 0, NULL, NULL);

            for (size_t i = 0; i < cfg->N; i++)
            {
                if (isnan(rt->Q[i]) || isinf(rt->Q[i]))
                {
                    rt->Q[i] = 0.0;
                }
            }

            retval = true;
        }
    }

    return retval;
}

private bool run_update_info(const iwt_runtime_t rt, const iwt_config_t cfg)
{
    bool retval = false;
    cl_kernel kernel = ocl_get_kernel(&rt->ocl, OCL_KERNEL_IWT_UPDATE_INFO);
    
    if (kernel != NULL)
    {
        // I_avg berechnen (einmal pro Iteration)
        double I_avg = 0.0;
        for (size_t i = 0; i < cfg->N; i++)
        {
            I_avg += rt->I[i];
        }
        I_avg /= cfg->N;

        int N = (int)cfg->N;
        int width = (int)sqrt(cfg->N);
        double DT = cfg->DT;
        double I_vac = cfg->I_vac;
        double phi_0 = cfg->phi_0;
        double omega_0 = cfg->omega_0;
        double D = cfg->D;
        double Z_0 = cfg->Z_0;

        clEnqueueWriteBuffer(rt->ocl.queue, rt->I_prev_gpu, CL_TRUE, 0,
            cfg->N * sizeof(double), rt->I, 0, NULL, NULL);
        clEnqueueWriteBuffer(rt->ocl.queue, rt->I_phase_prev_gpu, CL_TRUE, 0,
            cfg->N * sizeof(double), rt->I_phase, 0, NULL, NULL);

        clSetKernelArg(kernel, 0, sizeof(cl_mem), &rt->I_gpu);
        clSetKernelArg(kernel, 1, sizeof(cl_mem), &rt->I_phase_gpu);
        clSetKernelArg(kernel, 2, sizeof(cl_mem), &rt->I_prev_gpu);
        clSetKernelArg(kernel, 3, sizeof(cl_mem), &rt->I_phase_prev_gpu);
        clSetKernelArg(kernel, 4, sizeof(cl_mem), &rt->sumJ_gpu);
        clSetKernelArg(kernel, 5, sizeof(cl_mem), &rt->Q_gpu);
        clSetKernelArg(kernel, 6, sizeof(int), &N);
        clSetKernelArg(kernel, 7, sizeof(int), &width);
        clSetKernelArg(kernel, 8, sizeof(double), &DT);
        clSetKernelArg(kernel, 9, sizeof(double), &I_vac);
        clSetKernelArg(kernel, 10, sizeof(double), &phi_0);
        clSetKernelArg(kernel, 11, sizeof(double), &omega_0);
        clSetKernelArg(kernel, 12, sizeof(double), &D);
        clSetKernelArg(kernel, 13, sizeof(double), &Z_0);
        clSetKernelArg(kernel, 14, sizeof(double), &I_avg);

        size_t global = cfg->N;
        size_t local = 64;
        if (local > global) local = global;

        if (ocl_enqueue_kernel(&rt->ocl, kernel, global, local))
        {
            clEnqueueReadBuffer(rt->ocl.queue, rt->I_gpu, CL_TRUE,
                0, cfg->N * sizeof(double), rt->I, 0, NULL, NULL);
            clEnqueueReadBuffer(rt->ocl.queue, rt->I_phase_gpu, CL_TRUE,
                0, cfg->N * sizeof(double), rt->I_phase, 0, NULL, NULL);

            for (size_t i = 0; i < cfg->N; i++)
            {
                rt->I_prev[i] = rt->I[i];
                rt->I_phase_prev[i] = rt->I_phase[i];
            }

            retval = true;
        }
    }
    
    return retval;
}

bool run_simulation(const iwt_runtime_t rt, const iwt_config_t cfg)
{
    bool retval = false;

    // ============================================================
    // 100 Quellknoten mit I=1.0 initialisieren (Gesamtinformation = 100)
    // ============================================================
    for (int i = 0; i < 100 && i < (int)cfg->N; i++)
    {
        rt->I[i] = 1.0;
        rt->I_phase[i] = 0.0;
    }
    
	for (int i = 100; i < (int)cfg->N; i++)
    {
        rt->I[i] = 0.0;
        rt->I_phase[i] = 0.0;
    }

	string_clear_screen();
	string_set_cursor_position(1, 1);

    for (int iter = 0; iter < cfg->MAX_ITER; iter++)
    {
        // Vorherige Werte sichern
        for (int i = 0; i < 100 && i < (int)cfg->N; i++)
        {
            rt->I_prev[i] = rt->I[i];
            rt->I_phase_prev[i] = rt->I_phase[i];
        }

        // Daten auf GPU schreiben
        clEnqueueWriteBuffer(rt->ocl.queue, rt->I_gpu, CL_TRUE, 0,
            cfg->N * sizeof(double), rt->I, 0, NULL, NULL);
        clEnqueueWriteBuffer(rt->ocl.queue, rt->I_phase_gpu, CL_TRUE, 0,
            cfg->N * sizeof(double), rt->I_phase, 0, NULL, NULL);

        // ============================================================
        // SIMULATIONSSCHRITTE
        // ============================================================
        if (!run_flux_calculation(rt, cfg)) break;
        if (!run_q_calculation(rt, cfg)) break;
        if (!run_update_info(rt, cfg)) break;

        // ============================================================
        // STATISTIKEN BEREICHNEN
        // ============================================================
        double max_q = 0.0;
        double sum_abs = 0.0;
        double I_min = 1e30;
        double I_max = -1e30;

        for (size_t i = 0; i < cfg->N; i++)
        {
            sum_abs += rt->I[i];
            if (rt->I[i] < I_min) I_min = rt->I[i];
            if (rt->I[i] > I_max) I_max = rt->I[i];
            double abs_q = fabs(rt->Q[i]);
            if (abs_q > max_q) max_q = abs_q;
        }

        double mean_abs = sum_abs / cfg->N;
        double I_total = sum_abs;

        static double I_total_ref = -1.0;
        if (I_total_ref < 0.0) I_total_ref = I_total;
        double deviation = (I_total - I_total_ref) / (I_total_ref + 1e-30);

        iwt_print_status(rt, cfg, iter, max_q, I_total, I_min, I_max, mean_abs, deviation);

        retval = true;
    }

    return retval;
}
