#include <ocl/ocl.h>
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

        clSetKernelArg(kernel, 0, sizeof(cl_mem), &rt->I_gpu);
        clSetKernelArg(kernel, 1, sizeof(cl_mem), &rt->K_gpu);
        clSetKernelArg(kernel, 2, sizeof(cl_mem), &rt->sumJ_gpu);
        clSetKernelArg(kernel, 3, sizeof(int), &N);
        clSetKernelArg(kernel, 4, sizeof(int), &start);
        clSetKernelArg(kernel, 5, sizeof(int), &end);
        clSetKernelArg(kernel, 6, sizeof(double), &DT);
        clSetKernelArg(kernel, 7, sizeof(double), &I_max);

        size_t global = cfg->N;
        size_t local = 64;
        if (local > global) local = global;

        if (ocl_enqueue_kernel(&rt->ocl, kernel, global, local))
        {
            clEnqueueReadBuffer(rt->ocl.queue, rt->sumJ_gpu, CL_TRUE,
                0, cfg->N * sizeof(double), rt->sumJ, 0, NULL, NULL);
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
    // EINMALIGE EINSPEISUNG – NUR HIER!
    // ============================================================
    rt->I[0] = iwt_I_max();   // 1.0
    rt->I_phase[0] = 0.0;

    for (int iter = 0; iter < cfg->MAX_ITER; iter++)
    {
        rt->I_prev[0] = rt->I[0];
        rt->I_phase_prev[0] = rt->I_phase[0];

        clEnqueueWriteBuffer(rt->ocl.queue, rt->I_gpu, CL_TRUE, 0,
            cfg->N * sizeof(double), rt->I, 0, NULL, NULL);
        clEnqueueWriteBuffer(rt->ocl.queue, rt->I_phase_gpu, CL_TRUE, 0,
            cfg->N * sizeof(double), rt->I_phase, 0, NULL, NULL);

        if (!run_flux_calculation(rt, cfg)) break;
        if (!run_q_calculation(rt, cfg)) break;
        if (!run_update_info(rt, cfg)) break;

        // ============================================================
        // AUSGABE (unverändert)
        // ============================================================
        double max_q = 0.0;
        double sum_abs = 0.0;
        double sum_phase = 0.0;
        double sum_abs_sq = 0.0;
        double I_min = 1e30;
        double I_max = -1e30;

        for (size_t i = 0; i < cfg->N; i++)
        {
            double I_i = rt->I[i];
            sum_abs += I_i;
            sum_phase += rt->I_phase[i];
            sum_abs_sq += I_i * I_i;
            if (I_i < I_min) I_min = I_i;
            if (I_i > I_max) I_max = I_i;
            double abs_q = rt->Q[i] > 0 ? rt->Q[i] : -rt->Q[i];
            if (abs_q > max_q) max_q = abs_q;
        }

        double mean_abs = sum_abs / cfg->N;
        double mean_phase = sum_phase / cfg->N;
        double rms = sqrt(sum_abs_sq / cfg->N);
        double I_total = sum_abs;

        printf("Iter %3d: ", iter);
        printf("max|Q|=%8.3e ", max_q);
        printf("mean_abs=%8.3e ", mean_abs);
        printf("rms=%8.3e ", rms);
        printf("min=%8.3e ", I_min);
        printf("max=%8.3e ", I_max);
        printf("I_total=%8.3e\n", I_total);

        printf("         I[0..9] =");
        for (int i = 0; i < 10 && i < (int)cfg->N; i++)
        {
            printf(" %8.3e", rt->I[i]);
        }
        printf("\n");

        printf("         sumJ[0..4] =");
        for (int i = 0; i < 5 && i < (int)cfg->N; i++)
        {
            printf(" %8.3e", rt->sumJ[i]);
        }
        printf("\n");

        printf("         I_phase[0..4] =");
        for (int i = 0; i < 5 && i < (int)cfg->N; i++)
        {
            printf(" %8.3e", rt->I_phase[i]);
        }
        printf("\n");

        static double I_total_ref = -1.0;
        if (I_total_ref < 0.0) I_total_ref = I_total;
        double deviation = (I_total - I_total_ref) / (I_total_ref + 1e-30);
        printf("         Erhaltung: I_total/I_ref = %8.6f (Abw. %8.3e)\n",
               I_total / I_total_ref, deviation);

        retval = true;
    }

    return retval;
}
