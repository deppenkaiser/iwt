#include <ocl/ocl.h>
#include <stdio.h>
#include <math.h>
#include <api/api.h>
#include "iwt.h"

private bool run_flux_calculation_batched(const iwt_runtime_t rt, const iwt_config_t cfg);
private bool run_update_info(const iwt_runtime_t rt, const iwt_config_t cfg);
private double compute_energy(const iwt_runtime_t rt, const iwt_config_t cfg);

private bool run_flux_calculation_batched(const iwt_runtime_t rt, const iwt_config_t cfg)
{
    bool retval = false;
    cl_kernel kernel = ocl_get_kernel(&rt->ocl, OCL_KERNEL_IWT_FLUX);
    
    if (kernel != NULL)
    {
        bool all_ok = true;
        size_t num_batches = cfg->N / cfg->BATCH_SIZE;
        if (cfg->N % cfg->BATCH_SIZE != 0) num_batches++;

        clEnqueueWriteBuffer(rt->ocl.queue, rt->I_gpu, CL_TRUE, 0,
            cfg->N * sizeof(double), rt->I, 0, NULL, NULL);
        clEnqueueWriteBuffer(rt->ocl.queue, rt->I_phase_gpu, CL_TRUE, 0,
            cfg->N * sizeof(double), rt->I_phase, 0, NULL, NULL);
        clEnqueueWriteBuffer(rt->ocl.queue, rt->K_gpu, CL_TRUE, 0,
            cfg->N * cfg->N * sizeof(double), rt->K, 0, NULL, NULL);

		for (size_t batch = 0; (batch < num_batches) && all_ok; ++batch)
		{
			size_t batch_start = batch * cfg->BATCH_SIZE;
			size_t batch_end = batch_start + cfg->BATCH_SIZE;
			if (batch_end > cfg->N) batch_end = cfg->N;

			int N = (int)cfg->N;
			int start = (int)batch_start;
			int end = (int)batch_end;

			clSetKernelArg(kernel, 0, sizeof(cl_mem), &rt->I_gpu);
			clSetKernelArg(kernel, 1, sizeof(cl_mem), &rt->I_phase_gpu);
			clSetKernelArg(kernel, 2, sizeof(cl_mem), &rt->K_gpu);
			clSetKernelArg(kernel, 3, sizeof(cl_mem), &rt->sumJ_gpu);
			clSetKernelArg(kernel, 4, sizeof(int), &N);
			clSetKernelArg(kernel, 5, sizeof(int), &start);
			clSetKernelArg(kernel, 6, sizeof(int), &end);

			size_t global = batch_end - batch_start;
			size_t local = 64;
			if (local > global) local = global;

			all_ok = ocl_enqueue_kernel(&rt->ocl, kernel, global, local);
		}

        if (all_ok)
        {
            clEnqueueReadBuffer(rt->ocl.queue, rt->sumJ_gpu, CL_TRUE,
                0, cfg->N * 2 * sizeof(double), rt->sumJ, 0, NULL, NULL);
            retval = true;
        }
    }
    
    return retval;
}

private bool run_q_calculation(const iwt_runtime_t rt, const iwt_config_t cfg)
{
    // Q wird aus der Amplitude I berechnet (nicht aus I_phase)
    // Funktion bleibt beim Namen
    bool retval = false;
    cl_kernel kernel = ocl_get_kernel(&rt->ocl, OCL_KERNEL_IWT_Q);
    
    if (kernel != NULL)
    {
        double sum_abs_sq = 0.0;
        for (size_t i = 0; i < cfg->N; i++)
        {
            sum_abs_sq += rt->I[i] * rt->I[i];
        }
        if (sum_abs_sq < 1e-30) return false;

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
        int N = (int)cfg->N;
        int width = (int)sqrt(cfg->N);
        double DT = cfg->DT;
        double I_vac = cfg->I_vac;
        double phi_0 = cfg->phi_0;
        double omega_0 = cfg->omega_0;
        double D = cfg->D;

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

double compute_energy(const iwt_runtime_t rt, const iwt_config_t cfg)
{
    double E = 0.0;
    double DT = cfg->DT;

    for (size_t k = 0; k < cfg->N; k++)
    {
        double dI_dt = (rt->I[k] - rt->I_prev[k]) / DT;
        double E_kin = 0.5 * dI_dt * dI_dt;

        double E_pot = 0.0;
        for (size_t j = 0; j < cfg->N; j++)
        {
            double diff = rt->I[j] - rt->I[k];
            E_pot += 0.5 * rt->K[k * cfg->N + j] * diff * diff;
        }

        E += E_kin + E_pot;
    }

    return E;
}

bool run_simulation(const iwt_runtime_t rt, const iwt_config_t cfg)
{
    bool retval = false;

    for (int iter = 0; iter < cfg->MAX_ITER; iter++)
    {
        // Impuls an Knoten 0 (Amplitude)
        if (iter == 0)
        {
            rt->I[0] = iwt_I_max();
            rt->I_phase[0] = 0.0;
        }
        else
        {
            rt->I[0] = iwt_I_min();
            rt->I_phase[0] = 0.0;
        }
        rt->I_prev[0] = rt->I[0];
        rt->I_phase_prev[0] = rt->I_phase[0];

        clEnqueueWriteBuffer(rt->ocl.queue, rt->I_gpu, CL_TRUE, 0,
            cfg->N * sizeof(double), rt->I, 0, NULL, NULL);
        clEnqueueWriteBuffer(rt->ocl.queue, rt->I_phase_gpu, CL_TRUE, 0,
            cfg->N * sizeof(double), rt->I_phase, 0, NULL, NULL);

        if (!run_flux_calculation_batched(rt, cfg)) break;
        if (!run_q_calculation(rt, cfg)) break;
        if (!run_update_info(rt, cfg)) break;

        // Ausgabe
        double max_q = 0.0;
        double sum_abs = 0.0;
        double sum_phase = 0.0;
        for (size_t i = 0; i < cfg->N; i++)
        {
            sum_abs += rt->I[i];
            sum_phase += rt->I_phase[i];
            double abs_q = rt->Q[i] > 0 ? rt->Q[i] : -rt->Q[i];
            if (abs_q > max_q) max_q = abs_q;
        }

        double mean_abs = sum_abs / cfg->N;
        double mean_phase = sum_phase / cfg->N;

        printf("Iter %3d: max|Q| = %e, mean_abs = %f, mean_phase = %f\n",
               iter, max_q, mean_abs, mean_phase);

        retval = true;
    }

    return retval;
}
