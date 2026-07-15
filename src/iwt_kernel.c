#include <ocl/ocl.h>
#include <stdio.h>
#include <math.h>
#include <api/api.h>
#include "iwt.h"

private bool run_flux_calculation_batched(const iwt_runtime_t rt, const iwt_config_t cfg);
private bool run_q_calculation(const iwt_runtime_t rt, const iwt_config_t cfg);
private bool run_update_info(const iwt_runtime_t rt, const iwt_config_t cfg);
private bool run_update_coupling(const iwt_runtime_t rt, const iwt_config_t cfg);
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
            clSetKernelArg(kernel, 1, sizeof(cl_mem), &rt->K_gpu);
            clSetKernelArg(kernel, 2, sizeof(cl_mem), &rt->sumJ_gpu);
            clSetKernelArg(kernel, 3, sizeof(int), &N);
            clSetKernelArg(kernel, 4, sizeof(int), &start);
            clSetKernelArg(kernel, 5, sizeof(int), &end);

            size_t global = batch_end - batch_start;
            size_t local = 64;
            if (local > global) local = global;

            all_ok = ocl_enqueue_kernel(&rt->ocl, kernel, global, local);
        }

        if (all_ok)
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
        double sum_I = 0.0;
        for (size_t i = 0; i < cfg->N; i++)
        {
            sum_I += rt->I[i];
        }
        if (sum_I < 1e-30) return false;

        int N = (int)cfg->N;
        double hbar = 1.0;
        double m = 1.0;
        double prefactor = -(hbar * hbar) / (2.0 * m);

		clSetKernelArg(kernel, 0, sizeof(cl_mem), &rt->I_gpu);
		clSetKernelArg(kernel, 1, sizeof(cl_mem), &rt->Q_gpu);
		clSetKernelArg(kernel, 2, sizeof(int), &N);
		clSetKernelArg(kernel, 3, sizeof(double), &sum_I);
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
        double DT = cfg->DT;

        // I_prev auf GPU schreiben (vor dem Update)
        clEnqueueWriteBuffer(rt->ocl.queue, rt->I_prev_gpu, CL_TRUE, 0,
            cfg->N * sizeof(double), rt->I, 0, NULL, NULL);

        double GAMMA_ENTROPY = cfg->GAMMA_ENTROPY;
        double I0 = cfg->I0;
        double ETA_Q_POTENTIAL = 0.001;   // Q als Potential
        double ETA_SOURCE = 0.001;         // Materieentstehung durch Q

        clSetKernelArg(kernel, 0, sizeof(cl_mem), &rt->I_gpu);
        clSetKernelArg(kernel, 1, sizeof(cl_mem), &rt->I_prev_gpu);
        clSetKernelArg(kernel, 2, sizeof(cl_mem), &rt->sumJ_gpu);
        clSetKernelArg(kernel, 3, sizeof(cl_mem), &rt->Q_gpu);   // NEU
        clSetKernelArg(kernel, 4, sizeof(double), &DT);
        clSetKernelArg(kernel, 5, sizeof(double), &GAMMA_ENTROPY);
        clSetKernelArg(kernel, 6, sizeof(double), &I0);
        clSetKernelArg(kernel, 7, sizeof(double), &ETA_Q_POTENTIAL);
        clSetKernelArg(kernel, 8, sizeof(double), &ETA_SOURCE);
        clSetKernelArg(kernel, 9, sizeof(int), &N);

        size_t global = cfg->N;
        size_t local = 64;
        if (local > global) local = global;

        if (ocl_enqueue_kernel(&rt->ocl, kernel, global, local))
        {
            clEnqueueReadBuffer(rt->ocl.queue, rt->I_gpu, CL_TRUE,
                0, cfg->N * sizeof(double), rt->I, 0, NULL, NULL);

            // I_prev updaten (für nächsten Schritt)
            for (size_t i = 0; i < cfg->N; i++)
            {
                rt->I_prev[i] = rt->I[i];
            }

            // Begrenzung auf [I_min, I_max]
            double I_min = cfg->I_min;
            double I_max = cfg->I_max;
            for (size_t i = 0; i < cfg->N; i++)
            {
                if (rt->I[i] < I_min) rt->I[i] = I_min;
                if (rt->I[i] > I_max) rt->I[i] = I_max;
            }

            clEnqueueWriteBuffer(rt->ocl.queue, rt->I_gpu, CL_TRUE, 0,
                cfg->N * sizeof(double), rt->I, 0, NULL, NULL);

            retval = true;
        }
    }
    
    return retval;
}

private bool run_update_coupling(const iwt_runtime_t rt, const iwt_config_t cfg)
{
    bool retval = false;
    cl_kernel kernel = ocl_get_kernel(&rt->ocl, OCL_KERNEL_IWT_UPDATE_COUPLING);
    
    if (kernel != NULL)
    {
        bool all_ok = true;
        size_t num_batches = cfg->N / cfg->BATCH_SIZE;
        if (cfg->N % cfg->BATCH_SIZE != 0) num_batches++;

        double DT = cfg->DT;
        double ETA = cfg->ETA;
        double LAMBDA = cfg->LAMBDA;
        double MU = cfg->MU;
        double GAMMA = 1.0;
        double L = 1.0;
        double I_min = cfg->I_min;
        double I_max = cfg->I_max;

        for (size_t batch = 0; (batch < num_batches) && all_ok; ++batch)
        {
            size_t batch_start = batch * cfg->BATCH_SIZE;
            size_t batch_end = batch_start + cfg->BATCH_SIZE;
            if (batch_end > cfg->N) batch_end = cfg->N;

            int N = (int)cfg->N;
            int start = (int)batch_start;
            int end = (int)batch_end;

            clSetKernelArg(kernel, 0, sizeof(cl_mem), &rt->I_gpu);
            clSetKernelArg(kernel, 1, sizeof(cl_mem), &rt->Q_gpu);
            clSetKernelArg(kernel, 2, sizeof(cl_mem), &rt->K_gpu);
            clSetKernelArg(kernel, 3, sizeof(double), &DT);
            clSetKernelArg(kernel, 4, sizeof(double), &ETA);
            clSetKernelArg(kernel, 5, sizeof(double), &LAMBDA);
            clSetKernelArg(kernel, 6, sizeof(double), &MU);
            clSetKernelArg(kernel, 7, sizeof(double), &GAMMA);
            clSetKernelArg(kernel, 8, sizeof(double), &L);
            clSetKernelArg(kernel, 9, sizeof(double), &I_min);
            clSetKernelArg(kernel, 10, sizeof(double), &I_max);
            clSetKernelArg(kernel, 11, sizeof(int), &N);
            clSetKernelArg(kernel, 12, sizeof(int), &start);
            clSetKernelArg(kernel, 13, sizeof(int), &end);

            size_t global = batch_end - batch_start;
            size_t local = 64;
            if (local > global) local = global;

            all_ok = ocl_enqueue_kernel(&rt->ocl, kernel, global, local);
        }

        if (all_ok)
        {
            clEnqueueReadBuffer(rt->ocl.queue, rt->K_gpu, CL_TRUE,
                0, cfg->N * cfg->N * sizeof(double), rt->K, 0, NULL, NULL);
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
        // 1. Kinetische Energie: 0.5 * ((I_k - I_prev_k) / DT)^2
        double dI_dt = (rt->I[k] - rt->I_prev[k]) / DT;
        double E_kin = 0.5 * dI_dt * dI_dt;

        // 2. Potentielle Energie: 0.5 * sum_j K_kj * (I_j - I_k)^2
        double E_pot = 0.0;
        for (size_t j = 0; j < cfg->N; j++)
        {
            double diff = rt->I[j] - rt->I[k];
            E_pot += 0.5 * rt->K[k * cfg->N + j] * diff * diff;
        }

        // 3. Entropie-Energie: gamma * I_k * ln(I_k / I0)
        double I_k = rt->I[k];
        double E_entropy = cfg->GAMMA_ENTROPY * I_k * log(I_k / cfg->I0 + 1e-30);

        E += E_kin + E_pot + E_entropy;
    }

    return E;
}

bool run_simulation(const iwt_runtime_t rt, const iwt_config_t cfg)
{
    bool retval = false;

    for (int iter = 0; iter < cfg->MAX_ITER; iter++)
    {
		// Knoten 0 als periodischer Impuls
		if (iter == 1)
		{
			rt->I[0] = 1.0;
		}
		else
		{
			rt->I[0] = 0.01;
		}
		rt->I_prev[0] = rt->I[0];

        clEnqueueWriteBuffer(rt->ocl.queue, rt->I_gpu, CL_TRUE, 0,
            cfg->N * sizeof(double), rt->I, 0, NULL, NULL);

        if (!run_flux_calculation_batched(rt, cfg)) break;
        if (!run_q_calculation(rt, cfg)) break;
        if (!run_update_info(rt, cfg)) break;
        if (!run_update_coupling(rt, cfg)) break;

        // Ausgabe: Statistiken
        double max_q = 0.0;
        for (size_t i = 0; i < cfg->N; i++)
        {
            double abs_q = rt->Q[i] > 0 ? rt->Q[i] : -rt->Q[i];
            if (abs_q > max_q) max_q = abs_q;
        }

        double sum_I = 0.0;
        double sum_Q = 0.0;
        double sum_K = 0.0;
        double max_K = 0.0;
        double min_K = 0.0;

        for (size_t i = 0; i < cfg->N; i++)
        {
            sum_I += rt->I[i];
            sum_Q += rt->Q[i];
            for (size_t j = 0; j < cfg->N; j++)
            {
                double Kij = rt->K[i * cfg->N + j];
                sum_K += Kij;
                if (Kij > max_K) max_K = Kij;
                if (Kij < min_K) min_K = Kij;
            }
        }

        double mean_I = sum_I / cfg->N;
        double mean_Q = sum_Q / cfg->N;
        double mean_K = sum_K / (cfg->N * cfg->N);
		double E = compute_energy(rt, cfg);
		printf("Iter %2d: max|Q| = %e, mean_I = %f, mean_Q = %f, mean_K = %f, E = %f\n",
			iter, max_q, mean_I, mean_Q, mean_K, E);

        retval = true;
    }

    return retval;
}
