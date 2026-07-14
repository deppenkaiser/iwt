#include <ocl/ocl.h>
#include <stdio.h>
#include "iwt.h"

bool run_flux_calculation_batched(const iwt_runtime_t rt, const iwt_config_t cfg)
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

            printf("sumJ[0..9]:\n");
            for (size_t i = 0; i < 10 && i < cfg->N; i++)
            {
                printf("  sumJ[%ld] = %f\n", i, rt->sumJ[i]);
            }

            retval = true;
        }
    }
    
    return retval;
}

bool run_q_calculation(const iwt_runtime_t rt, const iwt_config_t cfg)
{
	bool retval = false;

	// Nach run_flux_calculation_batched:
	cl_kernel kernel_q = ocl_get_kernel(&rt->ocl, OCL_KERNEL_IWT_Q);
	if (kernel_q != NULL)
	{
		clSetKernelArg(kernel_q, 0, sizeof(cl_mem), &rt->sumJ_gpu);
		clSetKernelArg(kernel_q, 1, sizeof(cl_mem), &rt->Q_gpu);
		int N = (int)cfg->N;
		clSetKernelArg(kernel_q, 2, sizeof(int), &N);

		size_t global = cfg->N;
		size_t local = 64;
		if (local > global) local = global;

		if (ocl_enqueue_kernel(&rt->ocl, kernel_q, global, local))
		{
			clEnqueueReadBuffer(rt->ocl.queue, rt->Q_gpu, CL_TRUE,
				0, cfg->N * sizeof(double), rt->Q, 0, NULL, NULL);

			printf("Q[0..9]:\n");
			for (size_t i = 0; i < 10 && i < cfg->N; i++)
			{
				printf("  Q[%ld] = %f\n", i, rt->Q[i]);
			}

			retval = true;
		}
	}

	return retval;
}

// iwt_kernel.c – Quantisierung in run_update_info

bool run_update_info(const iwt_runtime_t rt, const iwt_config_t cfg)
{
    bool retval = false;
    cl_kernel kernel = ocl_get_kernel(&rt->ocl, OCL_KERNEL_IWT_UPDATE_INFO);
    
    if (kernel != NULL)
    {
        int N = (int)cfg->N;
        double DT = cfg->DT;

        clSetKernelArg(kernel, 0, sizeof(cl_mem), &rt->I_gpu);
        clSetKernelArg(kernel, 1, sizeof(cl_mem), &rt->sumJ_gpu);
        clSetKernelArg(kernel, 2, sizeof(double), &DT);
        clSetKernelArg(kernel, 3, sizeof(int), &N);

        size_t global = cfg->N;
        size_t local = 64;
        if (local > global) local = global;

        if (ocl_enqueue_kernel(&rt->ocl, kernel, global, local))
        {
            // I zurücklesen
            clEnqueueReadBuffer(rt->ocl.queue, rt->I_gpu, CL_TRUE,
                0, cfg->N * sizeof(double), rt->I, 0, NULL, NULL);

            // Quantisierung der Information
            double I_min = 0.1;
            double Delta_I = cfg->Delta_I;

            for (size_t i = 0; i < cfg->N; i++)
            {
                double I_quantized = I_min + round((rt->I[i] - I_min) / Delta_I) * Delta_I;
                rt->I[i] = I_quantized;
            }

            // Quantisierte Werte zurück auf die GPU schreiben
            clEnqueueWriteBuffer(rt->ocl.queue, rt->I_gpu, CL_TRUE, 0,
                cfg->N * sizeof(double), rt->I, 0, NULL, NULL);

            printf("I[0..9] nach Update und Quantisierung:\n");
            for (size_t i = 0; i < 10 && i < cfg->N; i++)
            {
                printf("  I[%ld] = %f\n", i, rt->I[i]);
            }

            retval = true;
        }
    }
    
    return retval;
}

bool run_q_dynamics(const iwt_runtime_t rt, const iwt_config_t cfg)
{
    bool retval = false;
    cl_kernel kernel = ocl_get_kernel(&rt->ocl, OCL_KERNEL_IWT_Q_DYNAMICS);

    if (kernel != NULL)
    {
        double I_min = 0.1;
        double DT = cfg->DT;
        double ETA_Q = cfg->ETA_Q;
        double LAMBDA_Q = cfg->LAMBDA_Q;
        double GAMMA_Q = cfg->GAMMA_Q;
        int N = (int)cfg->N;

        clSetKernelArg(kernel, 0, sizeof(cl_mem), &rt->Q_gpu);
        clSetKernelArg(kernel, 1, sizeof(cl_mem), &rt->I_gpu);
        clSetKernelArg(kernel, 2, sizeof(cl_mem), &rt->K_gpu);
        clSetKernelArg(kernel, 3, sizeof(double), &I_min);
        clSetKernelArg(kernel, 4, sizeof(double), &DT);
        clSetKernelArg(kernel, 5, sizeof(double), &ETA_Q);
        clSetKernelArg(kernel, 6, sizeof(double), &LAMBDA_Q);
        clSetKernelArg(kernel, 7, sizeof(double), &GAMMA_Q);
        clSetKernelArg(kernel, 8, sizeof(int), &N);

        size_t global = cfg->N;
        size_t local = 64;
        if (local > global) local = global;

        if (ocl_enqueue_kernel(&rt->ocl, kernel, global, local))
        {
            // Q zurücklesen
            clEnqueueReadBuffer(rt->ocl.queue, rt->Q_gpu, CL_TRUE,
                0, cfg->N * sizeof(double), rt->Q, 0, NULL, NULL);

            retval = true;
        }
    }

    return retval;
}

bool run_update_coupling(const iwt_runtime_t rt, const iwt_config_t cfg)
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
        double I_min = iwt_I_min();        // NEU
        double I_max = iwt_I_max();        // NEU

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
            clSetKernelArg(kernel, 2, sizeof(double), &DT);
            clSetKernelArg(kernel, 3, sizeof(double), &ETA);
            clSetKernelArg(kernel, 4, sizeof(double), &LAMBDA);
            clSetKernelArg(kernel, 5, sizeof(double), &I_min);
            clSetKernelArg(kernel, 6, sizeof(double), &I_max);
            clSetKernelArg(kernel, 7, sizeof(int), &N);
            clSetKernelArg(kernel, 8, sizeof(int), &start);
            clSetKernelArg(kernel, 9, sizeof(int), &end);

            size_t global = batch_end - batch_start;
            size_t local = 64;
            if (local > global) local = global;

            all_ok = ocl_enqueue_kernel(&rt->ocl, kernel, global, local);
        }

        if (all_ok)
        {
            clEnqueueReadBuffer(rt->ocl.queue, rt->K_gpu, CL_TRUE,
                0, cfg->N * cfg->N * sizeof(double), rt->K, 0, NULL, NULL);

            printf("K[0][0..4] nach Update:\n");
            for (size_t i = 0; i < 5 && i < cfg->N; i++)
            {
                printf("  K[0][%ld] = %f\n", i, rt->K[i]);
            }

            retval = true;
        }
    }

    return retval;
}
