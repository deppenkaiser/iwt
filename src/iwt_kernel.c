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

            printf("I[0..9] nach Update:\n");
            for (size_t i = 0; i < 10 && i < cfg->N; i++)
            {
                printf("  I[%ld] = %f\n", i, rt->I[i]);
            }

            retval = true;
        }
    }
    
    return retval;
}
