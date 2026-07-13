#include <stdio.h>
#include <ocl/ocl.h>
#include "iwt_kernel.h"

bool initialize_host_data(const iwt_runtime_t rt, const iwt_config_t cfg)
{
	bool retval = false;

	// Host-Speicher allozieren
	rt->I = malloc(cfg->N * sizeof(double));
	rt->K = malloc(cfg->N * cfg->N * sizeof(double));
	rt->sumJ = malloc(cfg->N * sizeof(double));
	rt->Q = malloc(cfg->N * sizeof(double));

	if ((rt->I != NULL) && (rt->K != NULL) && (rt->sumJ != NULL) && (rt->Q != NULL))
	{
		for (size_t i = 0; i < cfg->N; i++)
		{
			rt->I[i] = (double)(i + 1);  // 1, 2, 3, ...
		}

		for (size_t i = 0; i < cfg->N; i++)
		{
			for (size_t j = 0; j < cfg->N; j++)
			{
				rt->K[i * cfg->N + j] = 1.0;  // Einsermatrix
			}
		}

		retval = true;
	}

	return retval;	
}

void deinitialize_host_data(const iwt_runtime_t rt)
{
	free(rt->I);
	free(rt->K);
	free(rt->sumJ);
	free(rt->Q);
	rt->I = rt->K = rt->sumJ = rt->Q = NULL;
}

bool initialize_gpu_data(const iwt_runtime_t rt, const iwt_config_t cfg)
{
	// GPU-Speicher allozieren
	rt->I_gpu = ocl_create_buffer(&rt->ocl, OCL_BUF_READ_WRITE, cfg->N * sizeof(double), NULL);
	rt->K_gpu = ocl_create_buffer(&rt->ocl, OCL_BUF_READ_WRITE, cfg->N * cfg->N * sizeof(double), NULL);
	rt->sumJ_gpu = ocl_create_buffer(&rt->ocl, OCL_BUF_WRITE_ONLY, cfg->N * sizeof(double), NULL);
	rt->Q_gpu = ocl_create_buffer(&rt->ocl, OCL_BUF_READ_WRITE, cfg->N * sizeof(double), NULL);
	return (rt->I_gpu != NULL) && (rt->K_gpu != NULL) && (rt->sumJ_gpu != NULL) && (rt->Q_gpu != NULL);
}

void deinitialize_gpu_data(const iwt_runtime_t rt)
{
	clReleaseMemObject(rt->I_gpu);
	clReleaseMemObject(rt->K_gpu);
	clReleaseMemObject(rt->sumJ_gpu);
	rt->I_gpu = rt->K_gpu = rt->sumJ_gpu = NULL;
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

int main(void)
{
	int retval = 1;
    struct iwt_config cfg = {0};
    struct iwt_runtime rt = {0};

    cfg.N = 4096;
	cfg.BATCH_SIZE = 512;
    cfg.DT = 0.01;

    if (ocl_initialize(&rt.ocl))
	{
		if (ocl_compile(&rt.ocl))
		{
			if (ocl_load_kernels(&rt.ocl))
			{
				if (initialize_host_data(&rt, &cfg))
				{
					if (initialize_gpu_data(&rt, &cfg))
					{
						if (run_flux_calculation_batched(&rt, &cfg))
						{
							if (run_q_calculation(&rt, &cfg))
							{
								if (run_update_info(&rt, &cfg))
								{
									printf("N = %ld, DT = %f\n", cfg.N, cfg.DT);
									retval = 0;
								}
							}
						}

						deinitialize_gpu_data(&rt);
					}
				}

				deinitialize_host_data(&rt);
			}
		}
	
		ocl_deinitialize(&rt.ocl);
	}

    return retval;
}
