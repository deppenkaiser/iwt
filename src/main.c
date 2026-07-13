#include <stdio.h>
#include <ocl/ocl.h>

typedef struct iwt_config
{
    size_t N;
    double DT;
} *iwt_config_t;

typedef struct iwt_runtime
{
    double *I;
    double *K;
    double *sumJ;

    cl_mem I_gpu;
    cl_mem K_gpu;
    cl_mem sumJ_gpu;

    struct ocl_core ocl;
} *iwt_runtime_t;

bool initialize_host_data(const iwt_runtime_t rt, const iwt_config_t cfg)
{
	bool retval = false;

	// Host-Speicher allozieren
	rt->I = malloc(cfg->N * sizeof(double));
	rt->K = malloc(cfg->N * cfg->N * sizeof(double));
	rt->sumJ = malloc(cfg->N * sizeof(double));

	if ((rt->I != NULL) && (rt->K != NULL) && (rt->sumJ != NULL))
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
}

bool run_flux_calculation(const iwt_runtime_t rt, const iwt_config_t cfg)
{
	bool retval = false;
    cl_kernel kernel = ocl_get_kernel(&rt->ocl, OCL_KERNEL_IWT_FLUX);

    if (kernel != NULL)
    {
        int batch_start = 0;
        int batch_end = (int)cfg->N;
        int N = (int)cfg->N;

		clEnqueueWriteBuffer(rt->ocl.queue, rt->I_gpu, CL_TRUE, 0,
			cfg->N * sizeof(double), rt->I, 0, NULL, NULL);

		clEnqueueWriteBuffer(rt->ocl.queue, rt->K_gpu, CL_TRUE, 0,
			cfg->N * cfg->N * sizeof(double), rt->K, 0, NULL, NULL);

		clSetKernelArg(kernel, 0, sizeof(cl_mem), &rt->I_gpu);
        clSetKernelArg(kernel, 1, sizeof(cl_mem), &rt->K_gpu);
        clSetKernelArg(kernel, 2, sizeof(cl_mem), &rt->sumJ_gpu);
        clSetKernelArg(kernel, 3, sizeof(int), &N);
        clSetKernelArg(kernel, 4, sizeof(int), &batch_start);
        clSetKernelArg(kernel, 5, sizeof(int), &batch_end);

        size_t global = cfg->N;
        size_t local = 64;
        if (local > global) local = global;

        if (ocl_enqueue_kernel(&rt->ocl, kernel, global, local))
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

int main(void)
{
	int retval = 1;
    struct iwt_config cfg = {0};
    struct iwt_runtime rt = {0};

    cfg.N = 4096;
    cfg.DT = 0.01;

    if (ocl_initialize(&rt.ocl))
	{
		if (ocl_compile(&rt.ocl))
		{
			if (ocl_load_kernels(&rt.ocl))
			{
				if (initialize_host_data(&rt, &cfg))
				{
					// GPU-Speicher allozieren
					rt.I_gpu = ocl_create_buffer(&rt.ocl, OCL_BUF_READ_WRITE, cfg.N * sizeof(double), NULL);
					rt.K_gpu = ocl_create_buffer(&rt.ocl, OCL_BUF_READ_WRITE, cfg.N * cfg.N * sizeof(double), NULL);
					rt.sumJ_gpu = ocl_create_buffer(&rt.ocl, OCL_BUF_WRITE_ONLY, cfg.N * sizeof(double), NULL);

					if ((rt.I_gpu != NULL) && (rt.K_gpu != NULL) && (rt.sumJ_gpu != NULL))
					{
						if (run_flux_calculation(&rt, &cfg))
						{
							printf("N = %ld, DT = %f\n", cfg.N, cfg.DT);
							retval = 0;
						}
					}

					// Aufräumen
					clReleaseMemObject(rt.I_gpu);
					clReleaseMemObject(rt.K_gpu);
					clReleaseMemObject(rt.sumJ_gpu);
				}

				deinitialize_host_data(&rt);
			}
		}
	
		ocl_deinitialize(&rt.ocl);
	}

    return retval;
}
