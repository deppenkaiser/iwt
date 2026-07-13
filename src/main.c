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
	rt->I = rt->K = rt->sumJ = NULL;
}

bool initialize_gpu_data(const iwt_runtime_t rt, const iwt_config_t cfg)
{
	// GPU-Speicher allozieren
	rt->I_gpu = ocl_create_buffer(&rt->ocl, OCL_BUF_READ_WRITE, cfg->N * sizeof(double), NULL);
	rt->K_gpu = ocl_create_buffer(&rt->ocl, OCL_BUF_READ_WRITE, cfg->N * cfg->N * sizeof(double), NULL);
	rt->sumJ_gpu = ocl_create_buffer(&rt->ocl, OCL_BUF_WRITE_ONLY, cfg->N * sizeof(double), NULL);
	return (rt->I_gpu != NULL) && (rt->K_gpu != NULL) && (rt->sumJ_gpu != NULL);
}

void deinitialize_gpu_data(const iwt_runtime_t rt)
{
	clReleaseMemObject(rt->I_gpu);
	clReleaseMemObject(rt->K_gpu);
	clReleaseMemObject(rt->sumJ_gpu);
	rt->I_gpu = rt->K_gpu = rt->sumJ_gpu = NULL;
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
							printf("N = %ld, DT = %f\n", cfg.N, cfg.DT);
							retval = 0;
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
