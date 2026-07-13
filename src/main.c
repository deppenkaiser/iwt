#include <stdio.h>
#include <ocl/ocl.h>

struct iwt_config
{
    int N;
    double DT;
};

struct iwt_runtime
{
    double *I;
    double *K;
    double *sumJ;

    cl_mem I_gpu;
    cl_mem K_gpu;
    cl_mem sumJ_gpu;

    struct ocl_core ocl;
};

int main(void)
{
    struct iwt_config cfg = {0};
    struct iwt_runtime rt = {0};

    cfg.N = 4096;
    cfg.DT = 0.01;

    if (ocl_initialize(&rt.ocl))
	{
		// Host-Speicher allozieren
		rt.I = malloc(cfg.N * sizeof(double));
		rt.K = malloc(cfg.N * cfg.N * sizeof(double));
		rt.sumJ = malloc(cfg.N * sizeof(double));

		if ((rt.I != NULL) && (rt.K != NULL) && (rt.sumJ != NULL))
		{
			// GPU-Speicher allozieren
			rt.I_gpu = ocl_create_buffer(&rt.ocl, OCL_BUF_READ_WRITE, cfg.N * sizeof(double), NULL);
			rt.K_gpu = ocl_create_buffer(&rt.ocl, OCL_BUF_READ_WRITE, cfg.N * cfg.N * sizeof(double), NULL);
			rt.sumJ_gpu = ocl_create_buffer(&rt.ocl, OCL_BUF_WRITE_ONLY, cfg.N * sizeof(double), NULL);

			if ((rt.I_gpu != NULL) && (rt.K_gpu != NULL) && (rt.sumJ_gpu != NULL))
			{
    			printf("N = %d, DT = %f\n", cfg.N, cfg.DT);
			}

			// Aufräumen
			clReleaseMemObject(rt.I_gpu);
			clReleaseMemObject(rt.K_gpu);
			clReleaseMemObject(rt.sumJ_gpu);
		}

		free(rt.I);
		free(rt.K);
		free(rt.sumJ);

		ocl_deinitialize(&rt.ocl);
	}

    return 0;
}