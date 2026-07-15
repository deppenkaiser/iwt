#include "init.h"
#include <stdio.h>
#include <math.h>

bool initialize_host_data(const iwt_runtime_t rt, const iwt_config_t cfg)
{
    bool retval = false;

    rt->I = malloc(cfg->N * sizeof(double));
    rt->K = malloc(cfg->N * cfg->N * sizeof(double));
    rt->sumJ = malloc(cfg->N * sizeof(double));
    rt->Q = malloc(cfg->N * sizeof(double));

    if ((rt->I != NULL) && (rt->K != NULL) && (rt->sumJ != NULL) && (rt->Q != NULL))
    {
        // Initialisierung von I
		for (size_t i = 0; i < cfg->N; i++)
		{
			double r = (double)rand() / RAND_MAX;
			// 95% Vakuum, 5% höhere Werte
			if (r < 0.95)
			{
				rt->I[i] = 0.01 + 0.005 * ((double)rand() / RAND_MAX - 0.5);
			}
			else
			{
				// 5% der Knoten haben Werte zwischen 0.05 und 1.0
				rt->I[i] = 0.05 + 0.95 * ((double)rand() / RAND_MAX);
			}
		}

        // Anfangsquantisierung
        double I_min = iwt_I_min();      // 0.0
        double I_max = iwt_I_max();      // 1.0
        double Delta_I = iwt_delta_I();  // 2.3283064365e-10

        for (size_t i = 0; i < cfg->N; i++)
        {
            if (rt->I[i] < I_min) rt->I[i] = I_min;
            if (rt->I[i] > I_max) rt->I[i] = I_max;
            rt->I[i] = I_min + round((rt->I[i] - I_min) / Delta_I) * Delta_I;
        }

		printf("Initial I[0..9]:\n");
		for (size_t i = 0; i < cfg->N; i++)
		{
			double r = (double)rand() / RAND_MAX;
			rt->I[i] = 0.01 + 0.001 * (r - 0.5);
		}

        // Q aus I ableiten
        for (size_t i = 0; i < cfg->N; i++)
        {
            rt->Q[i] = -(rt->I[i] - I_min);
        }

        // K initialisieren
		for (size_t i = 0; i < cfg->N; i++)
		{
			double gi = iwt_g(rt->I[i]);
			for (size_t j = 0; j < cfg->N; j++)
			{
				double gj = iwt_g(rt->I[j]);
				rt->K[i * cfg->N + j] = gi * gj;
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
