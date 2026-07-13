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
        double ETA = 0.001;   // später als Parameter
        double LAMBDA = 0.1;

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
            clSetKernelArg(kernel, 5, sizeof(int), &N);
            clSetKernelArg(kernel, 6, sizeof(int), &start);
            clSetKernelArg(kernel, 7, sizeof(int), &end);

            size_t global = batch_end - batch_start;
            size_t local = 64;
            if (local > global) local = global;

            all_ok = ocl_enqueue_kernel(&rt->ocl, kernel, global, local);
        }

        if (all_ok)
        {
            // K zurücklesen (optional, zur Prüfung)
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

bool run_simulation(const iwt_runtime_t rt, const iwt_config_t cfg)
{
    bool retval = false;

    for (int iter = 0; iter < cfg->MAX_ITER; iter++)
    {
        if (!run_flux_calculation_batched(rt, cfg)) break;
        if (!run_q_calculation(rt, cfg)) break;
        if (!run_update_info(rt, cfg)) break;
        if (!run_update_coupling(rt, cfg)) break;

        // max|Q| berechnen (auf CPU)
        double max_q = 0.0;
        for (size_t i = 0; i < cfg->N; i++)
        {
            double abs_q = rt->Q[i] > 0 ? rt->Q[i] : -rt->Q[i];
            if (abs_q > max_q) max_q = abs_q;
        }

        printf("Iter %2d: max|Q| = %e\n", iter, max_q);

        if (max_q < cfg->THRESHOLD)
        {
            printf("Konvergenz erreicht nach %d Iterationen.\n", iter + 1);
            retval = true;
            break;
        }

        // I und K für nächste Iteration auf GPU neu schreiben
        clEnqueueWriteBuffer(rt->ocl.queue, rt->I_gpu, CL_TRUE, 0,
            cfg->N * sizeof(double), rt->I, 0, NULL, NULL);

        clEnqueueWriteBuffer(rt->ocl.queue, rt->K_gpu, CL_TRUE, 0,
            cfg->N * cfg->N * sizeof(double), rt->K, 0, NULL, NULL);
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
	cfg.THRESHOLD = 1e-6;
	cfg.MAX_ITER = 100;

	cfg.DT = 1e-5;
	cfg.ETA = 1e-6;
	cfg.LAMBDA = 1.0;

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
						if (run_simulation(&rt, &cfg))
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
