#include <stdio.h>
#include <ocl/ocl.h>
#include <math.h>
#include "iwt.h"
#include "iwt_kernel.h"

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
            if (i < cfg->N / 4)
                rt->I[i] = iwt_I_max();
            else
                rt->I[i] = iwt_I_min();
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
		for (size_t i = 0; i < 10 && i < cfg->N; i++)
		{
			printf("  I[%ld] = %f\n", i, rt->I[i]);
		}

        // Q aus I ableiten
        for (size_t i = 0; i < cfg->N; i++)
        {
            rt->Q[i] = -(rt->I[i] - I_min);
        }

        // K initialisieren
        for (size_t i = 0; i < cfg->N; i++)
        {
            for (size_t j = 0; j < cfg->N; j++)
            {
                rt->K[i * cfg->N + j] = 1.0;
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

bool run_simulation(const iwt_runtime_t rt, const iwt_config_t cfg)
{
    bool retval = false;

    for (int iter = 0; iter < cfg->MAX_ITER; iter++)
    {
        // Anfangsquantisierung (zu Beginn jedes Zeitschritts)
        double I_min = cfg->I_min;
        double Delta_I = cfg->Delta_I;
        for (size_t i = 0; i < cfg->N; i++)
        {
            rt->I[i] = I_min + round((rt->I[i] - I_min) / Delta_I) * Delta_I;
        }
        clEnqueueWriteBuffer(rt->ocl.queue, rt->I_gpu, CL_TRUE, 0,
            cfg->N * sizeof(double), rt->I, 0, NULL, NULL);

        if (!run_flux_calculation_batched(rt, cfg)) break;
        if (!run_q_calculation(rt, cfg)) break;
        if (!run_q_dynamics(rt, cfg)) break;
        if (!run_update_info(rt, cfg)) break;  // Endquantisierung

		printf("I[0..9] nach Iteration %d:\n", iter);
		for (size_t i = 0; i < 10 && i < cfg->N; i++)
		{
			printf("  I[%ld] = %f\n", i, rt->I[i]);
		}		

        if (!run_update_coupling(rt, cfg)) break;

        // Ausgabe
        double max_q = 0.0;
        for (size_t i = 0; i < cfg->N; i++)
        {
            double abs_q = rt->Q[i] > 0 ? rt->Q[i] : -rt->Q[i];
            if (abs_q > max_q) max_q = abs_q;
        }

        printf("Iter %2d: max|Q| = %e\n", iter, max_q);

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

        printf("Iter %2d: max|Q| = %e, mean_I = %f, mean_Q = %f, mean_K = %f, max_K = %f, min_K = %f\n",
            iter, max_q, mean_I, mean_Q, mean_K, max_K, min_K);

        if (max_q < cfg->THRESHOLD)
        {
            printf("Konvergenz erreicht nach %d Iterationen.\n", iter + 1);
            retval = true;
            break;
        }
    }

    return retval;
}

int main(void)
{
    int retval = 1;
    struct iwt_config cfg = {0};
    struct iwt_runtime rt = {0};

	// 1. Fundamentale Parameter setzen
	cfg.N = 4096;
	cfg.BATCH_SIZE = 512;
	cfg.D = iwt_fractal_dimension();
	cfg.l0 = 1.0;
	cfg.T = 1.0;
	cfg.alpha_IWT = 1.0;
	cfg.beta_IWT = 1.0;
	cfg.delta_t = 0.01;
	cfg.MAX_ITER = 10;
	cfg.I_min = iwt_I_min();
	cfg.I_max = iwt_I_max();
	cfg.Delta_I = iwt_delta_I();

	// 2. Abgeleitete Parameter berechnen
	cfg.THRESHOLD = iwt_I_min();
	cfg.DT = 0.001;
	cfg.ETA = 100.0;
	cfg.LAMBDA = 0.0;
	cfg.MU = 0.1;
	cfg.ETA_Q = 0.1;
	cfg.LAMBDA_Q = 0.1;
	cfg.GAMMA_Q = 0.1;

	printf("=== IWT Parameter (aus Theorie) ===\n");
	printf("D               = %.12f\n", cfg.D);
	printf("l0              = %.12e m\n", cfg.l0);
	printf("T               = %.12e s\n", cfg.T);
	printf("alpha_IWT       = %.12e\n", cfg.alpha_IWT);
	printf("beta_IWT        = %.12e\n", cfg.beta_IWT);
	printf("delta_t         = %.12f\n", cfg.delta_t);
	printf("I_min           = %.12f\n", iwt_I_min());
	printf("I_max           = %.12f\n", iwt_I_max());
	printf("THRESHOLD       = %.12f\n", cfg.THRESHOLD);
	printf("\n=== Abgeleitete Simulationsparameter ===\n");
	printf("DT              = %.12e\n", cfg.DT);
	printf("ETA             = %.12e\n", cfg.ETA);
	printf("LAMBDA          = %.12e\n", cfg.LAMBDA);
	printf("ETA_Q           = %.12e\n", cfg.ETA_Q);
	printf("LAMBDA_Q        = %.12e\n", cfg.LAMBDA_Q);
	printf("GAMMA_Q         = %.12e\n", cfg.GAMMA_Q);
	printf("========================================\n\n");

    // 3. OpenCL initialisieren
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