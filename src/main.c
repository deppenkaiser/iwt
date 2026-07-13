#include <stdio.h>
#include <ocl/ocl.h>
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
        double I_min = 0.1;
        for (size_t i = 0; i < cfg->N; i++)
        {
            rt->I[i] = I_min + 0.01 * (double)(i % 100);
        }

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
        double GAMMA = cfg->GAMMA;

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
            clSetKernelArg(kernel, 2, sizeof(cl_mem), &rt->Q_gpu);
            clSetKernelArg(kernel, 3, sizeof(double), &DT);
            clSetKernelArg(kernel, 4, sizeof(double), &ETA);
            clSetKernelArg(kernel, 5, sizeof(double), &LAMBDA);
            clSetKernelArg(kernel, 6, sizeof(double), &GAMMA);
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

bool run_simulation(const iwt_runtime_t rt, const iwt_config_t cfg)
{
    bool retval = false;

    for (int iter = 0; iter < cfg->MAX_ITER; iter++)
    {
        if (!run_flux_calculation_batched(rt, cfg)) break;
        if (!run_q_calculation(rt, cfg)) break;
        if (!run_update_info(rt, cfg)) break;
        if (!run_update_coupling(rt, cfg)) break;

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

    // 1. Fundamentale Parameter setzen
    cfg.N = 4096;
    cfg.BATCH_SIZE = 512;
    cfg.D = iwt_fractal_dimension();
    cfg.l0 = iwt_fundamental_length();
    cfg.T = iwt_fundamental_time();
    cfg.alpha_IWT = iwt_alpha_IWT();
    cfg.beta_IWT = iwt_beta_IWT();
    cfg.delta_t = 0.01;
    cfg.MAX_ITER = 10;

    // 2. Abgeleitete Parameter berechnen
    cfg.DT = cfg.T;
    cfg.ETA = cfg.alpha_IWT * cfg.DT;
    cfg.LAMBDA = cfg.beta_IWT * cfg.l0 / (cfg.T * cfg.T) * cfg.DT * cfg.DT;
    cfg.GAMMA = 0.0;//(cfg.alpha_IWT / cfg.beta_IWT) * cfg.DT;
    cfg.THRESHOLD = iwt_delta_I_min();

	printf("=== IWT Parameter (aus Theorie) ===\n");
	printf("D               = %.12f\n", cfg.D);
	printf("l0              = %.12e m\n", cfg.l0);
	printf("T               = %.12e s\n", cfg.T);
	printf("alpha_IWT       = %.12e\n", cfg.alpha_IWT);
	printf("beta_IWT        = %.12e\n", cfg.beta_IWT);
	printf("delta_t         = %.12f\n", cfg.delta_t);
	printf("Delta_I_min     = %.12f\n", iwt_delta_I_min());
	printf("THRESHOLD       = %.12f\n", cfg.THRESHOLD);
	printf("\n=== Abgeleitete Simulationsparameter ===\n");
	printf("DT              = %.12e\n", cfg.DT);
	printf("ETA             = %.12e\n", cfg.ETA);
	printf("LAMBDA          = %.12e\n", cfg.LAMBDA);
	printf("GAMMA           = %.12e\n", cfg.GAMMA);
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