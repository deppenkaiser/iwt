#include "iwt_kernel.h"
#include <ocl/ocl.h>
#include <stdio.h>

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
