#include <ocl/ocl.h>
#include <string/string.h>
#include <stdio.h>
#include <math.h>
#include <api/api.h>
#include "iwt.h"

// === NEUE HILFSFUNKTION: Box-Muller für Gauß'sche Zufallszahlen ===
static double box_muller(unsigned int* seed)
{
    // Box-Muller-Transformation: Erzeugt zwei unabhängige standard-normalverteilte Zufallszahlen
    // Verwendet den übergebenen Seed für reproduzierbare Ergebnisse
    double u1, u2;
    do {
        u1 = (double)rand_r(seed) / (double)RAND_MAX;
        u2 = (double)rand_r(seed) / (double)RAND_MAX;
    } while (u1 < 1e-30 || u2 < 1e-30); // Vermeidung von log(0)

    double z = sqrt(-2.0 * log(u1)) * cos(2.0 * iwt_pi() * u2);
    return z;
}

// === NEUE FUNKTION: Generierung der Fluktuationen auf der CPU ===
private bool generate_uncertainty_cpu(const iwt_runtime_t rt, const iwt_config_t cfg)
{
    if (!cfg->enable_fluctuations)
    {
        // Wenn Fluktuationen deaktiviert sind, setze alles auf 0
        for (size_t i = 0; i < cfg->N; i++)
        {
            rt->xi_real[i] = 0.0;
            rt->xi_imag[i] = 0.0;
            rt->uncertainty[i] = 0.0;
        }
        return true;
    }

    double scale = cfg->uncertainty_scale;

    // Seed für reproduzierbare Ergebnisse initialisieren
    unsigned int seed = cfg->seed;

    for (size_t i = 0; i < cfg->N; i++)
    {
        // Zwei unabhängige Gauß'sche Zufallszahlen für Real- und Imaginärteil
        double z_real = box_muller(&seed);
        double z_imag = box_muller(&seed);

        // Skalierung mit √(ℏ/(2T))
        rt->xi_real[i] = scale * z_real;
        rt->xi_imag[i] = scale * z_imag;

        // Kombinierter Unschärfe-Term (komplex als zwei double)
        // uncertainty[i] wird später im Update-Kernel verwendet
        rt->uncertainty[i] = 0.0; // Wird auf GPU berechnet
    }

    return true;
}

// === BESTEHENDE FUNKTIONEN ===
private bool run_flux_calculation(const iwt_runtime_t rt, const iwt_config_t cfg)
{
    bool retval = false;
    cl_kernel kernel = ocl_get_kernel(&rt->ocl, OCL_KERNEL_IWT_FLUX);
    
    if (kernel != NULL)
    {
        clEnqueueWriteBuffer(rt->ocl.queue, rt->I_gpu, CL_TRUE, 0,
            cfg->N * sizeof(double), rt->I, 0, NULL, NULL);
        clEnqueueWriteBuffer(rt->ocl.queue, rt->K_gpu, CL_TRUE, 0,
            cfg->N * cfg->N * sizeof(double), rt->K, 0, NULL, NULL);

        int N = (int)cfg->N;
        int start = 0;
        int end = N;
        double DT = cfg->DT;
        double I_max = cfg->I_max;
        double Z_0 = cfg->Z_0;
        double alpha = cfg->alpha_Z;

        clSetKernelArg(kernel, 0, sizeof(cl_mem), &rt->I_gpu);
        clSetKernelArg(kernel, 1, sizeof(cl_mem), &rt->K_gpu);
        clSetKernelArg(kernel, 2, sizeof(cl_mem), &rt->sumJ_gpu);
        clSetKernelArg(kernel, 3, sizeof(cl_mem), &rt->R_gpu);
        clSetKernelArg(kernel, 4, sizeof(cl_mem), &rt->T_gpu);
        clSetKernelArg(kernel, 5, sizeof(int), &N);
        clSetKernelArg(kernel, 6, sizeof(int), &start);
        clSetKernelArg(kernel, 7, sizeof(int), &end);
        clSetKernelArg(kernel, 8, sizeof(double), &DT);
        clSetKernelArg(kernel, 9, sizeof(double), &I_max);
        clSetKernelArg(kernel, 10, sizeof(double), &Z_0);
        clSetKernelArg(kernel, 11, sizeof(double), &alpha);

        size_t global = cfg->N;
        size_t local = 64;
        if (local > global) local = global;

        if (ocl_enqueue_kernel(&rt->ocl, kernel, global, local))
        {
            clEnqueueReadBuffer(rt->ocl.queue, rt->sumJ_gpu, CL_TRUE,
                0, cfg->N * sizeof(double), rt->sumJ, 0, NULL, NULL);
            clEnqueueReadBuffer(rt->ocl.queue, rt->R_gpu, CL_TRUE,
                0, cfg->N * sizeof(double), rt->R, 0, NULL, NULL);
            clEnqueueReadBuffer(rt->ocl.queue, rt->T_gpu, CL_TRUE,
                0, cfg->N * sizeof(double), rt->T, 0, NULL, NULL);
            retval = true;
        }
    }
    
    return retval;
}

private bool run_q_calculation(const iwt_runtime_t rt, const iwt_config_t cfg)
{
    bool retval = false;
    cl_kernel kernel = ocl_get_kernel(&rt->ocl, OCL_KERNEL_IWT_Q);
    
    if (kernel != NULL)
    {
        double sum_abs_sq = 0.0;
        for (size_t i = 0; i < cfg->N; i++)
        {
            sum_abs_sq += rt->I[i] * rt->I[i];
        }

        // Wenn keine Information vorhanden ist, setze Q auf 0 und kehre mit Erfolg zurück
        if (sum_abs_sq < 1e-30)
        {
            for (size_t i = 0; i < cfg->N; i++) rt->Q[i] = 0.0;
            return true;
        }

        int N = (int)cfg->N;
        double hbar = 1.0;
        double m = 1.0;
        double prefactor = -(hbar * hbar) / (2.0 * m);

        clSetKernelArg(kernel, 0, sizeof(cl_mem), &rt->I_gpu);
        clSetKernelArg(kernel, 1, sizeof(cl_mem), &rt->Q_gpu);
        clSetKernelArg(kernel, 2, sizeof(int), &N);
        clSetKernelArg(kernel, 3, sizeof(double), &sum_abs_sq);
        clSetKernelArg(kernel, 4, sizeof(double), &prefactor);

        size_t global = cfg->N;
        size_t local = 64;
        if (local > global) local = global;

        if (ocl_enqueue_kernel(&rt->ocl, kernel, global, local))
        {
            clEnqueueReadBuffer(rt->ocl.queue, rt->Q_gpu, CL_TRUE,
                0, cfg->N * sizeof(double), rt->Q, 0, NULL, NULL);

            for (size_t i = 0; i < cfg->N; i++)
            {
                if (isnan(rt->Q[i]) || isinf(rt->Q[i]))
                {
                    rt->Q[i] = 0.0;
                }
            }

            retval = true;
        }
    }

    return retval;
}

// === NEUE FUNKTION: Kopieren der Fluktuationen auf die GPU ===
private bool upload_uncertainty_to_gpu(const iwt_runtime_t rt, const iwt_config_t cfg)
{
    bool retval = false;

    // Zufallszahlen auf GPU schreiben
    if (clEnqueueWriteBuffer(rt->ocl.queue, rt->xi_real_gpu, CL_TRUE, 0,
        cfg->N * sizeof(double), rt->xi_real, 0, NULL, NULL) == CL_SUCCESS)
    {
        if (clEnqueueWriteBuffer(rt->ocl.queue, rt->xi_imag_gpu, CL_TRUE, 0,
            cfg->N * sizeof(double), rt->xi_imag, 0, NULL, NULL) == CL_SUCCESS)
        {
            retval = true;
        }
    }

    return retval;
}

// === BESTEHENDE FUNKTION: run_update_info (wird später erweitert) ===
private bool run_update_info(const iwt_runtime_t rt, const iwt_config_t cfg)
{
    bool retval = false;
    cl_kernel kernel = ocl_get_kernel(&rt->ocl, OCL_KERNEL_IWT_UPDATE_INFO);
    
    if (kernel != NULL)
    {
        // I_avg berechnen (einmal pro Iteration)
        double I_avg = 0.0;
        for (size_t i = 0; i < cfg->N; i++)
        {
            I_avg += rt->I[i];
        }
        I_avg /= cfg->N;

        int N = (int)cfg->N;
        int width = (int)sqrt(cfg->N);
        double DT = cfg->DT;
        double I_vac = cfg->I_vac;
        double phi_0 = cfg->phi_0;
        double omega_0 = cfg->omega_0;
        double D = cfg->D;
        double Z_0 = cfg->Z_0;

        clEnqueueWriteBuffer(rt->ocl.queue, rt->I_prev_gpu, CL_TRUE, 0,
            cfg->N * sizeof(double), rt->I, 0, NULL, NULL);
        clEnqueueWriteBuffer(rt->ocl.queue, rt->I_phase_prev_gpu, CL_TRUE, 0,
            cfg->N * sizeof(double), rt->I_phase, 0, NULL, NULL);

        // === NEU: Zufallszahlen auf die GPU kopieren ===
        if (!upload_uncertainty_to_gpu(rt, cfg))
        {
            return false;
        }

        clSetKernelArg(kernel, 0, sizeof(cl_mem), &rt->I_gpu);
        clSetKernelArg(kernel, 1, sizeof(cl_mem), &rt->I_phase_gpu);
        clSetKernelArg(kernel, 2, sizeof(cl_mem), &rt->I_prev_gpu);
        clSetKernelArg(kernel, 3, sizeof(cl_mem), &rt->I_phase_prev_gpu);
        clSetKernelArg(kernel, 4, sizeof(cl_mem), &rt->sumJ_gpu);
        clSetKernelArg(kernel, 5, sizeof(cl_mem), &rt->Q_gpu);
        // === NEU: GPU-Buffer für Fluktuationen als Argumente ===
        clSetKernelArg(kernel, 6, sizeof(cl_mem), &rt->xi_real_gpu);
        clSetKernelArg(kernel, 7, sizeof(cl_mem), &rt->xi_imag_gpu);
        clSetKernelArg(kernel, 8, sizeof(int), &N);
        clSetKernelArg(kernel, 9, sizeof(int), &width);
        clSetKernelArg(kernel, 10, sizeof(double), &DT);
        clSetKernelArg(kernel, 11, sizeof(double), &I_vac);
        clSetKernelArg(kernel, 12, sizeof(double), &phi_0);
        clSetKernelArg(kernel, 13, sizeof(double), &omega_0);
        clSetKernelArg(kernel, 14, sizeof(double), &D);
        clSetKernelArg(kernel, 15, sizeof(double), &Z_0);
        clSetKernelArg(kernel, 16, sizeof(double), &I_avg);
        // === NEU: Schalter für Fluktuationen ===
        clSetKernelArg(kernel, 17, sizeof(int), &cfg->enable_fluctuations);

        size_t global = cfg->N;
        size_t local = 64;
        if (local > global) local = global;

        if (ocl_enqueue_kernel(&rt->ocl, kernel, global, local))
        {
            clEnqueueReadBuffer(rt->ocl.queue, rt->I_gpu, CL_TRUE,
                0, cfg->N * sizeof(double), rt->I, 0, NULL, NULL);
            clEnqueueReadBuffer(rt->ocl.queue, rt->I_phase_gpu, CL_TRUE,
                0, cfg->N * sizeof(double), rt->I_phase, 0, NULL, NULL);

            for (size_t i = 0; i < cfg->N; i++)
            {
                rt->I_prev[i] = rt->I[i];
                rt->I_phase_prev[i] = rt->I_phase[i];
            }

            retval = true;
        }
    }
    
    return retval;
}

bool run_simulation(const iwt_runtime_t rt, const iwt_config_t cfg)
{
    bool retval = false;

    // Initialisierung auf reines Vakuum
    for (size_t i = 0; i < cfg->N; i++)
    {
        rt->I[i] = 0.0;
        rt->I_phase[i] = 0.0;
        rt->I_prev[i] = 0.0;
        rt->I_phase_prev[i] = 0.0;
    }

    double sum_I_sq_initial = 0.0;
    for (size_t i = 0; i < cfg->N; i++)
    {
        sum_I_sq_initial += rt->I[i] * rt->I[i];
    }

    // Cursor einmal auf (1,1) setzen (nach der Initialisierungsausgabe)
    string_set_cursor_position(1, 1);

    for (int iter = 0; iter < cfg->MAX_ITER; iter++)
    {
        for (size_t i = 0; i < cfg->N; i++)
        {
            rt->I_prev[i] = rt->I[i];
            rt->I_phase_prev[i] = rt->I_phase[i];
        }

        clEnqueueWriteBuffer(rt->ocl.queue, rt->I_gpu, CL_TRUE, 0,
            cfg->N * sizeof(double), rt->I, 0, NULL, NULL);
        clEnqueueWriteBuffer(rt->ocl.queue, rt->I_phase_gpu, CL_TRUE, 0,
            cfg->N * sizeof(double), rt->I_phase, 0, NULL, NULL);

        if (!run_flux_calculation(rt, cfg)) break;
        if (!run_q_calculation(rt, cfg)) break;
        if (!generate_uncertainty_cpu(rt, cfg)) break;
        if (!run_update_info(rt, cfg)) break;

        double max_q = 0.0;
        double sum_abs = 0.0;
        double sum_I_sq = 0.0;
        double I_min = 1e30;
        double I_max = -1e30;

        for (size_t i = 0; i < cfg->N; i++)
        {
            double val = rt->I[i];
            double abs_val = (val < 0.0) ? -val : val;

            sum_abs += abs_val;
            sum_I_sq += val * val;

            if (abs_val < I_min) I_min = abs_val;
            if (abs_val > I_max) I_max = abs_val;

            double abs_q = fabs(rt->Q[i]);
            if (abs_q > max_q) max_q = abs_q;
        }

        double I_total = sum_abs;

        static double sum_I_sq_ref = -1.0;
        if (sum_I_sq_ref < 0.0) sum_I_sq_ref = sum_I_sq_initial;
        double info_deviation = (sum_I_sq - sum_I_sq_ref) / (sum_I_sq_ref + 1e-30);

        static double I_total_ref = -1.0;
        if (I_total_ref < 0.0) I_total_ref = I_total;
        double deviation = (I_total - I_total_ref) / (I_total_ref + 1e-30);

        // Cursor auf (1,1) zurücksetzen (ohne Clear)
        string_set_cursor_position(1, 1);

        iwt_print_status(rt, cfg, iter, max_q, I_total, I_min, I_max,
                         deviation, sum_I_sq, info_deviation);

        retval = true;
    }

    return retval;
}
