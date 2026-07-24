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

private bool generate_uncertainty_cpu(const iwt_runtime_t rt, const iwt_config_t cfg)
{
    if (!cfg->enable_fluctuations)
    {
        for (size_t i = 0; i < cfg->N; i++)
        {
            rt->xi_real[i] = 0.0;
            rt->xi_imag[i] = 0.0;
            rt->uncertainty[i] = 0.0;
        }
        return true;
    }

    double scale = cfg->uncertainty_scale;
    unsigned int seed = cfg->seed;

    // 1. Rohdaten generieren
    double* raw_real = malloc(cfg->N * sizeof(double));
    double* raw_imag = malloc(cfg->N * sizeof(double));
    if (!raw_real || !raw_imag) return false;

    for (size_t i = 0; i < cfg->N; i++)
    {
        raw_real[i] = box_muller(&seed);
        raw_imag[i] = box_muller(&seed);
    }

    // 2. Mittelwert berechnen
    double mean_real = 0.0, mean_imag = 0.0;
    for (size_t i = 0; i < cfg->N; i++)
    {
        mean_real += raw_real[i];
        mean_imag += raw_imag[i];
    }
    mean_real /= cfg->N;
    mean_imag /= cfg->N;

    // 3. Mittelwert korrigieren und skalieren
    for (size_t i = 0; i < cfg->N; i++)
    {
        rt->xi_real[i] = scale * (raw_real[i] - mean_real);
        rt->xi_imag[i] = scale * (raw_imag[i] - mean_imag);
        rt->uncertainty[i] = 0.0;
    }

    free(raw_real);
    free(raw_imag);
    return true;
}

private bool run_flux_calculation(const iwt_runtime_t rt, const iwt_config_t cfg)
{
    bool retval = false;
    cl_kernel kernel = ocl_get_kernel(&rt->ocl, OCL_KERNEL_IWT_FLUX);
    
    if (kernel != NULL)
    {
        clEnqueueWriteBuffer(rt->ocl.queue, rt->I_real_gpu, CL_TRUE, 0,
            cfg->N * sizeof(double), rt->I_real, 0, NULL, NULL);
        clEnqueueWriteBuffer(rt->ocl.queue, rt->I_imag_gpu, CL_TRUE, 0,
            cfg->N * sizeof(double), rt->I_imag, 0, NULL, NULL);
        clEnqueueWriteBuffer(rt->ocl.queue, rt->K_gpu, CL_TRUE, 0,
            cfg->N * cfg->N * sizeof(double), rt->K, 0, NULL, NULL);

        int N = (int)cfg->N;
        double DT = cfg->DT;

        clSetKernelArg(kernel, 0, sizeof(cl_mem), &rt->I_real_gpu);
        clSetKernelArg(kernel, 1, sizeof(cl_mem), &rt->I_imag_gpu);
        clSetKernelArg(kernel, 2, sizeof(cl_mem), &rt->K_gpu);
        clSetKernelArg(kernel, 3, sizeof(cl_mem), &rt->sumJ_gpu);
        clSetKernelArg(kernel, 4, sizeof(int), &N);
        clSetKernelArg(kernel, 5, sizeof(double), &DT);

        size_t global = cfg->N;
        size_t local = 64;
        if (local > global) local = global;

        if (ocl_enqueue_kernel(&rt->ocl, kernel, global, local))
        {
            clEnqueueReadBuffer(rt->ocl.queue, rt->sumJ_gpu, CL_TRUE,
                0, cfg->N * sizeof(double), rt->sumJ, 0, NULL, NULL);
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
            sum_abs_sq += rt->I_real[i] * rt->I_real[i] + rt->I_imag[i] * rt->I_imag[i];
        }

        if (sum_abs_sq < 1e-30)
        {
            for (size_t i = 0; i < cfg->N; i++) rt->Q[i] = 0.0;
            return true;
        }

        int N = (int)cfg->N;
        double hbar = 1.0;
        double m = 1.0;
        double prefactor = -(hbar * hbar) / (2.0 * m);

        clSetKernelArg(kernel, 0, sizeof(cl_mem), &rt->I_real_gpu);
        clSetKernelArg(kernel, 1, sizeof(cl_mem), &rt->I_imag_gpu);
        clSetKernelArg(kernel, 2, sizeof(cl_mem), &rt->Q_gpu);
        clSetKernelArg(kernel, 3, sizeof(int), &N);
        clSetKernelArg(kernel, 4, sizeof(double), &sum_abs_sq);
        clSetKernelArg(kernel, 5, sizeof(double), &prefactor);

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

private bool run_update_info(const iwt_runtime_t rt, const iwt_config_t cfg)
{
    bool retval = false;
    cl_kernel kernel = ocl_get_kernel(&rt->ocl, OCL_KERNEL_IWT_UPDATE_INFO);
    
    if (kernel != NULL)
    {
        int N = (int)cfg->N;
        double DT = cfg->DT;

        clSetKernelArg(kernel, 0, sizeof(cl_mem), &rt->I_real_gpu);
        clSetKernelArg(kernel, 1, sizeof(cl_mem), &rt->I_imag_gpu);
        clSetKernelArg(kernel, 2, sizeof(cl_mem), &rt->I_phase_gpu);
        clSetKernelArg(kernel, 3, sizeof(cl_mem), &rt->sumJ_gpu);
        clSetKernelArg(kernel, 4, sizeof(cl_mem), &rt->Q_gpu);
        clSetKernelArg(kernel, 5, sizeof(int), &N);
        clSetKernelArg(kernel, 6, sizeof(double), &DT);

        size_t global = cfg->N;
        size_t local = 64;
        if (local > global) local = global;

        if (ocl_enqueue_kernel(&rt->ocl, kernel, global, local))
        {
            clEnqueueReadBuffer(rt->ocl.queue, rt->I_real_gpu, CL_TRUE,
                0, cfg->N * sizeof(double), rt->I_real, 0, NULL, NULL);
            clEnqueueReadBuffer(rt->ocl.queue, rt->I_imag_gpu, CL_TRUE,
                0, cfg->N * sizeof(double), rt->I_imag, 0, NULL, NULL);
            clEnqueueReadBuffer(rt->ocl.queue, rt->I_phase_gpu, CL_TRUE,
                0, cfg->N * sizeof(double), rt->I_phase, 0, NULL, NULL);

            retval = true;
        }
    }
    
    return retval;
}

private bool run_apply_fluctuations(const iwt_runtime_t rt, const iwt_config_t cfg)
{
    bool retval = false;
    cl_kernel kernel = ocl_get_kernel(&rt->ocl, OCL_KERNEL_IWT_APPLY_FLUCTUATIONS);
    
    if (kernel != NULL)
    {
        // Daten auf GPU schreiben (falls noch nicht geschehen)
        clEnqueueWriteBuffer(rt->ocl.queue, rt->I_real_gpu, CL_TRUE, 0,
            cfg->N * sizeof(double), rt->I_real, 0, NULL, NULL);
        clEnqueueWriteBuffer(rt->ocl.queue, rt->I_imag_gpu, CL_TRUE, 0,
            cfg->N * sizeof(double), rt->I_imag, 0, NULL, NULL);
        clEnqueueWriteBuffer(rt->ocl.queue, rt->I_phase_gpu, CL_TRUE, 0,
            cfg->N * sizeof(double), rt->I_phase, 0, NULL, NULL);

        if (!upload_uncertainty_to_gpu(rt, cfg))
        {
            return false;
        }

        int N = (int)cfg->N;
        int enable_fluctuations = cfg->enable_fluctuations ? 1 : 0;

        clSetKernelArg(kernel, 0, sizeof(cl_mem), &rt->I_real_gpu);
        clSetKernelArg(kernel, 1, sizeof(cl_mem), &rt->I_imag_gpu);
        clSetKernelArg(kernel, 2, sizeof(cl_mem), &rt->I_phase_gpu);
        clSetKernelArg(kernel, 3, sizeof(cl_mem), &rt->xi_real_gpu);
        clSetKernelArg(kernel, 4, sizeof(cl_mem), &rt->xi_imag_gpu);
        clSetKernelArg(kernel, 5, sizeof(int), &N);
        clSetKernelArg(kernel, 6, sizeof(int), &enable_fluctuations);

        size_t global = cfg->N;
        size_t local = 64;
        if (local > global) local = global;

        if (ocl_enqueue_kernel(&rt->ocl, kernel, global, local))
        {
            clEnqueueReadBuffer(rt->ocl.queue, rt->I_real_gpu, CL_TRUE,
                0, cfg->N * sizeof(double), rt->I_real, 0, NULL, NULL);
            clEnqueueReadBuffer(rt->ocl.queue, rt->I_imag_gpu, CL_TRUE,
                0, cfg->N * sizeof(double), rt->I_imag, 0, NULL, NULL);
            clEnqueueReadBuffer(rt->ocl.queue, rt->I_phase_gpu, CL_TRUE,
                0, cfg->N * sizeof(double), rt->I_phase, 0, NULL, NULL);

            retval = true;
        }
    }
    
    return retval;
}

private bool run_compute_mass_charge(const iwt_runtime_t rt, const iwt_config_t cfg)
{
    bool retval = false;
    cl_kernel kernel = ocl_get_kernel(&rt->ocl, OCL_KERNEL_IWT_MASS_CHARGE);
    
    if (kernel != NULL)
    {
        // Daten auf GPU schreiben
        clEnqueueWriteBuffer(rt->ocl.queue, rt->I_real_gpu, CL_TRUE, 0,
            cfg->N * sizeof(double), rt->I_real, 0, NULL, NULL);
        clEnqueueWriteBuffer(rt->ocl.queue, rt->I_imag_gpu, CL_TRUE, 0,
            cfg->N * sizeof(double), rt->I_imag, 0, NULL, NULL);
        clEnqueueWriteBuffer(rt->ocl.queue, rt->I_phase_gpu, CL_TRUE, 0,
            cfg->N * sizeof(double), rt->I_phase, 0, NULL, NULL);

        // Neue GPU-Buffer für Masse und Ladung (müssen in iwt.h hinzugefügt werden)
        // rt->mass_gpu und rt->charge_gpu werden benötigt

        int N = (int)cfg->N;
        double delta = 1.0; // Kopplungskonstante für Masse

        clSetKernelArg(kernel, 0, sizeof(cl_mem), &rt->I_real_gpu);
        clSetKernelArg(kernel, 1, sizeof(cl_mem), &rt->I_imag_gpu);
        clSetKernelArg(kernel, 2, sizeof(cl_mem), &rt->I_phase_gpu);
        clSetKernelArg(kernel, 3, sizeof(cl_mem), &rt->mass_gpu);
        clSetKernelArg(kernel, 4, sizeof(cl_mem), &rt->charge_gpu);
        clSetKernelArg(kernel, 5, sizeof(int), &N);
        clSetKernelArg(kernel, 6, sizeof(double), &delta);

        size_t global = cfg->N;
        size_t local = 64;
        if (local > global) local = global;

        if (ocl_enqueue_kernel(&rt->ocl, kernel, global, local))
        {
            clEnqueueReadBuffer(rt->ocl.queue, rt->mass_gpu, CL_TRUE,
                0, cfg->N * sizeof(double), rt->mass, 0, NULL, NULL);
            clEnqueueReadBuffer(rt->ocl.queue, rt->charge_gpu, CL_TRUE,
                0, cfg->N * sizeof(double), rt->charge, 0, NULL, NULL);
            retval = true;
        }
    }
    
    return retval;
}

bool run_simulation(const iwt_runtime_t rt, const iwt_config_t cfg)
{
    bool retval = false;

    // Initialisierung auf Vakuum: I_real = 0.01, I_imag = 0.0
    for (size_t i = 0; i < cfg->N; i++)
    {
        rt->I_real[i] = 0.01;
        rt->I_imag[i] = 0.0;
        rt->I_phase[i] = 0.0;
        rt->I_prev_real[i] = 0.01;
        rt->I_prev_imag[i] = 0.0;
        rt->I_phase_prev[i] = 0.0;
    }

    double sum_abs_sq_initial = 0.0;
    for (size_t i = 0; i < cfg->N; i++)
    {
        sum_abs_sq_initial += rt->I_real[i] * rt->I_real[i] + rt->I_imag[i] * rt->I_imag[i];
    }

    // CSV-Datei für Zeitreihen öffnen
    FILE* csv_file = fopen("iwt_timeseries.csv", "w");
    if (csv_file != NULL)
    {
        fprintf(csv_file, "iter,sum_I_sq,I_max\n");
    }
    else
    {
        printf("Warning: Could not open iwt_timeseries.csv for writing\n");
    }

    string_clear_screen();

    for (int iter = 0; iter < cfg->MAX_ITER; iter++)
    {
        for (size_t i = 0; i < cfg->N; i++)
        {
            rt->I_prev_real[i] = rt->I_real[i];
            rt->I_prev_imag[i] = rt->I_imag[i];
            rt->I_phase_prev[i] = rt->I_phase[i];
        }

        // ============================================================
        // 1. FLUKTUATIONEN GENERIEREN (CPU)
        // ============================================================
        if (!generate_uncertainty_cpu(rt, cfg)) break;

        // ============================================================
        // 2. FLUKTUATIONEN AUF I ANWENDEN (GPU)
        // ============================================================
        if (!run_apply_fluctuations(rt, cfg)) break;

        // ============================================================
        // 3. FLUSS AUS DEM NEUEN I BERECHNEN
        // ============================================================
        if (!run_flux_calculation(rt, cfg)) break;

        // ============================================================
        // 4. Q AUS DEM NEUEN I BERECHNEN
        // ============================================================
        if (!run_q_calculation(rt, cfg)) break;

        // ============================================================
        // 5. KONTINUITÄT (rho = |I|²) ANWENDEN
        // ============================================================
        if (!run_update_info(rt, cfg)) break;

        // ============================================================
        // 6. MASSE UND LADUNG BERECHNEN
        // ============================================================
        if (!run_compute_mass_charge(rt, cfg)) break;

        // ============================================================
        // STATISTIKEN
        // ============================================================
        double max_q = 0.0;
        double sum_abs_sq = 0.0;
        double sum_abs = 0.0;
        double I_min = 1e30;
        double I_max = -1e30;

        for (size_t i = 0; i < cfg->N; i++)
        {
            double abs_i = sqrt(rt->I_real[i] * rt->I_real[i] + rt->I_imag[i] * rt->I_imag[i] + 1e-30);
            double rho_i = abs_i * abs_i;

            sum_abs += abs_i;
            sum_abs_sq += rho_i;

            if (abs_i < I_min) I_min = abs_i;
            if (abs_i > I_max) I_max = abs_i;

            double abs_q = fabs(rt->Q[i]);
            if (abs_q > max_q) max_q = abs_q;
        }

        // CSV schreiben
        if (csv_file != NULL)
        {
            fprintf(csv_file, "%d,%.6f,%.6f\n", iter, sum_abs_sq, I_max);
        }

        static double sum_abs_sq_ref = -1.0;
        if (sum_abs_sq_ref < 0.0) sum_abs_sq_ref = sum_abs_sq_initial;
        double info_deviation = (sum_abs_sq - sum_abs_sq_ref) / (sum_abs_sq_ref + 1e-30);

        static double I_total_ref = -1.0;
        if (I_total_ref < 0.0) I_total_ref = sum_abs;
        double deviation = (sum_abs - I_total_ref) / (I_total_ref + 1e-30);

        // ============================================================
        // HEATMAP SPEICHERN (alle 100 Iterationen)
        // ============================================================
        if (iter % 10 == 0)
        {
            char filename[256];

            snprintf(filename, sizeof(filename), "heatmap_mass_%06d.pgm", iter);
            iwt_save_heatmap(rt->mass, cfg->N, filename, "mass");

            snprintf(filename, sizeof(filename), "heatmap_charge_%06d.pgm", iter);
            iwt_save_heatmap(rt->charge, cfg->N, filename, "charge");
        }

        // ============================================================
        // AUSGABE
        // ============================================================
        string_set_cursor_position(1, 1);
        iwt_print_status(rt, cfg, iter, max_q, sum_abs, I_min, I_max,
                         deviation, sum_abs_sq, info_deviation);

        fflush(stdout);

        retval = true;
    }

    // CSV-Datei schließen
    if (csv_file != NULL)
    {
        fclose(csv_file);
        printf("\nZeitreihe gespeichert in: iwt_timeseries.csv\n");
    }

    printf("\n");
    return retval;
}
