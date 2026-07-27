// ============================================================================
// iwt_kernel.c - Vollständige Implementierung der IWT-Simulation
// ============================================================================
// 
// Enthält:
//   - CPU-Funktionen für Quantenfluktuationen (Anhang O, P)
//   - GPU-Kernel-Aufrufe für Fluss, Q-Potential, Update
//   - Rotverschiebung als Massenänderung (Anhang Q)
//   - Hauptsimulationsschleife
//
// ============================================================================

#include <ocl/ocl.h>
#include <string/string.h>
#include <stdio.h>
#include <math.h>
#include <api/api.h>
#include "iwt.h"

// ============================================================================
// HILFSFUNKTIONEN FÜR DIE MASSE (aus Kapitel 3)
// ============================================================================

/**
 * Berechnet die Masse eines Knotens gemäß Kapitel 3, Gleichung (3.8):
 * 
 *   m_k = delta * sum_{l in N(k)} |I_k - I_l|^2
 * 
 * Die Masse ist ein Maß für die Informationsänderung zwischen benachbarten Knoten.
 */
static double compute_mass(const iwt_runtime_t rt, size_t k, size_t N)
{
    double mass = 0.0;
    double delta = 1.0;  // Kopplungskonstante für Masse
    
    // Nachbarn: k-1 und k+1 (1D-Netzwerk)
    if (k > 0) {
        double diff_re = rt->I_real[k] - rt->I_real[k-1];
        double diff_im = rt->I_imag[k] - rt->I_imag[k-1];
        mass += diff_re * diff_re + diff_im * diff_im;
    }
    if (k < N - 1) {
        double diff_re = rt->I_real[k] - rt->I_real[k+1];
        double diff_im = rt->I_imag[k] - rt->I_imag[k+1];
        mass += diff_re * diff_re + diff_im * diff_im;
    }
    
    return delta * mass;
}

/**
 * Kodiert eine Masse in eine Phase.
 * Die Phase ist ein Maß für die Informationsstruktur, die die Masse repräsentiert.
 */
static double encode_mass(double mass)
{
    double mass_0 = 1.0;  // Referenzmasse
    return atan(mass / mass_0);
}

/**
 * Prüft, ob ein Knoten ein Randknoten ist.
 * Im 1D-Netzwerk sind die Knoten 0 und N-1 die Ränder.
 */
static bool is_boundary_node(size_t k, size_t N)
{
    return (k == 0) || (k == N - 1);
}

// ============================================================================
// NEUE HILFSFUNKTION: Box-Muller für Gauß'sche Zufallszahlen
// ============================================================================

/**
 * Box-Muller-Transformation für standard-normalverteilte Zufallszahlen.
 * Wird für die intrinsischen Fluktuationen verwendet (Anhang P).
 */
static double box_muller(unsigned int* seed)
{
    double u1, u2;
    do {
        u1 = (double)rand_r(seed) / (double)RAND_MAX;
        u2 = (double)rand_r(seed) / (double)RAND_MAX;
    } while (u1 < 1e-30 || u2 < 1e-30);

    double z = sqrt(-2.0 * log(u1)) * cos(2.0 * iwt_pi() * u2);
    return z;
}

// ============================================================================
// CPU-FUNKTION: GENERIERUNG DER FLUKTUATIONEN (Anhang P)
// ============================================================================

/**
 * Generiert die intrinsischen Fluktuationen gemäß Anhang P, Gleichung (P.3):
 * 
 *   I_k^(n+1) = I_k^(n) + T * (lokaler Fluss + Bohm-Potential + nichtlinearer Term)
 *               + sqrt(hbar / (2*T)) * xi_k^(n)
 * 
 * Die Fluktuationen sind keine externe Störung, sondern eine Konsequenz der diskreten Zeit.
 */
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

    double* raw_real = malloc(cfg->N * sizeof(double));
    double* raw_imag = malloc(cfg->N * sizeof(double));
    if (!raw_real || !raw_imag) return false;

    for (size_t i = 0; i < cfg->N; i++)
    {
        raw_real[i] = box_muller(&seed);
        raw_imag[i] = box_muller(&seed);
    }

    double mean_real = 0.0, mean_imag = 0.0;
    for (size_t i = 0; i < cfg->N; i++)
    {
        mean_real += raw_real[i];
        mean_imag += raw_imag[i];
    }
    mean_real /= cfg->N;
    mean_imag /= cfg->N;

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

// ============================================================================
// GPU-FUNKTIONEN
// ============================================================================

/**
 * Lädt die Fluktuationen auf die GPU.
 */
private bool upload_uncertainty_to_gpu(const iwt_runtime_t rt, const iwt_config_t cfg)
{
    bool retval = false;
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

/**
 * Wendet die Fluktuationen auf das I-Feld an (GPU-Kernel).
 */
private bool run_apply_fluctuations(const iwt_runtime_t rt, const iwt_config_t cfg)
{
    bool retval = false;
    cl_kernel kernel = ocl_get_kernel(&rt->ocl, OCL_KERNEL_IWT_APPLY_FLUCTUATIONS);
    
    if (kernel != NULL)
    {
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

/**
 * Berechnet den Fluss (Weber-Kern) auf der GPU.
 */
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

/**
 * Berechnet das Bohm-Potential Q auf der GPU.
 */
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

/**
 * Führt das Kontinuitäts-Update durch (rho = |I|²).
 */
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

// ============================================================================
// run_apply_redshift_damping - GPU-Version (Anhang Q)
// ============================================================================

/**
 * Wendet die Rotverschiebung auf die Randknoten an (GPU).
 * 
 * Die Rotverschiebung in der IWT ist eine Massenänderung durch die fraktale Geometrie.
 * Die Amplitude |I| bleibt konstant (Informationserhaltung, Axiom 2).
 * 
 * Siehe Anhang Q, Gleichung (Q.16):
 * 
 *   |I_out|^2 = |I_in|^2
 *   m_out = m_in * (l0 / L_Q0)^(D-3)
 */
private bool run_apply_redshift_damping(const iwt_runtime_t rt, const iwt_config_t cfg)
{
    bool retval = false;
    cl_kernel kernel = ocl_get_kernel(&rt->ocl, OCL_KERNEL_IWT_REDSHIFT_DAMPING);
    
    if (kernel != NULL)
    {
        // Daten auf GPU schreiben
        clEnqueueWriteBuffer(rt->ocl.queue, rt->I_real_gpu, CL_TRUE, 0,
            cfg->N * sizeof(double), rt->I_real, 0, NULL, NULL);
        clEnqueueWriteBuffer(rt->ocl.queue, rt->I_imag_gpu, CL_TRUE, 0,
            cfg->N * sizeof(double), rt->I_imag, 0, NULL, NULL);

        // Parameter aus der Theorie (Anhang Q)
        int N = (int)cfg->N;
        double l0 = cfg->l0;
        double D = cfg->D;
        double L_Q0 = 2.0e46;  // Korrelationslänge des Q-Feldes (Anhang J)
        double delta = 1.0;    // Kopplungskonstante für Masse

        // Kernel-Argumente setzen
        clSetKernelArg(kernel, 0, sizeof(cl_mem), &rt->I_real_gpu);
        clSetKernelArg(kernel, 1, sizeof(cl_mem), &rt->I_imag_gpu);
        clSetKernelArg(kernel, 2, sizeof(int), &N);
        clSetKernelArg(kernel, 3, sizeof(double), &l0);
        clSetKernelArg(kernel, 4, sizeof(double), &D);
        clSetKernelArg(kernel, 5, sizeof(double), &L_Q0);
        clSetKernelArg(kernel, 6, sizeof(double), &delta);

        // Kernel ausführen
        size_t global = cfg->N;
        size_t local = 64;
        if (local > global) local = global;

        if (ocl_enqueue_kernel(&rt->ocl, kernel, global, local))
        {
            // Ergebnis zurücklesen
            clEnqueueReadBuffer(rt->ocl.queue, rt->I_real_gpu, CL_TRUE,
                0, cfg->N * sizeof(double), rt->I_real, 0, NULL, NULL);
            clEnqueueReadBuffer(rt->ocl.queue, rt->I_imag_gpu, CL_TRUE,
                0, cfg->N * sizeof(double), rt->I_imag, 0, NULL, NULL);
            retval = true;
        }
    }
    
    return retval;
}

// ============================================================================
// MASSE UND LADUNG BERECHNEN (GPU)
// ============================================================================

private bool run_compute_mass_charge(const iwt_runtime_t rt, const iwt_config_t cfg)
{
    bool retval = false;
    cl_kernel kernel = ocl_get_kernel(&rt->ocl, OCL_KERNEL_IWT_MASS_CHARGE);
    
    if (kernel != NULL)
    {
        clEnqueueWriteBuffer(rt->ocl.queue, rt->I_real_gpu, CL_TRUE, 0,
            cfg->N * sizeof(double), rt->I_real, 0, NULL, NULL);
        clEnqueueWriteBuffer(rt->ocl.queue, rt->I_imag_gpu, CL_TRUE, 0,
            cfg->N * sizeof(double), rt->I_imag, 0, NULL, NULL);
        clEnqueueWriteBuffer(rt->ocl.queue, rt->I_phase_gpu, CL_TRUE, 0,
            cfg->N * sizeof(double), rt->I_phase, 0, NULL, NULL);

        int N = (int)cfg->N;
        double delta = 1.0;

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

// ============================================================================
// HAUPT-SIMULATIONSSCHLEIFE
// ============================================================================

bool run_simulation(const iwt_runtime_t rt, const iwt_config_t cfg)
{
    bool retval = false;

    // Initialisierung auf Vakuum
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
        sum_abs_sq_initial += rt->I_real[i] * rt->I_real[i] + 
                              rt->I_imag[i] * rt->I_imag[i];
    }

    FILE* csv_file = fopen("iwt_timeseries.csv", "w");
    if (csv_file != NULL)
    {
        fprintf(csv_file, "iter,sum_I_sq,I_max,sum_mass\n");
    }

    string_clear_screen();

    for (int iter = 0; iter < cfg->MAX_ITER; iter++)
    {
        // Vorherige Werte speichern
        for (size_t i = 0; i < cfg->N; i++)
        {
            rt->I_prev_real[i] = rt->I_real[i];
            rt->I_prev_imag[i] = rt->I_imag[i];
            rt->I_phase_prev[i] = rt->I_phase[i];
        }

        // 1. Fluktuationen generieren (Anhang P)
        if (!generate_uncertainty_cpu(rt, cfg)) break;

        // 2. Fluktuationen anwenden (GPU)
        if (!run_apply_fluctuations(rt, cfg)) break;

        // 3. Fluss berechnen (Weber-Kern)
        if (!run_flux_calculation(rt, cfg)) break;

        // 4. Bohm-Potential Q berechnen
        if (!run_q_calculation(rt, cfg)) break;

        // 5. Kontinuitäts-Update (rho = |I|²)
        if (!run_update_info(rt, cfg)) break;

        // 6. Rotverschiebung (Massenänderung am Rand, Anhang Q)
        //    KEINE DÄMPFUNG!
        if (!run_apply_redshift_damping(rt, cfg)) break;

        // 7. Masse und Ladung berechnen
        if (!run_compute_mass_charge(rt, cfg)) break;

        // Statistiken
        double max_q = 0.0;
        double sum_abs_sq = 0.0;
        double sum_mass = 0.0;
        double I_min = 1e30;
        double I_max = -1e30;

        for (size_t i = 0; i < cfg->N; i++)
        {
            double abs_i = sqrt(rt->I_real[i] * rt->I_real[i] + 
                                rt->I_imag[i] * rt->I_imag[i] + 1e-30);
            double rho_i = abs_i * abs_i;

            sum_abs_sq += rho_i;
            sum_mass += compute_mass(rt, i, cfg->N);

            if (abs_i < I_min) I_min = abs_i;
            if (abs_i > I_max) I_max = abs_i;

            double abs_q = fabs(rt->Q[i]);
            if (abs_q > max_q) max_q = abs_q;
        }

        if (csv_file != NULL)
        {
            fprintf(csv_file, "%d,%.6f,%.6f,%.6f\n", 
                    iter, sum_abs_sq, I_max, sum_mass);
        }

        // Informationserhaltung prüfen (Axiom 2)
        static double sum_abs_sq_ref = -1.0;
        if (sum_abs_sq_ref < 0.0) sum_abs_sq_ref = sum_abs_sq_initial;
        double info_deviation = (sum_abs_sq - sum_abs_sq_ref) / (sum_abs_sq_ref + 1e-30);

        // Heatmaps speichern
        if (iter % 10 == 0)
        {
            char filename[256];
            snprintf(filename, sizeof(filename), "heatmap_mass_%06d.pgm", iter);
            iwt_save_heatmap(rt->mass, cfg->N, filename, "mass");

            snprintf(filename, sizeof(filename), "heatmap_charge_%06d.pgm", iter);
            iwt_save_heatmap(rt->charge, cfg->N, filename, "charge");
        }

        // Ausgabe
        string_set_cursor_position(1, 1);
        iwt_print_status(rt, cfg, iter, max_q, sum_abs_sq, I_min, I_max,
                         info_deviation, sum_abs_sq, info_deviation);

        fflush(stdout);
        retval = true;
    }

    if (csv_file != NULL)
    {
        fclose(csv_file);
        printf("\nZeitreihe gespeichert in: iwt_timeseries.csv\n");
    }

    printf("\n");
    return retval;
}