// ============================================================================
// iwt_kernel.c - Vollständige IWT-Simulation mit Energiekreislauf
// ============================================================================

#include <ocl/ocl.h>
#include <string/string.h>
#include <stdio.h>
#include <math.h>
#include <api/api.h>
#include "iwt.h"

// ============================================================================
// KONSTANTEN
// ============================================================================

#define ALPHA 1.0
#define BETA 1.0
#define DELTA 1.0
#define GAMMA 1.0

#define RHO_0 1e-6
#define RHO_MIN 1e-8
#define ALPHA_0 1e-6
#define ALPHA_MIN 1e-9
#define SCALE 0.70710678

// ============================================================================
// HILFSFUNKTIONEN
// ============================================================================

/**
 * Berechnet die Masse eines Knotens (Kapitel 3, Gleichung 3.8)
 */
static double compute_mass(const iwt_runtime_t rt, size_t k, size_t N)
{
    double mass = 0.0;
    
    if (k > 0)
	{
        double dr = rt->I_real[k] - rt->I_real[k-1];
        double di = rt->I_imag[k] - rt->I_imag[k-1];
        mass += dr * dr + di * di;
    }
    
	if (k < N - 1)
	{
        double dr = rt->I_real[k] - rt->I_real[k+1];
        double di = rt->I_imag[k] - rt->I_imag[k+1];
        mass += dr * dr + di * di;
    }
    
    return DELTA * mass;
}

/**
 * Berechnet die Energie eines Knotens (Kapitel 4, Gleichung 4.3)
 */
static double compute_energy(const iwt_runtime_t rt, size_t k, size_t N, double DT)
{
    // Amplitudenenergie
    double E_A = ALPHA * (rt->I_real[k] * rt->I_real[k] + rt->I_imag[k] * rt->I_imag[k]);
    
    // Phasenenergie
    double dphi = (k > 0) ? rt->I_phase[k] - rt->I_phase[k-1] : 0.0;
    double E_phi = BETA * (dphi * dphi) / (DT * DT);
    
    return E_A + E_phi;
}

/**
 * Kodiert eine Masse in eine Phase (Energiesenke, Anhang Q)
 */
static double encode_mass(double mass)
{
    double mass_0 = 1.0;
    return atan(mass / mass_0);
}

/**
 * Prüft, ob ein Knoten ein Randknoten ist
 */
static bool is_boundary_node(size_t k, size_t N)
{
    return (k == 0) || (k == N - 1);
}

// ============================================================================
// FLUKTUATIONEN (Anhang P) - ENERGIENEUTRAL
// ============================================================================

static double box_muller(unsigned int* seed)
{
    double u1, u2;
    do {
        u1 = (double)rand_r(seed) / (double)RAND_MAX;
        u2 = (double)rand_r(seed) / (double)RAND_MAX;
    } while (u1 < 1e-30 || u2 < 1e-30);
    return sqrt(-2.0 * log(u1)) * cos(2.0 * iwt_pi() * u2);
}

// ============================================================================
// FLUKTUATIONEN ALS DIFFUSION (Energieerhaltend)
// ============================================================================

bool generate_uncertainty_cpu(const iwt_runtime_t rt, const iwt_config_t cfg)
{
    if (!cfg->enable_fluctuations) {
        for (size_t i = 0; i < cfg->N; i++) {
            rt->xi_real[i] = 0.0;
            rt->xi_imag[i] = 0.0;
            rt->uncertainty[i] = 0.0;
        }
        return true;
    }

    double scale = SCALE;
    unsigned int seed = cfg->seed;

    for (size_t i = 0; i < cfg->N; i++)
    {
        double rho_i = rt->I_real[i] * rt->I_real[i] + 
                       rt->I_imag[i] * rt->I_imag[i] + 1e-30;

        // Fluktuation im Vakuum, nicht in Strukturen
        double fluct_strength = scale * (1.0 - rho_i / (RHO_0 + rho_i));

        double delta = box_muller(&seed) * fluct_strength;

        // Fluktuation unabhängig von der Amplitude
        rt->xi_real[i] = delta;
        rt->xi_imag[i] = delta;
        rt->uncertainty[i] = 0.0;
    }

    return true;
}

bool upload_uncertainty_to_gpu(const iwt_runtime_t rt, const iwt_config_t cfg)
{
    if (clEnqueueWriteBuffer(rt->ocl.queue, rt->xi_real_gpu, CL_TRUE, 0,
        cfg->N * sizeof(double), rt->xi_real, 0, NULL, NULL) != CL_SUCCESS)
        return false;
    if (clEnqueueWriteBuffer(rt->ocl.queue, rt->xi_imag_gpu, CL_TRUE, 0,
        cfg->N * sizeof(double), rt->xi_imag, 0, NULL, NULL) != CL_SUCCESS)
        return false;
    return true;
}

bool run_apply_fluctuations(const iwt_runtime_t rt, const iwt_config_t cfg)
{
    cl_kernel kernel = ocl_get_kernel(&rt->ocl, OCL_KERNEL_IWT_APPLY_FLUCTUATIONS);
    if (!kernel) return false;

    clEnqueueWriteBuffer(rt->ocl.queue, rt->I_real_gpu, CL_TRUE, 0,
        cfg->N * sizeof(double), rt->I_real, 0, NULL, NULL);
    clEnqueueWriteBuffer(rt->ocl.queue, rt->I_imag_gpu, CL_TRUE, 0,
        cfg->N * sizeof(double), rt->I_imag, 0, NULL, NULL);

    if (!upload_uncertainty_to_gpu(rt, cfg)) return false;

    int N = (int)cfg->N;
    int enable = cfg->enable_fluctuations ? 1 : 0;

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &rt->I_real_gpu);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &rt->I_imag_gpu);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &rt->xi_real_gpu);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), &rt->xi_imag_gpu);
    clSetKernelArg(kernel, 4, sizeof(int), &N);
    clSetKernelArg(kernel, 5, sizeof(int), &enable);

    size_t global = cfg->N;
    size_t local = 64;
    if (local > global) local = global;

    if (!ocl_enqueue_kernel(&rt->ocl, kernel, global, local)) return false;

    clEnqueueReadBuffer(rt->ocl.queue, rt->I_real_gpu, CL_TRUE,
        0, cfg->N * sizeof(double), rt->I_real, 0, NULL, NULL);
    clEnqueueReadBuffer(rt->ocl.queue, rt->I_imag_gpu, CL_TRUE,
        0, cfg->N * sizeof(double), rt->I_imag, 0, NULL, NULL);

    return true;
}

// ============================================================================
// GPU-KERNEL-AUFRUFE
// ============================================================================

bool run_flux_calculation(const iwt_runtime_t rt, const iwt_config_t cfg)
{
    cl_kernel kernel = ocl_get_kernel(&rt->ocl, OCL_KERNEL_IWT_FLUX);
    if (!kernel) return false;

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

    if (!ocl_enqueue_kernel(&rt->ocl, kernel, global, local)) return false;

    clEnqueueReadBuffer(rt->ocl.queue, rt->sumJ_gpu, CL_TRUE,
        0, cfg->N * sizeof(double), rt->sumJ, 0, NULL, NULL);
    return true;
}

bool run_q_calculation(const iwt_runtime_t rt, const iwt_config_t cfg)
{
    cl_kernel kernel = ocl_get_kernel(&rt->ocl, OCL_KERNEL_IWT_Q);
    if (!kernel) return false;

    double sum_abs_sq = 0.0;
    for (size_t i = 0; i < cfg->N; i++) {
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

    if (!ocl_enqueue_kernel(&rt->ocl, kernel, global, local)) return false;

    clEnqueueReadBuffer(rt->ocl.queue, rt->Q_gpu, CL_TRUE,
        0, cfg->N * sizeof(double), rt->Q, 0, NULL, NULL);

    for (size_t i = 0; i < cfg->N; i++)
	{
        if (isnan(rt->Q[i]) || isinf(rt->Q[i])) rt->Q[i] = 0.0;
    }
    return true;
}

bool run_update_info(const iwt_runtime_t rt, const iwt_config_t cfg)
{
    bool retval = false;
    cl_kernel kernel = ocl_get_kernel(&rt->ocl, OCL_KERNEL_IWT_UPDATE_INFO);
    
    if (kernel != NULL)
    {
        int N = (int)cfg->N;
        double DT = cfg->DT;

        // ============================================================
        // LEAPFROG (VERLET) - SYMPLEKTISCHER INTEGRATOR
        // ============================================================
        // 
        // 1. Halber Schritt für die Phase (mit altem rho)
        // 2. Vollständiger Schritt für rho (mit phase_half)
        // 3. Halber Schritt für die Phase (mit neuem rho)
        //
        // Der Kernel muss entsprechend angepasst werden!
        // ============================================================

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

bool run_compute_mass_charge(const iwt_runtime_t rt, const iwt_config_t cfg)
{
    cl_kernel kernel = ocl_get_kernel(&rt->ocl, OCL_KERNEL_IWT_MASS_CHARGE);
    if (!kernel) return false;

    clEnqueueWriteBuffer(rt->ocl.queue, rt->I_real_gpu, CL_TRUE, 0,
        cfg->N * sizeof(double), rt->I_real, 0, NULL, NULL);
    clEnqueueWriteBuffer(rt->ocl.queue, rt->I_imag_gpu, CL_TRUE, 0,
        cfg->N * sizeof(double), rt->I_imag, 0, NULL, NULL);
    clEnqueueWriteBuffer(rt->ocl.queue, rt->I_phase_gpu, CL_TRUE, 0,
        cfg->N * sizeof(double), rt->I_phase, 0, NULL, NULL);

    int N = (int)cfg->N;
    double delta = DELTA;

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

    if (!ocl_enqueue_kernel(&rt->ocl, kernel, global, local)) return false;

    clEnqueueReadBuffer(rt->ocl.queue, rt->mass_gpu, CL_TRUE,
        0, cfg->N * sizeof(double), rt->mass, 0, NULL, NULL);
    clEnqueueReadBuffer(rt->ocl.queue, rt->charge_gpu, CL_TRUE,
        0, cfg->N * sizeof(double), rt->charge, 0, NULL, NULL);
    return true;
}

// ============================================================================
// ROTVERSCHIEBUNG ALS ENERGIESSENKE (Anhang Q)
// ============================================================================

bool run_apply_redshift_damping(const iwt_runtime_t rt, const iwt_config_t cfg)
{
    double alpha_0 = ALPHA_0;
    double alpha_min = ALPHA_MIN;

    for (size_t i = 0; i < cfg->N; i++)
    {
        double rho_i = rt->I_real[i] * rt->I_real[i] + 
                       rt->I_imag[i] * rt->I_imag[i] + 1e-30;

        double anti_rho = 1.0 / rho_i;

        double alpha = alpha_0 * (anti_rho / (1.0 + anti_rho)) + alpha_min;

        rt->I_real[i] *= (1.0 - alpha);
        rt->I_imag[i] *= (1.0 - alpha);
    }

    return true;
}

// ============================================================================
// HAUPT-SIMULATIONSSCHLEIFE
// ============================================================================

bool run_simulation(const iwt_runtime_t rt, const iwt_config_t cfg)
{
    bool retval = false;

    // ============================================================
    // INITIALISIERUNG
    // ============================================================
    
    for (size_t i = 0; i < cfg->N; i++)
	{
        rt->I_real[i] = 0.01;
        rt->I_imag[i] = 0.0;
        rt->I_phase[i] = 0.0;
        rt->I_prev_real[i] = 0.01;
        rt->I_prev_imag[i] = 0.0;
        rt->I_phase_prev[i] = 0.0;
    }

    double sum_I_sq_initial = 0.0;
    for (size_t i = 0; i < cfg->N; i++)
	{
        sum_I_sq_initial += rt->I_real[i] * rt->I_real[i] + rt->I_imag[i] * rt->I_imag[i];
    }

    // ============================================================
    // CSV-LOGGING
    // ============================================================
    
    FILE* csv_file = fopen("iwt_timeseries.csv", "w");
    if (csv_file)
	{
        fprintf(csv_file, "iter,sum_I_sq,I_max,sum_mass,sum_E,E_deviation\n");
    }

    string_clear_screen();

    // ============================================================
    // SIMULATIONSSCHLEIFE
    // ============================================================
    
    for (int iter = 0; iter < cfg->MAX_ITER; iter++)
    {
        // Vorherige Werte
        for (size_t i = 0; i < cfg->N; i++)
		{
            rt->I_prev_real[i] = rt->I_real[i];
            rt->I_prev_imag[i] = rt->I_imag[i];
            rt->I_phase_prev[i] = rt->I_phase[i];
        }

        // 1. Fluktuationen (Anhang P) - bringt Energie aus dem Vakuum
        if (!generate_uncertainty_cpu(rt, cfg)) break;
        if (!run_apply_fluctuations(rt, cfg)) break;

        // 2. Fluss (Weber-Kern)
        if (!run_flux_calculation(rt, cfg)) break;

        // 3. Bohm-Potential Q
        if (!run_q_calculation(rt, cfg)) break;

        // 4. Kontinuitäts-Update
        if (!run_update_info(rt, cfg)) break;

        // 5. Rotverschiebung als Energiesenke (Anhang Q)
        //    - Masse ändert sich am Rand
        //    - Phase wird langsamer
        //    - Energie wird ans Vakuum abgegeben
        if (!run_apply_redshift_damping(rt, cfg)) break;

        // 6. Masse und Ladung berechnen
        if (!run_compute_mass_charge(rt, cfg)) break;

        // ============================================================
        // STATISTIK
        // ============================================================
        
        double max_q = 0.0;
        double sum_I_sq = 0.0;
        double sum_mass = 0.0;
        double sum_E = 0.0;
        double I_min = 1e30;
        double I_max = -1e30;

        for (size_t i = 0; i < cfg->N; i++)
        {
            double abs_i = sqrt(rt->I_real[i] * rt->I_real[i] + 
                                rt->I_imag[i] * rt->I_imag[i] + 1e-30);
            double rho_i = abs_i * abs_i;

            sum_I_sq += rho_i;
            sum_mass += compute_mass(rt, i, cfg->N);
            sum_E += compute_energy(rt, i, cfg->N, cfg->DT);

            if (abs_i < I_min) I_min = abs_i;
            if (abs_i > I_max) I_max = abs_i;

            double abs_q = fabs(rt->Q[i]);
            if (abs_q > max_q) max_q = abs_q;
        }

        // Informationserhaltung prüfen
        double I_deviation = (sum_I_sq - sum_I_sq_initial) / (sum_I_sq_initial + 1e-30);

        // Energieänderung (sollte negativ sein → Energiesenke)
        static double sum_E_ref = -1.0;
        if (sum_E_ref < 0.0) sum_E_ref = sum_E;
        double E_deviation = (sum_E - sum_E_ref) / (sum_E_ref + 1e-30);

        // CSV
        if (csv_file)
		{
            fprintf(csv_file, "%d,%.6f,%.6f,%.6f,%.6f,%.6f\n", 
                    iter, sum_I_sq, I_max, sum_mass, sum_E, E_deviation);
        }

        // Heatmaps
        if (iter % 10 == 0)
		{
            char filename[256];
            snprintf(filename, sizeof(filename), "heatmap_mass_%06d.pgm", iter);
            iwt_save_heatmap(rt->mass, cfg->N, filename, "mass");
            
            snprintf(filename, sizeof(filename), "heatmap_charge_%06d.pgm", iter);
            iwt_save_heatmap(rt->charge, cfg->N, filename, "charge");
            
            snprintf(filename, sizeof(filename), "heatmap_energy_%06d.pgm", iter);
            iwt_save_heatmap(rt->mass, cfg->N, filename, "energy");  // Platzhalter
        }

        // Ausgabe
        string_set_cursor_position(1, 1);
        iwt_print_status(rt, cfg, iter, max_q, sum_I_sq, I_min, I_max,
                         I_deviation, sum_I_sq, E_deviation);

        fflush(stdout);
        retval = true;
    }

    if (csv_file)
	{
        fclose(csv_file);
        printf("\nZeitreihe gespeichert in: iwt_timeseries.csv\n");
    }

    printf("\n");
    return retval;
}
