#include "iwt_kernel.h"
#include "iwt.h"
#include "iwt_detect_cluster.h"
#include "iwt_kernel_frozen.h"
#include "iwt_move_cluster.h"
#include <api/api.h>
#include <math.h>
#include <ocl/ocl.h>
#include <stdio.h>
#include <string/string.h>

/**
 * iwt_kernel.c - IWT Kernel-Funktionen (Simulations-Pipeline)
 *
 * THEORIE: Kap. 4 "Diskretes Informations-Lagrange-Funktional"
 *          Anhang P "Vollständige Evolutionsgleichung der IWT"
 *          Anhang Q "Rotverschiebung in der IWT"
 *
 * Diese Datei implementiert die vollständige Simulations-Pipeline:
 *
 * 1. Fluktuationen (frozen_generate_uncertainty_cpu)     -> Gleichung (P.3), Term 4
 * 2. Fluktuationen anwenden (frozen_run_apply_fluctuations) -> Gleichung (P.3), Term 4
 * 3. Redshift Damping (frozen_run_apply_redshift_damping)   -> Anhang Q
 * 4. Flussberechnung (run_flux_calculation)                 -> Gleichung (P.3), Term 1+3
 * 5. Bohm-Potential (run_q_calculation)                     -> Gleichung (P.3), Term 2
 * 6. Informations-Update (run_update_info)                  -> Gleichung (P.3)
 * 7. Masse & Ladung (run_compute_mass_charge)               -> Kap. 3.3
 * 8. Cluster-Erkennung (iwt_detect_clusters)                -> Kap. 2, Axiom 4
 * 9. Cluster-Bewegung (iwt_move_clusters)                   -> Kap. 8 (DSTT/Weber-Kräfte)
 *
 * Die vollständige Evolutionsgleichung der IWT (P.3):
 *
 * I_k^(n+1) = I_k^(n)
 *   + T * sum_l w_kl (I_l - I_k)                     // (1) Lokaler Fluss
 *   + T * lambda * Δ²I_k / I_k                       // (2) Globales Potential (Bohm)
 *   + T * mu * I_k * ln(|I_k|/I_0)                   // (3) Nichtlineare Strukturbildung
 *   + sqrt(ℏ/(2T)) * ξ_k^(n)                         // (4) Intrinsische Unschärfe
 */

#define ALPHA 1.0
#define BETA 1.0
#define DELTA 1.0
#define GAMMA 1.0

// GPU-Cache-Status der Kopplungsmatrix (statisch -> nur bei Bedarf uploaden)
static bool k_gpu_uploaded = false;

void iwt_k_gpu_set_uploaded(bool uploaded)
{
	k_gpu_uploaded = uploaded;
}

static size_t get_local_size(size_t global)
{
	size_t local = 64;
	if (local > global)
	{
		local = global;
	}
	return local;
}

static bool kernel_valid(cl_kernel k)
{
	return k != NULL;
}

static void sanitize_q_array(double* Q, size_t N, double Q_min)
{
	for (size_t i = 0; i < N; i++)
	{
		if (isnan(Q[i]) || isinf(Q[i]))
		{
			Q[i] = Q_min;
		}
	}
}

static double sum_abs_sq(const iwt_runtime_t rt, const iwt_config_t cfg)
{
	double sum = 0.0;
	for (size_t i = 0; i < cfg->N; i++)
	{
		sum += rt->I_real[i] * rt->I_real[i] + rt->I_imag[i] * rt->I_imag[i];
	}
	return sum;
}

/**
 * Berechnet das Bohm-Potential Q.
 * THEORIE: Gleichung (P.3), Term 2: T * lambda * Δ²I_k / I_k
 *
 * Q_k^(n) = -hbar²/(2m) * Δ_d² sqrt(|I_k|) / sqrt(|I_k|)  (Kap. 3.3)
 *
 * Das Bohm-Potential ist der globale, nicht-lokale Anteil der IWT-Dynamik.
 * Es organisiert die Fluktuationen zu stabilen Mustern und ermöglicht die
 * automatische Strukturbildung (Kap. 12).
 */
bool run_q_calculation(const iwt_runtime_t rt, const iwt_config_t cfg)
{
	cl_kernel kernel = ocl_get_kernel(&rt->ocl, OCL_KERNEL_IWT_Q);
	if (!kernel_valid(kernel))
	{
		return false;
	}

	double sum_abs_sq_val = sum_abs_sq(rt, cfg);

	if (sum_abs_sq_val < 1e-30)
	{
		for (size_t i = 0; i < cfg->N; i++)
		{
			rt->Q[i] = 1e-6;
		}
		return true;
	}

	int N = (int) cfg->N;
	double hbar = cfg->hbar;
	double m = 1.0;
	double beta = cfg->beta;
	double epsilon = 1e-6;
	double Q_min = 1e-6;
	double thresh = cfg->cluster_threshold;

	clSetKernelArg(kernel, 0, sizeof(cl_mem), &rt->I_real_gpu);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), &rt->I_imag_gpu);
	clSetKernelArg(kernel, 2, sizeof(cl_mem), &rt->K_gpu);
	clSetKernelArg(kernel, 3, sizeof(cl_mem), &rt->Q_gpu);
	clSetKernelArg(kernel, 4, sizeof(int), &N);
	clSetKernelArg(kernel, 5, sizeof(double), &sum_abs_sq_val);
	clSetKernelArg(kernel, 6, sizeof(double), &hbar);
	clSetKernelArg(kernel, 7, sizeof(double), &m);
	clSetKernelArg(kernel, 8, sizeof(double), &beta);
	clSetKernelArg(kernel, 9, sizeof(double), &epsilon);
	clSetKernelArg(kernel, 10, sizeof(double), &Q_min);
	clSetKernelArg(kernel, 11, sizeof(double), &thresh);

	size_t global = cfg->N;
	size_t local = get_local_size(global);

	if (!ocl_enqueue_kernel(&rt->ocl, kernel, global, local))
	{
		return false;
	}

	clEnqueueReadBuffer(rt->ocl.queue, rt->Q_gpu, CL_TRUE, 0,
						cfg->N * sizeof(double), rt->Q, 0, NULL, NULL);
	sanitize_q_array(rt->Q, cfg->N, Q_min);
	return true;
}

/**
 * Berechnet den Fluss sumJ.
 * THEORIE: Gleichung (P.3), Term 1+3:
 *   Term 1: T * sum_l w_kl (I_l - I_k)  (Lokaler Weber-Fluss)
 *   Term 3: T * mu * I_k * ln(|I_k|/I_0) (Nichtlineare Strukturbildung)
 *
 * Der Fluss treibt die Evolution des Informationsfeldes und realisiert die
 * emergente Dynamik der IWT (Kap. 3.4).
 */
bool run_flux_calculation(const iwt_runtime_t rt, const iwt_config_t cfg)
{
	cl_kernel kernel = ocl_get_kernel(&rt->ocl, OCL_KERNEL_IWT_FLUX);
	if (!kernel_valid(kernel))
	{
		return false;
	}

	// I-Arrays sind bereits auf dem Device (nach GPU-Redshift-Damping)
	// Q muss hochgeladen werden (Vorheriger Frame)
	clEnqueueWriteBuffer(rt->ocl.queue, rt->Q_gpu, CL_TRUE, 0,
						 cfg->N * sizeof(double), rt->Q, 0, NULL, NULL);

	// K-Matrix ist statisch: nur einmal bzw. nach Geometrie-Rebuild uploaden
	if (!k_gpu_uploaded)
	{
		clEnqueueWriteBuffer(rt->ocl.queue, rt->K_gpu, CL_TRUE, 0,
							 cfg->N * cfg->N * sizeof(double), rt->K, 0, NULL, NULL);
		k_gpu_uploaded = true;
	}

	int N = (int) cfg->N;
	double DT = cfg->DT;
	double gamma = cfg->gamma;
	double kappa = cfg->kappa;

	clSetKernelArg(kernel, 0, sizeof(cl_mem), &rt->I_real_gpu);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), &rt->I_imag_gpu);
	clSetKernelArg(kernel, 2, sizeof(cl_mem), &rt->I_phase_gpu);
	clSetKernelArg(kernel, 3, sizeof(cl_mem), &rt->Q_gpu);
	clSetKernelArg(kernel, 4, sizeof(cl_mem), &rt->K_gpu);
	clSetKernelArg(kernel, 5, sizeof(cl_mem), &rt->sumJ_gpu);
	clSetKernelArg(kernel, 6, sizeof(int), &N);
	clSetKernelArg(kernel, 7, sizeof(double), &DT);
	clSetKernelArg(kernel, 8, sizeof(double), &gamma);
	clSetKernelArg(kernel, 9, sizeof(double), &kappa);

	size_t global = cfg->N;
	size_t local = get_local_size(global);

	if (!ocl_enqueue_kernel(&rt->ocl, kernel, global, local))
	{
		return false;
	}

	clEnqueueReadBuffer(rt->ocl.queue, rt->sumJ_gpu, CL_TRUE, 0,
						cfg->N * sizeof(double), rt->sumJ, 0, NULL, NULL);
	return true;
}

/**
 * Informations-Update (Leapfrog/Verlet-Integrator).
 * THEORIE: Gleichung (P.3): I_k^(n+1) = I_k^(n) + T * Φ_k
 *
 * Verwendet einen symplektischen Leapfrog-Integrator für die Zeitentwicklung:
 *   1. Halber Schritt für die Phase (mit altem rho)
 *   2. Vollständiger Schritt für rho (mit phase_half)
 *   3. Halber Schritt für die Phase (mit rho_new)
 *
 * Die Phase wird auf [-π, π] gefaltet (Anhang R).
 */
bool run_update_info(const iwt_runtime_t rt, const iwt_config_t cfg)
{
	cl_kernel kernel = ocl_get_kernel(&rt->ocl, OCL_KERNEL_IWT_UPDATE_INFO);
	if (!kernel_valid(kernel))
	{
		return false;
	}

	int N = (int) cfg->N;
	double DT = cfg->DT;
	double DT_Q = cfg->phase_dt;
	double thresh = cfg->cluster_threshold;

	clSetKernelArg(kernel, 0, sizeof(cl_mem), &rt->I_real_gpu);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), &rt->I_imag_gpu);
	clSetKernelArg(kernel, 2, sizeof(cl_mem), &rt->I_phase_gpu);
	clSetKernelArg(kernel, 3, sizeof(cl_mem), &rt->sumJ_gpu);
	clSetKernelArg(kernel, 4, sizeof(cl_mem), &rt->Q_gpu);
	clSetKernelArg(kernel, 5, sizeof(cl_mem), &rt->K_gpu);
	clSetKernelArg(kernel, 6, sizeof(int), &N);
	clSetKernelArg(kernel, 7, sizeof(double), &DT);
	clSetKernelArg(kernel, 8, sizeof(double), &DT_Q);
	clSetKernelArg(kernel, 9, sizeof(double), &thresh);

	size_t global = cfg->N;
	size_t local = get_local_size(global);

	if (!ocl_enqueue_kernel(&rt->ocl, kernel, global, local))
	{
		return false;
	}

	clEnqueueReadBuffer(rt->ocl.queue, rt->I_real_gpu, CL_TRUE, 0,
						cfg->N * sizeof(double), rt->I_real, 0, NULL, NULL);
	clEnqueueReadBuffer(rt->ocl.queue, rt->I_imag_gpu, CL_TRUE, 0,
						cfg->N * sizeof(double), rt->I_imag, 0, NULL, NULL);
	clEnqueueReadBuffer(rt->ocl.queue, rt->I_phase_gpu, CL_TRUE, 0,
						cfg->N * sizeof(double), rt->I_phase, 0, NULL, NULL);
	return true;
}

/**
 * Berechnet emergente Masse und Ladung.
 * THEORIE: Kap. 3.3 "Information als Ursprung physikalischer Größen"
 *
 * Masse:   m_k = δ * sum_l |I_k - I_l|^2   (Gleichung 3.8)
 * Ladung:  q_k = sum_l (phi_k - phi_l)     (Anhang R, Gleichung R.2)
 *
 * Die Masse misst den Widerstand gegen Änderungen der lokalen Informationsstruktur.
 * Die Ladung ist die diskrete Divergenz der Phase (topologische Ladung, Anhang R).
 */
bool run_compute_mass_charge(const iwt_runtime_t rt, const iwt_config_t cfg)
{
	cl_kernel kernel = ocl_get_kernel(&rt->ocl, OCL_KERNEL_IWT_MASS_CHARGE);
	if (!kernel_valid(kernel))
	{
		return false;
	}

	// I-Arrays sind bereits auf dem Device (nach run_update_info)
	// Kein redundant noetig.

	int N = (int) cfg->N;
	double delta = DELTA;

	clSetKernelArg(kernel, 0, sizeof(cl_mem), &rt->I_real_gpu);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), &rt->I_imag_gpu);
	clSetKernelArg(kernel, 2, sizeof(cl_mem), &rt->I_phase_gpu);
	clSetKernelArg(kernel, 3, sizeof(cl_mem), &rt->mass_gpu);
	clSetKernelArg(kernel, 4, sizeof(cl_mem), &rt->charge_gpu);
	clSetKernelArg(kernel, 5, sizeof(int), &N);
	clSetKernelArg(kernel, 6, sizeof(double), &delta);

	size_t global = cfg->N;
	size_t local = get_local_size(global);

	if (!ocl_enqueue_kernel(&rt->ocl, kernel, global, local))
	{
		return false;
	}

	clEnqueueReadBuffer(rt->ocl.queue, rt->mass_gpu, CL_TRUE, 0,
						cfg->N * sizeof(double), rt->mass, 0, NULL, NULL);
	clEnqueueReadBuffer(rt->ocl.queue, rt->charge_gpu, CL_TRUE, 0,
						cfg->N * sizeof(double), rt->charge, 0, NULL, NULL);
	return true;
}

/**
 * Führt einen vollständigen diskreten Zeitschritt der IWT aus.
 * THEORIE: Die vollständige Evolutionsgleichung (P.3)
 *
 * Die Pipeline realisiert die geschlossene Theorie:
 * - Vakuumfluktuation ist intrinsisch (Anhang O, P)
 * - Strukturbildung ist automatisch (Kap. 2, Axiom 4)
 * - DBT emergiert im Grenzfall T→0 (Anhang F)
 * - Die Theorie ist geschlossen (Anhang P.5)
 */
bool run_simulation_step(const iwt_runtime_t rt, const iwt_config_t cfg)
{
	rt->n_steps++;

	// Speichere vorherigen Zustand für die Weber-Kraft (Kap. 3.4)
	for (size_t i = 0; i < cfg->N; i++)
	{
		rt->I_prev_real[i] = rt->I_real[i];
		rt->I_prev_imag[i] = rt->I_imag[i];
		rt->I_phase_prev[i] = rt->I_phase[i];
	}

	typedef bool (*step_fn)(const iwt_runtime_t, const iwt_config_t);
	step_fn steps[] = {
		frozen_generate_uncertainty_cpu,   // Gleichung (P.3), Term 4
		frozen_run_apply_fluctuations,	   // Gleichung (P.3), Term 4
		frozen_run_apply_redshift_damping, // Anhang Q (Energiesenke)
		run_flux_calculation,			   // Gleichung (P.3), Term 1+3
		run_q_calculation,				   // Gleichung (P.3), Term 2
		run_update_info,				   // Gleichung (P.3)
		run_compute_mass_charge			   // Kap. 3.3
	};

	for (size_t s = 0; s < sizeof(steps) / sizeof(steps[0]); s++)
	{
		if (!steps[s](rt, cfg))
		{
			return false;
		}
	}

	// Cluster-Erkennung (Kap. 2, Axiom 4) und Bewegung (Kap. 8)
	iwt_detect_clusters(rt, cfg);
	iwt_move_clusters(rt, cfg, cfg->DT);

	// Einmaliger GPU-Sync am Frame-Ende (statt clFinish nach jedem Kernel)
	ocl_finish_frame(&rt->ocl);

	return true;
}