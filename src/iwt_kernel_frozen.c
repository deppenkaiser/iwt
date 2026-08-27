#include "iwt_kernel_frozen.h"
#include "iwt.h"
#include <api/api.h>
#include <math.h>
#include <ocl/ocl.h>
#include <stdio.h>
#include <string/string.h>

/**
 * iwt_kernel_frozen.c - Eingefrorene IWT-Kernfunktionen
 *
 * Status: Eingefroren am 27.07.2026
 *
 * THEORIE: Anhang O "Herleitung der Unschärferelation aus diskreter Zeit"
 *          Anhang P "Vollständige Evolutionsgleichung der IWT"
 *          Anhang Q "Rotverschiebung in der IWT"
 *
 * Enthält zwei fundamentale Prozesse:
 *
 * 1. Vakuumfluktuationen (frozen_generate_uncertainty_cpu)
 *    THEORIE: Die diskrete Zeit T > 0 erzwingt eine minimale Energie-Unschärfe
 *             ΔE ~ ℏ/T (Anhang O). Diese manifestiert sich als intrinsische
 *             Fluktuation des Informationsfeldes ΔI_k ~ sqrt(ℏ/(2T)) * ξ_k.
 *             Gleichung (P.3), Term 4: sqrt(ℏ/(2T)) * ξ_k^(n)
 *
 * 2. Energiesenke / Redshift Damping (frozen_run_apply_redshift_damping)
 *    THEORIE: Die Rotverschiebung wirkt als Energiesenke, die Energie aus dem
 *             Materiesystem entfernt und an das Vakuum zurückgibt (Anhang Q).
 *             Dies schließt den Energiekreislauf: Vakuum -> Materie -> Vakuum.
 *             Gleichung (Q.9): Vakuum -> Fluktuationen -> Materie -> Rotverschiebung -> Vakuum
 */

#define RHO_0 1e-6
#define RHO_MIN 1e-8
#define ALPHA_0 1e-2
#define ALPHA_MIN 1e-9
#define SCALE 0.70710678

/**
 * Erzeugt intrinsische Vakuumfluktuationen.
 * THEORIE: Gleichung (P.3), Term 4: sqrt(ℏ/(2T)) * ξ_k^(n)
 *
 * Die Fluktuation ist KEINE externe Störung, sondern eine Eigenschaft der
 * diskreten Zeit T > 0. Sie ist die mathematische Manifestation des
 * bandbegrenzten Frequenzspektrums (Anhang O).
 *
 * Die Fluktuation wird NUR im Vakuum erzeugt (rho_i klein), da Strukturen
 * (rho_i groß) bereits stabil sind. Die Stärke skaliert mit
 * (1 - rho_i/(RHO_0 + rho_i)), so dass sie im Vakuum maximal und in
 * dichten Strukturen minimal ist.
 *
 * GPU-Port: Die Berechnung läuft jetzt im parallelen Kernel iwt_fluctuations
 * (SplitMix64 + Box-Muller mit pro Slot gemischtem Seed). Die Statistik ist
 * identisch zur früheren CPU-Fassung (Standard-Normalverteilung, ξ_k
 * unkorreliert über das Netzwerk: ⟨ξ_k · ξ_l⟩ = δ_kl).
 */
bool frozen_generate_uncertainty_cpu(const iwt_runtime_t rt, const iwt_config_t cfg)
{
	double scale = SCALE;
	double rho_0 = RHO_0;

	// IWT_NORM_FIX: Seed variiert pro Zeitschritt (vorher identische
	// Zufallssequenz in jedem Frame, da rand_r mit konstantem Seed startete).
	unsigned int seed = (unsigned int) (cfg->seed ^ (rt->n_steps * 2654435761ull));

	// I-Arrays für die Vakuum-Dichtebewertung auf die GPU laden
	clEnqueueWriteBuffer(rt->ocl.queue, rt->I_real_gpu, CL_TRUE, 0,
						 cfg->N * sizeof(double), rt->I_real, 0, NULL, NULL);
	clEnqueueWriteBuffer(rt->ocl.queue, rt->I_imag_gpu, CL_TRUE, 0,
						 cfg->N * sizeof(double), rt->I_imag, 0, NULL, NULL);

	cl_kernel kernel = ocl_get_kernel(&rt->ocl, OCL_KERNEL_IWT_FLUCTUATIONS);
	if (!kernel)
	{
		return false;
	}

	int N = (int) cfg->N;
	clSetKernelArg(kernel, 0, sizeof(cl_mem), &rt->I_real_gpu);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), &rt->I_imag_gpu);
	clSetKernelArg(kernel, 2, sizeof(cl_mem), &rt->xi_real_gpu);
	clSetKernelArg(kernel, 3, sizeof(cl_mem), &rt->xi_imag_gpu);
	clSetKernelArg(kernel, 4, sizeof(int), &N);
	clSetKernelArg(kernel, 5, sizeof(unsigned int), &seed);
	clSetKernelArg(kernel, 6, sizeof(double), &scale);
	clSetKernelArg(kernel, 7, sizeof(double), &rho_0);

	size_t global = cfg->N;
	size_t local = 64;
	if (local > global)
	{
		local = global;
	}

	if (!ocl_enqueue_kernel(&rt->ocl, kernel, global, local))
	{
		return false;
	}

	// xi zuruecklesen, damit die eingefrorene frozen_run_apply_fluctuations
	// sie unverändert auf die GPU hochladen kann
	clEnqueueReadBuffer(rt->ocl.queue, rt->xi_real_gpu, CL_TRUE, 0,
						cfg->N * sizeof(double), rt->xi_real, 0, NULL, NULL);
	clEnqueueReadBuffer(rt->ocl.queue, rt->xi_imag_gpu, CL_TRUE, 0,
						cfg->N * sizeof(double), rt->xi_imag, 0, NULL, NULL);

	return true;
}

/**
 * Transfer der Fluktuationen zur GPU.
 * THEORIE: Die Fluktuationen werden im OpenCL-Kernel iwt_apply_fluctuations
 *          auf das Informationsfeld angewendet (Gleichung P.3).
 */
bool frozen_upload_uncertainty_to_gpu(const iwt_runtime_t rt, const iwt_config_t cfg)
{
	if (clEnqueueWriteBuffer(rt->ocl.queue, rt->xi_real_gpu, CL_TRUE, 0,
							 cfg->N * sizeof(double), rt->xi_real, 0, NULL, NULL)
		!= CL_SUCCESS)
	{
		return false;
	}
	if (clEnqueueWriteBuffer(rt->ocl.queue, rt->xi_imag_gpu, CL_TRUE, 0,
							 cfg->N * sizeof(double), rt->xi_imag, 0, NULL, NULL)
		!= CL_SUCCESS)
	{
		return false;
	}
	return true;
}

/**
 * Wendet die intrinsischen Fluktuationen auf das Informationsfeld an.
 * THEORIE: Gleichung (P.3): I_k^(n+1) = ... + sqrt(ℏ/(2T)) * ξ_k^(n)
 *
 * Die Fluktuationen sind eine zwingende Konsequenz der diskreten Zeit.
 * Sie erzeugen Vakuumfluktuationen ohne externe Annahmen (Anhang P.5).
 */
bool frozen_run_apply_fluctuations(const iwt_runtime_t rt, const iwt_config_t cfg)
{
	cl_kernel kernel = ocl_get_kernel(&rt->ocl, OCL_KERNEL_IWT_APPLY_FLUCTUATIONS);
	if (!kernel)
	{
		return false;
	}

	clEnqueueWriteBuffer(rt->ocl.queue, rt->I_real_gpu, CL_TRUE, 0,
						 cfg->N * sizeof(double), rt->I_real, 0, NULL, NULL);
	clEnqueueWriteBuffer(rt->ocl.queue, rt->I_imag_gpu, CL_TRUE, 0,
						 cfg->N * sizeof(double), rt->I_imag, 0, NULL, NULL);

	if (!frozen_upload_uncertainty_to_gpu(rt, cfg))
	{
		return false;
	}

	int N = (int) cfg->N;
	clSetKernelArg(kernel, 0, sizeof(cl_mem), &rt->I_real_gpu);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), &rt->I_imag_gpu);
	clSetKernelArg(kernel, 2, sizeof(cl_mem), &rt->I_phase_gpu);
	clSetKernelArg(kernel, 3, sizeof(cl_mem), &rt->xi_real_gpu);
	clSetKernelArg(kernel, 4, sizeof(cl_mem), &rt->xi_imag_gpu);
	clSetKernelArg(kernel, 5, sizeof(int), &N);

	size_t global = cfg->N;
	size_t local = 64;
	if (local > global)
	{
		local = global;
	}

	if (!ocl_enqueue_kernel(&rt->ocl, kernel, global, local))
	{
		return false;
	}

	clEnqueueReadBuffer(rt->ocl.queue, rt->I_real_gpu, CL_TRUE,
						0, cfg->N * sizeof(double), rt->I_real, 0, NULL, NULL);
	clEnqueueReadBuffer(rt->ocl.queue, rt->I_imag_gpu, CL_TRUE,
						0, cfg->N * sizeof(double), rt->I_imag, 0, NULL, NULL);

	return true;
}

/**
 * Energiesenke / Redshift Damping (Materie-Zerstrahlung).
 * THEORIE: Anhang Q "Rotverschiebung in der IWT"
 *
 * Die Rotverschiebung wirkt als Energiesenke, die Energie aus dem Materiesystem
 * entfernt und an das Vakuum zurückgibt. Dies schließt den Energiekreislauf:
 *
 *   Vakuum --Fluktuationen--> Materie --Rotverschiebung--> Vakuum  (Q.9)
 *
 * Die Dämpfung wird NUR in Strukturen angewendet (rho_i groß), nicht im Vakuum.
 * Der Dämpfungsfaktor alpha skaliert mit anti_rho = 1/rho_i, so dass dichte
 * Strukturen stärker gedämpft werden als das Vakuum.
 *
 * Die irreversible Dämpfung realisiert den intrinsischen Zeitpfeil der
 * Ω-Theorie (Kap. 9). Die Gravitation ist irreversibel (Kap. 8).
 */
bool frozen_run_apply_redshift_damping(const iwt_runtime_t rt, const iwt_config_t cfg)
{
	cl_kernel kernel = ocl_get_kernel(&rt->ocl, OCL_KERNEL_IWT_REDSHIFT_DAMPING);
	if (!kernel)
	{
		return false;
	}

	// I-Arrays sind bereits auf dem Device (nach frozen_run_apply_fluctuations)
	int N = (int) cfg->N;
	double alpha_0 = ALPHA_0;
	double alpha_min = ALPHA_MIN;

	clSetKernelArg(kernel, 0, sizeof(cl_mem), &rt->I_real_gpu);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), &rt->I_imag_gpu);
	clSetKernelArg(kernel, 2, sizeof(cl_mem), &rt->I_phase_gpu);
	clSetKernelArg(kernel, 3, sizeof(int), &N);
	clSetKernelArg(kernel, 4, sizeof(double), &alpha_0);
	clSetKernelArg(kernel, 5, sizeof(double), &alpha_min);

	size_t global = cfg->N;
	size_t local = 64;
	if (local > global)
	{
		local = global;
	}

	if (!ocl_enqueue_kernel(&rt->ocl, kernel, global, local))
	{
		return false;
	}

	// Readback fuer sum_abs_sq in run_q_calculation (benötigt Host-I)
	clEnqueueReadBuffer(rt->ocl.queue, rt->I_real_gpu, CL_TRUE, 0,
						cfg->N * sizeof(double), rt->I_real, 0, NULL, NULL);
	clEnqueueReadBuffer(rt->ocl.queue, rt->I_imag_gpu, CL_TRUE, 0,
						cfg->N * sizeof(double), rt->I_imag, 0, NULL, NULL);

	return true;
}