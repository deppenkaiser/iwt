// ============================================================================
// iwt_kernel_frozen.c - Eingefrorene IWT-Kernfunktionen
// ============================================================================
//
// Status: Eingefroren am 27.07.2026
//
// Enthält NUR:
//   1. Fluktuationen (Vakuumfluktuation)
//   2. Energiesenke (Materie-Zerstrahlung)
//
// Diese Funktionen werden NICHT mehr geändert.
//
// ============================================================================

#include "iwt_kernel_frozen.h"
#include "iwt.h"
#include <api/api.h>
#include <math.h>
#include <ocl/ocl.h>
#include <stdio.h>
#include <string/string.h>

// ============================================================================
// KONSTANTEN (eingefroren)
// ============================================================================

#define RHO_0 1e-6
#define RHO_MIN 1e-8
#define ALPHA_0 1e-2
#define ALPHA_MIN 1e-9
#define SCALE 0.70710678

// ============================================================================
// HILFSFUNKTIONEN (eingefroren)
// ============================================================================

static double box_muller(unsigned int* seed)
{
	double u1, u2;
	do
	{
		u1 = (double) rand_r(seed) / (double) RAND_MAX;
		u2 = (double) rand_r(seed) / (double) RAND_MAX;
	} while (u1 < 1e-30 || u2 < 1e-30);
	return sqrt(-2.0 * log(u1)) * cos(2.0 * iwt_pi() * u2);
}

// ============================================================================
// 1. FLUKTUATIONEN (Vakuumfluktuation)
// ============================================================================

// Vakuumfluktuation ist intrinsisch: Gleichung (P.3) erzeugt immer Fluktuationen auch im leeren Vakuum.
// Die Vakuumfluktuation ist keine externe Annahme, sondern eine Eigenschaft der diskreten Zeit.
// Vgl. app:iwt_eq_konsequenzen § Die Vakuumfluktuation ist intrinsisch
// Diese Funktion realisiert den intrinsischen Fluktuationsterm xi_real/xi_imag für jeden Knoten.
bool frozen_generate_uncertainty_cpu(const iwt_runtime_t rt, const iwt_config_t cfg)
{
	double scale = SCALE;
	unsigned int seed = cfg->seed;

	// ================================================================
	// NEU: Knoten-Indizes in zufälliger Reihenfolge
	// ================================================================
	size_t* indices = malloc(cfg->N * sizeof(size_t));
	if (!indices)
	{
		return false;
	}

	for (size_t i = 0; i < cfg->N; i++)
	{
		indices[i] = i;
	}

	// Fisher-Yates Shuffle
	for (size_t i = cfg->N - 1; i > 0; i--)
	{
		size_t j = rand_r(&seed) % (i + 1);
		size_t temp = indices[i];
		indices[i] = indices[j];
		indices[j] = temp;
	}

	// ================================================================
	// Fluktuationen in zufälliger Reihenfolge erzeugen
	// ================================================================
	for (size_t n = 0; n < cfg->N; n++)
	{
		size_t i = indices[n];

		double rho_i = rt->I_real[i] * rt->I_real[i] + rt->I_imag[i] * rt->I_imag[i] + 1e-30;

		// Fluktuation NUR im Vakuum
		double fluct_strength = scale * (1.0 - rho_i / (RHO_0 + rho_i));

		double delta = box_muller(&seed) * fluct_strength;

		rt->xi_real[i] = delta;
		rt->xi_imag[i] = delta;
		rt->uncertainty[i] = 0.0;
	}

	free(indices);
	return true;
}

// Transfer der intrinsisch erzeugten Fluktuationen xi_real/xi_imag von Host zu GPU.
// Notwendig damit der OpenCL-Kernel iwt_apply_fluctuations die diskrete Unschärfe anwenden kann.
// Theorie: Fluktuationen sind Teil der geschlossenen Evolutionsgleichung P.3, vgl. app:iwt_eq_konsequenzen § Die Theorie ist geschlossen.
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

// Anwendung der intrinsischen Vakuumfluktuation auf das Informationsfeld I_real/I_imag.
// Realisiert den Term aus Gleichung P.3 der IWT: Fluktuationen werden immer erzeugt, auch im Vakuum.
// Theorie: Vakuumfluktuation ist intrinsisch, vgl. app:iwt_eq_konsequenzen § Die Vakuumfluktuation ist intrinsisch.
// IWT ist fundamental diskret, emergent kontinuierlich, vgl. sec:axiome_zusammenfassung.
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
	clSetKernelArg(kernel, 2, sizeof(cl_mem), &rt->xi_real_gpu);
	clSetKernelArg(kernel, 3, sizeof(cl_mem), &rt->xi_imag_gpu);
	clSetKernelArg(kernel, 4, sizeof(int), &N);

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

// ============================================================================
// 2. ENERGIESSENKE (Materie-Zerstrahlung)
// ============================================================================

// Energiesenke / Redshift Damping: Materie-Zerstrahlung als irreversibler Dissipationsterm.
// Realisiert die Irreversibilität der fundamentalen WDBT+ und damit der Ω-Theorie.
// Ω-Theorie vermeidet Singularitäten, kommt ohne dunkle Materie/Energie aus und bewahrt Irreversibilität.
// Vgl. sec:mg_zusammenhaenge und app:iwt_eq_konsequenzen § Die Theorie ist geschlossen.
bool frozen_run_apply_redshift_damping(const iwt_runtime_t rt, const iwt_config_t cfg)
{
	double alpha_0 = ALPHA_0;
	double alpha_min = ALPHA_MIN;

	for (size_t i = 0; i < cfg->N; i++)
	{
		double rho_i = rt->I_real[i] * rt->I_real[i] + rt->I_imag[i] * rt->I_imag[i] + 1e-30;

		double anti_rho = 1.0 / rho_i;

		// Energiesenke NUR in Strukturen
		double alpha = alpha_0 * (anti_rho / (1.0 + anti_rho)) + alpha_min;

		rt->I_real[i] *= (1.0 - alpha);
		rt->I_imag[i] *= (1.0 - alpha);
	}

	return true;
}
