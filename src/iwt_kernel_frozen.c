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

static double box_muller(unsigned int *seed)
{
	double u1, u2;
	do
	{
		u1 = (double)rand_r(seed) / (double)RAND_MAX;
		u2 = (double)rand_r(seed) / (double)RAND_MAX;
	} while (u1 < 1e-30 || u2 < 1e-30);
	return sqrt(-2.0 * log(u1)) * cos(2.0 * iwt_pi() * u2);
}

// ============================================================================
// 1. FLUKTUATIONEN (Vakuumfluktuation)
// ============================================================================

bool frozen_generate_uncertainty_cpu(const iwt_runtime_t rt, const iwt_config_t cfg)
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

	double scale = SCALE;
	unsigned int seed = cfg->seed;

	for (size_t i = 0; i < cfg->N; i++)
	{
		double rho_i = rt->I_real[i] * rt->I_real[i] +
					   rt->I_imag[i] * rt->I_imag[i] + 1e-30;

		// Fluktuation NUR im Vakuum
		double fluct_strength = scale * (1.0 - rho_i / (RHO_0 + rho_i));

		double delta = box_muller(&seed) * fluct_strength;

		rt->xi_real[i] = delta;
		rt->xi_imag[i] = delta;
		rt->uncertainty[i] = 0.0;
	}

	return true;
}

bool frozen_upload_uncertainty_to_gpu(const iwt_runtime_t rt, const iwt_config_t cfg)
{
	if (clEnqueueWriteBuffer(rt->ocl.queue, rt->xi_real_gpu, CL_TRUE, 0,
							 cfg->N * sizeof(double), rt->xi_real, 0, NULL, NULL) != CL_SUCCESS)
		return false;
	if (clEnqueueWriteBuffer(rt->ocl.queue, rt->xi_imag_gpu, CL_TRUE, 0,
							 cfg->N * sizeof(double), rt->xi_imag, 0, NULL, NULL) != CL_SUCCESS)
		return false;
	return true;
}

bool frozen_run_apply_fluctuations(const iwt_runtime_t rt, const iwt_config_t cfg)
{
	cl_kernel kernel = ocl_get_kernel(&rt->ocl, OCL_KERNEL_IWT_APPLY_FLUCTUATIONS);
	if (!kernel)
		return false;

	clEnqueueWriteBuffer(rt->ocl.queue, rt->I_real_gpu, CL_TRUE, 0,
						 cfg->N * sizeof(double), rt->I_real, 0, NULL, NULL);
	clEnqueueWriteBuffer(rt->ocl.queue, rt->I_imag_gpu, CL_TRUE, 0,
						 cfg->N * sizeof(double), rt->I_imag, 0, NULL, NULL);

	if (!frozen_upload_uncertainty_to_gpu(rt, cfg))
		return false;

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
	if (local > global)
		local = global;

	if (!ocl_enqueue_kernel(&rt->ocl, kernel, global, local))
		return false;

	clEnqueueReadBuffer(rt->ocl.queue, rt->I_real_gpu, CL_TRUE,
						0, cfg->N * sizeof(double), rt->I_real, 0, NULL, NULL);
	clEnqueueReadBuffer(rt->ocl.queue, rt->I_imag_gpu, CL_TRUE,
						0, cfg->N * sizeof(double), rt->I_imag, 0, NULL, NULL);

	return true;
}

// ============================================================================
// 2. ENERGIESSENKE (Materie-Zerstrahlung)
// ============================================================================

bool frozen_run_apply_redshift_damping(const iwt_runtime_t rt, const iwt_config_t cfg)
{
	double alpha_0 = ALPHA_0;
	double alpha_min = ALPHA_MIN;

	for (size_t i = 0; i < cfg->N; i++)
	{
		double rho_i = rt->I_real[i] * rt->I_real[i] +
					   rt->I_imag[i] * rt->I_imag[i] + 1e-30;

		double anti_rho = 1.0 / rho_i;

		// Energiesenke NUR in Strukturen
		double alpha = alpha_0 * (anti_rho / (1.0 + anti_rho)) + alpha_min;

		rt->I_real[i] *= (1.0 - alpha);
		rt->I_imag[i] *= (1.0 - alpha);
	}

	return true;
}
