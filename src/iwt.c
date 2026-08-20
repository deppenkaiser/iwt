// ============================================================================
// iwt.c - AUSGABE MIT STATISTIK STATT EINZELNER KNOTEN
// ============================================================================

#include "iwt.h"
#include <api/api.h>
#include <math.h>
#include <string/string.h>
#include <vector/vector.h>

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

double iwt_pi(void)
{
	return 4.0 * atan(1.0);
}

double iwt_fundamental_length(void)
{
	const double h = 6.62607015e-34;
	const double mp = 1.67262192e-27;
	const double c = 299792458.0;
	const double phi = (1.0 + sqrt(5.0)) / 2.0;
	const double V_dode = (15.0 + 7.0 * sqrt(5.0)) / 4.0;
	const double V_sphere = 4.0 * iwt_pi() / 3.0;
	const double V_ratio = V_dode / V_sphere;
	const double lambda_p = h / (mp * c);
	return pow(V_ratio, 1.0 / 3.0) * lambda_p;
}

double iwt_fundamental_time(void)
{
	const double c = 299792458.0;
	return iwt_fundamental_length() / c;
}

double iwt_fractal_dimension(void)
{
	const double phi = (1.0 + sqrt(5.0)) / 2.0;
	const double N = 20.0 * phi;
	const double s = 2.0 + phi;
	return log(N) / log(s);
}

double iwt_alpha_IWT(void)
{
	return 1;
}

double iwt_beta_IWT(void)
{
	return 1;
}

void iwt_compute_node_colors(const double* mass, const double* charge, size_t N, float* out_rgb)
{
	double mass_min = 1e30;
	double mass_max = -1e30;
	for (size_t i = 0; i < N; i++)
	{
		mass_min = MIN(mass_min, mass[i]);
		mass_max = MAX(mass_max, mass[i]);
	}
	double mass_range = mass_max - mass_min;
	if (mass_range < 1e-30)
	{
		mass_range = 1.0;
	}

	double max_abs = 0.0;
	for (size_t i = 0; i < N; i++)
	{
		max_abs = MAX(max_abs, fabs(charge[i]));
	}
	if (max_abs < 1e-30)
	{
		max_abs = 1.0;
	}

	// Helligkeit = Masse, Farbrichtung (Rot/Blau) = Ladung - gleiches Schema
	// wie iwt_build_overlay_rgb(), aber pro Knoten statt pro Pixel.
	for (size_t i = 0; i < N; i++)
	{
		double brightness = (mass[i] - mass_min) / mass_range;
		double charge_norm = charge[i] / max_abs;
		double abs_charge = fabs(charge_norm);

		double r = brightness * (charge_norm < 0.0 ? -charge_norm : (1.0 - abs_charge));
		double g = brightness * (1.0 - abs_charge);
		double b = brightness * (charge_norm > 0.0 ? charge_norm : (1.0 - abs_charge));

		out_rgb[i * 3 + 0] = (float) r;
		out_rgb[i * 3 + 1] = (float) g;
		out_rgb[i * 3 + 2] = (float) b;
	}
}
