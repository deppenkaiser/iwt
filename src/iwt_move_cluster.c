#include "iwt_move_cluster.h"

#include <api/api.h>
#include <math.h>

private struct vector_3d _iwt_compute_weber_force(const iwt_cluster_t a, const iwt_cluster_t b, double G, double c, double epsilon0)
{
	struct vector_3d r_vec = vector_sub(&b->pos, &a->pos);
	ld r_ld = vector_norm(&r_vec);
	double r = (double) r_ld;
	if (r < 1e-30)
	{
		return vector_clear(NULL);
	}

	// ================================================================
	// Relativgeschwindigkeit und -beschleunigung (vereinfacht)
	// ================================================================
	struct vector_3d dv = vector_sub(&b->vel, &a->vel);
	ld dr_ld = vector_dot(&r_vec, &dv) / r_ld;
	double dr = (double) dr_ld;

	// Beschleunigung (Differenz der Geschwindigkeiten, vereinfacht)
	double d2r = 0.0;

	// ================================================================
	// 1. WEBER-GRAVITATION (WG) - wirkt zwischen Massen
	//    β = 0.5 für massive Körper
	// ================================================================
	double F_WG_mag = G * a->mass * b->mass / (r * r);
	double beta_WG = 0.5;
	double factor_WG = 1.0 - (dr * dr) / (c * c) + beta_WG * (r * d2r) / (c * c);
	F_WG_mag *= factor_WG;

	struct vector_3d r_unit = vector_normalize(&r_vec);
	struct vector_3d F_WG_vec = vector_multiply_scalar(&r_unit, (cld) (-F_WG_mag));

	// ================================================================
	// 2. WEBER-ELEKTRODYNAMIK (WED) - wirkt zwischen Ladungen
	//    β = 2 für elektrische Ladungen
	// ================================================================
	double F_WED_mag = (a->charge * b->charge) / (4.0 * iwt_pi() * epsilon0 * r * r);
	double beta_WED = 2.0;
	double factor_WED = 1.0 - (dr * dr) / (c * c) + beta_WED * (r * d2r) / (c * c);
	F_WED_mag *= factor_WED;

	struct vector_3d F_WED_vec = vector_multiply_scalar(&r_unit, (cld) F_WED_mag);

	// ================================================================
	// 3. GESAMTKRAFT (WG + WED)
	// ================================================================
	struct vector_3d force = vector_add(&F_WG_vec, &F_WED_vec);
	return force;
}

private void _iwt_update_cluster_velocities(const iwt_runtime_t rt, double dt)
{
	for (size_t c = 0; c < rt->cluster_count; c++)
	{
		iwt_cluster_t cl = &rt->clusters[c];
		if (!cl->is_active)
		{
			continue;
		}
		struct vector_3d force = vector_clear(NULL);
		for (size_t d = 0; d < rt->cluster_count; d++)
		{
			if (c == d)
			{
				continue;
			}
			iwt_cluster_t other = &rt->clusters[d];
			if (!other->is_active)
			{
				continue;
			}
			struct vector_3d f_ij = _iwt_compute_weber_force(cl, other, 1.0, 1.0, 1.0);
			force = vector_add(&force, &f_ij);
		}
		if (cl->mass > 1e-30)
		{
			struct vector_3d dv = vector_multiply_scalar(&force, (cld) (dt / cl->mass));
			cl->vel = vector_add(&cl->vel, &dv);
		}
	}
}

private void _iwt_update_cluster_phases(const iwt_runtime_t rt, const iwt_config_t cfg, double dt)
{
	double PI = iwt_pi();
	double twoPI = 2.0 * PI;
	for (size_t c = 0; c < rt->cluster_count; c++)
	{
		iwt_cluster_t cl = &rt->clusters[c];
		if (!cl->is_active)
		{
			continue;
		}
		if (cl->node_count == 0)
		{
			continue;
		}
		ld v_speed_ld = vector_norm(&cl->vel);
		double v_speed_sq = (double) (v_speed_ld * v_speed_ld);
		for (size_t n = 0; n < cl->node_count; n++)
		{
			size_t i = cl->node_indices[n];
			struct vector_3d r_vec = vector_sub(&rt->pos[i], &cl->pos);
			ld r_ld = vector_norm(&r_vec);
			double r = (double) r_ld;
			if (r < 1e-30)
			{
				continue;
			}
			struct vector_3d v_vec = cl->vel;
			ld v_rad_ld = vector_dot(&r_vec, &v_vec) / r_ld;
			double v_rad = (double) v_rad_ld;
			double v_tan_sq = v_speed_sq - v_rad * v_rad;
			double v_tan = v_tan_sq > 0.0 ? sqrt(v_tan_sq) : 0.0;
			double lambda_rad = 1.0;
			double dphi_rad = (twoPI / lambda_rad) * v_rad * dt;
			double lambda_tan = 1.0;
			double dphi_tan = (twoPI / lambda_tan) * (v_tan / r) * dt;
			double dphi = dphi_rad + dphi_tan;
			rt->I_phase[i] += dphi;
			while (rt->I_phase[i] > PI)
			{
				rt->I_phase[i] -= twoPI;
			}
			while (rt->I_phase[i] < -PI)
			{
				rt->I_phase[i] += twoPI;
			}
		}
	}
}

private void _iwt_update_I_values(const iwt_runtime_t rt, const iwt_config_t cfg)
{
	for (size_t i = 0; i < cfg->N; i++)
	{
		double rho = rt->mass[i];
		double phi = rt->I_phase[i];
		rt->I_real[i] = sqrt(fabs(rho)) * cos(phi);
		rt->I_imag[i] = sqrt(fabs(rho)) * sin(phi);
	}
}

private void _iwt_update_cluster_centers(const iwt_runtime_t rt)
{
	for (size_t c = 0; c < rt->cluster_count; c++)
	{
		iwt_cluster_t cl = &rt->clusters[c];
		if (!cl->is_active)
		{
			continue;
		}
		if (cl->node_count == 0)
		{
			continue;
		}
		struct vector_3d new_pos = vector_clear(NULL);
		double new_mass = 0.0;
		for (size_t n = 0; n < cl->node_count; n++)
		{
			size_t i = cl->node_indices[n];
			double rho = rt->mass[i];
			struct vector_3d weighted = vector_multiply_scalar(&rt->pos[i], (cld) rho);
			new_pos.x += weighted.x;
			new_pos.y += weighted.y;
			new_pos.z += weighted.z;
			new_mass += rho;
		}
		if (new_mass > 1e-30)
		{
			cl->pos = vector_divide_scalar(&new_pos, (cld) new_mass);
		}
	}
}

private void _iwt_reset_cluster_node_counts(const iwt_runtime_t rt)
{
	for (size_t c = 0; c < rt->cluster_count; c++)
	{
		iwt_cluster_t cl = &rt->clusters[c];
		if (!cl->is_active)
		{
			continue;
		}
		cl->node_count = 0;
	}
}

void iwt_move_clusters(const iwt_runtime_t rt, const iwt_config_t cfg, double dt)
{
	// 1. GESCHWINDIGKEITEN AUS WEBER-KRÄFTEN BERECHNEN
	_iwt_update_cluster_velocities(rt, dt);

	// 2. PHASENVERSCHIEBUNG & I-Werte
	if (cfg->enable_motion)
	{
		_iwt_update_cluster_phases(rt, cfg, dt);
		_iwt_update_I_values(rt, cfg);
	}

	// 4. SCHWERPUNKT NEU BERECHNEN
	_iwt_update_cluster_centers(rt);

	// 5. KNOTENLISTE ZURÜCKSETZEN
	_iwt_reset_cluster_node_counts(rt);
}
