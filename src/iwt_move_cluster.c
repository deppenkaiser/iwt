#include "iwt_move_cluster.h"

#include <api/api.h>
#include <math.h>

/**
 * IWT_NORM: Führungsgeschwindigkeit (Bohm/Guidance) und quasi-teilchenhafter Drift.
 * Theorie: WDBT+ Führungsgleichung v = ∇S/m (de Broglie-Bohm, Teil II);
 *          Kap. 8 (Weber-Kräfte zwischen emergenten Objekten).
 * Implementierung:
 *   - Diskreter Phasengradient: grad = Σ_edges K_ij·ΔS_ij·r_ij (ΔS auf [-π,π] gefaltet)
 *   - Richtung = normierter Gradient, Betrag = GUIDANCE_SPEED·l0/DT
 *     (konstante Gleitgeschwindigkeit bei vorhandenem Gradienten)
 *   - Weber-Anteil wird separat integriert (vel_weber, gedämpft)
 *   - pos_offset integriert vel relativ zum knotenverankerten Schwerpunkt
 * Grund: Die Knoten des fraktalen Gitters sind statisch; sichtbare Bewegung
 * emergiert aus Feldtransport (Advektion im Flux-Kernel) + Objektdrift.
 */

#define WEBER_VELOCITY_DAMP 0.98
#define OFFSET_DECAY 0.98
#define GUIDANCE_SPEED 0.1
#define GUIDANCE_EPSILON 1e-12

static double phase_wrap(double dphi)
{
	const double two_pi = 2.0 * iwt_pi();
	return dphi - two_pi * rint(dphi / two_pi);
}

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
			cl->vel_weber = vector_add(&cl->vel_weber, &dv);
		}
		// Dämpfung hält den integrierten Impulsanteil beschränkt
		cl->vel_weber = vector_multiply_scalar(&cl->vel_weber, (cld) WEBER_VELOCITY_DAMP);
	}
}

/**
 * Führungsgeschwindigkeit aus dem diskreten Phasengradienten (Bohm).
 * v_c ∝ Σ_edges K_ij · ΔS_ij · r_ij, ΔS gefaltet auf [-π, π].
 */private void _iwt_apply_guidance(const iwt_runtime_t rt, const iwt_config_t cfg)
{
	double vmax = GUIDANCE_SPEED * cfg->l0 / cfg->DT;
	size_t N = cfg->N;

	for (size_t c = 0; c < rt->cluster_count; c++)
	{
		iwt_cluster_t cl = &rt->clusters[c];
		if (!cl->is_active || cl->node_count == 0)
		{
			continue;
		}

		struct vector_3d grad = vector_clear(NULL);
		for (size_t n = 0; n < cl->node_count; n++)
		{
			size_t i = cl->node_indices[n];
			const bool* row = &rt->adjacency[i * N];
			double phase_i = rt->I_phase[i];
			for (size_t j = 0; j < N; j++)
			{
				if (!row[j])
				{
					continue;
				}
				double ds = phase_wrap(rt->I_phase[j] - phase_i);
				if (fabs(ds) < GUIDANCE_EPSILON)
				{
					continue;
				}
				struct vector_3d r_vec = vector_sub(&rt->pos[j], &rt->pos[i]);
				ld r_ld = vector_norm(&r_vec);
				double r = (double) r_ld;
				if (r < GUIDANCE_EPSILON)
				{
					continue;
				}
				struct vector_3d dir = vector_multiply_scalar(&r_vec, (cld) (1.0 / r_ld));
				struct vector_3d contrib = vector_multiply_scalar(&dir, (cld) (rt->K[i * N + j] * ds));
				grad = vector_add(&grad, &contrib);
			}
		}

		ld mag = vector_norm(&grad);
		if (mag > GUIDANCE_EPSILON)
		{
			struct vector_3d dir = vector_multiply_scalar(&grad, (cld) (1.0 / mag));
			cl->vel = vector_multiply_scalar(&dir, (cld) vmax);
		}
		else
		{
			cl->vel = cl->vel_weber;
		}
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
			// Knotenverankerter Schwerpunkt + integrierter Drift-Offset
			struct vector_3d anchor = vector_divide_scalar(&new_pos, (cld) new_mass);
			cl->pos_offset = vector_multiply_scalar(&cl->pos_offset, (cld) OFFSET_DECAY);
			cl->pos = vector_add(&anchor, &cl->pos_offset);
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
	// 1. WEBER-IMPULSE (persistent, gedämpft)
	_iwt_update_cluster_velocities(rt, dt);

	// 2. FÜHRUNGSGESCHWINDIGKEIT aus dem Phasengradienten (Bohm)
	_iwt_apply_guidance(rt, cfg);

	// 4. SCHWERPUNKT + DRIFT-OFFSET NEU BERECHNEN
	_iwt_update_cluster_centers(rt);
	if (cfg->enable_motion)
	{
		for (size_t c = 0; c < rt->cluster_count; c++)
		{
			iwt_cluster_t cl = &rt->clusters[c];
			if (!cl->is_active || cl->node_count == 0)
			{
				continue;
			}
			struct vector_3d drift = vector_multiply_scalar(&cl->vel, (cld) dt);
			cl->pos_offset = vector_add(&cl->pos_offset, &drift);
			cl->pos = vector_add(&cl->pos, &drift);
		}
	}

	// 5. KNOTENLISTE ZURÜCKSETZEN
	_iwt_reset_cluster_node_counts(rt);
}
