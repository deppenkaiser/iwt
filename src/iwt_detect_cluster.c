#include "iwt_detect_cluster.h"

#include <api/api.h>
#include <stdio.h>
#include <memory.h>

/**
 * Cluster-Detektion mit Persistence, Glättung und Hysterese.
 *
 * IWT_NORM: Numerische Stabilisierung der Strukturerkennung.
 * Theorie: Axiom 4 (Strukturbildung ist automatisch), Kap. 2
 * Implementierung:
 *   1. EMA-Glättung der Knotenmasse (Zeitkonstante ~4 Schritte)
 *   2. Hysterese: Beitritt ab DETECT_JOIN, Verbleib bis DETECT_LEAVE
 *   3. Matching gegen Cluster des Vorframes: vel/id/type werden übernommen
 * Grund: Ohne Gedächtnis werden Cluster jeden Frame neu geboren (vel = 0);
 * die Weber-Integration liefert dann nur Brownsches Zittern ohne Drift.
 */

#define DETECT_EMA_ALPHA 0.25
#define DETECT_JOIN 1e-6
#define DETECT_LEAVE (3e-7)
#define DETECT_MATCH_RADIUS_FACTOR 3.0

static void flood_fill_process_node(const iwt_runtime_t rt, iwt_cluster_t c, size_t i);
static bool flood_fill_should_visit(const iwt_runtime_t rt, size_t i);
static void flood_fill_push_neighbors(const iwt_runtime_t rt, const iwt_config_t cfg, size_t i, size_t* stack, size_t* stack_ptr);
static void flood_fill(const iwt_runtime_t rt, const iwt_config_t cfg, size_t idx, iwt_cluster_t c);
static void detect_reset_visited(const iwt_runtime_t rt, const iwt_config_t cfg);
static void detect_reset_clusters(const iwt_runtime_t rt);
static bool detect_should_start_cluster(const iwt_runtime_t rt, size_t i);
static void detect_init_cluster(iwt_cluster_t c, size_t id);
static bool detect_finalize_cluster(const iwt_runtime_t rt, iwt_cluster_t c);
static void detect_save_previous(const iwt_runtime_t rt);
static void detect_smooth_mass(const iwt_runtime_t rt, const iwt_config_t cfg);
static void detect_update_membership(const iwt_runtime_t rt, const iwt_config_t cfg);
static double detect_threshold_for(const iwt_runtime_t rt, size_t i);
static void iwt_match_clusters(const iwt_runtime_t rt, const iwt_config_t cfg);

private void _flood_fill(const iwt_runtime_t rt, const iwt_config_t cfg, size_t idx, iwt_cluster_t c)
{
	flood_fill(rt, cfg, idx, c);
}

static void flood_fill(const iwt_runtime_t rt, const iwt_config_t cfg, size_t idx, iwt_cluster_t c)
{
	size_t stack[cfg->N];
	size_t stack_ptr = 0;
	stack[stack_ptr++] = idx;

	while (stack_ptr > 0)
	{
		size_t i = stack[--stack_ptr];
		if (!flood_fill_should_visit(rt, i))
		{
			continue;
		}
		flood_fill_process_node(rt, c, i);
		flood_fill_push_neighbors(rt, cfg, i, stack, &stack_ptr);
	}
}

static bool flood_fill_should_visit(const iwt_runtime_t rt, size_t i)
{
	if (rt->visited[i])
	{
		return false;
	}
	return rt->mass_smooth[i] >= detect_threshold_for(rt, i);
}

static void flood_fill_process_node(const iwt_runtime_t rt, iwt_cluster_t c, size_t i)
{
	rt->visited[i] = true;
	c->node_indices[c->node_count++] = i;
	c->mass += rt->mass[i];
	struct vector_3d weighted = vector_multiply_scalar(&rt->pos[i], (cld) rt->mass[i]);
	c->pos = vector_add(&c->pos, &weighted);
	c->charge += rt->charge[i];
	c->phase += rt->I_phase[i];
}

static void flood_fill_push_neighbors(const iwt_runtime_t rt, const iwt_config_t cfg, size_t i, size_t* stack, size_t* stack_ptr)
{
	int count = rt->adj_count[i];
	const int* neighbors = &rt->adj_flat[(size_t) i * IWT_ADJ_STRIDE];
	for (int n = 0; n < count; n++)
	{
		size_t j = (size_t) neighbors[n];
		if (!rt->visited[j] && rt->mass_smooth[j] >= detect_threshold_for(rt, j))
		{
			if (*stack_ptr >= cfg->N)
			{
				break;
			}
			stack[(*stack_ptr)++] = j;
		}
	}
}

// Erkennt Cluster aus dem Informationsfeld via Flood-Fill über Adjazenz.
// Realisiert automatische Strukturbildung: Fluktuationen werden lokal verstärkt und durch globales Potential zu stabilen Mustern organisiert.
// Theorie: Strukturbildung ist automatisch, benötigt keine externen Initialisierungen, vgl. app:iwt_eq_konsequenzen § Die Strukturbildung ist automatisch.
// Axiome: IWT vereinheitlicht alle Wechselwirkungen und macht testbare Vorhersagen, vgl. sec:axiome_zusammenfassung.
void iwt_detect_clusters(const iwt_runtime_t rt, const iwt_config_t cfg)
{
	detect_save_previous(rt);
	detect_smooth_mass(rt, cfg);
	detect_reset_visited(rt, cfg);
	detect_reset_clusters(rt);

	for (size_t i = 0; i < cfg->N; i++)
	{
		if (!detect_should_start_cluster(rt, i))
		{
			continue;
		}
		if (rt->cluster_count >= rt->cluster_capacity)
		{
			break;
		}
		iwt_cluster_t c = &rt->clusters[rt->cluster_count];
		detect_init_cluster(c, rt->cluster_count);
		_flood_fill(rt, cfg, i, c);
		if (detect_finalize_cluster(rt, c))
		{
			rt->cluster_count++;
		}
	}

	detect_update_membership(rt, cfg);
	iwt_match_clusters(rt, cfg);

	// printf im Hot-Path drosseln (stdout-Flush pro Frame ist teuer)
	static int detect_frame = 0;
	if ((detect_frame++ % 240) == 0)
	{
		printf("Gefundene Cluster: %d\n", rt->cluster_count);
	}
}

static void detect_reset_visited(const iwt_runtime_t rt, const iwt_config_t cfg)
{
	memset(rt->visited, 0, cfg->N * sizeof(bool));
}

static void detect_reset_clusters(const iwt_runtime_t rt)
{
	rt->cluster_count = 0;
}

static bool detect_should_start_cluster(const iwt_runtime_t rt, size_t i)
{
	if (rt->visited[i])
	{
		return false;
	}
	return rt->mass_smooth[i] >= detect_threshold_for(rt, i);
}

static double detect_threshold_for(const iwt_runtime_t rt, size_t i)
{
	return rt->was_member[i] ? DETECT_LEAVE : DETECT_JOIN;
}

static void detect_init_cluster(iwt_cluster_t c, size_t id)
{
	c->id = id;
	c->node_count = 0;
	c->mass = 0.0;
	c->charge = 0.0;
	c->phase = 0.0;
	vector_clear(&c->pos);
	vector_clear(&c->vel);
	vector_clear(&c->vel_weber);
	vector_clear(&c->pos_offset);
	c->external = false;
	c->is_active = true;
}

static bool detect_finalize_cluster(const iwt_runtime_t rt, iwt_cluster_t c)
{
	if (c->node_count == 0)
	{
		return false;
	}
	c->pos = vector_divide_scalar(&c->pos, (cld) c->mass);
	c->phase /= c->node_count;

	// Extern = Mitglieder aus >= 2 verschiedenen Generator-Zellen
	// (Fruehabbruch bei 32 verschiedenen IDs - extern ist damit sicher).
	int distinct[32];
	int nd = 0;
	for (size_t n = 0; n < c->node_count && nd < 32; n++)
	{
		int id = rt->node_cell[c->node_indices[n]];
		bool known = false;
		for (int d = 0; d < nd; d++)
		{
			if (distinct[d] == id)
			{
				known = true;
				break;
			}
		}
		if (!known)
		{
			distinct[nd++] = id;
		}
	}
	c->external = nd >= 2;
	return true;
}

static void detect_save_previous(const iwt_runtime_t rt)
{
	if (rt->cluster_count > 0)
	{
		memcpy(rt->clusters_prev, rt->clusters,
			   rt->cluster_count * sizeof(struct iwt_cluster));
	}
	rt->cluster_count_prev = rt->cluster_count;
}

static void detect_smooth_mass(const iwt_runtime_t rt, const iwt_config_t cfg)
{
	for (size_t i = 0; i < cfg->N; i++)
	{
		rt->mass_smooth[i] += DETECT_EMA_ALPHA * (rt->mass[i] - rt->mass_smooth[i]);
	}
}

static void detect_update_membership(const iwt_runtime_t rt, const iwt_config_t cfg)
{
	for (size_t i = 0; i < cfg->N; i++)
	{
		rt->was_member[i] = rt->visited[i];
	}
}

static void iwt_match_clusters(const iwt_runtime_t rt, const iwt_config_t cfg)
{
	double match_radius_sq = 0.0;
	if (cfg->l0 > 0.0)
	{
		double r = DETECT_MATCH_RADIUS_FACTOR * cfg->l0;
		match_radius_sq = r * r;
	}

	for (size_t c = 0; c < rt->cluster_count; c++)
	{
		iwt_cluster_t cl = &rt->clusters[c];
		if (!cl->is_active || cl->node_count == 0)
		{
			continue;
		}

		int best_prev = -1;
		double best_dist_sq = match_radius_sq;
		for (size_t p = 0; p < rt->cluster_count_prev; p++)
		{
			iwt_cluster_t old = &rt->clusters_prev[p];
			if (!old->is_active || old->node_count == 0)
			{
				continue;
			}
			struct vector_3d d = vector_sub(&cl->pos, &old->pos);
			ld dist_sq = vector_dot(&d, &d);
			if ((double) dist_sq < best_dist_sq)
			{
				best_dist_sq = (double) dist_sq;
				best_prev = (int) p;
			}
		}

		if (best_prev >= 0)
		{
			iwt_cluster_t old = &rt->clusters_prev[best_prev];
			cl->vel = old->vel;
			cl->vel_weber = old->vel_weber;
			cl->pos_offset = old->pos_offset;
			cl->id = old->id;
			cl->type = old->type;
			cl->external = old->external;
		}
	}
}
