#include "iwt_detect_cluster.h"

#include <api/api.h>
#include <stdio.h>
#include <memory.h>

static void flood_fill_process_node(const iwt_runtime_t rt, iwt_cluster_t c, size_t i);
static bool flood_fill_should_visit(const iwt_runtime_t rt, size_t i);
static void flood_fill_push_neighbors(const iwt_runtime_t rt, const iwt_config_t cfg, size_t i, size_t* stack, size_t* stack_ptr);
static void flood_fill(const iwt_runtime_t rt, const iwt_config_t cfg, size_t idx, iwt_cluster_t c);

static void detect_reset_visited(const iwt_runtime_t rt, const iwt_config_t cfg);
static void detect_reset_clusters(const iwt_runtime_t rt);
static bool detect_should_start_cluster(const iwt_runtime_t rt, size_t i);
static void detect_init_cluster(iwt_cluster_t c, size_t id);
static bool detect_finalize_cluster(const iwt_runtime_t rt, iwt_cluster_t c);

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
	if (rt->mass[i] < 1e-6)
	{
		return false;
	}
	return true;
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
	const bool* row = &rt->adjacency[i * cfg->N];
	for (size_t j = 0; j < cfg->N; j++)
	{
		if (row[j] && !rt->visited[j] && rt->mass[j] > 1e-6)
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

	printf("Gefundene Cluster: %d\n", rt->cluster_count);
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
	if (rt->mass[i] < 1e-6)
	{
		return false;
	}
	return true;
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
	return true;
}
