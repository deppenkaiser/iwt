#include "iwt_detect_cluster.h"

#include <api/api.h>
#include <stdio.h>
#include <memory.h>

private void _flood_fill(const iwt_runtime_t rt, const iwt_config_t cfg, size_t idx, iwt_cluster_t c)
{
	size_t stack[cfg->N]; // VLA - geht weil cfg->N bekannt ist
	size_t stack_ptr = 0;
	stack[stack_ptr++] = idx;

	while (stack_ptr > 0)
	{
		size_t i = stack[--stack_ptr];
		if (rt->visited[i])
		{
			continue;
		}
		if (rt->mass[i] < 1e-6)
		{
			continue;
		}

		rt->visited[i] = true;
		c->node_indices[c->node_count++] = i;

		c->mass += rt->mass[i];
		// Schwerpunkt akkumulieren mit Vector-Bibliothek
		struct vector_3d weighted = vector_multiply_scalar(&rt->pos[i], (cld) rt->mass[i]);
		c->pos = vector_add(&c->pos, &weighted);
		c->charge += rt->charge[i];
		c->phase += rt->I_phase[i];

		const bool* row = &rt->adjacency[i * cfg->N];
		for (size_t j = 0; j < cfg->N; j++)
		{
			if (row[j] && !rt->visited[j] && rt->mass[j] > 1e-6)
			{
				if (stack_ptr >= cfg->N)
				{
					break;
				}
				stack[stack_ptr++] = j;
			}
		}
	}
}

void iwt_detect_clusters(const iwt_runtime_t rt, const iwt_config_t cfg)
{
	// Alle Knoten als unbesucht markieren
	memset(rt->visited, 0, cfg->N * sizeof(bool));

	// Alte Cluster zurücksetzen
	rt->cluster_count = 0;

	// Flood-Fill für jeden Knoten
	for (size_t i = 0; i < cfg->N; i++)
	{
		if (rt->visited[i])
		{
			continue;
		}
		if (rt->mass[i] < 1e-6)
		{
			continue; // Schwellwert für Masse
		}

		// Neuen Cluster initialisieren
		if (rt->cluster_count >= rt->cluster_capacity)
		{
			break;
		}
		iwt_cluster_t c = &rt->clusters[rt->cluster_count];
		c->id = rt->cluster_count;
		c->node_count = 0;
		c->mass = 0.0;
		c->charge = 0.0;
		c->phase = 0.0;
		vector_clear(&c->pos);
		vector_clear(&c->vel);
		c->is_active = true;

		// Flood-Fill: Alle verbundenen Knoten sammeln
		_flood_fill(rt, cfg, i, c);

		// Schwerpunkt berechnen (gewichtet mit Masse)
		if (c->node_count > 0)
		{
			c->pos = vector_divide_scalar(&c->pos, (cld) c->mass);
			c->phase /= c->node_count;
			rt->cluster_count++;
		}
	}

	printf("Gefundene Cluster: %d\n", rt->cluster_count);
}
