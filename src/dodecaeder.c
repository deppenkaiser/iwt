// ================================================================
// DODEKAEDER-NETZWERK GENERIEREN
// ================================================================

#include "dodecaeder.h"
#include "iwt.h"
#include <math.h>
#include <stdlib.h>
#include <memory.h>

// ================================================================
// KONSTANTEN (compile-time)
// ================================================================

#define PHI 1.6180339887498948482
#define DODECA_N 20
#define DODECA_EDGES 30

// 20 Ecken des Dodekaeders (normiert auf Kantenlänge 1)
// PHI ist jetzt ein Makro – compile-time konstant
static const double dodeca_vertices[DODECA_N][3] = {
    { 1,  1,  1}, { 1,  1, -1}, { 1, -1,  1}, { 1, -1, -1},
    {-1,  1,  1}, {-1,  1, -1}, {-1, -1,  1}, {-1, -1, -1},
    { PHI,  1/PHI, 0}, { PHI, -1/PHI, 0}, {-PHI,  1/PHI, 0}, {-PHI, -1/PHI, 0},
    {0,  PHI,  1/PHI}, {0,  PHI, -1/PHI}, {0, -PHI,  1/PHI}, {0, -PHI, -1/PHI},
    { 1/PHI, 0,  PHI}, { 1/PHI, 0, -PHI}, {-1/PHI, 0,  PHI}, {-1/PHI, 0, -PHI}
};

// 30 Kanten des Dodekaeders
static const int dodeca_edges[DODECA_EDGES][2] = {
    {0,1}, {0,2}, {0,4}, {0,8}, {0,12},
    {1,3}, {1,5}, {1,9}, {1,13},
    {2,3}, {2,6}, {2,10}, {2,14},
    {3,7}, {3,11}, {3,15},
    {4,5}, {4,6}, {4,16}, {4,17},
    {5,7}, {5,18}, {5,19},
    {6,7}, {6,11}, {6,16},
    {8,9}, {8,12}, {8,13},
    {9,10}
};

// ================================================================
// 1. BASIS-DODEKAEDER INITIALISIEREN
// ================================================================

void dodeca_init_base(dodeca_graph_t g, double D)
{
    g->D = D;
    g->node_count = DODECA_N;
    g->max_nodes = 4096;
    g->nodes = malloc(g->max_nodes * sizeof(struct dodeca_node));
    g->K = malloc(g->max_nodes * g->max_nodes * sizeof(double));

    for (int i = 0; i < DODECA_N; i++)
    {
        g->nodes[i].x = dodeca_vertices[i][0];
        g->nodes[i].y = dodeca_vertices[i][1];
        g->nodes[i].z = dodeca_vertices[i][2];
        g->nodes[i].level = 0;
        g->nodes[i].index = i;
    }
}

// ================================================================
// 2. KANTEN SUBDIVIDIEREN (einen neuen Knoten in der Mitte einer Kante)
// ================================================================

int dodeca_subdivide_edge(dodeca_graph_t g, int a, int b)
{
    if (g->node_count >= g->max_nodes) return -1;

    int idx = g->node_count;
    g->nodes[idx].x = (g->nodes[a].x + g->nodes[b].x) / 2.0;
    g->nodes[idx].y = (g->nodes[a].y + g->nodes[b].y) / 2.0;
    g->nodes[idx].z = (g->nodes[a].z + g->nodes[b].z) / 2.0;
    g->nodes[idx].level = g->nodes[a].level + 1;
    g->nodes[idx].index = idx;
    g->node_count++;

    return idx;
}

// ================================================================
// 3. NETZWERK AUF 4096 KNOTEN ERWEITERN (iterative Subdivision)
// ================================================================

void dodeca_generate_network(dodeca_graph_t g)
{
    // Aktuelle Kantenliste
    int* edges = malloc(2 * DODECA_EDGES * sizeof(int));
    int edge_count = DODECA_EDGES;

    for (int i = 0; i < DODECA_EDGES; i++)
    {
        edges[2*i] = dodeca_edges[i][0];
        edges[2*i+1] = dodeca_edges[i][1];
    }

    // Solange Knoten < 4096, Kanten weiter teilen
    while (g->node_count < g->max_nodes)
    {
        int new_edge_count = 0;
        int* new_edges = malloc(4 * edge_count * sizeof(int));

        for (int i = 0; i < edge_count; i++)
        {
            int a = edges[2*i];
            int b = edges[2*i+1];

            // Neuen Knoten in der Mitte der Kante einfügen
            int mid = dodeca_subdivide_edge(g, a, b);
            if (mid == -1) break;

            // Neue Kanten: a-mid, mid-b
            new_edges[4*new_edge_count] = a;
            new_edges[4*new_edge_count+1] = mid;
            new_edges[4*new_edge_count+2] = mid;
            new_edges[4*new_edge_count+3] = b;
            new_edge_count++;
        }

        free(edges);
        edges = new_edges;
        edge_count = new_edge_count;
    }

    free(edges);
}

// ================================================================
// 4. KOPPLUNGSMATRIX BERECHNEN (fraktal)
// ================================================================

void dodeca_compute_kopplung(dodeca_graph_t g)
{
    double D = g->D;
    double alpha = 3.0 - D;

    for (int i = 0; i < g->node_count; i++)
    {
        for (int j = 0; j < g->node_count; j++)
        {
            if (i == j)
            {
                g->K[i * g->node_count + j] = 0.0;
                continue;
            }

            // Euklidische Distanz im 3D-Raum
            double dx = g->nodes[i].x - g->nodes[j].x;
            double dy = g->nodes[i].y - g->nodes[j].y;
            double dz = g->nodes[i].z - g->nodes[j].z;
            double dist = sqrt(dx*dx + dy*dy + dz*dz);

            // Fraktale Distanz (Skalierung mit Level)
            double d_ij = pow(dist, 1.0 / D);

            // Kopplung
            g->K[i * g->node_count + j] = 1.0 / pow(d_ij, alpha);
        }
    }
}

// ================================================================
// 5. DIE KOPPLUNGSMATRIX IN DIE IWT-STRUKTUR ÜBERNEHMEN
// ================================================================

void dodeca_apply_to_iwt(dodeca_graph_t g, iwt_runtime_t rt, iwt_config_t cfg)
{
    // Speicher für die IWT-Kopplungsmatrix freigeben
    free(rt->K);

    // Neue Kopplungsmatrix allokieren
    rt->K = malloc(g->node_count * g->node_count * sizeof(double));
    memcpy(rt->K, g->K, g->node_count * g->node_count * sizeof(double));

    // IWT-Konfiguration anpassen
    cfg->N = g->node_count;
    cfg->D = g->D;

    // IWT-Felder neu allokieren (weil N sich geändert hat)
    free(rt->I_real);
    free(rt->I_imag);
    // ... alle anderen Felder ebenfalls neu allokieren ...
}