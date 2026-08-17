#pragma once

// ================================================================
// STRUKTUR FÜR DEN DODEKAEDER-GRAPHEN
// ================================================================

typedef struct dodeca_node
{
    double x, y, z;          // 3D-Koordinaten
    int level;               // Subdivision-Level
    int index;               // Index im globalen Knoten-Array
} *dodeca_node_t;

typedef struct dodeca_graph
{
    dodeca_node_t nodes;
    int node_count;
    int max_nodes;
    double* K;               // Kopplungsmatrix
    double D;                // Fraktale Dimension
} *dodeca_graph_t;

void dodeca_init_base(dodeca_graph_t g, double D);
void dodeca_generate_network(dodeca_graph_t g);
void dodeca_compute_kopplung(dodeca_graph_t g);
