#pragma once

#include "iwt.h"

bool initialize_host_data(const iwt_runtime_t rt, const iwt_config_t cfg);
void deinitialize_host_data(const iwt_runtime_t rt);
bool initialize_gpu_data(const iwt_runtime_t rt, const iwt_config_t cfg);
void deinitialize_gpu_data(const iwt_runtime_t rt);

// Berechnet rt->adjacency aus der (bereits vorhandenen) K-Matrix und
// cfg->cluster_threshold neu. Positionen/K sind statisch, nur der
// Schwellwert kann sich (z.B. live per Spinbox) ändern.
void iwt_recompute_adjacency(const iwt_runtime_t rt, const iwt_config_t cfg);

// Baut die Geometrie komplett neu auf (Positionen, K-Matrix, Adjazenz)
// und lädt K erneut auf die GPU. Wird bei Änderung von extra_levels
// zur Laufzeit benötigt.
bool iwt_rebuild_geometry(const iwt_runtime_t rt, const iwt_config_t cfg);
