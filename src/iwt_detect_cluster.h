#pragma once

#include "iwt.h"

void iwt_detect_clusters(const iwt_runtime_t rt, const iwt_config_t cfg);

// Formatiert Spektrum + Groessen-Histogramm der aktuellen Cluster
// (mehrzeilig, monospace-geeignet) in den Puffer.
void iwt_format_stats(const iwt_runtime_t rt, char* buf, size_t buflen);
