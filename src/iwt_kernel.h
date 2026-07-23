#ifndef IWT_KERNEL_H
#define IWT_KERNEL_H

#include <stdbool.h>
#include "iwt.h"

// === BESTEHENDE FUNKTIONSDEKLARATION ===
bool run_simulation(const iwt_runtime_t rt, const iwt_config_t cfg);

// === NEUE FUNKTIONSDEKLARATIONEN FÜR QUANTENFLUKTUATIONEN ===
// Generiert Gauß'sche Zufallszahlen für die Fluktuationen (CPU-seitig)
bool generate_uncertainty_cpu(const iwt_runtime_t rt, const iwt_config_t cfg);

// Kopiert die generierten Fluktuationen auf die GPU
bool upload_uncertainty_to_gpu(const iwt_runtime_t rt, const iwt_config_t cfg);

#endif