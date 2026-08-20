// ============================================================================
// iwt_kernel_frozen.h - Eingefrorene IWT-Kernfunktionen
// ============================================================================
//
// Status: Eingefroren am 27.07.2026
//
// Enthält NUR:
//   1. Fluktuationen (Vakuumfluktuation)
//   2. Energiesenke (Materie-Zerstrahlung)
//
// Diese Funktionen werden NICHT mehr geändert.
//
// ============================================================================

#ifndef IWT_KERNEL_FROZEN_H
#define IWT_KERNEL_FROZEN_H

#include "iwt.h"
#include <stdbool.h>

// ============================================================================
// 1. FLUKTUATIONEN (Vakuumfluktuation)
// ============================================================================

bool frozen_generate_uncertainty_cpu(const iwt_runtime_t rt, const iwt_config_t cfg);
bool frozen_upload_uncertainty_to_gpu(const iwt_runtime_t rt, const iwt_config_t cfg);
bool frozen_run_apply_fluctuations(const iwt_runtime_t rt, const iwt_config_t cfg);

// ============================================================================
// 2. ENERGIESSENKE (Materie-Zerstrahlung)
// ============================================================================

bool frozen_run_apply_redshift_damping(const iwt_runtime_t rt, const iwt_config_t cfg);

#endif // IWT_KERNEL_FROZEN_H