// ============================================================================
// iwt_kernel.h - IWT Kernel-Funktionen (Entwicklungsversion)
// ============================================================================
//
// Enthält:
//   - Bohm-Potential
//   - Fluss (Weber-Kern)
//   - Kontinuität
//   - Masse & Ladung
//   - Hauptsimulation
//
// Die eingefrorenen Funktionen werden aus iwt_kernel_frozen.h importiert.
//
// ============================================================================

#ifndef IWT_KERNEL_H
#define IWT_KERNEL_H

#include "iwt.h"
#include <stdbool.h>

// ============================================================================
// BOHM-POTENTIAL (Strukturbildung)
// ============================================================================

bool run_q_calculation(const iwt_runtime_t rt, const iwt_config_t cfg);

// ============================================================================
// FLUSS (Transport)
// ============================================================================

bool run_flux_calculation(const iwt_runtime_t rt, const iwt_config_t cfg);

// ============================================================================
// KONTINUITÄT (Erhaltung)
// ============================================================================

bool run_update_info(const iwt_runtime_t rt, const iwt_config_t cfg);

// ============================================================================
// MASSE UND LADUNG
// ============================================================================

bool run_compute_mass_charge(const iwt_runtime_t rt, const iwt_config_t cfg);

// ============================================================================
// EIN EINZELNER SIMULATIONSSCHRITT (für Live-Anzeige)
// ============================================================================

bool run_simulation_step(const iwt_runtime_t rt, const iwt_config_t cfg);

// ============================================================================
// HAUPTSIMULATION
// ============================================================================

bool run_simulation(const iwt_runtime_t rt, const iwt_config_t cfg);

#endif // IWT_KERNEL_H