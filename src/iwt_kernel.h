#ifndef IWT_KERNEL_H
#define IWT_KERNEL_H

#include <stdbool.h>
#include "iwt.h"

bool run_simulation(const iwt_runtime_t rt, const iwt_config_t cfg);

// Fluktuationen (Anhang P)
bool generate_uncertainty_cpu(const iwt_runtime_t rt, const iwt_config_t cfg);
bool upload_uncertainty_to_gpu(const iwt_runtime_t rt, const iwt_config_t cfg);
bool run_apply_fluctuations(const iwt_runtime_t rt, const iwt_config_t cfg);

// Kern-Funktionen
bool run_flux_calculation(const iwt_runtime_t rt, const iwt_config_t cfg);
bool run_q_calculation(const iwt_runtime_t rt, const iwt_config_t cfg);
bool run_update_info(const iwt_runtime_t rt, const iwt_config_t cfg);
bool run_compute_mass_charge(const iwt_runtime_t rt, const iwt_config_t cfg);

// Rotverschiebung als Energiesenke (Anhang Q)
bool run_apply_redshift_damping(const iwt_runtime_t rt, const iwt_config_t cfg);

#endif