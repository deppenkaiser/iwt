#ifndef IWT_KERNEL_H
#define IWT_KERNEL_H

#include "iwt.h"

// Diese Funktionen rufen die ocl-Bibliothek auf.
// Sie werden später implementiert, sobald die Kernel geschrieben sind.

// Berechnet für einen Batch:
// sumJ_i = sum_j K_ij * (I_i - I_j)
bool iwt_kernel_flux(
    iwt_system_t *sys,
    size_t batch_start,
    size_t batch_end
);

// Berechnet Q_i = sum_j J_ji - sum_j J_ij
bool iwt_kernel_q(
    iwt_system_t *sys,
    size_t batch_start,
    size_t batch_end
);

// Aktualisiert I_i = I_i + sumJ_i * DT
bool iwt_kernel_update_info(
    iwt_system_t *sys,
    size_t batch_start,
    size_t batch_end
);

// Aktualisiert K_ij
bool iwt_kernel_update_coupling(
    iwt_system_t *sys,
    size_t batch_start,
    size_t batch_end
);

// Berechnet g_ij (optional)
bool iwt_kernel_metric(
    iwt_system_t *sys,
    size_t batch_start,
    size_t batch_end
);

#endif