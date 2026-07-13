#include "iwt_batch.h"
#include "iwt_kernel.h"
#include <math.h>
#include <stdio.h>

double iwt_process_batch(
    iwt_system_t *sys,
    size_t batch_id
) {
    size_t batch_start = batch_id * sys->cfg.BATCH_SIZE;
    size_t batch_end = batch_start + sys->cfg.BATCH_SIZE;
    if (batch_end > sys->cfg.N) batch_end = sys->cfg.N;

    // 1. Flüsse berechnen
    if (!iwt_kernel_flux(sys, batch_start, batch_end)) {
        fprintf(stderr, "Fehler in iwt_kernel_flux für Batch %zu\n", batch_id);
        return -1.0;
    }

    // 2. Q berechnen
    if (!iwt_kernel_q(sys, batch_start, batch_end)) {
        fprintf(stderr, "Fehler in iwt_kernel_q für Batch %zu\n", batch_id);
        return -1.0;
    }

    // 3. Information aktualisieren
    if (!iwt_kernel_update_info(sys, batch_start, batch_end)) {
        fprintf(stderr, "Fehler in iwt_kernel_update_info für Batch %zu\n", batch_id);
        return -1.0;
    }

    // 4. Kopplung aktualisieren
    if (!iwt_kernel_update_coupling(sys, batch_start, batch_end)) {
        fprintf(stderr, "Fehler in iwt_kernel_update_coupling für Batch %zu\n", batch_id);
        return -1.0;
    }

    // 5. Metrik berechnen (optional)
    if (sys->cfg.VERBOSE >= 2) {
        iwt_kernel_metric(sys, batch_start, batch_end);
    }

    // max|Q| für diesen Batch berechnen (auf der CPU)
    double max_q = 0.0;
    for (size_t i = batch_start; i < batch_end; i++) {
        double abs_q = fabs(sys->Q[i]);
        if (abs_q > max_q) max_q = abs_q;
    }
    return max_q;
}