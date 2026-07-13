#ifndef IWT_IO_H
#define IWT_IO_H

#include "iwt.h"

// Gibt die Konfiguration aus
void iwt_print_config(const iwt_config_t *cfg);

// Gibt den Fortschritt einer Iteration aus
void iwt_print_progress(
    int iteration,
    double max_q,
    double sum_i,
    double sum_i_initial
);

// Gibt die finalen Ergebnisse aus
void iwt_print_results(const iwt_system_t *sys);

// Hilfsfunktionen
double iwt_max_q(const iwt_system_t *sys);
double iwt_sum_i(const iwt_system_t *sys);

#endif // IWT_IO_H