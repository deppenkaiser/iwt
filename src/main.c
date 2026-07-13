#include "iwt.h"
#include "iwt_io.h"
#include "iwt_batch.h"   // <-- HIER FEHLTE DAS!
#include "iwt_init.h"
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    iwt_system_t sys = {0};

    // 1. Konfiguration laden
    sys.cfg = iwt_config_default();

    // 2. System initialisieren (Speicher + Daten)
    if (!iwt_init(&sys)) {
        fprintf(stderr, "Fehler: System-Initialisierung fehlgeschlagen.\n");
        return 1;
    }

    // 3. Konfiguration ausgeben
    iwt_print_config(&sys.cfg);

    // 4. Initiale Summe speichern
    double sum_i_initial = iwt_sum_i(&sys);

    // 5. Hauptschleife
    int iter;
    for (iter = 0; iter < sys.cfg.MAX_ITER; iter++) {
        double max_q = 0.0;

        // Alle Batches verarbeiten
        for (size_t batch = 0; batch < sys.cfg.NUM_BATCHES; batch++) {
            double batch_max_q = iwt_process_batch(&sys, batch);
            if (batch_max_q > max_q) max_q = batch_max_q;
        }

        // Fortschritt ausgeben
        double sum_i = iwt_sum_i(&sys);
        iwt_print_progress(iter, max_q, sum_i, sum_i_initial);

        // Konvergenzprüfung
        if (max_q < sys.cfg.THRESHOLD) {
            printf("\nKonvergenz erreicht nach %d Iterationen.\n", iter + 1);
            break;
        }
    }

    // 6. Endergebnisse ausgeben
    iwt_print_results(&sys);

    // 7. Aufräumen
    iwt_cleanup(&sys);

    return 0;
}