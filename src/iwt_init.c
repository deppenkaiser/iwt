#include "iwt_init.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>

// === Speicherallozierung ===
bool iwt_allocate(iwt_system_t *sys) {
    size_t N = sys->cfg.N;
    size_t K_size = N * N;

    sys->I = (double*)malloc(N * sizeof(double));
    sys->Q = (double*)malloc(N * sizeof(double));
    sys->sumJ = (double*)malloc(N * sizeof(double));
    sys->K = (double*)malloc(K_size * sizeof(double));

    if (!sys->I || !sys->Q || !sys->sumJ || !sys->K) {
        fprintf(stderr, "Fehler: Speicherallozierung fehlgeschlagen.\n");
        return false;
    }
    return true;
}

// === Initialisierung der Daten ===
void iwt_initialize_system(iwt_system_t *sys) {
    size_t N = sys->cfg.N;
    size_t K_size = N * N;
    double I_MIN = sys->cfg.I_MIN;
    double I_MAX = sys->cfg.I_MAX;
    double K_MIN = sys->cfg.K_MIN;
    double K_MAX = sys->cfg.K_MAX;

    srand((unsigned int)time(NULL));

    // I initialisieren (gleichverteilt)
    for (size_t i = 0; i < N; i++) {
        double r = (double)rand() / RAND_MAX;
        sys->I[i] = I_MIN + r * (I_MAX - I_MIN);
    }

    // K initialisieren (gleichverteilt, symmetrisch)
    for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < N; j++) {
            double r = (double)rand() / RAND_MAX;
            double val = K_MIN + r * (K_MAX - K_MIN);
            sys->K[i * N + j] = val;
            sys->K[j * N + i] = val; // symmetrisch
        }
    }

    // Q und sumJ auf Null setzen
    memset(sys->Q, 0, N * sizeof(double));
    memset(sys->sumJ, 0, N * sizeof(double));
}

// === Speicher freigeben ===
void iwt_free_system(iwt_system_t *sys) {
    free(sys->I);
    free(sys->Q);
    free(sys->sumJ);
    free(sys->K);
    sys->I = NULL;
    sys->Q = NULL;
    sys->sumJ = NULL;
    sys->K = NULL;
}

// === System initialisieren (öffentliche Funktion) ===
bool iwt_init(iwt_system_t *sys) {
    if (!iwt_allocate(sys)) return false;
    iwt_initialize_system(sys);
    return true;
}

// === System bereinigen (öffentliche Funktion) ===
void iwt_cleanup(iwt_system_t *sys) {
    iwt_free_system(sys);
}