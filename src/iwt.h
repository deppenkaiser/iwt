#ifndef IWT_H
#define IWT_H

#include "iwt_config.h"
#include <stdbool.h>

// Zentrale Systemstruktur
typedef struct {
    iwt_config_t cfg;

    // Daten (Host)
    double *I;          // Informationswerte, Größe N
    double *K;          // Kopplungsmatrix, Größe N * N
    double *Q;          // Quantenpotential, Größe N
    double *sumJ;       // Summe der Flüsse pro Knoten, Größe N

    // OpenCL-Handles (undurchsichtiger Zeiger)
    void *ocl;          // Zeiger auf ocl_core_t (später)
} iwt_system_t;

// Initialisiert das gesamte System
bool iwt_init(iwt_system_t *sys);

// Führt einen Zeitschritt (alle Batches) durch
bool iwt_step(iwt_system_t *sys, double *max_q);

// Gibt das System frei
void iwt_cleanup(iwt_system_t *sys);

#endif