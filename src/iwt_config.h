#ifndef IWT_CONFIG_H
#define IWT_CONFIG_H

#include <stddef.h>

typedef struct {
    // System
    size_t N;               // Gesamtknoten (z.B. 4096)
    size_t BATCH_SIZE;      // Knoten pro Batch (z.B. 512)
    size_t LOCAL_SIZE;      // OpenCL Workgroup-Größe (z.B. 64)
    size_t NUM_BATCHES;     // = N / BATCH_SIZE (wird berechnet)

    // Physik (IWT)
    double DT;              // Zeitschritt
    double ETA;             // Lernrate Kopplung
    double LAMBDA;          // Dämpfung Kopplung
    double THRESHOLD;       // Konvergenz max|Q|
    int MAX_ITER;           // Maximale Iterationen

    // Initialisierung
    double I_MIN;           // Min. Startwert I
    double I_MAX;           // Max. Startwert I
    double K_MIN;           // Min. Startwert K
    double K_MAX;           // Max. Startwert K
    double I_MIN_GLOBAL;    // Minimale Information (I_min > 0)

    // Ausgabe
    int VERBOSE;            // 0, 1, 2
    int SAVE_INTERMEDIATE;  // 0/1
} iwt_config_t;

// Gibt eine Standard-Konfiguration zurück
iwt_config_t iwt_config_default(void);

#endif // IWT_CONFIG_H