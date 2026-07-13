#include "iwt_config.h"

iwt_config_t iwt_config_default(void) {
    iwt_config_t cfg = {
        // System
        .N = 4096,
        .BATCH_SIZE = 512,
        .LOCAL_SIZE = 64,
        .NUM_BATCHES = 0,   // wird unten berechnet

        // Physik
        .DT = 0.01,
        .ETA = 0.001,
        .LAMBDA = 0.1,
        .THRESHOLD = 1e-6,
        .MAX_ITER = 100,

        // Initialisierung
        .I_MIN = 0.5,
        .I_MAX = 1.5,
        .K_MIN = 0.0,
        .K_MAX = 1.0,
        .I_MIN_GLOBAL = 0.1,

        // Ausgabe
        .VERBOSE = 1,
        .SAVE_INTERMEDIATE = 0
    };
    // NUM_BATCHES berechnen
    cfg.NUM_BATCHES = cfg.N / cfg.BATCH_SIZE;
    return cfg;
}