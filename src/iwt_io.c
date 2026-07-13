#include "iwt_io.h"
#include <stdio.h>
#include <math.h>

void iwt_print_config(const iwt_config_t *cfg) {
    printf("=== IWT Konfiguration ===\n");
    printf("N           : %zu\n", cfg->N);
    printf("BATCH_SIZE  : %zu\n", cfg->BATCH_SIZE);
    printf("LOCAL_SIZE  : %zu\n", cfg->LOCAL_SIZE);
    printf("DT          : %f\n", cfg->DT);
    printf("ETA         : %f\n", cfg->ETA);
    printf("LAMBDA      : %f\n", cfg->LAMBDA);
    printf("THRESHOLD   : %e\n", cfg->THRESHOLD);
    printf("MAX_ITER    : %d\n", cfg->MAX_ITER);
    printf("I_MIN       : %f\n", cfg->I_MIN);
    printf("I_MAX       : %f\n", cfg->I_MAX);
    printf("K_MIN       : %f\n", cfg->K_MIN);
    printf("K_MAX       : %f\n", cfg->K_MAX);
    printf("I_MIN_GLOBAL: %f\n", cfg->I_MIN_GLOBAL);
    printf("VERBOSE     : %d\n", cfg->VERBOSE);
    printf("========================\n");
}

void iwt_print_progress(int iteration, double max_q, double sum_i, double sum_i_initial) {
    printf("Iter %3d: max|Q| = %9.2e, sum(I) = %10.6f (delta = %+.3e)\n",
           iteration, max_q, sum_i, sum_i - sum_i_initial);
}

void iwt_print_results(const iwt_system_t *sys) {
    printf("\n=== Endergebnisse ===\n");
    printf("N = %zu\n", sys->cfg.N);
    printf("max|Q| = %e\n", iwt_max_q(sys));
    printf("sum(I)  = %f\n", iwt_sum_i(sys));
    printf("\nErste 10 I-Werte:\n");
    for (size_t i = 0; i < (sys->cfg.N < 10 ? sys->cfg.N : 10); i++) {
        printf("  I[%zu] = %f\n", i, sys->I[i]);
    }
    printf("====================\n");
}

double iwt_max_q(const iwt_system_t *sys) {
    double max = 0.0;
    for (size_t i = 0; i < sys->cfg.N; i++) {
        double abs_q = fabs(sys->Q[i]);
        if (abs_q > max) max = abs_q;
    }
    return max;
}

double iwt_sum_i(const iwt_system_t *sys) {
    double sum = 0.0;
    for (size_t i = 0; i < sys->cfg.N; i++) {
        sum += sys->I[i];
    }
    return sum;
}