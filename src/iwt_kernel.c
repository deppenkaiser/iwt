#include "iwt_kernel.h"
#include <stdio.h>

bool iwt_kernel_flux(iwt_system_t *sys, size_t batch_start, size_t batch_end) {
    // TODO: ocl-Aufruf für Flussberechnung
    fprintf(stderr, "WARNUNG: iwt_kernel_flux noch nicht implementiert.\n");
    return false;
}

bool iwt_kernel_q(iwt_system_t *sys, size_t batch_start, size_t batch_end) {
    fprintf(stderr, "WARNUNG: iwt_kernel_q noch nicht implementiert.\n");
    return false;
}

bool iwt_kernel_update_info(iwt_system_t *sys, size_t batch_start, size_t batch_end) {
    fprintf(stderr, "WARNUNG: iwt_kernel_update_info noch nicht implementiert.\n");
    return false;
}

bool iwt_kernel_update_coupling(iwt_system_t *sys, size_t batch_start, size_t batch_end) {
    fprintf(stderr, "WARNUNG: iwt_kernel_update_coupling noch nicht implementiert.\n");
    return false;
}

bool iwt_kernel_metric(iwt_system_t *sys, size_t batch_start, size_t batch_end) {
    fprintf(stderr, "WARNUNG: iwt_kernel_metric noch nicht implementiert.\n");
    return false;
}