#ifndef IWT_KERNEL_H
#define IWT_KERNEL_H

#include <stdbool.h>
#include "iwt.h"

bool run_flux_calculation_batched(const iwt_runtime_t rt, const iwt_config_t cfg);

#endif
