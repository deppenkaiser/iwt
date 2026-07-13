#ifndef IWT_KERNEL_H
#define IWT_KERNEL_H

#include <stdbool.h>
#include <ocl/ocl.h>
#include "iwt.h"

bool run_flux_calculation(const iwt_runtime_t rt, const iwt_config_t cfg);

#endif
