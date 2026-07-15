#pragma once

#include "iwt.h"

bool initialize_host_data(const iwt_runtime_t rt, const iwt_config_t cfg);
void deinitialize_host_data(const iwt_runtime_t rt);
bool initialize_gpu_data(const iwt_runtime_t rt, const iwt_config_t cfg);
void deinitialize_gpu_data(const iwt_runtime_t rt);
