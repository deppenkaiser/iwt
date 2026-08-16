#pragma once

#include <gui/gui.h>
#include "iwt.h"

typedef struct iwt_gui_data
{
    GtkWidget* gl_area;
    GtkWidget* window;

    struct iwt_config cfg;
    struct iwt_runtime rt;
} *iwt_gui_data_t;
