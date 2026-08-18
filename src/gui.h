#pragma once

#include <gui/gui.h>
#include "iwt.h"

typedef struct iwt_gui_data
{
    GtkWidget* gl_area;
    GtkWidget* window;

    GtkWidget* toggle_motion;
    GtkWidget* spin_beta;
    GtkWidget* spin_gamma;

    struct iwt_config cfg;
    struct iwt_runtime rt;

    float* node_colors;
    float* points_buffer;

    GLuint gl_program;
    GLuint gl_vao;
    GLuint gl_vbo;

    int iter;
} *iwt_gui_data_t;
