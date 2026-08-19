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
    GtkWidget* spin_cluster_threshold;

    struct iwt_config cfg;
    struct iwt_runtime rt;

    float* node_colors;
    float* points_buffer;
    float* cluster_points_buffer;

    GLuint gl_program;
    GLint gl_u_mvp;
    GLint gl_u_size_scale;
    GLuint gl_vao;
    GLuint gl_vbo;
    GLuint gl_vao_clusters;
    GLuint gl_vbo_clusters;

    int iter;
} *iwt_gui_data_t;
