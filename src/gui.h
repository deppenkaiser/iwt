#pragma once

#include "iwt.h"
#include <gui/gui.h>

typedef struct iwt_gui_data
{
	GtkWidget* gl_area;
	GtkWidget* window;

	GtkWidget* toggle_motion;
	GtkWidget* toggle_waves;
	GtkWidget* toggle_slice;
	GtkWidget* spin_beta;
	GtkWidget* spin_gamma;
	GtkWidget* spin_cluster_threshold;
	GtkWidget* spin_slice_pos;
	GtkWidget* spin_slice_delta;
	GtkWidget* spin_wave_k_min;
	GtkWidget* spin_extra_levels;
	GtkTextBuffer* stats_buffer;

	struct iwt_config cfg;
	struct iwt_runtime rt;

	float* node_colors;
	float* points_buffer;
	float* cluster_points_buffer;

	// EM-Wellenfront-Visualisierung (Aequipotenzial-Polylinien)
	float* wave_buffer;
	size_t wave_segment_count;

	GLuint gl_program;
	GLint gl_u_mvp;
	GLint gl_u_size_scale;
	GLuint gl_vao;
	GLuint gl_vbo;
	GLuint gl_vao_clusters;
	GLuint gl_vbo_clusters;
	GLuint gl_vao_wave;
	GLuint gl_vbo_wave;

	float zoom;
	float cam_yaw;
	float cam_pitch;
	double drag_last_x;
	double drag_last_y;

	int iter;

	// Analyse: Screenshot im naechsten Render-Durchlauf sichern (F12)
	bool request_screenshot;
}* iwt_gui_data_t;
