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
	GtkWidget* toggle_2d_mode;
	GtkWidget* spin_projection_plane;
	GtkTextBuffer* stats_buffer;

	struct iwt_config cfg;
	struct iwt_runtime rt;

	float* node_colors;
	float* points_buffer;
	float* cluster_points_buffer;

	// EM-Wellenfront-Visualisierung (Aequipotenzial-Polylinien)
	float* wave_buffer;
	size_t wave_segment_count;

	// GPU-Wellen: temporäre Positions-Arrays (einmal allokiert)
	double* wave_pos_x;
	double* wave_pos_y;
	double* wave_pos_z;

	// GPU-Wellen: Host-Spiegel der Crossing-Counts für Prefix-Sum
	int* wave_counts_host;

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

	// Analyse: Histogramm im naechsten Render-Durchlauf erzeugen (h)
	bool request_histogram;

	// Analyse: automatischer Screenshot nach N Sekunden (--auto-shot, 0 = aus)
	int auto_shot_delay_s;

	// Analyse: letzte Cluster-Anzahl fuer automatischen Trigger
	uint32_t last_cluster_count;

	// Performance: Stats-Text-Update throttling (jede N. Frame)
	int stats_update_counter;

	// Performance: FPS-Messung (exponentiell geglaettet)
	double fps_smooth;
	double fps_last_time;

	// 2D/3D Modus
	bool mode_2d;
	int projection_plane; // 0=XY, 1=XZ, 2=YZ

	// Throttling: Adjacency Recompute (debounced)
	guint adj_recompute_source;   // GLib-Timeout-ID für Debounce
	bool adj_recompute_pending;   // Neuberechnung ausstehend
}* iwt_gui_data_t;
