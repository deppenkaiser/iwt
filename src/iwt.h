// ============================================================================
// iwt.h
// ============================================================================

#ifndef IWT_H
#define IWT_H

#include <ocl/ocl.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <vector/vector.h>

// Maximale Nachbaranzahl pro Knoten in der komprimierten Liste
#define IWT_ADJ_STRIDE 64

// Entsprechend fuer das erweiterte Wellen-Kantennetz (schwerer Schweif)
#define IWT_WAVE_STRIDE 32

typedef struct iwt_cluster
{
	int id;
	int type; // 0 = Vakuum, 1 = Materie, 2 = Antimaterie, ...

	// Position (Schwerpunkt) – vector Bibliothek
	struct vector_3d pos;

	// Physikalische Eigenschaften
	double mass;
	double charge;
	double phase;

	// Geschwindigkeit – vector Bibliothek
	struct vector_3d vel;

	// Persistenter Impulsanteil aus Weber-Kräften (wird über Frames integriert)
	struct vector_3d vel_weber;

	// Drift-Offset relativ zum knotenverankerten Schwerpunkt (quasi-teilchenhaft)
	struct vector_3d pos_offset;

	// Knotenliste (für die Phasenverschiebung)
	size_t node_count;
	size_t* node_indices;

	// Externer Cluster: Mitglieder aus >= 2 verschiedenen Zellen
	bool external;

	bool is_active;
}* iwt_cluster_t;

typedef struct iwt_spectrum
{
	size_t count_vacuum;
	size_t count_electron;
	size_t count_proton;
	size_t count_u_quark;
	size_t count_d_quark;
	size_t count_other;
} iwt_spectrum_t;

typedef struct iwt_runtime
{
	uint32_t cluster_capacity;
	uint32_t cluster_count;
	iwt_cluster_t clusters;

	// Cluster des vorherigen Schritts (Persistence/Tracking)
	iwt_cluster_t clusters_prev;
	uint32_t cluster_count_prev;

	bool* visited;

	// Geglättete Masse für stabile Detektion (EMA) + Hysterese-Flag
	double* mass_smooth;
	bool* was_member;

	// Anzahl ausgeführter Simulationsschritte (Seed-Derivation pro Frame)
	uint64_t n_steps;

	double* I_real;
	double* I_imag;
	double* I_prev_real;
	double* I_prev_imag;
	double* I_phase;
	double* I_phase_prev;
	double* K;
	double* sumJ;
	double* Q;
	double* xi_real;
	double* xi_imag;
	double* uncertainty;
	double* mass;
	double* charge;

	// Dodekaeder-Knotenpositionen (3D) – vector Bibliothek
	struct vector_3d* pos;

	// Zell-ID je Knoten (BFS-Reihenfolge des Generators)
	int* node_cell;

	// Klassifikation der aktuellen Cluster (groessenbasiertes Spektrum)
	iwt_spectrum_t spectrum;

	// Komprimierte Nachbarschaftslisten (Performance): adj_flat[i*STRIDE..]
	// enthaelt die Knotenindizes der Nachbarn von i, adj_count[i] deren Zahl.
	int* adj_flat;
	int* adj_count;

	// Erweitertes Netz fuer das Wellen-Overlay: Fern-Kopplungen
	// (K > wave_k_min), je Knoten auf die naechsten begrenzt.
	int* wave_flat;
	int* wave_count;

	// Nachbarschafts-Adjazenz (K-Matrix-Schwellwert), N*N, statisch
	bool* adjacency;

	cl_mem mass_gpu;
	cl_mem charge_gpu;
	cl_mem I_real_gpu;
	cl_mem I_imag_gpu;
	cl_mem I_prev_real_gpu;
	cl_mem I_prev_imag_gpu;
	cl_mem I_phase_gpu;
	cl_mem I_phase_prev_gpu;
	cl_mem K_gpu;
	cl_mem sumJ_gpu;
	cl_mem Q_gpu;
	cl_mem xi_real_gpu;
	cl_mem xi_imag_gpu;
	cl_mem uncertainty_gpu;

	struct ocl_core ocl;
}* iwt_runtime_t;

typedef struct iwt_config
{
	size_t N;
	double gamma;
	double beta;
	double D;
	double l0;
	double T;
	double DT;
	double hbar;
	unsigned int seed;
	bool enable_motion;
	double cluster_threshold;

	// Advektions-Kopplungsstärke (Transport entlang des Phasengradienten)
	double kappa;

	// Effektiver Zeitschritt des Bohm-Phasen-Kicks (numerische Quantenrate)
	double phase_dt;

	// EM-Wellen-Overlay in der Visualisierung anzeigen
	bool show_waves;

	// 2D-Schnittmodus: nur Konturen nahe z = slice_pos, Knoten außerhalb gedimmt
	bool slice_mode;
	double slice_pos;

	// Halbe Dicke der Schnittscheibe
	double slice_delta;

	// Fraktale Selbstähnlichkeits-Stufen (0 = einzelner Dodekaeder)
	int extra_levels;

	// Kopplungs-Schwelle des erweiterten Wellen-Kantennetzes
	double wave_k_min;
}* iwt_config_t;

double iwt_pi(void);
double iwt_fundamental_length(void);
double iwt_fundamental_time(void);
double iwt_fractal_dimension(void);
double iwt_alpha_IWT(void);
double iwt_beta_IWT(void);

void iwt_compute_node_colors(const double* mass, const double* charge, size_t N, float* out_rgb);

#endif