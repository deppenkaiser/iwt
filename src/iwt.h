#ifndef IWT_H
#define IWT_H

#include <stddef.h>
#include <stdbool.h>
#include <ocl/ocl.h>

typedef struct iwt_config
{
    size_t N;
    size_t BATCH_SIZE;
    double D;
    double l0;
    double T;
    double alpha_IWT;
    double beta_IWT;

    double DT;
    double ETA_Q;
    double LAMBDA_Q;
    double GAMMA_Q;
	double GAMMA_ENTROPY;  // Stärke des Entropie-Terms
	double I0;             // Referenzwert für Entropie
	double ETA_Q_POTENTIAL;
	double ETA_SOURCE;
	int PULSE_INTERVAL;

    double I_min;
    double I_max;

    int MAX_ITER;
} *iwt_config_t;

typedef struct iwt_runtime
{
    double *I;
    double *I_prev;
    double *K;
    double *sumJ;
    double *Q;

	cl_mem I_gpu;
    cl_mem I_prev_gpu;
    cl_mem K_gpu;
    cl_mem sumJ_gpu;
    cl_mem Q_gpu;

	struct ocl_core ocl;
} *iwt_runtime_t;

typedef struct iwt_spectrum
{
    size_t count_vacuum;      // I ≈ 0.01
    size_t count_electron;    // I ≈ 0.25
    size_t count_proton;      // I ≈ 1.0
    size_t count_u_quark;     // I ≈ 0.05
    size_t count_d_quark;     // I ≈ 0.06
    size_t count_other;       // alles andere
} *iwt_spectrum_t;

typedef struct iwt_mds
{
    double* coords;      // Koordinaten (N * 2) im euklidischen Raum
    double* eigenvalues; // Eigenwerte der ersten 2 Dimensionen
    size_t dim;          // 2 für Visualisierung
    size_t N;            // Anzahl Knoten
} *iwt_mds_t;

bool iwt_mds_compute(const iwt_runtime_t rt, const iwt_config_t cfg, iwt_mds_t mds);
void iwt_mds_free(iwt_mds_t mds);
void iwt_mds_print(const iwt_mds_t mds, size_t n);
void iwt_mds_save_pgm(const iwt_mds_t mds, const double* I, size_t N, const char* filename);

double iwt_pi(void);
double iwt_fundamental_length(void);
double iwt_fundamental_time();
double iwt_alpha_IWT();
double iwt_fractal_dimension(void);
double iwt_I_min();
double iwt_I_max(void);
double iwt_delta_I(void);
double iwt_alpha_IWT();
double iwt_beta_IWT();
void iwt_compute_spectrum(const double* I, size_t N, struct iwt_spectrum* spec);
void iwt_print_spectrum(const struct iwt_spectrum* spec);
bool iwt_save_state(const iwt_runtime_t rt, const iwt_config_t cfg, const char* filename);
bool iwt_load_state(const iwt_runtime_t rt, const iwt_config_t cfg, const char* filename);

#endif
