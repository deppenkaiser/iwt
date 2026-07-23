#ifndef IWT_H
#define IWT_H

#include <stddef.h>
#include <stdbool.h>
#include <ocl/ocl.h>

typedef struct iwt_runtime
{
    // === Bestehende Felder ===
    double *I_real;      // EHEMALS I – jetzt Realteil von I
    double *I_imag;      // NEU – Imaginärteil von I
    double *I_prev_real; // EHEMALS I_prev – Realteil
    double *I_prev_imag; // NEU – Imaginärteil
    double *I_phase;     // Phase (bleibt)
    double *I_phase_prev;// Phase (bleibt)
    double *K;
    double *sumJ;
    double *Q;
    double *R;
    double *T;
    double *xi_real;
    double *xi_imag;
    double *uncertainty;

    // GPU-Buffer
    cl_mem I_real_gpu;
    cl_mem I_imag_gpu;
    cl_mem I_prev_real_gpu;
    cl_mem I_prev_imag_gpu;
    cl_mem I_phase_gpu;
    cl_mem I_phase_prev_gpu;
    cl_mem K_gpu;
    cl_mem sumJ_gpu;
    cl_mem Q_gpu;
    cl_mem R_gpu;
    cl_mem T_gpu;
    cl_mem xi_real_gpu;
    cl_mem xi_imag_gpu;
    cl_mem uncertainty_gpu;

    struct ocl_core ocl;
} *iwt_runtime_t;

typedef struct iwt_config
{
    // === Bestehende Felder ===
    size_t N;
    size_t BATCH_SIZE;
    double D;
    double l0;
    double T;
    double alpha_IWT;
    double beta_IWT;
    double DT;
    double I_min;
    double I_max;
    int MAX_ITER;
    double I_vac;
    double phi_0;
    double omega_0;
    double Z_0;
    double alpha_Z;

    // === NEU: Parameter für Quantenfluktuationen (Anhang O & P) ===
    double hbar;                // Plancksches Wirkungsquantum (in simulierten Einheiten)
    double uncertainty_scale;   // √(ℏ/(2·T)) – Skalierung der Fluktuationen
    bool enable_fluctuations;   // Fluktuationen aktivieren/deaktivieren
    unsigned int seed;          // Seed für Zufallsgenerator
} *iwt_config_t;

// === Restliche Deklarationen bleiben unverändert ===
typedef struct iwt_spectrum
{
    size_t count_vacuum;
    size_t count_electron;
    size_t count_proton;
    size_t count_u_quark;
    size_t count_d_quark;
    size_t count_other;
} *iwt_spectrum_t;

typedef struct iwt_mds
{
    double* coords;
    double* eigenvalues;
    size_t dim;
    size_t N;
} *iwt_mds_t;

// === Funktionsdeklarationen (bestehend + neue) ===
bool iwt_mds_compute(const iwt_runtime_t rt, const iwt_config_t cfg, iwt_mds_t mds);
void iwt_mds_free(iwt_mds_t mds);
void iwt_mds_print(const iwt_mds_t mds, size_t n);
void iwt_mds_save_pgm(const iwt_mds_t mds, const double* I, size_t N, const char* filename);

double iwt_pi(void);
double iwt_fundamental_length(void);
double iwt_fundamental_time(void);
double iwt_fractal_dimension(void);
double iwt_I_min(void);
double iwt_I_max(void);
double iwt_delta_I(void);
double iwt_alpha_IWT(void);
double iwt_beta_IWT(void);

void iwt_compute_spectrum(const double* I_real, const double* I_imag, size_t N, struct iwt_spectrum* spec);
int iwt_classify(double Re, double Im);
void iwt_print_spectrum(const struct iwt_spectrum* spec);
bool iwt_save_state(const iwt_runtime_t rt, const iwt_config_t cfg, const char* filename);
bool iwt_load_state(const iwt_runtime_t rt, const iwt_config_t cfg, const char* filename);
void iwt_print_border(void);
void iwt_print_status(const iwt_runtime_t rt, const iwt_config_t cfg, int iter,
    double max_q, double I_total, double I_min, double I_max,
    double deviation, double sum_I_sq, double info_deviation);
#endif
