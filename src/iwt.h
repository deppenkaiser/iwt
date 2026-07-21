#ifndef IWT_H
#define IWT_H

#include <stddef.h>
#include <stdbool.h>
#include <ocl/ocl.h>

typedef struct iwt_runtime
{
    double *I;
    double *I_prev;
    double *I_phase;
    double *I_phase_prev;
    double *K;
    double *sumJ;
    double *Q;
    double *R;  // NEU: Reflexionskoeffizient
    double *T;  // NEU: Transmissionskoeffizient

    cl_mem I_gpu;
    cl_mem I_prev_gpu;
    cl_mem I_phase_gpu;
    cl_mem I_phase_prev_gpu;
    cl_mem K_gpu;
    cl_mem sumJ_gpu;
    cl_mem Q_gpu;
    cl_mem R_gpu;  // NEU
    cl_mem T_gpu;  // NEU

    struct ocl_core ocl;
} *iwt_runtime_t;

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
    double I_min;
    double I_max;
    int MAX_ITER;
    double I_vac;
    double phi_0;
    double omega_0;
    double Z_0;
    double alpha_Z;
	size_t I_total_init;
} *iwt_config_t;

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

void iwt_compute_spectrum(const double* I, size_t N, struct iwt_spectrum* spec);
void iwt_print_spectrum(const struct iwt_spectrum* spec);
bool iwt_save_state(const iwt_runtime_t rt, const iwt_config_t cfg, const char* filename);
bool iwt_load_state(const iwt_runtime_t rt, const iwt_config_t cfg, const char* filename);
void iwt_print_border();
void iwt_print_status(const iwt_runtime_t rt, const iwt_config_t cfg, int iter, double max_q, double I_total,
    double I_min, double I_max, double mean_abs, double deviation);

#endif