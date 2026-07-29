// ============================================================================
// iwt.h
// ============================================================================

#ifndef IWT_H
#define IWT_H

#include <stddef.h>
#include <stdbool.h>
#include <ocl/ocl.h>

typedef struct iwt_runtime
{
    double *I_real;
    double *I_imag;
    double *I_prev_real;
    double *I_prev_imag;
    double *I_phase;
    double *I_phase_prev;
    double *K;
    double *sumJ;
    double *Q;
    double *xi_real;
    double *xi_imag;
    double *uncertainty;
    double *mass;
    double *charge;
    
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
    double hbar;
    double uncertainty_scale;
    bool enable_fluctuations;
    unsigned int seed;
} *iwt_config_t;

typedef struct iwt_spectrum
{
    size_t count_vacuum;
    size_t count_electron;
    size_t count_proton;
    size_t count_u_quark;
    size_t count_d_quark;
    size_t count_other;
} iwt_spectrum_t;

double iwt_pi(void);
double iwt_fundamental_length(void);
double iwt_fundamental_time(void);
double iwt_fractal_dimension(void);
double iwt_I_min(void);
double iwt_I_max(void);
double iwt_delta_I(void);
double iwt_alpha_IWT(void);
double iwt_beta_IWT(void);

void iwt_compute_spectrum(const double* I_real, const double* I_imag, size_t N, iwt_spectrum_t* spec);
int iwt_classify(double Re, double Im);
void iwt_save_heatmap_ppm(const double* data, size_t N, const char* filename, const char* type);
void iwt_save_charge_heatmap(const double *charge, size_t N, const char *filename);
void iwt_print_status(const iwt_runtime_t rt, const iwt_config_t cfg, int iter,
    double max_q, double I_total, double I_min, double I_max,
    double deviation, double sum_I_sq, double info_deviation);

#endif