#ifndef IWT_H
#define IWT_H

#include <stddef.h>
#include <stdbool.h>
#include <ocl/ocl.h>

typedef struct iwt_config
{
    size_t N;
    size_t BATCH_SIZE;
    double D;                // fraktale Dimension (2.704)
    double l0;               // fundamentale Länge (in IWT-Einheiten = 1.0)
    double T;                // fundamentale Zeit (in IWT-Einheiten = 1.0)
    double alpha_IWT;        // lokale Kopplung (aus Feinstrukturkonstante)
    double beta_IWT;         // globale Kopplung (aus Gravitationskonstante)
    double delta_t;          // Schrittweite (in IWT-Einheiten)

    double DT;               // = delta_t / T
    double ETA;              // = alpha_IWT * DT
    double LAMBDA;           // = beta_IWT * l0^(3-D) / T^2 * DT^2
    double GAMMA;            // = (alpha_IWT / beta_IWT) * DT

    double THRESHOLD;        // = (3.0 - D) / 3.0
    int MAX_ITER;
} *iwt_config_t;

typedef struct iwt_runtime
{
    double *I;
    double *K;
    double *sumJ;
    double *Q;

    cl_mem I_gpu;
    cl_mem K_gpu;
    cl_mem sumJ_gpu;
    cl_mem Q_gpu;

    struct ocl_core ocl;
} *iwt_runtime_t;

double iwt_pi(void);
double iwt_fundamental_length(void);
double iwt_fundamental_time();
double iwt_alpha_IWT();
double iwt_fractal_dimension(void);
double iwt_delta_I_min();
double iwt_alpha_IWT();
double iwt_beta_IWT();

#endif
