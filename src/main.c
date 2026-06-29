#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <ocl/ocl.h>

// ============================================================
// Parameter für die Simulation
// ============================================================

#define NUM_NODES 1024
#define NUM_EDGES (NUM_NODES * 12)
#define NUM_STEPS 10
#define OUTPUT_INTERVAL 1
#define SCALE_FACTOR 1.0e6        // Atomare Skala (l0 -> 1e-10 m)

// ============================================================
// IWT-Konstanten
// ============================================================

static inline long double iwt_phi(void)
{
    return (1.0L + sqrtl(5.0L)) / 2.0L;
}

static inline long double iwt_pi(void)
{
    return 4.0L * atanl(1.0L);
}

static inline double iwt_calculate_D(void)
{
    long double phi = iwt_phi();
    long double numerator = logl(20.0L * phi);
    long double denominator = logl(2.0L + phi);
    return (double)(numerator / denominator);
}

static inline double iwt_calculate_l0(double hbar, double m_p, double c)
{
    long double phi = iwt_phi();
    long double pi = iwt_pi();
    long double geometric_factor = (5.0L * phi * phi * phi) / (6.0L * pi);
    long double l0 = powl(geometric_factor, 1.0L / 3.0L) * ((long double)hbar / ((long double)m_p * (long double)c));
    return (double)l0;
}

// ============================================================
// IWT-Parameter
// ============================================================

typedef struct {
    double alpha;   // alpha_IWT
    double beta;    // beta_IWT
    double gamma;   // gamma_IWT
    double T;       // fundamentaler Zeitschrift
    double l0;      // fundamentale Länge
    double D;       // fraktale Dimension
} IWTParameters;

static inline IWTParameters iwt_calculate_parameters(double D, double l0)
{
    IWTParameters params;

    double c = 2.99792458e8;
    double G = 6.67430e-11;
    double alpha = 7.29735256e-3;
    double k_B = 1.380649e-23;

    params.T = l0 / c;
    params.beta = G * (params.T * params.T) / pow(l0, 3.0 - D);
    params.alpha = alpha * params.beta;
    double I_mean = 1.0;
    params.gamma = k_B * params.alpha * params.T / l0 * I_mean;
    params.D = D;
    params.l0 = l0;

    return params;
}

// ============================================================
// main
// ============================================================

int main(int argc, char *argv[])
{
    struct ocl_core ocl = {0};
    if (!ocl_initialize(&ocl))
    {
        fprintf(stderr, "Fehler: OpenCL-Initialisierung fehlgeschlagen.\n");
        return 1;
    }

    if (!ocl_compile(&ocl))
    {
        fprintf(stderr, "Fehler: OpenCL-Kompilierung fehlgeschlagen.\n");
        ocl_deinitialize(&ocl);
        return 1;
    }

    if (!ocl_load_kernels(&ocl))
    {
        fprintf(stderr, "Fehler: Kernel konnten nicht geladen werden.\n");
        ocl_deinitialize(&ocl);
        return 1;
    }

    // 1. IWT-Konstanten berechnen (fundamental)
    double D = iwt_calculate_D();
    double hbar = 1.054571817e-34;
    double m_p = 1.67262192369e-27;
    double c = 2.99792458e8;
    double l0 = iwt_calculate_l0(hbar, m_p, c);

    // 2. Auf atomare Skala skalieren
    double l0_atom = l0 * SCALE_FACTOR;
    double T_atom = (l0 / c) * SCALE_FACTOR;

    printf("IWT-Konstanten (fundamental):\n");
    printf("  D  = %.16f\n", D);
    printf("  l0 = %.4e m\n", l0);
    printf("  T  = %.4e s\n", T_atom / SCALE_FACTOR);

    printf("\nIWT-Konstanten (atomar):\n");
    printf("  l0 = %.4e m\n", l0_atom);
    printf("  T  = %.4e s\n", T_atom);

    // 3. IWT-Parameter berechnen
    IWTParameters params = iwt_calculate_parameters(D, l0);

    printf("\nIWT-Parameter:\n");
    printf("  alpha_IWT = %.4e\n", params.alpha);
    printf("  beta_IWT  = %.4e\n", params.beta);
    printf("  gamma_IWT = %.4e\n", params.gamma);

    // 4. Host-Daten initialisieren
    double *host_nodes = malloc(NUM_NODES * sizeof(double));
    uint32_t *host_adjacency = malloc(NUM_EDGES * sizeof(uint32_t));
    double *host_flows = malloc(NUM_EDGES * sizeof(double));

    // Knoten initialisieren (Massen in u)
    for (uint32_t i = 0; i < NUM_NODES; i++)
    {
        host_nodes[i] = 1.0 + 0.1 * (double)(i % 10);   // 1.0 u, 1.1 u, 1.2 u, ...
    }

    // Adjazenzliste (Ring mit 12 Nachbarn)
    for (uint32_t i = 0; i < NUM_NODES; i++)
    {
        host_adjacency[i * 2] = i * 12;
        host_adjacency[i * 2 + 1] = i * 12 + 12;
        for (uint32_t j = 0; j < 12; j++)
        {
            host_adjacency[i * 12 + j] = (i + j + 1) % NUM_NODES;
        }
    }

    // 5. GPU-Buffer erstellen
    cl_mem d_nodes = ocl_create_buffer(&ocl, OCL_BUF_READ_WRITE,
                                       NUM_NODES * sizeof(double), host_nodes);
    cl_mem d_adjacency = ocl_create_buffer(&ocl, OCL_BUF_READ_ONLY,
                                           NUM_EDGES * sizeof(uint32_t), host_adjacency);
    cl_mem d_flows = ocl_create_buffer(&ocl, OCL_BUF_WRITE_ONLY,
                                       NUM_EDGES * sizeof(double), NULL);

    cl_kernel kernel = ocl_get_kernel(&ocl, OCL_KERNEL_IWT_UPDATE);
    if (!kernel)
    {
        fprintf(stderr, "Fehler: Kernel 'iwt_update' nicht gefunden.\n");
        ocl_deinitialize(&ocl);
        return 1;
    }

    // 6. Parameter setzen (Kopplungen unverändert)
    double G_sim = params.beta;
    double k_sim = params.alpha;
    double Q_sim = params.gamma;
    double dt = T_atom;

    printf("\nSimulations-Parameter (atomar, Massen in u):\n");
    printf("  G_sim  = %.4e\n", G_sim);
    printf("  k_sim  = %.4e\n", k_sim);
    printf("  Q_sim  = %.4e\n", Q_sim);
    printf("  dt     = %.4e s\n", dt);

    ocl_set_parameter_iwt_update(kernel, d_nodes, d_adjacency, d_flows,
                                 D, l0_atom,
                                 G_sim, k_sim, Q_sim,
                                 NUM_NODES, dt);

    // 7. Simulation
    printf("\nSimulation läuft...\n");

    for (int step = 0; step < NUM_STEPS; step++)
    {
        if (!ocl_enqueue_kernel(&ocl, kernel, NUM_NODES, 256))
        {
            fprintf(stderr, "Fehler: Kernel-Ausführung bei Schritt %d fehlgeschlagen.\n", step);
            ocl_deinitialize(&ocl);
            return 1;
        }

        if (step % OUTPUT_INTERVAL == 0)
        {
            clEnqueueReadBuffer(ocl.queue, d_nodes, CL_TRUE, 0,
                                NUM_NODES * sizeof(double), host_nodes, 0, NULL, NULL);

            printf("Schritt %d:\n", step);
            for (int i = 0; i < 10; i++)
            {
                printf("  nodes[%d] = %.20f\n", i, host_nodes[i]);
            }
            printf("\n");
        }
    }

    // 8. Flüsse auslesen
    clEnqueueReadBuffer(ocl.queue, d_flows, CL_TRUE, 0,
                        NUM_EDGES * sizeof(double), host_flows, 0, NULL, NULL);

    printf("\nKräfte für Knoten 0 (erste 4 Flüsse):\n");
    for (int i = 0; i < 4; i++)
    {
        printf("  flows[%d] = %.4e\n", i, host_flows[i]);
    }

    // 9. Aufräumen
    free(host_nodes);
    free(host_adjacency);
    free(host_flows);
    clReleaseMemObject(d_nodes);
    clReleaseMemObject(d_adjacency);
    clReleaseMemObject(d_flows);
    ocl_deinitialize(&ocl);

    return 0;
}
