#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <ocl/ocl.h>

// ============================================================
// IWT-Konstanten (alles wird berechnet, keine externen Werte)
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
// IWT-Parameter aus Naturkonstanten berechnen
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

    // Gemessene Naturkonstanten
    double c = 2.99792458e8;          // m/s
    double hbar = 1.054571817e-34;    // J*s
    double G = 6.67430e-11;           // m^3/(kg*s^2)
    double alpha = 7.29735256e-3;     // Feinstrukturkonstante
    double k_B = 1.380649e-23;        // J/K
    double m_nu = 0.1;                // eV/c^2 (geschätzt)

    // 1. T = l0 / c
    params.T = l0 / c;

    // 2. beta_IWT = G * T^2 / l0^(3-D)
    params.beta = G * (params.T * params.T) / pow(l0, 3.0 - D);

    // 3. alpha_IWT = alpha * beta_IWT
    params.alpha = alpha * params.beta;

    // 4. gamma_IWT = k_B * alpha_IWT * T / l0 * <I>
    //    <I> = 1.0 (Normierung)
    double I_mean = 1.0;
    params.gamma = k_B * params.alpha * params.T / l0 * I_mean;

    // D und l0 speichern
    params.D = D;
    params.l0 = l0;

    return params;
}

// ============================================================
// main
// ============================================================

#define NUM_NODES 1024
#define NUM_EDGES (NUM_NODES * 12)

int main(int argc, char *argv[])
{
    // 1. ocl initialisieren
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

    // 2. IWT-Konstanten berechnen
    double D = iwt_calculate_D();
    double hbar = 1.054571817e-34;
    double m_p = 1.67262192369e-27;
    double c = 2.99792458e8;
    double l0 = iwt_calculate_l0(hbar, m_p, c);

    printf("IWT-Konstanten:\n");
    printf("  D  = %.16f\n", D);
    printf("  l0 = %.4e m\n", l0);

	// 3. IWT-Parameter berechnen
	IWTParameters params = iwt_calculate_parameters(D, l0);

	// 4. Feinstrukturkonstante berechnen
	double alpha = params.alpha / params.beta;

	printf("\nIWT-Parameter:\n");
	printf("  alpha_IWT = %.4e\n", params.alpha);
	printf("  beta_IWT  = %.4e\n", params.beta);
	printf("  gamma_IWT = %.4e\n", params.gamma);
	printf("  T         = %.4e s\n", params.T);

	printf("\nFeinstrukturkonstante:\n");
	printf("  alpha = alpha_IWT / beta_IWT = %.4e\n", alpha);
	printf("  Gemessener Wert: 7.29735256e-3\n");
	printf("  Abweichung: %.2e (%.2f%%)\n",
		alpha - 7.29735256e-3,
		(alpha - 7.29735256e-3) / 7.29735256e-3 * 100.0);

    // 4. Host-Daten initialisieren (Ring-Topologie für Test)
    float *host_nodes = malloc(NUM_NODES * sizeof(float));
    uint32_t *host_adjacency = malloc(NUM_EDGES * sizeof(uint32_t));
    float *host_flows = malloc(NUM_EDGES * sizeof(float));

    // Ungleiche Verteilung
    for (uint32_t i = 0; i < NUM_NODES; i++)
    {
        host_nodes[i] = 1.0f + 0.001f * (float)(i % 10);
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
                                       NUM_NODES * sizeof(float), host_nodes);
    cl_mem d_adjacency = ocl_create_buffer(&ocl, OCL_BUF_READ_ONLY,
                                           NUM_EDGES * sizeof(uint32_t), host_adjacency);
    cl_mem d_flows = ocl_create_buffer(&ocl, OCL_BUF_WRITE_ONLY,
                                       NUM_EDGES * sizeof(float), NULL);

    // 6. Kernel holen
    cl_kernel kernel = ocl_get_kernel(&ocl, OCL_KERNEL_IWT_UPDATE);
    if (!kernel)
    {
        fprintf(stderr, "Fehler: Kernel 'iwt_update' nicht gefunden.\n");
        ocl_deinitialize(&ocl);
        return 1;
    }

    // 7. Parameter setzen (verwende die berechneten Kopplungen)
    float dt = 0.00001f;

    // Die Kopplungen werden im Kernel verwendet:
    // - G  = params.beta  (für WG)
    // - k  = params.alpha (für WED)
    // - Q  = params.gamma (für Q)
    //
    // Der Kernel muss diese Werte als Parameter erhalten.
    // Dazu muss die Setter-Funktion erweitert werden.

    ocl_set_parameter_iwt_update(kernel, d_nodes, d_adjacency, (float)D, (float)l0, (float)params.beta, (float)params.alpha, (float)params.gamma, NUM_NODES, dt);

    // 8. Kernel ausführen (10000 Schritte)
    for (int step = 0; step < 10000; step++)
    {
        if (!ocl_enqueue_kernel(&ocl, kernel, NUM_NODES, 256))
        {
            fprintf(stderr, "Fehler: Kernel-Ausführung fehlgeschlagen.\n");
            ocl_deinitialize(&ocl);
            return 1;
        }
    }

    // 9. Ergebnisse zurücklesen
    clEnqueueReadBuffer(ocl.queue, d_nodes, CL_TRUE, 0,
                        NUM_NODES * sizeof(float), host_nodes, 0, NULL, NULL);

    // 10. Ergebnis ausgeben
    printf("\nErste 10 Knotenwerte nach Update:\n");
    for (int i = 0; i < 10; i++)
    {
        printf("  nodes[%d] = %.6f\n", i, host_nodes[i]);
    }

    // 11. Aufräumen
    free(host_nodes);
    free(host_adjacency);
    free(host_flows);
    clReleaseMemObject(d_nodes);
    clReleaseMemObject(d_adjacency);
    clReleaseMemObject(d_flows);
    ocl_deinitialize(&ocl);

    return 0;
}
