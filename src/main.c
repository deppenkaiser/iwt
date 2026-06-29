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

    // 3. Host-Daten initialisieren (Ring-Topologie für Test)
    float *host_nodes = malloc(NUM_NODES * sizeof(float));
    uint32_t *host_adjacency = malloc(NUM_EDGES * sizeof(uint32_t));
    float *host_flows = malloc(NUM_EDGES * sizeof(float));

    for (uint32_t i = 0; i < NUM_NODES; i++)
    {
        host_nodes[i] = 1.0f;
    }

    for (uint32_t i = 0; i < NUM_NODES; i++)
    {
        host_adjacency[i * 2] = i * 12;
        host_adjacency[i * 2 + 1] = i * 12 + 12;
        for (uint32_t j = 0; j < 12; j++)
        {
            host_adjacency[i * 12 + j] = (i + j + 1) % NUM_NODES;
        }
    }

    // 4. GPU-Buffer erstellen
    cl_mem d_nodes = ocl_create_buffer(&ocl, OCL_BUF_READ_WRITE,
                                       NUM_NODES * sizeof(float), host_nodes);
    cl_mem d_adjacency = ocl_create_buffer(&ocl, OCL_BUF_READ_ONLY,
                                           NUM_EDGES * sizeof(uint32_t), host_adjacency);
    cl_mem d_flows = ocl_create_buffer(&ocl, OCL_BUF_WRITE_ONLY,
                                       NUM_EDGES * sizeof(float), NULL);

    // 5. Kernel holen
    cl_kernel kernel = ocl_get_kernel(&ocl, OCL_KERNEL_IWT_UPDATE);
    if (!kernel)
    {
        fprintf(stderr, "Fehler: Kernel 'iwt_update' nicht gefunden.\n");
        ocl_deinitialize(&ocl);
        return 1;
    }

    // 6. Parameter setzen
    float dt = 0.01f;
	ocl_set_parameter_iwt_update(kernel, d_nodes, (float)D, (float)l0, NUM_NODES, dt);

    // 7. Kernel ausführen
    if (!ocl_enqueue_kernel(&ocl, kernel, NUM_NODES, 256))
    {
        fprintf(stderr, "Fehler: Kernel-Ausführung fehlgeschlagen.\n");
        ocl_deinitialize(&ocl);
        return 1;
    }

    // 8. Ergebnisse zurücklesen
    clEnqueueReadBuffer(ocl.queue, d_nodes, CL_TRUE, 0,
                        NUM_NODES * sizeof(float), host_nodes, 0, NULL, NULL);

    // 9. Ergebnis ausgeben
    printf("Erste 10 Knotenwerte nach Update:\n");
    for (int i = 0; i < 10; i++)
    {
        printf("  nodes[%d] = %.6f\n", i, host_nodes[i]);
    }

    // 10. Aufräumen
    free(host_nodes);
    free(host_adjacency);
    free(host_flows);
    clReleaseMemObject(d_nodes);
    clReleaseMemObject(d_adjacency);
    clReleaseMemObject(d_flows);
    ocl_deinitialize(&ocl);

    return 0;
}
