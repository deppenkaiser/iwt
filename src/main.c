#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <ocl/ocl.h>

// ============================================================
// Konstanten
// ============================================================

#define NUM_NODES 32768
#define NUM_STEPS 10000
#define OUTPUT_INTERVAL 100
#define CONVERGENCE_THRESHOLD 1e-12
#define BATCH_SIZE 128
#define M_PI 3.14159265358979323846

// ============================================================
// Hilfsfunktionen
// ============================================================

static inline double iwt_D(void)
{
    double phi = (1.0 + sqrt(5.0)) / 2.0;
    return log(20.0 * phi) / log(2.0 + phi);
}

static inline double rand_double(void)
{
    return (double)rand() / (double)RAND_MAX;
}

// ============================================================
// Parameter-Strukturen
// ============================================================

typedef struct
{
    double D;
    double I_min;
    double I_ges;
    double lambda;
    double eta;
    double dt;
    double K_init;
} IWTParams;

typedef struct
{
    double *I;
    double *K;
    double *J;
    double *Q;
} HostData;

typedef struct
{
    cl_mem I;
    cl_mem K;
    cl_mem J;
    cl_mem Q;
} GPUBuffers;

// ============================================================
// Umrechnung IWT → SI
// ============================================================

typedef struct
{
    double I_phys;
    double M_phys;
    double E_phys;
    double L_phys;
    double T_phys;
    double rho_phys;
} PhysicalQuantities;

PhysicalQuantities convert_iwt_to_si(double I_IWT, double D, double dt_IWT, int step, double I0)
{
    PhysicalQuantities pq = {0};
    double l0_SI = 1.8e-15;
    double c_SI = 2.99792458e8;
    double T_SI = l0_SI / c_SI;
    double m_p_SI = 1.67262192369e-27;

    pq.T_phys = (double)step * dt_IWT * T_SI;
    pq.L_phys = l0_SI;
    pq.I_phys = I_IWT * I0;
    pq.M_phys = pq.I_phys * m_p_SI;
    pq.E_phys = pq.M_phys * c_SI * c_SI;
    pq.rho_phys = pq.M_phys / (l0_SI * l0_SI * l0_SI);
    return pq;
}

// ============================================================
// Initialisierung
// ============================================================

IWTParams iwt_params_init(void)
{
    double D = iwt_D();
    IWTParams p = {
        .D = D,
        .I_min = 1e-6,
        .I_ges = 1.0,
        .lambda = 0.01,
        .eta = 0.001,
        .dt = 0.01,
        .K_init = 0.001
    };
    return p;
}

HostData* host_data_alloc(void)
{
    HostData *h = malloc(sizeof(HostData));
    if (!h) return NULL;

    size_t N = NUM_NODES;
    h->I = malloc(N * sizeof(double));
    h->K = malloc(N * N * sizeof(double));
    h->J = malloc(N * N * sizeof(double));
    h->Q = malloc(N * sizeof(double));

    if (!h->I || !h->K || !h->J || !h->Q)
    {
        free(h->I); free(h->K); free(h->J); free(h->Q);
        free(h);
        return NULL;
    }
    return h;
}

void host_data_init(HostData *h, IWTParams *p)
{
    size_t N = NUM_NODES;
    srand(42);

    for (uint32_t i = 0; i < N; i++)
    {
        double fluctuation = 0.001 * (rand_double() - 0.5);
        h->I[i] = p->I_min + fluctuation;
    }

    double sum_I = 0.0;
    for (uint32_t i = 0; i < N; i++) sum_I += h->I[i];
    double scale = p->I_ges / sum_I;
    for (uint32_t i = 0; i < N; i++) h->I[i] *= scale;

    for (uint32_t i = 0; i < N; i++)
    {
        for (uint32_t j = 0; j < N; j++)
        {
            h->K[i * N + j] = (i == j) ? 1.0 : p->K_init;
        }
    }

    for (uint32_t i = 0; i < N * N; i++) h->J[i] = 0.0;
    for (uint32_t i = 0; i < N; i++) h->Q[i] = 0.0;
}

void host_data_free(HostData *h)
{
    if (!h) return;
    free(h->I); free(h->K); free(h->J); free(h->Q);
    free(h);
}

// ============================================================
// GPU-Buffer
// ============================================================

GPUBuffers gpu_buffers_create(struct ocl_core *ocl, HostData *h)
{
    size_t N = NUM_NODES;
    GPUBuffers b = {0};

    b.I = ocl_create_buffer(ocl, OCL_BUF_READ_WRITE, N * sizeof(double), h->I);
    b.K = ocl_create_buffer(ocl, OCL_BUF_READ_WRITE, N * N * sizeof(double), h->K);
    b.J = ocl_create_buffer(ocl, OCL_BUF_READ_WRITE, N * N * sizeof(double), h->J);
    b.Q = ocl_create_buffer(ocl, OCL_BUF_READ_WRITE, N * sizeof(double), h->Q);

    return b;
}

void gpu_buffers_free(GPUBuffers *b)
{
    clReleaseMemObject(b->I);
    clReleaseMemObject(b->K);
    clReleaseMemObject(b->J);
    clReleaseMemObject(b->Q);
}

// ============================================================
// Export-Funktionen
// ============================================================

void export_metadata(IWTParams *p)
{
    printf("\n=== IWT-Parameter ===\n");
    printf("  D       = %.6f\n", p->D);
    printf("  I_min   = %.6e\n", p->I_min);
    printf("  I_ges   = %.6f\n", p->I_ges);
    printf("  lambda  = %.6f\n", p->lambda);
    printf("  eta     = %.6f\n", p->eta);
    printf("  dt      = %.6f\n", p->dt);
    printf("  K_init  = %.6f\n", p->K_init);
}

void export_q_distribution(HostData *h, int step)
{
    char filename[64];
    snprintf(filename, sizeof(filename), "q_step_%d.csv", step);
    FILE *f = fopen(filename, "w");
    if (!f) return;

    fprintf(f, "index,Q\n");
    for (uint32_t i = 0; i < NUM_NODES; i++)
    {
        fprintf(f, "%u,%.12e\n", i, h->Q[i]);
    }
    fclose(f);
    printf("Q-Verteilung exportiert: %s\n", filename);
}

void export_metrix(HostData *h, int step)
{
    char filename[64];
    snprintf(filename, sizeof(filename), "metrix_step_%d.csv", step);
    FILE *f = fopen(filename, "w");
    if (!f) return;

    size_t N = NUM_NODES;
    fprintf(f, "i,j,g_ij\n");
    for (uint32_t i = 0; i < N; i++)
    {
        for (uint32_t j = 0; j < N; j++)
        {
            if (i == j) continue;
            double g_ij = h->K[i * N + j] / sqrt(h->K[i * N + i] * h->K[j * N + j]);
            if (g_ij > 1e-6)
                fprintf(f, "%u,%u,%.12e\n", i, j, g_ij);
        }
    }
    fclose(f);
    printf("Metrik exportiert: %s\n", filename);
}

void export_information(HostData *h, int step)
{
    char filename[64];
    snprintf(filename, sizeof(filename), "info_step_%d.csv", step);
    FILE *f = fopen(filename, "w");
    if (!f) return;

    fprintf(f, "index,I\n");
    for (uint32_t i = 0; i < NUM_NODES; i++)
    {
        fprintf(f, "%u,%.12e\n", i, h->I[i]);
    }
    fclose(f);
    printf("Information exportiert: %s\n", filename);
}

// ============================================================
// Hauptsimulation
// ============================================================

void run_simulation(
    struct ocl_core *ocl,
    cl_kernel kernel_flux,
    cl_kernel kernel_q,
    cl_kernel kernel_update_I,
    cl_kernel kernel_update_K,
    GPUBuffers *buf,
    HostData *h,
    IWTParams *p
)
{
    size_t N = NUM_NODES;

    double *host_I = malloc(N * sizeof(double));
    double *host_Q = malloc(N * sizeof(double));

    if (!host_I || !host_Q) return;

    for (int step = 0; step < NUM_STEPS; step++)
    {
        // 1. Flux: Batch über i und j
        for (uint32_t oi = 0; oi < N; oi += BATCH_SIZE)
        {
            uint32_t bi = (oi + BATCH_SIZE > N) ? N - oi : BATCH_SIZE;
            for (uint32_t oj = 0; oj < N; oj += BATCH_SIZE)
            {
                uint32_t bj = (oj + BATCH_SIZE > N) ? N - oj : BATCH_SIZE;

                ocl_set_parameter_compute_flux(
                    kernel_flux,
                    buf->I,
                    buf->K,
                    buf->J,
                    (uint32_t)N,
                    oi, oj, bi, bj
                );

                if (!ocl_enqueue_kernel(ocl, kernel_flux, bi * bj, 256))
                {
                    fprintf(stderr, "Fehler: Flux Batch bei Schritt %d\n", step);
                    free(host_I); free(host_Q);
                    return;
                }
            }
        }

        // 2. Q: Batch über i
        for (uint32_t offset = 0; offset < N; offset += BATCH_SIZE)
        {
            uint32_t batch = (offset + BATCH_SIZE > N) ? N - offset : BATCH_SIZE;

            ocl_set_parameter_compute_q(
                kernel_q,
                buf->J,
                buf->Q,
                (uint32_t)N,
                offset,
                batch
            );

            if (!ocl_enqueue_kernel(ocl, kernel_q, batch, 256))
            {
                fprintf(stderr, "Fehler: Q Batch bei Schritt %d\n", step);
                free(host_I); free(host_Q);
                return;
            }
        }

        // 3. Update I: Batch über i
        for (uint32_t offset = 0; offset < N; offset += BATCH_SIZE)
        {
            uint32_t batch = (offset + BATCH_SIZE > N) ? N - offset : BATCH_SIZE;

            ocl_set_parameter_update_I(
                kernel_update_I,
                buf->I,
                buf->J,
                p->dt,
                (uint32_t)N,
                offset,
                batch
            );

            if (!ocl_enqueue_kernel(ocl, kernel_update_I, batch, 256))
            {
                fprintf(stderr, "Fehler: Update I Batch bei Schritt %d\n", step);
                free(host_I); free(host_Q);
                return;
            }
        }

        // 4. Update K: Batch über i und j
        for (uint32_t oi = 0; oi < N; oi += BATCH_SIZE)
        {
            uint32_t bi = (oi + BATCH_SIZE > N) ? N - oi : BATCH_SIZE;
            for (uint32_t oj = 0; oj < N; oj += BATCH_SIZE)
            {
                uint32_t bj = (oj + BATCH_SIZE > N) ? N - oj : BATCH_SIZE;

                ocl_set_parameter_update_K(
                    kernel_update_K,
                    buf->K,
                    buf->I,
                    p->eta,
                    p->lambda,
                    p->dt,
                    (uint32_t)N,
                    oi, oj, bi, bj
                );

                if (!ocl_enqueue_kernel(ocl, kernel_update_K, bi * bj, 256))
                {
                    fprintf(stderr, "Fehler: Update K Batch bei Schritt %d\n", step);
                    free(host_I); free(host_Q);
                    return;
                }
            }
        }

        // 5. Ausgabe
        if (step % OUTPUT_INTERVAL == 0)
        {
            clEnqueueReadBuffer(ocl->queue, buf->I, CL_TRUE, 0, N * sizeof(double), host_I, 0, NULL, NULL);
            clEnqueueReadBuffer(ocl->queue, buf->Q, CL_TRUE, 0, N * sizeof(double), host_Q, 0, NULL, NULL);

            double sum_I = 0.0, max_Q = 0.0;
            for (uint32_t i = 0; i < N; i++)
            {
                sum_I += host_I[i];
                if (fabs(host_Q[i]) > max_Q) max_Q = fabs(host_Q[i]);
            }

            printf("Schritt %d: sum_I = %.12f, max_Q = %.12e\n", step, sum_I, max_Q);

            if (max_Q < CONVERGENCE_THRESHOLD)
            {
                printf("Konvergenz bei Schritt %d\n", step);
                clEnqueueReadBuffer(ocl->queue, buf->I, CL_TRUE, 0, N * sizeof(double), h->I, 0, NULL, NULL);
                clEnqueueReadBuffer(ocl->queue, buf->Q, CL_TRUE, 0, N * sizeof(double), h->Q, 0, NULL, NULL);
                clEnqueueReadBuffer(ocl->queue, buf->K, CL_TRUE, 0, N * N * sizeof(double), h->K, 0, NULL, NULL);
                export_q_distribution(h, step);
                export_metrix(h, step);
                export_information(h, step);
                break;
            }
        }
    }

    free(host_I);
    free(host_Q);
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

    cl_kernel kernel_flux      = ocl_get_kernel(&ocl, OCL_KERNEL_COMPUTE_FLUX);
    cl_kernel kernel_q         = ocl_get_kernel(&ocl, OCL_KERNEL_COMPUTE_Q);
    cl_kernel kernel_update_I  = ocl_get_kernel(&ocl, OCL_KERNEL_UPDATE_I);
    cl_kernel kernel_update_K  = ocl_get_kernel(&ocl, OCL_KERNEL_UPDATE_K);

    if (!kernel_flux || !kernel_q || !kernel_update_I || !kernel_update_K)
    {
        fprintf(stderr, "Fehler: Mindestens ein Kernel nicht gefunden.\n");
        ocl_deinitialize(&ocl);
        return 1;
    }

    IWTParams p = iwt_params_init();
    export_metadata(&p);

    HostData *h = host_data_alloc();
    if (!h)
    {
        fprintf(stderr, "Fehler: Host-Speicher fehlgeschlagen.\n");
        ocl_deinitialize(&ocl);
        return 1;
    }
    host_data_init(h, &p);

    GPUBuffers buf = gpu_buffers_create(&ocl, h);
    if (!buf.I || !buf.K || !buf.J || !buf.Q)
    {
        fprintf(stderr, "Fehler: GPU-Buffer fehlgeschlagen.\n");
        host_data_free(h);
        ocl_deinitialize(&ocl);
        return 1;
    }

    run_simulation(&ocl, kernel_flux, kernel_q, kernel_update_I, kernel_update_K,
                   &buf, h, &p);

    printf("\n=== Umrechnung in SI-Einheiten ===\n");
    double I0 = 1.0;
    for (int i = 0; i < 10; i++)
    {
        PhysicalQuantities pq = convert_iwt_to_si(h->I[i], p.D, p.dt, NUM_STEPS, I0);
        printf("  Knoten %d: I_phys = %.6e, M_phys = %.6e kg\n",
               i, pq.I_phys, pq.M_phys);
    }

    gpu_buffers_free(&buf);
    host_data_free(h);
    ocl_deinitialize(&ocl);

    return 0;
}