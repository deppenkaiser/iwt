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
#define BATCH_SIZE 16384
#define NUM_EDGES (NUM_NODES * 12)
#define NUM_STEPS 100
#define OUTPUT_INTERVAL 10
#define MAX_SPECTRUM_DIST 250
#define M_PI 3.14159265358979323846

// ============================================================
// Hilfsfunktionen
// ============================================================

static inline double iwt_D(void)
{
    double phi = (1.0 + sqrt(5.0)) / 2.0;
    return log(20.0 * phi) / log(2.0 + phi);
}

// ============================================================
// Parameter-Strukturen
// ============================================================

typedef struct
{
    double D;
    double l0;
    double G;
    double k;
    double Q;
    double dt;
} IWTParams;

typedef struct
{
    double *nodes;
    double *x, *y, *z;
    double *vx, *vy, *vz;
    uint32_t *adjacency;
    double *flows;
} HostData;

typedef struct
{
    cl_mem nodes;
    cl_mem x, y, z;
    cl_mem vx, vy, vz;
    cl_mem adjacency;
    cl_mem flows;
    cl_mem q_potential;
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
// Zufallszahlen
// ============================================================

static inline double rand_double(void)
{
    return (double)rand() / (double)RAND_MAX;
}

// ============================================================
// Initialisierungen
// ============================================================

IWTParams iwt_params_init(void)
{
    IWTParams p =
    {
        .D = iwt_D(),
        .l0 = 1.0,
        .G = 1.0,
        .k = 1.0,
        .Q = 1.0,
        .dt = 1.0 // T!
    };
    return p;
}

HostData* host_data_alloc(void)
{
    HostData *h = malloc(sizeof(HostData));
    if (!h) return NULL;

    h->nodes    = malloc(NUM_NODES * sizeof(double));
    h->x        = malloc(NUM_NODES * sizeof(double));
    h->y        = malloc(NUM_NODES * sizeof(double));
    h->z        = malloc(NUM_NODES * sizeof(double));
    h->vx       = malloc(NUM_NODES * sizeof(double));
    h->vy       = malloc(NUM_NODES * sizeof(double));
    h->vz       = malloc(NUM_NODES * sizeof(double));
    h->adjacency = malloc(NUM_EDGES * sizeof(uint32_t));
    h->flows    = malloc(NUM_EDGES * sizeof(double));

    if (!h->nodes || !h->x || !h->y || !h->z ||
        !h->vx || !h->vy || !h->vz ||
        !h->adjacency || !h->flows)
    {
        free(h->nodes); free(h->x); free(h->y); free(h->z);
        free(h->vx); free(h->vy); free(h->vz);
        free(h->adjacency); free(h->flows); free(h);
        return NULL;
    }
    return h;
}

void host_data_init(HostData *h)
{
    srand(42); // Reproduzierbar

    // Knoten-Information (wie gehabt)
    for (uint32_t i = 0; i < NUM_NODES; i++)
    {
        h->nodes[i] = 1.0 + 0.1 * (double)(i % 10); // 0.1 = (3-D)/3
    }

    // Positionen: Zufällig in einer Kugel vom Radius 10
    for (uint32_t i = 0; i < NUM_NODES; i++)
    {
        // Gleichverteilung in einer Kugel
        double theta = 2.0 * M_PI * rand_double();
        double phi = acos(2.0 * rand_double() - 1.0);
        double r = 10.0 * cbrt(rand_double()); // Volumen gleichverteilt

        h->x[i] = r * sin(phi) * cos(theta);
        h->y[i] = r * sin(phi) * sin(theta);
        h->z[i] = r * cos(phi);

        h->vx[i] = 0.0;
        h->vy[i] = 0.0;
        h->vz[i] = 0.0;
    }

    // Adjazenzliste (12 Nachbarn pro Knoten)
    for (uint32_t i = 0; i < NUM_NODES; i++)
    {
        for (uint32_t j = 0; j < 12; j++)
        {
            h->adjacency[i * 12 + j] = (i + j + 1) % NUM_NODES;
        }
    }
}

void host_data_free(HostData *h)
{
    if (!h) return;
    free(h->nodes);
    free(h->x); free(h->y); free(h->z);
    free(h->vx); free(h->vy); free(h->vz);
    free(h->adjacency);
    free(h->flows);
    free(h);
}

// ============================================================
// GPU-Buffer (erweitert)
// ============================================================

GPUBuffers gpu_buffers_create(struct ocl_core *ocl, HostData *h)
{
    GPUBuffers b = {0};

    b.nodes      = ocl_create_buffer(ocl, OCL_BUF_READ_WRITE, NUM_NODES * sizeof(double), h->nodes);
    b.x          = ocl_create_buffer(ocl, OCL_BUF_READ_WRITE, NUM_NODES * sizeof(double), h->x);
    b.y          = ocl_create_buffer(ocl, OCL_BUF_READ_WRITE, NUM_NODES * sizeof(double), h->y);
    b.z          = ocl_create_buffer(ocl, OCL_BUF_READ_WRITE, NUM_NODES * sizeof(double), h->z);
    b.vx         = ocl_create_buffer(ocl, OCL_BUF_READ_WRITE, NUM_NODES * sizeof(double), h->vx);
    b.vy         = ocl_create_buffer(ocl, OCL_BUF_READ_WRITE, NUM_NODES * sizeof(double), h->vy);
    b.vz         = ocl_create_buffer(ocl, OCL_BUF_READ_WRITE, NUM_NODES * sizeof(double), h->vz);
    b.adjacency  = ocl_create_buffer(ocl, OCL_BUF_READ_ONLY,  NUM_EDGES * sizeof(uint32_t), h->adjacency);
    b.flows      = ocl_create_buffer(ocl, OCL_BUF_WRITE_ONLY, NUM_EDGES * sizeof(double), NULL);
    b.q_potential = ocl_create_buffer(ocl, OCL_BUF_READ_WRITE, NUM_NODES * sizeof(double), NULL);

    return b;
}

void gpu_buffers_free(GPUBuffers *b)
{
    clReleaseMemObject(b->nodes);
    clReleaseMemObject(b->x); clReleaseMemObject(b->y); clReleaseMemObject(b->z);
    clReleaseMemObject(b->vx); clReleaseMemObject(b->vy); clReleaseMemObject(b->vz);
    clReleaseMemObject(b->adjacency);
    clReleaseMemObject(b->flows);
    clReleaseMemObject(b->q_potential);
}

// ============================================================
// Q-Feld (Bohm-Potential) – CPU-Version (wie gehabt)
// ============================================================

void compute_bohm_potential(double* nodes, double* q_potential, uint32_t num_nodes, double D)
{
    double sum_I = 0.0;
    for (uint32_t i = 0; i < num_nodes; i++) sum_I += nodes[i];
    if (sum_I < 1e-30) return;

    double* sqrt_rho = malloc(num_nodes * sizeof(double));
    for (uint32_t i = 0; i < num_nodes; i++) sqrt_rho[i] = sqrt(nodes[i] / sum_I);

    double* laplace = malloc(num_nodes * sizeof(double));
    for (uint32_t i = 0; i < num_nodes; i++)
    {
        double sum = 0.0;
        for (uint32_t j = 0; j < num_nodes; j++)
        {
            if (i == j) continue;
            double dist = (double)(j - i);
            if (dist < 0.0) dist = -dist;
            if (dist < 1.0) dist = 1.0;
            sum += (sqrt_rho[j] - sqrt_rho[i]) / (dist * dist);
        }
        laplace[i] = sum;
    }

    double hbar = 1.054571817e-34;
    double mass = 1.0;
    double prefactor = -hbar * hbar / (2.0 * mass);

    for (uint32_t i = 0; i < num_nodes; i++)
    {
        q_potential[i] = (sqrt_rho[i] > 1e-30)
            ? prefactor * laplace[i] / sqrt_rho[i]
            : 0.0;
    }

    free(sqrt_rho);
    free(laplace);
}

// ============================================================
// Kraftspektrum (wie gehabt)
// ============================================================

void compute_force_spectrum(HostData *h, double D)
{
    double *spectrum = calloc(NUM_NODES, sizeof(double));
    if (!spectrum) return;

    for (uint32_t i = 0; i < NUM_NODES; i++)
    {
        double I_i = h->nodes[i];
        uint32_t start = i * 12;
        uint32_t end = i * 12 + 12;

        for (uint32_t e = start; e < end; e++)
        {
            uint32_t j = h->adjacency[e];
            if (j == i) continue;
            double I_j = h->nodes[j];
            double dx = h->x[j] - h->x[i];
            double dy = h->y[j] - h->y[i];
            double dz = h->z[j] - h->z[i];
            double dist = sqrt(dx*dx + dy*dy + dz*dz);
            if (dist < 1.0) dist = 1.0;
            if (dist >= MAX_SPECTRUM_DIST) continue;

            double diff = I_i - I_j;
            double r = pow(dist, D - 2.0);
            double F_wed = diff / pow(r, D - 1.0);
            double F_wg  = -diff / pow(r, D - 2.0);
            double F_q   = diff / pow(r, D - 2.0);
            double F_total = F_wed + F_wg + F_q;

            int idx = (int)dist;
            spectrum[idx] += F_total;
        }
    }

    printf("\nKraftspektrum (Distanz -> akkumulierte Kraft):\n");
    for (int d = 1; d < MAX_SPECTRUM_DIST; d++)
    {
        if (spectrum[d] != 0.0)
            printf("  dist %2d: %.6e\n", d, spectrum[d]);
    }
    free(spectrum);
}

// ============================================================
// WDBT+-Konstanten (wie gehabt)
// ============================================================

void compute_wdbt_constants(double D)
{
    printf("\n=== Schritt 3: Korrekte Kalibrierung der IWT-Parameter ===\n");

    double hbar_SI = 1.054571817e-34;
    double c_SI = 2.99792458e8;
    double G_SI = 6.67430e-11;
    double alpha_SI = 7.29735256e-3;
    double k_B_SI = 1.380649e-23;

    double l0_SI = 1.8e-15;
    double T_SI = l0_SI / c_SI;
    double I_mean = 1.0;

    double beta_IWT = G_SI * T_SI * T_SI / pow(l0_SI, 3.0 - D);
    double alpha_IWT = alpha_SI * beta_IWT;
    double gamma_IWT = alpha_IWT * k_B_SI * T_SI / l0_SI * I_mean;
    double Delta_I_min = hbar_SI / (alpha_IWT * l0_SI * l0_SI);

    double m_nu_eV = 0.1;
    double m_nu_kg = m_nu_eV * 1.782662e-36;
    double prefactor = hbar_SI / (l0_SI * c_SI);
    double exponent = (3.0 - D) / 2.0;
    double L_Q0_SI = l0_SI * pow(prefactor / m_nu_kg, 1.0 / exponent);

    double rho0 = 4.58e-14;
    double d_cmb = 8.85e25;
    double d_ceph = 1.0e26;
    double d_ref = 1.36e26;
    double d_rand = L_Q0_SI;

    double H_factor = (4.0 * M_PI * G_SI * rho0 * pow(l0_SI, 3.0 - D))
                    / (D * (D - 1.0) * c_SI);

    double H_cmb = H_factor * pow(d_cmb, D - 2.0);
    double H_ceph = H_factor * pow(d_ceph, D - 2.0);
    double H_ref = H_factor * pow(d_ref, D - 2.0);
    double H_rand = H_factor * pow(d_rand, D - 2.0);

    double conv = 3.08567758e19;
    double H_cmb_km = H_cmb * conv;
    double H_ceph_km = H_ceph * conv;
    double H_ref_km = H_ref * conv;
    double H_rand_km = H_rand * conv;

    printf("IWT-Parameter (dimensionslos):\n");
    printf("  α_IWT   = %.6e\n", alpha_IWT);
    printf("  β_IWT   = %.6e\n", beta_IWT);
    printf("  γ_IWT   = %.6e\n", gamma_IWT);
    printf("  ΔI_min  = %.6e\n", Delta_I_min);
    printf("  l0      = %.6e m\n", l0_SI);
    printf("  T       = %.6e s\n", T_SI);
    printf("  ⟨I⟩     = %.6f\n", I_mean);

    printf("\nRückgerechnete SI-Konstanten:\n");
    printf("  c  = %.6e m/s\n", l0_SI / T_SI);
    printf("  ℏ  = %.6e J·s (CODATA: %.6e)\n",
           alpha_IWT * Delta_I_min * l0_SI * l0_SI, hbar_SI);
    printf("  G  = %.6e m³/(kg·s²) (CODATA: %.6e)\n",
           beta_IWT * pow(l0_SI, 3.0 - D) / (T_SI * T_SI), G_SI);
    printf("  α  = %.6e (CODATA: %.6e)\n", alpha_IWT / beta_IWT, alpha_SI);
    printf("  k_B= %.6e J/K (CODATA: %.6e)\n",
           (gamma_IWT / alpha_IWT) * (l0_SI / T_SI) * (1.0 / I_mean), k_B_SI);

    printf("\nNeutrinomasse (nach Anhang J):\n");
    printf("  m_ν = %.6f eV/c²\n", m_nu_eV);
    printf("  L_Q0 = %.6e m\n", L_Q0_SI);

    printf("\nEffektive Hubble-Konstante:\n");
    printf("  CMB (d = %.2e m):   H = %.2f km/s/Mpc\n", d_cmb, H_cmb_km);
    printf("  Cepheiden (d = %.2e m): H = %.2f km/s/Mpc\n", d_ceph, H_ceph_km);
    printf("  SN Refsdal (d = %.2e m): H = %.2f km/s/Mpc\n", d_ref, H_ref_km);
    printf("  Rand (d = %.2e m):   H = %.2f km/s/Mpc\n", d_rand, H_rand_km);
    printf("\n  Hubble-Spannung: H_Cepheiden / H_CMB = %.3f\n", H_ceph_km / H_cmb_km);
    printf("  Beobachtung: 73.5 / 67.4 = %.3f\n", 73.5 / 67.4);
}

// ============================================================
// Export: Positionen
// ============================================================

void export_positions(HostData *h, int step)
{
    char filename[64];
    snprintf(filename, sizeof(filename), "positions_step_%d.csv", step);
    FILE *f = fopen(filename, "w");
    if (!f) return;

    fprintf(f, "index,x,y,z\n");
    
    // ALLE Knoten exportieren (nicht nur die ersten 100)
    for (uint32_t i = 0; i < NUM_NODES; i++)
    {
        fprintf(f, "%u,%.6f,%.6f,%.6f\n", i, h->x[i], h->y[i], h->z[i]);
    }
    fclose(f);
    printf("Positionen exportiert: %s (%u Knoten)\n", filename, NUM_NODES);
}

// ============================================================
// Simulation (Motion-Kernel)
// ============================================================

void run_simulation_motion(
    struct ocl_core *ocl,
    cl_kernel kernel_motion,
    cl_kernel kernel_q,
    GPUBuffers *buf,
    HostData *h,
    IWTParams *p,
    double initial_sum
)
{
    double *host_q = malloc(NUM_NODES * sizeof(double));
    if (!host_q) return;

    for (int step = 0; step < NUM_STEPS; step++)
    {
        // ----- 1. Q-Feld berechnen (CPU) -----
        clEnqueueReadBuffer(ocl->queue, buf->nodes, CL_TRUE, 0,
                            NUM_NODES * sizeof(double), h->nodes, 0, NULL, NULL);
        compute_bohm_potential(h->nodes, host_q, NUM_NODES, p->D);
        clEnqueueWriteBuffer(ocl->queue, buf->q_potential, CL_TRUE, 0,
                             NUM_NODES * sizeof(double), host_q, 0, NULL, NULL);

        // ----- 2. Motion-Kernel in Batches -----
        for (uint32_t offset = 0; offset < NUM_NODES; offset += BATCH_SIZE)
        {
            uint32_t current_batch_size = BATCH_SIZE;
            if (offset + current_batch_size > NUM_NODES)
                current_batch_size = NUM_NODES - offset;

            ocl_set_parameter_iwt_update_motion(
                kernel_motion,
                buf->nodes,
                buf->x, buf->y, buf->z,
                buf->vx, buf->vy, buf->vz,
                buf->adjacency,
                buf->q_potential,
                p->D, p->l0, p->G, p->k, p->Q,
                NUM_NODES, p->dt, offset
            );

            if (!ocl_enqueue_kernel(ocl, kernel_motion, current_batch_size, 256))
            {
                fprintf(stderr, "Fehler: Motion-Batch bei Offset %u, Schritt %d\n", offset, step);
                free(host_q);
                return;
            }
        }

        // ----- 3. Ergebnisse zurücklesen -----
        clEnqueueReadBuffer(ocl->queue, buf->nodes, CL_TRUE, 0,
                            NUM_NODES * sizeof(double), h->nodes, 0, NULL, NULL);
        clEnqueueReadBuffer(ocl->queue, buf->x, CL_TRUE, 0,
                            NUM_NODES * sizeof(double), h->x, 0, NULL, NULL);
        clEnqueueReadBuffer(ocl->queue, buf->y, CL_TRUE, 0,
                            NUM_NODES * sizeof(double), h->y, 0, NULL, NULL);
        clEnqueueReadBuffer(ocl->queue, buf->z, CL_TRUE, 0,
                            NUM_NODES * sizeof(double), h->z, 0, NULL, NULL);

        // ----- 4. Informationserhaltung prüfen -----
        double sum = 0.0;
        for (uint32_t i = 0; i < NUM_NODES; i++) sum += h->nodes[i];
        double deviation = sum - initial_sum;

        if (step % OUTPUT_INTERVAL == 0)
        {
            printf("Schritt %d:\n", step);
            for (int i = 0; i < 10; i++)
                printf("  I[%d] = %.10f\n", i, h->nodes[i]);
            printf("  Summe I: %.12f\n", sum);
            printf("  Abweichung: %.12e\n", deviation);
            printf("\n");
            export_positions(h, step);
        }
    }

    free(host_q);
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

    cl_kernel kernel_motion = ocl_get_kernel(&ocl, OCL_KERNEL_IWT_UPDATE_MOTION);
    cl_kernel kernel_q = ocl_get_kernel(&ocl, OCL_KERNEL_Q_FIELD);

    if (!kernel_motion)
    {
        fprintf(stderr, "Fehler: Kernel 'iwt_update_motion' nicht gefunden.\n");
        ocl_deinitialize(&ocl);
        return 1;
    }

    if (!kernel_q)
    {
        fprintf(stderr, "Fehler: Kernel 'compute_q_field' nicht gefunden.\n");
        ocl_deinitialize(&ocl);
        return 1;
    }

    IWTParams p = iwt_params_init();
    printf("IWT-Parameter (dimensionslos):\n");
    printf("  D  = %.6f\n", p.D);
    printf("  dt = %.6f\n", p.dt);
    printf("\n");

    HostData *h = host_data_alloc();
    if (!h)
    {
        fprintf(stderr, "Fehler: Host-Speicher fehlgeschlagen.\n");
        ocl_deinitialize(&ocl);
        return 1;
    }
    host_data_init(h);

    GPUBuffers buf = gpu_buffers_create(&ocl, h);
    if (!buf.nodes || !buf.x || !buf.y || !buf.z ||
        !buf.vx || !buf.vy || !buf.vz ||
        !buf.adjacency || !buf.flows || !buf.q_potential)
    {
        fprintf(stderr, "Fehler: GPU-Buffer fehlgeschlagen.\n");
        host_data_free(h);
        ocl_deinitialize(&ocl);
        return 1;
    }

    double initial_sum = 0.0;
    for (uint32_t i = 0; i < NUM_NODES; i++) initial_sum += h->nodes[i];
    printf("Initiale Summe I: %.12f\n", initial_sum);
    printf("\n");

    run_simulation_motion(&ocl, kernel_motion, kernel_q, &buf, h, &p, initial_sum);

    double final_sum = 0.0;
    for (uint32_t i = 0; i < NUM_NODES; i++) final_sum += h->nodes[i];
    printf("Endgültige Summe I: %.12f\n", final_sum);
    printf("Endgültige Abweichung: %.12e\n", final_sum - initial_sum);
    printf("Relative Abweichung: %.12e\n", (final_sum - initial_sum) / initial_sum);

    compute_force_spectrum(h, p.D);
    compute_wdbt_constants(p.D);

    // Letzte Positionen exportieren
    export_positions(h, NUM_STEPS);

    gpu_buffers_free(&buf);
    host_data_free(h);
    ocl_deinitialize(&ocl);

    return 0;
}