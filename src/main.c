#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <ocl/ocl.h>

// ============================================================
// Konstanten
// ============================================================

#define BATCH_SIZE 32768
#define NUM_NODES (BATCH_SIZE * 2)
#define NUM_EDGES (NUM_NODES * 12)
#define NUM_STEPS 10
#define OUTPUT_INTERVAL 1
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
    uint32_t *adjacency;
    double *flows;
} HostData;

typedef struct
{
    cl_mem nodes;
    cl_mem adjacency;
    cl_mem flows;
    cl_mem q_potential;
} GPUBuffers;

// ============================================================
// Umrechnung von IWT-Einheiten in SI-Einheiten
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
// Host-seitige Initialisierung (nur noch für Setup)
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
        .dt = 0.0001
    };
    return p;
}

HostData* host_data_alloc(void)
{
    HostData *h = malloc(sizeof(HostData));
    if (!h) return NULL;
    h->nodes = malloc(NUM_NODES * sizeof(double));
    h->adjacency = malloc(NUM_EDGES * sizeof(uint32_t));
    h->flows = malloc(NUM_EDGES * sizeof(double));
    if (!h->nodes || !h->adjacency || !h->flows)
    {
        free(h->nodes);
        free(h->adjacency);
        free(h->flows);
        free(h);
        return NULL;
    }
    return h;
}

void host_data_init(HostData *h)
{
    // Knotenwerte
    for (uint32_t i = 0; i < NUM_NODES; i++)
    {
        h->nodes[i] = 1.0 + 0.001 * (double)(i % 10);
    }

    // Adjazenzliste: Jeder Knoten hat 12 Nachbarn
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
    free(h->adjacency);
    free(h->flows);
    free(h);
}

// ============================================================
// GPU-Buffer
// ============================================================

GPUBuffers gpu_buffers_create(struct ocl_core *ocl, HostData *h)
{
    GPUBuffers b = {0};
    b.nodes = ocl_create_buffer(ocl, OCL_BUF_READ_WRITE, NUM_NODES * sizeof(double), h->nodes);
    b.adjacency = ocl_create_buffer(ocl, OCL_BUF_READ_ONLY, NUM_EDGES * sizeof(uint32_t), h->adjacency);
    b.flows = ocl_create_buffer(ocl, OCL_BUF_WRITE_ONLY, NUM_EDGES * sizeof(double), NULL);
    b.q_potential = ocl_create_buffer(ocl, OCL_BUF_READ_WRITE, NUM_NODES * sizeof(double), NULL);
    return b;
}

void gpu_buffers_free(GPUBuffers *b)
{
    clReleaseMemObject(b->nodes);
    clReleaseMemObject(b->adjacency);
    clReleaseMemObject(b->flows);
    clReleaseMemObject(b->q_potential);
}

// ============================================================
// Kraftspektrum (Host-seitig)
// ============================================================

void compute_force_spectrum(HostData *h, double D)
{
    double *spectrum = calloc(NUM_NODES, sizeof(double));
    if (!spectrum)
    {
        fprintf(stderr, "Fehler: Spektrum-Allokation fehlgeschlagen.\n");
        return;
    }

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
            double dist = (double)(j - i);
            if (dist < 0.0) dist = -dist;
            if (dist < 1.0) dist = 1.0;
            if (dist >= MAX_SPECTRUM_DIST) continue;

            double r = pow(dist, D - 2.0);
            double diff = I_i - I_j;
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
        {
            printf("  dist %2d: %.6e\n", d, spectrum[d]);
        }
    }
    free(spectrum);
}

// ============================================================
// WDBT+-Konstanten
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
    printf("  l0      = %.6e m (fundamentale Gitterkonstante)\n", l0_SI);
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
    printf("  L_Q0 = %.6e m (Korrelationslänge des Q-Feldes = Größe des Universums)\n", L_Q0_SI);

    printf("\nEffektive Hubble-Konstante:\n");
    printf("  ρ0 = %.6e kg/m³\n", rho0);
    printf("  CMB (d = %.2e m):   H = %.2f km/s/Mpc\n", d_cmb, H_cmb_km);
    printf("  Cepheiden (d = %.2e m): H = %.2f km/s/Mpc\n", d_ceph, H_ceph_km);
    printf("  SN Refsdal (d = %.2e m): H = %.2f km/s/Mpc\n", d_ref, H_ref_km);
    printf("  Rand (d = %.2e m):   H = %.2f km/s/Mpc\n", d_rand, H_rand_km);
    printf("\n  Hubble-Spannung: H_Cepheiden / H_CMB = %.3f\n", H_ceph_km / H_cmb_km);
    printf("  Beobachtung: 73.5 / 67.4 = %.3f\n", 73.5 / 67.4);
}

// ============================================================
// Simulation (Q-Feld auf GPU)
// ============================================================

void run_simulation(
    struct ocl_core *ocl,
    cl_kernel kernel_iwt,
    cl_kernel kernel_q,
    GPUBuffers *buf,
    HostData *h,
    IWTParams *p,
    double initial_sum
)
{
    // sqrt_rho-Buffer auf der GPU
    cl_mem sqrt_rho_buf = clCreateBuffer(
        ocl->context,
        CL_MEM_READ_ONLY,
        NUM_NODES * sizeof(double),
        NULL,
        NULL
    );

    if (!sqrt_rho_buf)
    {
        fprintf(stderr, "Fehler: sqrt_rho-Buffer konnte nicht erstellt werden.\n");
        return;
    }

    double *sqrt_rho_host = malloc(NUM_NODES * sizeof(double));
    if (!sqrt_rho_host)
    {
        fprintf(stderr, "Fehler: sqrt_rho_host konnte nicht allokiert werden.\n");
        clReleaseMemObject(sqrt_rho_buf);
        return;
    }

    double hbar = 1.054571817e-34;
    double mass = 1.0;

    for (int step = 0; step < NUM_STEPS; step++)
    {
        // ----- 1. nodes von GPU lesen für sum_I und sqrt_rho -----
        clEnqueueReadBuffer(
            ocl->queue,
            buf->nodes,
            CL_TRUE,
            0,
            NUM_NODES * sizeof(double),
            h->nodes,
            0,
            NULL,
            NULL
        );

        double sum_I = 0.0;
        for (uint32_t i = 0; i < NUM_NODES; i++)
        {
            sum_I += h->nodes[i];
        }

        for (uint32_t i = 0; i < NUM_NODES; i++)
        {
            sqrt_rho_host[i] = sqrt(h->nodes[i] / sum_I);
        }

        // sqrt_rho auf GPU kopieren
        clEnqueueWriteBuffer(
            ocl->queue,
            sqrt_rho_buf,
            CL_TRUE,
            0,
            NUM_NODES * sizeof(double),
            sqrt_rho_host,
            0,
            NULL,
            NULL
        );

        // ----- 2. Q-Feld in Batches berechnen (global) -----
        for (uint32_t offset = 0; offset < NUM_NODES; offset += BATCH_SIZE)
        {
            uint32_t current_batch_size = BATCH_SIZE;
            if (offset + current_batch_size > NUM_NODES)
            {
                current_batch_size = NUM_NODES - offset;
            }

            ocl_set_parameter_q_field(
                kernel_q,
                buf->nodes,
                sqrt_rho_buf,
                buf->q_potential,
                NUM_NODES,
                hbar,
                mass,
                offset
            );

            if (!ocl_enqueue_kernel(ocl, kernel_q, current_batch_size, 256))
            {
                fprintf(
                    stderr,
                    "Fehler: Q-Feld-Batch bei Offset %u, Schritt %d\n",
                    offset,
                    step
                );
                clReleaseMemObject(sqrt_rho_buf);
                free(sqrt_rho_host);
                return;
            }
        }

        // ----- 3. IWT-Update in Batches -----
        for (uint32_t offset = 0; offset < NUM_NODES; offset += BATCH_SIZE)
        {
            uint32_t current_batch_size = BATCH_SIZE;
            if (offset + current_batch_size > NUM_NODES)
            {
                current_batch_size = NUM_NODES - offset;
            }

            ocl_set_parameter_iwt_update(
                kernel_iwt,
                buf->nodes,
                buf->adjacency,
                buf->flows,
                buf->q_potential,
                p->D,
                p->l0,
                p->G,
                p->k,
                p->Q,
                NUM_NODES,
                p->dt,
                offset
            );

            if (!ocl_enqueue_kernel(ocl, kernel_iwt, current_batch_size, 256))
            {
                fprintf(
                    stderr,
                    "Fehler: IWT-Batch bei Offset %u, Schritt %d\n",
                    offset,
                    step
                );
                clReleaseMemObject(sqrt_rho_buf);
                free(sqrt_rho_host);
                return;
            }
        }

        // ----- 4. Ergebnisse zurücklesen -----
        clEnqueueReadBuffer(
            ocl->queue,
            buf->nodes,
            CL_TRUE,
            0,
            NUM_NODES * sizeof(double),
            h->nodes,
            0,
            NULL,
            NULL
        );

        double sum = 0.0;
        for (uint32_t i = 0; i < NUM_NODES; i++)
        {
            sum += h->nodes[i];
        }
        double deviation = sum - initial_sum;

        if (step % OUTPUT_INTERVAL == 0)
        {
            printf("Schritt %d:\n", step);
            for (int i = 0; i < 10; i++)
            {
                printf("  I[%d] = %.10f\n", i, h->nodes[i]);
            }
            printf("  Summe I: %.12f\n", sum);
            printf("  Abweichung: %.12e\n", deviation);
            printf("\n");
        }
    }

    clReleaseMemObject(sqrt_rho_buf);
    free(sqrt_rho_host);
}

// Nach der Simulation: Exportiere die Drift als CSV
void export_drift_csv(HostData *h, int num_steps)
{
    FILE *f = fopen("drift.csv", "w");
    fprintf(f, "index,start,end,drift_per_step\n");
    
    for (uint32_t i = 0; i < 100; i++) // erste 100 Knoten
    {
        double start_val = 1.0 + 0.001 * (double)(i % 10);
        double end_val = h->nodes[i];
        double drift = (end_val - start_val) / (double)num_steps;
        fprintf(f, "%u,%.10f,%.10f,%.6e\n", i, start_val, end_val, drift);
    }
    fclose(f);
    printf("Drift-Daten gespeichert: drift.csv\n");
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

    // Kernel laden
    cl_kernel kernel_iwt = ocl_get_kernel(&ocl, OCL_KERNEL_IWT_UPDATE);
    cl_kernel kernel_q = ocl_get_kernel(&ocl, OCL_KERNEL_Q_FIELD);

    if (!kernel_iwt)
    {
        fprintf(stderr, "Fehler: Kernel 'iwt_update' nicht gefunden.\n");
        ocl_deinitialize(&ocl);
        return 1;
    }

    if (!kernel_q)
    {
        fprintf(stderr, "Fehler: Kernel 'compute_q_field' nicht gefunden.\n");
        ocl_deinitialize(&ocl);
        return 1;
    }

    // Parameter
    IWTParams p = iwt_params_init();
    printf("IWT-Parameter (dimensionslos):\n");
    printf("  D  = %.6f\n", p.D);
    printf("  dt = %.6f\n", p.dt);
    printf("\n");

    // Host-Daten
    HostData *h = host_data_alloc();
    if (!h)
    {
        fprintf(stderr, "Fehler: Host-Speicher fehlgeschlagen.\n");
        ocl_deinitialize(&ocl);
        return 1;
    }
    host_data_init(h);

    // GPU-Buffer
    GPUBuffers buf = gpu_buffers_create(&ocl, h);
    if (!buf.nodes || !buf.adjacency || !buf.flows || !buf.q_potential)
    {
        fprintf(stderr, "Fehler: GPU-Buffer fehlgeschlagen.\n");
        host_data_free(h);
        ocl_deinitialize(&ocl);
        return 1;
    }

    // Initiale Summe
    double initial_sum = 0.0;
    for (uint32_t i = 0; i < NUM_NODES; i++)
    {
        initial_sum += h->nodes[i];
    }
    printf("Initiale Summe I: %.12f\n", initial_sum);
    printf("\n");

    // Simulation
    run_simulation(&ocl, kernel_iwt, kernel_q, &buf, h, &p, initial_sum);
	export_drift_csv(h, NUM_STEPS);

    // Endsumme
    double final_sum = 0.0;
    for (uint32_t i = 0; i < NUM_NODES; i++)
    {
        final_sum += h->nodes[i];
    }
    printf("Endgültige Summe I: %.12f\n", final_sum);
    printf("Endgültige Abweichung: %.12e\n", final_sum - initial_sum);
    printf("Relative Abweichung: %.12e\n", (final_sum - initial_sum) / initial_sum);

    // Kraftspektrum
    compute_force_spectrum(h, p.D);

    // Flüsse
    clEnqueueReadBuffer(
        ocl.queue,
        buf.flows,
        CL_TRUE,
        0,
        NUM_EDGES * sizeof(double),
        h->flows,
        0,
        NULL,
        NULL
    );
    printf("\nKräfte für Knoten 0 (erste 4 Flüsse):\n");
    for (int i = 0; i < 4; i++)
    {
        printf("  flows[%d] = %.6e\n", i, h->flows[i]);
    }

    // WDBT+-Konstanten
    compute_wdbt_constants(p.D);

    // SI-Umrechnung
    printf("\n=== Umrechnung in SI-Einheiten ===\n");
    double I0 = 1.0;
    double I_IWT = h->nodes[0];
    PhysicalQuantities pq = convert_iwt_to_si(I_IWT, p.D, p.dt, NUM_STEPS, I0);
    printf("Knoten 0 nach %d Schritten:\n", NUM_STEPS);
    printf("  I_IWT    = %.10f (dimensionslos)\n", I_IWT);
    printf("  I_phys   = %.6e (Normierung I0 = %.3e)\n", pq.I_phys, I0);
    printf("  Zeit     = %.6e s\n", pq.T_phys);
    printf("  Länge    = %.6e m (fundamental)\n", pq.L_phys);
    printf("  Masse    = %.6e kg\n", pq.M_phys);
    printf("  Energie  = %.6e J\n", pq.E_phys);
    printf("  Dichte   = %.6e kg/m³\n", pq.rho_phys);

    printf("\nAlle Knoten (I_phys in Normierung I0=%.3e):\n", I0);
    for (int i = 0; i < 10; i++)
    {
        PhysicalQuantities pq_i = convert_iwt_to_si(h->nodes[i], p.D, p.dt, NUM_STEPS, I0);
        printf("  nodes[%d] = %.6e\n", i, pq_i.I_phys);
    }

    // Aufräumen
    gpu_buffers_free(&buf);
    host_data_free(h);
    ocl_deinitialize(&ocl);

    return 0;
}