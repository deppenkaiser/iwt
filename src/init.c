#include "init.h"
#include "iwt.h"
#include "dodecahedron.h"
#include <stddef.h>
#include <stdio.h>
#include <math.h>
#include <api/api.h>

bool initialize_host_data(const iwt_runtime_t rt, const iwt_config_t cfg)
{
    bool retval = false;

    // === IWT-Kernfelder (komplex) ===
    rt->I_real = calloc(cfg->N, sizeof(double));
    rt->I_imag = calloc(cfg->N, sizeof(double));
    rt->I_prev_real = calloc(cfg->N, sizeof(double));
    rt->I_prev_imag = calloc(cfg->N, sizeof(double));
    rt->I_phase = calloc(cfg->N, sizeof(double));
    rt->I_phase_prev = calloc(cfg->N, sizeof(double));

    // === Kopplungsmatrix und Hilfsfelder ===
    rt->K = malloc(cfg->N * cfg->N * sizeof(double));
    rt->sumJ = malloc(cfg->N * sizeof(double));
    rt->Q = calloc(cfg->N, sizeof(double));

    // === Dodekaeder-Knotenpositionen (3D) ===
    rt->pos_x = malloc(cfg->N * sizeof(double));
    rt->pos_y = malloc(cfg->N * sizeof(double));
    rt->pos_z = malloc(cfg->N * sizeof(double));

    // === Nachbarschafts-Adjazenz (statisch, wird einmalig berechnet) ===
    rt->adjacency = malloc(cfg->N * cfg->N * sizeof(bool));

    // === Masse und Ladung ===
    rt->mass = calloc(cfg->N, sizeof(double));
    rt->charge = calloc(cfg->N, sizeof(double));

    // === Quantenfluktuationen (Anhang O & P) ===
    rt->xi_real = calloc(cfg->N, sizeof(double));
    rt->xi_imag = calloc(cfg->N, sizeof(double));
    rt->uncertainty = calloc(cfg->N, sizeof(double));

    // === Cluster-Verwaltung ===
    rt->cluster_capacity = 100;
    rt->clusters = calloc(rt->cluster_capacity, sizeof(struct iwt_cluster));
	rt->visited = calloc(cfg->N, sizeof(bool));

    for (size_t i = 0; i < rt->cluster_capacity; i++)
    {
        rt->clusters[i].node_indices = calloc(cfg->N, sizeof(size_t));
        rt->clusters[i].is_active = false;
    }

    // === Prüfung aller Allokationen ===
    if ((rt->I_real != NULL) && (rt->I_imag != NULL) &&
        (rt->I_prev_real != NULL) && (rt->I_prev_imag != NULL) &&
        (rt->I_phase != NULL) && (rt->I_phase_prev != NULL) &&
        (rt->K != NULL) && (rt->sumJ != NULL) &&
        (rt->Q != NULL) &&
        (rt->pos_x != NULL) && (rt->pos_y != NULL) && (rt->pos_z != NULL) &&
        (rt->adjacency != NULL) &&
        (rt->mass != NULL) && (rt->charge != NULL) &&
        (rt->xi_real != NULL) && (rt->xi_imag != NULL) &&
        (rt->uncertainty != NULL) && (rt->clusters != NULL) &&
		(rt->visited != NULL))
    {
        // ================================================================
        // FRAKTALE DODEKAEDER-KNOTENPOSITIONEN (mehrere Wurzeln im Gitter)
        // ================================================================
        bool points_ok = dodecahedron_generate_multi_root_points(
            rt->pos_x, rt->pos_y, rt->pos_z, cfg->N, cfg->l0,
            2, 2, 1, 3.0);

        if (points_ok)
        {
            // ================================================================
            // FRAKTALE KOPPLUNGSMATRIX (IWT-Kern)
            // ================================================================

            double D = cfg->D;
            double alpha = 3.0 - D;

            // 1. Alle Kopplungen auf 0 setzen
            for (size_t i = 0; i < cfg->N; i++)
            {
                for (size_t j = 0; j < cfg->N; j++)
                {
                    rt->K[i * cfg->N + j] = 0.0;
                }
            }

            // 2. Fraktale Kopplungen berechnen (3D-Dodekaeder-Abstaende)
            for (size_t i = 0; i < cfg->N; i++)
            {
                for (size_t j = 0; j < cfg->N; j++)
                {
                    if (i == j) continue;

                    double dx = rt->pos_x[i] - rt->pos_x[j];
                    double dy = rt->pos_y[i] - rt->pos_y[j];
                    double dz = rt->pos_z[i] - rt->pos_z[j];
                    double dist_3d = sqrt(dx * dx + dy * dy + dz * dz);
                    if (dist_3d < 1e-9) dist_3d = 1e-9;

                    // Fraktale Distanz
                    double d_ij = pow(dist_3d, 1.0 / D);
                    rt->K[i * cfg->N + j] = 1.0 / pow(d_ij, alpha);
                }
            }

            // 3. KEINE NORMIERUNG - die Kopplungsmatrix bleibt fraktal
            //    (Die Normierung würde die fraktale Struktur zerstören)

            // 4. Nachbarschafts-Adjazenz aus K-Matrix-Schwellwert ableiten
            //    (kann später jederzeit per iwt_recompute_adjacency() neu
            //    berechnet werden, z.B. wenn der Schwellwert live geändert wird)
            iwt_recompute_adjacency(rt, cfg);

            retval = true;
        }
    }

    return retval;
}

void iwt_recompute_adjacency(const iwt_runtime_t rt, const iwt_config_t cfg)
{
    for (size_t i = 0; i < cfg->N; i++)
    {
        for (size_t j = 0; j < cfg->N; j++)
        {
            rt->adjacency[i * cfg->N + j] =
                (i != j) && (rt->K[i * cfg->N + j] > cfg->cluster_threshold);
        }
    }
}

bool initialize_gpu_data(const iwt_runtime_t rt, const iwt_config_t cfg)
{
    // === IWT-Kernfelder (komplex) ===
    rt->I_real_gpu = ocl_create_buffer(&rt->ocl, OCL_BUF_READ_WRITE, cfg->N * sizeof(double), NULL);
    rt->I_imag_gpu = ocl_create_buffer(&rt->ocl, OCL_BUF_READ_WRITE, cfg->N * sizeof(double), NULL);
    rt->I_prev_real_gpu = ocl_create_buffer(&rt->ocl, OCL_BUF_READ_WRITE, cfg->N * sizeof(double), NULL);
    rt->I_prev_imag_gpu = ocl_create_buffer(&rt->ocl, OCL_BUF_READ_WRITE, cfg->N * sizeof(double), NULL);
    rt->I_phase_gpu = ocl_create_buffer(&rt->ocl, OCL_BUF_READ_WRITE, cfg->N * sizeof(double), NULL);
    rt->I_phase_prev_gpu = ocl_create_buffer(&rt->ocl, OCL_BUF_READ_WRITE, cfg->N * sizeof(double), NULL);

    // === Kopplungsmatrix und Hilfsfelder ===
    rt->K_gpu = ocl_create_buffer(&rt->ocl, OCL_BUF_READ_WRITE, cfg->N * cfg->N * sizeof(double), NULL);
    rt->sumJ_gpu = ocl_create_buffer(&rt->ocl, OCL_BUF_WRITE_ONLY, cfg->N * sizeof(double), NULL);
    rt->Q_gpu = ocl_create_buffer(&rt->ocl, OCL_BUF_READ_WRITE, cfg->N * sizeof(double), NULL);

    // === Masse und Ladung ===
    rt->mass_gpu = ocl_create_buffer(&rt->ocl, OCL_BUF_READ_WRITE, cfg->N * sizeof(double), NULL);
    rt->charge_gpu = ocl_create_buffer(&rt->ocl, OCL_BUF_READ_WRITE, cfg->N * sizeof(double), NULL);

    // === Quantenfluktuationen (Anhang O & P) ===
    rt->xi_real_gpu = ocl_create_buffer(&rt->ocl, OCL_BUF_READ_WRITE, cfg->N * sizeof(double), NULL);
    rt->xi_imag_gpu = ocl_create_buffer(&rt->ocl, OCL_BUF_READ_WRITE, cfg->N * sizeof(double), NULL);
    rt->uncertainty_gpu = ocl_create_buffer(&rt->ocl, OCL_BUF_READ_WRITE, cfg->N * sizeof(double), NULL);

    // === Prüfung aller GPU-Buffer ===
    bool all_buffers_valid =
        (rt->I_real_gpu != NULL) && (rt->I_imag_gpu != NULL) &&
        (rt->I_prev_real_gpu != NULL) && (rt->I_prev_imag_gpu != NULL) &&
        (rt->I_phase_gpu != NULL) && (rt->I_phase_prev_gpu != NULL) &&
        (rt->K_gpu != NULL) && (rt->sumJ_gpu != NULL) &&
        (rt->Q_gpu != NULL) &&
        (rt->mass_gpu != NULL) && (rt->charge_gpu != NULL) &&
        (rt->xi_real_gpu != NULL) && (rt->xi_imag_gpu != NULL) &&
        (rt->uncertainty_gpu != NULL);

    return all_buffers_valid;
}

private void _free_memory(void** pp)
{
    if (*pp != NULL)
    {
        free(*pp);
        *pp = NULL;
    }
}

private void _free_gpu_memory(cl_mem* pp)
{
    if (*pp != NULL)
    {
        clReleaseMemObject(*pp);
        *pp = NULL;
    }
}

void deinitialize_host_data(const iwt_runtime_t rt)
{
    // === IWT-Kernfelder (komplex) ===
    _free_memory((void**)&rt->I_real);
    _free_memory((void**)&rt->I_imag);
    _free_memory((void**)&rt->I_prev_real);
    _free_memory((void**)&rt->I_prev_imag);
    _free_memory((void**)&rt->I_phase);
    _free_memory((void**)&rt->I_phase_prev);

    // === Kopplungsmatrix und Hilfsfelder ===
    _free_memory((void**)&rt->K);
    _free_memory((void**)&rt->sumJ);
    _free_memory((void**)&rt->Q);

    // === Dodekaeder-Knotenpositionen (3D) ===
    _free_memory((void**)&rt->pos_x);
    _free_memory((void**)&rt->pos_y);
    _free_memory((void**)&rt->pos_z);

    // === Nachbarschafts-Adjazenz ===
    _free_memory((void**)&rt->adjacency);

    // === Masse und Ladung ===
    _free_memory((void**)&rt->mass);
    _free_memory((void**)&rt->charge);

    // === Quantenfluktuationen (Anhang O & P) ===
    _free_memory((void**)&rt->xi_real);
    _free_memory((void**)&rt->xi_imag);
    _free_memory((void**)&rt->uncertainty);

    // === Cluster ===
	_free_memory((void**)&rt->visited);

    if (rt->clusters != NULL)
    {
        for (size_t i = 0; i < rt->cluster_capacity; i++)
        {
            _free_memory((void**)&rt->clusters[i].node_indices);
        }
        _free_memory((void**)&rt->clusters);
    }
}

void deinitialize_gpu_data(const iwt_runtime_t rt)
{
    // === IWT-Kernfelder (komplex) ===
    _free_gpu_memory(&rt->I_real_gpu);
    _free_gpu_memory(&rt->I_imag_gpu);
    _free_gpu_memory(&rt->I_prev_real_gpu);
    _free_gpu_memory(&rt->I_prev_imag_gpu);
    _free_gpu_memory(&rt->I_phase_gpu);
    _free_gpu_memory(&rt->I_phase_prev_gpu);

    // === Kopplungsmatrix und Hilfsfelder ===
    _free_gpu_memory(&rt->K_gpu);
    _free_gpu_memory(&rt->sumJ_gpu);
    _free_gpu_memory(&rt->Q_gpu);

    // === Masse und Ladung ===
    _free_gpu_memory(&rt->mass_gpu);
    _free_gpu_memory(&rt->charge_gpu);

    // === Quantenfluktuationen (Anhang O & P) ===
    _free_gpu_memory(&rt->xi_real_gpu);
    _free_gpu_memory(&rt->xi_imag_gpu);
    _free_gpu_memory(&rt->uncertainty_gpu);
}
