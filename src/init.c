#include "init.h"
#include "dodecahedron.h"
#include "iwt.h"
#include <api/api.h>
#include <math.h>
#include <stddef.h>
#include <stdio.h>

/**
 * init.c - Initialisierung des IWT-Informationsnetzwerks
 *
 * THEORIE: Kap. 2 "Axiome der IWT"
 *          Kap. 3 "Diskrete Informationsdynamik"
 *          Kap. 6 "Fraktale Informationsdimension D"
 *          Anhang A "Mathematische Grundlagen"
 *
 * Diese Datei implementiert die Initialisierung des diskreten Netzwerks:
 * - Allokation der Host- und GPU-Arrays für I_real, I_imag, I_phase, K, Q, etc.
 * - Erzeugung der fraktalen Knotenpositionen via dodecahedron.c
 * - Berechnung der Kopplungsmatrix K_kl = 1/d^(3-D) (Anhang A.2)
 * - Berechnung der Adjazenzmatrix aus K und cluster_threshold
 *
 * Die Kopplungsmatrix K_kl ist die Grundlage für:
 * - Die diskrete Metrik g_kl = K_kl / sqrt(K_kk * K_ll) (Kap. 5.2)
 * - Die lokale Weber-Dynamik (Kap. 8)
 * - Die Cluster-Erkennung via Flood-Fill (iwt_detect_cluster.c)
 */

static bool allocate_host_arrays(const iwt_runtime_t rt, const iwt_config_t cfg);
static bool init_positions(const iwt_runtime_t rt, const iwt_config_t cfg);
static void zero_coupling_matrix(const iwt_runtime_t rt, const iwt_config_t cfg);
static void compute_coupling_matrix(const iwt_runtime_t rt, const iwt_config_t cfg);
static bool init_host_memory(const iwt_runtime_t rt, const iwt_config_t cfg);
static bool allocate_gpu_buffers(const iwt_runtime_t rt, const iwt_config_t cfg);
static void free_cluster_arrays(const iwt_runtime_t rt);

static bool all_ptrs_valid(void* ptrs[], size_t n)
{
	for (size_t i = 0; i < n; i++)
	{
		if (!ptrs[i])
		{
			return false;
		}
	}
	return true;
}

private
void _free_memory(void** pp)
{
	if (*pp != NULL)
	{
		free(*pp);
		*pp = NULL;
	}
}

private
void _free_gpu_memory(cl_mem* pp)
{
	if (*pp != NULL)
	{
		clReleaseMemObject(*pp);
		*pp = NULL;
	}
}

bool initialize_host_data(const iwt_runtime_t rt, const iwt_config_t cfg)
{
	if (!allocate_host_arrays(rt, cfg))
	{
		return false;
	}
	if (!init_host_memory(rt, cfg))
	{
		return false;
	}
	return true;
}

/**
 * Allokiert alle Host-Arrays.
 * THEORIE: Die Arrays entsprechen den diskreten Feldern der IWT:
 *   I_real, I_imag   -> komplexes Informationsfeld I_k (Kap. 2, Axiom 1)
 *   I_phase          -> Phase phi_k (Anhang R)
 *   K                -> fraktale Kopplungsmatrix K_kl = 1/d^(3-D) (Anhang A.2)
 *   Q                -> Bohm-Potential (Kap. 12)
 *   mass, charge     -> emergente Masse und Ladung (Kap. 3.3)
 *   xi_real, xi_imag -> intrinsische Fluktuationen (Anhang P, Gleichung P.3)
 */
static bool allocate_host_arrays(const iwt_runtime_t rt, const iwt_config_t cfg)
{
	rt->I_real = calloc(cfg->N, sizeof(double));
	rt->I_imag = calloc(cfg->N, sizeof(double));
	rt->I_prev_real = calloc(cfg->N, sizeof(double));
	rt->I_prev_imag = calloc(cfg->N, sizeof(double));
	rt->I_phase = calloc(cfg->N, sizeof(double));
	rt->I_phase_prev = calloc(cfg->N, sizeof(double));

	rt->K = malloc(cfg->N * cfg->N * sizeof(double));
	rt->sumJ = malloc(cfg->N * sizeof(double));
	rt->Q = calloc(cfg->N, sizeof(double));

	rt->pos = calloc(cfg->N, sizeof(struct vector_3d));
	rt->adjacency = malloc(cfg->N * cfg->N * sizeof(bool));

	rt->mass = calloc(cfg->N, sizeof(double));
	rt->charge = calloc(cfg->N, sizeof(double));

	// Anhang P: Intrinsische Unschärfe aus diskreter Zeit
	rt->xi_real = calloc(cfg->N, sizeof(double));
	rt->xi_imag = calloc(cfg->N, sizeof(double));
	rt->uncertainty = calloc(cfg->N, sizeof(double));

	rt->cluster_capacity = 400;
	rt->clusters = calloc(rt->cluster_capacity, sizeof(struct iwt_cluster));
	rt->visited = calloc(cfg->N, sizeof(bool));

	for (size_t i = 0; i < rt->cluster_capacity; i++)
	{
		rt->clusters[i].node_indices = calloc(cfg->N, sizeof(size_t));
		rt->clusters[i].is_active = false;
	}

	void* ptrs[] = {
		rt->I_real, rt->I_imag, rt->I_prev_real, rt->I_prev_imag,
		rt->I_phase, rt->I_phase_prev,
		rt->K, rt->sumJ, rt->Q,
		rt->pos, rt->adjacency,
		rt->mass, rt->charge,
		rt->xi_real, rt->xi_imag, rt->uncertainty,
		rt->clusters, rt->visited};

	if (!all_ptrs_valid(ptrs, sizeof(ptrs) / sizeof(ptrs[0])))
	{
		return false;
	}

	for (size_t i = 0; i < rt->cluster_capacity; i++)
	{
		if (!rt->clusters[i].node_indices)
		{
			return false;
		}
	}

	return true;
}

static bool init_host_memory(const iwt_runtime_t rt, const iwt_config_t cfg)
{
	if (!init_positions(rt, cfg))
	{
		return false;
	}
	zero_coupling_matrix(rt, cfg);
	compute_coupling_matrix(rt, cfg); // K_kl = 1/d^(3-D) (Anhang A.2)
	iwt_recompute_adjacency(rt, cfg);
	return true;
}

/**
 * Erzeugt die Knotenpositionen auf dem fraktalen Dodekaeder-Gitter.
 * THEORIE: Die Positionen sind die Knoten des fraktalen Netzwerks.
 *          Die fraktale Distanz d_kl im Indexraum wird aus diesen Positionen
 *          berechnet (Anhang A.2).
 *          D ≈ 2,704 ist die Hausdorff-Dimension (Kap. 6.3).
 */
static bool init_positions(const iwt_runtime_t rt, const iwt_config_t cfg)
{
	double* tmp_x = malloc(cfg->N * sizeof(double));
	double* tmp_y = malloc(cfg->N * sizeof(double));
	double* tmp_z = malloc(cfg->N * sizeof(double));
	if (!tmp_x || !tmp_y || !tmp_z)
	{
		free(tmp_x);
		free(tmp_y);
		free(tmp_z);
		return false;
	}

	// extra_levels=1: 12 sichtbare "Wurzeln" (Selbstähnlichkeit)
	bool points_ok = dodecahedron_generate_points_ex(tmp_x, tmp_y, tmp_z, cfg->N, cfg->l0, 1);

	if (points_ok)
	{
		for (size_t i = 0; i < cfg->N; i++)
		{
			rt->pos[i].x = (ld) tmp_x[i];
			rt->pos[i].y = (ld) tmp_y[i];
			rt->pos[i].z = (ld) tmp_z[i];
		}
	}

	free(tmp_x);
	free(tmp_y);
	free(tmp_z);
	return points_ok;
}

static void zero_coupling_matrix(const iwt_runtime_t rt, const iwt_config_t cfg)
{
	for (size_t i = 0; i < cfg->N; i++)
	{
		for (size_t j = 0; j < cfg->N; j++)
		{
			rt->K[i * cfg->N + j] = 0.0;
		}
	}
}

/**
 * Berechnet die fraktale Kopplungsmatrix K_kl = 1 / d_kl^(3-D).
 * 
 * THEORIE: Anhang A.2, Gleichung (A.7); Kap. 2, Axiom 3; Kap. 5.2; Kap. 8.
 * 
 * Die Distanz d_kl ist die FRAKTALE DISTANZ IM INDEXRAUM.
 * Implementierung: d_kl = (||pos_k - pos_l|| / l0)^(1/D)
 * 
 * Diese Formel gilt FÜR ALLE KNOTENPAARE – auch über Dodekaeder-Grenzen hinweg.
 * Dadurch wird die Kopplung über Zellgrenzen korrekt berechnet und
 * die Information kann sich frei im gesamten fraktalen Netzwerk bewegen.
 * 
 * Die Kopplungsmatrix ist die Grundlage für:
 * - Die diskrete Metrik g_kl (Kap. 5.2)
 * - Die lokale Weber-Dynamik (Kap. 8)
 * - Die Adjazenzmatrix für die Cluster-Erkennung
 */
static void compute_coupling_matrix(const iwt_runtime_t rt, const iwt_config_t cfg)
{
    // Theoretische Werte: D ≈ 2.704, alpha = 3 - D ≈ 0.296
    double D = cfg->D;
    double alpha = 3.0 - D;         
    double l0 = cfg->l0;            // Fundamentale Längenskala (Kap. 6.4)

    for (size_t i = 0; i < cfg->N; i++) {
        for (size_t j = 0; j < cfg->N; j++) {
            if (i == j) {
                // Diagonale = 0 (keine Selbstkopplung, wie in der Theorie)
                rt->K[i * cfg->N + j] = 0.0;
                continue;
            }

            // Euklidische Distanz im 3D-Raum (wie ursprünglich)
            struct vector_3d vi = rt->pos[i];
            struct vector_3d vj = rt->pos[j];
            struct vector_3d dvec = vector_sub(&vi, &vj);
            ld dist_ld = vector_norm(&dvec);
            double dist_3d = (double)dist_ld;

            // Vermeidung von Singularitäten
            if (dist_3d < 1e-12) {
                dist_3d = 1e-12;
            }

            // ================================================================
            // KORREKTUR NACH THEORIE:
            // Fraktale Distanz im Indexraum: d_kl = (dist_3d / l0)^(1/D)
            // K_kl = 1 / d_kl^(3-D)
            // ================================================================
            
            // 1. Fraktale Distanz berechnen
            double d_ij = pow(dist_3d / l0, 1.0 / D);
            
            // 2. Kopplung berechnen (exakt nach Gleichung A.7)
            rt->K[i * cfg->N + j] = 1.0 / pow(d_ij, alpha);

            // ================================================================
            // ALTE (FALSCHE) IMPLEMENTIERUNG (auskommentiert zur Referenz):
            // double d_ij = pow(dist_3d, 1.0 / D);
            // rt->K[i * cfg->N + j] = 1.0 / pow(d_ij, alpha);
            // FEHLER: Fehlende Normierung auf l0 führte zu viel zu kleinen
            // Kopplungen zwischen entfernten Knoten, wodurch die Information
            // in den Dodekaeder-Zellen gefangen blieb.
            // ================================================================
        }
    }
}

/**
 * Berechnet die Adjazenzmatrix aus K und cluster_threshold.
 * THEORIE: Zwei Knoten sind verbunden, wenn ihre Kopplung K_kl > threshold.
 *          Dies wird für die Cluster-Erkennung via Flood-Fill verwendet
 *          (iwt_detect_cluster.c).
 *
 *          Die Cluster-Erkennung realisiert die automatische Strukturbildung
 *          aus der Evolutionsgleichung P.3 (Anhang P).
 */
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
	return allocate_gpu_buffers(rt, cfg);
}

static bool allocate_gpu_buffers(const iwt_runtime_t rt, const iwt_config_t cfg)
{
	rt->I_real_gpu = ocl_create_buffer(&rt->ocl, OCL_BUF_READ_WRITE, cfg->N * sizeof(double), NULL);
	rt->I_imag_gpu = ocl_create_buffer(&rt->ocl, OCL_BUF_READ_WRITE, cfg->N * sizeof(double), NULL);
	rt->I_prev_real_gpu = ocl_create_buffer(&rt->ocl, OCL_BUF_READ_WRITE, cfg->N * sizeof(double), NULL);
	rt->I_prev_imag_gpu = ocl_create_buffer(&rt->ocl, OCL_BUF_READ_WRITE, cfg->N * sizeof(double), NULL);
	rt->I_phase_gpu = ocl_create_buffer(&rt->ocl, OCL_BUF_READ_WRITE, cfg->N * sizeof(double), NULL);
	rt->I_phase_prev_gpu = ocl_create_buffer(&rt->ocl, OCL_BUF_READ_WRITE, cfg->N * sizeof(double), NULL);

	rt->K_gpu = ocl_create_buffer(&rt->ocl, OCL_BUF_READ_WRITE, cfg->N * cfg->N * sizeof(double), NULL);
	rt->sumJ_gpu = ocl_create_buffer(&rt->ocl, OCL_BUF_WRITE_ONLY, cfg->N * sizeof(double), NULL);
	rt->Q_gpu = ocl_create_buffer(&rt->ocl, OCL_BUF_READ_WRITE, cfg->N * sizeof(double), NULL);

	rt->mass_gpu = ocl_create_buffer(&rt->ocl, OCL_BUF_READ_WRITE, cfg->N * sizeof(double), NULL);
	rt->charge_gpu = ocl_create_buffer(&rt->ocl, OCL_BUF_READ_WRITE, cfg->N * sizeof(double), NULL);

	rt->xi_real_gpu = ocl_create_buffer(&rt->ocl, OCL_BUF_READ_WRITE, cfg->N * sizeof(double), NULL);
	rt->xi_imag_gpu = ocl_create_buffer(&rt->ocl, OCL_BUF_READ_WRITE, cfg->N * sizeof(double), NULL);
	rt->uncertainty_gpu = ocl_create_buffer(&rt->ocl, OCL_BUF_READ_WRITE, cfg->N * sizeof(double), NULL);

	void* gpu_ptrs[] = {
		rt->I_real_gpu, rt->I_imag_gpu, rt->I_prev_real_gpu, rt->I_prev_imag_gpu,
		rt->I_phase_gpu, rt->I_phase_prev_gpu,
		rt->K_gpu, rt->sumJ_gpu, rt->Q_gpu,
		rt->mass_gpu, rt->charge_gpu,
		rt->xi_real_gpu, rt->xi_imag_gpu, rt->uncertainty_gpu};

	return all_ptrs_valid(gpu_ptrs, sizeof(gpu_ptrs) / sizeof(gpu_ptrs[0]));
}

void deinitialize_host_data(const iwt_runtime_t rt)
{
	_free_memory((void**) &rt->I_real);
	_free_memory((void**) &rt->I_imag);
	_free_memory((void**) &rt->I_prev_real);
	_free_memory((void**) &rt->I_prev_imag);
	_free_memory((void**) &rt->I_phase);
	_free_memory((void**) &rt->I_phase_prev);
	_free_memory((void**) &rt->K);
	_free_memory((void**) &rt->sumJ);
	_free_memory((void**) &rt->Q);
	_free_memory((void**) &rt->pos);
	_free_memory((void**) &rt->adjacency);
	_free_memory((void**) &rt->mass);
	_free_memory((void**) &rt->charge);
	_free_memory((void**) &rt->xi_real);
	_free_memory((void**) &rt->xi_imag);
	_free_memory((void**) &rt->uncertainty);
	_free_memory((void**) &rt->visited);

	if (rt->clusters != NULL)
	{
		for (size_t i = 0; i < rt->cluster_capacity; i++)
		{
			_free_memory((void**) &rt->clusters[i].node_indices);
		}
		_free_memory((void**) &rt->clusters);
	}
}

void deinitialize_gpu_data(const iwt_runtime_t rt)
{
	_free_gpu_memory(&rt->I_real_gpu);
	_free_gpu_memory(&rt->I_imag_gpu);
	_free_gpu_memory(&rt->I_prev_real_gpu);
	_free_gpu_memory(&rt->I_prev_imag_gpu);
	_free_gpu_memory(&rt->I_phase_gpu);
	_free_gpu_memory(&rt->I_phase_prev_gpu);
	_free_gpu_memory(&rt->K_gpu);
	_free_gpu_memory(&rt->sumJ_gpu);
	_free_gpu_memory(&rt->Q_gpu);
	_free_gpu_memory(&rt->mass_gpu);
	_free_gpu_memory(&rt->charge_gpu);
	_free_gpu_memory(&rt->xi_real_gpu);
	_free_gpu_memory(&rt->xi_imag_gpu);
	_free_gpu_memory(&rt->uncertainty_gpu);
}