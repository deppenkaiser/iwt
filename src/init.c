#include "init.h"
#include "dodecahedron.h"
#include "iwt.h"
#include "iwt_kernel.h"
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
	rt->adj_flat = calloc((size_t) cfg->N * IWT_ADJ_STRIDE, sizeof(int));
	rt->adj_count = calloc(cfg->N, sizeof(int));
	rt->wave_flat = calloc((size_t) cfg->N * IWT_WAVE_STRIDE, sizeof(int));
	rt->wave_count = calloc(cfg->N, sizeof(int));

	rt->mass = calloc(cfg->N, sizeof(double));
	rt->charge = calloc(cfg->N, sizeof(double));

	// Anhang P: Intrinsische Unschärfe aus diskreter Zeit
	rt->xi_real = calloc(cfg->N, sizeof(double));
	rt->xi_imag = calloc(cfg->N, sizeof(double));
	rt->uncertainty = calloc(cfg->N, sizeof(double));

	rt->cluster_capacity = 400;
	rt->clusters = calloc(rt->cluster_capacity, sizeof(struct iwt_cluster));
	rt->visited = calloc(cfg->N, sizeof(bool));

	// Persistence: Schatten-Array für Cluster des vorherigen Schritts.
	// node_indices bleibt NULL (nur pos/vel/id werden fuer das Matching benoetigt).
	rt->clusters_prev = calloc(rt->cluster_capacity, sizeof(struct iwt_cluster));
	rt->cluster_count_prev = 0;

	// EMA-geglättete Masse + Hysterese-Flag fuer flimmerfreie Cluster-Detektion
	rt->mass_smooth = calloc(cfg->N, sizeof(double));
	rt->was_member = calloc(cfg->N, sizeof(bool));
	rt->n_steps = 0;

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
		rt->clusters, rt->clusters_prev, rt->visited,
		rt->mass_smooth, rt->was_member,
		rt->adj_flat, rt->adj_count,
		rt->wave_flat, rt->wave_count};

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

	// extra_levels aus cfg: 0 = einzelner Dodekaeder, 1 = 12 Wurzeln, ...
	bool points_ok = dodecahedron_generate_points_ex(tmp_x, tmp_y, tmp_z, cfg->N, cfg->l0, cfg->extra_levels);

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
	double D = cfg->D;
	double alpha = 3.0 - D;
	double s = 2.0 + (1.0 + sqrt(5.0)) / 2.0;
	double l0 = cfg->l0;
	double scale_coupling = 1.0 / pow(s, alpha);

	// 1. Skala jedes Knotens bestimmen
	int* scale = calloc(cfg->N, sizeof(int));
	if (!scale)
	{
		return;
	}

	for (size_t i = 0; i < cfg->N; i++)
	{
		struct vector_3d pos = rt->pos[i];
		double r = (double) vector_norm(&pos);
		scale[i] = (int) (log(r / l0 + 1.0) / log(s));
		if (scale[i] < 0)
		{
			scale[i] = 0;
		}
	}

	// 2. Kopplungsmatrix berechnen
	for (size_t i = 0; i < cfg->N; i++)
	{
		for (size_t j = 0; j < cfg->N; j++)
		{
			if (i == j)
			{
				rt->K[i * cfg->N + j] = 0.0;
				continue;
			}

			int scale_i = scale[i];
			int scale_j = scale[j];
			int scale_diff = abs(scale_i - scale_j);

			// ============================================================
			// SCHRITT 1: Kopplung aus der fraktalen Distanz (BLEIBT)
			// ============================================================
			struct vector_3d vi = rt->pos[i];
			struct vector_3d vj = rt->pos[j];
			struct vector_3d dvec = vector_sub(&vi, &vj);
			ld dist_ld = vector_norm(&dvec);
			double dist_3d = (double) dist_ld;
			if (dist_3d < 1e-12)
			{
				dist_3d = 1e-12;
			}

			// Fraktale Distanz (korrigiert, bleibt erhalten)
			double d_ij = pow(dist_3d / l0, 1.0 / D);
			double K_ij = 1.0 / pow(d_ij, alpha);

			// ============================================================
			// SCHRITT 2: Skalen-Normierung (NEU, ERGÄNZEND)
			// ============================================================
			if (scale_diff == 0)
			{
				// Gleiche Skala: Kopplung bleibt wie berechnet (oder wird verstärkt)
				// Hier belassen wir K_ij, weil es die Feinstruktur innerhalb der Skala abbildet
				rt->K[i * cfg->N + j] = K_ij;
			}
			else
			{
				// Unterschiedliche Skalen: Kopplung wird durch die Skalendifferenz begrenzt
				// Die Kopplung darf nicht größer sein als die theoriekonforme Skalen-Kopplung
				double K_scale = pow(scale_coupling, scale_diff);
				rt->K[i * cfg->N + j] = fmin(K_ij, K_scale);
			}
		}
	}

	free(scale);
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

	// Komprimierte Nachbarschaftslisten pflegen (O(N·deg) statt O(N²)-Scans)
	for (size_t i = 0; i < cfg->N; i++)
	{
		int count = 0;
		const bool* row = &rt->adjacency[i * cfg->N];
		for (size_t j = 0; j < cfg->N && count < IWT_ADJ_STRIDE; j++)
		{
			if (row[j])
			{
				rt->adj_flat[i * IWT_ADJ_STRIDE + count++] = (int) j;
			}
		}
		rt->adj_count[i] = count;
	}

	// ================================================================
	// Erweitertes Wellen-Kantennetz: Alle Knoten mit K > wave_k_min sind
	// Kandidaten (schwerer Kopplungsschweif reicht weit in den Raum),
	// pro Knoten werden die IWT_WAVE_STRIDE naechsten ausgewaehlt.
	// IWT_NORM: Visualisierungs-Graph, keine Dynamik-Aenderung.
	// Theorie: Konturen brauchen Kanten; die fraktale Kopplung klingt
	// nur langsam ab (alpha = 3-D ≈ 0.296), daher ueberbruecken die
	// staerksten Fern-Kopplungen die Zwischenraeume der Packung.
	// ================================================================
	for (size_t i = 0; i < cfg->N; i++)
	{
		double best_d2[IWT_WAVE_STRIDE];
		int best_j[IWT_WAVE_STRIDE];
		int cnt = 0;
		struct vector_3d pi_pos = rt->pos[i];

		for (size_t j = 0; j < cfg->N; j++)
		{
			if (j == i || !(rt->K[i * cfg->N + j] > cfg->wave_k_min))
			{
				continue;
			}
			struct vector_3d dvec = vector_sub(&rt->pos[j], &pi_pos);
			double d2 = (double) vector_dot(&dvec, &dvec);

			if (cnt < IWT_WAVE_STRIDE)
			{
				int p = cnt++;
				while (p > 0 && best_d2[p - 1] > d2)
				{
					best_d2[p] = best_d2[p - 1];
					best_j[p] = best_j[p - 1];
					p--;
				}
				best_d2[p] = d2;
				best_j[p] = (int) j;
			}
			else if (d2 < best_d2[cnt - 1])
			{
				int p = cnt - 1;
				while (p > 0 && best_d2[p - 1] > d2)
				{
					best_d2[p] = best_d2[p - 1];
					best_j[p] = best_j[p - 1];
					p--;
				}
				best_d2[p] = d2;
				best_j[p] = (int) j;
			}
		}

		rt->wave_count[i] = cnt;
		for (int n = 0; n < cnt; n++)
		{
			rt->wave_flat[(size_t) i * IWT_WAVE_STRIDE + n] = best_j[n];
		}
	}
}

bool initialize_gpu_data(const iwt_runtime_t rt, const iwt_config_t cfg)
{
	return allocate_gpu_buffers(rt, cfg);
}

bool iwt_rebuild_geometry(const iwt_runtime_t rt, const iwt_config_t cfg)
{
	if (!init_host_memory(rt, cfg))
	{
		return false;
	}
	iwt_k_gpu_set_uploaded(false);
	return clEnqueueWriteBuffer(rt->ocl.queue, rt->K_gpu, CL_TRUE, 0,
								cfg->N * cfg->N * sizeof(double), rt->K, 0, NULL,
								NULL) == CL_SUCCESS;
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

	_free_memory((void**) &rt->mass_smooth);
	_free_memory((void**) &rt->was_member);
	_free_memory((void**) &rt->clusters_prev);
	_free_memory((void**) &rt->adj_flat);
	_free_memory((void**) &rt->adj_count);
	_free_memory((void**) &rt->wave_flat);
	_free_memory((void**) &rt->wave_count);
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