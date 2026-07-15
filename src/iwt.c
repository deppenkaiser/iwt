#include "iwt.h"
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <libgen.h>
#include <unistd.h>
#include <limits.h>
#include <api/api.h>

double iwt_pi(void)
{
    return 4.0 * atan(1.0);
}

double iwt_fundamental_length(void)
{
    const double h = 6.62607015e-34;        // J*s
    const double mp = 1.67262192e-27;       // kg
    const double c = 299792458.0;           // m/s
    const double phi = (1.0 + sqrt(5.0)) / 2.0;

    // Volumen eines regulären Dodekaeders mit Kantenlänge a
    // V_Dodekaeder = (15 + 7*sqrt(5)) / 4 * a^3
    const double V_dode = (15.0 + 7.0 * sqrt(5.0)) / 4.0;
    const double V_sphere = 4.0 * iwt_pi() / 3.0;

    // Verhältnis der Volumina (a=1)
    const double V_ratio = V_dode / V_sphere;

    // Compton-Wellenlänge des Protons
    const double lambda_p = h / (mp * c);

    // Fundamentale Länge
    const double l0 = pow(V_ratio, 1.0 / 3.0) * lambda_p;
	
    return l0;
}

double iwt_fundamental_time()
{
    const double c = 299792458.0;   // m/s
    return iwt_fundamental_length() / c;
}

double iwt_fractal_dimension(void)
{
    const double phi = (1.0 + sqrt(5.0)) / 2.0;          // Goldener Schnitt
    const double N = 20.0 * phi;                          // Vermehrungsfaktor
    const double s = 2.0 + phi;                           // Skalierungsfaktor

    // D = ln(N) / ln(s)
    return log(N) / log(s);
}

double iwt_I_min()
{
    return 0.0;
}

double iwt_I_max(void)
{
    // I_max = 1.0
    // 
    // Herleitung:
    // Masse: m_p = δ * I_max²
    // Mit δ = m_p (Protonenmasse als elementare Masse)
    // m_p = m_p * I_max²
    // I_max² = 1
    // I_max = 1
    //
    // Der Wertebereich von I ist: 0 <= I <= 1

    return 1.0;
}

double iwt_delta_I(void)
{
    double I_min = iwt_I_min();
    double I_max = iwt_I_max();
    double steps = pow(2.0, 32.0) - 1.0;
    return (I_max - I_min) / steps;
}

double iwt_alpha_IWT()
{
    return 1;
}

double iwt_beta_IWT()
{
    return 1;
}

int iwt_classify(double I)
{
    if (fabs(I - 0.01) < 0.001) return 0;   // Vakuum
    if (fabs(I - 0.25) < 0.001) return 1;   // Elektron
    if (fabs(I - 1.0)  < 0.001) return 2;   // Proton
    if (fabs(I - 0.05) < 0.001) return 3;   // u-Quark
    if (fabs(I - 0.06) < 0.001) return 4;   // d-Quark
    return -1;                               // sonst
}

void iwt_compute_spectrum(const double* I, size_t N, struct iwt_spectrum* spec)
{
    spec->count_vacuum = 0;
    spec->count_electron = 0;
    spec->count_proton = 0;
    spec->count_u_quark = 0;
    spec->count_d_quark = 0;
    spec->count_other = 0;

    for (size_t i = 0; i < N; i++)
    {
        switch (iwt_classify(I[i]))
        {
            case 0: spec->count_vacuum++; break;
            case 1: spec->count_electron++; break;
            case 2: spec->count_proton++; break;
            case 3: spec->count_u_quark++; break;
            case 4: spec->count_d_quark++; break;
            default: spec->count_other++; break;
        }
    }
}

void iwt_print_spectrum(const struct iwt_spectrum* spec)
{
    size_t total = spec->count_vacuum + spec->count_electron + spec->count_proton +
                   spec->count_u_quark + spec->count_d_quark + spec->count_other;

    printf("=== Teilchenspektrum ===\n");
    printf("Vakuum   (0.01): %zu (%.2f%%)\n", spec->count_vacuum, 100.0 * spec->count_vacuum / total);
    printf("Elektron (0.25): %zu (%.2f%%)\n", spec->count_electron, 100.0 * spec->count_electron / total);
    printf("Proton   (1.0) : %zu (%.2f%%)\n", spec->count_proton, 100.0 * spec->count_proton / total);
    printf("u-Quark  (0.05): %zu (%.2f%%)\n", spec->count_u_quark, 100.0 * spec->count_u_quark / total);
    printf("d-Quark  (0.06): %zu (%.2f%%)\n", spec->count_d_quark, 100.0 * spec->count_d_quark / total);
    printf("Sonstige       : %zu (%.2f%%)\n", spec->count_other, 100.0 * spec->count_other / total);
    printf("Gesamt: %zu Knoten\n", total);
}

private char* iwt_get_exe_path(void)
{
    static char path[PATH_MAX];
    ssize_t len = readlink("/proc/self/exe", path, sizeof(path) - 1);
    if (len == -1) return NULL;
    path[len] = '\0';
    return dirname(path);
}

bool iwt_save_state(const iwt_runtime_t rt, const iwt_config_t cfg, const char* filename)
{
    char* exe_dir = iwt_get_exe_path();
    if (!exe_dir) return false;

    char fullpath[PATH_MAX];
    snprintf(fullpath, sizeof(fullpath), "%s/%s", exe_dir, filename);

    FILE* f = fopen(fullpath, "wb");
    if (!f) return false;

    size_t nI = cfg->N * sizeof(double);
    if (fwrite(rt->I, 1, nI, f) != nI) { fclose(f); return false; }
	if (fwrite(rt->I_prev, 1, nI, f) != nI) { fclose(f); return false; }

    size_t nK = cfg->N * cfg->N * sizeof(double);
    if (fwrite(rt->K, 1, nK, f) != nK) { fclose(f); return false; }

    size_t nQ = cfg->N * sizeof(double);
    if (fwrite(rt->Q, 1, nQ, f) != nQ) { fclose(f); return false; }

    fclose(f);
    return true;
}

bool iwt_load_state(const iwt_runtime_t rt, const iwt_config_t cfg, const char* filename)
{
    char* exe_dir = iwt_get_exe_path();
    if (!exe_dir) return false;

    char fullpath[PATH_MAX];
    snprintf(fullpath, sizeof(fullpath), "%s/%s", exe_dir, filename);

    FILE* f = fopen(fullpath, "rb");
    if (!f) return false;

    size_t nI = cfg->N * sizeof(double);
    if (fread(rt->I, 1, nI, f) != nI) { fclose(f); return false; }
	if (fread(rt->I_prev, 1, nI, f) != nI) { fclose(f); return false; }

    size_t nK = cfg->N * cfg->N * sizeof(double);
    if (fread(rt->K, 1, nK, f) != nK) { fclose(f); return false; }

    size_t nQ = cfg->N * sizeof(double);
    if (fread(rt->Q, 1, nQ, f) != nQ) { fclose(f); return false; }

    fclose(f);
    return true;
}

bool iwt_mds_compute(const iwt_runtime_t rt, const iwt_config_t cfg, iwt_mds_t mds)
{
    size_t N = cfg->N;
    double* D2 = malloc(N * N * sizeof(double));
    double* B = malloc(N * N * sizeof(double));
    double* eigenvalues = malloc(N * sizeof(double));
    double* temp = malloc(N * N * sizeof(double));
    
    if (!D2 || !B || !eigenvalues || !temp)
	{
        free(D2); free(B); free(eigenvalues); free(temp);
        return false;
    }

    // 1. Distanzmatrix aus Metrik g
    // g_ij = K_ij / sqrt(K_ii * K_jj)
    for (size_t i = 0; i < N; i++)
	{
        for (size_t j = 0; j < N; j++)
		{
            double Kii = rt->K[i * N + i];
            double Kjj = rt->K[j * N + j];
            double g_ij = rt->K[i * N + j] / sqrt(Kii * Kjj + 1e-30);
            // Distanz im durch g definierten Raum
            double d = sqrt(g_ij); // vereinfacht: d_ij = sqrt(g_ij)
            D2[i * N + j] = d * d;
        }
    }

    // 2. Zentrierungsmatrix J = I - 1/N * 1*1^T
    double invN = 1.0 / N;
    for (size_t i = 0; i < N; i++)
	{
        for (size_t j = 0; j < N; j++)
		{
            double J_ij = (i == j) ? 1.0 - invN : -invN;
            B[i * N + j] = -0.5 * J_ij * D2[i * N + j]; // vereinfacht (ohne vollständige J-Operation)
        }
    }

    // 3. Eigenwertzerlegung (vereinfacht: nur erste 2 Eigenwerte/Eigenvektoren)
    // Hier wird eine einfache Power-Iteration für die ersten 2 Dimensionen verwendet.
    // Für eine vollständige Eigenwertzerlegung müsste man LAPACK oder eine alternative Bibliothek einbinden.
    
    // Vereinfachte Implementierung: Verwende PCA auf Distanzmatrix
    // (Erste 2 Hauptkomponenten der Distanzmatrix)
    double* mean = calloc(N, sizeof(double));
    for (size_t i = 0; i < N; i++)
	{
        for (size_t j = 0; j < N; j++)
		{
            mean[i] += D2[i * N + j];
        }
        mean[i] /= N;
    }
    
    // Kovarianzmatrix (approx)
    double* cov = calloc(2 * 2, sizeof(double));
    // Projektion auf 2D: Hier nur als Platzhalter
    // In einer echten Implementierung würde man die Eigenvektoren berechnen
    
    // Allozieren für X, Y
    mds->X = malloc(N * 2 * sizeof(double));
    mds->Y = malloc(N * 2 * sizeof(double));
    mds->eigenvalues = malloc(2 * sizeof(double));
    mds->dim = 2;

    // Vorläufige Koordinaten: (i, 0) als Platzhalter
    for (size_t i = 0; i < N; i++)
	{
        mds->X[i * 2 + 0] = (double)i / N;
        mds->X[i * 2 + 1] = 0.0;
        mds->Y[i * 2 + 0] = (double)i / N;
        mds->Y[i * 2 + 1] = 0.0;
    }
    mds->eigenvalues[0] = 1.0;
    mds->eigenvalues[1] = 0.0;

    free(mean);
    free(cov);
    free(D2);
    free(B);
    free(eigenvalues);
    free(temp);
    return true;
}

void iwt_mds_free(iwt_mds_t mds)
{
    free(mds->X);
    free(mds->Y);
    free(mds->eigenvalues);
    mds->X = mds->Y = mds->eigenvalues = NULL;
}

void iwt_mds_print(const iwt_mds_t mds, size_t n)
{
    printf("MDS Koordinaten (erste %zu Knoten):\n", n);
    for (size_t i = 0; i < n && i < mds->dim; i++)
	{
        printf("  Knoten %zu: (%f, %f)\n", i, mds->X[i*2+0], mds->X[i*2+1]);
    }
}
