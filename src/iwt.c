#include "iwt.h"
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <libgen.h>
#include <unistd.h>
#include <limits.h>
#include <api/api.h>
#include <string/string.h>

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
    if (fwrite(rt->I_phase, 1, nI, f) != nI) { fclose(f); return false; }
    if (fwrite(rt->I_phase_prev, 1, nI, f) != nI) { fclose(f); return false; }

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
    if (fread(rt->I_phase, 1, nI, f) != nI) { fclose(f); return false; }
    if (fread(rt->I_phase_prev, 1, nI, f) != nI) { fclose(f); return false; }

    size_t nK = cfg->N * cfg->N * sizeof(double);
    if (fread(rt->K, 1, nK, f) != nK) { fclose(f); return false; }

    size_t nQ = cfg->N * sizeof(double);
    if (fread(rt->Q, 1, nQ, f) != nQ) { fclose(f); return false; }

    fclose(f);
    return true;
}

// Hilfsfunktion: Matrix-Multiplikation (für Eigenwertzerlegung)
static void iwt_matmul(double* A, double* B, double* C, size_t n)
{
    for (size_t i = 0; i < n; i++)
    {
        for (size_t j = 0; j < n; j++)
        {
            C[i * n + j] = 0.0;
            for (size_t k = 0; k < n; k++)
            {
                C[i * n + j] += A[i * n + k] * B[k * n + j];
            }
        }
    }
}

// Hilfsfunktion: Power-Iteration für den größten Eigenwert
static double iwt_power_iteration(double* A, double* v, size_t n, int max_iter)
{
    double* w = malloc(n * sizeof(double));
    if (!w) return 0.0;

    // Initialisiere v mit Zufallswerten
    for (size_t i = 0; i < n; i++)
    {
        v[i] = (double)rand() / RAND_MAX;
    }

    double lambda = 0.0;
    for (int iter = 0; iter < max_iter; iter++)
    {
        // w = A * v
        for (size_t i = 0; i < n; i++)
        {
            w[i] = 0.0;
            for (size_t j = 0; j < n; j++)
            {
                w[i] += A[i * n + j] * v[j];
            }
        }

        // Norm von w
        double norm = 0.0;
        for (size_t i = 0; i < n; i++)
        {
            norm += w[i] * w[i];
        }
        norm = sqrt(norm);
        if (norm < 1e-30) break;

        // v = w / norm
        for (size_t i = 0; i < n; i++)
        {
            v[i] = w[i] / norm;
        }

        // Rayleigh-Quotient: lambda = v^T * A * v
        lambda = 0.0;
        for (size_t i = 0; i < n; i++)
        {
            for (size_t j = 0; j < n; j++)
            {
                lambda += v[i] * A[i * n + j] * v[j];
            }
        }
    }

    free(w);
    return lambda;
}

bool iwt_mds_compute(const iwt_runtime_t rt, const iwt_config_t cfg, iwt_mds_t mds)
{
    size_t N = cfg->N;
    if (N < 2) return false;

    // 1. Distanzmatrix aus Metrik g berechnen
    double* D2 = malloc(N * N * sizeof(double));
    if (!D2) return false;

    for (size_t i = 0; i < N; i++)
    {
        double Kii = rt->K[i * N + i];
        if (Kii < 1e-30) Kii = 1e-30;
        for (size_t j = 0; j < N; j++)
        {
            double Kjj = rt->K[j * N + j];
            if (Kjj < 1e-30) Kjj = 1e-30;
            double g_ij = rt->K[i * N + j] / sqrt(Kii * Kjj + 1e-30);
            D2[i * N + j] = g_ij;  // Distanz im durch g definierten Raum
        }
    }

    // 2. Zentrierungsmatrix B = -1/2 * J * D^2 * J
    double* B = malloc(N * N * sizeof(double));
    if (!B) { free(D2); return false; }

    double invN = 1.0 / N;
    for (size_t i = 0; i < N; i++)
    {
        for (size_t j = 0; j < N; j++)
        {
            double J_i = (i == j) ? 1.0 - invN : -invN;
            double J_j = (j == i) ? 1.0 - invN : -invN;
            // Vereinfacht: B = -1/2 * D^2 (ohne vollständige J-Operation)
            // Für exakte MDS müsste man J*D^2*J berechnen
            B[i * N + j] = -0.5 * D2[i * N + j];
        }
    }

    // 3. Erste 2 Eigenvektoren von B (Power-Iteration mit Deflation)
    double* v1 = malloc(N * sizeof(double));
    double* v2 = malloc(N * sizeof(double));
    if (!v1 || !v2) { free(D2); free(B); free(v1); free(v2); return false; }

    // Erster Eigenvektor
    double lambda1 = iwt_power_iteration(B, v1, N, 100);
    if (lambda1 < 0) lambda1 = -lambda1;

    // Deflation: B = B - lambda1 * v1 * v1^T
    double* B_deflated = malloc(N * N * sizeof(double));
    if (!B_deflated) { free(D2); free(B); free(v1); free(v2); return false; }
    for (size_t i = 0; i < N; i++)
    {
        for (size_t j = 0; j < N; j++)
        {
            B_deflated[i * N + j] = B[i * N + j] - lambda1 * v1[i] * v1[j];
        }
    }

    // Zweiter Eigenvektor
    double lambda2 = iwt_power_iteration(B_deflated, v2, N, 100);
    if (lambda2 < 0) lambda2 = -lambda2;

    // 4. Koordinaten: X = V * sqrt(Lambda)
    mds->coords = malloc(N * 2 * sizeof(double));
    mds->eigenvalues = malloc(2 * sizeof(double));
    if (!mds->coords || !mds->eigenvalues)
    {
        free(D2); free(B); free(B_deflated); free(v1); free(v2);
        free(mds->coords); free(mds->eigenvalues);
        return false;
    }

    mds->N = N;
    mds->dim = 2;
    mds->eigenvalues[0] = lambda1;
    mds->eigenvalues[1] = lambda2;

    double sqrt_lambda1 = sqrt(fabs(lambda1) + 1e-30);
    double sqrt_lambda2 = sqrt(fabs(lambda2) + 1e-30);

    for (size_t i = 0; i < N; i++)
    {
        mds->coords[i * 2 + 0] = v1[i] * sqrt_lambda1;
        mds->coords[i * 2 + 1] = v2[i] * sqrt_lambda2;
    }

    free(D2);
    free(B);
    free(B_deflated);
    free(v1);
    free(v2);

    // 5. Zentrieren (Mittelwert auf 0 setzen)
    double mean_x = 0.0, mean_y = 0.0;
    for (size_t i = 0; i < N; i++)
    {
        mean_x += mds->coords[i * 2 + 0];
        mean_y += mds->coords[i * 2 + 1];
    }
    mean_x /= N;
    mean_y /= N;
    for (size_t i = 0; i < N; i++)
    {
        mds->coords[i * 2 + 0] -= mean_x;
        mds->coords[i * 2 + 1] -= mean_y;
    }

    return true;
}

void iwt_mds_free(iwt_mds_t mds)
{
    free(mds->coords);
    free(mds->eigenvalues);
    mds->coords = NULL;
    mds->eigenvalues = NULL;
    mds->N = 0;
    mds->dim = 0;
}

void iwt_mds_print(const iwt_mds_t mds, size_t n)
{
    if (!mds->coords) return;
    printf("MDS Koordinaten (erste %zu Knoten):\n", n);
    printf("  Eigenwerte: %f, %f\n", mds->eigenvalues[0], mds->eigenvalues[1]);
    for (size_t i = 0; i < n && i < mds->N; i++)
    {
        printf("  Knoten %4zu: (%8.4f, %8.4f)\n",
               i, mds->coords[i * 2 + 0], mds->coords[i * 2 + 1]);
    }
}

// Speichert I als PGM-Bild (2D-Plot mit MDS-Koordinaten)
void iwt_mds_save_pgm(const iwt_mds_t mds, const double* I, size_t N, const char* filename)
{
    if (!mds->coords || !I || N != mds->N) return;

    // Bounding Box der Koordinaten
    double min_x = 0.0, max_x = 1.0;
    double min_y = 0.0, max_y = 1.0;
    for (size_t i = 0; i < N; i++)
    {
        double x = mds->coords[i * 2 + 0];
        double y = mds->coords[i * 2 + 1];
        if (i == 0 || x < min_x) min_x = x;
        if (i == 0 || x > max_x) max_x = x;
        if (i == 0 || y < min_y) min_y = y;
        if (i == 0 || y > max_y) max_y = y;
    }
    double range_x = max_x - min_x;
    double range_y = max_y - min_y;
    if (range_x < 1e-10) range_x = 1.0;
    if (range_y < 1e-10) range_y = 1.0;

    // Bildgröße
    int width = 512;
    int height = 512;
    unsigned char* image = malloc(width * height * sizeof(unsigned char));
    if (!image) return;

    // Pixel initialisieren (schwarz)
    memset(image, 0, width * height);

    // Knoten in das Bild zeichnen
    for (size_t i = 0; i < N; i++)
    {
        double x = (mds->coords[i * 2 + 0] - min_x) / range_x;
        double y = (mds->coords[i * 2 + 1] - min_y) / range_y;
        int px = (int)(x * (width - 1));
        int py = (int)((1.0 - y) * (height - 1));
        if (px < 0) px = 0;
        if (px >= width) px = width - 1;
        if (py < 0) py = 0;
        if (py >= height) py = height - 1;

        // Helligkeit aus I
        double brightness = I[i];
        if (brightness < 0.0) brightness = 0.0;
        if (brightness > 1.0) brightness = 1.0;
        image[py * width + px] = (unsigned char)(brightness * 255.0);
    }

    // PGM-Datei schreiben
    FILE* f = fopen(filename, "wb");
    if (f)
    {
        fprintf(f, "P5\n%d %d\n255\n", width, height);
        fwrite(image, 1, width * height, f);
        fclose(f);
    }

    free(image);
}

private void iwt_print_int(uint32_t* x, uint32_t y, const char* label, int value)
{
    string_set_cursor_position(*x, y);
    printf(BLACK_ON_WHITE "%s:" BLUE_ON_WHITE "%8d" COLOR_RESET, label, value);
    *x += strlen(label) + 10;
}

private void iwt_print_double(uint32_t* x, uint32_t y, const char* label, double value)
{
    string_set_cursor_position(*x, y);
    char buffer[256] = {0};
    sprintf(buffer, "%12.3f", value);
    printf(BLACK_ON_WHITE "%s:" BLUE_ON_WHITE "%s" COLOR_RESET, label, buffer);
    *x += strlen(label) + strlen(buffer) + 5;
}

// ============================================================
// ERWEITERTE iwt_print_status MIT FLUKTUATIONS-ANZEIGE
// ============================================================

void iwt_print_status(const iwt_runtime_t rt, const iwt_config_t cfg, int iter,
    double max_q, double I_total, double I_min, double I_max,
    double deviation, double sum_I_sq, double info_deviation)
{
    // Statische Arrays für Zeitreihen (letzte 10 Werte)
    static double sum_I_sq_history[10] = {0};
    static double I_max_history[10] = {0};
    static int history_idx = 0;
    static int history_count = 0;

    // Aktuellen Wert speichern
    sum_I_sq_history[history_idx] = sum_I_sq;
    I_max_history[history_idx] = I_max;
    history_idx = (history_idx + 1) % 10;
    if (history_count < 10) history_count++;

    uint32_t col = 1, row = 1;

    // ZEILE 1
    iwt_print_int(&col, row, "#", iter);
    iwt_print_double(&col, row, "|Q_max|", max_q);
    iwt_print_double(&col, row, "I_total", I_total);
    iwt_print_double(&col, row, "I_min", I_min);
    iwt_print_double(&col, row, "I_max", I_max);
    iwt_print_double(&col, row, "Deviation", deviation);

    // ZEILE 2: Informationserhaltung
    row += 2; col = 1;
    iwt_print_double(&col, row, "ΣI²", sum_I_sq);
    iwt_print_double(&col, row, "Info_Dev", info_deviation);

    // ZEILE 3: Fluktuations-Statistiken
    row += 2; col = 1;
    double xi_real_mean = 0.0, xi_imag_mean = 0.0;
    double xi_real_var = 0.0, xi_imag_var = 0.0;

    for (size_t i = 0; i < cfg->N; i++)
    {
        xi_real_mean += rt->xi_real[i];
        xi_imag_mean += rt->xi_imag[i];
    }
    xi_real_mean /= cfg->N;
    xi_imag_mean /= cfg->N;

    for (size_t i = 0; i < cfg->N; i++)
    {
        double dr = rt->xi_real[i] - xi_real_mean;
        double di = rt->xi_imag[i] - xi_imag_mean;
        xi_real_var += dr * dr;
        xi_imag_var += di * di;
    }
    xi_real_var = sqrt(xi_real_var / cfg->N);
    xi_imag_var = sqrt(xi_imag_var / cfg->N);

    iwt_print_double(&col, row, "ξ_real_mean", xi_real_mean);
    iwt_print_double(&col, row, "ξ_imag_mean", xi_imag_mean);
    iwt_print_double(&col, row, "ξ_real_σ", xi_real_var);
    iwt_print_double(&col, row, "ξ_imag_σ", xi_imag_var);
    iwt_print_double(&col, row, "scale", cfg->uncertainty_scale);

    // ZEILE 4: Zeitreihe ΣI² (letzte 10 Werte)
    row += 2; col = 1;
    string_set_cursor_position(col, row);
    printf(BLACK_ON_WHITE "ΣI² (letzte 10):" COLOR_RESET);

    int start = history_count < 10 ? 0 : history_idx;
    for (int i = 0; i < history_count; i++)
    {
        int idx = (start + i) % 10;
        string_set_cursor_position(col + 18 + i * 12, row);
        printf(BLUE_ON_WHITE "%10.1f" COLOR_RESET, sum_I_sq_history[idx]);
    }

    // ZEILE 5: I_max
    row += 2; col = 1;
    string_set_cursor_position(col, row);
    printf(BLACK_ON_WHITE "I_max (letzte 10):" COLOR_RESET);

    for (int i = 0; i < history_count; i++)
    {
        int idx = (start + i) % 10;
        string_set_cursor_position(col + 18 + i * 12, row);
        printf(BLUE_ON_WHITE "%10.1f" COLOR_RESET, I_max_history[idx]);
    }

    // ZEILE 7: I-Werte
    row += 2; col = 1;
    iwt_print_double(&col, row, "I[0]", rt->I[0]);
    iwt_print_double(&col, row, "I[101]", rt->I[101]);
    iwt_print_double(&col, row, "I[102]", rt->I[102]);
    iwt_print_double(&col, row, "I[103]", rt->I[103]);
    iwt_print_double(&col, row, "I[104]", rt->I[104]);

    // ZEILE 9: I_phase
    row += 2; col = 1;
    iwt_print_double(&col, row, "I_phase[0]", rt->I_phase[0] / iwt_pi() * 180.0);
    iwt_print_double(&col, row, "I_phase[101]", rt->I_phase[101] / iwt_pi() * 180.0);
    iwt_print_double(&col, row, "I_phase[102]", rt->I_phase[102] / iwt_pi() * 180.0);
    iwt_print_double(&col, row, "I_phase[103]", rt->I_phase[103] / iwt_pi() * 180.0);
    iwt_print_double(&col, row, "I_phase[104]", rt->I_phase[104] / iwt_pi() * 180.0);

    // ZEILE 11: sumJ
    row += 2; col = 1;
    iwt_print_double(&col, row, "sum_J[0]", rt->sumJ[0]);
    iwt_print_double(&col, row, "sum_J[101]", rt->sumJ[101]);
    iwt_print_double(&col, row, "sum_J[102]", rt->sumJ[102]);
    iwt_print_double(&col, row, "sum_J[103]", rt->sumJ[103]);
    iwt_print_double(&col, row, "sum_J[104]", rt->sumJ[104]);

    // ZEILE 13: T
    row += 2; col = 1;
    iwt_print_double(&col, row, "T[0]", rt->T[0]);
    iwt_print_double(&col, row, "T[101]", rt->T[101]);
    iwt_print_double(&col, row, "T[102]", rt->T[102]);
    iwt_print_double(&col, row, "T[103]", rt->T[103]);
    iwt_print_double(&col, row, "T[104]", rt->T[104]);

    // ZEILE 15: R
    row += 2; col = 1;
    iwt_print_double(&col, row, "R[0]", rt->R[0]);
    iwt_print_double(&col, row, "R[101]", rt->R[101]);
    iwt_print_double(&col, row, "R[102]", rt->R[102]);
    iwt_print_double(&col, row, "R[103]", rt->R[103]);
    iwt_print_double(&col, row, "R[104]", rt->R[104]);

    // ZEILE 17: Teilchen
    row += 2; col = 1;
    struct iwt_spectrum spec = {0};
    iwt_compute_spectrum(rt->I, cfg->N, &spec);
    iwt_print_int(&col, row, "Elektronen", spec.count_electron);
    iwt_print_int(&col, row, "Protonen", spec.count_proton);

    // ZEILE 19: Fluktuationswerte ξ_r
    row += 2; col = 1;
    iwt_print_double(&col, row, "ξ_r[0]", rt->xi_real[0]);
    iwt_print_double(&col, row, "ξ_r[101]", rt->xi_real[101]);
    iwt_print_double(&col, row, "ξ_r[102]", rt->xi_real[102]);
    iwt_print_double(&col, row, "ξ_r[103]", rt->xi_real[103]);
    iwt_print_double(&col, row, "ξ_r[104]", rt->xi_real[104]);

    // ZEILE 21: Fluktuationswerte ξ_i
    row += 2; col = 1;
    iwt_print_double(&col, row, "ξ_i[0]", rt->xi_imag[0]);
    iwt_print_double(&col, row, "ξ_i[101]", rt->xi_imag[101]);
    iwt_print_double(&col, row, "ξ_i[102]", rt->xi_imag[102]);
    iwt_print_double(&col, row, "ξ_i[103]", rt->xi_imag[103]);
    iwt_print_double(&col, row, "ξ_i[104]", rt->xi_imag[104]);

    fflush(stdout);
}
