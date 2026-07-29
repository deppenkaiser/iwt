// ============================================================================
// iwt.c - AUSGABE MIT STATISTIK STATT EINZELNER KNOTEN
// ============================================================================

#include "iwt.h"
#include <api/api.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string/string.h>

double iwt_pi(void) { return 4.0 * atan(1.0); }

double iwt_fundamental_length(void)
{
    const double h = 6.62607015e-34;
    const double mp = 1.67262192e-27;
    const double c = 299792458.0;
    const double phi = (1.0 + sqrt(5.0)) / 2.0;
    const double V_dode = (15.0 + 7.0 * sqrt(5.0)) / 4.0;
    const double V_sphere = 4.0 * iwt_pi() / 3.0;
    const double V_ratio = V_dode / V_sphere;
    const double lambda_p = h / (mp * c);
    return pow(V_ratio, 1.0 / 3.0) * lambda_p;
}

double iwt_fundamental_time(void)
{
    const double c = 299792458.0;
    return iwt_fundamental_length() / c;
}

double iwt_fractal_dimension(void)
{
    const double phi = (1.0 + sqrt(5.0)) / 2.0;
    const double N = 20.0 * phi;
    const double s = 2.0 + phi;
    return log(N) / log(s);
}

double iwt_I_min(void) { return 0.0; }

double iwt_I_max(void) { return 1.0; }

double iwt_delta_I(void)
{
    double I_min = iwt_I_min();
    double I_max = iwt_I_max();
    double steps = pow(2.0, 32.0) - 1.0;
    return (I_max - I_min) / steps;
}

double iwt_alpha_IWT(void) { return 1; }

double iwt_beta_IWT(void) { return 1; }

static void iwt_print_int(uint32_t *x, uint32_t y, const char *label, int value)
{
    string_set_cursor_position(*x, y);
    printf(BLACK_ON_WHITE "%s:" BLUE_ON_WHITE "%8d" COLOR_RESET, label, value);
    *x += strlen(label) + 10;
}

static void iwt_print_double(uint32_t *x, uint32_t y, const char *label, double value)
{
    string_set_cursor_position(*x, y);
    char buffer[256] = {0};
    sprintf(buffer, "%12.3f", value);
    printf(BLACK_ON_WHITE "%s:" BLUE_ON_WHITE "%s" COLOR_RESET, label, buffer);
    *x += strlen(label) + strlen(buffer) + 5;
}

void iwt_print_status(const iwt_runtime_t rt, const iwt_config_t cfg, int iter,
                      double max_q, double I_total, double I_min, double I_max,
                      double deviation, double sum_I_sq, double info_deviation)
{
    static double sum_I_sq_history[10] = {0};
    static double I_max_history[10] = {0};
    static int history_idx = 0;
    static int history_count = 0;

    sum_I_sq_history[history_idx] = sum_I_sq;
    I_max_history[history_idx] = I_max;
    history_idx = (history_idx + 1) % 10;
    if (history_count < 10)
        history_count++;

    uint32_t col = 1, row = 1;

    // ZEILE 1
    iwt_print_int(&col, row, "#", iter);
    iwt_print_double(&col, row, "|Q_max|", max_q);
    iwt_print_double(&col, row, "I_total", I_total);
    iwt_print_double(&col, row, "I_min", I_min);
    iwt_print_double(&col, row, "I_max", I_max);
    iwt_print_double(&col, row, "Deviation", deviation);

    // ZEILE 2
    row += 2;
    col = 1;
    iwt_print_double(&col, row, "ΣI²", sum_I_sq);
    iwt_print_double(&col, row, "Info_Dev", info_deviation);

    // ZEILE 3: Fluktuationen
    row += 2;
    col = 1;
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

    // ZEILE 4: ΣI² History
    row += 2;
    col = 1;
    string_set_cursor_position(col, row);
    printf(BLACK_ON_WHITE "ΣI² (letzte 10):" COLOR_RESET);
    int start = history_count < 10 ? 0 : history_idx;
    for (int i = 0; i < history_count; i++)
    {
        int idx = (start + i) % 10;
        string_set_cursor_position(col + 18 + i * 12, row);
        printf(BLUE_ON_WHITE "%10.1f" COLOR_RESET, sum_I_sq_history[idx]);
    }

    // ZEILE 5: I_max History
    row += 2;
    col = 1;
    string_set_cursor_position(col, row);
    printf(BLACK_ON_WHITE "I_max (letzte 10):" COLOR_RESET);
    for (int i = 0; i < history_count; i++)
    {
        int idx = (start + i) % 10;
        string_set_cursor_position(col + 18 + i * 12, row);
        printf(BLUE_ON_WHITE "%10.1f" COLOR_RESET, I_max_history[idx]);
    }

    // ZEILE 7: Re(I) Statistik
    row += 2;
    col = 1;
    double Re_mean = 0.0, Re_min = 1e30, Re_max = -1e30;
    for (size_t i = 0; i < cfg->N; i++)
    {
        double v = rt->I_real[i];
        Re_mean += v;
        if (v < Re_min)
            Re_min = v;
        if (v > Re_max)
            Re_max = v;
    }
    Re_mean /= cfg->N;
    iwt_print_double(&col, row, "Re_mean", Re_mean);
    iwt_print_double(&col, row, "Re_min", Re_min);
    iwt_print_double(&col, row, "Re_max", Re_max);

    // ZEILE 9: Im(I) Statistik
    row += 2;
    col = 1;
    double Im_mean = 0.0, Im_min = 1e30, Im_max = -1e30;
    for (size_t i = 0; i < cfg->N; i++)
    {
        double v = rt->I_imag[i];
        Im_mean += v;
        if (v < Im_min)
            Im_min = v;
        if (v > Im_max)
            Im_max = v;
    }
    Im_mean /= cfg->N;
    iwt_print_double(&col, row, "Im_mean", Im_mean);
    iwt_print_double(&col, row, "Im_min", Im_min);
    iwt_print_double(&col, row, "Im_max", Im_max);

    // ZEILE 11: Phase Statistik
    row += 2;
    col = 1;
    double phase_mean = 0.0, phase_min = 1e30, phase_max = -1e30;
    for (size_t i = 0; i < cfg->N; i++)
    {
        double v = rt->I_phase[i] / iwt_pi() * 180.0;
        phase_mean += v;
        if (v < phase_min)
            phase_min = v;
        if (v > phase_max)
            phase_max = v;
    }
    phase_mean /= cfg->N;
    iwt_print_double(&col, row, "φ_mean", phase_mean);
    iwt_print_double(&col, row, "φ_min", phase_min);
    iwt_print_double(&col, row, "φ_max", phase_max);

    // ZEILE 13: sumJ Statistik
    row += 2;
    col = 1;
    double J_mean = 0.0, J_min = 1e30, J_max = -1e30;
    for (size_t i = 0; i < cfg->N; i++)
    {
        double v = rt->sumJ[i];
        J_mean += v;
        if (v < J_min)
            J_min = v;
        if (v > J_max)
            J_max = v;
    }
    J_mean /= cfg->N;
    iwt_print_double(&col, row, "J_mean", J_mean);
    iwt_print_double(&col, row, "J_min", J_min);
    iwt_print_double(&col, row, "J_max", J_max);

    // ZEILE 15: Q Statistik
    row += 2;
    col = 1;
    double Q_mean = 0.0;
    for (size_t i = 0; i < cfg->N; i++)
    {
        Q_mean += rt->Q[i];
    }
    Q_mean /= cfg->N;
    iwt_print_double(&col, row, "Q_mean", Q_mean);

    // ZEILE 17: xi_real Statistik
    row += 2;
    col = 1;
    double xir_mean = 0.0, xir_min = 1e30, xir_max = -1e30;
    for (size_t i = 0; i < cfg->N; i++)
    {
        double v = rt->xi_real[i];
        xir_mean += v;
        if (v < xir_min)
            xir_min = v;
        if (v > xir_max)
            xir_max = v;
    }
    xir_mean /= cfg->N;
    iwt_print_double(&col, row, "ξr_mean", xir_mean);
    iwt_print_double(&col, row, "ξr_min", xir_min);
    iwt_print_double(&col, row, "ξr_max", xir_max);

    // ZEILE 19: xi_imag Statistik
    row += 2;
    col = 1;
    double xii_mean = 0.0, xii_min = 1e30, xii_max = -1e30;
    for (size_t i = 0; i < cfg->N; i++)
    {
        double v = rt->xi_imag[i];
        xii_mean += v;
        if (v < xii_min)
            xii_min = v;
        if (v > xii_max)
            xii_max = v;
    }
    xii_mean /= cfg->N;
    iwt_print_double(&col, row, "ξi_mean", xii_mean);
    iwt_print_double(&col, row, "ξi_min", xii_min);
    iwt_print_double(&col, row, "ξi_max", xii_max);

    fflush(stdout);
}

void iwt_save_heatmap_ppm(const double *data, size_t N, const char *filename, const char *type)
{
    int width = 512;
    int height = 512;

    // Speicher mit 0 initialisieren (calloc statt malloc)
    unsigned char *image = calloc(width * height * 3, sizeof(unsigned char));
    if (!image) return;

    double min_val = 1e30;
    double max_val = -1e30;
    for (size_t i = 0; i < N; i++)
    {
        if (data[i] < min_val) min_val = data[i];
        if (data[i] > max_val) max_val = data[i];
    }
    double range = max_val - min_val;
    if (range < 1e-30) range = 1.0;

    int grid_size = (int)sqrt(N);
    for (size_t i = 0; i < N; i++)
    {
        int x = i % grid_size;
        int y = i / grid_size;
        if (x >= width || y >= height) continue;

        double normalized = (data[i] - min_val) / range;
        unsigned char value = (unsigned char)(normalized * 255.0);

        image[(y * width + x) * 3 + 0] = value;
        image[(y * width + x) * 3 + 1] = value;
        image[(y * width + x) * 3 + 2] = value;
    }

    FILE *f = fopen(filename, "wb");
    if (f)
    {
        fprintf(f, "P6\n%d %d\n255\n", width, height);
        fwrite(image, 1, width * height * 3, f);
        fclose(f);
    }
    free(image);
}

void iwt_save_charge_heatmap(const double *charge, size_t N, const char *filename)
{
    int width = 512;
    int height = 512;

    // Speicher mit 0 initialisieren (calloc statt malloc)
    unsigned char *image = calloc(width * height * 3, sizeof(unsigned char));
    if (!image) return;

    double max_abs = 0.0;
    for (size_t i = 0; i < N; i++)
    {
        if (fabs(charge[i]) > max_abs) max_abs = fabs(charge[i]);
    }
    if (max_abs < 1e-30) max_abs = 1.0;

    int grid_size = (int)sqrt(N);
    for (size_t i = 0; i < N; i++)
    {
        int x = i % grid_size;
        int y = i / grid_size;
        if (x >= width || y >= height) continue;

        double normalized = charge[i] / max_abs;
        unsigned char r, g, b;

        if (normalized > 0.01)
        {
            // Positiv: Blau
            r = 0;
            g = 0;
            b = (unsigned char)(normalized * 255.0);
        }
        else if (normalized < -0.01)
        {
            // Negativ: Rot
            r = (unsigned char)(-normalized * 255.0);
            g = 0;
            b = 0;
        }
        else
        {
            // Null: Weiß
            r = 255;
            g = 255;
            b = 255;
        }

        image[(y * width + x) * 3 + 0] = r;
        image[(y * width + x) * 3 + 1] = g;
        image[(y * width + x) * 3 + 2] = b;
    }

    FILE *f = fopen(filename, "wb");
    if (f)
    {
        fprintf(f, "P6\n%d %d\n255\n", width, height);
        fwrite(image, 1, width * height * 3, f);
        fclose(f);
    }
    free(image);
}

