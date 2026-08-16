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

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

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

double iwt_alpha_IWT(void) { return 1; }

double iwt_beta_IWT(void) { return 1; }

private void _iwt_print_int(uint32_t *x, uint32_t y, const char *label, int value)
{
    string_set_cursor_position(*x, y);
    printf(BLACK_ON_WHITE "%s:" BLUE_ON_WHITE "%8d" COLOR_RESET, label, value);
    *x += strlen(label) + 10;
}

private void _iwt_print_double(uint32_t *x, uint32_t y, const char *label, double value)
{
    string_set_cursor_position(*x, y);
    char buffer[256] = {0};
    sprintf(buffer, "%12.3f", value);
    printf(BLACK_ON_WHITE "%s:" BLUE_ON_WHITE "%s" COLOR_RESET, label, buffer);
    *x += strlen(label) + strlen(buffer) + 5;
}

private void _iwt_print_double_triple(uint32_t *x, uint32_t y, const char *labels[3], const double values[3])
{
    for (int i = 0; i < 3; i++)
    {
        _iwt_print_double(x, y, labels[i], values[i]);
    }
}

private void _iwt_print_double_quad(uint32_t *x, uint32_t y, const char *labels[4], const double values[4])
{
    for (int i = 0; i < 4; i++)
    {
        _iwt_print_double(x, y, labels[i], values[i]);
    }
}

void iwt_print_status(const iwt_runtime_t rt, const iwt_config_t cfg, int iter,
    double max_q, double I_total, double I_min, double I_max, double sum_I_sq)
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
    _iwt_print_int(&col, row, "#", iter);
    {
        const char *labels[4] = { "|Q_max|", "I_total", "I_min", "I_max" };
        const double values[4] = { max_q, I_total, I_min, I_max };
        _iwt_print_double_quad(&col, row, labels, values);
    }

    // ZEILE 2
    row += 2;
    col = 1;
    _iwt_print_double(&col, row, "ΣI²", sum_I_sq);

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

    {
        const char *labels[4] = { "ξ_real_mean", "ξ_imag_mean", "ξ_real_σ", "ξ_imag_σ" };
        const double values[4] = { xi_real_mean, xi_imag_mean, xi_real_var, xi_imag_var };
        _iwt_print_double_quad(&col, row, labels, values);
    }

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
        Re_min = MIN(Re_min, v);
        Re_max = MAX(Re_max, v);
    }
    Re_mean /= cfg->N;
    {
        const char *labels[3] = { "Re_mean", "Re_min", "Re_max" };
        const double values[3] = { Re_mean, Re_min, Re_max };
        _iwt_print_double_triple(&col, row, labels, values);
    }

    // ZEILE 9: Im(I) Statistik
    row += 2;
    col = 1;
    double Im_mean = 0.0, Im_min = 1e30, Im_max = -1e30;
    for (size_t i = 0; i < cfg->N; i++)
    {
        double v = rt->I_imag[i];
        Im_mean += v;
        Im_min = MIN(Im_min, v);
        Im_max = MAX(Im_max, v);
    }
    Im_mean /= cfg->N;
    {
        const char *labels[3] = { "Im_mean", "Im_min", "Im_max" };
        const double values[3] = { Im_mean, Im_min, Im_max };
        _iwt_print_double_triple(&col, row, labels, values);
    }

    // ZEILE 11: Phase Statistik
    row += 2;
    col = 1;
    double phase_mean = 0.0, phase_min = 1e30, phase_max = -1e30;
    for (size_t i = 0; i < cfg->N; i++)
    {
        double v = rt->I_phase[i] / iwt_pi() * 180.0;
        phase_mean += v;
        phase_min = MIN(phase_min, v);
        phase_max = MAX(phase_max, v);
    }
    phase_mean /= cfg->N;
    {
        const char *labels[3] = { "φ_mean", "φ_min", "φ_max" };
        const double values[3] = { phase_mean, phase_min, phase_max };
        _iwt_print_double_triple(&col, row, labels, values);
    }

    // ZEILE 13: sumJ Statistik
    row += 2;
    col = 1;
    double J_mean = 0.0, J_min = 1e30, J_max = -1e30;
    for (size_t i = 0; i < cfg->N; i++)
    {
        double v = rt->sumJ[i];
        J_mean += v;
        J_min = MIN(J_min, v);
        J_max = MAX(J_max, v);
    }
    J_mean /= cfg->N;
    {
        const char *labels[3] = { "J_mean", "J_min", "J_max" };
        const double values[3] = { J_mean, J_min, J_max };
        _iwt_print_double_triple(&col, row, labels, values);
    }

    // ZEILE 15: Q Statistik
    row += 2;
    col = 1;
    double Q_mean = 0.0;
    for (size_t i = 0; i < cfg->N; i++)
    {
        Q_mean += rt->Q[i];
    }
    Q_mean /= cfg->N;
    _iwt_print_double(&col, row, "Q_mean", Q_mean);

    // ZEILE 17: xi_real Statistik
    row += 2;
    col = 1;
    double xir_mean = 0.0, xir_min = 1e30, xir_max = -1e30;
    for (size_t i = 0; i < cfg->N; i++)
    {
        double v = rt->xi_real[i];
        xir_mean += v;
        xir_min = MIN(xir_min, v);
        xir_max = MAX(xir_max, v);
    }
    xir_mean /= cfg->N;
    {
        const char *labels[3] = { "ξr_mean", "ξr_min", "ξr_max" };
        const double values[3] = { xir_mean, xir_min, xir_max };
        _iwt_print_double_triple(&col, row, labels, values);
    }

    // ZEILE 19: xi_imag Statistik
    row += 2;
    col = 1;
    double xii_mean = 0.0, xii_min = 1e30, xii_max = -1e30;
    for (size_t i = 0; i < cfg->N; i++)
    {
        double v = rt->xi_imag[i];
        xii_mean += v;
        xii_min = MIN(xii_min, v);
        xii_max = MAX(xii_max, v);
    }
    xii_mean /= cfg->N;
    {
        const char *labels[3] = { "ξi_mean", "ξi_min", "ξi_max" };
        const double values[3] = { xii_mean, xii_min, xii_max };
        _iwt_print_double_triple(&col, row, labels, values);
    }

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
        min_val = MIN(min_val, data[i]);
        max_val = MAX(max_val, data[i]);
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
        max_abs = MAX(max_abs, fabs(charge[i]));
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

