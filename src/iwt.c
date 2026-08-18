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

private void _flood_fill(const iwt_runtime_t rt, const iwt_config_t cfg, 
                         size_t idx, bool* visited, iwt_cluster_t c)
{
    // Dynamischer Stack (wie von Dir bereits angepasst)
    size_t* stack = malloc(cfg->N * sizeof(size_t));
    if (!stack) return;
    int stack_ptr = 0;
    stack[stack_ptr++] = idx;

    while (stack_ptr > 0)
    {
        size_t i = stack[--stack_ptr];
        if (visited[i]) continue;
        if (rt->mass[i] < 1e-6) continue;
        
        visited[i] = true;
        
        // Knoten zum Cluster hinzufügen
        c->node_indices[c->node_count] = i;
        c->node_count++;
        
        // Masse und Schwerpunkt akkumulieren (3D-Dodekaeder-Positionen)
        double m = rt->mass[i];
        c->mass += m;
        c->x += m * rt->pos_x[i];
        c->y += m * rt->pos_y[i];
        c->z += m * rt->pos_z[i];
        c->charge += rt->charge[i];
        c->phase += rt->I_phase[i];
        
        // ================================================================
        // Nachbarn ueber die vorberechnete K-Matrix-Adjazenz sammeln,
        // in zufaelliger Reihenfolge (Fisher-Yates)
        // ================================================================
        
        size_t* neighbors = malloc(cfg->N * sizeof(size_t));
        int neighbor_count = 0;
        
        if (neighbors != NULL)
        {
            const bool* row = &rt->adjacency[i * cfg->N];
            for (size_t j = 0; j < cfg->N; j++)
            {
                if (row[j] && !visited[j] && rt->mass[j] > 1e-6)
                {
                    neighbors[neighbor_count++] = j;
                }
            }

            // Nachbarn zufaellig mischen (Fisher-Yates)
            for (int n = neighbor_count - 1; n > 0; n--)
            {
                int r = rand() % (n + 1);
                size_t temp = neighbors[n];
                neighbors[n] = neighbors[r];
                neighbors[r] = temp;
            }

            // Nachbarn in zufaelliger Reihenfolge auf den Stack legen
            for (int n = 0; n < neighbor_count; n++)
            {
                stack[stack_ptr++] = neighbors[n];
            }

            free(neighbors);
        }
    }
    
    free(stack);
}

void iwt_detect_clusters(const iwt_runtime_t rt, const iwt_config_t cfg)
{
    static bool* visited = NULL;
    static size_t visited_size = 0;
    
    if (visited == NULL || visited_size != cfg->N)
    {
        if (visited != NULL) free(visited);
        visited = calloc(cfg->N, sizeof(bool));
        visited_size = cfg->N;
    }
    
    // Alle Knoten als unbesucht markieren
    memset(visited, 0, cfg->N * sizeof(bool));
    
    // Alte Cluster zurücksetzen
    rt->cluster_count = 0;
    
    // Flood-Fill für jeden Knoten
    for (size_t i = 0; i < cfg->N; i++)
    {
        if (visited[i]) continue;
        if (rt->mass[i] < 1e-6) continue;  // Schwellwert für Masse
        
        // Neuen Cluster initialisieren
        if (rt->cluster_count >= rt->cluster_capacity) break;
        iwt_cluster_t c = &rt->clusters[rt->cluster_count];
        c->id = rt->cluster_count;
        c->node_count = 0;
        c->mass = 0.0;
        c->charge = 0.0;
        c->phase = 0.0;
        c->x = 0.0;
        c->y = 0.0;
        c->z = 0.0;
        c->vx = 0.0;
        c->vy = 0.0;
        c->vz = 0.0;
        c->is_active = true;
        
        // Flood-Fill: Alle verbundenen Knoten sammeln
        _flood_fill(rt, cfg, i, visited, c);
        
        // Schwerpunkt berechnen (gewichtet mit Masse)
        if (c->node_count > 0)
        {
            c->x /= c->mass;
            c->y /= c->mass;
            c->z /= c->mass;
            c->phase /= c->node_count;
            rt->cluster_count++;
        }
    }
}

private void _iwt_compute_weber_force(const iwt_cluster_t a, const iwt_cluster_t b, 
                             double G, double c, double epsilon0, 
                             double* Fx, double* Fy, double* Fz)
{
    double dx = b->x - a->x;
    double dy = b->y - a->y;
    double dz = b->z - a->z;
    double r = sqrt(dx * dx + dy * dy + dz * dz);
    if (r < 1e-30) { *Fx = 0.0; *Fy = 0.0; *Fz = 0.0; return; }
    
    // ================================================================
    // Relativgeschwindigkeit und -beschleunigung (vereinfacht)
    // ================================================================
    double dvx = b->vx - a->vx;
    double dvy = b->vy - a->vy;
    double dvz = b->vz - a->vz;
    double dr = (dx * dvx + dy * dvy + dz * dvz) / r;      // radiale Geschwindigkeit
    
    // Beschleunigung (Differenz der Geschwindigkeiten, vereinfacht)
    double dax = 0.0;  // müsste aus den Kräften berechnet werden
    double day = 0.0;
    double daz = 0.0;
    double d2r = (dx * dax + dy * day + dz * daz) / r;     // radiale Beschleunigung
    
    // ================================================================
    // 1. WEBER-GRAVITATION (WG) - wirkt zwischen Massen
    //    β = 0.5 für massive Körper
    // ================================================================
    double F_WG_mag = G * a->mass * b->mass / (r * r);
    double beta_WG = 0.5;
    double factor_WG = 1.0 - (dr * dr) / (c * c) + beta_WG * (r * d2r) / (c * c);
    F_WG_mag *= factor_WG;
    
    double F_WG_x = F_WG_mag * (-dx / r);  // anziehend (Minus-Zeichen)
    double F_WG_y = F_WG_mag * (-dy / r);
    double F_WG_z = F_WG_mag * (-dz / r);
    
    // ================================================================
    // 2. WEBER-ELEKTRODYNAMIK (WED) - wirkt zwischen Ladungen
    //    β = 2 für elektrische Ladungen
    // ================================================================
    double F_WED_mag = (a->charge * b->charge) / (4.0 * 3.141592653589793 * epsilon0 * r * r);
    double beta_WED = 2.0;
    double factor_WED = 1.0 - (dr * dr) / (c * c) + beta_WED * (r * d2r) / (c * c);
    F_WED_mag *= factor_WED;
    
    double F_WED_x = F_WED_mag * (dx / r);   // abstoßend/anziehend je nach Vorzeichen
    double F_WED_y = F_WED_mag * (dy / r);
    double F_WED_z = F_WED_mag * (dz / r);
    
    // ================================================================
    // 3. GESAMTKRAFT (WG + WED)
    // ================================================================
    *Fx = F_WG_x + F_WED_x;
    *Fy = F_WG_y + F_WED_y;
    *Fz = F_WG_z + F_WED_z;
}

void iwt_move_clusters(const iwt_runtime_t rt, const iwt_config_t cfg, double dt)
{
    double PI = 4.0 * atan(1.0);
    double twoPI = 2.0 * PI;

    // ================================================================
    // 1. GESCHWINDIGKEITEN AUS WEBER-KRÄFTEN BERECHNEN (3D)
    // ================================================================
    for (size_t c = 0; c < rt->cluster_count; c++)
    {
        iwt_cluster_t cl = &rt->clusters[c];
        if (!cl->is_active) continue;

        double Fx = 0.0;
        double Fy = 0.0;
        double Fz = 0.0;

        for (size_t d = 0; d < rt->cluster_count; d++)
        {
            if (c == d) continue;
            iwt_cluster_t other = &rt->clusters[d];
            if (!other->is_active) continue;

            double Fx_ij, Fy_ij, Fz_ij;
            _iwt_compute_weber_force(cl, other, 1.0, 1.0, 1.0, &Fx_ij, &Fy_ij, &Fz_ij);
            Fx += Fx_ij;
            Fy += Fy_ij;
            Fz += Fz_ij;
        }

        if (cl->mass > 1e-30)
        {
            cl->vx += (Fx / cl->mass) * dt;
            cl->vy += (Fy / cl->mass) * dt;
            cl->vz += (Fz / cl->mass) * dt;
        }
    }

    // ================================================================
    // 2. PHASENVERSCHIEBUNG (NUR WENN enable_motion == true)
    //    Radial = entlang der Knoten-Zentrum-Achse (wie bisher, jetzt 3D)
    //    Tangential = Anteil von v senkrecht zu v_rad (Ebene senkrecht
    //    zum Cluster-Impuls), unsigniert (Pythagoras)
    // ================================================================
    if (cfg->enable_motion)
    {
        for (size_t c = 0; c < rt->cluster_count; c++)
        {
            iwt_cluster_t cl = &rt->clusters[c];
            if (!cl->is_active) continue;
            if (cl->node_count == 0) continue;

            double vx = cl->vx;
            double vy = cl->vy;
            double vz = cl->vz;
            double v_speed_sq = vx * vx + vy * vy + vz * vz;

            for (size_t n = 0; n < cl->node_count; n++)
            {
                size_t i = cl->node_indices[n];

                double dx = rt->pos_x[i] - cl->x;
                double dy = rt->pos_y[i] - cl->y;
                double dz = rt->pos_z[i] - cl->z;
                double r = sqrt(dx * dx + dy * dy + dz * dz);
                if (r < 1e-30) continue;

                double v_rad = (vx * dx + vy * dy + vz * dz) / r;
                double v_tan_sq = v_speed_sq - v_rad * v_rad;
                double v_tan = v_tan_sq > 0.0 ? sqrt(v_tan_sq) : 0.0;

                double lambda_rad = 1.0;
                double dphi_rad = (twoPI / lambda_rad) * v_rad * dt;

                double lambda_tan = 1.0;
                double dphi_tan = (twoPI / lambda_tan) * (v_tan / r) * dt;

                double dphi = dphi_rad + dphi_tan;

                rt->I_phase[i] += dphi;
                while (rt->I_phase[i] > PI) rt->I_phase[i] -= twoPI;
                while (rt->I_phase[i] < -PI) rt->I_phase[i] += twoPI;
            }
        }

        // 3. I-Werte neu berechnen (nur wenn Bewegung aktiv)
        for (size_t i = 0; i < cfg->N; i++)
        {
            double rho = rt->mass[i];
            double phi = rt->I_phase[i];
            rt->I_real[i] = sqrt(fabs(rho)) * cos(phi);
            rt->I_imag[i] = sqrt(fabs(rho)) * sin(phi);
        }
    }

    // ================================================================
    // 4. SCHWERPUNKT NEU BERECHNEN (immer, 3D)
    // ================================================================
    for (size_t c = 0; c < rt->cluster_count; c++)
    {
        iwt_cluster_t cl = &rt->clusters[c];
        if (!cl->is_active) continue;
        if (cl->node_count == 0) continue;

        double new_x = 0.0;
        double new_y = 0.0;
        double new_z = 0.0;
        double new_mass = 0.0;

        for (size_t n = 0; n < cl->node_count; n++)
        {
            size_t i = cl->node_indices[n];
            double rho = rt->mass[i];

            new_x += rho * rt->pos_x[i];
            new_y += rho * rt->pos_y[i];
            new_z += rho * rt->pos_z[i];
            new_mass += rho;
        }

        if (new_mass > 1e-30)
        {
            cl->x = new_x / new_mass;
            cl->y = new_y / new_mass;
            cl->z = new_z / new_mass;
        }
    }

    // ================================================================
    // 5. KNOTENLISTE ZURÜCKSETZEN (immer)
    // ================================================================
    for (size_t c = 0; c < rt->cluster_count; c++)
    {
        iwt_cluster_t cl = &rt->clusters[c];
        if (!cl->is_active) continue;
        cl->node_count = 0;
    }
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

void iwt_compute_node_colors(const double *mass, const double *charge, size_t N, float *out_rgb)
{
    double mass_min = 1e30;
    double mass_max = -1e30;
    for (size_t i = 0; i < N; i++)
    {
        mass_min = MIN(mass_min, mass[i]);
        mass_max = MAX(mass_max, mass[i]);
    }
    double mass_range = mass_max - mass_min;
    if (mass_range < 1e-30) mass_range = 1.0;

    double max_abs = 0.0;
    for (size_t i = 0; i < N; i++)
    {
        max_abs = MAX(max_abs, fabs(charge[i]));
    }
    if (max_abs < 1e-30) max_abs = 1.0;

    // Helligkeit = Masse, Farbrichtung (Rot/Blau) = Ladung - gleiches Schema
    // wie iwt_build_overlay_rgb(), aber pro Knoten statt pro Pixel.
    for (size_t i = 0; i < N; i++)
    {
        double brightness = (mass[i] - mass_min) / mass_range;
        double charge_norm = charge[i] / max_abs;
        double abs_charge = fabs(charge_norm);

        double r = brightness * (charge_norm < 0.0 ? -charge_norm : (1.0 - abs_charge));
        double g = brightness * (1.0 - abs_charge);
        double b = brightness * (charge_norm > 0.0 ? charge_norm : (1.0 - abs_charge));

        out_rgb[i * 3 + 0] = (float)r;
        out_rgb[i * 3 + 1] = (float)g;
        out_rgb[i * 3 + 2] = (float)b;
    }
}

