// ============================================================================
// iwt.c - AUSGABE MIT STATISTIK STATT EINZELNER KNOTEN
// ============================================================================

#include "iwt.h"
#include <api/api.h>
#include <math.h>
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

private void _flood_fill(const iwt_runtime_t rt, const iwt_config_t cfg, 
                         size_t idx, iwt_cluster_t c)
{
    size_t stack[cfg->N];  // VLA - geht weil cfg->N bekannt ist
    size_t stack_ptr = 0;
    stack[stack_ptr++] = idx;

    while (stack_ptr > 0)
    {
        size_t i = stack[--stack_ptr];
        if (rt->visited[i]) continue;
        if (rt->mass[i] < 1e-6) continue;
        
        rt->visited[i] = true;
        c->node_indices[c->node_count++] = i;
        
        c->mass += rt->mass[i];
        c->x += rt->mass[i] * rt->pos_x[i];
        c->y += rt->mass[i] * rt->pos_y[i];
        c->z += rt->mass[i] * rt->pos_z[i];
        c->charge += rt->charge[i];
        c->phase += rt->I_phase[i];
        
        const bool* row = &rt->adjacency[i * cfg->N];
        for (size_t j = 0; j < cfg->N; j++)
        {
            if (row[j] && !rt->visited[j] && rt->mass[j] > 1e-6)
            {
                if (stack_ptr >= cfg->N) break;
                stack[stack_ptr++] = j;
            }
        }
    }
}

void iwt_detect_clusters(const iwt_runtime_t rt, const iwt_config_t cfg)
{   
    // Alle Knoten als unbesucht markieren
    memset(rt->visited, 0, cfg->N * sizeof(bool));
    
    // Alte Cluster zurücksetzen
    rt->cluster_count = 0;
    
    // Flood-Fill für jeden Knoten
    for (size_t i = 0; i < cfg->N; i++)
    {
        if (rt->visited[i]) continue;
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
        _flood_fill(rt, cfg, i, c);
        
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
