#include "iwt.h"
#include <math.h>
#include <stdio.h>

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

double iwt_gamma_IWT(void)
{
    return 1.0;
}

double iwt_g(double I)
{
	const double S = 1e6;
    // Nullstellen bei Fixpunkten: 0.01, 0.25, 1.0
    // Minima bei instabilen Teilchen: 0.05, 0.06
    double term1 = (I - 0.25) * (I - 0.25);
    double term2 = (I - 1.0) * (I - 1.0);
    double term3 = (I - 0.05) * (I - 0.05) + 0.01;
    double term4 = (I - 0.06) * (I - 0.06) + 0.01;
    return S * term1 * term2 * term3 * term4;
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
