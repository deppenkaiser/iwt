#include "iwt.h"
#include <math.h>

#include <math.h>

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

double iwt_delta_I_min()
{
    return (3.0 - iwt_fractal_dimension()) / 3.0;
}

double iwt_alpha_IWT()
{
    const double h = 6.62607015e-34;        // J*s
    const double D = iwt_fractal_dimension();
    const double Delta_I_min = iwt_delta_I_min();
	const double l0 = iwt_fundamental_length();

    // alpha_IWT = h / (Delta_I_min * l0^2)
    return h / (Delta_I_min * l0 * l0);
}

double iwt_beta_IWT()
{
    const double G = 6.67430e-11;           // m^3/(kg*s^2)
	const double T = iwt_fundamental_time();
	const double l0 = iwt_fundamental_length();
	const double D = iwt_fractal_dimension();

    // beta_IWT = G * T^2 / l0^(3-D)
    return G * T * T / pow(l0, 3.0 - D);
}
