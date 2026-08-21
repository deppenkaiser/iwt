#pragma once

#include <stdbool.h>
#include <stddef.h>

// Erzeugt N Knotenpositionen nach dem fraktalen Dodekaeder-Verfahren:
// Level 0 = ein Dodekaeder (Radius R0) im Ursprung. Jede VOLLSTAENDIG
// gefuellte Schale erzeugt an ihren 12 Flaechenzentren (= Ecken des
// eingepassten dualen Ikosaeders) je ein um 1/s verkleinertes, um den
// Goldenen Winkel Psi (um die jeweilige Spoke-Achse) gedrehtes
// Kind-Dodekaeder. Breadth-First bis exakt N Punkte erzeugt sind
// (Wurzel + alle Zwischenstufen kombiniert; letzte Schale ggf.
// unvollstaendig und wird deterministisch angeschnitten).
bool dodecahedron_generate_points(double* pos_x, double* pos_y, double* pos_z, size_t N, double R0);

// Anzahl der "Ueberstufen" (extra_levels): jede zusaetzliche Stufe skaliert
// R0 um den Faktor s (=2+phi) nach oben, wodurch die bisherige "eine
// Wurzel" automatisch zu einer von 12 Kind-Ebenen eines (unsichtbaren)
// groesseren Eltern-Dodekaeders wird - dieselbe Rekursionsregel wie überall
// sonst, kein separates Gitter noetig. extra_levels=1 -> 12 sichtbare
// "Wurzeln", extra_levels=2 -> 12*12=144, usw.
bool dodecahedron_generate_points_ex(double* pos_x, double* pos_y, double* pos_z, size_t N, double R0, int extra_levels);
