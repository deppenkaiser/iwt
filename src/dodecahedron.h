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

// Wie oben, zusaetzlich mit Ausgabe der Zell-ID je Knoten (darf NULL sein):
// Alle 20 Ecken einer BFS-Zelle teilen dieselbe ID; die IDs sind
// fortlaufend in Enqueue-Reihenfolge (Wurzel = 0).
bool dodecahedron_generate_points_ex(double* pos_x, double* pos_y, double* pos_z, size_t N, double R0, int extra_levels, int* out_cell_ids);
