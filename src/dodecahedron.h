#pragma once

#include <stddef.h>
#include <stdbool.h>

// Erzeugt N Knotenpositionen nach dem fraktalen Dodekaeder-Verfahren:
// Level 0 = ein Dodekaeder (Radius R0) im Ursprung. Jede VOLLSTAENDIG
// gefuellte Schale erzeugt an ihren 12 Flaechenzentren (= Ecken des
// eingepassten dualen Ikosaeders) je ein um 1/s verkleinertes, um den
// Goldenen Winkel Psi (um die jeweilige Spoke-Achse) gedrehtes
// Kind-Dodekaeder. Breadth-First bis exakt N Punkte erzeugt sind
// (Wurzel + alle Zwischenstufen kombiniert; letzte Schale ggf.
// unvollstaendig und wird deterministisch angeschnitten).
bool dodecahedron_generate_points(double* pos_x, double* pos_y, double* pos_z, size_t N, double R0);

// Wie dodecahedron_generate_points(), aber mit mehreren Wurzel-Dodekaedern,
// angeordnet in einem regelmäßigen nx*ny*nz-Gitter (Abstand `spacing`
// zwischen den Zentren). N wird gleichmäßig auf alle Wurzeln aufgeteilt
// (Rest geht an die ersten Wurzeln), jede Wurzel bildet ihr eigenes
// vollständiges Fraktal.
bool dodecahedron_generate_multi_root_points(double* pos_x, double* pos_y, double* pos_z, size_t N, double R0, int nx, int ny, int nz, double spacing);
