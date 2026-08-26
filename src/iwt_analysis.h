// ============================================================================
// iwt_analysis.h
// ============================================================================

#ifndef IWT_ANALYSIS_H
#define IWT_ANALYSIS_H

#include "gui.h"

/*
 * Exportiert den aktuellen Simulationszustand als CSV-Satz unter
 * experiments/exp_<Zeitstempel>/ :
 *   spectrum.csv  - Klassenzahlen pro Partikeltyp je Schritt
 *   clusters.csv  - alle Cluster mit Masse, Ladung, Phase, Position, Velocity
 *   nodes.csv     - alle Knoten mit Position und Informationswerten
 */
void iwt_analysis_export_csv(iwt_gui_data_t data);

/*
 * Sichert den aktuellen OpenGL-Framebuffers als PNG unter shots/ .
 * Muss innerhalb des GL-Kontextes aufgerufen werden (nach dem Draw).
 */
void iwt_analysis_capture_screenshot(iwt_gui_data_t data);

/*
 * Erzeugt ein Histogramm fuer das aktuelle Experiment (Hotkey 'h').
 */
void iwt_analysis_generate_histogram(iwt_gui_data_t data);

#endif
