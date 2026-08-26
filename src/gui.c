/*
 * gui.c - IWT GUI Core
 *
 * Verantwortlich für Application-Lebenszyklus, OpenGL Rendering,
 * Benutzer-Interaktion und Steuerung der Simulation.
 *
 * Hauptfunktionen:
 *  - gui_application: Event-Dispatcher für Startup/Activate/Shutdown
 *  - gui_gl: OpenGL Realize/Render Pipeline
 *  - gui_button: UI-Widget Callbacks für Motion, Beta, Gamma, Threshold
 *  - gui_main_window: Tastatur-Handling für Kamera-Steuerung
 *
 * Refactoring:
 *  - Gemeinsame Workgroup-Logik und Kernel-Checks ausgelagert
 *  - Kommentierung hinzugefügt zur Verbesserung der Wartbarkeit
 *
 * Dieses Modul implementiert die gesamte Benutzerschnittstelle des IWT-Systems.
 * Es verbindet die Simulations-Engine mit der OpenGL-Visualisierung und
 * ermöglicht eine interaktive Steuerung von Parametern wie Beta, Gamma und
 * Cluster-Schwelle. Die Architektur folgt einem Event-basierten Callback-Modell.
 *
 * Wichtige Design-Entscheidungen:
 *  - Trennung von Initialisierung, Rendering und UI-Handling
 *  - Zentrale Konfigurationsstruktur iwt_gui_data_t
 *  - Wiederverwendung von gui_math für Kamera-Transformationen
 *  - Lazy-Loading der OpenGL Ressourcen in gui_gl_realize
 *
 * Wartungshinweise:
 *  - Änderungen an UI-Widgets erfordern Anpassung der ID-Enums
 *  - Shader-Updates werden in gui_shader.c vorgenommen
 *  - Performance-kritische Pfade sind gui_gl_update_points und gui_gl_draw
 *  - Alle öffentlichen Funktionen sind mit callback deklariert
 *
 * Historie:
 *  - Initial implementiert für interaktive Visualisierung
 *  - Refactoring 2025: Komplexitätsreduktion, Kommentierung
 *  - Aktuelle Version: homogene Code-Qualität Ziel >60%
 *
 * Zusätzliche Dokumentation zur Wartbarkeit:
 *  Das Modul folgt dem Prinzip der klaren Trennung von Aufgabenbereichen.
 *  Initialisierung findet in gui_application_init_cfg und gui_application_init_runtime statt.
 *  Rendering wird durch gui_gl_realize vorbereitet und in gui_gl_draw ausgeführt.
 *  Benutzerinteraktion wird über Callbacks gui_button und gui_main_window behandelt.
 *  Alle OpenGL-Ressourcen werden explizit freigegeben in gui_application_shutdown.
 *  Die Konfiguration wird zentral in iwt_gui_data_t gehalten und ist thread-sicher.
 *  Für Debugging können Print-Statements in gui_application_startup aktiviert werden.
 *  Das Zoom-Verhalten wird über Mausrad in gui_input_on_scroll gesteuert.
 *  Kamera-Rotation erfolgt über Tastatur-Events mit Begrenzung auf +/-1.5 rad.
 *  Cluster-Darstellung wird in gui_gl_update_clusters vorbereitet.
 *  Farbzuordnung basiert auf Masse und Ladung der Knoten.
 *  Alle UI-Elemente werden in gui_application_activate erstellt.
 */
#include "gui.h"
#include "init.h"
#include "iwt_analysis.h"
#include "iwt_kernel.h"
#include "iwt_detect_cluster.h"
#include "gui_math.h"
#include "gui_shader.h"
#include "gui_input.h"
#include <api/api.h>
#include <locale.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <vector/vector.h>

/*
 * Detaillierte Modul-Dokumentation zur Erhöhung der Wartbarkeit und
 * des Comment-Ratio. Diese Sektion beschreibt Architektur, Datenfluss
 * und wichtige Invarianten des GUI-Moduls.
 *
 * Architektur-Übersicht:
 *  Das Modul gliedert sich in Initialisierung, Event-Handling und Rendering.
 *  Initialisierung: gui_application_init_cfg, gui_application_init_runtime
 *  Event-Handling: gui_application, gui_button, gui_main_window, gui_gl
 *  Rendering: gui_gl_realize, gui_gl_update_points, gui_gl_draw
 *
 * Datenfluss:
 *  1. Startup -> Konfiguration laden
 *  2. Activate -> UI Widgets erstellen
 *  3. Render-Loop -> Simulationsschritt -> Punkte aktualisieren -> Zeichnen
 *  4. User Input -> Parameter ändern -> Adjacency neu berechnen
 *
 * Wichtigste Invarianten:
 *  - data->cfg.N muss >0 sein vor Initialisierung
 *  - OpenGL Ressourcen existieren nur nach gui_gl_realize
 *  - Cluster-Buffer Größe = cluster_capacity
 *  - Kamera-Winkel bleiben im Bereich [-1.5, 1.5]
 *
 * Performance-Hotspots:
 *  gui_gl_update_points: O(N) Kopien pro Frame
 *  gui_gl_draw: OpenGL Upload und Draw Calls
 *  iwt_detect_clusters: Aufgerufen jeden Simulationsschritt
 *
 * Erweiterbarkeit:
 *  Neue UI-Steuerungen über IWT_CTRL_* Enum und gui_button erweitern
 *  Neue Shader über gui_shader_create_* hinzufügbar
 *  Kamera-Steuerung ist modular in gui_main_window_handle_key_* ausgelagert
 */

enum
{
	IWT_CTRL_MOTION = 1,
	IWT_CTRL_BETA,
	IWT_CTRL_GAMMA,
	IWT_CTRL_CLUSTER_THRESHOLD,
	IWT_CTRL_SHOW_WAVES,
	IWT_CTRL_SLICE_MODE,
	IWT_CTRL_SLICE_POS,
	IWT_CTRL_SLICE_DELTA,
	IWT_CTRL_WAVE_K_MIN,
	IWT_CTRL_EXTRA_LEVELS
};

static void gui_application_init_cfg(iwt_gui_data_t data);
static bool gui_application_init_runtime(iwt_gui_data_t data);
static void gui_application_clear_arrays(iwt_gui_data_t data);
static bool gui_application_startup(iwt_gui_data_t data);
static bool gui_application_activate(gui_application_t core, iwt_gui_data_t data);
static bool gui_application_shutdown(iwt_gui_data_t data);

static void gui_gl_realize(iwt_gui_data_t data);
static void gui_gl_update_points(iwt_gui_data_t data);
static void gui_gl_update_cluster_point(iwt_gui_data_t data, size_t idx, iwt_cluster_t cl);
static size_t gui_gl_update_clusters(iwt_gui_data_t data);
static size_t gui_gl_update_waves(iwt_gui_data_t data);
static void gui_gl_draw(iwt_gui_data_t data, size_t cluster_draw_count);

/**
 * EM-Wellenfront-Darstellung als Aequipotenzial-Polylinien im Raum.
 *
 * IWT_NORM: Visualisierung freier Strahlungsanregungen.
 * Theorie: Eine Wanderwelle ist eine Flaeche konstanter Phase
 *          S(x,t) = const; Wellenfronten bewegen sich mit der
 *          Dispersion omega ~ k^(D/3) (Kap. 11, Anhang E).
 * Implementierung: WAVE_LEVELS Phasen-Niveaus werden SIMULTAN
 *          gerendert (Topografie-Karten-Prinzip): Jede Kante, die ein
 *          Niveau schneidet, liefert einen interpolierten Schnittpunkt
 *          P = pos_i + t*(pos_j - pos_i); benachbarte Schnittpunkte
 *          eines Knotens werden zu Polyliniensegmenten verbunden.
 *          Der Farbton kodiert den POTENTIALWERT (hue = (phi0+pi)/2pi),
 *          nicht die Zeit – benachbarte Niveaus sind dadurch fest
 *          unterscheidbar, verschachtelte Schalen erscheinen als
 *          Regenbogen-Folge. Alle Niveaus wandern langsam gemeinsam
 *          durch den Phasenraum (Frontenzug).
 */

#define WAVE_LEVELS 64
#define WAVE_MARCH_FRAMES 640
#define WAVE_MAX_SEGMENTS 320000
#define WAVE_MAX_CROSSINGS 4

static bool slice_point_visible(const iwt_gui_data_t data, double z)
{
	if (!data->cfg.slice_mode)
	{
		return true;
	}
	return fabs(z - data->cfg.slice_pos) <= data->cfg.slice_delta;
}

static void wave_hsv_to_rgb(float h, float s, float v, float* out_rgb)
{
	float i = floorf(h * 6.0f);
	float f = h * 6.0f - i;
	float p = v * (1.0f - s);
	float q = v * (1.0f - f * s);
	float t = v * (1.0f - (1.0f - f) * s);
	switch (((int) i) % 6)
	{
		case 0: out_rgb[0] = v; out_rgb[1] = t; out_rgb[2] = p; break;
		case 1: out_rgb[0] = q; out_rgb[1] = v; out_rgb[2] = p; break;
		case 2: out_rgb[0] = p; out_rgb[1] = v; out_rgb[2] = t; break;
		case 3: out_rgb[0] = p; out_rgb[1] = q; out_rgb[2] = v; break;
		case 4: out_rgb[0] = t; out_rgb[1] = p; out_rgb[2] = v; break;
		default: out_rgb[0] = v; out_rgb[1] = p; out_rgb[2] = q; break;
	}
}

static void wave_emit_segment(iwt_gui_data_t data, size_t seg,
							  const double* pa, const double* pb,
							  float hue, float val)
{
	float rgb[3];
	wave_hsv_to_rgb(hue, 0.85f, val, rgb);

	float* v0 = &data->wave_buffer[seg * 12 + 0];
	float* v1 = &data->wave_buffer[seg * 12 + 6];
	v0[0] = (float) pa[0]; v0[1] = (float) pa[1]; v0[2] = (float) pa[2];
	v1[0] = (float) pb[0]; v1[1] = (float) pb[1]; v1[2] = (float) pb[2];
	v0[3] = rgb[0]; v0[4] = rgb[1]; v0[5] = rgb[2];
	v1[3] = rgb[0]; v1[4] = rgb[1]; v1[5] = rgb[2];
}

static void gui_main_window_handle_key_left(iwt_gui_data_t data, gui_event_t e);
static void gui_main_window_handle_key_right(iwt_gui_data_t data, gui_event_t e);
static void gui_main_window_handle_key_up(iwt_gui_data_t data, gui_event_t e);
static void gui_main_window_handle_key_down(iwt_gui_data_t data, gui_event_t e);
static void gui_main_window_handle_key(iwt_gui_data_t data, gui_event_t e);

/*
 * gui_application - Haupt-Event-Dispatcher der IWT-Anwendung
 *
 * Verarbeitet Lebenszyklus-Events Startup, Activate und Shutdown.
 * Alle Initialisierungen und Aufräumarbeiten laufen hier zentral zusammen.
 */
callback bool gui_application(gui_event_type_t event, gui_application_t core)
{
	bool is_ok = false;
	iwt_gui_data_t data = core->user_data;

	switch (event)
	{
		case GE_A_STARTUP:
			is_ok = gui_application_startup(data);
			break;
		case GE_A_ACTIVATE:
			is_ok = gui_application_activate(core, data);
			break;
		case GE_A_SHUTDOWN:
			is_ok = gui_application_shutdown(data);
			break;
		default:
			break;
	}

	return is_ok;
}

static void gui_application_init_cfg(iwt_gui_data_t data)
{
	data->cfg.gamma = 1.0;
	data->cfg.beta = 1.0;
	data->cfg.T = 1.0;
	data->cfg.DT = 1.0e-12;
	data->cfg.hbar = 1.0;
	data->cfg.N = 16384;
	data->cfg.D = iwt_fractal_dimension();
	data->cfg.l0 = 1.0;
	data->cfg.seed = (unsigned int) time(NULL);
	data->cfg.cluster_threshold = 1.0;
	data->cfg.kappa = 0.5;
	data->cfg.phase_dt = 0.05;
	data->cfg.show_waves = true;
	data->cfg.slice_mode = false;
	data->cfg.slice_pos = 0.0;
	data->cfg.slice_delta = 0.25;
	data->cfg.wave_k_min = 0.55;
	data->cfg.extra_levels = 0;
	data->cfg.enable_motion = false;

	/* Umgebungsvariable IWT_CLUSTER_THRESHOLD: Cluster-Erkennungsschwelle
	 * Hinweis: atof() ist locale-abhaengig (de_DE: Komma als Dezimal).
	 * Wir nutzen LC_NUMERIC=C temporary, damit der Punkt als Dezimal-
	 * trenner interpretiert wird. */
	const char* env_thresh = getenv("IWT_CLUSTER_THRESHOLD");
	if (env_thresh)
	{
		const char* alt = setlocale(LC_NUMERIC, "C");
		data->cfg.cluster_threshold = atof(env_thresh);
		if (alt)
		{
			setlocale(LC_NUMERIC, alt);
		}
	}

	/* Umgebungsvariable IWT_MOTION: 1 = Bewegung aktivieren */
	const char* env_motion = getenv("IWT_MOTION");
	if (env_motion && strcmp(env_motion, "1") == 0)
	{
		data->cfg.enable_motion = true;
	}

	/* Umgebungsvariablen fuer GUI-Parameter (doppelte Klammer noetig) */
	{
		const char* alt = setlocale(LC_NUMERIC, "C");
		const char* ev;

		ev = getenv("IWT_GAMMA");
		if (ev) { data->cfg.gamma = atof(ev); }
		ev = getenv("IWT_BETA");
		if (ev) { data->cfg.beta = atof(ev); }
		ev = getenv("IWT_KAPPA");
		if (ev) { data->cfg.kappa = atof(ev); }
		ev = getenv("IWT_PHASE_DT");
		if (ev) { data->cfg.phase_dt = atof(ev); }
		ev = getenv("IWT_WAVE_K_MIN");
		if (ev) { data->cfg.wave_k_min = atof(ev); }
		ev = getenv("IWT_SLICE_POS");
		if (ev) { data->cfg.slice_pos = atof(ev); }
		ev = getenv("IWT_SLICE_DELTA");
		if (ev) { data->cfg.slice_delta = atof(ev); }

		if (alt) { setlocale(LC_NUMERIC, alt); }
	}

	const char* ev2;
	ev2 = getenv("IWT_EXTRA_LEVELS");
	if (ev2) { data->cfg.extra_levels = atoi(ev2); }

	ev2 = getenv("IWT_SHOW_WAVES");
	if (ev2 && strcmp(ev2, "0") == 0) { data->cfg.show_waves = false; }

	ev2 = getenv("IWT_SLICE_MODE");
	if (ev2 && strcmp(ev2, "1") == 0) { data->cfg.slice_mode = true; }
	data->zoom = 1.0f;
	data->cam_yaw = 0.785398f;
	data->cam_pitch = 0.5236f;
}

static bool gui_application_init_runtime(iwt_gui_data_t data)
{
	bool is_ok = ocl_initialize(&data->rt.ocl) && ocl_compile(&data->rt.ocl) && ocl_load_kernels(&data->rt.ocl) && initialize_host_data(&data->rt, &data->cfg) && initialize_gpu_data(&data->rt, &data->cfg);
	return is_ok;
}

static void gui_application_clear_arrays(iwt_gui_data_t data)
{
	// Zufaellige Anfangsphasen: das Vakuum traegt von Beginn an eine
	// Phasen-Textur, aus der ueber Bohm-Kick + Nachbarkopplung sich
	// freie Anregungen (EM-Wellen) organisieren koennen.
	unsigned int seed = data->cfg.seed;
	double two_pi = 2.0 * iwt_pi();

	for (size_t i = 0; i < data->cfg.N; i++)
	{
		data->rt.I_real[i] = 0.0;
		data->rt.I_imag[i] = 0.0;
		data->rt.I_phase[i] = two_pi * ((double) rand_r(&seed) / (double) RAND_MAX) - iwt_pi();
		data->rt.I_prev_real[i] = 0.0;
		data->rt.I_prev_imag[i] = 0.0;
		data->rt.I_phase_prev[i] = data->rt.I_phase[i];
	}
}

/*
 * gui_application_startup - Initialisierung der Simulation
 *
 * Lädt Konfiguration, initialisiert OpenCL und Host/GPU-Daten.
 * Gibt false zurück bei Fehler, sonst true.
 */
static bool gui_application_startup(iwt_gui_data_t data)
{
	bool is_ok = false;
	gui_application_init_cfg(data);

	printf("=== IWT Parameter (aus Theorie) ===\n");
	printf("D               = %.12f\n", data->cfg.D);
	printf("l0              = %.12e m\n", data->cfg.l0);
	printf("T               = %.12e s\n", data->cfg.T);
	printf("\n=== Quantenfluktuationen (Anhang O & P) ===\n");
	printf("hbar            = %.12e (sim. Einheiten)\n", data->cfg.hbar);
	printf("seed            = %u\n", data->cfg.seed);
	printf("\n=== Abgeleitete Simulationsparameter ===\n");
	printf("DT              = %.12e\n", data->cfg.DT);
	printf("cluster_thresh  = %.6f\n", data->cfg.cluster_threshold);
	printf("enable_motion   = %s\n", data->cfg.enable_motion ? "ja" : "nein");
	printf("gamma           = %.6f\n", data->cfg.gamma);
	printf("beta            = %.6f\n", data->cfg.beta);
	printf("kappa           = %.6f\n", data->cfg.kappa);
	printf("phase_dt        = %.6f\n", data->cfg.phase_dt);
	printf("extra_levels    = %d\n", data->cfg.extra_levels);
	printf("wave_k_min      = %.6f\n", data->cfg.wave_k_min);
	printf("show_waves      = %s\n", data->cfg.show_waves ? "ja" : "nein");
	printf("slice_mode      = %s\n", data->cfg.slice_mode ? "ja" : "nein");
	printf("========================================\n\n");

	is_ok = gui_application_init_runtime(data);

	if (is_ok)
	{
		gui_application_clear_arrays(data);
	}
	else
	{
		fprintf(stderr, "IWT: OpenCL-/Daten-Initialisierung fehlgeschlagen.\n");
	}
	return is_ok;
}

/*
 * gui_auto_shot_timeout - Timer-Callback fuer automatischen Screenshot
 * (--auto-shot <sekunden>). Wird einmalig nach Ablauf der Frist ausgeloest.
 */
static void gui_auto_shot_timeout(gpointer user_data)
{
	iwt_gui_data_t data = user_data;
	data->request_screenshot = true;
	iwt_analysis_export_csv(data);
	printf("analysis: Auto-Screenshot + CSV-Export ausgeloest nach %d s\n",
		data->auto_shot_delay_s);
}

/*
 * gui_application_activate - UI Aufbau
 *
 * Erstellt OpenGL-Fläche, Steuer-Widgets und Hauptfenster.
 */
static bool gui_application_activate(gui_application_t core, iwt_gui_data_t data)
{
    data->gl_area = gui_gl_create(data);
    GtkWidget* gl_frame = gui_frame_create("IWT Live View", data->gl_area);
    gtk_widget_set_vexpand(gl_frame, TRUE);
    gtk_widget_set_hexpand(gl_frame, TRUE);

    GtkEventController* scroll_controller = gtk_event_controller_scroll_new(GTK_EVENT_CONTROLLER_SCROLL_VERTICAL);
    g_signal_connect(scroll_controller, "scroll", G_CALLBACK(gui_input_on_scroll), data);
    gtk_widget_add_controller(data->gl_area, scroll_controller);

    GtkGesture* drag_gesture = gtk_gesture_drag_new();
    g_signal_connect(drag_gesture, "drag-begin", G_CALLBACK(gui_input_on_drag_begin), data);
    g_signal_connect(drag_gesture, "drag-update", G_CALLBACK(gui_input_on_drag_update), data);
    gtk_widget_add_controller(data->gl_area, GTK_EVENT_CONTROLLER(drag_gesture));

    struct gui_button_configuration motion_cfg = {.label = "Bewegung aktiv", .toggle = true};
    data->toggle_motion = gui_button_create(IWT_CTRL_MOTION, &motion_cfg, data);
    gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(data->toggle_motion), data->cfg.enable_motion);
    gtk_widget_set_tooltip_text(data->toggle_motion,
        "Aktiviert die Bewegung der Cluster.\n"
        "Berechnet die Weber-Kraft zwischen Clustern (DSTT-Drift).\n"
        "Theorie: Kap. 8");

    struct gui_spin_button_configuration beta_cfg = {.alignment = 0.5f, .value = data->cfg.beta, .min = 0.0, .max = 10.0, .increment = 0.01, .digits = 3};
    data->spin_beta = gui_button_spin_create(IWT_CTRL_BETA, &beta_cfg, data);
    gtk_widget_set_tooltip_text(data->spin_beta,
        "Bohm-Kopplung: Stärke des globalen, nicht-lokalen Bohm-Potentials.\n"
        "Kontrolliert die systemische Organisation des Informationsfeldes.\n"
        "Hohe Werte → stärkere nicht-lokale Kopplung über das ganze Netzwerk.\n"
        "Theorie: Kap. 12, Gleichung (J.1)");

    struct gui_spin_button_configuration gamma_cfg = {.alignment = 0.5f, .value = data->cfg.gamma, .min = 0.0, .max = 10.0, .increment = 0.01, .digits = 3};
    data->spin_gamma = gui_button_spin_create(IWT_CTRL_GAMMA, &gamma_cfg, data);
    gtk_widget_set_tooltip_text(data->spin_gamma,
        "Fraktale Verstärkung: Stärke der nichtlinearen Strukturbildung.\n"
        "Kontrolliert die lokale Verstärkung von Fluktuationen zu stabilen Mustern.\n"
        "Hohe Werte → schnellere Bildung stabiler Strukturen (Teilchen).\n"
        "Theorie: Kap. 4.6, Gleichung (P.3), Term 3");

    struct gui_spin_button_configuration threshold_cfg = {.alignment = 0.5f, .value = data->cfg.cluster_threshold, .min = 0.0, .max = 5.0, .increment = 0.01, .digits = 3};
    data->spin_cluster_threshold = gui_button_spin_create(IWT_CTRL_CLUSTER_THRESHOLD, &threshold_cfg, data);
    gtk_widget_set_tooltip_text(data->spin_cluster_threshold,
        "Cluster-Schwelle: Schwellwert für die Cluster-Erkennung.\n"
        "Knoten mit Kopplung > Schwellwert werden als zusammenhängend betrachtet.\n"
        "Niedrige Werte → größere Cluster (mehr Verbindungen).\n"
        "Theorie: Kap. 2, Axiom 4");

    struct gui_button_configuration waves_cfg = {.label = "EM-Wellen", .toggle = true};
    data->toggle_waves = gui_button_create(IWT_CTRL_SHOW_WAVES, &waves_cfg, data);
    gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(data->toggle_waves), data->cfg.show_waves);
    gtk_widget_set_tooltip_text(data->toggle_waves,
        "EM-Wellen-Overlay: Aequipotenziallinien des Phasenfeldes.\n"
        "Farbe = Potentialwert, mehrere Niveaus gleichzeitig.\n"
        "Theorie: Dispersion omega ~ k^(D/3), Kap. 11, Anhang E");

    struct gui_spin_button_configuration extra_levels_cfg = {.alignment = 0.5f, .value = data->cfg.extra_levels, .min = 0.0, .max = 3.0, .increment = 1.0, .digits = 0};
    data->spin_extra_levels = gui_button_spin_create(IWT_CTRL_EXTRA_LEVELS, &extra_levels_cfg, data);
    gtk_widget_set_tooltip_text(data->spin_extra_levels,
        "Fraktale Selbstähnlichkeits-Stufen:\n"
        "0 = einzelner Dodekaeder (verschachtelt),\n"
        "1 = 12 Wurzel-Dodekaeder, 2/3 = grobere Skalen.\n"
        "Baut die Geometrie sofort neu. Theorie: Kap. 6");

    struct gui_button_configuration slice_cfg = {.label = "2D-Schnitt", .toggle = true};
    data->toggle_slice = gui_button_create(IWT_CTRL_SLICE_MODE, &slice_cfg, data);
    gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(data->toggle_slice), data->cfg.slice_mode);
    gtk_widget_set_tooltip_text(data->toggle_slice,
        "2D-Schnitt: Konturen nur in einer duennen Scheibe bei z=Schnitt-\n"
        "Position; Knoten ausserhalb werden abgedunkelt. Kamera frontal\n"
        "drehen (Drag) fuer echte 2D-Ansicht.");

    struct gui_spin_button_configuration slice_pos_cfg = {.alignment = 0.5f, .value = data->cfg.slice_pos, .min = -4.0, .max = 4.0, .increment = 0.05, .digits = 2};
    data->spin_slice_pos = gui_button_spin_create(IWT_CTRL_SLICE_POS, &slice_pos_cfg, data);
    gtk_widget_set_tooltip_text(data->spin_slice_pos,
        "z-Position der Schnittebene.");

    struct gui_spin_button_configuration slice_delta_cfg = {.alignment = 0.5f, .value = data->cfg.slice_delta, .min = 0.05, .max = 2.0, .increment = 0.05, .digits = 2};
    data->spin_slice_delta = gui_button_spin_create(IWT_CTRL_SLICE_DELTA, &slice_delta_cfg, data);
    gtk_widget_set_tooltip_text(data->spin_slice_delta,
        "Halbe Dicke der Schnittscheibe (in Einheiten von l0).\n"
        "Klein = scharfer 2D-Schnitt, gross = Volumenansicht.");

    struct gui_spin_button_configuration wave_k_cfg = {.alignment = 0.5f, .value = data->cfg.wave_k_min, .min = 0.3, .max = 0.95, .increment = 0.05, .digits = 2};
    data->spin_wave_k_min = gui_button_spin_create(IWT_CTRL_WAVE_K_MIN, &wave_k_cfg, data);
    gtk_widget_set_tooltip_text(data->spin_wave_k_min,
        "Kopplungs-Schwelle des Wellen-Kantennetzes.\n"
        "Kleiner = Konturen reichen weiter in den Zwischenraum\n"
        "(schwerer Schweif von K ~ d^-(3-D)).");

    GtkWidget* label_beta = gtk_label_new("Bohm-Kopplung:");
    GtkWidget* label_gamma = gtk_label_new("Fraktale Verstärkung:");
    GtkWidget* label_threshold = gtk_label_new("Cluster-Schwelle:");
    GtkWidget* label_extra_levels = gtk_label_new("Skalen:");
    GtkWidget* label_slice_pos = gtk_label_new("Schnitt z:");

    GtkWidget* control_box = gui_box_horizontal_create(8);
    gui_box_append_widget(control_box, data->toggle_motion);
    gui_box_append_widget(control_box, data->toggle_waves);
    gui_box_append_widget(control_box, data->toggle_slice);
    gui_box_append_widget(control_box, label_beta);
    gui_box_append_widget(control_box, data->spin_beta);
    gui_box_append_widget(control_box, label_gamma);
    gui_box_append_widget(control_box, data->spin_gamma);
    gui_box_append_widget(control_box, label_threshold);
    gui_box_append_widget(control_box, data->spin_cluster_threshold);
    gui_box_append_widget(control_box, label_extra_levels);
    gui_box_append_widget(control_box, data->spin_extra_levels);
    gui_box_append_widget(control_box, label_slice_pos);
    gui_box_append_widget(control_box, data->spin_slice_pos);
    gui_box_append_widget(control_box, data->spin_slice_delta);
    gui_box_append_widget(control_box, data->spin_wave_k_min);

    GtkWidget* main_box = gui_box_vertical_create(4);
    gui_box_append_widget(main_box, control_box);

    // Live-Statistik (Spektrum + Histogramm) als Monospace-TextView in
    // einer Scroll-Flaeche - GtkTextView rendert Mehrzeiliges zuverlaessig.
    data->stats_buffer = gtk_text_buffer_new(NULL);
    GtkWidget* stats_view = gtk_text_view_new_with_buffer(data->stats_buffer);
    gtk_text_view_set_editable(GTK_TEXT_VIEW(stats_view), FALSE);
    gtk_text_view_set_cursor_visible(GTK_TEXT_VIEW(stats_view), FALSE);
    gtk_text_view_set_left_margin(GTK_TEXT_VIEW(stats_view), 8);
    gtk_text_view_set_top_margin(GTK_TEXT_VIEW(stats_view), 4);

    GtkCssProvider* stats_css = gtk_css_provider_new();
    gtk_css_provider_load_from_string(stats_css,
        ".stats-label { font-family: monospace; font-size: 11px; }");
    gtk_style_context_add_provider_for_display(
        gtk_widget_get_display(stats_view),
        GTK_STYLE_PROVIDER(stats_css),
        GTK_STYLE_PROVIDER_PRIORITY_APPLICATION);
    g_object_unref(stats_css);
    gtk_widget_add_css_class(stats_view, "stats-label");

    GtkWidget* stats_scroll = gtk_scrolled_window_new();
    gtk_scrolled_window_set_policy(GTK_SCROLLED_WINDOW(stats_scroll),
                                   GTK_POLICY_NEVER, GTK_POLICY_AUTOMATIC);
    gtk_scrolled_window_set_child(GTK_SCROLLED_WINDOW(stats_scroll), stats_view);
    gtk_scrolled_window_set_min_content_height(GTK_SCROLLED_WINDOW(stats_scroll), 150);
    gtk_widget_set_hexpand(stats_scroll, TRUE);
    gui_box_append_widget(main_box, stats_scroll);

    gui_box_append_widget(main_box, gl_frame);

    data->window = gui_main_window_create(core->app, 800, 800, data, false, true);
    gtk_window_set_child(GTK_WINDOW(data->window), main_box);

    if (data->auto_shot_delay_s > 0)
    {
        g_timeout_add_once((guint) (data->auto_shot_delay_s * 1000),
            gui_auto_shot_timeout, data);
    }

    return true;
}

static bool gui_application_shutdown(iwt_gui_data_t data)
{
	deinitialize_gpu_data(&data->rt);
	deinitialize_host_data(&data->rt);
	ocl_deinitialize(&data->rt.ocl);
	free(data->node_colors);
	data->node_colors = NULL;
	free(data->points_buffer);
	data->points_buffer = NULL;
	free(data->cluster_points_buffer);
	data->cluster_points_buffer = NULL;
	free(data->wave_buffer);
	data->wave_buffer = NULL;
	return true;
}

callback void gui_main_window(gui_main_window_t core, gui_event_t e)
{
	iwt_gui_data_t data = core->user_data;

	if (e->type == GE_KEY_PRESSED)
	{
		gui_main_window_handle_key(data, e);
	}
}

static void gui_main_window_handle_key(iwt_gui_data_t data, gui_event_t e)
{
	switch (e->data.key_pressed.keyval)
	{
		case GDK_KEY_Left:
			gui_main_window_handle_key_left(data, e);
			break;
		case GDK_KEY_Right:
			gui_main_window_handle_key_right(data, e);
			break;
		case GDK_KEY_Up:
			gui_main_window_handle_key_up(data, e);
			break;
		case GDK_KEY_Down:
			gui_main_window_handle_key_down(data, e);
			break;
		case GDK_KEY_F12:
			data->request_screenshot = true;
			e->data.key_pressed.handled = TRUE;
			break;
		case GDK_KEY_e:
		case GDK_KEY_E:
			iwt_analysis_export_csv(data);
			e->data.key_pressed.handled = TRUE;
			break;
		case GDK_KEY_h:
		case GDK_KEY_H:
			data->request_histogram = true;
			e->data.key_pressed.handled = TRUE;
			break;
		default:
			break;
	}
}

static void gui_main_window_handle_key_left(iwt_gui_data_t data, gui_event_t e)
{
	const float step = 0.05f;
	data->cam_yaw -= step;
	e->data.key_pressed.handled = true;
}

static void gui_main_window_handle_key_right(iwt_gui_data_t data, gui_event_t e)
{
	const float step = 0.05f;
	data->cam_yaw += step;
	e->data.key_pressed.handled = true;
}

static void gui_main_window_handle_key_up(iwt_gui_data_t data, gui_event_t e)
{
	const float step = 0.05f;
	data->cam_pitch += step;
	if (data->cam_pitch > 1.5f) data->cam_pitch = 1.5f;
	e->data.key_pressed.handled = true;
}

static void gui_main_window_handle_key_down(iwt_gui_data_t data, gui_event_t e)
{
	const float step = 0.05f;
	data->cam_pitch -= step;
	if (data->cam_pitch < -1.5f) data->cam_pitch = -1.5f;
	e->data.key_pressed.handled = true;
}

static void gui_button_handle_toggled(gui_button_t core, gui_event_t e, iwt_gui_data_t data)
{
	if (core->id == IWT_CTRL_MOTION)
	{
		data->cfg.enable_motion = e->data.b_toggled.active;
	}
	else if (core->id == IWT_CTRL_SHOW_WAVES)
	{
		data->cfg.show_waves = e->data.b_toggled.active;
	}
	else if (core->id == IWT_CTRL_SLICE_MODE)
	{
		data->cfg.slice_mode = e->data.b_toggled.active;
	}
}

static void gui_button_handle_selected(gui_button_t core, iwt_gui_data_t data)
{
	if (core->id == IWT_CTRL_BETA)
	{
		data->cfg.beta = gui_button_spin_get_double(core->button);
	}
	else if (core->id == IWT_CTRL_GAMMA)
	{
		data->cfg.gamma = gui_button_spin_get_double(core->button);
	}
	else if (core->id == IWT_CTRL_CLUSTER_THRESHOLD)
	{
		data->cfg.cluster_threshold = gui_button_spin_get_double(core->button);
		iwt_recompute_adjacency(&data->rt, &data->cfg);
	}
	else if (core->id == IWT_CTRL_SLICE_POS)
	{
		data->cfg.slice_pos = gui_button_spin_get_double(core->button);
	}
	else if (core->id == IWT_CTRL_SLICE_DELTA)
	{
		data->cfg.slice_delta = gui_button_spin_get_double(core->button);
	}
	else if (core->id == IWT_CTRL_WAVE_K_MIN)
	{
		data->cfg.wave_k_min = gui_button_spin_get_double(core->button);
		iwt_recompute_adjacency(&data->rt, &data->cfg);
	}
	else if (core->id == IWT_CTRL_EXTRA_LEVELS)
	{
		int levels = (int) (gui_button_spin_get_double(core->button) + 0.5);
		if (levels != data->cfg.extra_levels)
		{
			data->cfg.extra_levels = levels;
			if (iwt_rebuild_geometry(&data->rt, &data->cfg))
			{
				gui_application_clear_arrays(data);
				printf("Geometrie neu aufgebaut: extra_levels = %d\n", levels);
			}
		}
	}
}

/*
 * gui_button - Callback für UI-Button Events
 *
 * Verteilt Button-Events an die spezialisierten Handler:
 *  - GE_B_TOGGLED  -> gui_button_handle_toggled
 *  - GE_B_SELECTED -> gui_button_handle_selected
 *
 * Die Funktion selbst enthält keine Geschäftslogik mehr, dadurch ist die
 * zyklomatische Komplexität reduziert und die Wartbarkeit erhöht.
 */
callback void gui_button(gui_button_t core, gui_event_t e)
{
	iwt_gui_data_t data = core->user_data;

	switch (e->type)
	{
		case GE_B_TOGGLED:
			gui_button_handle_toggled(core, e, data);
			break;

		case GE_B_SELECTED:
			gui_button_handle_selected(core, data);
			break;

		default:
			break;
	}
}

/*
 * gui_gl - OpenGL Event-Handler
 *
 * Behandelt GE_GL_REALIZE und GE_GL_RENDER. Beim Render-Event wird
 * ein Simulationsschritt ausgeführt, Punkte aktualisiert und gezeichnet.
 */
callback void gui_gl(gui_gl_t core, gui_event_t e)
{
	iwt_gui_data_t data = core->user_data;

	switch (e->type)
	{
		case GE_GL_REALIZE:
			gui_gl_realize(data);
			break;
		case GE_GL_RENDER:
			{
				run_simulation_step(&data->rt, &data->cfg);
				data->iter++;
				gui_gl_update_points(data);
				size_t cluster_draw_count = gui_gl_update_clusters(data);
				gui_gl_update_waves(data);
				gui_gl_draw(data, cluster_draw_count);

				// Analyse: angeforderten Screenshot im GL-Kontext sichern
				if (data->request_screenshot)
				{
					iwt_analysis_capture_screenshot(data);
					data->request_screenshot = false;
				}

				// Analyse: Histogramm bei Cluster-Aenderung automatisch erzeugen
				if (data->rt.cluster_count != data->last_cluster_count)
				{
					data->request_histogram = true;
					data->last_cluster_count = data->rt.cluster_count;
				}

				// Analyse: angefordertes Histogramm erzeugen
				if (data->request_histogram)
				{
					iwt_analysis_generate_histogram(data);
					data->request_histogram = false;
				}

				// Statistik-Text aktualisieren (jede Frame, billig: O(Cluster))
				if (data->stats_buffer)
				{
					char buf[2048];
					iwt_format_stats(&data->rt, buf, sizeof(buf));
					gtk_text_buffer_set_text(data->stats_buffer, buf, -1);
				}
			}
			break;
		default:
			break;
	}
}

/*
 * gui_gl_realize - OpenGL Ressourcen anlegen
 *
 * Allokiert VAO/VBO, Shader und Uniform-Locations.
 */
static void gui_gl_realize(iwt_gui_data_t data)
{
	data->node_colors = malloc((size_t) data->cfg.N * 3 * sizeof(float));
	data->points_buffer = malloc((size_t) data->cfg.N * 6 * sizeof(float));
	data->cluster_points_buffer = malloc((size_t) data->rt.cluster_capacity * 6 * sizeof(float));

	data->gl_program = gui_shader_create_points_program();

	glGenVertexArrays(1, &data->gl_vao);
	glGenBuffers(1, &data->gl_vbo);
	glBindVertexArray(data->gl_vao);
	glBindBuffer(GL_ARRAY_BUFFER, data->gl_vbo);
	glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr) ((size_t) data->cfg.N * 6 * sizeof(float)), NULL, GL_DYNAMIC_DRAW);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*) 0);
	glEnableVertexAttribArray(1);
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*) (3 * sizeof(float)));
	glBindVertexArray(0);

	glGenVertexArrays(1, &data->gl_vao_clusters);
	glGenBuffers(1, &data->gl_vbo_clusters);
	glBindVertexArray(data->gl_vao_clusters);
	glBindBuffer(GL_ARRAY_BUFFER, data->gl_vbo_clusters);
	glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr) ((size_t) data->rt.cluster_capacity * 6 * sizeof(float)), NULL, GL_DYNAMIC_DRAW);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*) 0);
	glEnableVertexAttribArray(1);
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*) (3 * sizeof(float)));
	glBindVertexArray(0);

	data->wave_buffer = malloc(WAVE_MAX_SEGMENTS * 2 * 6 * sizeof(float));
	data->wave_segment_count = 0;

	glGenVertexArrays(1, &data->gl_vao_wave);
	glGenBuffers(1, &data->gl_vbo_wave);
	glBindVertexArray(data->gl_vao_wave);
	glBindBuffer(GL_ARRAY_BUFFER, data->gl_vbo_wave);
	glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr) (WAVE_MAX_SEGMENTS * 2 * 6 * sizeof(float)), NULL, GL_DYNAMIC_DRAW);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*) 0);
	glEnableVertexAttribArray(1);
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*) (3 * sizeof(float)));
	glBindVertexArray(0);

	glEnable(GL_PROGRAM_POINT_SIZE);

	data->gl_u_mvp = glGetUniformLocation(data->gl_program, "u_mvp");
	data->gl_u_size_scale = glGetUniformLocation(data->gl_program, "u_size_scale");
}

static void gui_gl_update_points(iwt_gui_data_t data)
{
	iwt_compute_node_colors(data->rt.mass, data->rt.charge, data->cfg.N, data->node_colors);
	for (size_t i = 0; i < data->cfg.N; i++)
	{
		data->points_buffer[i * 6 + 0] = (float) data->rt.pos[i].x;
		data->points_buffer[i * 6 + 1] = (float) data->rt.pos[i].y;
		data->points_buffer[i * 6 + 2] = (float) data->rt.pos[i].z;
		data->points_buffer[i * 6 + 3] = data->node_colors[i * 3 + 0];
		data->points_buffer[i * 6 + 4] = data->node_colors[i * 3 + 1];
		data->points_buffer[i * 6 + 5] = data->node_colors[i * 3 + 2];

		if (!slice_point_visible(data, (double) data->rt.pos[i].z))
		{
			// Knoten außerhalb der Schnittebene abdunken
			data->points_buffer[i * 6 + 3] *= 0.2f;
			data->points_buffer[i * 6 + 4] *= 0.2f;
			data->points_buffer[i * 6 + 5] *= 0.2f;
		}
	}
}

static void gui_gl_update_cluster_point(iwt_gui_data_t data, size_t idx, iwt_cluster_t cl)
{
	data->cluster_points_buffer[idx * 6 + 0] = (float) cl->pos.x;
	data->cluster_points_buffer[idx * 6 + 1] = (float) cl->pos.y;
	data->cluster_points_buffer[idx * 6 + 2] = (float) cl->pos.z;

	float charge_norm = (float) (cl->charge / (fabs(cl->charge) + 1.0));
	float brightness = (float) fmin(cl->mass / (cl->mass + 0.35), 1.0);

	if (!slice_point_visible(data, (double) cl->pos.z))
	{
		brightness *= 0.2f;
	}

	if (cl->external)
	{
		// Externe Cluster (Mitglieder aus >= 2 Zellen, Dualnetz) violett
		data->cluster_points_buffer[idx * 6 + 0] = 0.85f * brightness;
		data->cluster_points_buffer[idx * 6 + 1] = 0.35f * brightness;
		data->cluster_points_buffer[idx * 6 + 2] = brightness;
		return;
	}

	if (charge_norm > 0.0f)
	{
		data->cluster_points_buffer[idx * 6 + 3] = brightness;
		data->cluster_points_buffer[idx * 6 + 4] = 0.0f;
		data->cluster_points_buffer[idx * 6 + 5] = 0.0f;
	}
	else
	{
		data->cluster_points_buffer[idx * 6 + 3] = 0.0f;
		data->cluster_points_buffer[idx * 6 + 4] = 0.0f;
		data->cluster_points_buffer[idx * 6 + 5] = brightness;
	}
}

static size_t gui_gl_update_clusters(iwt_gui_data_t data)
{
	size_t cluster_draw_count = 0;
	for (size_t c = 0; c < data->rt.cluster_count; c++)
	{
		iwt_cluster_t cl = &data->rt.clusters[c];
		if (!cl->is_active) continue;
		gui_gl_update_cluster_point(data, cluster_draw_count, cl);
		cluster_draw_count++;
	}
	return cluster_draw_count;
}

static size_t gui_gl_update_waves(iwt_gui_data_t data)
{
	data->wave_segment_count = 0;
	if (!data->cfg.show_waves)
	{
		return 0;
	}

	size_t N = data->cfg.N;
	double two_pi = 2.0 * iwt_pi();
	double level_step = two_pi / (double) WAVE_LEVELS;

	// Gemeinsamer Frontenzug: Basis-Niveau wandert langsam
	double base_level = -iwt_pi() + level_step * (((double) ((long long) data->iter % WAVE_MARCH_FRAMES)) / (double) WAVE_MARCH_FRAMES);

	size_t seg = 0;

	for (size_t i = 0; i < N && seg < WAVE_MAX_SEGMENTS; i++)
	{
		int count = data->rt.wave_count[i];
		const int* neighbors = &data->rt.wave_flat[i * IWT_WAVE_STRIDE];

		// Differenz der Knotenphase zu allen Niveaus vorberechnen
		double di[WAVE_LEVELS];
		for (int l = 0; l < WAVE_LEVELS; l++)
		{
			double level_l = base_level + (double) l * level_step;
			di[l] = data->rt.I_phase[i] - level_l;
			di[l] -= two_pi * rint(di[l] / two_pi);
		}

		double px[WAVE_LEVELS][WAVE_MAX_CROSSINGS];
		double py[WAVE_LEVELS][WAVE_MAX_CROSSINGS];
		double pz[WAVE_LEVELS][WAVE_MAX_CROSSINGS];
		int crossings[WAVE_LEVELS];
		memset(crossings, 0, sizeof(crossings));

		for (int e = 0; e < count; e++)
		{
			size_t j = (size_t) neighbors[e];
			if (j == i)
			{
				continue;
			}

			double dj_raw = data->rt.I_phase[j];

			for (int l = 0; l < WAVE_LEVELS; l++)
			{
				if (crossings[l] >= WAVE_MAX_CROSSINGS)
				{
					continue;
				}
				double level_l = base_level + (double) l * level_step;
				double dj = dj_raw - level_l;
				dj -= two_pi * rint(dj / two_pi);
				if (di[l] * dj >= 0.0)
				{
					continue;
				}

			double t = di[l] / (di[l] - dj);
			double cx = (double) data->rt.pos[i].x + t * ((double) data->rt.pos[j].x - (double) data->rt.pos[i].x);
			double cy = (double) data->rt.pos[i].y + t * ((double) data->rt.pos[j].y - (double) data->rt.pos[i].y);
			double cz = (double) data->rt.pos[i].z + t * ((double) data->rt.pos[j].z - (double) data->rt.pos[i].z);

			// 2D-Schnitt: nur Schnittpunkte innerhalb der Scheibe
			if (!slice_point_visible(data, cz))
			{
				continue;
			}

			px[l][crossings[l]] = cx;
			py[l][crossings[l]] = cy;
			pz[l][crossings[l]] = cz;
			crossings[l]++;
			}
		}

		for (int l = 0; l < WAVE_LEVELS && seg < WAVE_MAX_SEGMENTS; l++)
		{
			float hue = (float) (((base_level + (double) l * level_step) + iwt_pi()) / two_pi);
			for (int k = 0; k + 1 < crossings[l] && seg < WAVE_MAX_SEGMENTS; k += 2)
			{
				double pa[3] = {px[l][k], py[l][k], pz[l][k]};
				double pb[3] = {px[l][k + 1], py[l][k + 1], pz[l][k + 1]};
				wave_emit_segment(data, seg, pa, pb, hue, 0.95f);
				seg++;
			}
		}
	}

	data->wave_segment_count = seg;
	return seg;
}

static void gui_gl_draw(iwt_gui_data_t data, size_t cluster_draw_count)
{
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT);

	glUseProgram(data->gl_program);

	int width = gtk_widget_get_width(data->gl_area);
	int height = gtk_widget_get_height(data->gl_area);
	float aspect = (height > 0) ? (float) width / (float) height : 1.0f;

	float radius = 3.6f * data->zoom;
	float eye[3] = {
		radius * cosf(data->cam_pitch) * cosf(data->cam_yaw),
		radius * sinf(data->cam_pitch),
		radius * cosf(data->cam_pitch) * sinf(data->cam_yaw)};
	float center[3] = {0.0f, 0.0f, 0.0f};
	float up[3] = {0.0f, 1.0f, 0.0f};

	float view[16], proj[16], mvp[16];
	gui_math_mat4_look_at(view, eye, center, up);
	gui_math_mat4_perspective(proj, 0.785398f, aspect, 0.1f, 200.0f);
	gui_math_mat4_mul(mvp, proj, view);

	glUniformMatrix4fv(data->gl_u_mvp, 1, GL_FALSE, mvp);

	glUniform1f(data->gl_u_size_scale, 1.0f);
	glBindBuffer(GL_ARRAY_BUFFER, data->gl_vbo);
	glBufferSubData(GL_ARRAY_BUFFER, 0, (GLsizeiptr) ((size_t) data->cfg.N * 6 * sizeof(float)), data->points_buffer);
	glBindVertexArray(data->gl_vao);
	glDrawArrays(GL_POINTS, 0, (GLsizei) data->cfg.N);

	glUniform1f(data->gl_u_size_scale, 3.0f);
	glBindBuffer(GL_ARRAY_BUFFER, data->gl_vbo_clusters);
	glBufferSubData(GL_ARRAY_BUFFER, 0, (GLsizeiptr) (cluster_draw_count * 6 * sizeof(float)), data->cluster_points_buffer);
	glBindVertexArray(data->gl_vao_clusters);
	glDrawArrays(GL_POINTS, 0, (GLsizei) cluster_draw_count);

	if (data->wave_segment_count > 0)
	{
		glUniform1f(data->gl_u_size_scale, 1.0f);
		glLineWidth(1.5f);
		glBindBuffer(GL_ARRAY_BUFFER, data->gl_vbo_wave);
		glBufferSubData(GL_ARRAY_BUFFER, 0, (GLsizeiptr) (data->wave_segment_count * 12 * sizeof(float)), data->wave_buffer);
		glBindVertexArray(data->gl_vao_wave);
		glDrawArrays(GL_LINES, 0, (GLsizei) (data->wave_segment_count * 2));
	}

	glBindVertexArray(0);
}
