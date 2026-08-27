/*
 * iwt_analysis.c - Analyse-Exporte der IWT-Simulation
 *
 * Stellt die Bruecke zwischen der laufenden Simulation und externen
 * Auswertewerkzeugen her:
 *  - iwt_analysis_export_csv: Zustandsexport als CSV-Satz unter
 *    experiments/exp_<Zeitstempel>/ (Spektrum, Cluster, Knoten)
 *  - iwt_analysis_capture_screenshot: Framebuffer-Grab als PNG unter shots/
 *
 * Beide Exporte dienen der reproduzierbaren Dokumentation von
 * Simulationsexperimenten (Histogramme, Projektionen, Spektralanalyse).
 * Die Ausgabeverzeichnisse liegen neben der Binaerdatei - analog zum
 * .cache-Muster aus main.c.
 *
 * Pfadaufbau ausschliesslich ueber die string-Bibliothek (string_copy/
 * string_cat mit Groessenargument): keine snprintf-Komposition, keine
 * Truncations-Warnungen, einheitliches Hausmuster.
 */

#include "iwt_analysis.h"

#include <epoxy/gl.h>
#include <gdk-pixbuf/gdk-pixbuf.h>
#include <locale.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <string/string.h>

/*
 * analysis_base_dir - Liefert das Verzeichnis der Binaerdatei
 *
 * Analog zum .cache-Muster in main.c werden Exporte relativ zur
 * Ausfuehrungsdatei abgelegt, damit sie unabhaengig vom Arbeitsverzeichnis
 * auffindbar bleiben.
 */
static void analysis_base_dir(char* out, size_t out_len)
{
	char exe_path[STRING_MAXLEN];
	string_get_exe_path(exe_path, sizeof(exe_path));

	char exe_path_copy[STRING_MAXLEN];
	string_copy(exe_path_copy, sizeof(exe_path_copy), exe_path);

	const char* base_dir = string_dirname_from_filepath(exe_path_copy);
	string_copy(out, out_len, base_dir ? base_dir : ".");
}

/*
 * analysis_timestamp - Erzeugt einen Zeitstempel fuer Datei-/Verzeichnisnamen
 */
static void analysis_timestamp(char* out, size_t out_len)
{
	time_t now = time(NULL);
	struct tm tm_v;
	localtime_r(&now, &tm_v);
	strftime(out, out_len, "%Y%m%d_%H%M%S", &tm_v);
}

/*
 * analysis_spiegeln - Kippt den Pixelblock vertikal (OpenGL-Ursprung unten)
 */
static void analysis_spiegeln(GLubyte* pixels, int width, int height,
	size_t row_bytes)
{
	GLubyte* zeile_tmp = malloc(row_bytes);
	for (int y = 0; y < height / 2; ++y)
	{
		GLubyte* oben = pixels + (size_t) y * row_bytes;
		GLubyte* unten = pixels + (size_t) (height - 1 - y) * row_bytes;
		memcpy(zeile_tmp, oben, row_bytes);
		memcpy(oben, unten, row_bytes);
		memcpy(unten, zeile_tmp, row_bytes);
	}
	free(zeile_tmp);
}

/*
 * analysis_max_helligkeit - Groesster Farbwert (Alpha ausgenommen) im
 * RGBA-Block; 0 bedeutet "leerer Framebuffer".
 */
static GLubyte analysis_max_helligkeit(const GLubyte* pixels, size_t anzahl)
{
	GLubyte m = 0;
	for (size_t i = 0; i < anzahl; i += 4)
	{
		if (pixels[i] > m)
		{
			m = pixels[i];
		}
		if (pixels[i + 1] > m)
		{
			m = pixels[i + 1];
		}
		if (pixels[i + 2] > m)
		{
			m = pixels[i + 2];
		}
	}
	return m;
}

void iwt_analysis_capture_screenshot(iwt_gui_data_t data)
{
	string_t base;
	string_t stamp;
	string_t file_path;

	analysis_base_dir(base, sizeof(base));

	/*
	 * Dimensionen: Der Viewport ist hier unzuverlaessig (GtkGLArea rendert
	 * intern in ein eigenes FBO und niemand im Code ruft je glViewport auf).
	 * Wir leiten die Groesse daher vom Widget ab (Breite x Scale-Faktor)
	 * und nutzen den Viewport nur als Fallback bzw. fuer die Diagnose.
	 */
	GLint viewport[4];
	glGetIntegerv(GL_VIEWPORT, viewport);

	const int scale = gtk_widget_get_scale_factor(data->gl_area);
	int width = gtk_widget_get_width(data->gl_area) * scale;
	int height = gtk_widget_get_height(data->gl_area) * scale;

	GLint read_fbo = 0;
	glGetIntegerv(GL_READ_FRAMEBUFFER_BINDING, &read_fbo);

	if (width < 1 || height < 1)
	{
		width = viewport[2];
		height = viewport[3];
	}

	if (width < 1 || height < 1)
	{
		fprintf(stderr, "analysis: Ungueltige Framebuffer-Dimensionen (%dx%d) "
			"- Screenshot uebersprungen\n", viewport[2], viewport[3]);
		return;
	}

	const size_t row_bytes = (size_t) width * 4;
	const size_t pixel_anzahl = row_bytes * (size_t) height;
	GLubyte* pixels = malloc(pixel_anzahl);
	if (!pixels)
	{
		fprintf(stderr, "analysis: Speicher fuer Screenshot fehlgeschlagen\n");
		return;
	}

	/*
	 * GLES-Kompatibilitaet + saubere Fehlerlage:
	 * Der Kontext kann ein GLES-Kontext sein - dort ist die erlaubte
	 * Format-/Typkombination beim Lesen beschraenkt (GL_INVALID_ENUM).
	 * Wichtig: glGetError arbeitet wie eine Queue - ohne vorheriges
	 * Leeren wuerde hier ein ALTFEHLER frueherer Aufrufe gemeldet statt
	 * des Status unseres Reads.
	 */
	unsigned int altfehler = 0;
	while (glGetError() != GL_NO_ERROR)
	{
		++altfehler;
	}

	glPixelStorei(GL_PACK_ALIGNMENT, 4);
	glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
	const GLenum fehler_nachher = glGetError();
	if (fehler_nachher != GL_NO_ERROR)
	{
		fprintf(stderr, "analysis: glReadPixels fehlgeschlagen (err=%u, "
			"%u Altfehler verworfen) - Screenshot uebersprungen\n",
			(unsigned) fehler_nachher, altfehler);
		free(pixels);
		return;
	}
	if (altfehler > 0)
	{
		printf("analysis: %u vorbestehende GL-Fehler verworfen "
			"(Queueschmutz aus frueheren Aufrufen)\n", altfehler);
	}

	for (size_t i = 0; i < pixel_anzahl; i += 4)
	{
		pixels[i + 3] = 255;
	}

	const GLubyte hell = analysis_max_helligkeit(pixels, pixel_anzahl);

	analysis_spiegeln(pixels, width, height, row_bytes);

	GdkPixbuf* pixbuf = gdk_pixbuf_new_from_data(pixels, GDK_COLORSPACE_RGB, TRUE,
		8, width, height, (int) row_bytes, NULL, NULL);
	if (!pixbuf)
	{
		free(pixels);
		fprintf(stderr, "analysis: Pixbuf fehlgeschlagen\n");
		return;
	}

	// Zielverzeichnis anlegen und Pfad zusammensetzen
	const char* shots_dir = string_append_directory_to_path_and_create(base, "shots");
	if (!shots_dir)
	{
		g_object_unref(pixbuf);
		free(pixels);
		return;
	}

	analysis_timestamp(stamp, sizeof(stamp));
	string_copy(file_path, sizeof(file_path), shots_dir);
	string_cat(file_path, sizeof(file_path), "/shot_");
	string_cat(file_path, sizeof(file_path), stamp);
	string_cat(file_path, sizeof(file_path), ".png");

	GError* error = NULL;
	gboolean ok = gdk_pixbuf_save(pixbuf, file_path, "png", &error, NULL);
	if (!ok)
	{
		fprintf(stderr, "analysis: PNG fehlgeschlagen: %s\n",
			error ? error->message : "?");
		if (error)
		{
			g_error_free(error);
		}
	}
	else
	{
		printf("analysis: Screenshot gespeichert: %s (%dx%d, max=%u, "
			"vp=%dx%d, fbo=%u, err=%u)\n", file_path, width, height,
			(unsigned) hell, viewport[2], viewport[3],
			(unsigned) read_fbo, (unsigned) fehler_nachher);
	}

	g_object_unref(pixbuf);
	free(pixels);
}

void iwt_analysis_export_csv(iwt_gui_data_t data)
{
	/* LC_NUMERIC auf C setzen, damit %.9e immer Punkt als Dezimaltrenner
	 * liefert (ansonsten waere auf de_DE das Komma waehle, was Pandas
	 * als Feldtrenner interpretieren wuerde). */
	const char* alt_locale = setlocale(LC_NUMERIC, "C");

	string_t base;
	string_t stamp;
	string_t dir_path;
	string_t file_path;

	analysis_base_dir(base, sizeof(base));
	analysis_timestamp(stamp, sizeof(stamp));

	const char* exp_root = string_append_directory_to_path_and_create(
		base, "experiments");
	if (!exp_root)
	{
		return;
	}

	string_copy(dir_path, sizeof(dir_path), exp_root);
	string_cat(dir_path, sizeof(dir_path), "/exp_");
	string_cat(dir_path, sizeof(dir_path), stamp);
	string_directory_create(dir_path);

	const iwt_runtime_t rt = &data->rt;
	const iwt_config_t cfg = &data->cfg;

	// --- spectrum.csv ---------------------------------------------------
	string_copy(file_path, sizeof(file_path), dir_path);
	string_cat(file_path, sizeof(file_path), "/spectrum.csv");
	FILE* f = fopen(file_path, "w");
	if (!f)
	{
		perror(file_path);
		return;
	}

	fprintf(f, "step;vacuum;electron;proton;u_quark;d_quark;other;total\n");
	for (size_t i = 0; i < rt->spectrum_history_count; ++i)
	{
		const iwt_spectrum_t* s = &rt->spectrum_history[i];
		size_t total = s->count_proton + s->count_electron + s->count_other;
		fprintf(f, "%llu;%zu;%zu;%zu;%zu;%zu;%zu;%zu\n",
			(unsigned long long) s->step, s->count_vacuum,
			s->count_electron, s->count_proton,
			s->count_u_quark, s->count_d_quark,
			s->count_other, total);
	}
	// Aktuellen Stand noch anhaengen falls er nicht im Puffer liegt
	if (rt->spectrum_history_count == 0 ||
		rt->spectrum_history[rt->spectrum_history_count - 1].step != rt->n_steps)
	{
		size_t total = rt->spectrum.count_proton + rt->spectrum.count_electron
					+ rt->spectrum.count_other;
		fprintf(f, "%llu;%zu;%zu;%zu;%zu;%zu;%zu;%zu\n",
			(unsigned long long) rt->n_steps, rt->spectrum.count_vacuum,
			rt->spectrum.count_electron, rt->spectrum.count_proton,
			rt->spectrum.count_u_quark, rt->spectrum.count_d_quark,
			rt->spectrum.count_other, total);
	}
	fclose(f);

	// --- clusters.csv ---------------------------------------------------
	string_copy(file_path, sizeof(file_path), dir_path);
	string_cat(file_path, sizeof(file_path), "/clusters.csv");
	f = fopen(file_path, "w");
	if (!f)
	{
		perror(file_path);
		return;
	}

	fprintf(f, "id;type;node_count;mass;charge;phase;x;y;z;vx;vy;vz\n");
	for (uint32_t i = 0; i < rt->cluster_count; ++i)
	{
		const struct iwt_cluster* c = &rt->clusters[i];
		fprintf(f, "%d;%d;%zu;%.9e;%.9e;%.9e;%.9e;%.9e;%.9e;%.9e;%.9e;%.9e\n",
			c->id, c->type, c->node_count, c->mass, c->charge, c->phase,
			(double) c->pos.x, (double) c->pos.y, (double) c->pos.z,
			(double) c->vel.x, (double) c->vel.y, (double) c->vel.z);
	}
	fclose(f);

	// --- nodes.csv ------------------------------------------------------
	string_copy(file_path, sizeof(file_path), dir_path);
	string_cat(file_path, sizeof(file_path), "/nodes.csv");
	f = fopen(file_path, "w");
	if (!f)
	{
		perror(file_path);
		return;
	}

	fprintf(f, "idx;x;y;z;i_real;i_imag;rho;mass;charge\n");
	for (size_t i = 0; i < cfg->N; ++i)
	{
		const double rho = rt->I_real[i] * rt->I_real[i] + rt->I_imag[i] * rt->I_imag[i];
		fprintf(f, "%zu;%.9e;%.9e;%.9e;%.9e;%.9e;%.9e;%.9e;%.9e\n", i,
			(double) rt->pos[i].x, (double) rt->pos[i].y,
			(double) rt->pos[i].z, rt->I_real[i], rt->I_imag[i], rho,
			rt->mass[i], rt->charge[i]);
	}
	fclose(f);

	printf("analysis: Export nach %s (%u Cluster, %zu Knoten, Schritt %llu)\n",
		dir_path, rt->cluster_count, cfg->N, (unsigned long long) rt->n_steps);

	if (alt_locale)
	{
		setlocale(LC_NUMERIC, alt_locale);
	}
}

void iwt_analysis_generate_histogram(iwt_gui_data_t data)
{
	string_t base;
	string_t python_path;
	string_t script_path;
	string_t cmd;

	analysis_base_dir(base, sizeof(base));
	string_copy(python_path, sizeof(python_path), base);
	string_cat(python_path, sizeof(python_path), "/src/python/analyse.py");

	string_copy(script_path, sizeof(script_path), base);
	string_cat(script_path, sizeof(script_path), "/build/bin/experiments/");

	// Letztes Experiment finden (neuestes)
	string_t find_cmd;
	string_copy(find_cmd, sizeof(find_cmd), "ls -td ");
	string_cat(find_cmd, sizeof(find_cmd), script_path);
	string_cat(find_cmd, sizeof(find_cmd), "/exp_* 2>/dev/null | head -1");
	FILE* f = popen(find_cmd, "r");
	if (!f)
	{
		return;
	}

	char latest_exp[512];
	if (fgets(latest_exp, sizeof(latest_exp), f))
	{
		// Newline entfernen
		size_t len = strlen(latest_exp);
		if (len > 0 && latest_exp[len - 1] == '\n')
		{
			latest_exp[len - 1] = '\0';
		}

		string_copy(cmd, sizeof(cmd), "python3 \"");
		string_cat(cmd, sizeof(cmd), python_path);
		string_cat(cmd, sizeof(cmd), "\" \"");
		string_cat(cmd, sizeof(cmd), latest_exp);
		string_cat(cmd, sizeof(cmd), "\" --histogram &");
		printf("analysis: Erzeuge Histogramm fuer %s\n", latest_exp);
		system(cmd);
	}

	pclose(f);
}
