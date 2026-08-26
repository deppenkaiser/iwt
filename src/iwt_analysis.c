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
 */

#include "iwt_analysis.h"

#include <errno.h>
#include <epoxy/gl.h>
#include <gdk-pixbuf/gdk-pixbuf.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
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
	snprintf(out, out_len, "%s", base_dir ? base_dir : ".");
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
 * analysis_ensure_dir - Legt ein Verzeichnis an, existiert es bereits wird
 * das toleriert.
 */
static bool analysis_ensure_dir(const char* path)
{
	if (mkdir(path, 0755) == 0 || errno == EEXIST)
	{
		return true;
	}

	perror(path);
	return false;
}

void iwt_analysis_capture_screenshot(iwt_gui_data_t data)
{
	char base[STRING_MAXLEN];
	char stamp[32];
	char dir_path[STRING_MAXLEN * 2];
	char file_path[STRING_MAXLEN * 2];

	analysis_base_dir(base, sizeof(base));

	GLint viewport[4];	glGetIntegerv(GL_VIEWPORT, viewport);

	const int width = viewport[2];
	const int height = viewport[3];

	GLubyte* pixels = malloc((size_t) width * (size_t) height * 3);
	if (!pixels)
	{
		fprintf(stderr, "analysis: Speicher fuer Screenshot fehlgeschlagen\n");
		return;
	}

	glPixelStorei(GL_PACK_ALIGNMENT, 1);
	glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, pixels);

	// Vertikal spiegeln - OpenGL legt den Ursprung unten links
	const size_t row_bytes = (size_t) width * 3;
	GLubyte* row_tmp = malloc(row_bytes);
	for (int y = 0; y < height / 2; ++y)
	{
		GLubyte* top = pixels + (size_t) y * row_bytes;
		GLubyte* bottom = pixels + (size_t) (height - 1 - y) * row_bytes;
		memcpy(row_tmp, top, row_bytes);
		memcpy(top, bottom, row_bytes);
		memcpy(bottom, row_tmp, row_bytes);
	}
	free(row_tmp);

	GdkPixbuf* pixbuf = gdk_pixbuf_new_from_data(pixels, GDK_COLORSPACE_RGB, FALSE,
		8, width, height, (int) row_bytes, NULL, NULL);
	if (!pixbuf)
	{
		free(pixels);
		fprintf(stderr, "analysis: Pixbuf fehlgeschlagen\n");
		return;
	}

	analysis_timestamp(stamp, sizeof(stamp));
	snprintf(dir_path, sizeof(dir_path), "%s/shots", base);
	if (!analysis_ensure_dir(dir_path))
	{
		g_object_unref(pixbuf);
		free(pixels);
		return;
	}

	snprintf(file_path, sizeof(file_path), "%s/shot_%s.png", dir_path, stamp);
	GError* error = NULL;
	if (!gdk_pixbuf_save(pixbuf, file_path, "png", &error, NULL))
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
		printf("analysis: Screenshot gespeichert: %s (%dx%d)\n", file_path,
			width, height);
	}

	g_object_unref(pixbuf);
	free(pixels);
}

void iwt_analysis_export_csv(iwt_gui_data_t data)
{
	char base[STRING_MAXLEN];
	char stamp[32];
	char dir_path[STRING_MAXLEN * 2];
	char file_path[STRING_MAXLEN * 2];

	analysis_base_dir(base, sizeof(base));

	analysis_timestamp(stamp, sizeof(stamp));
	snprintf(dir_path, sizeof(dir_path), "%s/experiments/exp_%s", base, stamp);
	if (!analysis_ensure_dir(dir_path))
	{
		return;
	}

	const iwt_runtime_t rt = &data->rt;
	const iwt_config_t cfg = &data->cfg;

	// --- spectrum.csv ---------------------------------------------------
	snprintf(file_path, sizeof(file_path), "%s/spectrum.csv", dir_path);
	FILE* f = fopen(file_path, "w");
	if (!f)
	{
		perror(file_path);
		return;
	}

	fprintf(f, "step,vacuum,electron,proton,u_quark,d_quark,other,total\n");
	fprintf(f, "%llu,%zu,%zu,%zu,%zu,%zu,%zu,%u\n",
		(unsigned long long) rt->n_steps, rt->spectrum.count_vacuum,
		rt->spectrum.count_electron, rt->spectrum.count_proton,
		rt->spectrum.count_u_quark, rt->spectrum.count_d_quark,
		rt->spectrum.count_other, rt->cluster_count);
	fclose(f);

	// --- clusters.csv ---------------------------------------------------
	snprintf(file_path, sizeof(file_path), "%s/clusters.csv", dir_path);
	f = fopen(file_path, "w");
	if (!f)
	{
		perror(file_path);
		return;
	}

	fprintf(f, "id,type,node_count,mass,charge,phase,x,y,z,vx,vy,vz\n");
	for (uint32_t i = 0; i < rt->cluster_count; ++i)
	{
		const struct iwt_cluster* c = &rt->clusters[i];
		fprintf(f, "%d,%d,%zu,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e\n",
			c->id, c->type, c->node_count, c->mass, c->charge, c->phase,
			(double) c->pos.x, (double) c->pos.y, (double) c->pos.z,
			(double) c->vel.x, (double) c->vel.y, (double) c->vel.z);
	}
	fclose(f);

	// --- nodes.csv ------------------------------------------------------
	snprintf(file_path, sizeof(file_path), "%s/nodes.csv", dir_path);
	f = fopen(file_path, "w");
	if (!f)
	{
		perror(file_path);
		return;
	}

	fprintf(f, "idx,x,y,z,i_real,i_imag,rho,mass,charge\n");
	for (size_t i = 0; i < cfg->N; ++i)
	{
		const double rho = rt->I_real[i] * rt->I_real[i] + rt->I_imag[i] * rt->I_imag[i];
		fprintf(f, "%zu,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e\n", i,
			(double) rt->pos[i].x, (double) rt->pos[i].y,
			(double) rt->pos[i].z, rt->I_real[i], rt->I_imag[i], rho,
			rt->mass[i], rt->charge[i]);
	}
	fclose(f);

	printf("analysis: Export nach %s (%u Cluster, %zu Knoten, Schritt %llu)\n",
		dir_path, rt->cluster_count, cfg->N, (unsigned long long) rt->n_steps);
}
