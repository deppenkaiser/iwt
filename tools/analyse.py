#!/usr/bin/env python3
# ============================================================================
# analyse.py - Auswertewerkzeug fuer IWT-Simulationsexporte
# ============================================================================
#
# Liest den CSV-Export der Simulation (Taste 'E' in der GUI, siehe
# src/iwt_analysis.c) und erzeugt daraus drei Artefakte:
#
#   1. cluster_histogramm.png
#      Verteilung der Clustergroessen (Knotenanzahl je Cluster).
#      Die Theorie sagt stabile Strukturen bei ca. 1500 Knoten (Elektron)
#      und ca. 150 Knoten (Proton) voraus - beide werden als Referenzlinie
#      eingezeichnet, sofern sie im Wertebereich liegen.
#
#   2. projektion_<achse>.png
#      Gewichtetes 2D-Projektionsbild: Alle 3D-Knoten werden auf eine Ebene
#      projiziert, wobei entlang der Projektionsachse die Informationsdichte
#      rho = |I|^2 akkumuliert wird ("Roentgenaufnahme" des Raums).
#      Grundlage fuer den Vergleich mit CMB-aehnlichen Strukturen.
#
#   3. spektrum_<achse>.png  (optional, Flag --spektrum)
#      Radial gemitteltes Leistungsspektrum (2D-FFT) der Projektion,
#      doppelt-logarithmisch. Dient dem spaeteren quantitativen Vergleich
#      mit der Winkel-Leistungsdichte des CMB.
#
# Aufruf:
#   python3 tools/analyse.py                          # neuester Export
#   python3 tools/analyse.py bin/experiments/exp_XY   # bestimmter Export
#   python3 tools/analyse.py --achse x --aufloesung 512 --spektrum
#
# Ausgaben landen standardmaessig im Exportverzeichnis selbst.
# ============================================================================

import argparse
import glob
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")  # Headless: kein Fenster, nur Dateiausgabe
import matplotlib.pyplot as plt

# Von der Theorie vorhergesagte Clustergroessen (Knotenanzahl):
# siehe README/Wiki - Elektron ~ 1500 Knoten, Proton ~ 150 Knoten.
ERWARTET = {
    "Elektron (~1500)": 1500,
    "Proton (~150)": 150,
}


def finde_neuesten_export(basis):
	"""Sucht unter <basis>/experiments/exp_* das zeitlich neueste Verzeichnis.

	Fallback: Falls dort nichts liegt, wird auch direkt unter <basis> gesucht
	(erlaubt den Aufruf mit einem bereits exportierten Verzeichnis als Argument).
	"""
	kandidaten = sorted(glob.glob(os.path.join(basis, "experiments", "exp_*")))
	if not kandidaten:
		kandidaten = sorted(glob.glob(os.path.join(basis, "exp_*")))
	if not kandidaten:
		sys.exit(f"Fehler: Kein Export unter {basis} gefunden "
			 f"(erst in der Simulation 'E' druecken).")
	return kandidaten[-1]


def lade_csv(pfad):
	"""Laedt eine Export-CSV als strukturiertes NumPy-Array (mit Spaltennamen)."""
	if not os.path.isfile(pfad):
		sys.exit(f"Fehler: Datei fehlt: {pfad}")
	return np.genfromtxt(pfad, delimiter=",", names=True)


def erstelle_histogramm(clusters, ausgabe_pfad):
	"""Zeichnet die Verteilung der Clustergroessen und markiert die Erwartung."""
	groessen = clusters["node_count"].astype(int)
	groessen = groessen[groessen > 0]
	if groessen.size == 0:
		print("Warnung: Keine Cluster im Export - Histogramm uebersprungen.")
		return None

	fig, ax = plt.subplots(figsize=(10, 6))
	bins = min(80, max(20, int(np.sqrt(groessen.size)) * 2))
	ax.hist(groessen, bins=bins, color="#3b6ea5", edgecolor="black", linewidth=0.4)

	# Theoretisch erwartete Strukturen als gestrichelte Referenzlinien
	for label, wert in ERWARTET.items():
		if groessen.min() <= wert <= groessen.max():
			ax.axvline(wert, color="red", linestyle="--", linewidth=1.2)
			ax.text(wert, ax.get_ylim()[1] * 0.92, f" {label}",
					color="red", fontsize=9, va="top")

	ax.set_xlabel("Knotenanzahl je Cluster")
	ax.set_ylabel("Anzahl Cluster")
	ax.set_yscale("log")
	ax.set_title(f"Cluster-Groessenverteilung (N = {groessen.size})")
	ax.grid(alpha=0.3)
	fig.tight_layout()
	fig.savefig(ausgabe_pfad, dpi=150)
	plt.close(fig)

	# Kennzahlen auf der Konsole: die am staerksten besetzten Groessenklassen
	hist, kanten = np.histogram(groessen, bins=bins)
	reihenfolge = np.argsort(hist)[::-1][:5]
	print("Staerkste Groessenklassen (Zentrum -> Anzahl):")
	for idx in reihenfolge:
		if hist[idx] > 0:
			zentrum = 0.5 * (kanten[idx] + kanten[idx + 1])
			print(f"  ~{zentrum:9.1f} Knoten : {hist[idx]:5d} Cluster")
	return ausgabe_pfad


def erstelle_projektion(nodes, achse, aufloesung, ausgabe_pfad):
	"""Gewichtete 2D-Projektion: Akkumulation von rho = |I|^2 entlang `achse`.

	Die beiden verbleibenden Achsen spannen die Bildebene auf. Die Gewichtung
	mit der Informationsdichte entspricht der physikalischen Deutung: Nur
	Abweichungen vom leeren Raum sind sichtbar (relationale Metrik, Axiom 3).
	"""
	# Bildachsen je Projektionsachse: entlang 'z' wird z. B. auf (x, y) projiziert
	ebenen = {"x": ("y", "z"), "y": ("x", "z"), "z": ("x", "y")}
	hx, hy = ebenen[achse]

	x = nodes[hx].astype(float)
	y = nodes[hy].astype(float)
	rho = nodes["rho"]

	grid, x_kanten, y_kanten = np.histogram2d(
		x, y,
		bins=aufloesung,
		weights=rho,
		range=[[x.min(), x.max()], [y.min(), y.max()]],
	)

	fig, ax = plt.subplots(figsize=(8, 8))
	im = ax.imshow(
		grid.T,
		origin="lower",
		extent=[x_kanten[0], x_kanten[-1], y_kanten[0], y_kanten[-1]],
		cmap="inferno",
		interpolation="nearest",
	)
	ax.set_xlabel(hx)
	ax.set_ylabel(hy)
	ax.set_title(f"Gewichtete Projektion entlang {achse} "
			 f"(rho = |I|^2, N = {rho.size})")
	fig.colorbar(im, ax=ax, label="akkumulierte Dichte")
	fig.tight_layout()
	fig.savefig(ausgabe_pfad, dpi=150)
	plt.close(fig)

	# Grid fuer das Leistungsspektrum zurueckgeben
	return grid


def erstelle_leistungsspektrum(grid, ausgabe_pfad):
	"""Radial gemitteltes 2D-Leistungsspektrum (FFT) der Projektion.

	Vorgehen: 2D-FFT, Betragsquadrat, dann Mittelung ueber konzentrische
	Ringe (Wellenzahl k). Das Ergebnis ist mit dem Winkel-Leistungsspektrum
	des CMB vergleichbar, wenn Skalenachse und Normierung angepasst werden.
	"""
	feld = grid - grid.mean()  # Gleichanteil entfernen (sonst dominiert k=0)
	spektrum = np.abs(np.fft.fftshift(np.fft.fft2(feld))) ** 2

	nz, nx = spektrum.shape
	ky = np.fft.fftshift(np.fft.fftfreq(nz))[:, None]
	kx = np.fft.fftshift(np.fft.fftfreq(nx))[None, :]
	k = np.sqrt(kx ** 2 + ky ** 2)

	# Radiale Binning: Mittelung des Spektrums ueber Ring gleicher Wellenzahl
	k_max = k.max()
	anzahl_bins = min(nz, nx) // 2
	k_grenzen = np.linspace(0, k_max, anzahl_bins + 1)
	k_mitte = 0.5 * (k_grenzen[:-1] + k_grenzen[1:])
	indices = np.digitize(k.ravel(), k_grenzen) - 1

	p_mittel = np.array([
		spektrum.ravel()[indices == b].mean() if np.any(indices == b) else np.nan
		for b in range(anzahl_bins)
	])

	fig, ax = plt.subplots(figsize=(8, 6))
	ax.loglog(k_mitte[1:], p_mittel[1:], marker=".", linestyle="-")
	ax.set_xlabel("Wellenzahl k [1/Einheit]")
	ax.set_ylabel("Leistung P(k)")
	ax.set_title("Radiales Leistungsspektrum der Projektion")
	ax.grid(alpha=0.3, which="both")
	fig.tight_layout()
	fig.savefig(ausgabe_pfad, dpi=150)
	plt.close(fig)


def main():
	parser = argparse.ArgumentParser(
		description="Auswertung eines IWT-Simulationsexports (siehe oben).")
	parser.add_argument("verzeichnis", nargs="?", default=None,
						help="Exportverzeichnis (Standard: neuestes unter bin/experiments)")
	parser.add_argument("--basis", default=os.path.join(os.path.dirname(__file__), "..", "bin"),
						help="Basisverzeichnis mit experiments/ (Standard: bin/)")
	parser.add_argument("--achse", default="z", choices=["x", "y", "z"],
						help="Projektionsachse (Default: z)")
	parser.add_argument("--aufloesung", type=int, default=256,
						help="Bildbreite der Projektion in Pixel/Kacheln (Default: 256)")
	parser.add_argument("--spektrum", action="store_true",
						help="Zusaetzlich radiales Leistungsspektrum erzeugen")
	args = parser.parse_args()

	ziel = args.verzeichnis or finde_neuesten_export(args.basis)
	if not os.path.isdir(ziel):
		sys.exit(f"Fehler: Verzeichnis nicht gefunden: {ziel}")
	print(f"Auswerte Export: {ziel}")

	nodes_pfad = os.path.join(ziel, "nodes.csv")
	clusters_pfad = os.path.join(ziel, "clusters.csv")

	nodes = lade_csv(nodes_pfad)
	clusters = lade_csv(clusters_pfad)

	histogramm = erstelle_histogramm(clusters, os.path.join(ziel, "cluster_histogramm.png"))
	if histogramm:
		print(f"Erzeugt: {histogramm}")

	grid = erstelle_projektion(
		nodes, args.achse, args.aufloesung,
		os.path.join(ziel, f"projektion_{args.achse}.png"))
	print(f"Erzeugt: {os.path.join(ziel, f'projektion_{args.achse}.png')}")

	if args.spektrum:
		spektrum_pfad = os.path.join(ziel, f"spektrum_{args.achse}.png")
		erstelle_leistungsspektrum(grid, spektrum_pfad)
		print(f"Erzeugt: {spektrum_pfad}")


if __name__ == "__main__":
	main()
