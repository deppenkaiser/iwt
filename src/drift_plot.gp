set terminal pngcairo size 1200,600 enhanced font "Arial,12"
set output "/home/czymic/Dokumente/daten/sandbox/apps/iwt/drift_final.png"

set datafile separator comma
set multiplot layout 1,2

# Linkes Diagramm: Alle 100 Knoten (mit Farben)
set title "(a) Alle 100 Knoten"
set xlabel "Knoten-Index"
set ylabel "Drift pro Schritt"
set grid
set boxwidth 0.8
set style fill solid 0.5
set xrange [-2:102]

# Farbige Balken: rot für negativ, grün für positiv
plot "/home/czymic/Dokumente/daten/sandbox/apps/iwt/build/bin/drift.csv" using 1:($4 < 0 ? $4 : 0):(sprintf("%.0f", $1)) with boxes lc rgb "red" notitle, \
     "" using 1:($4 >= 0 ? $4 : 0) with boxes lc rgb "green" notitle

# Rechtes Diagramm: Erste 10 Knoten (Detail)
set title "(b) Erste 10 Knoten (Detail)"
set xlabel "Knoten-Index"
set ylabel "Drift pro Schritt"
set grid
set boxwidth 0.6
set style fill solid 0.5
set xrange [-1:10]

plot "/home/czymic/Dokumente/daten/sandbox/apps/iwt/build/bin/drift.csv" using 1:($4 < 0 ? $4 : 0) every ::0::9 with boxes lc rgb "red" notitle, \
     "" using 1:($4 >= 0 ? $4 : 0) every ::0::9 with boxes lc rgb "green" notitle

unset multiplot