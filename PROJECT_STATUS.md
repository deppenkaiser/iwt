# Projektstatus: iwt – GPU-Performance & Optimierungsstand
## Übergabe an die nächste KI (erstellt: 2026-08-27)

> Diese Datei ist die **verlässliche Projektbeschreibung** für die nächste KI-Session.
> Sie ersetzt nicht die README.md (Theorie/Numerik), sondern dokumentiert den
> **Arbeits- und Optimierungsstand** der GPU-Implementierung.

---

## 1. Zweck des Projekts

IWT = **Informations-Weber-Theorie** (Autor: Michael Czybor).
Diskretes, **global gekoppeltes** Punktmodell zur Emergenz von Raum, Masse,
Ladung und Quantenpotential. OpenCL-beschleunigt, GTK4-Rendering.

Zwei getrennte Git-Repos (getrennt committen!):
- **App:** `/home/czymic/Dokumente/daten/sandbox/apps/iwt`
- **OpenCL-Bibliothek:** `/home/czymic/Dokumente/daten/sandbox/libraries/ocl`

---

## 2. Aktuelle Performance-Messung (Commit `dee02a6`)
Gemessen auf NVIDIA GeForce RTX 5060 Ti (OpenCL 3.0 CUDA 13.0.100), N=16384.
Pro Frame (~133 ms Simulation → **~8 FPS**) in Millisekunden:

```
flux         = ~66 ms   ← O(N²), ENGPASS 1
q            = ~35 ms   ← O(N²), ENGPASS 2
update_info  = ~22 ms   ← O(N²), ENGPASS 3
clusters     = ~9 ms
mass/charge  = ~0.2 ms
wellen       = ~7 ms (Draw gesamt ~3-10 ms)
Sim gesamt   = ~133 ms => ~8 FPS
```

Der Engpass liegt **allein in den drei O(N²)-Kernels** (flux, q, update_info)
= ~95 % der Framezeit. Wellen und Fluktuationen sind bereits auf GPU (billig).

---

## 3. Grundlegender Befund (WICHTIG – nicht wiederholen)

**Die O(N²)-Kernels sind SPEICHERBANDWERTEN-BEGRENZT, nicht rechnen-begrenzt.**

- K-Matrix ist **voll besetzt**: fast alle 16384² Paare sind gekoppelt
  (K = 1/d^(3-D), nicht-lokal; kein Sparsity-Gewinn möglich).
- K-Matrix = 16384² × 8 Byte = **2 GB** (statisch, einmalig hochgeladen).
- **2D-Parallel-Reduktion wurde getestet und wieder zurückgerollt**: brachte
  flux nur 64→57 ms, weil die Kern-Parallelität NICHT der Engpass ist –
  die dominierende Speicherlast (K-Matrix lesen) bleibt unverändert.
  → Neuer Kernel `iwt_flux_stage1`/`iwt_flux_reduce` ist aus der ocl-Bibliothek
  entfernt, `IWT_FLUX_TILE` und `flux_partial_gpu`-Buffer entfernt.

**Konsequenz:** ~8 FPS bei N=16384 ist **nahe dem Speicherbandbreiten-Limit**
der RTX 5060 Ti. Es gibt keinen Weg zu deutlich mehr FPS ohne Modelländerung.

---

## 4. Bereits abgeschlossene Optimierungen (alle committet)

| Commit (App) | Inhalt |
|--------------|--------|
| `3e83b5c` | Wellen auf GPU (SIGFPE-Fix): Zwei-Pass `iwt_wave_count_points` (2D-Range, keine Atomics) + `iwt_wave_emit` mit CPU-Prefix-Sum. Behebt NVIDIA-Treiber-Crash. |
| `4d34245` | Wellen-GPU-Berechnung alle 2 Frames (halbiert synchrone PCIe-Transfers). |
| `63351dc` | Fluktuationen auf GPU: `frozen_generate_uncertainty_cpu` = dünner Wrapper (upload I, Kernel `iwt_fluctuations`, readback xi). Box-Muller+Xorshift64, identische Statistik. `box_muller2`/gestrichen. |
| `c4758e7` | `iwt_analysis.c`: Shell-Kommandos über `string_copy`/`string_cat` (string-Bibliothek) statt `snprintf` → behebt sämtliche `-Wformat-truncation`-Warnungen. |
| `dee02a6` | Profiling-Instrumentierung: `[kprof]` je Schritt, `[prof]` je Render-Phase. |

| Commit (ocl-Lib) | Inhalt |
|------------------|--------|
| `54018ae` | Kernel `iwt_fluctuations` (Xorshift64+Box-Muller) mit OpenCL-nativen `ulong`/`UL`. **Wichtig:** C99-`ull`-Literale werden von OpenCL C **nicht** unterstützt → erzeugen NaN. Immer `ulong` + `UL`-Suffix verwenden. |
| `456a89a` | Wellen-Kernel aufgeteilt in `iwt_wave_count_points` + `iwt_wave_emit`. |

---

## 5. Verbleibende Probleme / Herausforderungen

1. **Nur ~8 FPS** bei N=16384 (physikalisches Bandbreiten-Limit, s. Abschnitt 3).
2. **Kein Sparsity-Potenzial:** Komprimierte Nachbarschaftsliste (`IWT_ADJ_STRIDE`=64)
   erfasst nur ~64/5375 Kopplungen → unbrauchbar ohne Physik-Bruch.
   `true_avg=5375` (K > cluster_threshold), `avg_deg=64` (Liste gesättigt).

---

## 6. Optionen, falls mehr FPS gewünscht (Entscheidung offen)

Falls der Autor mehr FPS will, sind die realistischen Wege (mit Physik-Kompromiss):

1. **N verkleinern** (z.B. N=8192 → ~4× schneller, da O(N²) → ~30 FPS).
   Physik treu, aber gröbere Auflösung.
2. **Kopplung ausdünnen** (nur starke Kopplungen behalten) → **ändert das Bild/Physik**
   (bricht Nichtlokalität, Kap. 12). Vom Autor bisher abgelehnt (Physik muss treu bleiben).
3. **8 FPS bei N=16384 akzeptieren** (aktueller Stand, stabil und korrekt).

Bisher hat der Autor wiederholt "die Physik identisch lassen" betont → Option 2
ist nur mit ausdrücklicher Freigabe zulässig.

---

## 7. Konventionen & Regeln (STRENG einhalten)

- **Code-Kommentare auf DEUTSCH** (kein Englisch). Der Autor ist Österreicher
  (Dipl.-Ing. FH), WDBT+-Urheber. Er richtet sich explizit: *"Ich möchte keine
  englischen Texte im Code."*
- Physik **niemals ändern** ohne ausdrückliche Freigabe (Standard = identische
  Statistik/Semantik).
- Warnungen gelten streng (s. Abschnitt 4: `-Wformat-truncation` war Blockade).
- Build: `cmake --build build -j4` in App- *und* ocl-Repo (Bibliothek zuerst).
- **Getrennte Commits** für die zwei Repos.
- Diagnose-/Messcode vor Produktiv-Commit entfernen, außer `[kprof]`/`[prof]`
  (bei `dee02a6` bewusst beibehalten).

---

## 8. Nützliche Messtechnik

- Profiling geht nach stderr: `iwt 2>&1 | grep -E "kprof|prof"`.
- `[kprof] *` = je Simulationsschritt (alle 30 Frames bei Profil-Logging).
- `[prof] sim=...` = Gesamt-Simzeit je Frame → 1000/`sim` = FPS.
- Wellen-Neuberechnung nur alle 2 Frames → `iter & 1` in `gui_gl_update_waves`.
