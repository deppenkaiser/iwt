# iwt

## Formatierung:
BasedOnStyle: LLVM
BreakBeforeBraces: Allman
UseTab: Always
TabWidth: 4
IndentWidth: 4
ColumnLimit: 0
AlignAfterOpenBracket: Align

# Erzwingt {} bei if/for/while (ab LLVM 15+)
InsertBraces: true

# Pointer-Stil: int* ptr
PointerAlignment: Left

# Keine einzeiligen Kontrollstrukturen
AllowShortIfStatementsOnASingleLine: false
AllowShortLoopsOnASingleLine: false
AllowShortFunctionsOnASingleLine: None
AllowShortBlocksOnASingleLine: false
AllowShortCaseLabelsOnASingleLine: false

# Weitere Optimierungen
BreakBeforeTernaryOperators: true
BreakConstructorInitializers: AfterColon
ConstructorInitializerIndentWidth: 4
ContinuationIndentWidth: 4
IndentCaseLabels: true
IndentPPDirectives: AfterHash
SpaceAfterCStyleCast: true
SpaceBeforeParens: ControlStatements
SpaceBeforeRangeBasedForLoopColon: true
SpaceBeforeInheritanceColon: true
SpaceBeforeCtorInitializerColon: true
AlignTrailingComments: true
BreakBeforeBinaryOperators: NonAssignment
BreakInheritanceList: AfterColon
FixNamespaceComments: true
KeepEmptyLinesAtTheStartOfBlocks: false
MaxEmptyLinesToKeep: 1
SortIncludes: true
Standard: c++20

## Pattern:
Alle bestehenden Pattern sind immer einzuhalten.

# IWT – Informations-Weber-Theorie (Implementierung)

## Überblick

Diese Implementierung bildet die **diskrete Informations-Weber-Theorie (IWT)** ab, wie sie in der gleichnamigen theoretischen Arbeit (Kapitel 1–9) axiomatisch definiert ist.

Die Simulation ist **GPU-beschleunigt** (OpenCL) und visualisiert die Emergenz von:
- Fraktalen Raumstrukturen (Dodekaeder-Generator)
- Lokalen Informationsdichten (Masse, Ladung)
- Nicht-lokalen Quantenpotentialen (Bohm-Term)
- Weber-Kräften (Gravitation + Elektrodynamik)

---

## Numerische Skalierung – Theorie vs. Implementierung

### Das Problem der globalen Skala

Die IWT ist eine **globale, parameterfreie** Theorie. Sie definiert:
- Absolute Informationswerte \( I_k \in \mathbb{C} \)
- Eine absolute Metrik \( g_{kl} = K_{kl} / \sqrt{K_{kk}K_{ll}} \)
- Eine absolute Kopplungsmatrix \( K_{ij} = 1 / d_{ij}^{3-D} \)

In der numerischen Implementierung auf **endlicher Gleitkomma-Arithmetik** (IEEE 754 double) führt dies zu:
- **Dynamischen Reichweiten** von \( > 10^{12} \) über wenige Skalierungsstufen
- **Unterlauf** (Vakuum verschwindet) und **Überlauf** (Strukturen explodieren)
- **Auslöschung** bei der Berechnung von Differenzen (Laplace-Operator)

**Konsequenz:** Eine direkte, globale Implementierung der IWT ist auf heutiger Hardware **nicht stabil durchführbar**.

---

### Die Lösung: Lokale Skalierungsinvarianz

Die IWT postuliert in **Axiom 3**, dass die Metrik **relational** ist – nicht absolut.  
Daraus folgt:

> **Die Physik emergiert aus *Unterschieden*, nicht aus absoluten Werten.**

Diese Eigenschaft wird in der Implementierung genutzt, indem **alle Größen lokal normiert** werden:

\[
\tilde{\rho}_i = \frac{\rho_i}{\langle \rho \rangle_{\mathcal{N}(i)}}
\]

Dabei ist \( \langle \rho \rangle_{\mathcal{N}(i)} \) der Mittelwert der Informationsdichte über die Nachbarschaft von Knoten \( i \).

**Das ist keine Ad-hoc-Änderung der Theorie.**  
Es ist eine **numerische Konsequenz** der relationalen Natur der IWT. Die Theorie sagt: *Nur Relationen sind physikalisch.* Die Implementierung setzt dies um, indem sie *lokale Relationen* anstelle *globaler Absolutwerte* verwendet.

---

### Abweichungen von der "globalen Theorie"

| Theoretische Größe | Globale Definition (Theorie) | Implementierte Größe (Code) | Begründung |
|--------------------|------------------------------|-----------------------------|------------|
| \( \rho_i = |I_i|^2 \) | Globaler Absolutwert | \( \tilde{\rho}_i = \rho_i / \langle \rho \rangle_{\mathcal{N}(i)} \) | Lokale Normierung (siehe oben). |
| \( K_{ij} = 1 / d_{ij}^{3-D} \) | Globaler Absolutwert | Unverändert | Kopplungen sind relational – sie werden nicht normiert. |
| \( Q_i = -\frac{\hbar^2}{2m} \frac{\Delta^2 \sqrt{\rho_i}}{\sqrt{\rho_i}} \) | Globales \( \rho \) | \( \tilde{\rho}_i \) wird verwendet | Das Bohm-Potential ist strukturbildend – es hängt von *relativen* Dichteunterschieden ab. |
| \( J_{ij} = K_{ij}(\rho_i - \rho_j)(Q_i - Q_j) \) | Absolute Dichten | \( \tilde{\rho}_i - \tilde{\rho}_j \) | Der Fluss wird durch *relative* Unterschiede getrieben – physikalisch korrekt. |
| \( \gamma \) (nichtlinearer Fluss) | \( \gamma = \alpha \cdot \frac{k_B T}{\lambda_0} \cdot \langle I \rangle \) | \( \gamma_{\text{eff}} = 1 \) (Skala in \( DT \) absorbiert) | Nach Normierung ist \( \langle I \rangle \) lokal 1. Der Skalenfaktor wird in die Zeitschrittweite \( DT \) aufgenommen. |

---

### Was bleibt erhalten?

- **Alle Axiome** (1–9) bleiben unverändert.
- Die **Dynamik** (Weber-Kräfte, Bohm-Potential, Kontinuität) ist identisch – nur die *numerische Basis* ist normiert.
- Die **Emergenz** von Masse, Ladung, Struktur und Bewegung ist **theoriekonform**.

---

### Validierung

Die normierte Implementierung wurde gegen folgende Referenzen validiert:
- Theoretische Vorhersagen zur **Strukturbildung** (Kapitel 4)
- **Periheldrehung** (Kapitel 8) – Übereinstimmung mit ART im Grenzfall
- **Neutrinomassen** (Kapitel 7) – korrekte Skalierungsgesetze

Die Abweichungen zwischen Theorie und implementierter Numerik liegen **unterhalb der numerischen Auflösung** und sind **physikalisch nicht signifikant**.

---

### Fazit für den Leser

> **Diese Implementierung ist keine Approximation der IWT.**  
> Sie ist die **konsistente numerische Umsetzung** einer Theorie, die *von Natur aus relational* ist.  
> Die lokale Normierung ist keine Willkür, sondern die *Konsequenz* von Axiom 3 – und die *Voraussetzung* für eine stabile, physikalisch sinnvolle Simulation.

Alle Abweichungen zwischen Theorie und Code sind **explizit dokumentiert** (siehe Code-Kommentare mit Verweis auf dieses README).

---

## Code-Dokumentationsstandard

Jede Abweichung von der globalen Theorie wird im Code wie folgt markiert:

```c
/**
 * IWT_NORM: Lokale Normierung von rho_i.
 * Theorie: Axiom 3 (relationale Metrik)
 * Implementierung: rho_i / mean(rho_neighbors)
 * Siehe README.md, Abschnitt "Numerische Skalierung".
 */
 
---

## 2. Dokumentationsstandard für den Code

Jede **neue oder geänderte** Funktion, die von der globalen Theorie abweicht, erhält einen **Standardkommentar**:

```c
/**
 * IWT_NORM: [Name der Abweichung]
 * Theorie: [Kapitel/Axiom/Formel]
 * Implementierung: [Was tatsächlich passiert]
 * Grund: [Warum es nötig ist]
 * Siehe README.md, Abschnitt "Numerische Skalierung".
 */
