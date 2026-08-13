# DIY 3D Line-Laser Scanner

Projekt in Zusammenarbeit mit OpenAI-ChatGPT haben wir ein Günstigen(Wirklich Günstigen) 3D Scanner gebaut. Die Auflösung lässt noch zu Wünschen übrig und das Tracking wurde bisher nur auf dem Charuco Board getestet.
Es besteht derzeit ein Großer Offset in der Z-Ebene welcher meiner Vermutung nach mit der Günstigen Hardware zusammenhängt.
Getestet mit:
- Günstigen Ali-Express Linienlaser mit Kollimatorlinse (~7€)
- Rollei R100 Kamera (~5-10€)

Ein experimenteller handgeführter 3D-Scanner auf Basis von:

- USB-Kamera
- rotem Linienlaser
- ChArUco-Referenzbrett
- Python
- OpenCV
- Open3D

Der Scanner bestimmt 3D-Punkte durch **Lasertriangulation**.  
Die Position und Orientierung des Scannerkopfs wird während des Scans über ein fest im Raum liegendes ChArUco-Board bestimmt.

> **Projektstatus:** Funktionsfähiger Prototyp / Entwicklungsstand  
> Das System ist für Versuche, Messungen und Weiterentwicklung gedacht und derzeit kein kalibriertes industrielles Messmittel.

---

# 1. Funktionsprinzip

Der Scanner besteht aus einer Kamera und einem fest damit verbundenen Linienlaser.

Der Linienlaser projiziert eine rote Linie auf das zu erfassende Objekt. Die Kamera erkennt die Lage dieser Linie im Bild.

Aus

1. der Kamerakalibrierung,
2. der bekannten Laserebene und
3. dem Kamerastrahl eines erkannten Laserpixels

wird für jeden gültigen Laserpunkt ein 3D-Punkt berechnet.

Während der Scanner bewegt wird, bleibt das ChArUco-Board fest liegen. Dadurch kann OpenCV die aktuelle Position und Orientierung der Kamera relativ zum Board bestimmen.

Die einzelnen Messpunkte werden anschließend in ein gemeinsames Weltkoordinatensystem transformiert.

Vereinfacht:

```text
Kamera + Laser
      |
      | erkennt Laserlinie
      v
  Kamerabild
      |
      | Kamera-Ray + Laserebene
      v
3D-Punkt im Kamerasystem
      |
      | ChArUco-Pose
      v
3D-Punkt im Weltkoordinatensystem
      |
      v
   Punktwolke
```

---

# 2. Hardware

Der aktuelle Aufbau verwendet:

- **Kamera:** Rollei R100 USB-Kamera
- **Auflösung:** 1920 × 1080
- **Bildrate:** ca. 30 FPS angefordert
- **Linienlaser:** roter Linienlaser
- **Referenz:** ChArUco-Board
- Kamera und Laser sind mechanisch fest miteinander verbunden.

Der typische Arbeitsbereich liegt aktuell ungefähr bei:

```text
300 ... 500 mm
```

Im Live-Scanner ist ein größerer zulässiger Messbereich hinterlegt:

```python
MIN_Z_MM = 200.0
MAX_Z_MM = 700.0
```

## Mechanischer Aufbau

Für eine stabile Messung ist wichtig:

- Kamera und Laser dürfen sich relativ zueinander **nicht bewegen**.
- Nach jeder mechanischen Änderung muss die **Laserkalibrierung neu durchgeführt** werden.
- Wird die Kamera selbst verstellt, fokussiert oder optisch verändert, sollte auch die **Kamerakalibrierung neu durchgeführt** werden.
- Der Laser sollte möglichst scharf fokussiert sein.
- Kamera und Laser sollten sich mit einem ausreichenden Triangulationswinkel schneiden.

Der aktuelle Aufbau verwendet ungefähr einen Winkel im Bereich von etwa 25°.

---

# 3. Software-Voraussetzungen

Entwickelt und getestet unter Windows.

Empfohlen:

- Windows 10 / Windows 11
- Python 3.12
- OpenCV mit ArUco/ChArUco-Unterstützung
- NumPy
- Open3D

Im aktuellen Entwicklungsstand wurde unter anderem verwendet:

```text
OpenCV 5.0.0
Python 3.12
```

Benötigte Python-Pakete sind insbesondere:

```bash
pip install numpy open3d
```

OpenCV muss eine Installation sein, die `cv2.aruco` enthält.

Je nach Python-/OpenCV-Installation kann beispielsweise zusätzlich nötig sein:

```bash
pip install opencv-contrib-python
```

> Achtung: Nicht mehrere inkompatible OpenCV-Pakete gleichzeitig installieren.

---

# 4. Empfohlene Projektstruktur

Die Programme erwarten ungefähr folgende Struktur:

```text
3DScanner/
│
├─ modules/
│  ├─ camera_calibration.py
│  ├─ laser_calibration.py
│  └─ scan_live.py
│
├─ output/
│  │
│  ├─ charuco/
│  │  └─ charuco_config.json
│  │
│  ├─ calibration/
│  │  ├─ camera_calibration.json
│  │  ├─ laser_calibration.json
│  │  └─ laser_images_auto/
│  │
│  └─ scans/
│     ├─ scan_YYYYMMDD_HHMMSS_raw.csv
│     ├─ scan_YYYYMMDD_HHMMSS_raw.ply
│     ├─ scan_YYYYMMDD_HHMMSS_filtered.csv
│     └─ scan_YYYYMMDD_HHMMSS_filtered.ply
│
└─ README.md
```

Die Skripte bestimmen den Projektpfad über:

```python
ROOT = Path(__file__).resolve().parents[1]
```

Daher sollten die Python-Dateien im Ordner `modules/` liegen.

---

# 5. ChArUco-Board

Der aktuell verwendete Aufbau basiert auf:

```text
6 × 9 Squares
Square Length: 28.0 mm
Marker Length: 20.0 mm
Dictionary: DICT_4X4_50
```

Physische Boardgröße:

```text
168 × 252 mm
```

Das Board dient gleichzeitig als:

- Referenz für die Kamerakalibrierung,
- bekannte Ebene bei der Laserkalibrierung,
- Weltkoordinatensystem beim Live-Scan.

## Wichtig beim Drucken

Das Board muss **maßhaltig** gedruckt werden.

Kein:

- "An Seite anpassen"
- automatisches Skalieren
- Verkleinern auf Druckbereich

Nach dem Druck sollte eine bekannte Kontrollstrecke mit einem Lineal oder Messschieber geprüft werden.

Ein falsch skaliertes Board erzeugt direkt falsche 3D-Maße.

---

# 6. Kalibrierreihenfolge

Die Reihenfolge ist zwingend:

```text
1. ChArUco-Board prüfen
        ↓
2. Kamera kalibrieren
        ↓
3. Laser kalibrieren
        ↓
4. Testscan
        ↓
5. Objekt scannen
```

Eine Laserkalibrierung darf **nicht** mit alten Kameraparametern weiterverwendet werden, wenn die Kamera neu kalibriert wurde.

---

# 7. Kamerakalibrierung

Die Kamerakalibrierung bestimmt:

- Brennweite `fx`
- Brennweite `fy`
- Hauptpunkt `cx`
- Hauptpunkt `cy`
- Linsenverzeichnung / Distortion

Die Kamera zeigt eine deutliche Weitwinkel-/Tonnenverzeichnung. Diese wird über die OpenCV-Distortion-Koeffizienten berücksichtigt.

Beispiel:

```json
{
  "fx": 1456.9,
  "fy": 1458.5,
  "cx": 970.8,
  "cy": 545.1
}
```

Die Kalibrierung sollte viele unterschiedliche Ansichten enthalten:

- frontal
- links
- rechts
- oben
- unten
- diagonal
- Board in unterschiedlichen Bildbereichen
- verschiedene Abstände

Der automatische Kalibrierassistent führt den Anwender durch die benötigten Positionen.

## Gute Kamerakalibrierung erkennen

Wichtige Kennwerte sind:

```text
RMS Reprojection Error
Mean Reprojection Error
Median Reprojection Error
Max Reprojection Error
```

Der aktuelle gute Kalibrierstand lag ungefähr bei:

```text
RMS:                 ~0.49 px
Mean Reprojection:   ~0.43 px
Median Reprojection: ~0.43 px
```

Je kleiner der Reprojection Error, desto besser.

Ein niedriger Fehler allein garantiert jedoch noch keine perfekte Kalibrierung. Wichtig ist ebenfalls eine gute räumliche Verteilung der aufgenommenen Boardpositionen.

---

# 8. Laser-Kalibrierung

## Ziel

Der Linienlaser wird mathematisch als Ebene beschrieben:

```text
a·X + b·Y + c·Z + d = 0
```

Die Parameter werden in:

```text
output/calibration/laser_calibration.json
```

gespeichert.

Die Laserebene liegt im **Kamerakoordinatensystem**.

---

# 9. Robuste Erkennung der Laserlinie

Ein wichtiger Punkt des Projekts ist die Bestimmung der exakten Position der Laserlinie.

Eine reale Laserlinie besitzt im Kamerabild keine Breite von exakt einem Pixel.

Typische Ursachen:

- reale Strahlbreite
- Fokus
- Blooming
- Reflexionen
- Überbelichtung
- Oberflächenstruktur
- Kameraoptik

Daher wird nicht einfach der hellste Pixel verwendet.

Die aktuelle Version bestimmt stattdessen ein **robustes intensitätsgewichtetes Profilzentrum**.

Vereinfacht:

```text
Intensität

          /\
        /    \
______/      \______
          ↑
     Profilzentrum
```

Ablauf pro Bildspalte:

1. rote Pixel erkennen
2. zusammenhängende Pixelbereiche bilden
3. unplausible Linienbreiten verwerfen
4. lokalen Hintergrund bestimmen
5. Hintergrund abziehen
6. schwache Profilanteile entfernen
7. intensitätsgewichtetes Zentrum berechnen
8. Sättigung prüfen
9. Symmetrie / Profilqualität bewerten
10. nur gültige Punkte weiterverwenden

Dadurch wird das Zentrum subpixelgenau bestimmt.

## Aktuelle Qualitätsparameter

```python
MIN_LINE_WIDTH_PX = 1
MAX_LINE_WIDTH_PX = 14

LASER_BACKGROUND_MARGIN_PX = 10

MIN_PROFILE_PROMINENCE = 35.0

MAX_PROFILE_SATURATED_RATIO = 0.70

PROFILE_RELATIVE_FLOOR = 0.20

MIN_PROFILE_QUALITY = 0.30
```

### Maximale Linienbreite

```python
MAX_LINE_WIDTH_PX = 14
```

Laserbereiche breiter als 14 Pixel werden verworfen.

Das hilft insbesondere gegen:

- starke Reflexionen
- rote Flächen
- Blooming
- aufgeweitete Laserflecken

Für eine stärker präzisionsorientierte Einstellung kann dieser Wert später beispielsweise auf 8 Pixel reduziert werden.

Dies reduziert eventuell die Punktzahl, kann aber die Messqualität erhöhen.

---

# 10. Wichtig: identische Laser-Erkennung

**Laser-Kalibrierung und Live-Scanner müssen exakt dieselbe Definition der Laserposition verwenden.**

Nicht zulässig wäre beispielsweise:

```text
Kalibrierung: obere Kante
Scan: Linienmitte
```

oder:

```text
Kalibrierung: stärkster Pixel
Scan: Profilzentrum
```

Die aktuelle Version verwendet in beiden Programmen:

```text
robustes intensitätsgewichtetes Profilzentrum
```

Wird die Laser-Erkennung geändert, muss anschließend die **Laserkalibrierung erneut durchgeführt** werden.

---

# 11. Geführte Laser-Kalibrierung

Die Laser-Kalibrierung führt automatisch durch mehrere Positionen.

Typische Positionen:

```text
frontal
von links
von rechts
von oben
von unten
diagonal
nah
weit
Board links im Bild
Board rechts im Bild
```

Die Anzeige gibt konkrete Hinweise, zum Beispiel:

```text
SCANNER NAEHER ZUM BOARD
SCANNER WEITER VOM BOARD WEG
MEHR NACH LINKS GEHEN
SCANNER HOEHER HALTEN
LASERLINIE AUF DAS BOARD RICHTEN
RICHTIG - RUHIG HALTEN
```

Zusätzlich wird die gewünschte Boardposition im Kamerabild dargestellt.

## Automatische Aufnahme

Eine Pose wird automatisch gespeichert, wenn:

- genügend ChArUco-Corners erkannt werden,
- die Boardpose gültig ist,
- Abstand im Sollbereich liegt,
- Winkel im Sollbereich liegt,
- Board im richtigen Bildbereich liegt,
- genügend Laserlinie erkannt wird,
- genügend 3D-Punkte erzeugt werden,
- die Bedingungen mehrere Frames stabil bleiben.

Aktuelle Grundwerte:

```python
MIN_CHARUCO_CORNERS = 18
MIN_LASER_PIXELS = 120

MAX_POSE_ERROR_PX = 1.0

STABLE_FRAMES_REQUIRED = 5

ANGLE_TOLERANCE_DEG = 10.0
DISTANCE_TOLERANCE_MM = 90.0
POSITION_TOLERANCE_NORM = 0.20
```

---

# 12. Bedienung der Laser-Kalibrierung

Das Programm läuft weitgehend automatisch.

Tasten:

```text
M = Lasermaske anzeigen / ausblenden
R = komplette Kalibriersequenz zurücksetzen
Q = Programm beenden
ESC = Programm beenden
```

Nach erfolgreicher Sequenz wird automatisch erzeugt:

```text
output/calibration/laser_calibration.json
```

Zusätzlich können Bilder der aufgenommenen Kalibrierpositionen gespeichert werden.

---

# 13. Live-Scanner starten

Nach erfolgreicher Kamera- und Laserkalibrierung kann der Live-Scanner gestartet werden.

Beispiel:

```bash
python modules/scan_live.py
```

Vorher prüfen:

```text
camera_calibration.json vorhanden
laser_calibration.json vorhanden
charuco_config.json vorhanden
Kamera frei
ChArUco-Board sichtbar
Laser eingeschaltet
```

---

# 14. Bedienung des Live-Scanners

Die Steuerung erfolgt über globale Windows-Hotkeys.

| Taste | Funktion |
|---|---|
| `SPACE` halten | Punkte aufnehmen |
| `F` | gefilterte / ungefilterte Live-Punktwolke umschalten |
| `S` | aktuellen Scan speichern |
| `R` | aktuelle Punktwolke löschen / Scan zurücksetzen |
| `Q` | Scanner beenden |
| `ESC` | Scanner beenden |

## Aufnahme

Nur solange `SPACE` gedrückt wird, werden neue Punkte gespeichert.

Zusätzlich müssen gleichzeitig gültig sein:

```text
ChArUco-Tracking OK
Laser erkannt
gültige 3D-Punkte vorhanden
neuer Kameraframe vorhanden
```

Dadurch werden keine mehrfach verwendeten alten Frames in die Punktwolke geschrieben.

---

# 15. Live-Anzeige

Während des Scans werden unter anderem angezeigt:

```text
FPS
Anzahl ChArUco-Corners
Trackingstatus
Pose-Reprojection-Error
Positionsänderung
Rotationsänderung
Pose-Geschwindigkeit
Laserbreite
Laser-Profilqualität
Objektabstand
Kamera-Z
```

Beispiel:

```text
TRACKING OK
Pose err: 0.42 px
dPos: 1.25 mm
dRot: 0.31 deg
vPose: 145 mm/s | 9.5 deg/s

Laserbreite: 5.40 px
Profilqualitaet: 0.87
```

Diese Werte sind besonders bei der Fehlersuche hilfreich.

---

# 16. Robustes Pose-Tracking

Die ChArUco-Pose wird mit OpenCV bestimmt.

Verwendet wird:

```python
cv2.SOLVEPNP_ITERATIVE
```

Anschließend erfolgt eine Verfeinerung über:

```python
cv2.solvePnPRefineLM(...)
```

Der vorherige gültige Posewert wird als Startwert für den nächsten Frame verwendet.

Das verbessert die zeitliche Stabilität.

---

# 17. Anti-Jitter-Tracking

Ein Hauptproblem eines handgeführten Scanners sind falsche Einzelposen.

Eine einzelne fehlerhafte Pose kann eine komplette Laserlinie mehrere Millimeter versetzen.

Daher besitzt der Scanner eine zusätzliche Plausibilitätsprüfung.

Aktuelle Werte:

```python
MIN_CHARUCO_CORNERS = 14

MAX_POSE_ERROR_PX = 1.00

SOFT_TRANSLATION_JUMP_MM = 6.0
SOFT_ROTATION_JUMP_DEG = 2.0

PENDING_CONFIRM_TRANSLATION_MM = 8.0
PENDING_CONFIRM_ROTATION_DEG = 3.0

MAX_TRANSLATION_SPEED_MM_S = 1800.0
MAX_ROTATION_SPEED_DEG_S = 220.0

MAX_TRANSLATION_JUMP_MM = 45.0
MAX_ROTATION_JUMP_DEG = 14.0

POSE_RESET_AFTER_BAD_FRAMES = 8
```

## Kleine Bewegungen

Kleine plausible Änderungen werden direkt akzeptiert.

Dadurch entsteht möglichst wenig Verzögerung.

## Größere Bewegungen

Eine größere Bewegung wird zunächst als verdächtig behandelt.

Der Frame wird **noch nicht gespeichert**.

Erst wenn der nächste Frame eine ähnliche Pose bestätigt, wird die Bewegung akzeptiert.

Dadurch werden einzelne Pose-Spikes unterdrückt.

## Harte Sprünge

Unrealistische Änderungen werden vollständig verworfen.

Beispiele:

```text
zu großer Translationssprung
zu großer Rotationssprung
unrealistisch hohe Posegeschwindigkeit
schlechter Reprojection Error
zu wenige ChArUco-Corners
```

Nach mehreren ungültigen Frames wird das Tracking neu initialisiert.

---

# 18. Koordinatensysteme

## Kamerakoordinatensystem

Die aus Lasertriangulation erzeugten Punkte liegen zunächst im Kamerasystem.

Typisch:

```text
X = horizontal
Y = vertikal
Z = Blickrichtung der Kamera
```

## Weltkoordinatensystem

Über die ChArUco-Pose werden die Punkte ins Board-Koordinatensystem transformiert.

Die Umrechnung erfolgt sinngemäß über:

```python
R.T @ (camera_points - t)
```

Das ChArUco-Board bildet somit die feste Weltreferenz.

---

# 19. Punktwolke

Die aufgenommenen 3D-Punkte werden zunächst ungefiltert gespeichert.

Zusätzlich kann eine gefilterte Version erstellt werden.

Aktuell:

```python
VOXEL_SIZE_MM = 0.5

STAT_NB_NEIGHBORS = 20
STAT_STD_RATIO = 1.5
```

Verwendete Open3D-Verfahren:

1. Voxel Downsampling
2. Statistical Outlier Removal

Wichtig:

> Der Filter kann einzelne Ausreißer entfernen, aber keine systematischen Messfehler reparieren.

Wenn ganze Scanlinien falsch liegen, muss die Ursache im Tracking oder in der Kalibrierung behoben werden.

---

# 20. Ausgabeformate

Beim Speichern werden Punktwolken typischerweise als:

```text
PLY
CSV
```

ausgegeben.

CSV-Struktur:

```text
X_mm;Y_mm;Z_mm
```

Beispiel:

```csv
X_mm;Y_mm;Z_mm
42.183;128.442;0.381
42.722;128.517;0.417
43.260;128.592;0.402
```

Die Rohdaten werden mit `_raw` gespeichert.

Die gefilterte Punktwolke erhält `_filtered`.

---

# 21. Empfohlener Scanablauf

## Vorbereitung

1. ChArUco-Board flach und fest positionieren.
2. Board darf sich während des Scans nicht bewegen.
3. Scanner einschalten.
4. Laser einschalten.
5. Kamera freigeben.
6. Live-Scanner starten.
7. Tracking prüfen.

## Scan

1. Scanner auf Objekt richten.
2. Sicherstellen, dass das ChArUco-Board ausreichend erkannt wird.
3. `SPACE` gedrückt halten.
4. Scanner langsam und gleichmäßig bewegen.
5. Keine abrupten Bewegungen durchführen.
6. Laserlinie über das Objekt führen.
7. `SPACE` loslassen.
8. Punktwolke kontrollieren.
9. Mit `S` speichern.

---

# 22. Geschwindigkeit beim Scannen

Der Scanner funktioniert besser bei:

- langsamen,
- kontinuierlichen,
- gleichmäßigen Bewegungen.

Ungünstig:

- ruckartige Bewegungen,
- starkes Verdrehen,
- kurzfristiges Verdecken des Boards,
- extrem schnelle Richtungswechsel.

Auch wenn das Tracking schnelle Bewegungen teilweise akzeptieren kann, steigt die Messqualität bei ruhiger Führung deutlich.

---

# 23. Qualitätskontrolle

Vor echten Objektmessungen sollte ein bekanntes flaches Objekt gescannt werden.

Ideal:

```text
flaches ChArUco-Board
oder
ebene Platte
```

Danach kann eine Ebene an die Punktwolke angepasst werden.

Wichtige Kennwerte:

```text
Ebenenneigung
RMS-Abweichung
mittlere absolute Abweichung
95%-Bereich
Maximalabweichung
```

---

# 24. Aktueller Entwicklungsstand der Genauigkeit

Im Entwicklungsverlauf konnten mehrere Fehlerquellen getrennt werden.

## Vor neuer Kamera-/Laser-Kalibrierung

Es traten starke systematische Verformungen auf.

Beispielsweise:

```text
~17° scheinbare Ebenenneigung
mehrere Zentimeter Höhenfehler über das Board
```

Nach erneuter Kamera- und Laserkalibrierung wurde diese systematische Geometrieabweichung weitgehend beseitigt.

## Aktueller flacher bewegter Scan

Typische Größenordnung:

```text
Ebenenneigung < 1°
Residual RMS etwa 1 ... 2 mm
```

## Stationärer Test

Bei vollständig stillgehaltenem Scanner wurde ungefähr erreicht:

```text
RMS etwa 0.27 mm
90 % etwa innerhalb ±0.41 mm
95 % etwa innerhalb ±0.49 mm
```

Das zeigt:

> Die Lasertriangulation selbst ist im Stillstand bereits deutlich genauer als ein bewegter Scan.

Die derzeit größte verbleibende Fehlerquelle ist daher das Pose-Tracking während der Bewegung.

---

# 25. Typische Fehler

## Kamera kann nicht geöffnet werden

Beispiel:

```text
Kamera konnte nicht geoeffnet werden
```

Prüfen:

- Windows Kamera-App geschlossen?
- Browser benutzt Kamera?
- OBS geöffnet?
- anderes Python-Skript läuft noch?
- Kamera an anderem USB-Port?
- richtiger `CAMERA_INDEX`?

Nur ein Programm sollte gleichzeitig auf die Kamera zugreifen.

---

## ChArUco wird schlecht erkannt

Mögliche Ursachen:

- Board zu klein im Bild
- Bewegungsunschärfe
- schlechte Beleuchtung
- Reflexion
- Laser überstrahlt Marker
- Board teilweise verdeckt

Maßnahmen:

- langsamer bewegen
- besser beleuchten
- Board größer im Bild halten
- Kameraexposure reduzieren
- Laserleistung reduzieren

---

## Laser wird nicht erkannt

Prüfen:

```python
MIN_RED
MIN_RED_DIFFERENCE
MIN_PROFILE_PROMINENCE
```

Auch die Lasermaske mit `M` anzeigen.

---

## Zu viele Reflexionen

Mögliche Maßnahmen:

- Laserleistung reduzieren
- Kameraexposure reduzieren
- Scanwinkel verändern
- maximale Linienbreite reduzieren

Beispiel für strengere Einstellung:

```python
MAX_LINE_WIDTH_PX = 8
MAX_PROFILE_SATURATED_RATIO = 0.50
MIN_PROFILE_QUALITY = 0.45
```

Das führt zu weniger Punkten, aber möglicherweise höherer Genauigkeit.

---

## Punktwolke springt in Z

Wenn ganze Linien oder Flächen in der Höhe springen:

Nicht sofort den Open3D-Filter erhöhen.

Zuerst prüfen:

```text
Pose Error
dPos
dRot
vPose
Laserbreite
Profilqualität
ChArUco-Corners
```

Ist der Scanner im Stillstand sauber, aber während der Bewegung nicht, deutet dies stark auf Pose-/Trackingfehler hin.

---

# 26. Kalibrierung neu durchführen, wenn ...

## Kamera neu kalibrieren

Wenn:

- Kamera gewechselt wird
- Auflösung geändert wird
- Zoom geändert wird
- Fokus stark verändert wird
- Optik verändert wird

Danach muss ebenfalls der Laser neu kalibriert werden.

## Laser neu kalibrieren

Wenn:

- Laser relativ zur Kamera bewegt wird
- Halterung verändert wird
- Fokus des Lasers verändert wird
- Laser aus- und wieder anders eingebaut wird
- Laser-Erkennungsalgorithmus geändert wird
- Kamerakalibrierung geändert wurde

---

# 27. Was nicht verändert werden darf

Während einer gültigen Laserkalibrierung müssen Kamera und Laser eine starre Einheit bilden.

Nicht während des Betriebs:

```text
Laserhalter verdrehen
Kamera relativ zum Laser verschieben
Kameramodul lösen
Laser neu fokussieren
```

Andernfalls stimmt die gespeicherte Laserebene nicht mehr.

---

# 28. Debugging-Strategie

Bei schlechter Messqualität sollte nicht alles gleichzeitig verändert werden.

Empfohlene Reihenfolge:

```text
1. Kamerakalibrierung prüfen
2. Laserprofil im Bild prüfen
3. Laserkalibrierung prüfen
4. stationären Scan durchführen
5. bewegten Scan durchführen
6. Pose-Tracking analysieren
7. erst danach Punktwolkenfilter anpassen
```

Damit lässt sich die Fehlerquelle gezielt eingrenzen.

---

# 29. Stationärer Referenztest

Ein sehr nützlicher Test:

1. Scanner auf eine ebene Fläche richten.
2. Scanner **nicht bewegen**.
3. `SPACE` mehrere Sekunden gedrückt halten.
4. Scan speichern.
5. Streuung der wiederholt gemessenen Linie auswerten.

Wenn der stationäre Scan gut ist, aber ein bewegter Scan schlecht:

```text
Lasertriangulation wahrscheinlich OK
Pose-/Bewegungstracking wahrscheinlich Hauptfehler
```

Wenn bereits der stationäre Scan stark streut:

```text
Lasererkennung
Laserkalibrierung
Kamera
Belichtung
Reflexion
```

weiter untersuchen.

---

# 30. Entwicklungsziele

Mögliche nächste Verbesserungen:

- bewegungsabhängiger Posefilter
- Pose-Vorhersage aus Geschwindigkeit
- adaptive Trackinggrenzen
- bessere Bewertung einzelner ChArUco-Corners
- Belichtungssteuerung
- automatische Laserprofil-Diagnose
- automatische Qualitätskarte der Punktwolke
- Scanlinien-Interpolation
- Mesh-Rekonstruktion
- automatische Oberflächennormalen
- ICP-Unterstützung als zusätzliche Registrierung
- GUI für Kalibrierung und Scan
- gemeinsame zentrale Konfigurationsdatei

---

# 31. Wichtige Grundregel des Projekts

Bei Änderungen immer daran denken:

> **Kamera-, Laser- und Posemodell bilden gemeinsam eine Messkette.**

Eine Änderung an einem Teil kann die gesamte Geometrie beeinflussen.

Besonders wichtig:

```text
Neue Kamerakalibrierung
        ↓
Laser neu kalibrieren
        ↓
Testscan
        ↓
erst dann Objekt scannen
```

und:

```text
Laser-Erkennung geändert
        ↓
Laser neu kalibrieren
        ↓
Scan mit exakt derselben Laser-Erkennung
```

---

# 32. Kurzstart

Wenn bereits alle Dateien vorhanden sind:

```text
1. ChArUco-Board fest aufstellen
2. Kamera + Laser anschließen
3. Laser einschalten
4. scan_live.py starten
5. TRACKING OK abwarten
6. SPACE halten und langsam scannen
7. SPACE loslassen
8. S drücken
9. Dateien unter output/scans prüfen
```

Bei geänderter Mechanik:

```text
1. Kamera prüfen / ggf. kalibrieren
2. Laser neu kalibrieren
3. flaches Board testen
4. erst danach Objekt scannen
```

---

# 33. Sicherheit

Ein Linienlaser ist kein Spielzeug.

- Nicht direkt in den Laserstrahl sehen.
- Laser nicht auf Personen richten.
- Reflexionen an spiegelnden Oberflächen beachten.
- Geeignete Laserklasse und lokale Sicherheitsvorgaben beachten.
- Bei unbekannter oder hoher Laserleistung geeigneten Augenschutz verwenden.

---

# 34. Hinweis zur Messgenauigkeit

Die vom Scanner ausgegebenen Millimeterwerte sind geometrisch berechnete Messwerte.

Der aktuelle Aufbau ist jedoch noch ein Entwicklungsprototyp.

Für Anwendungen mit verbindlichen Toleranzanforderungen ist eine zusätzliche Rückführung auf bekannte Referenzkörper und eine Messsystemanalyse erforderlich.

Insbesondere sollte nicht allein aus einer niedrigen Punktwolkenstreuung geschlossen werden, dass die absolute Genauigkeit identisch hoch ist.

---

# 35. Lizenz / Verwendung

Noch nicht festgelegt.

Vor Veröffentlichung des Repositories sollte hier eine gewünschte Lizenz ergänzt werden, beispielsweise:

```text
MIT
Apache-2.0
GPL-3.0
```

---

# 36. Zusammenfassung

Das Projekt kombiniert:

```text
ChArUco Pose Tracking
        +
Kamerakalibrierung
        +
robuste Laserlinienerkennung
        +
Lasertriangulation
        +
Anti-Jitter-Tracking
        +
Open3D Punktwolkenverarbeitung
```

zu einem handgeführten experimentellen 3D-Laserscanner.

Der aktuelle Fokus liegt auf einer möglichst reproduzierbaren geometrischen Messung und der Reduktion von Posefehlern während der Bewegung.
