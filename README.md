# Draftion

**Brawl Stars Ranked Drafting Tool** von Kian Gray.

Trainiertes ML-Modell, das aus Map + bisherigen Picks/Bans einer Ranked-Draft
den vielversprechendsten nächsten Ban oder Pick vorschlägt, plus interaktives
Web-Tool (Draft-Empfehlungen live, Auswertung nach dem Draft).

## Wichtige Einschränkungen der offiziellen API

- **Kein Draft-Verlauf.** Battlelog-Einträge enthalten nur die finale
  3v3-Zusammensetzung, die Map und das Ergebnis - nicht welche Brawler gebannt
  wurden oder in welcher Reihenfolge gepickt wurde. Das Modell lernt deshalb
  "Siegwahrscheinlichkeit von Team A gegeben (Map, Team A, Team B)" aus
  abgeschlossenen Matches, nicht die Zugreihenfolge selbst. Die eigentliche
  Draft-Empfehlung entsteht später, indem man mit diesem Modell als
  Bewertungsfunktion über mögliche nächste Picks sucht.
- **API-Key ist IP-gebunden.** Ändert sich deine öffentliche IP, bekommst du
  403-Fehler und musst dir auf developer.brawlstars.com einen neuen Key für
  die neue IP generieren.
- **Kein Bulk-Endpoint.** `battlelog` zeigt nur die letzten ~25 Kämpfe pro
  Spieler, mit bis zu 30 Min Verzögerung für neue. Um genug Ranked-Matches zu
  sammeln, muss der Crawler über Tage/Wochen wiederholt laufen (z.B. per
  Cronjob) statt einmalig.

## Setup

1. Account auf https://developer.brawlstars.com erstellen, Key für deine
   aktuelle öffentliche IP generieren.
2. `.env.example` nach `.env` kopieren und den Token eintragen.
3. Virtuelle Umgebung anlegen und Pakete installieren:
   `python -m venv .venv-compat`, dann
   `.venv-compat/Scripts/python.exe -m pip install -r requirements.txt`.
   Alle Befehle unten mit diesem Python starten (statt `python`).

**Windows + Smart App Control:** Die Versionen in `requirements.txt` sind mit
Absicht auf ältere, weit verbreitete Releases festgenagelt. Ist Smart App
Control aktiv, blockiert Windows die neuesten scikit-learn/scipy/numpy-Dateien
(unbekannter Ruf), und der Server stürzt beim Start mit `DLL load failed ... An
Application Control policy has blocked this file` ab. Passiert das trotzdem,
installiere die Pakete in ein frisches venv mit genau diesen Versionen - die
Einstellung selbst musst du dafür nicht ändern.

## Nutzung

```bash
# Crawler starten (sammelt Kämpfe über Top-Spieler + Snowball-Sampling)
python -m src.crawler --max-players 2000

# Folgeläufe: bekannte Ranked-Spieler ab Lobby-Rang 12 erneut abklappern (per Cron)
python -m src.crawler --max-players 2000 --reseed-from-rank 12 --recrawl-hours 2

# Nach ein paar Durchläufen prüfen, was in der Datenbank steckt (siehe unten:
# Ranked-Drafts sind Typ `soloRanked`, ihr Rang steckt in brawler.trophies)
python -m src.extract types
python -m src.extract ranks           # Standard-Typ ist soloRanked; anderer Typ: `ranks <typ>`
python -m src.extract status          # Fortschritt: nutzbare Kämpfe (Ziel 40 000+), Tempo, abgefragte Spieler

# Komplette Brawler-Liste + Portraits holen -> data/brawlers.json. Wird von
# Modell, Draft-Engine, API und Frontend gemeinsam genutzt. Funktioniert auch
# ohne BS_API_TOKEN (Brawlify), mit Token kommen die Namen von der offiziellen API.
python -m src.brawlers fetch

# Map-Liste + Map-Bilder holen -> data/maps.json (Brawlify, kein Key nötig).
# Nur aktuelle Maps der 3v3-Modi Gem Grab, Brawl Ball, Knockout, Hot Zone,
# Heist, Bounty (`--modes ...` ändert das)
python -m src.maps fetch

# Modell trainieren, sobald genug Ranked-Daten da sind. Jeder Lauf schreibt eine NEUE
# Version nach data/models/win_model_<UTC-Zeit>.joblib (die letzten 5 bleiben); ein
# laufender Server lädt die neueste ohne Neustart. Die Regularisierung C wählt das
# Training selbst per Zeitreihen-CV (--c erzwingt einen Wert).
python -m src.model train --min-rank 12 --backend logreg   # Typ-Standard: soloRanked

# Trainings-Stellschrauben (min_rank, weight_step, C, Kalibrierung) und den Vergleich
# Modell <-> Rohstatistik auf den bisher gesammelten Daten messen; bei viel mehr Daten
# wieder laufen lassen, die besten Werte wandern mit der Datenmenge
python -m src.tune

# Unit-Tests (Suchalgorithmus gegen eine künstliche Bewertungsfunktion, Regeln,
# Fallback-Modell, Modellversionen/-auswahl, Map-Pool, API) - brauchen keine echten Daten
python -m unittest discover -s tests -t . -v

# Backend + Web-UI starten (unter einer URL, http://localhost:8000, Abschnitt 0)
uvicorn src.api:app --reload          # nur Web-App
python -m src.serve                   # zusätzlich Crawler + Neutraining im Hintergrund (braucht BS_API_TOKEN)
```

Web-UI (englisch, Abschnitt 11): Startseite mit **First Pick dieses Matches**
(mein Team / Gegner - wird pro Match gewählt) und Map (anklicken/suchen, nach
Modus filtern, oder überspringen), danach nur noch Brawler anklicken -
Ban/Pick und Team leitet die App selbst aus der Zugreihenfolge ab
(`GET /rules?first_pick=A|B`), die Leiste oben aktualisiert sich sofort.
Sind alle Picks drin, ersetzt ein **Ergebnis-Screen** das Brawler-Grid
(`POST /analyze`): großer Gewinnchancen-Balken (z. B. 60 % - 40 %), Matchup-Matrix
"wer schlägt wen", beste/schwerste Matchups fürs eigene Team, Beitrag jedes
Brawlers und Teamkollegen-Synergien. Alles kommt aus demselben Modell
(Teil-Aufstellungen als Sonden, siehe `src/draft_engine/analysis.py`) - eine
Modell-Einschätzung, keine gemessene Winrate.
Rechts stehen die 8 besten Vorschläge (Klick auf eine Zeile übernimmt sie), die
Top 3 sind zusätzlich im Grid umrandet. <kbd>Enter</kbd> übernimmt die
Top-Empfehlung ohne Maus - oder den ersten Treffer, wenn im Suchfeld etwas steht.
<kbd>Esc</kbd> leert zuerst das Suchfeld, sonst macht es die letzte Aktion
rückgängig. <kbd>/</kbd> springt ins Suchfeld.

**Karten-Auswahl = aktueller Ranked-Pool.** Die Startseite zeigt standardmäßig nur die
Maps, die in den letzten 7 Tagen (`MAP_WINDOW_DAYS`) mindestens 3-mal
(`MAP_MIN_BATTLES`) in gecrawlten Ranked-Kämpfen gespielt wurden - bei den ersten
echten Daten waren das 26 von 132 (4 je Modus, 6 bei Hot Zone). Ein Umschalter unter den
Modus-Chips ("Current Ranked pool: 26 maps · show all 132") blendet den Rest ein; wer eine
Map außerhalb des Pools eintippt und Enter drückt, bekommt trotzdem die richtig
geschriebene Map. Solange der Crawl weniger als 6 aktive Maps gesehen hat, wird nichts
versteckt. (`GET /events/rotation` ist dafür **nicht** geeignet, siehe unten.)

**Wie sehr man den Zahlen trauen darf**, sagt die Oberfläche selbst (Banner oben plus
Badge neben Empfehlungen und Ergebnis; `model_confidence` in `GET /health`):

| Stufe | Wann | Anzeige |
|---|---|---|
| `demo` | Modell auf synthetischen Daten | gelb: "Demo model" - Platzhalter |
| `low` | zu wenig Daten für ein Modell, das die Rohstatistik schlägt: es antwortet die **Rohstatistik** (`src/baseline.py`, nur Brawler-Winrates, gemessen und geschrumpft) | "Preliminary · low confidence" |
| `early` | trainiertes Modell auf < 10 000 Kämpfen | "Early model" |
| `ok` | trainiertes Modell auf ≥ 10 000 Kämpfen | kein Hinweis |

Welches Modell antwortet, entscheidet `src/model_select.py`: ein echt trainiertes
Modell mit ≥ `MIN_TRAIN_ROWS` Kämpfen, **das die Rohstatistik in der beim Training
gemessenen Zeitreihen-CV nicht verliert**; sonst die Rohstatistik (ab
`MIN_RAW_BATTLES` echten Kämpfen); sonst das mitgelieferte Demo-Modell. Es gibt keine
feste Schwelle, die man nachpflegen müsste - sobald das Modell besser ist, übernimmt es
von selbst.

Vollständige Spezifikation inkl. Draft-Engine-Anforderungen (Denial-Bans,
Lookahead-Picks, Minimax-Suche) und offener Punkte: siehe
`Brawl_Stars_Draft_AI_Buildprompt.pdf`.

## Pipeline-Test ohne echte Daten

`data/brawl.db` existiert erst, sobald der Crawler eine Weile gegen einen
echten API-Key gelaufen ist. Um Feature-Encoding, Modell-Training,
Draft-Suche (Minimax + Denial/Lookahead) und die FastAPI-Endpunkte trotzdem
end-to-end zu prüfen, gibt es einen synthetischen Smoke-Test:

```bash
python -m src.synthetic run
```

Erzeugt einen künstlichen Ranked-Datensatz mit eingebautem Bias (ein paar
Brawler bewusst stärker, ein Synergie-Paar, ein Counter-Paar), trainiert
darauf ein Modell, prüft ob der Bias in der Vorhersage wiederzufinden ist,
lässt die Draft-Suche einmal laufen und schickt eine echte Anfrage gegen
`/recommend`. Alle Artefakte landen unter `data/_smoke/` (git-ignoriert,
gefahrlos löschbar) - kein Ersatz für echte Daten, nur ein
Pipeline-Korrektheitscheck.

## Architektur

| Datei | Abschnitt | Zweck |
|---|---|---|
| `src/api_client.py`, `src/db.py`, `src/crawler.py` | 2-4 | Datensammlung |
| `src/extract.py`, `src/stats.py` | 5-7 | Trainingstabelle, explorative Stats |
| `src/brawlers.py` | - | Brawler-Liste + Portraits (ID/Name von der API, Bilder von Brawlify; nicht geraten) |
| `src/features.py` | 8 | Sparse Feature-Encoding (Brawler + Paar-Interaktionen, siehe Abschnitt 6) |
| `src/model.py` | 8 | Win-Probability-Modell (logreg/GBM), Zeit-Split, `C` per Zeitreihen-CV, Kalibrierung (Platt/Isotonic, erst ab genug Daten), versionierte Modell-Dateien |
| `src/baseline.py` | 5 | Fallback-Modell aus rohen Brawler-Winrates (`stats.brawler_winrate`), solange das trainierte Modell nicht besser ist |
| `src/model_select.py` | 5 | Welches Modell antwortet: trainiert / Rohstatistik / Demo, plus `model_confidence` fürs UI |
| `src/config.py` | - | Alle Umgebungsvariablen an einer Stelle (API und Worker teilen sie) |
| `src/tune.py` | 8 | Misst `min_rank`, `weight_step`, `C`, Kalibrierung und Modell-gegen-Rohstatistik per Zeitreihen-CV |
| `tests/` | 9 | Unit-Tests: Suche gegen erschöpfendes Minimax mit künstlicher Bewertung, Regeln, Fallback, Versionierung, API |
| `src/draft_engine/rules.py` | 3, 9 | Konfigurierbare Ban/Pick-Zugreihenfolge (Annahme, nicht verifiziert - siehe unten) |
| `src/draft_engine/state.py` | 9 | Draft-Zustand |
| `src/draft_engine/analysis.py` | 11 | Ergebnis-Screen: Gewinnchance, Matchup-Matrix, Impact, Synergien (aus dem Modell abgeleitet) |
| `src/serve.py` | 12 | Runner für Railway: Web-App + Hintergrund-Job (crawlen, neu trainieren, Modell hot-laden) |
| `src/maps.py` | 11 | Map-Liste + Bilder (Brawlify), gefiltert auf die 3v3-Ranked-Modi |
| `src/draft_engine/search.py` | 1, 9 | Minimax + Alpha-Beta + Top-N-Pruning, iterative Vertiefung mit Zeitbudget, gebatchte Modellaufrufe: Denial-Bans und Lookahead-Picks |
| `src/predictor.py`, `src/api.py` | 10 | FastAPI-Backend (inkl. StaticFiles-Mount fürs Frontend), lädt Modell einmal beim Start |
| `web/` | 11 | Vanilla-JS-Frontend: Map-Auswahl, State-Machine-Draft-UI, Timeline, Icon-Grid mit Empfehlungs-Rahmen, Keyboard-Shortcuts |
| `src/synthetic.py` | - | Synthetischer Datensatz + End-to-End-Smoke-Test |

Abschnitt 0 des Specs legt den Stack verbindlich fest (Python/FastAPI/Uvicorn
Backend, Vanilla-HTML/CSS/JS-Frontend ohne Build-Step, ein Service via
`StaticFiles`, SQLite) - entspricht 1:1 dem, was hier gebaut ist.

## Offene, dokumentierte Annahmen (vor echtem Draft-Einsatz verifizieren)

- **Ban/Pick-Reihenfolge** (`src/draft_engine/rules.py`, `DraftRules`):
  **Bans** (bestätigt von Kian aus dem Spiel): erst bannt das ganze eigene Team
  (A) alle 3, dann das andere (B) alle 3 - A-A-A-B-B-B. **Picks** (bestätigt,
  entgegen dem Spec): **1-2-2-1** - das First-Pick-Team pickt 1, das andere 2,
  das erste 2, das andere 1. Wer First Pick hat, ist pro Match wählbar (Schalter
  auf der Startseite). Noch nicht am Live-Client geprüft: ob das Gegnerteam auch
  dann zuerst bannt, wenn es First Pick hat - bisher bannt immer Team A zuerst.
  Alles steht nur in `DraftRules` (`ban_first`, `pick_first`, `pick_pattern`),
  der Rest des Codes liest die Reihenfolge nur darüber aus.
- **Seitenunabhängiges Modell**: Im Battlelog hat die Reihenfolge der beiden Teams
  keine Bedeutung, deshalb wird jeder Kampf beim Training auch gespiegelt
  verwendet (Teams getauscht, Sieger gedreht). Ohne das lernte das Modell
  "Seite A" und "Seite B" getrennt, und dieselbe Aufstellung bekam je nach Seite
  eine andere Gewinnchance (im Test 0,35 gegen 0,50 statt 0,65). Der Smoke-Test
  prüft jetzt P(A schlägt B) + P(B schlägt A) = 1.
- **`min_rank`/Ranked-`type`-Wert**: **verifiziert** (2026-09-19), siehe "Was die ersten
  echten Daten gezeigt haben". `MIN_RANK` = 12 ist der per CV gemessene Standard, nicht
  Legendary III (= 18): so hohe Lobbys gibt es kaum (der Mittelwert der sechs Spieler
  liegt fast nie bei ≥ 16).
- **Checkliste, Stand 2026-09-19** - was gemessen und was nur angenommen ist:

  | Punkt | Stand |
  |---|---|
  | Ranked-`type`/`mode` im Battlelog | gemessen: `soloRanked`, sechs 3v3-Modi |
  | Zahl → Liga (`min_rank`) | gemessen über `GET /players/{tag}` (`rankedRank` + `rankedRankName`): 7-9 Gold, 10-12 Diamond, 13-15 Mythic, 16-18 Legendary (**III = 18**), 19-21 Masters, 22 Pro; der Wert im Battlelog ist derselbe (61 von 115 Spielern exakt gleich, der Rest ±1 wegen späterem Auf-/Abstieg) |
  | Ban-/Pick-Reihenfolge | **nicht per API prüfbar** (das Battlelog enthält keine Bans, keine Reihenfolge, nur `mode/type/result/duration/starPlayer/teams`); 3 Bans je Team als Blöcke und 1-2-2-1 stammen von Kian aus dem Spiel; offen: bannt das Gegnerteam bei eigenem First Pick zuerst? |
  | `weight_step` | gemessen: **wirkungslos** (mittlerer CV-Log-Loss 0,7064-0,7066 über alle Stufen); vorher wirkten Gewichte als heimliches größeres `C`, jetzt sind sie auf Mittelwert 1 normiert |
  | `C` (Regularisierung) | gemessen: **entscheidend**. sklearn-Standard `C=1` überfittet stark (Log-Loss 0,95 unkalibriert, 0,70 mit Platt; Raten = 0,693), `C=0,01` gibt 0,691. Das beste `C` wächst mit den Daten, deshalb wählt jedes Training es neu per CV |
  | Kalibrierung | gemessen: Platt auf einem 15-%-Slice (~330 Kämpfe) verschlechtert stark regularisierte Modelle (+0,002 bis +0,005); erst ab 1 000 Kämpfen im Slice aktiv. Ist sie aus, wird der Slice nicht mehr abgeschnitten (sonst gingen 15 % der Trainingsdaten verloren) |
  | Zeitbasierter Split | ja: 80/20 nach `battle_time`, dazu Zeitreihen-CV (`src/tune.py`); berichtet werden AUC, Brier, Log-Loss und Kalibrierungs-Bins, nicht nur Accuracy |
  | Denial-Bans / Lookahead-Picks | nachvollzogen (Details unten) und per Unit-Test gegen erschöpfendes Minimax abgesichert |
- **Brawler-Roster**: `src/brawlers.py` fällt ohne `data/brawlers.json` auf
  eine kurze, bewusst unvollständige Seed-Liste (ohne Bilder) zurück -
  `python -m src.brawlers fetch` ausführen. **Ohne `BS_API_TOKEN`** kommt die
  Liste von Brawlify, mit großgeschriebenen Namen ("Shelly" -> "SHELLY"), weil
  die offizielle API/das Battlelog so schreibt. Das ist unverifiziert: stimmt
  die Schreibweise nicht mit den gecrawlten Daten überein, würde das Modell
  *keinen* Brawler erkennen. `DraftPredictor` warnt deshalb beim Start, wenn
  Pool-Brawler im Training nie vorkamen. Sobald du einen Token hast, `fetch`
  nochmal laufen lassen (offizielle Namen, Bilder weiter per ID von Brawlify).
  Brandneue Brawler haben bei Brawlify manchmal noch keine Bilddatei (404);
  `fetch` erkennt das, die UI zeigt dann Initialen, ein späteres `fetch` holt sie.
- **Maps** (`src/maps.py`): Die offizielle API hat keinen Endpoint für Map-Listen,
  die Liste kommt von Brawlify. Welche Modi im Ranked rotieren
  (`RANKED_MODES`), ist eine Annahme - zu viele Modi machen nur die Liste
  länger (es gibt Modus-Filter), ein fehlender Modus lässt sich als Freitext
  eintippen. Dazu mischt `/maps` automatisch alle Map-Namen ein, die in
  gecrawlten Ranked-Kämpfen vorkommen, denn nur deren Schreibweise kennt das
  Modell sicher (das Modell sieht nur den Namen, nicht den Modus).
- **Map-Rotation**: `GET /events/rotation` ist **nicht** die Ranked-Rotation. Gemessen
  gegen die echten Kämpfe: von 16 Events dort tauchen nur 5 der 26 tatsächlich in Ranked
  gespielten Maps auf, Hot Zone (ein Ranked-Modus) fehlt ganz - es ist die Trophäen-
  Rotation samt Showdown & Co. Den Ranked-Pool liefern deshalb die Kämpfe selbst
  (`db.active_ranked_maps`, siehe oben). Ein Ranked-Leaderboard gibt es ebenfalls nicht:
  `GET /rankings/{global,US,CH}/powerplay/seasons[/latest|/<id>]` antwortet überall mit 404
  (Kontrollabfragen desselben Keys: 200), das Power-Play-Ranking ist abgeschafft.
- **Such-Tiefe/Zeitbudget** (`DraftPredictor(depth=4, top_n=16, inner_top_n=8,
  time_budget=1.0)`, Budget per Umgebungsvariable `DRAFT_TIME_BUDGET`): die
  Suche vertieft iterativ (1, 2, 3, ... Halbzüge) und antwortet mit der letzten
  fertigen Tiefe. Gemessen mit 109 Brawlern: ein Suchlauf mit fester Tiefe 4
  brauchte 2 s beim ersten Ban, 8 s mitten in den Bans und **145 s** beim ersten
  Pick; mit Zeitbudget sind es ≤ ~1 s, dafür nur 2 Halbzüge (eigener Pick +
  beste Antwort des Gegners) beim ersten Pick und 4 beim ersten Ban. Mehr
  Budget = tiefer, aber träger. `python -m src.synthetic run` gibt die
  Latenzen pro Draft-Phase aus.
- **Kalibrierung** (`--calibrate` in `python -m src.model train`, Default
  `sigmoid`): Abschnitt 8 nennt Platt-Scaling und Isotonic Regression als
  Optionen. Isotonic ist auf dem synthetischen Smoke-Test-Datensatz
  (~480 Kalibrierungs-Zeilen) tatsächlich problematisch gewesen - hat ein
  klar gelerntes Signal (Counter-Pick-Erkennung) auf praktisch nicht mehr
  unterscheidbare Werte zusammengestaucht, weil die non-parametrische
  Treppenfunktion bei wenig Daten übergeneralisiert. Deshalb Default
  `sigmoid` (parametrisch, robuster bei wenig Daten). Auf echten Daten hat sich
  außerdem gezeigt, dass auch Platt auf einem kleinen Slice schadet - beide
  Methoden sind daher erst ab 1 000 Kämpfen im Kalibrier-Slice aktiv
  (`_MIN_CALIBRATION_SAMPLES` in `src/model.py`).
- **Denial-Bans und Lookahead-Picks - nachvollzogen** (echtes Modell, Map Hard Rock
  Mine, Gegner hat First Pick): Der beste erste Ban ist **EMZ**, der stärkste
  Eröffnungs-Pick des Gegners (Platz 1 von 108); sein Ban senkt dessen bestes Opening
  von 53,9 % auf 52,7 %, der Ban seines unnützesten Openers ändert nichts. Bei den
  Picks stimmt die Suche auf einem auf 9 Brawler eingeschränkten Pool mit einem
  erschöpfenden Minimax über alle 60 480 Reihenfolgen **exakt** überein (max.
  Abweichung 0,0): BO ist der beste Erst-Pick, sein naiver Wert von 52,7 % fällt nach der
  besten Gegenantwort (8-BIT + BELLE) auf 50,3 %, und SIRIUS wirkt naiv gut (Platz 4),
  ist nach der Antwort aber der schlechteste Pick. Dabei fiel ein Fehler auf, der behoben
  ist: die Bewertung beim Suchabbruch vor den Picks füllte die Teams strikt A/B/A/B mit
  A zuerst statt in der echten Pick-Reihenfolge - hat der Gegner First Pick, war damit
  der Wert eines Bans falsch (`tests/test_search.py`).

## Was die ersten echten Daten gezeigt haben (2026-09-19)

Ein Test-Crawl mit dem echten Key (ca. 3 000 Ranked-Kämpfe) hat mehrere Annahmen
aus dem Spec korrigiert - deshalb ist die Pipeline jetzt daran ausgerichtet:

- **Ranked-Draft = Typ `soloRanked`**, nur die Modi Gem Grab, Brawl Ball,
  Knockout, Hot Zone, Heist, Bounty (kein Wipeout). Der Typ `ranked` ist normales
  Trophäen-Spiel in allen Modi inkl. Showdown - **nicht** der Draft. Standard
  überall: `soloRanked`.
- **`battle.rank` gibt es bei Ranked-Kämpfen nicht** (bei `ranked` ist es die
  Showdown-Platzierung). Der Rang steckt bei jedem Spieler in `brawler.trophies`
  (Werte 4..18 gesehen; bei Trophäen-Kämpfen sind es echte Trophäen). Als
  `rank` einer Partie gilt der Mittelwert der 6 Spieler (`db.battle_rank`).
  Die Zahl→Liga-Zuordnung ist über das Spielerprofil (`GET /players/{tag}`:
  `rankedRank` + `rankedRankName`) **belegt**: 7-9 Gold, 10-12 Diamond, 13-15 Mythic,
  16-18 Legendary (**III = 18**), 19-21 Masters, 22 Pro. Das Profil enthält außerdem
  `rankedElo` (ein feineres, kontinuierliches Skill-Maß, z. B. Diamond III = 4 030-4 260,
  Legendary III = 7 500-7 700) - das ließe sich später als Gewicht nutzen.
- **Namen**: Alle 108 offiziellen Brawler-Namen entsprechen den großgeschriebenen
  Brawlify-Namen. Brawlify hat einen Extra-Eintrag (BUZZ LIGHTYEAR), den es in der
  offiziellen API nicht gibt. Bei Maps weicht Brawlify ab ("Belles Rock" statt
  "Belle's Rock") - `/maps` nimmt deshalb die Schreibweise des Spiels.
- **Modell auf echten Daten - ehrlich gemessen.** Der erste Wert (AUC 0,57-0,58 auf ~2 800
  Kämpfen) kam von einem einzelnen Zeit-Split und war Glück; die Zeitreihen-CV
  (`python -m src.tune`, 4-5 Folds, immer vorwärts in der Zeit) ergibt **AUC 0,53-0,54 und
  Log-Loss 0,691** (Raten: 0,693) - bei ~2 800 Kämpfen trägt das Modell kaum Signal. Die
  einfachen Brawler-Siegquoten (`src/baseline.py`) liegen bei ~1 500 Kämpfen mit Log-Loss
  0,685 / AUC 0,57 sogar leicht darüber: Die tausenden Paar-/Matchup-Features fügen
  bei so wenig Daten nur Rauschen hinzu. Deshalb antwortet jetzt zunächst die
  Rohstatistik ("Preliminary"), bis das trainierte Modell sie in der CV schlägt. Grober
  Bedarf: ~100 Brawler mit je ±1 % Winrate-Genauigkeit brauchen ~2 500 Spiele pro Brawler,
  also **etwa 40 000+ Kämpfe** für ein Brawler-Modell und deutlich mehr für Synergien und
  Counter - deshalb muss der Crawler dauerhaft laufen (Railway).
- Gefundene und behobene Fehler: `b.map` in `build_training_table` war die
  Pandas-Methode statt der Map (das Modell hat die Map nie benutzt), die
  Trainingstabelle war quadratisch langsam (jetzt linear, 85× schneller bei 2 400
  Zeilen), und ein per `python -m src.model train` trainiertes Modell ließ sich vom
  Server nicht laden (Klasse als `__main__` gespeichert).

## Auf Railway hosten (erst die Web-App, dann der API-Key)

Die Reihenfolge ist so gewählt, dass der API-Key nur **einmal** angelegt werden muss:

1. Repo nach GitHub pushen. In Railway: *New Project → Deploy from GitHub repo →
   Draftion*. Railway baut das `Dockerfile` selbst. Es ist noch kein Key nötig:
   Roster und Maps kommen aus `deploy/seed/` (ursprünglich von Brawlify), dazu
   ein Demo-Modell - die Seite zeigt deshalb das gelbe "Demo model"-Banner.
2. *Settings → Networking → Generate Domain* für die öffentliche URL, optional
   Healthcheck-Pfad `/health`. **Für eine reine Demo (z. B. als Bewerbungs-Link) reichen
   Schritt 1 und 2**: kein Key, kein Volume, keine Variablen. Die Seite läuft dann mit dem
   Demo-Modell auf synthetischen Daten und sagt das oben selbst. Der Start aus einem
   leeren Ordner mit nur `src/`, `web/`, `deploy/` (also wie im Image) wurde lokal nachgestellt
   und funktioniert; das Image selbst ist nicht gebaut.
3. Key anlegen auf developer.brawlstars.com, bei "Allowed IP" **`45.79.218.79`**
   eintragen (der Proxy von RoyaleAPI). Der Key gilt dadurch von überall, auch von
   deinem PC - ein zweiter Key für deine Heim-IP ist nicht nötig.
4. Denselben Key an zwei Stellen eintragen: in Railway unter *Variables* und
   lokal in `.env`. Beide brauchen zusätzlich die Basis-URL des Proxys:
   `BS_API_TOKEN=...` und `BS_API_BASE_URL=https://bsproxy.royaleapi.dev/v1`.
   Der Proxy sieht den Key; ohne `BS_API_BASE_URL` gehen Anfragen direkt an die
   offizielle API und scheitern mit 403.
5. Crawlen und trainieren auf Railway: Sobald `BS_API_TOKEN` gesetzt ist, startet
   `python -m src.serve` (der Startbefehl im `Dockerfile`) zusätzlich zur Web-App
   einen Hintergrund-Job: alle 2 h crawlen, danach - ab 1 000 brauchbaren Kämpfen -
   einmal pro Intervall neu trainieren (aber nur, wenn seit dem letzten Modell
   mindestens 100 neue Kämpfe dazukamen) und jedes Ergebnis als **neue Modell-Version**
   unter `data/models/` ablegen; der Server lädt die neueste ohne Neustart. Nach jedem
   Crawl wählt der Server sein Modell neu (die Rohstatistik wächst mit den Daten). Dafür
   in Railway ein **Volume auf `/app/data`** anlegen (sonst ist die Datenbank nach jedem
   Deploy weg). Es läuft im selben Service, weil ein Volume nur an einen Service passt.
   Einstellbar per Variablen (alle optional, Liste in `src/config.py`): `RANKED_TYPE`
   (soloRanked), `MIN_RANK` (12), `CRAWL_INTERVAL_HOURS` (2), `CRAWL_MAX_PLAYERS` (1500),
   `RETRAIN_INTERVAL_HOURS` (24; `168` = wöchentlich), `RETRAIN_MIN_NEW_BATTLES` (100),
   `KEEP_MODELS` (5), `MIN_TRAIN_ROWS` (1000), `TRAIN_BACKEND` (logreg),
   `DRAFTION_WORKER=0` schaltet den Job ab. Ein zusätzlicher Cron-Service ist nicht nötig
   (und ginge nicht: ein Volume passt nur an einen Service); wer trotzdem extern trainiert
   (`python -m src.model train`), schreibt in dasselbe Verzeichnis und wird automatisch
   geladen. Lokal getestet (Crawl, Training, Versionen, Hot-Reload); auf Railway selbst
   noch nie gelaufen.

Technisches: `data/` ist im Repo ignoriert, und ein Railway-Volume auf `/app/data`
würde alles verdecken, was im Image dort liegt. Deshalb liegen Startdaten in
`deploy/seed/`, und der Startbefehl im `Dockerfile` kopiert fehlende Dateien nach
`data/` (`cp -n`: überschreibt nie, ein auf dem Volume trainiertes Modell bleibt
erhalten). Der Docker-Build selbst ist lokal nicht getestet (kein Docker
installiert); die Startlogik wurde in einem leeren Ordner nachgestellt.

## Roadmap

1. **Datensammlung** (Crawler) - bis genug Ranked-Matches in `data/brawl.db`
   liegen (ein paar Tausend als grober Richtwert für erste Experimente).
2. **`src/extract.py`** - `battle_type`/`min_rank` gegen echte Daten bestätigen.
3. **Explorative Statistik** (`src/stats.py`) - Sanity-Check, nicht die
   finale Empfehlungsquelle.
4. **Modell** (`src/model.py`) - fertig, sobald genug echte Daten da sind:
   `python -m src.model train --min-rank <bestätigt>`.
5. **Draft-Suche** (`src/draft_engine/`) - fertig; Ban/Pick-Reihenfolge nach
   Live-Client-Verifikation ggf. in `rules.py` anpassen.
6. **Backend** (`src/api.py`) - fertig: `/recommend`, `/brawlers`, `/maps`.
7. **Web-UI** (`web/`) - fertig: Map, Klick-Interface, Live-Empfehlungspanel.

Schritte 4-7 sind mit synthetischen Daten durchgetestet (`python -m
src.synthetic run`); scharf werden sie erst mit einem Modell aus echten
Ranked-Daten aus Schritt 1.
#   D r a f t i o n  
 #   D r a f t i o n  
 