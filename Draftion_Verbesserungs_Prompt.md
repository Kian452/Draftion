# Draftion – Verbesserungs-Prompt für Claude Code

Das Projekt in diesem Ordner ist bereits vollständig gebaut (Backend,
Frontend, Crawler, Modell) auf Basis einer früheren Spezifikation. Das
Folgende sind gezielte Verbesserungen und offene Verifikationen am
**bestehenden** Code – kein Neubau. Lies zuerst den vorhandenen Code, bevor
du etwas änderst, und baue nichts neu, was schon funktioniert.

Arbeite die beiden Prioritätsstufen in Reihenfolge ab – Stufe 1 liefert
Informationen, die Entscheidungen in Stufe 2 direkt beeinflussen.

## Priorität 1 – Zuerst empirisch verifizieren

**1. Möglicher direkter Ranked-Leaderboard-Endpoint testen.**
Rufe mit dem echten API-Key `GET /rankings/global/powerplay/seasons` auf.
- Ist die neueste zurückgegebene Season aktuell/laufend: das ist eine echte
  Ranked-Bestenliste. Passe `seed_tags()` im Crawler so an, dass es zuerst
  versucht, Tags direkt über `GET /rankings/{country}/powerplay/seasons/{seasonId}`
  (seasonId `latest` verwenden) zu holen, und nur bei leerer/fehlender
  Antwort auf die bisherige Trophäen-Rangliste zurückfällt. Das ersetzt
  unsicheres Proxy-Seeding durch echte Ranked-Tags.
- Ist die neueste Season alt/inaktiv (z.B. schon Jahre her): totes
  Legacy-Feature, ignorieren, beim bisherigen Trophäen-Seed bleiben.

**2. Checkliste wirklich verifiziert, nicht nur angenommen?**
Bestätige für jeden Punkt explizit, ob er empirisch geprüft wurde oder nur
als Platzhalter/Annahme im Code steht – bei Bedarf jetzt nachholen:
- Ranked `type`/`mode`-Wert im Battlelog bestätigt?
- `min_rank` (numerischer Wert für Legendary III) bestätigt?
- Exakte Ban/Pick-Sequenz (3 Bans/Team, dann alternierend – Reihenfolge und
  Gleichzeitigkeit) gegen den echten Live-Client verifiziert?
- `weight_step`/`min_samples` anhand der tatsächlich gesammelten Datenmenge
  kalibriert, nicht nur Default-Werte?
- Modell-Kalibrierung mit zeitbasiertem Split geprüft (nicht nur Accuracy)?
- Denial-Bans und Lookahead-Picks an mindestens einem konkreten
  Beispiel-Draft manuell nachvollzogen und für korrekt befunden?

## Priorität 2 – Konkrete Verbesserungen

**3. Map-Rotation statt statischer Liste.**
`GET /events/rotation` einbinden. Die im Frontend wählbaren Maps sollen der
aktuellen Ranked-Rotation entsprechen, nicht eine feste Liste aller jemals
existierenden Maps sein.

**4. Modell-Refresh-Strategie.**
Periodischen Retraining-Job einrichten (z.B. wöchentlich, als eigener
Cron-Service oder Erweiterung des bestehenden Crawler-Cron-Setups), der ein
neues, versioniertes Modell-File schreibt. Backend lädt beim Start die
jeweils neueste Version, statt dauerhaft an einem einmal geladenen Modell
festzuhalten.

**5. Fallback bei dünner Datenlage.**
Solange das trainierte Modell noch wenig Daten hat: auf die bereits
vorhandenen Rohstatistiken (`brawler_winrate()` aus `stats.py`)
zurückfallen, im Frontend klar als "geringe Konfidenz / vorläufig"
gekennzeichnet – statt gar keine Empfehlung zu zeigen.

**6. Such-Logik unabhängig vom Modell testen.**
Unit-Tests für die Minimax-Suche (Denial-Bans, Lookahead-Picks) mit einer
künstlichen, festen Bewertungsfunktion statt dem echten trainierten Modell
schreiben. Trennt "ist der Suchalgorithmus korrekt" von "ist das Modell
gut" – lässt sich sofort testen, unabhängig vom Datenstand.

---
Bei jedem Punkt: kurz zusammenfassen, was geändert wurde und was das
Testergebnis aus Priorität 1 war, bevor mit dem nächsten Punkt weitergemacht
wird.
