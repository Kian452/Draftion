// Draftion - frontend.
//
// Whose turn it is comes from the ban/pick sequence served by GET /rules
// (defined only in src/draft_engine/rules.py) and the number of actions so
// far, so the bar updates instantly on every click. Only the recommendations
// need a /recommend round trip (up to about a second), and they never block
// you from clicking: responses that arrive after the draft has moved on are
// dropped. Once the draft is complete, /analyze produces the result view.
//
// Team A is always your team, team B the enemy.

const REC_COUNT = 8;       // suggestions listed in the right-hand panel
const GRID_HIGHLIGHT = 3;  // how many of them also get a ring on their grid portrait

const state = {
  map: null,          // chosen map's name, null = no map chosen
  mapMode: null,      // its game mode, if known (display only - the model only sees the name)
  maps: [],           // [{name, mode, image, active}] - active = in the current Ranked pool
  showAllMaps: false, // home screen: also list maps that are not in the current Ranked pool
  mapFilter: null,    // game-mode chip currently active on the home screen, null = all
  confidence: null,   // /health model_confidence: demo | low | early | ok
  modelKind: null,    // /health model_kind: "trained" | "raw-stats" (the preliminary fallback)
  brawlers: [],       // [{name, id, image}]
  byName: new Map(),
  // The pick order depends on which team has first pick, which is decided per
  // match - so both variants are loaded up front and the home screen picks one.
  rulesByFirstPick: {},   // { A: {turns, bansPerTeam, picksPerTeam}, B: {...} }
  firstPick: "A",         // "A" = your team, "B" = the enemy team
  bans_a: [], bans_b: [], picks_a: [], picks_b: [],
  recommendations: [], // last /recommend result for the *current* draft state, best first
  history: [],         // stack of {key, name} for Undo
  loading: false,
  error: null,
  requestSeq: 0,       // lets us drop responses for a draft state that no longer exists
  analysis: null,      // /analyze result for the completed draft
  analysisLoading: false,
  analysisError: null,
  analysisSeq: 0,
};

const $ = (id) => document.getElementById(id);
const el = {
  mapScreen: $("map-screen"),
  mapSearch: $("map-search"),
  mapGrid: $("map-grid"),
  modeChips: $("mode-chips"),
  mapScope: $("map-scope"),
  mapHint: $("map-hint"),
  skipMapBtn: $("skip-map-btn"),
  firstPickSwitch: $("first-pick-switch"),

  draftScreen: $("draft-screen"),
  demoBanner: $("demo-banner"),
  teamA: $("team-a-block"),
  teamB: $("team-b-block"),
  phasePill: $("phase-pill"),
  turnText: $("turn-text"),
  mapName: $("map-name"),
  firstPickText: $("first-pick-text"),
  newDraftBtn: $("new-draft-btn"),
  undoBtn: $("undo-btn"),

  mainGrid: $("main-grid"),
  brawlerSearch: $("brawler-search"),
  poolCount: $("pool-count"),
  brawlerGrid: $("brawler-grid"),
  recSubtitle: $("rec-subtitle"),
  recList: $("rec-list"),
  recStatus: $("rec-status"),

  resultPanel: $("result-panel"),
  winBar: $("win-bar"),
  winAText: $("win-a-text"),
  winBText: $("win-b-text"),
  winVerdict: $("win-verdict"),
  resultBody: $("result-body"),
  resultStatus: $("result-status"),
};

let tiles = new Map(); // brawler name -> its grid tile (built once, updated in place)

// ---------- Helpers ----------

async function fetchJSON(path, opts) {
  const res = await fetch(path, opts);
  if (!res.ok) throw new Error(`${path} -> ${res.status}`);
  return res.json();
}

function esc(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

const teamName = (team) => (team === "A" ? "Your team" : "Enemy team");
const pct = (p) => `${Math.round(p * 100)}%`;
const pts = (v) => `${v >= 0 ? "+" : "−"}${Math.abs(v * 100).toFixed(1)}`;

// Fallback avatar (initials on a per-name colour) for brawlers/maps without a
// portrait, or whose image fails to load.
function nameColor(name) {
  let hash = 0;
  for (let i = 0; i < name.length; i++) hash = (hash * 31 + name.charCodeAt(i)) >>> 0;
  return `hsl(${hash % 360}, 36%, 34%)`;
}
function initials(name) {
  const parts = name.trim().split(/\s+/).filter(Boolean);
  if (parts.length >= 2) return (parts[0][0] + parts[1][0]).toUpperCase();
  return name.slice(0, 2).toUpperCase();
}
function initialsHTML(name) {
  return `<span class="initials" style="background:${nameColor(name)}">${esc(initials(name))}</span>`;
}
function portraitHTML(name) {
  const b = state.byName.get(name);
  if (b && b.image) {
    return `<img src="${esc(b.image)}" alt="" draggable="false" decoding="async" data-name="${esc(name)}">`;
  }
  return initialsHTML(name);
}
const portrait = (name, size = "") => `<span class="portrait ${size}">${portraitHTML(name)}</span>`;

// <img> error events don't bubble, hence the capture phase.
document.addEventListener("error", (e) => {
  const img = e.target;
  if (img instanceof HTMLImageElement && img.dataset.name) img.outerHTML = initialsHTML(img.dataset.name);
}, true);

function usedBrawlers() {
  return new Set([...state.bans_a, ...state.bans_b, ...state.picks_a, ...state.picks_b]);
}
function activeRules() {
  return state.rulesByFirstPick[state.firstPick];
}
function actionCount() {
  return state.bans_a.length + state.bans_b.length + state.picks_a.length + state.picks_b.length;
}
function getTurn() {
  return activeRules().turns[actionCount()] || null; // {phase: "ban"|"pick", team: "A"|"B"} or null once the draft is complete
}

// ---------- Startup ----------

// How far to trust the numbers is decided by the backend (/health model_confidence,
// see src/model_select.py): demo = synthetic data, low = raw brawler win rates only
// (not enough data for a trained model yet), early = a trained model on fewer than
// ~40,000 battles (measured: AUC 0.58 at ~18k, only a small edge over a coin flip),
// ok = no warning.
const CONFIDENCE_TAG = {
  demo: "Demo data",
  low: "Preliminary · low confidence",
  early: "Early model",
};

// Say so whenever the recommendations are not yet trustworthy.
function modelNotice(health) {
  const n = (health.model_training_battles || 0).toLocaleString("en-US");
  switch (health.model_confidence) {
    case "demo":
      return "<strong>Demo model.</strong> The recommendations come from a model trained on synthetic data, not real matches, so they are illustrative. The real-data pipeline (crawler → training → evaluation) is part of the project – see the README.";
    case "low":
      return `<strong>Preliminary – low confidence.</strong> Based only on raw brawler win rates from ${n} real battles, with no counter, synergy or map modelling yet. Expect win chances close to 50%; a trained model takes over once enough battles are collected.`;
    case "early":
      return `<strong>Early model.</strong> Trained on ${n} real battles – the edge over a coin flip is still small, so the order of the top recommendations is noisy, and the map does not influence the numbers yet. It improves as more data is collected.`;
    default:
      return "";
  }
}

// Small pill repeated next to the numbers (recommendations, result) so the warning
// isn't only in a banner that scrolls out of mind.
function confidenceTag() {
  const label = CONFIDENCE_TAG[state.confidence];
  return label ? ` <span class="conf-tag conf-${state.confidence}">${label}</span>` : "";
}

async function init() {
  try {
    const [rulesA, rulesB, brawlers, maps, health] = await Promise.all([
      fetchJSON("/rules?first_pick=A"),
      fetchJSON("/rules?first_pick=B"),
      fetchJSON("/brawlers"),
      fetchJSON("/maps").catch(() => []),
      fetchJSON("/health").catch(() => ({})),
    ]);
    state.confidence = health.model_confidence || null;
    state.modelKind = health.model_kind || null;
    const notice = modelNotice(health);
    el.demoBanner.innerHTML = notice;
    el.demoBanner.classList.toggle("hidden", !notice);
    const asRules = (r) => ({ turns: r.turns, bansPerTeam: r.bans_per_team, picksPerTeam: r.picks_per_team });
    state.rulesByFirstPick = { A: asRules(rulesA), B: asRules(rulesB) };
    state.brawlers = brawlers;
    state.byName = new Map(brawlers.map((b) => [b.name, b]));
    state.maps = maps;
  } catch (e) {
    showMapHint("Could not reach the backend. Start it with: uvicorn src.api:app");
    return;
  }
  buildBrawlerGrid();
  renderMapChips();
  renderMapScope();
  renderMapGrid();
  if (state.maps.length === 0) {
    showMapHint("No maps loaded yet – run `python -m src.maps fetch`, or type a name and press Enter, or just start.");
  }
  el.mapSearch.focus();
}

function showMapHint(text) {
  el.mapHint.textContent = text;
  el.mapHint.classList.remove("hidden");
}

// ---------- Home screen ----------

// The maps on offer: the current Ranked pool by default (what /maps marks `active`),
// everything the backend knows once "show all" is switched on.
function visibleMaps() {
  return state.showAllMaps ? state.maps : state.maps.filter((m) => m.active !== false);
}

function filterMaps(query, from = visibleMaps()) {
  const q = query.trim().toLowerCase();
  return from.filter(
    (m) => (!state.mapFilter || m.mode === state.mapFilter) && (!q || m.name.toLowerCase().includes(q))
  );
}

// Switch between "current Ranked pool" and "all maps"; hidden when the two are the same.
function renderMapScope() {
  const active = state.maps.filter((m) => m.active !== false).length;
  const hidden = state.maps.length - active;
  el.mapScope.classList.toggle("hidden", hidden === 0);
  el.mapScope.textContent = state.showAllMaps
    ? `Showing all ${state.maps.length} maps · show current Ranked pool only (${active})`
    : `Current Ranked pool: ${active} maps · show all ${state.maps.length}`;
}

// One chip per game mode present in the list (plus "All"); hidden when there's nothing to filter.
function renderMapChips() {
  const modes = [...new Set(visibleMaps().map((m) => m.mode).filter(Boolean))];
  el.modeChips.innerHTML = modes.length < 2 ? "" : [null, ...modes]
    .map((mode) => `<button class="chip${state.mapFilter === mode ? " active" : ""}" data-mode="${mode ? esc(mode) : ""}">${mode ? esc(mode) : "All"}</button>`)
    .join("");
}

function renderMapGrid() {
  const shown = filterMaps(el.mapSearch.value);
  // Searching for a map outside the current Ranked pool: say where it went instead of showing nothing.
  const elsewhere = shown.length === 0 && el.mapSearch.value.trim() ? filterMaps(el.mapSearch.value, state.maps).length : 0;
  if (elsewhere) {
    el.mapGrid.innerHTML = `<p class="hint" style="grid-column: 1 / -1">No match in the current Ranked pool – ${elsewhere} matching map${elsewhere > 1 ? "s" : ""} in “show all”. Enter opens ${elsewhere > 1 ? "the first" : "it"} anyway.</p>`;
    return;
  }
  el.mapGrid.innerHTML = shown
    .map((m) => `
      <button class="tile map-tile" data-map="${esc(m.name)}" data-mode="${esc(m.mode || "")}">
        <span class="map-thumb">${m.image
          ? `<img src="${esc(m.image)}" alt="" draggable="false" decoding="async" loading="lazy" data-name="${esc(m.name)}">`
          : initialsHTML(m.name)}</span>
        <span class="tile-label">${esc(m.name)}</span>
        ${m.mode ? `<span class="tile-mode">${esc(m.mode)}</span>` : ""}
      </button>`)
    .join("");
}

function selectMap(name, mode) {
  state.map = name || null;
  state.mapMode = name ? mode || null : null;
  el.mapScreen.classList.add("hidden");
  el.draftScreen.classList.remove("hidden");
  renderAll();
  refreshRecommendation();
  el.brawlerSearch.focus();
}

// ---------- Draft bar ----------

function slotHTML(kind, name, active) {
  const cls = `slot ${kind}${name ? " filled" : ""}${active ? " active" : ""}`;
  if (!name) return `<div class="${cls}"><span class="slot-empty">${kind === "ban" ? "Ban" : "Pick"}</span></div>`;
  const label = kind === "pick" ? `<span class="slot-name">${esc(name)}</span>` : "";
  return `<div class="${cls}" title="${esc(name)}">${portrait(name)}${label}</div>`;
}

function teamBlockHTML(team, turn) {
  const onClock = !!turn && turn.team.toLowerCase() === team;
  const row = (kind, label, slots) => {
    const items = state[`${kind}s_${team}`];
    const activeIdx = onClock && turn.phase === kind ? items.length : -1;
    const cells = Array.from({ length: slots }, (_, i) => slotHTML(kind, items[i], i === activeIdx)).join("");
    return `<div><div class="row-label">${label}</div><div class="slot-row">${cells}</div></div>`;
  };
  return `
    <div class="team-title"><span class="team-name">${teamName(team.toUpperCase())}</span>${onClock ? '<span class="turn-tag">To move</span>' : ""}</div>
    <div class="team-rows">
      ${row("ban", "Bans", activeRules().bansPerTeam)}
      ${row("pick", "Picks", activeRules().picksPerTeam)}
    </div>`;
}

function renderDraftBar() {
  const turn = getTurn();
  el.teamA.innerHTML = teamBlockHTML("a", turn);
  el.teamB.innerHTML = teamBlockHTML("b", turn);
  el.teamA.classList.toggle("on-clock", !!turn && turn.team === "A");
  el.teamB.classList.toggle("on-clock", !!turn && turn.team === "B");

  el.mapName.textContent = state.map ? state.map + (state.mapMode ? ` · ${state.mapMode}` : "") : "None";
  el.firstPickText.textContent = teamName(state.firstPick);
  el.turnText.className = "turn-text";
  el.phasePill.className = "phase-pill";

  if (!turn) {
    el.phasePill.textContent = "Draft complete";
    el.turnText.textContent = "All bans and picks are in";
    return;
  }
  const inPhase = activeRules().turns.filter((t) => t.phase === turn.phase);
  const step = activeRules().turns.slice(0, actionCount()).filter((t) => t.phase === turn.phase).length + 1;
  el.phasePill.textContent = `${turn.phase === "ban" ? "Ban" : "Pick"} ${step} of ${inPhase.length}`;
  el.phasePill.classList.add(turn.phase);
  el.turnText.textContent = `${teamName(turn.team)} to ${turn.phase}`;
  el.turnText.classList.add(`team-${turn.team.toLowerCase()}`);
}

// ---------- Brawler grid + recommendations ----------

function buildBrawlerGrid() {
  el.brawlerGrid.innerHTML = state.brawlers
    .map((b) => `
      <button class="tile" data-name="${esc(b.name)}" title="${esc(b.name)}">
        ${portrait(b.name)}
        <span class="tile-label">${esc(b.name)}</span>
      </button>`)
    .join("");
  tiles = new Map([...el.brawlerGrid.children].map((t) => [t.dataset.name, t]));
}

// Updates the grid in place (rather than rebuilding it) so portraits don't
// reload or flicker on every click.
function updateBrawlerGrid() {
  const query = el.brawlerSearch.value.trim().toLowerCase();
  const used = usedBrawlers();
  const turn = getTurn();
  const highlighted = state.loading ? [] : state.recommendations.slice(0, GRID_HIGHLIGHT).map((r) => r.brawler);

  for (const [name, tile] of tiles) {
    tile.hidden = !!query && !name.toLowerCase().includes(query);
    tile.disabled = used.has(name) || !turn;
    tile.classList.toggle("top-pick", name === highlighted[0]);
    tile.classList.toggle("alt-pick", highlighted.slice(1).includes(name));
  }
  el.poolCount.textContent = `${tiles.size - used.size} of ${tiles.size} available`;
}

// The list is sorted best-first, so compare each option with #1. (The API's
// `delta` - vs. the position before the move - is negative for the best ban
// at the very start, which reads as if the top suggestion were bad.)
function vsBest(r, best) {
  const diff = (r.win_probability - best.win_probability) * 100;
  if (r === best) return { text: "Best option", cls: "" };
  if (Math.abs(diff) < 0.05) return { text: "Same as #1", cls: "" };
  return { text: `${diff.toFixed(1)} pts vs. #1`, cls: "neg" };
}

function formatDelta(delta) {
  return `${delta >= 0 ? "+" : ""}${(delta * 100).toFixed(1)} pts vs. before this move`;
}

function renderRecommendations() {
  const turn = getTurn();
  el.recList.innerHTML = "";
  el.recStatus.textContent = "";
  el.recSubtitle.innerHTML = turn
    ? `Estimated win chance for ${turn.team === "A" ? "your team" : "the enemy team"} after each ${turn.phase}${confidenceTag()}`
    : "";

  if (!turn) return;
  if (state.error) { el.recStatus.textContent = state.error; return; }
  if (state.loading) { el.recStatus.textContent = "Calculating…"; return; }

  const best = state.recommendations[0];
  el.recList.innerHTML = state.recommendations
    .map((r, i) => {
      const rel = vsBest(r, best);
      return `
      <li class="rec-item rank-${i + 1}" data-name="${esc(r.brawler)}" title="${esc(formatDelta(r.delta))}">
        <span class="rec-rank">${i + 1}</span>
        ${portrait(r.brawler)}
        <span class="rec-main">
          <span class="rec-name">${esc(r.brawler)}</span>
          <span class="rec-delta ${rel.cls}">${rel.text}</span>
        </span>
        <span class="rec-side">
          <span class="rec-prob">${(r.win_probability * 100).toFixed(1)}%</span>
          ${i === 0 ? '<span class="rec-enter">Enter ↵</span>' : ""}
        </span>
      </li>`;
    })
    .join("");
}

// ---------- Result view (draft complete) ----------

// Background for a matchup cell: green where your brawler is favoured, red where it isn't.
function heat(p) {
  const strength = Math.min(Math.abs(p - 0.5) * 3.2, 0.75);
  return p >= 0.5 ? `rgba(63, 178, 127, ${strength})` : `rgba(229, 100, 95, ${strength})`;
}

function matchupTable(r) {
  const head = r.team_b.map((y) => `<th scope="col"><div class="mx-head col">${portrait(y, "sm")}<span>${esc(y)}</span></div></th>`).join("");
  const rows = r.team_a
    .map((x, i) => `
      <tr>
        <th scope="row"><div class="mx-head">${portrait(x, "sm")}<span>${esc(x)}</span></div></th>
        ${r.matchups[i].map((p) => `<td style="background:${heat(p)}">${pct(p)}</td>`).join("")}
      </tr>`)
    .join("");
  return `<table class="matrix"><thead><tr><th class="corner"><span>Your team ↓</span><span>Enemy →</span></th>${head}</tr></thead><tbody>${rows}</tbody></table>`;
}

function matchupRow(m) {
  return `
    <li class="mu-row">
      <span class="pair">${portrait(m.x, "sm")}${portrait(m.y, "sm")}</span>
      <span class="mu-names"><b>${esc(m.x)}</b> vs ${esc(m.y)}</span>
      <span class="mu-pct ${m.p >= 0.5 ? "pos" : "neg"}">${pct(m.p)}</span>
    </li>`;
}

function impactRow(d, scale) {
  const width = Math.min(50, (Math.abs(d.value) / scale) * 50);
  const pos = d.value >= 0;
  return `
    <li class="impact-row">
      ${portrait(d.brawler, "sm")}
      <span class="impact-name">${esc(d.brawler)}</span>
      <span class="impact-bar"><i class="${pos ? "pos" : "neg"}" style="left:${pos ? 50 : 50 - width}%;width:${width}%"></i></span>
      <span class="impact-val ${pos ? "pos" : "neg"}">${pts(d.value)}</span>
    </li>`;
}

function synergyRow(s) {
  const pos = s.value >= 0;
  return `
    <li class="synergy-row">
      <span class="pair">${portrait(s.pair[0], "sm")}${portrait(s.pair[1], "sm")}</span>
      <span class="impact-name">${esc(s.pair[0])} + ${esc(s.pair[1])}</span>
      <span class="impact-val ${pos ? "pos" : "neg"}">${pts(s.value)}</span>
    </li>`;
}

// The preliminary raw-win-rate model has no counter or synergy terms at all, so those parts of
// the result would only show zeros - say so instead of presenting them as findings.
const isRawModel = () => state.modelKind === "raw-stats";

function teamCard(title, impact, synergy, scale) {
  const synergyBlock = isRawModel()
    ? `<p class="card-note">Not modelled yet – teammate synergies need a trained model on more battles.</p>`
    : `<p class="card-note">Teammate pairs: bonus beyond the two brawlers' individual strength (points).</p>
       <ul class="synergy-list">${synergy.map(synergyRow).join("")}</ul>`;
  return `
    <section class="card card-half">
      <h3>${title}</h3>
      <p class="card-note">Impact: how much each brawler changes its team's win chance versus the same draft without it (points).</p>
      <ul class="impact-list">${impact.map((d) => impactRow(d, scale)).join("")}</ul>
      <h4>Synergy</h4>
      ${synergyBlock}
    </section>`;
}

function renderResult() {
  const complete = !getTurn();
  el.mainGrid.classList.toggle("hidden", complete);
  el.resultPanel.classList.toggle("hidden", !complete);
  if (!complete) return;

  const r = state.analysis;
  el.resultStatus.textContent = state.analysisError || (state.analysisLoading ? "Analysing draft…" : "");
  if (!r) {
    el.winBar.style.gridTemplateColumns = "1fr 1fr";
    el.winAText.textContent = el.winBText.textContent = "–";
    el.winVerdict.textContent = "";
    el.resultBody.innerHTML = "";
    return;
  }

  const pa = r.win_probability_a;
  const a = Math.round(pa * 100);
  el.winBar.style.gridTemplateColumns = `minmax(72px, ${pa}fr) minmax(72px, ${1 - pa}fr)`;
  el.winAText.textContent = `${a}%`;
  el.winBText.textContent = `${100 - a}%`;
  const lead = Math.abs(pa - 0.5) * 200; // percentage-point gap between the teams
  el.winVerdict.innerHTML = (lead < 2
    ? "The model sees an even draft."
    : `The model favours ${pa >= 0.5 ? "your team" : "the enemy team"} by ${lead.toFixed(0)} points.`) + confidenceTag();

  const pairs = r.team_a.flatMap((x, i) => r.team_b.map((y, j) => ({ x, y, p: r.matchups[i][j] })));
  const best = [...pairs].sort((m, n) => n.p - m.p).slice(0, 3);
  const toughest = [...pairs].sort((m, n) => m.p - n.p).slice(0, 3);
  const scale = Math.max(0.02, ...[...r.impact_a, ...r.impact_b].map((d) => Math.abs(d.value)));

  el.resultBody.innerHTML = `
    <section class="card card-matrix">
      <h3>Who beats whom</h3>
      <p class="card-note">Chance that your brawler beats the enemy brawler 1v1, according to the model. ${isRawModel()
        ? "The preliminary model only knows each brawler's general win rate, so this shows strength differences, not counters."
        : "Includes general strength and counter effects."}</p>
      ${matchupTable(r)}
    </section>
    <section class="card card-matchups">
      <h3>Best matchups for your team</h3>
      <ol class="mu-list">${best.map(matchupRow).join("")}</ol>
      <h3 class="second">Toughest matchups</h3>
      <ol class="mu-list">${toughest.map(matchupRow).join("")}</ol>
    </section>
    ${teamCard("Your team", r.impact_a, r.synergy_a, scale)}
    ${teamCard("Enemy team", r.impact_b, r.synergy_b, scale)}`;
}

function renderAll() {
  renderDraftBar();
  updateBrawlerGrid();
  renderRecommendations();
  renderResult();
}

// ---------- Server round trips ----------

async function refreshRecommendation() {
  const seq = ++state.requestSeq;
  state.recommendations = [];
  state.error = null;
  state.loading = !!getTurn();
  if (state.loading) {
    // back in the draft (e.g. after Undo): drop any result view and in-flight analysis
    state.analysisSeq++;
    state.analysis = null;
    state.analysisLoading = false;
    state.analysisError = null;
  }
  renderAll();
  if (!state.loading) { refreshAnalysis(); return; } // draft complete - show the result instead

  try {
    const data = await fetchJSON("/recommend", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        map: state.map,
        bans_a: state.bans_a, bans_b: state.bans_b,
        picks_a: state.picks_a, picks_b: state.picks_b,
        num_alternatives: REC_COUNT - 1,
        first_pick: state.firstPick,
      }),
    });
    if (seq !== state.requestSeq) return; // the draft moved on while this was computing
    state.recommendations = data.recommendations || [];
  } catch (e) {
    if (seq !== state.requestSeq) return;
    state.error = "Backend unreachable, or no trained model yet (see README).";
  }
  state.loading = false;
  renderAll();
}

async function refreshAnalysis() {
  const seq = ++state.analysisSeq;
  state.analysis = null;
  state.analysisError = null;
  state.analysisLoading = true;
  renderAll();
  try {
    const data = await fetchJSON("/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ map: state.map, picks_a: state.picks_a, picks_b: state.picks_b }),
    });
    if (seq !== state.analysisSeq) return;
    state.analysis = data;
  } catch (e) {
    if (seq !== state.analysisSeq) return;
    state.analysisError = "Backend unreachable, or no trained model yet (see README).";
  }
  state.analysisLoading = false;
  renderAll();
}

// ---------- Actions ----------

function applyAction(name) {
  const turn = getTurn();
  if (!turn || !name || usedBrawlers().has(name)) return;
  const key = `${turn.phase}s_${turn.team.toLowerCase()}`;
  state[key].push(name);
  state.history.push({ key, name });
  el.brawlerSearch.value = "";
  el.brawlerSearch.focus();
  refreshRecommendation();
}

function undo() {
  const last = state.history.pop();
  if (!last) return;
  const arr = state[last.key];
  const idx = arr.lastIndexOf(last.name);
  if (idx !== -1) arr.splice(idx, 1);
  refreshRecommendation();
}

// Enter: with something typed in the search box, lock in the first visible
// match; otherwise lock in the top recommendation (once it has arrived).
function confirmSelection() {
  if (el.brawlerSearch.value.trim()) {
    const match = [...el.brawlerGrid.children].find((t) => !t.hidden && !t.disabled);
    if (match) applyAction(match.dataset.name);
    return;
  }
  if (!state.loading && state.recommendations.length > 0) applyAction(state.recommendations[0].brawler);
}

function newDraft() {
  state.requestSeq++;
  state.analysisSeq++;
  state.map = null;
  state.mapMode = null;
  state.bans_a = []; state.bans_b = []; state.picks_a = []; state.picks_b = [];
  state.history = [];
  state.recommendations = [];
  state.loading = false;
  state.error = null;
  state.analysis = null;
  state.analysisLoading = false;
  state.analysisError = null;
  el.brawlerSearch.value = "";
  el.draftScreen.classList.add("hidden");
  el.mapScreen.classList.remove("hidden");
  el.mapSearch.value = "";
  renderMapGrid();
  el.mapSearch.focus();
}

// ---------- Events ----------

el.mapSearch.addEventListener("input", renderMapGrid);
el.mapSearch.addEventListener("keydown", (e) => {
  if (e.key !== "Enter") return;
  e.preventDefault();
  const typed = el.mapSearch.value.trim();
  // A name that only exists outside the current Ranked pool still resolves to the real map
  // (with its exact spelling, which is what the model knows) instead of becoming free text.
  const found = filterMaps(typed)[0] || (typed ? filterMaps(typed, state.maps)[0] : null);
  if (found) selectMap(found.name, found.mode);
  else if (typed) selectMap(typed); // free-text map name that isn't in the known list
  else selectMap(null);
});
el.mapGrid.addEventListener("click", (e) => {
  const tile = e.target.closest(".tile");
  if (tile) selectMap(tile.dataset.map, tile.dataset.mode);
});
el.modeChips.addEventListener("click", (e) => {
  const chip = e.target.closest(".chip");
  if (!chip) return;
  state.mapFilter = chip.dataset.mode || null;
  renderMapChips();
  renderMapGrid();
});
el.mapScope.addEventListener("click", () => {
  state.showAllMaps = !state.showAllMaps;
  if (state.mapFilter && !visibleMaps().some((m) => m.mode === state.mapFilter)) state.mapFilter = null;
  renderMapScope();
  renderMapChips();
  renderMapGrid();
});
el.skipMapBtn.addEventListener("click", () => selectMap(null));
el.firstPickSwitch.addEventListener("click", (e) => {
  const btn = e.target.closest(".seg");
  if (!btn) return;
  state.firstPick = btn.dataset.team;
  for (const b of el.firstPickSwitch.children) b.classList.toggle("active", b === btn);
});

el.brawlerSearch.addEventListener("input", updateBrawlerGrid);
el.brawlerGrid.addEventListener("click", (e) => {
  const tile = e.target.closest(".tile");
  if (tile && !tile.disabled) applyAction(tile.dataset.name);
});
el.recList.addEventListener("click", (e) => {
  const item = e.target.closest(".rec-item");
  if (item) applyAction(item.dataset.name);
});
el.undoBtn.addEventListener("click", undo);
el.newDraftBtn.addEventListener("click", newDraft);

// Enter / Esc / "/" - the point is to never need the mouse during a live draft.
document.addEventListener("keydown", (e) => {
  const homeActive = !el.mapScreen.classList.contains("hidden");

  if (e.key === "/") {
    const active = document.activeElement;
    const typing = active && (active.tagName === "INPUT" || active.tagName === "TEXTAREA");
    if (!typing) {
      e.preventDefault();
      (homeActive ? el.mapSearch : el.brawlerSearch).focus();
    }
    return;
  }

  if (homeActive) return; // Enter on the home screen is handled by the map search box

  if (e.key === "Enter") {
    e.preventDefault();
    confirmSelection();
  } else if (e.key === "Escape") {
    e.preventDefault();
    if (el.brawlerSearch.value) {
      el.brawlerSearch.value = "";
      updateBrawlerGrid();
    } else {
      undo();
    }
  }
});

init();
