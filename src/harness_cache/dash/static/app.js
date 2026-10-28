// CAMA dashboard. Read only. Every number comes from the server, computed by the same code as the reports.
"use strict";
const $ = (s, r = document) => r.querySelector(s);
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const num = (v, d = 0) => (v == null || Number.isNaN(v)) ? null : Number(v).toLocaleString("en-US", { maximumFractionDigits: d, minimumFractionDigits: d });
const show = (v, d = 0, unit = "") => { const n = num(v, d); return n == null ? "n/a" : n + unit; };
const usd = (v) => v == null ? "n/a" : "$" + (v < 0.01 ? v.toFixed(5) : v.toFixed(4));
const pct = (v) => v == null ? "n/a" : (100 * v).toFixed(0) + "%";
const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const api = async (p) => { const r = await fetch(p, { cache: "no-store" }); if (!r.ok) throw new Error(`${p}: ${r.status}`); return r.json(); };
const S = { view: "overview", params: new URLSearchParams(), runs: [], open: new Set(), lastOk: 0 };

const TIPS = {
  turn: "Time the user waits for one turn: routing, assembling the request, every model call and tool call.",
  memory: "Part of the turn spent outside the model: routing, embeddings, assembly, the harness itself.",
  cached: "Share of prompt tokens the provider served from its cache (cheaper, billed at the cache read price).",
  cost: "Money for the run: hot path (the turns) plus cold path (memory extraction, cache creation and storage).",
  h3: "Tool calls answered from the tool cache instead of running the tool again.",
  pc: "Turns whose stable prefix (instructions, memory, tool declarations) was sent by cache name.",
  stable: "Share of the request identical to the previous one, from the start: what a prefix cache can reuse.",
  backlog: "Cold path tasks waiting: extraction, cache creation, replication.",
};
const GLOSSARY = [
  ["Hot path", "Everything the user waits for during a turn."],
  ["Cold path", "Work done after the answer, while nobody waits: memory extraction, cache creation, replication."],
  ["H1 routing", "Finding the memory a question needs: words first, then embeddings of the active blocks, then the archive (L3)."],
  ["H2 embedding cache", "A vector computed once for a text is reused, also by other workers."],
  ["H3 tool cache", "A tool result is reused while still valid: forever for pure tools, for a time for others, never for actions like a booking."],
  ["H4 prefix cache", "The stable start of every request is stored by the model provider and sent by name, at a lower price."],
  ["Active set", "The bounded memory sent with every request, rendered in an order that does not change when a block is read."],
  ["Page fault", "A question needs a block that is only in the archive (L3): it is brought back into the active set."],
  ["Ablation", "The same run with one mechanism switched off, to measure what that mechanism adds."],
  ["Confidence interval", "The range where the median would likely fall on a repeat; two runs differ only if their ranges do not overlap."],
  ["Lease and epoch", "Only one worker may write an agent's memory; a takeover raises the epoch and stale writers are refused."],
  ["RPO and RTO", "Memory lost at a crash (rows not yet replicated) and time until another worker answers for the agent."],
];

// ── charts ────────────────────────────────────────────────────────────────
function lineChart(series, { h = 170, yFmt = (v) => show(v), title = "" } = {}) {
  const W = 1000, pad = { l: 56, r: 12, t: 18, b: 24 };
  const pts = series.flatMap((s) => s.points);
  if (!pts.length) return `<div class="absent-note">No data for ${esc(title)}</div>`;
  const xs = pts.map((p) => p[0]), ys = pts.map((p) => p[1]);
  const x0 = Math.min(...xs), x1 = Math.max(...xs), y1 = Math.max(...ys, 1e-9);
  const X = (x) => pad.l + (W - pad.l - pad.r) * (x1 === x0 ? 0.5 : (x - x0) / (x1 - x0));
  const Y = (y) => pad.t + (h - pad.t - pad.b) * (1 - y / y1);
  let g = `<svg class="chart" viewBox="0 0 ${W} ${h}" role="img" aria-label="${esc(title)}">`;
  for (let i = 0; i <= 3; i++) {
    const y = (y1 * i) / 3;
    g += `<line class="grid" x1="${pad.l}" x2="${W - pad.r}" y1="${Y(y)}" y2="${Y(y)}"/><text x="${pad.l - 6}" y="${Y(y) + 4}" text-anchor="end">${esc(yFmt(y))}</text>`;
  }
  g += `<text x="${pad.l}" y="${h - 4}">${x0}</text><text x="${W - pad.r}" y="${h - 4}" text-anchor="end">${x1}</text>`;
  for (const s of series) {
    if (!s.points.length) continue;
    const d = s.points.map((p, i) => `${i ? "L" : "M"}${X(p[0]).toFixed(1)},${Y(p[1]).toFixed(1)}`).join("");
    g += `<path d="${d}" fill="none" stroke="${s.color}" stroke-width="2" ${s.dash ? 'stroke-dasharray="5 4"' : ""}><title>${esc(s.name)}</title></path>`;
  }
  g += "</svg>";
  return g + `<div class="legend">${series.map((s) => `<span><span class="sw" style="background:${s.color}"></span>${esc(s.name)}</span>`).join("")}</div>`;
}

function stackedBars(rows, parts, { h = 170, title = "" } = {}) {
  if (!rows.length) return `<div class="absent-note">No data for ${esc(title)}</div>`;
  const W = 1000, pad = { l: 56, r: 12, t: 18, b: 24 };
  const tot = rows.map((r) => parts.reduce((a, p) => a + (r[p.key] || 0), 0));
  const y1 = Math.max(...tot, 1e-9), bw = Math.max(2, (W - pad.l - pad.r) / rows.length - 2);
  const Y = (v) => (h - pad.t - pad.b) * (v / y1);
  let g = `<svg class="chart" viewBox="0 0 ${W} ${h}" role="img" aria-label="${esc(title)}">`;
  for (let i = 0; i <= 3; i++) {
    const v = (y1 * i) / 3, y = h - pad.b - Y(v);
    g += `<line class="grid" x1="${pad.l}" x2="${W - pad.r}" y1="${y}" y2="${y}"/><text x="${pad.l - 6}" y="${y + 4}" text-anchor="end">${show(v)}</text>`;
  }
  rows.forEach((r, i) => {
    let base = h - pad.b;
    const x = pad.l + i * ((W - pad.l - pad.r) / rows.length);
    for (const p of parts) {
      const v = r[p.key] || 0, hh = Y(v);
      if (hh > 0) g += `<rect x="${x}" y="${base - hh}" width="${bw}" height="${hh}" fill="${p.color(r)}"><title>turn ${r.turn}: ${esc(p.name)} ${show(v)}</title></rect>`;
      base -= hh;
    }
  });
  g += "</svg>";
  return g + `<div class="legend">${parts.map((p) => `<span><span class="sw" style="background:${p.color({})}"></span>${esc(p.name)}</span>`).join("")}</div>`;
}

const kpi = (v, l, tip, absent = false) => `<div class="kpi${absent ? " absent" : ""}" ${tip ? `data-tip title="${esc(TIPS[tip] || tip)}"` : ""}><div class="v">${v}</div><div class="l">${esc(l)}</div></div>`;
const med = (d, unit = " ms") => d ? `${show(d.median)}${unit}` : "n/a";
const ci = (d) => d ? `<span class="small muted">[${show(d.ci_lo)}, ${show(d.ci_hi)}]</span>` : "";

// ── views ─────────────────────────────────────────────────────────────────
async function overview() {
  const frozen = S.runs.filter((r) => r.source === "frozen" && r.kind === "turns").slice(0, 12);
  const details = await Promise.all(frozen.map((r) => api(`/api/run/${encodeURIComponent(r.id)}`).catch(() => null)));
  const rows = details.filter(Boolean).map((d) => {
    const m = d.metrics, c = m.cost || {}, p = m.cost_projected || {};
    return `<tr><td class="l"><a href="#live?run=${encodeURIComponent(d.id)}">${esc(d.id)}</a></td><td class="l">${esc(m.pin.mode)}</td>
      <td class="l">${esc(m.pin.model)}</td><td>${med(m.hot_ms)}</td><td>${pct(m.tokens?.cached_share)}</td>
      <td>${c.priced ? usd(c.total_usd) : p.priced ? usd(p.total_usd) + " *" : "n/a"}</td></tr>`;
  }).join("");
  return `
  <section class="panel fade"><p class="lead">An agent pays, on every step, for more than the model: finding its memory, embedding text,
  calling tools, and sending the same long instructions again. <b>CAMA keeps what is stable reusable</b> across turns and steps,
  measures what each reuse saves, and shows where it stops working.</p></section>
  <section class="panel"><h2>One turn</h2><div class="flow">
    <div class="step hot"><b>1. Find memory</b>words, then active block vectors, then the archive (H1, H2)</div>
    <div class="step hot"><b>2. Build the request</b>stable part first, in an order reads never change</div>
    <div class="step hot"><b>3. Model and tools</b>the stable part sent by cache name (H4), tool results reused (H3)</div>
    <div class="step cold"><b>4. After the answer</b>extract new facts, prepare the next cache</div>
    <div class="step cold"><b>5. Replicate</b>the journal to the shared tier, every second</div>
  </div></section>
  <section class="grid g4">
    <div class="panel level h1"><h3>H1 routing</h3><div class="small muted">Cheapest evidence first: matching words cost nothing, vectors of the active blocks are in memory, the archive is scanned last.</div></div>
    <div class="panel level h2"><h3>H2 embeddings</h3><div class="small muted">A vector is computed once per text and model, then reused, also by other workers and after a restart.</div></div>
    <div class="panel level h3"><h3>H3 tools</h3><div class="small muted">Results reused while valid; a booking or a payment is never served from cache.</div></div>
    <div class="panel level h4"><h3>H4 prefix</h3><div class="small muted">The provider stores the stable start of the request; created after the answer, only once it proved stable.</div></div>
  </section>
  <section class="panel"><h2>Frozen measurements</h2><div class="tbl"><table><tr><th class="l">run</th><th class="l">mode</th><th class="l">model</th>
    <th>turn median</th><th>cached share</th><th>cost</th></tr>${rows || '<tr><td colspan="6" class="l muted">none</td></tr>'}</table></div>
    <p class="small muted">* projected: an echo run priced as if it had used the priced model. Answer quality is not measured in any run.</p></section>
  <section class="panel"><h2>Glossary</h2><dl class="gloss">${GLOSSARY.map(([t, d]) => `<dt>${esc(t)}</dt><dd>${esc(d)}</dd>`).join("")}</dl></section>`;
}

function runSelect(id, kind = "turns", param = "run") {
  const opts = S.runs.filter((r) => r.kind === kind).map((r) => `<option value="${esc(r.id)}"${r.id === id ? " selected" : ""}>${esc(r.id)} · ${esc(r.mode || "")} · ${r.source}</option>`).join("");
  return `<select data-param="${param}">${opts}</select>`;
}

async function live() {
  const id = S.params.get("run") || S.runs.find((r) => r.kind === "turns" && r.source === "live")?.id || S.runs[0]?.id;
  if (!id) return `<div class="empty">No run yet.</div>`;
  const d = await api(`/api/run/${encodeURIComponent(id)}`), m = d.metrics, st = d.store, ser = d.series;
  const last = ser[ser.length - 1], prev = ser[ser.length - 2];
  const h = m.harness || {}, c = m.cost || {};
  const kp = [
    kpi(med(m.hot_ms) + " " + ci(m.hot_ms), "turn, median", "turn"),
    kpi(med(m.memory_ms), "outside the model, median", "memory"),
    kpi(pct(m.tokens?.cached_share), "prompt tokens from cache", "cached"),
    kpi(m.prefix_cache ? `${m.prefix_cache.turns_hit}/${m.prefix_cache.turns}` : "off", "turns on the prefix cache", "pc", !m.prefix_cache),
    kpi(h.tool_calls ? `${h.h3_hits}/${h.tool_calls}` : "n/a", "tool calls from cache", "h3", !h.tool_calls),
    kpi(c.priced ? usd(c.total_usd) : "not priced", "cost of the run", "cost", !c.priced),
    kpi(show(m.n_turns), "turns"),
    kpi(st ? show(st.counts.unsynced) : "n/a", "journal rows not yet replicated", "backlog", !st),
  ].join("");
  let bar = "";
  if (st && last) {
    const keys = last.block_keys || [], pk = prev?.block_keys || [];
    let brk = keys.findIndex((k, i) => k !== pk[i]); if (brk < 0) brk = keys.length;
    const byId = Object.fromEntries(st.active.map((b) => [b.id, b]));
    const colour = { fondamental: "--k-fond", projet: "--k-proj", temp: "--k-temp" };
    const segs = keys.map((k, i) => { const b = byId[k.split("@")[0]]; return { t: b?.tokens || 40, c: css(colour[b?.kind] || "--k-win"), stale: prev && i >= brk, label: k }; });
    segs.push({ t: Math.max(20, last.request_est - segs.reduce((a, s) => a + s.t, 0)), c: css("--k-win"), stale: true, label: "instructions, recent turns, question" });
    const tot = segs.reduce((a, s) => a + s.t, 0);
    bar = `<div class="pbar">${segs.map((s) => `<div class="${s.stale ? "stale" : ""}" title="${esc(s.label)}" style="width:${(100 * s.t) / tot}%;background:${s.c}"></div>`).join("")}</div>
      <div class="legend"><span><span class="sw" style="background:var(--k-fond)"></span>fundamental</span><span><span class="sw" style="background:var(--k-proj)"></span>project</span>
      <span><span class="sw" style="background:var(--k-temp)"></span>temporary</span><span>hatched: changed since the previous request${prev ? (brk >= keys.length ? " (memory unchanged)" : ` (from block ${brk + 1})`) : ""}</span></div>`;
  }
  const blocks = st ? st.active.map((b) => block(b, false)).join("") : `<div class="absent-note">Frozen run: block states were not kept, only the figures.</div>`;
  const tiers = { lexical: "--h1", active: "--h2", l3_fault: "--hot", miss: "--k-win" };
  return `<section class="panel"><div class="bar" style="padding:0">${runSelect(id)}<span class="small muted">${esc(m.pin.model)} · ${esc(m.pin.embedder)} · prefix cache ${esc(m.pin.cache_mode)} · ${esc(m.pin.date)} · code ${esc(m.pin.code_sha)}</span></div></section>
  <section class="kpis">${kp}</section>
  <section class="grid g2">
    <div class="panel"><h2>Memory as the model sees it (render order)</h2>${bar || '<div class="absent-note">No live store for this run.</div>'}<div class="blocks" style="margin-top:12px">${blocks}</div></div>
    <div class="panel"><h2>Archive (L3)</h2><div class="blocks">${st ? st.l3.map((b) => block(b, true)).join("") || '<span class="muted small">empty</span>' : ""}</div>
      <h2 style="margin-top:16px">Cold path, latest tasks</h2>${taskTable(d.tasks)}</div>
  </section>
  <section class="panel"><h2>Per turn: where the time goes</h2>${stackedBars(ser.map((p) => ({ ...p, harness: Math.max(0, p.hot_ms - p.llm_ms - p.tools_ms) })), [
    { key: "harness", name: "harness (routing, assembly)", color: (r) => css(tiers[r.tier] || "--h1") },
    { key: "tools_ms", name: "tools", color: () => css("--h3") }, { key: "llm_ms", name: "model calls", color: () => css("--k-win") }], { title: "latency" })}</section>
  <section class="grid g2">
    <div class="panel"><h2>Tokens per turn</h2>${lineChart([
      { name: "prompt tokens (provider)", color: css("--hot"), points: ser.map((p) => [p.turn, p.prompt]) },
      { name: "of which from cache", color: css("--h4"), points: ser.map((p) => [p.turn, p.cached]) },
      { name: "stable part of the request (estimate)", color: css("--k-proj"), dash: true, points: ser.map((p) => [p.turn, p.stable_est]) }], { title: "tokens" })}</div>
    <div class="panel"><h2>Cumulative cost</h2>${ser.some((p) => p.usd != null) ? lineChart([{ name: "hot path, cumulative USD", color: css("--hot"),
      points: ser.reduce((a, p) => { a.push([p.turn, (a.length ? a[a.length - 1][1] : 0) + (p.usd || 0)]); return a; }, []) }], { title: "cost", yFmt: (v) => "$" + v.toFixed(3) })
      : '<div class="absent-note">Not priced for this run (frozen, or a model without a price).</div>'}</div>
  </section>`;
}

function block(b, inL3) {
  return `<div class="blk ${esc(b.kind)}${S.open.has(b.id) ? " open" : ""}" data-id="${esc(b.id)}"><div class="top"><b>${esc(b.title)}</b>
    ${b.pinned ? '<span class="chip">pinned</span>' : ""}<span class="meta">${esc(b.kind)} · ${esc(b.id)} · v${b.version} · ${b.tokens} tok${inL3 ? "" : ` · recency #${b.recency + 1}`}</span></div><pre>${esc(b.content)}</pre></div>`;
}

function taskTable(tasks) {
  if (!tasks?.length) return '<div class="muted small">none</div>';
  return `<div class="tbl"><table><tr><th class="l">task</th><th>ms</th><th class="l">effect</th></tr>${tasks.slice(-12).reverse().map((t) => {
    const eff = t.ok === false ? `<span class="chip c-bad">${esc(t.error || "failed")}</span>`
      : t.task.startsWith("cache") ? ({ create: `<span class="chip c-h4">created ${show(t.tokens)} tok</span>`, reuse: "kept", skip: `skipped: ${esc(t.reason)}` }[t.action] || esc(t.action))
      : t.task.startsWith("extract") ? `${t.exchanges || 1} exchange(s): ${(t.created || []).length} created, ${(t.appended || []).length} updated`
      : t.synced != null ? `${t.synced} rows replicated` : "";
    return `<tr><td class="l mono small">${esc(t.task)}</td><td>${show(t.run_ms)}</td><td class="l small">${eff}</td></tr>`; }).join("")}</table></div>`;
}

async function compare() {
  const turnRuns = S.runs.filter((r) => r.kind === "turns");
  const a = S.params.get("a") || turnRuns[1]?.id, b = S.params.get("b") || turnRuns[0]?.id;
  if (!a || !b) return `<div class="empty">Two runs are needed.</div>`;
  const d = await api(`/api/compare?a=${encodeURIComponent(a)}&b=${encodeURIComponent(b)}`);
  const A = d.a.metrics, B = d.b.metrics;
  const row = (label, fa, fb) => `<tr><td class="l">${label}</td><td>${fa(A)}</td><td>${fb ? fb(B) : fa(B)}</td></tr>`;
  const diff = Object.keys({ ...A.pin, ...B.pin }).filter((k) => !["run_id", "date"].includes(k) && JSON.stringify(A.pin[k]) !== JSON.stringify(B.pin[k]));
  const colA = css("--cold"), colB = css("--hot");
  const sa = d.a.series, sb = d.b.series;
  return `<section class="panel"><div class="bar" style="padding:0"><b>A</b>${runSelect(a, "turns", "a")}<b>B</b>${runSelect(b, "turns", "b")}<button class="btn" id="swap">swap</button></div></section>
  <section class="panel"><h2>What the data says</h2><div class="verdicts">${d.verdicts.map((v) => `<div class="verdict ${v.strength.split(" ")[0]}">${esc(v.text)}</div>`).join("")}</div></section>
  <section class="grid g2">
    <div class="panel"><h2>Side by side</h2><div class="tbl"><table><tr><th class="l"></th><th>A</th><th>B</th></tr>
      ${row("turn median [95% CI]", (m) => med(m.hot_ms) + " " + ci(m.hot_ms))}
      ${row("outside the model", (m) => med(m.memory_ms))}
      ${row("prompt tokens", (m) => show(m.tokens?.prompt_sum))}
      ${row("served from cache", (m) => pct(m.tokens?.cached_share))}
      ${row("model calls", (m) => show(m.harness?.model_calls))}
      ${row("tool calls from cache", (m) => m.harness ? `${m.harness.h3_hits}/${m.harness.tool_calls}` : "n/a")}
      ${row("cost", (m) => m.cost?.priced ? usd(m.cost.total_usd) : m.cost_projected?.priced ? usd(m.cost_projected.total_usd) + " *" : "n/a")}
      ${row("of which cold path", (m) => m.cost?.priced ? usd(m.cost.cold_usd) : "n/a")}
      ${row("turns", (m) => show(m.n_turns))}
    </table></div><p class="small muted">* projected price. Intervals overlap: no measured difference.</p></div>
    <div class="panel"><h2>What differs in the setup</h2>${diff.length ? `<div class="tbl"><table><tr><th class="l">setting</th><th class="l">A</th><th class="l">B</th></tr>${diff.map((k) => `<tr><td class="l">${esc(k)}</td><td class="l mono small">${esc(A.pin[k])}</td><td class="l mono small">${esc(B.pin[k])}</td></tr>`).join("")}</table></div>` : '<div class="muted">Same setup: only run to run noise separates them.</div>'}</div>
  </section>
  <section class="grid g2">
    <div class="panel"><h2>Prompt tokens per turn</h2>${lineChart([{ name: "A", color: colA, points: sa.map((p) => [p.turn, p.prompt]) }, { name: "B", color: colB, points: sb.map((p) => [p.turn, p.prompt]) },
      { name: "A from cache", color: colA, dash: true, points: sa.map((p) => [p.turn, p.cached]) }, { name: "B from cache", color: colB, dash: true, points: sb.map((p) => [p.turn, p.cached]) }], { title: "tokens" })}</div>
    <div class="panel"><h2>Turn latency</h2>${lineChart([{ name: "A", color: colA, points: sa.map((p) => [p.turn, p.hot_ms]) }, { name: "B", color: colB, points: sb.map((p) => [p.turn, p.hot_ms]) }], { title: "latency", yFmt: (v) => show(v) + " ms" })}</div>
  </section>`;
}

async function costs() {
  const id = S.params.get("run") || S.runs.find((r) => r.kind === "turns")?.id;
  const [d, pricing] = await Promise.all([api(`/api/run/${encodeURIComponent(id)}`), api("/api/pricing")]);
  const c = d.metrics.cost?.priced ? d.metrics.cost : d.metrics.cost_projected;
  const posts = c?.priced ? [["hot: uncached input", c.hot.input, "--hot"], ["hot: cache reads", c.hot.cache_read, "--h4"], ["hot: output", c.hot.output, "--k-temp"],
    ["cold: memory extraction", c.cold.extraction, "--cold"], ["cold: cache creation", c.cold.cache_write, "--h1"], ["cold: cache storage", c.cold.cache_storage, "--h2"]] : [];
  const tot = posts.reduce((a, p) => a + p[1], 0) || 1;
  return `<section class="panel"><div class="bar" style="padding:0">${runSelect(id)}${d.metrics.cost?.priced ? "" : c?.priced ? '<span class="chip c-warn">projected price</span>' : ""}</div></section>
  ${c?.priced ? `<section class="kpis">${kpi(usd(c.total_usd), "total", "cost")}${kpi(usd(c.hot_usd), "hot path")}${kpi(usd(c.cold_usd), "cold path")}${kpi(usd(c.total_usd / Math.max(1, d.metrics.n_turns)), "per turn")}</section>
  <section class="panel"><h2>Where the money goes</h2><div class="pbar">${posts.map((p) => `<div title="${esc(p[0])}: ${usd(p[1])}" style="width:${(100 * p[1]) / tot}%;background:var(${p[2]})"></div>`).join("")}</div>
    <div class="legend">${posts.map((p) => `<span><span class="sw" style="background:var(${p[2]})"></span>${esc(p[0])} ${usd(p[1])} (${((100 * p[1]) / tot).toFixed(0)}%)</span>`).join("")}</div>
    ${(c.assumptions || []).map((a) => `<div class="absent-note" style="margin-top:8px">Assumption: ${esc(a)}</div>`).join("")}</section>`
  : '<section class="panel"><div class="absent-note">This run is not priced: its model has no price in pricing.toml, so no money figure is shown rather than a false zero.</div></section>'}
  <section class="panel"><h2>Price table and its sources</h2><div class="tbl"><table><tr><th class="l">model</th><th class="l">status</th><th>input</th><th>output</th><th>cache read</th><th>storage /h</th><th class="l">explicit cache</th><th>implicit min</th><th class="l">read on</th></tr>
    ${Object.entries(pricing).map(([k, p]) => `<tr><td class="l mono">${esc(k)}</td><td class="l"><span class="chip ${p.status === "official" ? "c-ok" : p.status === "unpriced" ? "c-warn" : ""}">${esc(p.status)}</span></td>
    <td>${p.input ?? "n/a"}</td><td>${p.output ?? "n/a"}</td><td>${p.cache_read ?? "n/a"}</td><td>${p.cache_storage_hour ?? "n/a"}</td><td class="l">${p.explicit_cache ? "yes" : "no"}</td><td>${p.implicit_min_tokens || "unknown"}</td><td class="l small">${esc(p.read_on)}</td></tr>`).join("")}
  </table></div><p class="small muted">USD per million tokens. No provider returns a price: this table is ours, with its source and date.</p></section>`;
}

async function cluster() {
  const c = await api("/api/cluster");
  if (!c.available) return `<section class="panel"><div class="absent-note">Cluster view unavailable: ${esc(c.reason)}. Start it with <span class="mono">docker compose --profile cluster up -d</span>.</div></section>`;
  const now = Date.now() / 1000;
  return `<section class="kpis">${kpi(show(c.workers.length), "workers")}${kpi(show(c.leases.length), "agents with a lease")}${kpi(show(c.counts.journal), "replicated journal rows")}
    ${kpi(show(c.counts.prefixes), "shared prefix caches")}${kpi(show(c.counts.tools), "shared tool results")}${kpi(show(c.counts.embeddings), "shared vectors")}</section>
  <section class="grid g3">${c.workers.map((w) => { const age = now - w.seen; return `<div class="panel"><h3>${esc(w.worker)} <span class="chip ${age < 10 ? "c-ok" : "c-bad"}">${age < 10 ? "alive" : `silent ${show(age)} s`}</span></h3>
    <div class="small muted">agents ${esc((w.info.agents || []).join(", ") || "none")}</div><div class="small">turns ${show(w.info.turns)} · RSS ${show(w.info.rss_mb)} MB · backlog ${show(w.info.backlog)} · unreplicated ${show(w.info.unsynced)}</div></div>`; }).join("")}</section>
  <section class="grid g2"><div class="panel"><h2>Leases</h2><div class="tbl"><table><tr><th class="l">agent</th><th class="l">worker</th><th>epoch</th><th>expires in</th></tr>
    ${c.leases.map((l) => `<tr><td class="l">${esc(l.agent_id)}</td><td class="l">${esc(l.worker)}</td><td>${l.epoch}</td><td>${l.until > now ? show(l.until - now, 1) + " s" : '<span class="chip c-warn">expired</span>'}</td></tr>`).join("")}</table></div></div>
    <div class="panel"><h2>Events</h2><div class="tbl"><table>${c.events.map((e) => `<tr><td class="l mono small">${new Date(e.ts * 1000).toLocaleTimeString()}</td><td class="l">${esc(e.worker)}</td><td class="l"><span class="chip">${esc(e.kind)}</span></td><td class="l">${esc(e.agent_id || "")}</td><td class="l small muted">${esc(JSON.stringify(e.detail)).slice(0, 90)}</td></tr>`).join("")}</table></div></div></section>`;
}

async function stress() {
  const loads = await api("/api/loads");
  if (!loads.length) return `<div class="empty">No load run yet.</div>`;
  return `<section class="panel"><h2>Load and fault runs</h2><div class="tbl"><table><tr><th class="l">run</th><th>agents</th><th>requests</th><th>errors</th><th>failovers</th><th>takeovers</th><th>req/s</th><th>p50</th><th>p95</th><th>p99</th></tr>
    ${loads.map((l) => { const s = l.summary; return `<tr><td class="l mono small">${esc(l.id)}</td><td>${show(l.config.agents)}</td><td>${s.requests}</td><td>${s.errors ? `<span class="chip c-bad">${s.errors}</span>` : 0}</td><td>${s.failovers}</td><td>${s.takeovers ?? "n/a"}</td>
      <td>${show(s.throughput_rps, 2)}</td><td>${show(s.p50_ms)} ms</td><td>${show(s.p95_ms)} ms</td><td>${show(s.p99_ms)} ms</td></tr>`; }).join("")}</table></div>
    <p class="small muted">Load runs use the echo model with a latency model fitted on real calls; the observed limits and their causes are in docs/measured/stress-*.md and docs/limits.md.</p></section>`;
}

async function trace() {
  const ev = await api("/api/ledger");
  const colour = { action: "", run: "c-h1", measurement: "c-h4", decision: "c-ok", fault: "c-bad", review: "c-warn" };
  return `<section class="panel"><h2>The ledger: every action, decision, run and fault</h2><div class="timeline">${ev.map((e) => `<div class="ev"><div class="when">#${e.seq}<br>${esc(e.ts.slice(0, 16).replace("T", " "))}</div>
    <div><span class="chip ${colour[e.kind] || ""}">${esc(e.kind)}</span> <b>${esc(e.title)}</b> <span class="small muted">by ${esc(e.who)} · ${esc(e.result)}</span>
    <div class="small muted">${esc(e.why)}</div>${e.live ? `<div class="small">${esc(e.live)}</div>` : ""}${e.detail ? `<div class="small muted">${esc(e.detail)}</div>` : ""}</div></div>`).join("") || '<div class="muted">The ledger is not mounted.</div>'}</div></section>`;
}

// ── shell ─────────────────────────────────────────────────────────────────
const VIEWS = { overview, live, compare, costs, cluster, stress, trace };
let rendering = false;
async function render() {
  if (rendering) return; rendering = true;
  try {
    const [v, q] = (location.hash.slice(1) || "overview").split("?");
    S.view = VIEWS[v] ? v : "overview"; S.params = new URLSearchParams(q || "");
    document.querySelectorAll("#nav button").forEach((b) => b.classList.toggle("on", b.dataset.view === S.view));
    S.runs = await api("/api/runs");
    $("#main").innerHTML = await VIEWS[S.view]();
    S.lastOk = Date.now(); $("#upd").textContent = "updated " + new Date().toLocaleTimeString();
  } catch (e) { $("#upd").textContent = "error: " + e.message; } finally { rendering = false; }
}
document.addEventListener("click", (e) => {
  const nav = e.target.closest("#nav button"); if (nav) { location.hash = nav.dataset.view; return; }
  const blk = e.target.closest(".blk"); if (blk) { const id = blk.dataset.id; S.open.has(id) ? S.open.delete(id) : S.open.add(id); blk.classList.toggle("open"); return; }
  if (e.target.id === "swap") { const a = S.params.get("a"), b = S.params.get("b"); S.params.set("a", b || ""); S.params.set("b", a || ""); location.hash = `compare?${S.params}`; }
  if (e.target.id === "theme") { const r = document.documentElement; r.dataset.theme = r.dataset.theme === "dark" ? "light" : "dark"; try { localStorage.setItem("hc-theme", r.dataset.theme); } catch (_) {} render(); }
});
document.addEventListener("change", (e) => {
  const p = e.target.dataset.param; if (!p) return;
  if (S.view === "compare" && !S.params.get("a")) { const opts = [...document.querySelectorAll("select[data-param]")]; opts.forEach((s) => S.params.set(s.dataset.param, s.value)); }
  S.params.set(p, e.target.value); location.hash = `${S.view}?${S.params}`;
});
window.addEventListener("hashchange", render);
try { const t = localStorage.getItem("hc-theme"); if (t) document.documentElement.dataset.theme = t; } catch (_) {}
const es = new EventSource("/api/stream");
let pending = null;
es.addEventListener("runs", () => { if (["live", "overview", "costs"].includes(S.view)) { clearTimeout(pending); pending = setTimeout(render, 300); } });
setInterval(() => { $("#dot").className = "dot" + (Date.now() - S.lastOk > 15000 ? " stale" : ""); if (S.view === "cluster") render(); }, 2000);
render();
