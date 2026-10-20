/* ==========================================================================
   StudentRisk AI — frontend logic
   Schema-driven form, debounced predictions, TreeSHAP chart, CSV class scan.
   ========================================================================== */
"use strict";

const $ = (sel, el = document) => el.querySelector(sel);
const $$ = (sel, el = document) => [...el.querySelectorAll(sel)];
const icon = (id) => `<svg class="ico" aria-hidden="true"><use href="#i-${id}"/></svg>`;
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

const TONE_COLOR = { high: "#8e3b2f", moderate: "#a9762a", low: "#3c6e51" };
const DOWN_COLOR = "#33526b";
const fmtProb = (p) => (p < 0.005 ? "< 1%" : `${Math.round(p * 100)}%`);

let SCHEMA = null;
let lastResult = null;
let lastBatch = null;
let predictTimer = null;
let toastTimer = null;
let displayedProb = 0;

const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

/* ---------------------------------------------------------------- bootstrap */

async function init() {
  try {
    SCHEMA = await (await fetch("/api/schema")).json();
  } catch (err) {
    toast("Could not reach the model API — is the server running?", true);
    return;
  }

  renderPresets();
  renderForm();
  renderBands();
  renderMetrics();
  wireForm();
  wireClassScan();
  wireReveals();

  // First prediction with the schema defaults — the page opens with a verdict.
  schedulePredict(80);
}

/* ------------------------------------------------------------------ render */

function renderPresets() {
  const box = $("#presets");
  box.innerHTML = Object.values(SCHEMA.presets)
    .map((p) => `<button type="button" class="preset-chip" data-preset="${p.id}" data-tone="${p.tone}">
        <span class="preset-dot"></span>${esc(p.label)}</button>`)
    .join("");

  box.addEventListener("click", (e) => {
    const chip = e.target.closest("[data-preset]");
    if (!chip) return;
    const preset = SCHEMA.presets[chip.dataset.preset];
    applyValues(preset.values);
    $$(".preset-chip", box).forEach((c) => c.classList.toggle("active", c === chip));
    schedulePredict(30);
  });
}

function renderForm() {
  const root = $("#form-groups");
  root.innerHTML = SCHEMA.groups.map((group) => {
    const selects = group.selects
      .map((id) => SCHEMA.selects.find((s) => s.id === id))
      .filter(Boolean)
      .map(renderSelect)
      .join("");

    const fields = group.fields.map(renderField).join("");
    const checks = group.fields.every((f) => f in SCHEMA.binaries)
      ? `<div class="checks">${fields}</div>`
      : `<div class="group-fields">${fields}</div>`;

    return `<section class="form-group" data-group="${group.id}">
      <header class="group-head">
        <span class="group-icon">${icon(group.icon)}</span>
        <span class="group-title">${esc(group.title)}</span>
        <span class="group-note">${esc(group.note)}</span>
      </header>
      ${selects ? `<div class="seg-row">${selects}</div>` : ""}
      ${group.fields.length ? checks : ""}
    </section>`;
  }).join("");
}

function renderSelect(sel) {
  const buttons = sel.options
    .map((opt) => `<button type="button" class="seg-btn${opt === sel.default ? " active" : ""}"
        data-select="${sel.id}" data-value="${esc(opt)}">${esc(opt)}</button>`)
    .join("");
  return `<div class="seg-field">
      <span class="field-label">${esc(sel.title)}</span>
      <div class="segmented" data-select-group="${sel.id}" role="group" aria-label="${esc(sel.title)}">${buttons}</div>
    </div>`;
}

function renderField(name) {
  const meta = SCHEMA.numeric[name];
  if (meta) {
    return `<label class="field" for="f-${name}">
      <span class="field-label">${esc(meta.label)}<span class="field-value" id="v-${name}">${meta.default}</span></span>
      <input type="range" id="f-${name}" data-field="${name}" min="${meta.low}" max="${meta.high}"
             step="1" value="${meta.default}">
    </label>`;
  }
  const bin = SCHEMA.binaries[name];
  return `<label class="check">
      <input type="checkbox" data-field="${name}"${bin.default ? " checked" : ""}>
      <span class="check-box">${icon("check")}</span>
      <span>${esc(bin.label)}</span>
    </label>`;
}

function renderBands() {
  const thresholds = SCHEMA.bands;
  const order = ["low", "moderate", "high"];
  const labels = { low: "Low risk", moderate: "Moderate risk", high: "High risk" };
  const thresholdsText = {
    low: `< ${Math.round(thresholds.moderate * 100)}%`,
    moderate: `${Math.round(thresholds.moderate * 100)}–${Math.round(thresholds.high * 100)}%`,
    high: `≥ ${Math.round(thresholds.high * 100)}%`,
  };
  $("#band-grid").innerHTML = order
    .map((tone) => {
      const label = labels[tone];
      return `<article class="band-card" data-tone="${tone}">
        <h3>${label}<span class="threshold">${thresholdsText[tone]}</span></h3>
        <p>${esc(SCHEMA.band_guide[label])}</p>
      </article>`;
    })
    .join("");
}

function renderMetrics() {
  const m = SCHEMA.metrics;
  const accuracy = `${(m.accuracy * 100).toFixed(1)}%`;
  $("#m-accuracy").textContent = accuracy;
  $("#m-auc").textContent = m.roc_auc.toFixed(2);
  $("#m-split").textContent = `${m.n_train} / ${m.n_test}`;
  $("#m-risk").textContent = `${(m.at_risk_rate * 100).toFixed(1)}%`;
  $("#stat-accuracy").textContent = accuracy;
  $("#stat-auc").textContent = m.roc_auc.toFixed(2);
}

/* ------------------------------------------------------------------- input */

function wireForm() {
  const form = $("#student-form");

  form.addEventListener("submit", (e) => {
    e.preventDefault();
    schedulePredict(10);
  });

  form.addEventListener("input", (e) => {
    const field = e.target.dataset.field;
    if (!field) return;
    if (e.target.type === "range") {
      $(`#v-${field}`).textContent = e.target.value;
    }
    $$(".preset-chip").forEach((c) => c.classList.remove("active"));
    schedulePredict();
  });

  form.addEventListener("click", (e) => {
    const btn = e.target.closest(".seg-btn");
    if (!btn) return;
    const group = btn.closest("[data-select-group]");
    $$(".seg-btn", group).forEach((b) => b.classList.toggle("active", b === btn));
    $$(".preset-chip").forEach((c) => c.classList.remove("active"));
    schedulePredict();
  });

  $("#png-btn").addEventListener("click", exportPNG);
}

function applyValues(values) {
  for (const [key, value] of Object.entries(values)) {
    const range = $(`#f-${key}`);
    if (range) {
      range.value = value;
      $(`#v-${key}`).textContent = value;
      continue;
    }
    const check = $(`input[type="checkbox"][data-field="${key}"]`);
    if (check) {
      check.checked = Boolean(Number(value));
      continue;
    }
    const seg = $(`[data-select-group="${key}"]`);
    if (seg) {
      $$(".seg-btn", seg).forEach((b) => b.classList.toggle("active", b.dataset.value === value));
    }
  }
}

function collect() {
  const values = {};
  $$("#student-form [data-field]").forEach((el) => {
    if (el.type === "checkbox") values[el.dataset.field] = el.checked ? 1 : 0;
    else values[el.dataset.field] = Number(el.value);
  });
  $$("[data-select-group]").forEach((group) => {
    const active = $(".seg-btn.active", group);
    if (active) values[group.dataset.selectGroup] = active.dataset.value;
  });
  return values;
}

function schedulePredict(delay = 240) {
  clearTimeout(predictTimer);
  predictTimer = setTimeout(predict, delay);
}

async function predict() {
  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ values: collect() }),
    });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    lastResult = await response.json();
    renderResult(lastResult);
  } catch (err) {
    toast("Prediction failed — please try again.", true);
  }
}

/* ------------------------------------------------------------------ result */

function renderResult(r) {
  $("#result-empty").hidden = true;
  $("#result-body").hidden = false;

  const badge = $("#band-badge");
  badge.textContent = r.band;
  badge.dataset.tone = r.tone;

  animateProb(r.probability);  // also drives the meter width (min sliver for tiny values)
  $("#meter-fill").dataset.tone = r.tone;

  renderPlain(r.explain);
  renderChart(r.drivers.slice(0, 12));

  const m = SCHEMA.metrics;
  $("#chart-note").innerHTML =
    `Exact TreeSHAP values in log-odds. <strong>${icon("trend-up")} red</strong> pushes the student toward being at risk, ` +
    `<strong>${icon("trend-down")} blue</strong> toward being on track — measured from the average student ` +
    `(base value ${r.base_value.toFixed(2)}).`;
  $("#result-foot").textContent =
    `Model: XGBoost · test accuracy ${(m.accuracy * 100).toFixed(1)}% · ROC-AUC ${m.roc_auc.toFixed(2)} · ` +
    `trained on ${m.n_train} students (UCI Student Performance).`;
}

function animateProb(target) {
  const el = $("#prob-value");
  const start = displayedProb;
  displayedProb = target;

  const fmt = fmtProb;
  $("#meter-fill").style.width = `${Math.max(target * 100, 0.5).toFixed(1)}%`;
  // Background tabs throttle requestAnimationFrame to zero — skip the count-up
  // there so the final value always lands.
  if (reducedMotion || document.hidden) {
    el.textContent = fmt(target);
    return;
  }
  const t0 = performance.now();
  const duration = 550;
  function frame(now) {
    const p = Math.min((now - t0) / duration, 1);
    const eased = 1 - Math.pow(1 - p, 3);
    const value = start + (target - start) * eased;
    el.textContent = fmt(value);
    if (p < 1) requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
}

function renderPlain(explain) {
  const list = $("#plain-list");
  if (explain.empty) {
    list.innerHTML = `<li>No single factor dominated this prediction.</li>`;
    return;
  }
  const ups = explain.ups.map((u, i) =>
    `<li data-dir="up">${icon("trend-up")}<span><b>${esc(u.label)}</b> <span class="rec">(recorded: ${u.value})</span> ` +
    `${i === 0 ? "pushed the risk <b>up</b> the most" : "also pushed the risk <b>up</b>"}.</span></li>`
  );
  const downs = explain.downs.map((d, i) =>
    `<li data-dir="down">${icon("trend-down")}<span><b>${esc(d.label)}</b> <span class="rec">(recorded: ${d.value})</span> ` +
    `${i === 0 ? "worked most strongly <b>in the student's favour</b>" : "also worked <b>in the student's favour</b>"}.</span></li>`
  );
  list.innerHTML = [...ups, ...downs].join("");
}

function renderChart(drivers) {
  const W = 720;
  const rowH = 30;
  const padTop = 24;
  const padBottom = 34;
  const labelRight = 268; // labels end here
  const axisX = 470;
  const half = 200;
  const H = padTop + drivers.length * rowH + padBottom;

  const maxAbs = Math.max(...drivers.map((d) => Math.abs(d.contribution)), 0.0001);

  const rows = drivers.map((d, i) => {
    const y = padTop + i * rowH;
    const barY = y + 4;
    const magnitude = Math.abs(d.contribution);
    const barW = Math.max((magnitude / maxAbs) * half, 1.5);
    const up = d.contribution > 0;
    const color = up ? TONE_COLOR.high : DOWN_COLOR;
    const barX = up ? axisX : axisX - barW;
    const textX = up ? axisX + barW + 7 : axisX - barW - 7;
    const anchor = up ? "start" : "end";
    const value = `${d.label} = ${d.value}`;
    return `
      <text x="${labelRight}" y="${barY + 11}" text-anchor="end" font-family="Inter, sans-serif"
            font-size="12" fill="#4c463b">${esc(value)}</text>
      <rect x="${barX}" y="${barY}" width="${barW}" height="15" rx="2" fill="${color}"/>
      <text x="${textX}" y="${barY + 11.5}" text-anchor="${anchor}" font-family="Fraunces, Georgia, serif"
            font-size="11.5" fill="${color}">${d.contribution > 0 ? "+" : ""}${d.contribution.toFixed(2)}</text>`;
  });

  const svg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${W} ${H}" role="img"
      aria-label="Feature contributions to the risk prediction">
    <rect width="${W}" height="${H}" fill="#fffdf8"/>
    <line x1="${axisX}" y1="${padTop - 12}" x2="${axisX}" y2="${H - padBottom + 14}"
          stroke="#1d1a14" stroke-width="1.2"/>
    <text x="${axisX - half}" y="${H - padBottom + 28}" font-family="Inter, sans-serif" font-size="10.5"
          letter-spacing="1.5" fill="#8b8271" text-anchor="start">← PULLS IT DOWN</text>
    <text x="${axisX + half}" y="${H - padBottom + 28}" font-family="Inter, sans-serif" font-size="10.5"
          letter-spacing="1.5" fill="#8b8271" text-anchor="end">PUSHES IT UP →</text>
    <text x="${axisX}" y="${padTop - 14}" font-family="Inter, sans-serif" font-size="10.5"
          letter-spacing="1.5" fill="#8b8271" text-anchor="middle">LOG-ODDS IMPACT</text>
    ${rows.join("")}
  </svg>`;
  $("#shap-chart").innerHTML = svg;
}

/* -------------------------------------------------------------- PNG export */

function exportPNG() {
  if (!lastResult) return;
  const svg = $("#shap-chart svg");
  if (!svg) return;

  const vb = svg.viewBox.baseVal;
  const svgW = 880;
  const svgH = Math.round((vb.height / vb.width) * svgW);
  const headH = 132;
  const footH = 64;
  const canvas = document.createElement("canvas");
  canvas.width = svgW;
  canvas.height = headH + svgH + footH;
  const ctx = canvas.getContext("2d");

  ctx.fillStyle = "#faf7f0";
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  // Header
  ctx.fillStyle = "#1d1a14";
  ctx.font = "520 30px Fraunces, Georgia, serif";
  ctx.fillText("StudentRisk AI — risk explanation", 36, 52);
  ctx.font = "600 15px Inter, sans-serif";
  ctx.fillStyle = TONE_COLOR[lastResult.tone];
  ctx.fillText(`${lastResult.band} · ${fmtProb(lastResult.probability)} probability of failing`, 36, 84);
  ctx.strokeStyle = "#cec3a8";
  ctx.beginPath();
  ctx.moveTo(36, 106);
  ctx.lineTo(canvas.width - 36, 106);
  ctx.stroke();

  // Footer
  ctx.fillStyle = "#8b8271";
  ctx.font = "400 13px Inter, sans-serif";
  ctx.fillText(
    `Exact TreeSHAP · base value ${lastResult.base_value.toFixed(2)} · generated ${new Date().toISOString().slice(0, 10)} · decision support only`,
    36,
    canvas.height - 26
  );

  const blob = new Blob([new XMLSerializer().serializeToString(svg)], { type: "image/svg+xml;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const img = new Image();
  img.onload = () => {
    ctx.drawImage(img, 0, headH, svgW, svgH);
    URL.revokeObjectURL(url);
    canvas.toBlob((png) => downloadBlob(png, "risk-explanation.png"), "image/png");
  };
  img.onerror = () => {
    URL.revokeObjectURL(url);
    toast("Could not render the PNG in this browser.", true);
  };
  img.src = url;
}

/* -------------------------------------------------------------- class scan */

function wireClassScan() {
  const dropzone = $("#dropzone");
  const input = $("#csv-input");

  dropzone.addEventListener("click", () => input.click());
  dropzone.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      input.click();
    }
  });
  ["dragover", "dragenter"].forEach((type) =>
    dropzone.addEventListener(type, (e) => {
      e.preventDefault();
      dropzone.classList.add("dragover");
    })
  );
  ["dragleave", "drop"].forEach((type) =>
    dropzone.addEventListener(type, (e) => {
      e.preventDefault();
      dropzone.classList.remove("dragover");
    })
  );
  dropzone.addEventListener("drop", (e) => {
    const file = e.dataTransfer.files[0];
    if (file) handleCSV(file);
  });
  input.addEventListener("change", () => {
    if (input.files[0]) handleCSV(input.files[0]);
    input.value = "";
  });

  $("#sample-btn").addEventListener("click", downloadSample);
  $("#csv-dl-btn").addEventListener("click", downloadReport);
}

async function handleCSV(file) {
  let rows;
  try {
    rows = parseCSV(await file.text());
  } catch (err) {
    toast("That file could not be read as CSV.", true);
    return;
  }
  if (rows.length < 2) {
    toast("The CSV needs a header row and at least one student.", true);
    return;
  }

  const header = rows[0].map((h) => h.trim());
  const missing = SCHEMA.feature_order.filter((c) => !header.includes(c));
  if (missing.length) {
    toast(`Missing columns: ${missing.slice(0, 5).join(", ")}${missing.length > 5 ? "…" : ""}`, true);
    return;
  }

  const objects = rows.slice(1).filter((r) => r.some((cell) => cell.trim() !== ""))
    .map((row) => Object.fromEntries(header.map((h, i) => [h, row[i] ?? ""])));

  if (objects.length > 2000) {
    toast("That roster is over 2,000 rows — please split the file.", true);
    return;
  }

  try {
    const response = await fetch("/api/batch", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ rows: objects }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || `HTTP ${response.status}`);
    lastBatch = payload;
    renderBatch(payload);
    toast(`Scored ${payload.results.length} students.`);
  } catch (err) {
    toast(err.message || "Scoring the class failed.", true);
  }
}

function renderBatch(payload) {
  $("#class-counts").hidden = false;
  $("#count-high").textContent = payload.counts.high;
  $("#count-moderate").textContent = payload.counts.moderate;
  $("#count-low").textContent = payload.counts.low;

  const tbody = $("#report-table tbody");
  tbody.innerHTML = payload.results
    .map(
      (r) => `<tr>
        <td class="student">${esc(r.student)}</td>
        <td class="num">${(r.probability * 100).toFixed(1)}%</td>
        <td><span class="lozenge" data-tone="${r.tone}">${r.band}</span></td>
        <td>${esc(r.top_factor)}</td>
      </tr>`
    )
    .join("");
  $("#table-wrap").hidden = false;
  $("#csv-dl-btn").hidden = false;
}

function downloadSample() {
  const lines = ["student," + SCHEMA.feature_order.join(",")];
  const presets = Object.values(SCHEMA.presets);
  presets.forEach((preset, i) => {
    const v = preset.values;
    const encoded = SCHEMA.feature_order.map((f) => {
      if (f === "is_female") return v.sex === "Female" ? 1 : 0;
      if (f === "is_urban") return v.area === "Urban" ? 1 : 0;
      return v[f];
    });
    lines.push(`Example ${i + 1},` + encoded.join(","));
  });
  // One tweaked row so the roster isn't just the three presets.
  const high = presets[0].values;
  const tweaked = SCHEMA.feature_order.map((f) => {
    if (f === "is_female") return high.sex === "Female" ? 1 : 0;
    if (f === "is_urban") return high.area === "Urban" ? 1 : 0;
    if (f === "absences") return 12;
    if (f === "G1") return 9;
    if (f === "G2") return 9;
    return high[f];
  });
  lines.push(`Example 4,` + tweaked.join(","));

  downloadBlob(new Blob([lines.join("\n")], { type: "text/csv" }), "sample_class.csv");
}

function downloadReport() {
  if (!lastBatch) return;
  const header = ["student", "risk_probability", "risk_band", "top_factor"];
  const rows = lastBatch.results.map((r) =>
    [r.student, r.probability, r.band, r.top_factor].map(csvCell).join(",")
  );
  downloadBlob(new Blob([[header.join(","), ...rows].join("\n")], { type: "text/csv" }), "class_risk_report.csv");
}

function parseCSV(text) {
  const rows = [];
  let row = [];
  let cell = "";
  let quoted = false;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (quoted) {
      if (c === '"') {
        if (text[i + 1] === '"') { cell += '"'; i++; }
        else quoted = false;
      } else cell += c;
    } else if (c === '"') quoted = true;
    else if (c === ",") { row.push(cell); cell = ""; }
    else if (c === "\n" || c === "\r") {
      if (c === "\r" && text[i + 1] === "\n") i++;
      row.push(cell);
      cell = "";
      rows.push(row);
      row = [];
    } else cell += c;
  }
  if (cell !== "" || row.length) { row.push(cell); rows.push(row); }
  return rows;
}

const csvCell = (value) => {
  const s = String(value);
  return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
};

function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 4000);
}

/* ------------------------------------------------------- chrome: toast etc */

function toast(message, isError = false) {
  const el = $("#toast");
  el.textContent = message;
  el.classList.toggle("error", isError);
  el.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => el.classList.remove("show"), 3600);
}

function wireReveals() {
  const items = $$(".reveal");
  if (!("IntersectionObserver" in window) || reducedMotion) {
    items.forEach((el) => el.classList.add("in"));
    return;
  }
  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add("in");
          observer.unobserve(entry.target);
        }
      });
    },
    { threshold: 0.12 }
  );
  items.forEach((el) => observer.observe(el));
}

document.addEventListener("DOMContentLoaded", init);
