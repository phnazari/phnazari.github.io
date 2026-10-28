// Progressive enhancement for the LoRes post. Measured timing data is kept below.
(() => {
  "use strict";
  const root = document.querySelector(".lores");
  if (!root) return;
  const $ = (selector) => root.querySelector(selector);
  const $$ = (selector) => [...root.querySelectorAll(selector)];
  const setPressed = (buttons, active) => buttons.forEach((button) => button.setAttribute("aria-pressed", String(button === active)));
  const exampleBits = 4;

  // Keep links into collapsed aspects and their nested examples usable.
  const revealAspect = (hash) => {
    if (!hash) return;
    let target;
    try {
      target = document.getElementById(decodeURIComponent(hash.slice(1)));
    } catch {
      return;
    }
    if (!target || !root.contains(target) || !target.closest(".lr-aspect")) return;
    let opened = false;
    for (let panel = target.closest("details"); panel; panel = panel.parentElement.closest("details")) {
      if (!panel.open) {
        panel.open = true;
        opened = true;
      }
    }
    if (opened) target.scrollIntoView({ block: "start" });
  };
  root.addEventListener("click", (event) => {
    const link = event.target.closest('a[href^="#"]');
    if (link) revealAspect(link.getAttribute("href"));
  });
  window.addEventListener("hashchange", () => revealAspect(window.location.hash));
  revealAspect(window.location.hash);

  const modelButtons = $$("[data-lr-model]");
  const selectModel = (button) => {
    setPressed(modelButtons, button);
    $$("[data-lr-model-panel]").forEach((panel) => {
      panel.hidden = panel.dataset.lrModelPanel !== button.dataset.lrModel;
    });
  };
  modelButtons.forEach((button) => button.addEventListener("click", () => selectModel(button)));
  selectModel(modelButtons[0]);
  $("#lr-model-controls").hidden = false;

  // SVG labels and coordinates come only from constants and local numeric values.
  let width = Math.max(280, Math.min(660, root.clientWidth - 28));
  const height = 280;
  const margin = { left: 65, right: 22, top: 36, bottom: 45 };
  let plotWidth = width - margin.left - margin.right;
  const plotHeight = height - margin.top - margin.bottom;
  const y = (value, max) => margin.top + plotHeight * (1 - value / max);
  const line = (x1, y1, x2, y2, extra = "") => `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" ${extra}/>`;
  const label = (x, yy, text, extra = "") => `<text x="${x}" y="${yy}" ${extra}>${text}</text>`;
  function frame(id, title, description, body) {
    return `<svg viewBox="0 0 ${width} ${height}" role="img" aria-labelledby="${id}-title ${id}-desc"><title id="${id}-title">${title}</title><desc id="${id}-desc">${description}</desc>${body}</svg>`;
  }

  function randomUnitVector() {
    const vector = Array.from({ length: 4 }, () => 2 * Math.random() - 1);
    const norm = Math.hypot(...vector);
    return norm > 0 ? vector.map((value) => value / norm) : [1, 0, 0, 0];
  }
  // Keep the sampled inputs and added vector fixed while comparing the controls.
  let groupingState = {
    matrix: [[1, -0.7, 0.5, 0.2], [0.8, 0.6, -0.3, 0.4], [0.5, -0.9, 0.7, 0.2], [-0.4, 0.3, 0.6, 0.8]],
    query: [0.5, 0.5, 0.5, 0.5],
    addedVector: randomUnitVector(),
  };
  function randomGroupingState() {
    const matrix = Array.from({ length: 4 }, () => Array.from({ length: 4 }, () => (Math.floor(Math.random() * 21) - 10) / 10));
    return { matrix, query: randomUnitVector(), addedVector: randomUnitVector() };
  }
  // Single-step readout variance from the paper, for the current unit-norm query.
  function groupingExample(magnitude, bits, axis, perturbAxis) {
    const added = groupingState.matrix.map((row, i) => row.map((_, j) => {
      const selected = perturbAxis === "value" ? i === 0 : j === 0;
      const component = groupingState.addedVector[perturbAxis === "value" ? j : i];
      return selected ? magnitude * component : 0;
    }));
    const matrix = groupingState.matrix.map((row, i) => row.map((value, j) => value + added[i][j]));
    const query = groupingState.query;
    const rowMax = matrix.map((row) => Math.max(...row.map(Math.abs)));
    const columnMax = matrix[0].map((_, j) => Math.max(...matrix.map((row) => Math.abs(row[j]))));
    const levels = 2 ** (bits - 1) - 1;
    const keyRms = Math.sqrt(columnMax.reduce((sum, value, j) => sum + query[j] ** 2 * value ** 2, 0)) / (Math.sqrt(12) * levels);
    const rms = rowMax.map((value) => axis === "value" ? value / (Math.sqrt(12) * levels) : keyRms);
    const output = matrix.map((row) => row.reduce((sum, value, j) => sum + value * query[j], 0));
    return { matrix, added, query, output, rms };
  }
  const groupingButtons = $$("[data-lr-grouping]");
  const perturbButtons = $$("[data-lr-perturb]");
  function drawGrouping() {
    const axis = groupingButtons.find((button) => button.getAttribute("aria-pressed") === "true").dataset.lrGrouping;
    const perturbAxis = perturbButtons.find((button) => button.getAttribute("aria-pressed") === "true").dataset.lrPerturb;
    const magnitude = Number($("#lr-perturbation-size").value);
    const example = groupingExample(magnitude, exampleBits, axis, perturbAxis);
    $("#lr-perturbation-value").value = magnitude.toFixed(2);
    $("#lr-perturbation-size").setAttribute("aria-valuetext", `${magnitude.toFixed(2)} vector magnitude`);
    const rows = example.matrix.map((row, i) => `<tr><th scope="row">Row ${i + 1}</th>${row.map((value, j) => {
      const selected = perturbAxis === "value" ? i === 0 : j === 0;
      const delta = example.added[i][j];
      const signedDelta = `${delta < 0 && Math.abs(delta) >= 0.005 ? "−" : "+"}${Math.abs(delta).toFixed(2)}`;
      const description = `Row ${i + 1}, column ${j + 1}: ${value.toFixed(2)}` + (selected ? `, original ${groupingState.matrix[i][j].toFixed(2)}, added ${signedDelta}` : "");
      return `<td data-group="${axis === "value" ? i : j}" data-perturbed="${selected}" aria-label="${description}"><span class="lr-state-value">${value.toFixed(2)}</span>${selected ? `<span class="lr-state-addition">Δ ${signedDelta}</span>` : ""}</td>`;
    }).join("")}</tr>`).join("");
    $("#lr-state-matrix").innerHTML = `<table class="lr-matrix" data-axis="${axis}" data-perturb="${perturbAxis}" aria-label="Toy state matrix, random vector of magnitude ${magnitude.toFixed(2)} added to ${perturbAxis === "value" ? "value row" : "key column"} 1, grouped by ${axis === "value" ? "value rows" : "key columns"}"><tbody>${rows}</tbody></table>`;
    $("#lr-query-vector").innerHTML = example.query.map((value, i) => `<div class="lr-vector-entry" role="listitem" aria-label="Query entry ${i + 1}: ${value}">${value.toFixed(3)}</div>`).join("");
    $("#lr-output-vector").innerHTML = example.output.map((value, i) => `<div class="lr-vector-entry" role="listitem" aria-label="Output entry ${i + 1}: ${value.toFixed(2)}, predicted RMS error ${example.rms[i].toFixed(3)}"><span>${value.toFixed(2)}</span><span class="lr-output-error">±${example.rms[i].toFixed(3)}</span></div>`).join("");
  }
  groupingButtons.forEach((button) => button.addEventListener("click", () => { setPressed(groupingButtons, button); drawGrouping(); }));
  perturbButtons.forEach((button) => button.addEventListener("click", () => { setPressed(perturbButtons, button); drawGrouping(); }));
  $("#lr-perturbation-size").addEventListener("input", drawGrouping);
  $("#lr-grouping-resample").addEventListener("click", () => { groupingState = randomGroupingState(); drawGrouping(); });
  drawGrouping();
  $("#lr-grouping-explorer").hidden = false;

  function drawDecay() {
    const alpha = Number($("#lr-decay").value) / 1000;
    const x = (step) => margin.left + plotWidth * step / 100;
    let svg = "";
    for (const fraction of [0, 0.25, 0.5, 0.75, 1]) {
      svg += line(margin.left, y(fraction, 1), width - margin.right, y(fraction, 1), 'class="lr-axis"');
      svg += label(margin.left - 10, y(fraction, 1) + 4, `${fraction * 100}%`, 'text-anchor="end"');
    }
    for (const step of [0, 25, 50, 75, 100]) svg += label(x(step), height - 24, step, 'text-anchor="middle"');
    svg += label(margin.left, 17, "Surviving error");
    svg += label(width - margin.right, height - 4, "Updates since injection", 'text-anchor="end"');
    const points = Array.from({ length: 101 }, (_, n) => `${x(n)},${y(alpha ** n, 1)}`).join(" ");
    svg += `<polyline points="${points}" fill="none" stroke="var(--lr-decay)" stroke-width="3"/>`;
    const remaining = 100 * alpha ** 100;
    const halfLife = alpha < 1 ? Math.log(0.5) / Math.log(alpha) : Infinity;
    const summary = `After 100 updates: ${remaining.toFixed(2)}% remains. ` + (Number.isFinite(halfLife) ? `Half-life: ${halfLife.toFixed(1)} updates.` : "No decay.");
    $("#lr-decay-value").value = alpha.toFixed(3);
    $("#lr-decay").setAttribute("aria-valuetext", alpha.toFixed(3));
    $("#lr-decay-chart").innerHTML = frame("lr-decay-plot", `Error decay with alpha ${alpha.toFixed(3)}`, summary, svg);
  }
  $("#lr-decay").addEventListener("input", drawDecay);
  drawDecay();
  $("#lr-decay-explorer").hidden = false;

  // Seeded sampling keeps slider comparisons reproducible; the button changes the seed.
  function randomGenerator(seed) {
    return () => {
      let value = seed += 0x6D2B79F5;
      value = Math.imul(value ^ value >>> 15, value | 1);
      value ^= value + Math.imul(value ^ value >>> 7, value | 61);
      return ((value ^ value >>> 14) >>> 0) / 4294967296;
    };
  }
  // A scalar coordinate of a key/value outer-product write, sampled independently per token.
  // Within-token correlation is illustrative, not fitted to model statistics.
  const keyValueModel = { correlation: 0.7 };
  function sampleKeyValueUpdates(scale, seed, steps = 100) {
    const { correlation } = keyValueModel;
    const updateRandom = randomGenerator(seed ^ 0x9E3779B9);
    const normalPair = () => {
      const radius = Math.sqrt(-2 * Math.log(1 - updateRandom()));
      const angle = 2 * Math.PI * updateRandom();
      return [radius * Math.cos(angle), radius * Math.sin(angle)];
    };
    const independentValue = Math.sqrt(1 - correlation ** 2);
    return Array.from({ length: steps }, () => {
      const [z, w] = normalPair();
      const key = z;
      const value = correlation * z + independentValue * w;
      return scale * value * key;
    });
  }
  // The paper's signed remainder: accumulated squared error minus the decayed
  // sum of squared fresh injections. Compute it before averaging SR runs.
  function roundingRemainder(updates, alpha, random = null) {
    let reference = 0, state = 0, individual = 0;
    for (const update of updates) {
      reference = alpha * reference + update;
      const input = alpha * state + update;
      const lower = Math.floor(input);
      state = random ? lower + Number(random() < input - lower) : Math.round(input);
      const injection = state - input;
      individual = alpha ** 2 * individual + injection ** 2;
    }
    return (state - reference) ** 2 - individual;
  }
  function roundingExample(scale, alpha, seed, steps = 100, trials = 1024, trajectories = 128) {
    const nearest = [], mean = [];
    for (let index = 0; index < trajectories; index++) {
      const trajectorySeed = (seed + Math.imul(index, 0x9E3779B9)) >>> 0;
      const updates = sampleKeyValueUpdates(scale, trajectorySeed, steps);
      const random = randomGenerator(trajectorySeed ^ 0x85EBCA6B);
      nearest.push(roundingRemainder(updates, alpha));
      let average = 0;
      for (let trial = 0; trial < trials; trial++) average += roundingRemainder(updates, alpha, random) / trials;
      mean.push(average);
    }
    return { nearest, mean };
  }
  // Common histogram bins on a signed-log axis keep both small signed SR
  // remainders and the larger RTN tail visible, as in the paper's figure.
  function roundingHistogram(series) {
    const transform = (value) => Math.sign(value) * Math.log10(1 + Math.abs(value) / 0.01);
    const inverse = (value) => Math.sign(value) * 0.01 * (10 ** Math.abs(value) - 1);
    const values = series.flatMap((sample) => sample.map(transform));
    const binWidth = 0.2;
    const first = Math.floor(Math.min(-0.4, ...values) / binWidth);
    const last = Math.floor(Math.max(0.4, ...values) / binWidth) + 1;
    const histograms = series.map((sample) => {
      const counts = Array(last - first).fill(0);
      for (const value of sample) counts[Math.floor(transform(value) / binWidth) - first]++;
      return counts.map((count) => count / sample.length);
    });
    return { histograms, first, last, binWidth, transform, inverse };
  }
  let roundingSeed = 2026;
  let roundingCache = { key: "", paths: null };
  function drawRounding() {
    const scale = Number($("#lr-update-size").value) / 100;
    const alpha = Number($("#lr-rounding-decay").value) / 1000;
    const key = `${scale}:${alpha}:${roundingSeed}`;
    if (roundingCache.key !== key) roundingCache = { key, paths: roundingExample(scale, alpha, roundingSeed) };
    const samples = roundingCache.paths;
    const histogram = roundingHistogram([samples.nearest, samples.mean]);
    const { histograms, first, last, binWidth, transform, inverse } = histogram;
    const left = margin.left, right = width - margin.right;
    const x = (coordinate) => left + (right - left) * (coordinate / binWidth - first) / (last - first);
    const curves = [
      { title: "RTN", bins: histograms[0], sample: samples.nearest, color: "var(--lr-orange)", style: "" },
      { title: "SR", bins: histograms[1], sample: samples.mean, color: "var(--lr-green)", style: 'stroke-dasharray="6 4"' },
    ];
    const peak = Math.max(...histograms.flat(), 0.01);
    const tickMagnitude = 10 ** Math.floor(Math.log10(peak / 4));
    const tick = [1, 2, 5, 10].find((step) => step * tickMagnitude >= peak / 4) * tickMagnitude;
    const max = Math.ceil(peak / tick) * tick;
    let svg = "";
    const summaries = [];
    for (let index = 0; index <= Math.round(max / tick); index++) {
      const value = index * tick;
      svg += line(left, y(value, max), right, y(value, max), 'class="lr-axis"');
      svg += label(left - 10, y(value, max) + 4, `${Math.round(100 * value)}%`, 'text-anchor="end"');
    }
    svg += label(left, 17, "Fraction of trajectories");
    svg += label(right, height - 4, "Remaining error", 'text-anchor="end"');
    const ticks = [0];
    for (const sign of [-1, 1]) {
      let previous = x(0);
      for (let power = -2; power <= 6; power++) {
        const value = sign * 10 ** power, position = x(transform(value));
        if (position >= left && position <= right && Math.abs(position - previous) >= 42) {
          ticks.push(value);
          previous = position;
        }
      }
    }
    for (const value of ticks) svg += label(x(transform(value)), height - 24, value, 'text-anchor="middle"');
    for (const { title, bins, sample, color, style } of curves) {
      let path = `M ${x(first * binWidth)} ${y(0, max)}`;
      bins.forEach((fraction, index) => {
        path += ` L ${x((first + index) * binWidth)} ${y(fraction, max)} L ${x((first + index + 1) * binWidth)} ${y(fraction, max)}`;
      });
      path += ` L ${x(last * binWidth)} ${y(0, max)}`;
      svg += `<path d="${path} Z" fill="${color}" fill-opacity="0.12" stroke="none"/>`;
      svg += `<path d="${path}" fill="none" stroke="${color}" stroke-width="2.5" ${style}/>`;
      summaries.push(`${title}: ${sample.filter((value) => value < 0).length} negative, ${sample.filter((value) => value > 0).length} positive, range ${Math.min(...sample).toPrecision(4)} to ${Math.max(...sample).toPrecision(4)}`);
    }
    svg += line(x(0), margin.top, x(0), height - margin.bottom, 'stroke="var(--lr-muted)" stroke-width="1.5" stroke-dasharray="3 3"');
    // Tooltips report actual values and frequencies, not transformed coordinates.
    for (let index = 0; index < last - first; index++) {
      const lower = inverse((first + index) * binWidth), upper = inverse((first + index + 1) * binWidth);
      const binLabel = `Remaining error ${lower.toPrecision(3)} to ${upper.toPrecision(3)}: RTN ${(100 * histograms[0][index]).toFixed(1)}%, SR ${(100 * histograms[1][index]).toFixed(1)}% of trajectories`;
      svg += `<rect x="${x((first + index) * binWidth)}" y="${margin.top}" width="${x((first + index + 1) * binWidth) - x((first + index) * binWidth)}" height="${plotHeight}" fill="transparent"><title>${binLabel}</title></rect>`;
    }
    const description = `Distribution of the signed interaction remainder after 100 updates across 128 synthetic scalar trajectories. The remainder isolates interactions between rounding errors from different steps. RTN can reinforce these errors, producing positive values. Under SR, these interactions vanish in expectation. For each trajectory, RTN uses one deterministic run and SR averages the remainder over 1,024 rounding runs with the same updates. Update scale ${scale.toFixed(2)}, decay ${alpha.toFixed(3)}. Common histogram bins on a signed-log horizontal axis. The vertical reference marks zero. ${summaries.join(". ")}.`;
    $("#lr-update-value").value = scale.toFixed(2);
    $("#lr-update-size").setAttribute("aria-valuetext", `${scale.toFixed(2)} times the key-value product`);
    $("#lr-rounding-decay-value").value = alpha.toFixed(3);
    $("#lr-rounding-decay").setAttribute("aria-valuetext", `${alpha.toFixed(3)} decay factor`);
    $("#lr-rounding-chart").innerHTML = frame("lr-rounding-plot", "Remaining error: RTN versus SR", description, svg);
  }
  ["#lr-update-size", "#lr-rounding-decay"].forEach((selector) => $(selector).addEventListener("input", drawRounding));
  $("#lr-rounding-resample").addEventListener("click", () => { roundingSeed++; drawRounding(); });
  drawRounding();
  $("#lr-rounding-explorer").hidden = false;

  // Actual finite-grid SR, including saturation when reference scales are too small.
  function feedbackQuantize(values, step, levels, draws) {
    if (step <= 0) return values.map(() => 0);
    return values.map((value, j) => {
      const position = Math.max(-levels, Math.min(levels, value / step));
      const nearest = Math.round(position);
      if (Math.abs(position - nearest) < 1e-10) return nearest * step;
      const lower = Math.floor(position);
      return step * (lower + (draws[j] < position - lower ? 1 : 0));
    });
  }
  function feedbackExample(alpha, updateScale, bits, seed, steps = 200) {
    const random = randomGenerator(seed);
    const levels = 2 ** (bits - 1) - 1;
    let reference = Array.from({ length: 64 }, () => 2 * random() - 1);
    let noisy = [...reference], oracle = [...reference];
    // One correlated key/value pair per component, with independent component streams.
    const updates = reference.map((_, index) => sampleKeyValueUpdates(
      updateScale, (seed + Math.imul(index, 0x9E3779B9)) >>> 0, steps,
    ));
    const scale = (state) => Math.max(...state.map(Math.abs)) / levels;
    const error = (state) => Math.hypot(...state.map((value, j) => value - reference[j]));
    const initialStep = scale(reference);
    const history = [{ noisyStep: initialStep, referenceStep: initialStep, noisyError: 0, referenceError: 0 }];
    for (let t = 1; t <= steps; t++) {
      const update = updates.map((trajectory) => trajectory[t - 1]);
      const draws = reference.map(() => random());
      const advance = (state) => state.map((value, j) => alpha * value + update[j]);
      reference = advance(reference);
      const noisyInput = advance(noisy), oracleInput = advance(oracle);
      const noisyStep = scale(noisyInput), referenceStep = scale(reference);
      noisy = feedbackQuantize(noisyInput, noisyStep, levels, draws);
      oracle = feedbackQuantize(oracleInput, referenceStep, levels, draws);
      history.push({ noisyStep, referenceStep, noisyError: error(noisy), referenceError: error(oracle) });
    }
    return history;
  }
  let feedbackSeed = 2026;
  let feedbackCache = { key: "", history: [] };
  function drawFeedback() {
    const alpha = Number($("#lr-feedback-decay").value) / 1000;
    const updateScale = Number($("#lr-feedback-update-scale").value);
    const key = `${alpha}:${updateScale}:${feedbackSeed}`;
    if (feedbackCache.key !== key) feedbackCache = { key, history: feedbackExample(alpha, updateScale, exampleBits, feedbackSeed) };
    const history = feedbackCache.history;
    const lastStep = history.length - 1;
    const final = history[lastStep];
    $("#lr-feedback-update-value").value = updateScale.toFixed(3);
    $("#lr-feedback-update-scale").setAttribute("aria-valuetext", `${updateScale.toFixed(3)} times the key-value product`);
    $("#lr-feedback-decay-value").value = alpha.toFixed(3);
    $("#lr-feedback-decay").setAttribute("aria-valuetext", `${alpha.toFixed(3)} decay factor`);
    const left = 62, right = width - 20, plotHeight = 188;
    const x = (step) => left + (right - left) * step / lastStep;
    const specs = [
      { title: "Quantization scale", fields: ["noisyStep", "referenceStep"], top: 34 },
    ];
    let svg = "";
    for (const { title, fields, top } of specs) {
      const peak = Math.max(...history.flatMap((row) => fields.map((field) => row[field])), 1e-8);
      const magnitude = 10 ** Math.floor(Math.log10(peak / 4));
      const tick = [1, 2, 5, 10].find((step) => step * magnitude >= peak / 4) * magnitude;
      const max = Math.ceil(peak / tick) * tick;
      const yy = (value) => top + plotHeight * (1 - value / max);
      svg += label(left, top - 16, title, 'class="lr-loss-value"');
      for (let index = 0; index <= Math.round(max / tick); index++) {
        const value = index * tick;
        svg += line(left, yy(value), right, yy(value), 'class="lr-axis"');
        svg += label(left - 10, yy(value) + 4, Number(value.toPrecision(3)), 'text-anchor="end"');
      }
      fields.forEach((field, index) => {
        const color = index === 0 ? "var(--lr-orange)" : "var(--lr-green)";
        const name = index === 0 ? "Scale from noisy state" : "Scale from unquantized copy";
        const points = history.map((row, t) => `${x(t)},${yy(row[field])}`).join(" ");
        svg += `<polyline points="${points}" fill="none" stroke="${color}" stroke-width="2.5"${index === 1 ? ' stroke-dasharray="6 4"' : ""}/>`;
        const value = final[field];
        svg += `<circle cx="${x(lastStep)}" cy="${yy(value)}" r="${index === 0 ? 3.5 : 5}" fill="${index === 0 ? color : "var(--lr-surface)"}" stroke="${color}" stroke-width="2"><title>${name}, update ${lastStep}: ${value.toPrecision(4)}</title></circle>`;
      });
      for (const fraction of [0, 0.25, 0.5, 0.75, 1]) svg += label(x(fraction * lastStep), top + plotHeight + 21, fraction * lastStep, 'text-anchor="middle"');
    }
    svg += label(right, 271, "Number of updates", 'text-anchor="end"');
    const description = `64-component toy recurrence with key–value pairs correlated within each token and independent across tokens, stochastic rounding, update scale ${updateScale.toFixed(3)}, and decay ${alpha.toFixed(3)}. All ${lastStep} updates shown. Final scale from noisy state: ${final.noisyStep.toPrecision(4)}. Scale from unquantized copy: ${final.referenceStep.toPrecision(4)}. Both runs share the initial state, updates, and random draws.`;
    $("#lr-feedback-chart").innerHTML = `<svg viewBox="0 0 ${width} 280" role="img" aria-labelledby="lr-feedback-plot-title lr-feedback-plot-desc"><title id="lr-feedback-plot-title">Quantization scales over repeated updates</title><desc id="lr-feedback-plot-desc">${description}</desc>${svg}</svg>`;
  }
  ["#lr-feedback-decay", "#lr-feedback-update-scale"].forEach((selector) => $(selector).addEventListener("input", drawFeedback));
  $("#lr-feedback-resample").addEventListener("click", () => { feedbackSeed++; drawFeedback(); });
  $("#lr-feedback-explorer").hidden = false;
  drawFeedback();

  // Measured TPOT (ms) from the paper, with two-token prompts; null means not measured.
  const batches = [1, 2, 4, 8, 16, 32, 64, 128, 256];
  const timings = {
    "Nemotron-H · 16/16/32": [6.31, 6.41, 6.62, 6.96, 7.63, 8.78, 11.4, 16.83, 29.35],
    "Nemotron-H · 16/16/8": [6.27, 6.32, 6.44, 6.67, 6.98, 7.66, 9.01, 11.18, 17.32],
    "Nemotron-H · 4/16/32": [4.46, 4.52, 4.73, 4.99, 5.62, 6.77, 9.86, null, null],
    "Nemotron-H · 4/16/8": [4.41, 4.44, 4.57, 4.74, 4.98, 5.57, 7.5, null, null],
    "Qwen-3.5 · 16/16/32": [6.99, 7.05, 7.15, 7.39, 7.77, 8.44, 9.6, 12.19, 18.23],
    "Qwen-3.5 · 16/16/8": [6.98, 7.01, 7.06, 7.2, 7.48, 7.92, 8.67, 9.99, 14.21],
    "Qwen-3.5 · 4/16/32": [5.7, 5.79, 5.89, 6.11, 6.46, 7.22, 9.57, null, null],
    "Qwen-3.5 · 4/16/8": [5.67, 5.74, 5.81, 5.96, 6.23, 6.76, 8.59, null, null],
  };

  // Both speedups use the same BF16/FP32 baseline, shown at 1×.
  // The orange segment spans the weights-only and combined speedups.
  function speedContributions(model) {
    const baseline = timings[`${model} · 16/16/32`];
    const weightsOnly = timings[`${model} · 4/16/32`];
    const weightsAndState = timings[`${model} · 4/16/8`];
    return batches.flatMap((batch, index) => {
      if (![baseline[index], weightsOnly[index], weightsAndState[index]].every((value) => Number.isFinite(value) && value > 0)) return [];
      const weightSpeedup = baseline[index] / weightsOnly[index];
      const totalSpeedup = baseline[index] / weightsAndState[index];
      return [{ batch, weightSpeedup, totalSpeedup }];
    });
  }
  const contributionButtons = $$("[data-lr-contribution-model]");
  const contributionTooltip = $("#lr-contributions-tooltip");
  const hideContributionTooltip = () => { contributionTooltip.hidden = true; };
  function drawContributions(button) {
    hideContributionTooltip();
    setPressed(contributionButtons, button);
    const model = button.dataset.lrContributionModel;
    const contributions = speedContributions(model);
    const peak = Math.max(...contributions.map(({ totalSpeedup }) => totalSpeedup));
    const rawStep = (peak - 1) / 4;
    const magnitude = 10 ** Math.floor(Math.log10(rawStep));
    const tickStep = [1, 2, 5, 10].find((step) => step * magnitude >= rawStep) * magnitude;
    const tickCount = Math.ceil((peak - 1) / tickStep);
    const maxGain = tickCount * tickStep;
    const speedY = (speedup) => y(speedup - 1, maxGain);
    const slot = plotWidth / contributions.length;
    const barWidth = Math.min(44, slot * 0.68);
    let svg = "";
    for (let tick = 0; tick <= tickCount; tick++) {
      const speedup = 1 + tick * tickStep;
      svg += line(margin.left, speedY(speedup), width - margin.right, speedY(speedup), 'class="lr-axis"');
      svg += label(margin.left - 10, speedY(speedup) + 4, `${speedup.toFixed(1)}×`, 'text-anchor="end"');
    }
    svg += label(margin.left, 17, "Speedup (×)");
    svg += label(width - margin.right, height - 4, "Batch size", 'text-anchor="end"');
    const summaries = contributions.map(({ batch, weightSpeedup, totalSpeedup }, index) => {
      const center = margin.left + slot * (index + 0.5);
      const summary = `Batch ${batch}: INT4 weights alone give ${weightSpeedup.toFixed(2)}× speedup. With INT8 states added, the combined speedup is ${totalSpeedup.toFixed(2)}×.`;
      svg += `<g class="lr-contribution-bar" data-lr-batch="${batch}" tabindex="0" role="img" aria-label="${summary}">`;
      svg += `<rect x="${center - barWidth / 2}" y="${speedY(weightSpeedup)}" width="${barWidth}" height="${speedY(1) - speedY(weightSpeedup)}" fill="var(--lr-green)" data-lr-contribution="weights" data-lr-speedup="${weightSpeedup}"/>`;
      svg += `<rect x="${center - barWidth / 2}" y="${speedY(totalSpeedup)}" width="${barWidth}" height="${speedY(weightSpeedup) - speedY(totalSpeedup)}" fill="var(--lr-orange)" data-lr-contribution="state" data-lr-speedup="${totalSpeedup}"/>`;
      svg += "</g>";
      svg += label(center, height - 24, batch, 'text-anchor="middle"');
      return summary;
    });
    const description = `End-to-end vLLM decode speedup relative to BF16 weights and FP32 states, with BF16 activations. The axis starts at the 1× baseline. Green reaches the INT4-weight speedup. Orange extends to the combined speedup with INT8 states. ${summaries.join(" ")}`;
    $("#lr-contributions-chart").innerHTML = frame("lr-contributions-plot", `${model}: weight and state contributions`, description, svg);
    $$(".lr-contribution-bar").forEach((bar, index) => {
      const { batch, weightSpeedup, totalSpeedup } = contributions[index];
      const positionTooltip = (event) => {
        const bounds = bar.getBoundingClientRect();
        const container = contributionTooltip.parentElement.getBoundingClientRect();
        const hasPointer = event && typeof event.clientX === "number";
        const left = hasPointer ? event.clientX - container.left + 12 : bounds.left - container.left + bounds.width / 2 - contributionTooltip.offsetWidth / 2;
        contributionTooltip.style.left = `${Math.max(0, Math.min(container.width - contributionTooltip.offsetWidth, left))}px`;
        contributionTooltip.style.top = `${(hasPointer ? event.clientY : bounds.bottom) - container.top + 16}px`;
      };
      const showTooltip = (event) => {
        contributionTooltip.innerHTML = `<div class="lr-tooltip-title">Batch ${batch}</div><div class="lr-tooltip-row"><span><i class="lr-bar-key lr-weight-key"></i>Weights only</span><span>${weightSpeedup.toFixed(2)}×</span></div><div class="lr-tooltip-row lr-tooltip-total"><span><i class="lr-bar-key lr-state-key"></i>Weights + state</span><span>${totalSpeedup.toFixed(2)}×</span></div>`;
        contributionTooltip.hidden = false;
        positionTooltip(event);
      };
      bar.addEventListener("pointerenter", showTooltip);
      bar.addEventListener("pointermove", positionTooltip);
      bar.addEventListener("pointerleave", hideContributionTooltip);
      bar.addEventListener("focus", showTooltip);
      bar.addEventListener("blur", hideContributionTooltip);
      bar.addEventListener("click", showTooltip);
      bar.addEventListener("keydown", (event) => { if (event.key === "Escape") hideContributionTooltip(); });
    });
  }
  contributionButtons.forEach((button) => button.addEventListener("click", () => drawContributions(button)));
  drawContributions(contributionButtons[0]);
  $("#lr-speed-contributions").hidden = false;

  if (typeof ResizeObserver !== "undefined") {
    new ResizeObserver(() => {
      const nextWidth = Math.max(280, Math.min(660, root.clientWidth - 28));
      if (nextWidth === width) return;
      width = nextWidth;
      plotWidth = width - margin.left - margin.right;
      drawDecay();
      drawRounding();
      drawFeedback();
      drawContributions(contributionButtons.find((button) => button.getAttribute("aria-pressed") === "true"));
    }).observe(root);
  }

  const copy = $("#lr-copy");
  const citation = $("#lr-bibtex");
  const copyStatus = $("#lr-copy-status");
  if (copy && citation?.textContent.trim() && navigator.clipboard && window.isSecureContext) {
    copy.hidden = false;
    copy.addEventListener("click", async () => {
      try {
        await navigator.clipboard.writeText(citation.textContent);
        if (copyStatus) copyStatus.textContent = "Citation copied.";
      } catch {
        if (copyStatus) copyStatus.textContent = "Copy was unavailable. Select the citation text to copy it manually.";
      }
    });
  }
})();
