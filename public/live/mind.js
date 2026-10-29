// The animal mind, drawn as the instruments of a laboratory: the counterpart
// to heaven. A spike raster of the neurons that judge, region traces like an
// EEG, and a specimen tag. Everything here is plain measurement.

const POPS = [
  ["reward", "reward", +1], ["approach", "approach", +1], ["satiety", "satiety", +1],
  ["punishment", "punish", -1], ["avoidance", "avoid", -1], ["escape", "escape", -1],
];

// Spike raster: one row per neuron, a tick each time it fires, grouped by
// population, with each population's combined firing rate drawn as a line in
// its band. Neurons adapt, as real ones do: each tracks its own recent level
// (over ~2 s) and fires mostly in proportion to how far its activity has just
// risen above it, so bursts mark the moments the view changes in a way that
// matters to that cell, and the raster quiets between them. Silenced neurons
// fall dark. (The brain model gives activity, not spikes; the ticks are drawn
// as a Poisson process from these rates.)
const SCROLL = 45;          // px per second
const SPONT = 0.3;          // Hz, spontaneous
const BURST = 30;           // Hz, at a rise of >= 1.2 sigma above the recent level
const TONIC = 1.2;          // Hz, for sustained activity

export class Raster {
  constructor(canvas, judge, judgeNames, maxRowsPerPop = 40, labels = true) {
    this.c = canvas;
    this.ctx = canvas.getContext("2d");
    this.labels = labels;
    this.rows = [];         // { idx, sign, mean }
    this.groups = [];
    for (const [key, short, sign] of POPS) {
      const p = judgeNames.indexOf(key);
      const members = [];
      judge.forEach((j, i) => j === p && members.push(i));
      const step = Math.max(1, members.length / maxRowsPerPop);
      const start = this.rows.length;
      for (let k = 0; k < members.length; k += step) this.rows.push({ idx: members[Math.floor(k)], sign, mean: null });
      this.groups.push({ short, sign, start, end: this.rows.length, n: members.length, lastY: null });
    }
    this.carry = 0;
  }

  layout(h, dpr) {
    const gap = (this.labels ? 10 : 3) * dpr;
    const rowH = (h - gap * (this.groups.length - 1)) / this.rows.length;
    let y = 0;
    for (const g of this.groups) {
      g.y0 = y;
      g.y1 = y + (g.end - g.start) * rowH;
      y = g.y1 + gap;
    }
    return rowH;
  }

  resize() {
    const dpr = devicePixelRatio || 1;
    const w = Math.round(this.c.clientWidth * dpr), h = Math.round(this.c.clientHeight * dpr);
    if (this.c.width !== w || this.c.height !== h) { this.c.width = w; this.c.height = h; this.ctx.clearRect(0, 0, w, h); }
    return [w, h, dpr];
  }

  // z: Int8Array (σ x 40) for all constellation neurons, or null.
  update(dt, z) {
    const [w, h, dpr] = this.resize();
    const ctx = this.ctx;
    const gutter = this.labels ? 70 * dpr : 0;
    const rowH = this.layout(h, dpr);
    // Adaptation runs every frame, whether or not the raster scrolls.
    const k = 1 - Math.exp(-dt / 2);
    for (const row of this.rows) {
      const s = z ? z[row.idx] / 40 : 0;
      row.s = s;
      row.mean = row.mean === null ? s : row.mean + (s - row.mean) * k;
    }
    this.carry += dt * SCROLL * dpr;
    const shift = Math.floor(this.carry);
    if (shift <= 0) return;
    this.carry -= shift;
    // Scroll by copying the canvas onto itself one step left. The copy must
    // replace what is there, not composite over it, or old ticks smear into
    // solid lines.
    ctx.globalCompositeOperation = "copy";
    ctx.drawImage(this.c, gutter + shift, 0, w - gutter - shift, h, gutter, 0, w - gutter - shift, h);
    ctx.globalCompositeOperation = "source-over";
    ctx.clearRect(w - shift, 0, shift, h);
    const colT = shift / (SCROLL * dpr);                     // seconds this new column spans
    const x = w - shift;
    for (const g of this.groups) {
      let sum = 0;
      for (let r = g.start; r < g.end; r++) {
        const { s, mean } = this.rows[r];
        const rise = Math.max(0, Math.min(1, (s - mean) / 1.2));
        const rate = s >= 0 ? SPONT + BURST * rise ** 1.5 + TONIC * Math.min(1, s / 3)
                            : SPONT * (1 - Math.min(1, -s / 3));
        sum += rate;
        if (Math.random() < rate * colT) {
          ctx.fillStyle = g.sign > 0 ? "rgba(238,234,224,0.95)" : "rgba(155,175,212,0.95)";
          const y = g.y0 + (r - g.start) * rowH;
          ctx.fillRect(x, y + rowH * 0.15, Math.max(1, dpr), Math.max(1, rowH * 0.7));
        }
      }
      // Population rate, as a line through the band.
      const norm = Math.min(1, sum / (g.end - g.start) / 12);
      const y = g.y1 - 1 - norm * (g.y1 - g.y0) * 0.9;
      ctx.strokeStyle = g.sign > 0 ? "rgba(233,196,106,0.85)" : "rgba(130,155,215,0.9)";
      ctx.lineWidth = dpr;
      ctx.beginPath();
      ctx.moveTo(x - 1, g.lastY ?? y);
      ctx.lineTo(x + shift, y);
      ctx.stroke();
      g.lastY = y;
      // Band floor, scrolling with the raster.
      ctx.fillStyle = "rgba(255,255,255,0.07)";
      ctx.fillRect(x, g.y1, shift, Math.max(1, dpr * 0.5));
    }
    if (this.labels) {
      ctx.clearRect(0, 0, gutter, h);
      ctx.font = `${9.5 * dpr}px "IBM Plex Mono", Consolas, monospace`;
      ctx.textBaseline = "middle";
      for (const g of this.groups) {
        const mid = (g.y0 + g.y1) / 2;
        ctx.fillStyle = g.sign > 0 ? "rgba(238,234,224,0.75)" : "rgba(155,175,212,0.8)";
        ctx.fillText(g.short, 0, mid - 6 * dpr);
        ctx.fillStyle = "rgba(255,255,255,0.32)";
        ctx.fillText(`${g.end - g.start}/${g.n}`, 0, mid + 6 * dpr);
      }
    }
  }
}

// Region traces: each region's activity level over the last seconds, stacked.
const TRACE_ORDER = [
  ["retina R", "retina"], ["optic lobe R", "optic lobe"], ["visual projection R", "vis. projection"],
  ["central complex", "central cx"], ["mushroom body", "mushroom body"], ["lateral horn", "lateral horn"],
  ["antennal lobe", "antennal lobe"], ["central brain", "central brain"], ["neuroendocrine", "neuroendocrine"],
  ["descending", "descending"],
];

export class Traces {
  constructor(canvas) {
    this.c = canvas;
    this.ctx = canvas.getContext("2d");
    this.hist = TRACE_ORDER.map(() => []);
    this.carry = 0;
  }

  update(dt, levels) {
    const dpr = devicePixelRatio || 1;
    const w = Math.round(this.c.clientWidth * dpr), h = Math.round(this.c.clientHeight * dpr);
    if (this.c.width !== w || this.c.height !== h) { this.c.width = w; this.c.height = h; }
    const gutter = 96 * dpr, cols = Math.max(2, Math.floor((w - gutter) / dpr / 2));
    this.carry += dt * 24;                                   // samples per second
    while (this.carry >= 1) {
      this.carry -= 1;
      TRACE_ORDER.forEach(([r], k) => {
        const H = this.hist[k];
        H.push(levels?.[r] ?? 0);
        if (H.length > cols) H.shift();
      });
    }
    const ctx = this.ctx;
    ctx.clearRect(0, 0, w, h);
    const pad = 10 * dpr;
    const rowH = (h - pad) / TRACE_ORDER.length;
    ctx.font = `${9.5 * dpr}px "IBM Plex Mono", Consolas, monospace`;
    ctx.textBaseline = "middle";
    TRACE_ORDER.forEach(([, label], k) => {
      const H = this.hist[k], y0 = pad + k * rowH, mid = y0 + rowH * 0.55;
      ctx.fillStyle = "rgba(255,255,255,0.45)";
      ctx.fillText(label, 0, mid);
      ctx.fillStyle = "rgba(255,255,255,0.06)";
      ctx.fillRect(gutter, mid, w - gutter, Math.max(1, dpr * 0.5));
      if (H.length < 2) return;
      let lo = Infinity, hi = -Infinity;
      for (const v of H) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
      const span = Math.max(hi - lo, 0.05);
      ctx.strokeStyle = "rgba(226,230,236,0.85)";
      ctx.lineWidth = dpr;
      ctx.beginPath();
      H.forEach((v, i) => {
        const x = w - (H.length - 1 - i) * 2 * dpr;
        const y = mid - ((v - lo) / span - 0.5) * rowH * 0.8;
        i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
      });
      ctx.stroke();
    });
  }
}

// The arched window onto heaven: a dark field with a round-headed arch cut
// out of it, drawn into an SVG overlay.
export function drawArch(svg) {
  const W = innerWidth, H = innerHeight;
  const aw = Math.min(W * 0.48, H * 0.86), r = aw / 2;
  const x0 = (W - aw) / 2, x1 = x0 + aw, top = H * 0.06, bottom = H * 0.9;
  const spring = top + r;
  const arch = `M ${x0} ${bottom} L ${x0} ${spring} A ${r} ${r} 0 0 1 ${x1} ${spring} L ${x1} ${bottom} Z`;
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  svg.innerHTML = `
    <path d="M 0 0 H ${W} V ${H} H 0 Z ${arch}" fill-rule="evenodd" fill="rgba(7,7,8,0.94)"/>
    <path d="${arch}" fill="none" stroke="rgba(255,255,255,0.16)" stroke-width="1"/>`;
  return { x0, x1, top, bottom };
}
