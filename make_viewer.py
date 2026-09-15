"""
make_viewer.py  —  Convert phase_*/volume.npy outputs to an interactive HTML viewer.

Usage:
    python make_viewer.py --output_dir ./outputs/real
    cd ./outputs/real && python -m http.server 8080
    # open http://localhost:8080/viewer.html

Generates:
    <output_dir>/viewer.html
    <output_dir>/viewer_data/index.json
    <output_dir>/viewer_data/phase_X.XX.bin   (one uint8 binary per phase)
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>CBCT Volume Viewer</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  background: #0f0f1a;
  color: #d0d0e8;
  font-family: 'Segoe UI', system-ui, sans-serif;
  height: 100vh;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

/* ── top bar ── */
#topbar {
  background: #16162a;
  border-bottom: 1px solid #2a2a4a;
  padding: 10px 18px;
  display: flex;
  align-items: center;
  gap: 16px;
  flex-shrink: 0;
}
#topbar h1 { font-size: 15px; color: #7eb8f7; font-weight: 600; white-space: nowrap; }
#topbar .sep { flex: 1; }

/* ── phase strip ── */
#phase-strip {
  background: #13132a;
  border-bottom: 1px solid #2a2a4a;
  padding: 7px 18px;
  display: flex;
  gap: 6px;
  flex-wrap: wrap;
  flex-shrink: 0;
}
#phase-strip label { font-size: 11px; color: #606080; align-self: center; margin-right: 4px; }
.phase-btn {
  padding: 3px 11px;
  border-radius: 12px;
  border: 1px solid #2a2a4a;
  background: #1a1a30;
  color: #9090b8;
  cursor: pointer;
  font-size: 12px;
  transition: all .15s;
}
.phase-btn:hover { border-color: #7eb8f7; color: #c0d8f8; }
.phase-btn.active { background: #2a4a7f; border-color: #7eb8f7; color: #e8f2ff; font-weight: 600; }
#loading-badge {
  font-size: 11px; color: #f0a060; padding: 3px 8px;
  background: #2a1a00; border-radius: 10px; border: 1px solid #4a3010;
  display: none;
}

/* ── main layout ── */
#main { flex: 1; display: flex; overflow: hidden; }

/* ── sidebar ── */
#sidebar {
  width: 190px;
  background: #13132a;
  border-right: 1px solid #2a2a4a;
  padding: 14px 12px;
  display: flex;
  flex-direction: column;
  gap: 14px;
  flex-shrink: 0;
  overflow-y: auto;
}
.section-label {
  font-size: 10px;
  text-transform: uppercase;
  letter-spacing: .08em;
  color: #505070;
  margin-bottom: 4px;
}

/* direction buttons */
.dir-group { display: flex; flex-direction: column; gap: 4px; }
.dir-btn {
  padding: 7px 10px;
  border-radius: 5px;
  border: 1px solid #2a2a4a;
  background: #1a1a30;
  color: #9090b8;
  cursor: pointer;
  font-size: 13px;
  text-align: left;
  transition: all .15s;
  display: flex;
  justify-content: space-between;
  align-items: center;
}
.dir-btn:hover { border-color: #4a6a9a; color: #b8d0f0; }
.dir-btn.active { background: #1e2e50; border-color: #7eb8f7; color: #c8e0ff; }
.dir-btn .axis-label { font-size: 10px; color: #506080; }

/* slice slider */
.ctrl { display: flex; flex-direction: column; gap: 5px; }
.ctrl-row { display: flex; justify-content: space-between; align-items: center; }
.ctrl-val { font-size: 12px; color: #7eb8f7; font-variant-numeric: tabular-nums; }
input[type=range] {
  width: 100%;
  height: 4px;
  appearance: none;
  background: #2a2a50;
  border-radius: 2px;
  outline: none;
  cursor: pointer;
}
input[type=range]::-webkit-slider-thumb {
  appearance: none;
  width: 13px; height: 13px;
  border-radius: 50%;
  background: #7eb8f7;
  cursor: pointer;
}
input[type=range]::-webkit-slider-runnable-track { border-radius: 2px; }

/* zoom buttons */
.zoom-row {
  display: flex;
  gap: 5px;
  align-items: center;
}
.zoom-btn {
  width: 26px; height: 26px;
  border-radius: 4px;
  border: 1px solid #2a2a4a;
  background: #1a1a30;
  color: #9090b8;
  cursor: pointer;
  font-size: 16px;
  line-height: 1;
  display: flex; align-items: center; justify-content: center;
  transition: all .1s;
  flex-shrink: 0;
}
.zoom-btn:hover { border-color: #7eb8f7; color: #c0d8f8; background: #1e2e50; }
#zoom-reset-btn {
  flex: 1;
  padding: 4px 6px;
  border-radius: 4px;
  border: 1px solid #2a2a4a;
  background: #1a1a30;
  color: #9090b8;
  cursor: pointer;
  font-size: 11px;
  transition: all .1s;
}
#zoom-reset-btn:hover { border-color: #7eb8f7; color: #c0d8f8; }

/* colormap */
select {
  width: 100%;
  padding: 4px 6px;
  background: #1a1a30;
  border: 1px solid #2a2a4a;
  color: #c0c0e0;
  border-radius: 4px;
  font-size: 12px;
  cursor: pointer;
}
select:focus { outline: none; border-color: #7eb8f7; }

/* info box */
#info-box {
  background: #0f0f1e;
  border: 1px solid #2a2a4a;
  border-radius: 4px;
  padding: 7px 9px;
  font-size: 11px;
  color: #6060a0;
  line-height: 1.7;
  margin-top: auto;
}
#info-box span { color: #9090c8; }

/* ── canvas area ── */
#canvas-wrap {
  flex: 1;
  background: #080810;
  overflow: hidden;
  position: relative;
}
canvas#vol-canvas {
  position: absolute;
  top: 0; left: 0;
  width: 100%; height: 100%;
  cursor: crosshair;
}
canvas#vol-canvas.panning { cursor: grabbing; }

/* overlays */
.canvas-overlay {
  position: absolute;
  pointer-events: none;
  font-size: 11px;
  background: rgba(8,8,16,.72);
  padding: 3px 8px;
  border-radius: 3px;
  color: #506080;
}
#crosshair-info { bottom: 10px; right: 12px; }
#zoom-badge     { bottom: 10px; left: 12px; color: #7eb8f7; }

#no-data-msg {
  position: absolute;
  top: 50%; left: 50%;
  transform: translate(-50%,-50%);
  color: #404060;
  font-size: 14px;
  pointer-events: none;
}

/* ── playback bar ── */
#playback-bar {
  background: #111124;
  border-bottom: 1px solid #2a2a4a;
  padding: 6px 18px;
  display: flex;
  align-items: center;
  gap: 10px;
  flex-shrink: 0;
}
.play-btn {
  padding: 4px 12px;
  border-radius: 5px;
  border: 1px solid #2a2a4a;
  background: #1a1a30;
  color: #9090b8;
  cursor: pointer;
  font-size: 13px;
  transition: all .15s;
  white-space: nowrap;
}
.play-btn:hover { border-color: #7eb8f7; color: #c0d8f8; }
.play-btn.playing { background: #1e3a20; border-color: #60c870; color: #90f0a0; }
#playback-bar .section-label { white-space: nowrap; }
#fps-slider { width: 64px; }
#fps-val { font-size: 12px; color: #6060a0; min-width: 2em; }
#time-slider-val {
  font-size: 12px; color: #7eb8f7;
  min-width: 3.2em; font-variant-numeric: tabular-nums;
}

/* ── dual-range time slider ── */
#time-range-wrap {
  flex: 1;
  position: relative;
  height: 22px;
  min-width: 80px;
}
#time-track-bg {
  position: absolute;
  left: 7px; right: 7px;
  top: 50%; transform: translateY(-50%);
  height: 4px;
  background: #2a2a50;
  border-radius: 2px;
  pointer-events: none;
}
#time-track-fill {
  position: absolute;
  top: 50%; transform: translateY(-50%);
  height: 4px;
  background: #2a4a80;
  border-radius: 2px;
  pointer-events: none;
}
#time-pos-dot {
  position: absolute;
  top: 50%; transform: translate(-50%, -50%);
  width: 9px; height: 9px;
  border-radius: 50%;
  background: #60e870;
  border: 2px solid #0f0f1a;
  pointer-events: none;
  z-index: 4;
}
#time-range-start,
#time-range-end {
  position: absolute;
  width: 100%;
  top: 50%; transform: translateY(-50%);
  appearance: none; -webkit-appearance: none;
  background: transparent;
  height: 4px;
  pointer-events: none;
  margin: 0; padding: 0;
  outline: none;
}
#time-range-start { z-index: 2; }
#time-range-end   { z-index: 3; }
#time-range-start::-webkit-slider-thumb,
#time-range-end::-webkit-slider-thumb {
  appearance: none; -webkit-appearance: none;
  pointer-events: all;
  width: 14px; height: 14px;
  border-radius: 50%;
  cursor: pointer;
  border: 2px solid #0f0f1a;
}
#time-range-start::-webkit-slider-thumb { background: #7eb8f7; }
#time-range-end::-webkit-slider-thumb   { background: #f0a060; }
#time-range-start::-moz-range-thumb,
#time-range-end::-moz-range-thumb {
  pointer-events: all;
  width: 12px; height: 12px;
  border-radius: 50%;
  cursor: pointer;
  border: 2px solid #0f0f1a;
}
#time-range-start::-moz-range-thumb { background: #7eb8f7; }
#time-range-end::-moz-range-thumb   { background: #f0a060; }
</style>
</head>
<body>

<div id="topbar">
  <h1>&#x1F4F7; CBCT Volume Viewer</h1>
  <span class="sep"></span>
  <span id="loading-badge">Loading…</span>
</div>

<div id="phase-strip">
  <label>Phase</label>
  <!-- phase buttons injected here -->
</div>

<div id="playback-bar">
  <button class="play-btn" id="play-btn" title="Play / Pause (Space)">&#x25B6; Play</button>
  <button class="play-btn" id="stop-btn" title="Stop &amp; rewind to t=0">&#x23F9;</button>
  <span class="section-label">fps</span>
  <input type="range" id="fps-slider" min="1" max="24" value="8">
  <span id="fps-val">8</span>
  <span style="flex:1"></span>
  <span class="section-label" title="Blue = loop start  Orange = loop end  Green dot = current">t =</span>
  <div id="time-range-wrap">
    <div id="time-track-bg"></div>
    <div id="time-track-fill"></div>
    <div id="time-pos-dot"></div>
    <input type="range" id="time-range-start" min="0" max="0" step="1" value="0" title="Loop start (blue)">
    <input type="range" id="time-range-end"   min="0" max="0" step="1" value="0" title="Loop end (orange)">
  </div>
  <span id="time-slider-val">0.00</span>
</div>

<div id="main">
  <div id="sidebar">

    <div>
      <div class="section-label">Direction</div>
      <div class="dir-group">
        <button class="dir-btn active" data-dir="axial">
          Axial <span class="axis-label">z</span>
        </button>
        <button class="dir-btn" data-dir="coronal">
          Coronal <span class="axis-label">y</span>
        </button>
        <button class="dir-btn" data-dir="sagittal">
          Sagittal <span class="axis-label">x</span>
        </button>
      </div>
    </div>

    <div class="ctrl">
      <div class="section-label">Slice</div>
      <div class="ctrl-row">
        <span class="section-label" style="margin:0">index</span>
        <span class="ctrl-val" id="slice-val">—</span>
      </div>
      <input type="range" id="slice-slider" min="0" max="127" value="64">
    </div>

    <div class="ctrl">
      <div class="section-label">Zoom</div>
      <div class="zoom-row">
        <button class="zoom-btn" id="zoom-out-btn" title="Zoom out (-)">−</button>
        <button id="zoom-reset-btn" title="Reset zoom (double-click canvas)">Fit</button>
        <button class="zoom-btn" id="zoom-in-btn"  title="Zoom in (+)">+</button>
      </div>
    </div>

    <div class="ctrl">
      <div class="section-label">Window</div>
      <div class="ctrl-row">
        <span class="section-label" style="margin:0">min</span>
        <span class="ctrl-val" id="wmin-val">0</span>
      </div>
      <input type="range" id="wmin-slider" min="0" max="254" value="0">
      <div class="ctrl-row" style="margin-top:6px">
        <span class="section-label" style="margin:0">max</span>
        <span class="ctrl-val" id="wmax-val">255</span>
      </div>
      <input type="range" id="wmax-slider" min="1" max="255" value="255">
    </div>

    <div class="ctrl">
      <div class="section-label">Colormap</div>
      <select id="colormap-sel">
        <option value="gray">Grayscale</option>
        <option value="hot">Hot</option>
        <option value="viridis">Viridis</option>
        <option value="bone">Bone</option>
      </select>
    </div>

    <div id="info-box">
      Phase: <span id="info-phase">—</span><br>
      Direction: <span id="info-dir">Axial</span><br>
      Slice: <span id="info-slice">—</span><br>
      Shape: <span id="info-shape">—</span>
    </div>

  </div><!-- #sidebar -->

  <div id="canvas-wrap">
    <canvas id="vol-canvas"></canvas>
    <div class="canvas-overlay" id="crosshair-info"></div>
    <div class="canvas-overlay" id="zoom-badge">1.0×</div>
    <div id="no-data-msg">Select a phase to begin.</div>
  </div>
</div><!-- #main -->

<script>
// ── state ────────────────────────────────────────────────────────────────────
const state = {
  phases:      [],
  cache:       {},      // file → Uint8Array
  offscreen:   null,    // OffscreenCanvas holding latest colorised slice
  lastSliceKey: null,   // string key — rebuild offscreen only when it changes
  activePhase: null,
  direction:   'axial',
  sliceIdx:    0,
  wMin: 0, wMax: 255,
  colormap: 'gray',
  // zoom/pan
  zoom: 1.0,
  pan:  { x: 0, y: 0 },
  isPanning:  false,
  panOrigin:  { x: 0, y: 0 },   // mousedown canvas position
  panStart:   { x: 0, y: 0 },   // pan value at mousedown
  // playback
  playback: { playing: false, timerId: null, fps: 8, loopStart: 0, loopEnd: 0 },
};

// ── colormaps (256-entry RGB LUTs) ───────────────────────────────────────────
const CMAPS = (() => {
  function make(fn) {
    const lut = new Uint8Array(256 * 3);
    for (let i = 0; i < 256; i++) {
      const [r, g, b] = fn(i / 255);
      lut[i*3]   = Math.round(Math.min(1, r) * 255);
      lut[i*3+1] = Math.round(Math.min(1, g) * 255);
      lut[i*3+2] = Math.round(Math.min(1, b) * 255);
    }
    return lut;
  }
  return {
    gray:    make(t => [t, t, t]),
    hot:     make(t => [Math.min(1,t*3), Math.min(1,Math.max(0,t*3-1)), Math.min(1,Math.max(0,t*3-2))]),
    viridis: make(t => {
      const r = Math.max(0, 0.280 + t*(0.116 + t*( 1.284 - t*0.895)));
      const g = Math.max(0, -0.090 + t*(2.000 - t*1.600));
      const b = Math.max(0, 0.430 + t*(0.670 - t*1.430));
      return [r, g, b];
    }),
    bone: make(t => [
      Math.min(1, t*0.937 + 0.063*Math.min(1,t*3)),
      Math.min(1, t*0.937 + 0.063*Math.min(1,t*3-1)),
      Math.min(1, t),
    ]),
  };
})();

// ── slice extraction ─────────────────────────────────────────────────────────
// Vol stored in C order: vol[x,y,z] = buf[x*Ny*Nz + y*Nz + z]
function extractSlice(buf, shape, direction, idx) {
  const [Nx, Ny, Nz] = shape;
  let data, width, height;
  if (direction === 'axial') {
    width = Nx; height = Ny;
    data = new Uint8Array(Nx * Ny);
    const z = idx;
    for (let y = 0; y < Ny; y++)
      for (let x = 0; x < Nx; x++)
        data[y*Nx + x] = buf[x*Ny*Nz + y*Nz + z];
  } else if (direction === 'coronal') {
    width = Nx; height = Nz;
    data = new Uint8Array(Nx * Nz);
    const y = idx;
    for (let z = 0; z < Nz; z++)
      for (let x = 0; x < Nx; x++)
        data[(Nz-1-z)*Nx + x] = buf[x*Ny*Nz + y*Nz + z];
  } else {  // sagittal
    width = Ny; height = Nz;
    data = new Uint8Array(Ny * Nz);
    const x = idx;
    for (let z = 0; z < Nz; z++)
      for (let y = 0; y < Ny; y++)
        data[(Nz-1-z)*Ny + y] = buf[x*Ny*Nz + y*Nz + z];
  }
  return { data, width, height };
}

// ── offscreen canvas (colorised slice) ───────────────────────────────────────
function rebuildOffscreen() {
  if (!state.activePhase) return;
  const ph  = state.phases.find(p => p.time === state.activePhase);
  const buf = state.cache[ph.file];
  if (!buf) return;

  const { data, width, height } = extractSlice(
    buf, ph.shape, state.direction, state.sliceIdx
  );
  state.lastSliceData = { data, width, height };   // keep for pixel query

  const oc  = new OffscreenCanvas(width, height);
  const ctx = oc.getContext('2d');
  const img = ctx.createImageData(width, height);
  const lut = CMAPS[state.colormap];
  const span = Math.max(1, state.wMax - state.wMin);

  for (let i = 0; i < data.length; i++) {
    const t = Math.max(0, Math.min(255, Math.round((data[i] - state.wMin) / span * 255)));
    img.data[i*4]   = lut[t*3];
    img.data[i*4+1] = lut[t*3+1];
    img.data[i*4+2] = lut[t*3+2];
    img.data[i*4+3] = 255;
  }
  ctx.putImageData(img, 0, 0);
  state.offscreen = oc;
}

// ── display canvas (applies zoom/pan) ────────────────────────────────────────
function redrawCanvas() {
  const canvas = document.getElementById('vol-canvas');
  const noMsg  = document.getElementById('no-data-msg');

  if (!state.offscreen) {
    noMsg.style.display = 'block';
    return;
  }
  noMsg.style.display = 'none';

  const W = canvas.offsetWidth, H = canvas.offsetHeight;
  if (canvas.width !== W || canvas.height !== H) {
    canvas.width = W;
    canvas.height = H;
  }

  const { width: imgW, height: imgH } = state.offscreen;
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, W, H);

  // image rendered centred + pan offset
  const destX = (W - imgW * state.zoom) / 2 + state.pan.x;
  const destY = (H - imgH * state.zoom) / 2 + state.pan.y;

  ctx.imageSmoothingEnabled = state.zoom < 2;
  ctx.drawImage(state.offscreen, 0, 0, imgW, imgH,
                destX, destY, imgW * state.zoom, imgH * state.zoom);

  document.getElementById('zoom-badge').textContent = `${state.zoom.toFixed(2)}×`;
}

// convenience: rebuild then redraw
function renderSlice() {
  rebuildOffscreen();
  redrawCanvas();
  // update info panel
  if (state.activePhase !== null) {
    const ph = state.phases.find(p => p.time === state.activePhase);
    document.getElementById('info-phase').textContent = state.activePhase.toFixed(2);
    document.getElementById('info-dir').textContent   = capitalize(state.direction);
    document.getElementById('info-slice').textContent = state.sliceIdx;
    if (ph) document.getElementById('info-shape').textContent = ph.shape.join(' × ');
  }
}

// ── zoom helpers ─────────────────────────────────────────────────────────────
const ZOOM_MIN = 0.1, ZOOM_MAX = 32;

function zoomBy(factor, pivotX, pivotY) {
  // pivotX/Y in canvas element coordinates (centre of canvas if not given)
  const canvas = document.getElementById('vol-canvas');
  const cx = (pivotX !== undefined) ? pivotX : canvas.offsetWidth  / 2;
  const cy = (pivotY !== undefined) ? pivotY : canvas.offsetHeight / 2;

  const newZoom = Math.max(ZOOM_MIN, Math.min(ZOOM_MAX, state.zoom * factor));
  const actualFactor = newZoom / state.zoom;

  // keep the pivot point stationary: pan adjusts so the image point under the
  // cursor doesn't move.  Derivation: screenPt = (imgPt - imgW/2)*zoom + W/2 + pan
  // → pan_new = (cx - W/2)*(1 - actualFactor) + pan*actualFactor
  state.pan.x = (cx - canvas.offsetWidth  / 2) * (1 - actualFactor) + state.pan.x * actualFactor;
  state.pan.y = (cy - canvas.offsetHeight / 2) * (1 - actualFactor) + state.pan.y * actualFactor;
  state.zoom  = newZoom;
  redrawCanvas();
}

function resetZoom() {
  state.zoom  = 1.0;
  state.pan.x = 0;
  state.pan.y = 0;
  redrawCanvas();
}

// ── phase loading ─────────────────────────────────────────────────────────────
async function loadPhase(ph) {
  if (state.cache[ph.file]) return;
  const badge = document.getElementById('loading-badge');
  badge.style.display = 'inline';
  try {
    const resp = await fetch(ph.file);
    state.cache[ph.file] = new Uint8Array(await resp.arrayBuffer());
  } finally {
    badge.style.display = 'none';
  }
}

async function activatePhase(ph) {
  document.querySelectorAll('.phase-btn').forEach(b => b.classList.remove('active'));
  document.querySelector(`.phase-btn[data-time="${ph.time}"]`).classList.add('active');
  await loadPhase(ph);
  state.activePhase = ph.time;
  updateSliderRange(ph.shape);
  renderSlice();
  // sync time range UI
  document.getElementById('time-slider-val').textContent = ph.time.toFixed(2);
  updateTimeRangeUI();
}

function updateSliderRange(shape) {
  const [Nx, Ny, Nz] = shape;
  const maxIdx = { axial: Nz-1, coronal: Ny-1, sagittal: Nx-1 }[state.direction];
  const slider = document.getElementById('slice-slider');
  slider.max = maxIdx;
  if (state.sliceIdx < 0 || state.sliceIdx > maxIdx)
    state.sliceIdx = Math.floor(maxIdx / 2);
  slider.value = state.sliceIdx;
  document.getElementById('slice-val').textContent = state.sliceIdx;
}

// ── screen → image coordinate ─────────────────────────────────────────────────
function screenToImage(sx, sy) {
  if (!state.offscreen) return null;
  const canvas = document.getElementById('vol-canvas');
  const { width: imgW, height: imgH } = state.offscreen;
  const destX = (canvas.offsetWidth  - imgW * state.zoom) / 2 + state.pan.x;
  const destY = (canvas.offsetHeight - imgH * state.zoom) / 2 + state.pan.y;
  const ix = Math.floor((sx - destX) / state.zoom);
  const iy = Math.floor((sy - destY) / state.zoom);
  if (ix < 0 || ix >= imgW || iy < 0 || iy >= imgH) return null;
  return { x: ix, y: iy };
}

// ── playback ─────────────────────────────────────────────────────────────────
function currentPhaseIndex() {
  return state.phases.findIndex(p => p.time === state.activePhase);
}

function stepPlay() {
  const n = state.phases.length;
  if (n === 0) return;
  const { loopStart, loopEnd } = state.playback;
  const cur = currentPhaseIndex();
  let next = cur + 1;
  if (next > loopEnd || next >= n) next = loopStart;
  activatePhase(state.phases[next]);
}

function startPlay() {
  if (state.playback.playing) return;
  state.playback.playing = true;
  const btn = document.getElementById('play-btn');
  btn.innerHTML = '&#x23F8; Pause';
  btn.classList.add('playing');
  state.playback.timerId = setInterval(stepPlay, 1000 / state.playback.fps);
}

function stopPlay() {
  if (!state.playback.playing) return;
  clearInterval(state.playback.timerId);
  state.playback.timerId = null;
  state.playback.playing = false;
  const btn = document.getElementById('play-btn');
  btn.innerHTML = '&#x25B6; Play';
  btn.classList.remove('playing');
}

function togglePlay() {
  state.playback.playing ? stopPlay() : startPlay();
}

function updateTimeRangeUI() {
  const n = state.phases.length;
  if (n <= 1) return;
  const { loopStart, loopEnd } = state.playback;
  const cur = Math.max(0, currentPhaseIndex());
  const pct = i => (i / (n - 1)) * 100;
  // track fill between the two handles (accounts for 7px padding on each side)
  const inner = 100 - 14 / document.getElementById('time-range-wrap').offsetWidth * 100;
  const toLeft = i => 7 / document.getElementById('time-range-wrap').offsetWidth * 100
                    + pct(i) * inner / 100;
  const fillL = toLeft(loopStart);
  const fillR = toLeft(loopEnd);
  const fill = document.getElementById('time-track-fill');
  fill.style.left  = fillL + '%';
  fill.style.width = Math.max(0, fillR - fillL) + '%';
  document.getElementById('time-pos-dot').style.left = toLeft(cur) + '%';
}

// ── init ─────────────────────────────────────────────────────────────────────
async function init() {
  const resp = await fetch('viewer_data/index.json');
  const data = await resp.json();
  state.phases = data.phases;

  const strip = document.getElementById('phase-strip');
  data.phases.forEach(ph => {
    const btn = document.createElement('button');
    btn.className = 'phase-btn';
    btn.dataset.time = ph.time;
    btn.textContent  = `t = ${ph.time.toFixed(2)}`;
    btn.addEventListener('click', () => activatePhase(ph));
    strip.appendChild(btn);
  });

  // size canvas once before first render
  const canvas = document.getElementById('vol-canvas');
  canvas.width  = canvas.offsetWidth;
  canvas.height = canvas.offsetHeight;

  // configure dual-range time slider
  const lastIdx = Math.max(0, data.phases.length - 1);
  document.getElementById('time-range-start').max = lastIdx;
  document.getElementById('time-range-end').max   = lastIdx;
  document.getElementById('time-range-start').value = 0;
  document.getElementById('time-range-end').value   = lastIdx;
  state.playback.loopStart = 0;
  state.playback.loopEnd   = lastIdx;

  if (data.phases.length > 0) {
    state.sliceIdx = -1;
    await activatePhase(data.phases[0]);
  }
}

// ── event wiring ──────────────────────────────────────────────────────────────

// direction buttons
document.querySelectorAll('.dir-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.dir-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    state.direction = btn.dataset.dir;
    if (state.activePhase) {
      const ph = state.phases.find(p => p.time === state.activePhase);
      state.sliceIdx = -1;
      updateSliderRange(ph.shape);
    }
    resetZoom();
    renderSlice();
  });
});

// slice slider
document.getElementById('slice-slider').addEventListener('input', e => {
  state.sliceIdx = +e.target.value;
  document.getElementById('slice-val').textContent = state.sliceIdx;
  renderSlice();
});

// window sliders
document.getElementById('wmin-slider').addEventListener('input', e => {
  state.wMin = +e.target.value;
  if (state.wMin >= state.wMax) { state.wMax = state.wMin + 1; document.getElementById('wmax-slider').value = state.wMax; }
  document.getElementById('wmin-val').textContent = state.wMin;
  renderSlice();
});
document.getElementById('wmax-slider').addEventListener('input', e => {
  state.wMax = +e.target.value;
  if (state.wMax <= state.wMin) { state.wMin = state.wMax - 1; document.getElementById('wmin-slider').value = state.wMin; }
  document.getElementById('wmax-val').textContent = state.wMax;
  renderSlice();
});

// colormap
document.getElementById('colormap-sel').addEventListener('change', e => {
  state.colormap = e.target.value;
  renderSlice();
});

// zoom buttons
document.getElementById('zoom-in-btn') .addEventListener('click', () => zoomBy(1.5));
document.getElementById('zoom-out-btn').addEventListener('click', () => zoomBy(1/1.5));
document.getElementById('zoom-reset-btn').addEventListener('click', resetZoom);

// playback controls
document.getElementById('play-btn').addEventListener('click', togglePlay);
document.getElementById('stop-btn').addEventListener('click', () => {
  stopPlay();
  const ph = state.phases[state.playback.loopStart];
  if (ph) activatePhase(ph);
});
document.getElementById('fps-slider').addEventListener('input', e => {
  state.playback.fps = +e.target.value;
  document.getElementById('fps-val').textContent = state.playback.fps;
  if (state.playback.playing) { stopPlay(); startPlay(); }
});
document.getElementById('time-range-start').addEventListener('input', e => {
  let val = +e.target.value;
  if (val >= state.playback.loopEnd) {
    val = Math.max(0, state.playback.loopEnd - 1);
    e.target.value = val;
  }
  state.playback.loopStart = val;
  updateTimeRangeUI();
  if (currentPhaseIndex() < val) { stopPlay(); activatePhase(state.phases[val]); }
});
document.getElementById('time-range-end').addEventListener('input', e => {
  let val = +e.target.value;
  if (val <= state.playback.loopStart) {
    val = Math.min(state.phases.length - 1, state.playback.loopStart + 1);
    e.target.value = val;
  }
  state.playback.loopEnd = val;
  updateTimeRangeUI();
  if (currentPhaseIndex() > val) { stopPlay(); activatePhase(state.phases[val]); }
});

// ── canvas mouse interactions ─────────────────────────────────────────────────
const canvas = document.getElementById('vol-canvas');

// wheel zoom — zoom toward cursor
canvas.addEventListener('wheel', e => {
  e.preventDefault();
  const rect = canvas.getBoundingClientRect();
  const mx = e.clientX - rect.left;
  const my = e.clientY - rect.top;
  const factor = e.deltaY < 0 ? 1.15 : 1 / 1.15;
  zoomBy(factor, mx, my);
}, { passive: false });

// drag to pan
canvas.addEventListener('mousedown', e => {
  if (e.button !== 0) return;
  state.isPanning  = true;
  state.panOrigin  = { x: e.clientX, y: e.clientY };
  state.panStart   = { x: state.pan.x, y: state.pan.y };
  canvas.classList.add('panning');
});
window.addEventListener('mousemove', e => {
  const rect = canvas.getBoundingClientRect();
  const sx = e.clientX - rect.left;
  const sy = e.clientY - rect.top;

  // pixel value readout
  if (!state.isPanning) {
    const pt = screenToImage(sx, sy);
    if (pt && state.lastSliceData) {
      const { data, width } = state.lastSliceData;
      const val = data[pt.y * width + pt.x] ?? '—';
      document.getElementById('crosshair-info').textContent =
        `img (${pt.x}, ${pt.y})  val: ${val}`;
    } else {
      document.getElementById('crosshair-info').textContent = '';
    }
  }

  // pan
  if (!state.isPanning) return;
  state.pan.x = state.panStart.x + (e.clientX - state.panOrigin.x);
  state.pan.y = state.panStart.y + (e.clientY - state.panOrigin.y);
  redrawCanvas();
});
window.addEventListener('mouseup', () => {
  state.isPanning = false;
  canvas.classList.remove('panning');
});

// double-click to reset zoom
canvas.addEventListener('dblclick', resetZoom);

// resize observer — keep canvas pixel size synced to element size
new ResizeObserver(() => {
  const canvas = document.getElementById('vol-canvas');
  canvas.width  = canvas.offsetWidth;
  canvas.height = canvas.offsetHeight;
  redrawCanvas();
}).observe(document.getElementById('canvas-wrap'));

// keyboard shortcuts
document.addEventListener('keydown', e => {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
  const slider = document.getElementById('slice-slider');
  if (e.key === 'ArrowUp' || e.key === 'ArrowRight') {
    slider.value = Math.min(+slider.max, +slider.value + 1);
    slider.dispatchEvent(new Event('input'));
  } else if (e.key === 'ArrowDown' || e.key === 'ArrowLeft') {
    slider.value = Math.max(0, +slider.value - 1);
    slider.dispatchEvent(new Event('input'));
  } else if (e.key === ' ') {
    e.preventDefault();
    togglePlay();
  } else if (e.key === '+' || e.key === '=') {
    zoomBy(1.3);
  } else if (e.key === '-') {
    zoomBy(1/1.3);
  } else if (e.key === '0') {
    resetZoom();
  }
});

function capitalize(s) { return s.charAt(0).toUpperCase() + s.slice(1); }

init().catch(e => {
  document.getElementById('no-data-msg').textContent = 'Error: ' + e.message;
  document.getElementById('no-data-msg').style.display = 'block';
});
</script>
</body>
</html>
"""


def main():
    ap = argparse.ArgumentParser(description="Build CBCT volume HTML viewer from phase_*/volume.npy outputs")
    ap.add_argument('--output_dir', '-o', required=True,
                    help="Directory containing phase_*/volume.npy (output of test_real.py)")
    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        raise FileNotFoundError(f"output_dir not found: {output_dir}")

    # discover phase volumes
    vol_paths = sorted(output_dir.glob("phase_*/volume.npy"),
                       key=lambda p: float(re.sub(r'[^0-9.]', '', p.parent.name.replace('phase_', ''))))
    if not vol_paths:
        raise FileNotFoundError(f"No phase_*/volume.npy files found in {output_dir}")
    print(f"Found {len(vol_paths)} phase(s): {[p.parent.name for p in vol_paths]}")

    viewer_data = output_dir / "viewer_data"
    viewer_data.mkdir(exist_ok=True)

    index = {"phases": []}

    for vol_path in vol_paths:
        t_str = vol_path.parent.name[len("phase_"):]
        t = float(t_str)

        print(f"  processing {vol_path.parent.name} ...", end=" ", flush=True)
        vol = np.load(vol_path).astype(np.float32)

        vmin, vmax = float(vol.min()), float(vol.max())
        if vmax > vmin:
            vol_norm = (vol - vmin) / (vmax - vmin)
        else:
            vol_norm = np.zeros_like(vol)

        vol_u8 = (vol_norm * 255).clip(0, 255).astype(np.uint8)
        bin_name = f"phase_{t_str}.bin"
        bin_path = viewer_data / bin_name

        with open(bin_path, "wb") as fout:
            fout.write(vol_u8.tobytes())

        index["phases"].append({
            "time":  t,
            "file":  f"viewer_data/{bin_name}",
            "shape": list(vol_u8.shape),
        })
        print(f"shape={vol_u8.shape}  size={bin_path.stat().st_size // 1024} KB")

    with open(viewer_data / "index.json", "w") as f:
        json.dump(index, f, indent=2)

    html_path = output_dir / "viewer.html"
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(HTML_TEMPLATE)

    print(f"\nViewer written to: {html_path}")
    print(f"\nTo view:")
    print(f"  cd {output_dir.resolve()}")
    print(f"  python -m http.server 8080")
    print(f"  # then open http://localhost:8080/viewer.html")


if __name__ == "__main__":
    main()
