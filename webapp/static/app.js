/* Shutter smear player front end.
 *
 * The shutter control set is built from the server's ui_schema (which comes
 * from ShutterKernel.ui_schema()), so this file contains no hard-coded shutter
 * parameters.  Shutter angle and exposure are kept in sync client-side; any
 * server-side ShutterError is rendered back inside the shutter panel.
 */
"use strict";

const state = {
  config: null,
  scene: null,
  method: "rolling",
  frame: 4,
  angle: 180,
  samples: 32,
  readout: 0.9,
  readoutDir: "top_down",
  exposureDir: "top_down",
  profile: "trapezoid",
  showPaths: true,
  playing: false,
  playTimer: null,
  pending: null,
  abortCtrl: null,
};

const $ = (id) => document.getElementById(id);

async function init() {
  const res = await fetch("/api/config");
  state.config = await res.json();
  const d = state.config.defaults;
  state.scene = d.scene;
  state.method = d.method;
  state.frame = d.frame_index;
  state.angle = d.shutter_angle_deg;
  state.samples = d.num_samples;
  state.readout = d.readout_frac;
  state.readoutDir = d.readout_direction;
  state.exposureDir = d.exposure_direction;
  state.profile = d.exposure_profile;
  state.showPaths = d.show_paths;

  fillSelect($("scene"), state.config.scenes, state.scene);
  fillSelect($("method"), state.config.methods, state.method);
  fillSelect($("ctl-readout-dir"), state.config.directions, state.readoutDir);
  fillSelect($("ctl-exposure-dir"), state.config.directions, state.exposureDir);
  fillSelect($("ctl-profile"), state.config.profiles, state.profile);
  buildKernelControls(state.config.ui_schema);

  $("frame-slider").max = state.config.num_frames - 1;
  $("frame-slider").value = state.frame;
  $("show-paths").checked = state.showPaths;
  $("ctl-readout").value = state.readout;
  updateReadoutLabel();

  $("scene").onchange = (e) => { state.scene = e.target.value; schedule(); };
  $("method").onchange = (e) => { state.method = e.target.value; schedule(); };
  $("ctl-readout-dir").onchange = (e) => {
    state.readoutDir = e.target.value; schedule();
  };
  $("ctl-exposure-dir").onchange = (e) => {
    state.exposureDir = e.target.value; schedule();
  };
  $("ctl-profile").onchange = (e) => {
    state.profile = e.target.value; schedule();
  };
  $("ctl-readout").oninput = (e) => {
    state.readout = parseFloat(e.target.value);
    updateReadoutLabel();
    schedule();
  };
  $("show-paths").onchange = (e) => {
    state.showPaths = e.target.checked;
    schedule();
  };
  $("frame-slider").oninput = (e) => {
    state.frame = parseInt(e.target.value, 10);
    schedule();
  };
  $("step-back").onclick = () => jumpFrame(-1);
  $("step-fwd").onclick = () => jumpFrame(1);
  $("play").onclick = togglePlay;
  $("snapshot").onclick = downloadFrame;
  $("export-mp4").onclick = () => exportClip("mp4");
  $("export-gif").onclick = () => exportClip("gif");

  await loadAssets();
  await render(true);
}

function updateReadoutLabel() {
  const dt = state.config.frame_interval;
  $("val-readout").textContent =
    `${state.readout.toFixed(2)} · R=${(state.readout * dt * 1000).toFixed(2)} ms`;
}

async function loadAssets() {
  const host = $("asset-list");
  try {
    const res = await fetch("/api/assets");
    const payload = await res.json();
    if (!payload.assets.length) {
      host.textContent = "未发现样片资源（先运行 CLI author）";
      return;
    }
    host.innerHTML = "";
    for (const asset of payload.assets) {
      const item = document.createElement("div");
      item.className = "asset-item";
      const verified = Object.values(asset.hashes_ok).every(Boolean);
      item.innerHTML =
        `<div><strong>${asset.name}</strong> ` +
        `<span class="${verified ? "ok" : "bad"}">` +
        `${verified ? "哈希已核验" : "哈希不符"}</span></div>` +
        `<div class="meta">变体: ${asset.variants.join(", ")} · ` +
        `源轨迹采样 ${asset.trajectory_samples} 点 · 点击载入转盘场景</div>`;
      item.onclick = () => {
        state.scene = "dial";
        $("scene").value = "dial";
        schedule();
      };
      host.appendChild(item);
    }
  } catch (err) {
    host.textContent = "样片列表加载失败: " + String(err);
  }
}

function fillSelect(sel, values, chosen) {
  sel.innerHTML = "";
  for (const v of values) {
    const opt = document.createElement("option");
    opt.value = v;
    opt.textContent = v;
    if (v === chosen) opt.selected = true;
    sel.appendChild(opt);
  }
}

function buildKernelControls(schema) {
  const host = $("kernel-controls");
  host.innerHTML = "";
  for (const spec of schema) {
    const wrap = document.createElement("div");
    wrap.className = "control";
    const label = document.createElement("label");
    const name = document.createElement("span");
    name.textContent = spec.label;
    const val = document.createElement("span");
    val.className = "val";
    val.id = `val-${spec.name}`;
    label.append(name, val);

    let input;
    if (spec.readonly) {
      input = document.createElement("input");
      input.type = "number";
      input.readOnly = true;
      input.disabled = true;
      input.value = spec.value;
    } else if (Number.isInteger(spec.step)) {
      input = document.createElement("input");
      input.type = "range";
      input.min = spec.min;
      input.max = spec.max;
      input.step = spec.step;
      input.value = spec.value;
      addNumericSpin(input, spec, val);
    } else {
      input = document.createElement("input");
      input.type = "range";
      input.min = spec.min;
      input.max = spec.max;
      input.step = spec.step;
      input.value = spec.value;
      addNumericSpin(input, spec, val);
    }
    input.id = `ctl-${spec.name}`;
    input.dataset.name = spec.name;
    wrap.append(label, input);
    host.appendChild(wrap);
    val.textContent = formatValue(spec, input.value);
  }
}

function addNumericSpin(input, spec, val) {
  input.addEventListener("input", () => {
    val.textContent = formatValue(spec, input.value);
    if (spec.name === "shutter_angle_deg") {
      state.angle = parseFloat(input.value);
      syncExposureFromAngle();
    } else if (spec.name === "exposure") {
      const exposure = parseFloat(input.value);
      state.angle = 360.0 * exposure / state.config.frame_interval;
      syncAngleFromExposure();
    } else if (spec.name === "num_samples") {
      state.samples = parseInt(input.value, 10);
    }
    schedule();
  });
}

function formatValue(spec, raw) {
  const n = Number(raw);
  if (spec.unit === "s") return `${(n * 1000).toFixed(2)} ms`;
  if (spec.unit === "deg") return `${n.toFixed(0)}°`;
  return `${n}${spec.unit ? " " + spec.unit : ""}`;
}

function syncExposureFromAngle() {
  const exposure = state.angle / 360.0 * state.config.frame_interval;
  const ctl = $("ctl-exposure");
  const val = $("val-exposure");
  if (ctl) {
    ctl.value = exposure;
    val.textContent = formatValue(
      state.config.ui_schema.find((s) => s.name === "exposure"), exposure);
  }
}

function syncAngleFromExposure() {
  const ctl = $("ctl-shutter_angle_deg");
  const val = $("val-shutter_angle_deg");
  if (ctl) {
    ctl.value = state.angle;
    val.textContent = `${state.angle.toFixed(0)}°`;
  }
}

function query(extra = {}) {
  const p = new URLSearchParams({
    scene: state.scene,
    frame: state.frame,
    method: state.method,
    angle: state.angle,
    samples: state.samples,
    readout: state.readout,
    readout_dir: state.readoutDir,
    exposure_dir: state.exposureDir,
    profile: state.profile,
    paths: state.showPaths ? "1" : "0",
    ...extra,
  });
  return p.toString();
}

function schedule() {
  clearTimeout(state.pending);
  state.pending = setTimeout(() => render(false), 90);
}

function jumpFrame(delta) {
  const n = state.config.num_frames;
  state.frame = (state.frame + delta + n) % n;
  $("frame-slider").value = state.frame;
  schedule();
}

function togglePlay() {
  state.playing = !state.playing;
  const btn = $("play");
  if (state.playing) {
    btn.textContent = "⏸ 暂停 Pause";
    state.playTimer = setInterval(() => jumpFrame(1),
                                  state.config.frame_interval * 1000);
  } else {
    btn.textContent = "▶ 播放 Play";
    clearInterval(state.playTimer);
  }
}

async function render(initial) {
  if (state.abortCtrl) state.abortCtrl.abort();
  const ctrl = new AbortController();
  state.abortCtrl = ctrl;
  setBusy(true);
  try {
    const res = await fetch(`/api/render?${query()}`, { signal: ctrl.signal });
    const payload = await res.json();
    if (!res.ok) {
      showError(payload);
      return;
    }
    clearError();
    await loadImage(payload.image_png);
    updateMetrics(payload);
  } catch (err) {
    if (err.name !== "AbortError") showError({ error: String(err) });
  } finally {
    if (state.abortCtrl === ctrl) setBusy(false);
  }
}

function loadImage(url) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => {
      const canvas = $("result-canvas");
      canvas.width = img.naturalWidth;
      canvas.height = img.naturalHeight;
      const ctx = canvas.getContext("2d");
      ctx.drawImage(img, 0, 0);
      $("frame-tag").textContent =
        `frame ${state.frame} / ${state.config.num_frames - 1}`;
      resolve();
    };
    img.onerror = reject;
    img.src = url;
  });
}

function updateMetrics(payload) {
  $("m-rmse").textContent = payload.metrics.rmse_vs_gt.toFixed(5);
  $("m-contour").textContent = payload.metrics.contour_energy.toFixed(3);
  $("m-seg").textContent =
    payload.metrics.pixels_with_multiple_segments.toLocaleString();
  $("m-weight").textContent = payload.metrics.mean_sample_weight.toFixed(4);
  const k = payload.kernel;
  $("kernel-readout").textContent =
    `θ=${k.shutter_angle_deg.toFixed(1)}° · E=${(k.exposure * 1000)
      .toFixed(2)} ms · dt=${(k.frame_interval * 1000).toFixed(2)} ms ` +
    `· R=${(k.readout_duration * 1000).toFixed(2)} ms ` +
    `(${k.is_rolling ? "rolling" : "global"}) · ` +
    `轮廓=${k.exposure_profile} · 读出=${k.readout_direction} · ` +
    `曝光=${k.exposure_direction} · N=${k.num_samples}`;
  $("m-rowoffset").textContent =
    `${(k.row_offset_first * 1000).toFixed(2)} ms → ` +
    `${(k.row_offset_last * 1000).toFixed(2)} ms`;
}

function setBusy(on) {
  $("busy").innerHTML = on ? '<span class="spin"></span>积分中 integrating…' : "";
}

function showError(payload) {
  const box = $("error-box");
  box.classList.add("show");
  $("error-text").textContent =
    `${payload.type ? "(" + payload.type + ") " : ""}${payload.error}`;
}

function clearError() {
  $("error-box").classList.remove("show");
}

function downloadFrame() {
  window.open(`/api/frame?${query()}`, "_blank");
}

async function exportClip(kind) {
  setBusy(true);
  try {
    const res = await fetch(`/api/export?${query({ kind })}`);
    if (!res.ok) {
      showError(await res.json());
      return;
    }
    const blob = await res.blob();
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `shutter_${state.scene}_${state.method}.${kind}`;
    a.click();
    URL.revokeObjectURL(a.href);
  } catch (err) {
    showError({ error: String(err) });
  } finally {
    setBusy(false);
  }
}

init().catch((err) => showError({ error: String(err) }));
