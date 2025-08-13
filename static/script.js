const $ = (sel) => document.querySelector(sel);

const CONFIG_KEYS = [
  "epochs",
  "initial_learning_rate",
  "learning_rate_scaling",
  "scaling_frequency",
  "validation_patience",
  "batch_size",
  "test_ratio",
  "val_ratio",
  "root_folder",
  "model_save_path",
  "reduce_dimensions",
  "gaussian_smoothing",
  "wavelength_range",
  "evaluation_graphics",
  "randomizer_seed"
];

function boolToStr(v) { return v ? "true" : "false"; }
function strToBool(s) { return String(s).toLowerCase() === "true"; }

async function fetchJSON(url, opts = {}) {
  const res = await fetch(url, opts);
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json();
}

async function loadConfigIntoForm() {
  const cfg = await fetchJSON("/config");
  for (const key of CONFIG_KEYS) {
    const input = document.querySelector(`[name="${key}"]`);
    if (!input) continue;

    let val = cfg[key];

    if (key === "wavelength_range" && Array.isArray(val)) {
      input.value = `${val[0]},${val[1]}`;
      continue;
    }
    if (input.tagName === "SELECT") {
      input.value = boolToStr(val);
    } else {
      input.value = val;
    }
  }
}

function formToPayload() {
  const payload = {};
  for (const key of CONFIG_KEYS) {
    const input = document.querySelector(`[name="${key}"]`);
    if (!input) continue;
    let val = input.value;

    if (key === "reduce_dimensions" || key === "gaussian_smoothing" || key === "evaluation_graphics") {
      val = strToBool(val);
    } else if (key === "wavelength_range") {
      const parts = val.split(",").map(s => s.trim());
      if (parts.length === 2) {
        val = [parseInt(parts[0], 10), parseInt(parts[1], 10)];
      }
    } else if (["epochs","scaling_frequency","validation_patience","batch_size","randomizer_seed"].includes(key)) {
      val = parseInt(val, 10);
    } else if (["initial_learning_rate","learning_rate_scaling","test_ratio","val_ratio"].includes(key)) {
      val = parseFloat(val);
    }
    payload[key] = val;
  }
  return payload;
}

function setStatus(text) {
  $("#train-status").textContent = text;
}

async function handleSaveConfig(e) {
  e.preventDefault();
  const payload = formToPayload();
  await fetchJSON("/config", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(payload)
  });
  setStatus("Config saved");
}

async function handleReload() {
  await loadConfigIntoForm();
  setStatus("Config reloaded");
}

async function handleUpload() {
  const f = $("#file-input").files[0];
  if (!f) return alert("Pick a .txt file first.");
  const form = new FormData();
  form.append("file", f);
  const res = await fetch("/upload-config", { method: "POST", body: form });
  if (!res.ok) {
    const txt = await res.text();
    return alert(`Upload failed: ${txt}`);
  }
  await loadConfigIntoForm();
  setStatus("Config loaded from file");
}

function handleDownload() {
  // Just navigate to endpoint; browser will download
  window.location.href = "/download-config";
}

let pollTimer = null;
let lastEpoch = -1;

async function pollStatus() {
  try {
    const s = await fetchJSON("/status");

    if (s.running) {
      setStatus("Training…");
    } else {
      setStatus("Idle");
    }

    const epochInfo = [];
    if (typeof s.epoch === "number") epochInfo.push(`epoch=${s.epoch}`);
    if (typeof s.loss === "number") epochInfo.push(`loss=${s.loss.toFixed(5)}`);
    if (typeof s.val_loss === "number") epochInfo.push(`val_loss=${s.val_loss.toFixed(5)}`);
    $("#epoch-info").textContent = epochInfo.join(" • ");

    // When training stops, fetch results once
    if (!s.running && lastEpoch !== s.epoch && s.epoch > 0) {
      await loadResults();
    }
    lastEpoch = s.epoch ?? lastEpoch;
  } catch (err) {
    console.error(err);
  }
}

async function handleStartTraining() {
  const resp = await fetchJSON("/train", { method: "POST" });
  if (resp.status === "already_running") {
    setStatus("Already running");
  } else {
    setStatus("Started");
  }
}

function renderConfusionMatrix(cm) {
  const wrap = $("#cm-table-wrapper");
  wrap.innerHTML = "";
  if (!Array.isArray(cm) || cm.length === 0) {
    wrap.textContent = "No confusion matrix.";
    return;
  }
  const table = document.createElement("table");
  const tbody = document.createElement("tbody");
  for (const row of cm) {
    const tr = document.createElement("tr");
    for (const cell of row) {
      const td = document.createElement("td");
      td.textContent = cell;
      tr.appendChild(td);
    }
    tbody.appendChild(tr);
  }
  table.appendChild(tbody);
  wrap.appendChild(table);
}

async function loadResults() {
  const r = await fetchJSON("/results");
  if (!r.ready) return;
  $("#accuracy").textContent = (typeof r.accuracy === "number") ? r.accuracy.toFixed(4) : "—";
  renderConfusionMatrix(r.confusion_matrix);
}

async function init() {
  $("#config-form").addEventListener("submit", handleSaveConfig);
  $("#btn-refresh").addEventListener("click", handleReload);
  $("#btn-upload").addEventListener("click", handleUpload);
  $("#btn-download").addEventListener("click", handleDownload);
  $("#btn-train").addEventListener("click", handleStartTraining);

  await loadConfigIntoForm();
  await loadResults(); // in case a previous run exists

  pollTimer = setInterval(pollStatus, 1500);
}
document.addEventListener("DOMContentLoaded", init);

async function updateProgress() {
    const response = await fetch('/training_status');
    const status = await response.json();

    const progressBar = document.getElementById('training-progress');
    const progressText = document.getElementById('progress-text');

    let percent = 0;
    if (status.running && status.total_epochs > 0) {
        percent = Math.floor((status.epoch / status.total_epochs) * 100);
        if (percent > 100) percent = 100;
        progressBar.value = percent;
        progressText.textContent = `${percent}%`;
    } else if (status.total_epochs > 0) {
        progressBar.value = 100;
        progressText.textContent = "100% (Completed)";
    } else {
        progressBar.value = 0;
        progressText.textContent = "0%";
    }

    setTimeout(updateProgress, 1000);
}

updateProgress();

