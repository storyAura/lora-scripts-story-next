/**
 * Quick Anima inference panel.
 *
 * Injected via sd-trainer-brand.js (both served no-cache). Sidebar entry is
 * appended at the end of the 「训练」 group's children (same hydration rule as
 * the training queue). Disabled while training occupies the GPU.
 */
(function () {
  if (window.__sdTrainerInferLoaded) return;
  window.__sdTrainerInferLoaded = true;

  const OVERLAY_ID = "sd-infer-overlay";
  const STYLE_ID = "sd-infer-style";
  const NAV_ID = "sd-infer-nav";
  const TOAST_ID = "sd-infer-toast";
  const POLL_MS = 2000;

  let status = null;
  let overlayOpen = false;
  let pollTimer = null;
  let lastTaskId = null;
  let loraInfo = null;
  let loraInfoSeq = 0;

  function isEnglish() {
    return (document.documentElement.dataset.sdUiLocale || "") === "en-US";
  }

  function normalize(text) {
    return (text || "").replace(/\s+/g, " ").trim();
  }

  function escapeHtml(text) {
    return String(text == null ? "" : text)
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;");
  }

  async function api(method, url, body) {
    const res = await fetch(url, {
      method: method,
      headers: body !== undefined ? { "Content-Type": "application/json" } : undefined,
      body: body !== undefined ? JSON.stringify(body) : undefined,
    });
    return res.json();
  }

  function toast(message, ok) {
    let el = document.getElementById(TOAST_ID);
    if (!el) {
      el = document.createElement("div");
      el.id = TOAST_ID;
      document.body.appendChild(el);
    }
    el.textContent = message;
    el.className = ok === false ? "sd-infer-toast-error" : "sd-infer-toast-ok";
    el.style.opacity = "1";
    clearTimeout(el._hideTimer);
    el._hideTimer = setTimeout(() => { el.style.opacity = "0"; }, 3200);
  }

  function injectStyle() {
    if (document.getElementById(STYLE_ID)) return;
    const style = document.createElement("style");
    style.id = STYLE_ID;
    style.textContent = `
#${NAV_ID} {
  margin-top: 0.45rem; padding-top: 0.45rem;
  border-top: 1px dashed var(--c-border, #dcdfe6);
}
#${NAV_ID}.sd-infer-disabled a { opacity: .55; }
#${TOAST_ID} {
  position: fixed; right: 22px; bottom: 22px; z-index: 3200;
  max-width: 360px; padding: 10px 14px; border-radius: 10px;
  font-size: 13px; line-height: 1.45; pointer-events: none;
  transition: opacity .25s ease; opacity: 0;
  box-shadow: 0 8px 24px rgba(45,36,17,.18);
}
#${TOAST_ID}.sd-infer-toast-ok { background: #574d38; color: #fff7df; }
#${TOAST_ID}.sd-infer-toast-error { background: #8b3a3a; color: #fff7df; }

#${OVERLAY_ID} {
  position: fixed; top: 0; right: 0; bottom: 0; left: 0;
  z-index: 1800; display: flex; flex-direction: column;
  background: #ffffff; color: #2d2411; font-family: inherit;
}
html.dark #${OVERLAY_ID} { background: #1a130a; color: #f1e5cd; }
#${OVERLAY_ID} .sdi-head {
  display: flex; align-items: center; gap: 14px; padding: 16px 26px;
  border-bottom: 1px solid #f1e5cd; flex: none;
}
html.dark #${OVERLAY_ID} .sdi-head { border-color: #574d38; }
#${OVERLAY_ID} .sdi-title { font-size: 19px; font-weight: 700; }
#${OVERLAY_ID} .sdi-state { font-size: 13px; color: #847964; }
#${OVERLAY_ID} .sdi-close {
  margin-left: auto; border: 1px solid #dfd4bc; background: transparent; color: inherit;
  border-radius: 8px; padding: 5px 14px; font-size: 14px; cursor: pointer;
}
#${OVERLAY_ID} .sdi-close:hover { background: #fff7df; }
html.dark #${OVERLAY_ID} .sdi-close:hover { background: #2d2411; }
#${OVERLAY_ID} .sdi-body {
  flex: 1; overflow-y: auto; padding: 18px 26px 30px;
  display: grid; grid-template-columns: minmax(280px, 420px) 1fr; gap: 22px;
  max-width: 1180px;
}
@media (max-width: 900px) {
  #${OVERLAY_ID} .sdi-body { grid-template-columns: 1fr; }
}
#${OVERLAY_ID} .sdi-form label {
  display: block; font-size: 12px; color: #847964; margin: 10px 0 4px;
}
#${OVERLAY_ID} .sdi-form input, #${OVERLAY_ID} .sdi-form select, #${OVERLAY_ID} .sdi-form textarea {
  width: 100%; box-sizing: border-box; border: 1px solid #dfd4bc; border-radius: 8px;
  padding: 7px 10px; background: #fff; color: inherit; font: inherit;
}
html.dark #${OVERLAY_ID} .sdi-form input,
html.dark #${OVERLAY_ID} .sdi-form select,
html.dark #${OVERLAY_ID} .sdi-form textarea {
  background: #211809; border-color: #574d38;
}
#${OVERLAY_ID} .sdi-form textarea { min-height: 72px; resize: vertical; }
#${OVERLAY_ID} .sdi-row { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
#${OVERLAY_ID} .sdi-actions { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 16px; }
#${OVERLAY_ID} .sdi-btn {
  border: 1px solid #dfd4bc; border-radius: 8px; padding: 7px 16px; cursor: pointer;
  background: #ffffff; color: #574d38; font-size: 13px;
}
#${OVERLAY_ID} .sdi-btn.primary { background: #574d38; border-color: #574d38; color: #fff7df; }
#${OVERLAY_ID} .sdi-btn:disabled { opacity: .45; cursor: not-allowed; }
html.dark #${OVERLAY_ID} .sdi-btn { background: transparent; color: #dfd4bc; border-color: #574d38; }
html.dark #${OVERLAY_ID} .sdi-btn.primary { background: #dfd4bc; border-color: #dfd4bc; color: #2d2411; }
#${OVERLAY_ID} .sdi-warn {
  margin-top: 12px; padding: 10px 12px; border-radius: 10px; font-size: 13px; line-height: 1.5;
  background: #fff7df; border: 1px solid #f1e5cd; color: #574d38;
}
html.dark #${OVERLAY_ID} .sdi-warn { background: #241b0f; border-color: #574d38; color: #dfd4bc; }
#${OVERLAY_ID} .sdi-meta {
  margin: 6px 0 2px; font-size: 12px; line-height: 1.5; color: #847964;
}
#${OVERLAY_ID} .sdi-meta.bad { color: #8b3a3a; }
html.dark #${OVERLAY_ID} .sdi-meta.bad { color: #f0a0a0; }
#${OVERLAY_ID} .sdi-gallery {
  min-height: 220px; border: 1px dashed #dfd4bc; border-radius: 12px; padding: 14px;
}
html.dark #${OVERLAY_ID} .sdi-gallery { border-color: #574d38; }
#${OVERLAY_ID} .sdi-gallery img {
  max-width: 100%; border-radius: 10px; display: block; margin-bottom: 12px;
  border: 1px solid #f1e5cd;
}
html.dark #${OVERLAY_ID} .sdi-gallery img { border-color: #574d38; }
#${OVERLAY_ID} .sdi-empty { color: #847964; font-size: 13px; line-height: 1.6; }
`;
    document.head.appendChild(style);
  }

  function ensureNav() {
    const existing = document.getElementById(NAV_ID);
    if (existing) {
      if (existing.querySelector('a[href="#sd-infer"]')) {
        renderNavState();
        return;
      }
      existing.removeAttribute("id");
    }
    const sidebar = document.querySelector(".sidebar .sidebar-items");
    if (!sidebar) return;
    let trainChildren = null;
    sidebar.querySelectorAll(":scope > li").forEach((li) => {
      if (trainChildren) return;
      const heading = li.querySelector(":scope > p.sidebar-item.sidebar-heading");
      const text = normalize(heading && heading.textContent);
      if (text === "训练" || text === "Training") {
        trainChildren = li.querySelector(":scope > ul.sidebar-item-children");
      }
    });
    if (!trainChildren) return;

    const en = isEnglish();
    const li = document.createElement("li");
    li.id = NAV_ID;
    const a = document.createElement("a");
    a.className = "sidebar-item sidebar-heading";
    a.href = "#sd-infer";
    a.setAttribute("aria-label", en ? "Quick Infer" : "快速推理");
    a.appendChild(document.createTextNode(en ? " Quick Infer " : " 快速推理 "));
    a.addEventListener("click", (ev) => {
      ev.preventDefault();
      ev.stopPropagation();
      openOverlay();
    });
    li.appendChild(a);
    trainChildren.appendChild(li);
    renderNavState();
  }

  function renderNavState() {
    const nav = document.getElementById(NAV_ID);
    if (!nav) return;
    nav.classList.toggle("sd-infer-disabled", !!(status && status.busy_training));
  }

  function formValues() {
    const root = document.getElementById(OVERLAY_ID);
    if (!root) return {};
    const g = (id) => {
      const el = root.querySelector("#" + id);
      return el ? el.value : "";
    };
    return {
      lora_path: g("sdi-lora"),
      dit: g("sdi-dit"),
      vae: g("sdi-vae"),
      text_encoder: g("sdi-te"),
      prompt: g("sdi-prompt"),
      negative_prompt: g("sdi-neg"),
      width: Number(g("sdi-w") || 1024),
      height: Number(g("sdi-h") || 1024),
      steps: Number(g("sdi-steps") || 40),
      cfg: Number(g("sdi-cfg") || 4.5),
      seed: g("sdi-seed"),
      flow_shift: Number(g("sdi-fs") || 5),
      scheduler: g("sdi-sch") || "simple",
      sampler: g("sdi-ss") || "euler",
      lora_multiplier: Number(g("sdi-mult") || 1),
      attn_mode: g("sdi-attn") || "torch",
    };
  }

  function ensureOverlay() {
    if (document.getElementById(OVERLAY_ID)) return;
    const en = isEnglish();
    const el = document.createElement("div");
    el.id = OVERLAY_ID;
    el.style.display = "none";
    el.innerHTML = `
<div class="sdi-head">
  <span class="sdi-title">${en ? "Quick Infer" : "快速推理"}</span>
  <span class="sdi-state"></span>
  <button type="button" class="sdi-close">${en ? "Close" : "关闭"}</button>
</div>
<div class="sdi-body">
  <div class="sdi-form">
    <label>${en ? "Recent LoRA" : "近期 LoRA"}</label>
    <select id="sdi-lora"></select>
    <div class="sdi-meta" id="sdi-meta"></div>
    <label>${en ? "DiT / base model" : "DiT / 底模"}</label>
    <input id="sdi-dit" />
    <label>VAE</label>
    <input id="sdi-vae" />
    <label>${en ? "Text encoder (qwen3)" : "文本编码器 (qwen3)"}</label>
    <input id="sdi-te" />
    <label>${en ? "Prompt" : "正面提示词"}</label>
    <textarea id="sdi-prompt"></textarea>
    <label>${en ? "Negative prompt" : "负面提示词"}</label>
    <textarea id="sdi-neg"></textarea>
    <div class="sdi-row">
      <div><label>W</label><input id="sdi-w" type="number" min="256" step="64" /></div>
      <div><label>H</label><input id="sdi-h" type="number" min="256" step="64" /></div>
    </div>
    <div class="sdi-row">
      <div><label>Steps</label><input id="sdi-steps" type="number" min="1" /></div>
      <div><label>CFG</label><input id="sdi-cfg" type="number" min="0" step="0.1" /></div>
    </div>
    <div class="sdi-row">
      <div><label>flow_shift</label><input id="sdi-fs" type="number" min="0" step="0.1" /></div>
      <div><label>Seed</label><input id="sdi-seed" placeholder="random" /></div>
    </div>
    <div class="sdi-row">
      <div>
        <label>${en ? "Scheduler" : "调度器"}</label>
        <select id="sdi-sch">
          <option value="simple">simple</option>
          <option value="beta">beta</option>
          <option value="normal">normal</option>
        </select>
      </div>
      <div>
        <label>${en ? "Sampler" : "采样器"}</label>
        <select id="sdi-ss">
          <option value="euler">euler</option>
          <option value="heun">heun</option>
        </select>
      </div>
    </div>
    <div class="sdi-row">
      <div><label>${en ? "LoRA strength" : "LoRA 倍率"}</label><input id="sdi-mult" type="number" min="0" step="0.05" /></div>
      <div>
        <label>Attn</label>
        <select id="sdi-attn">
          <option value="torch">torch</option>
          <option value="sdpa">sdpa</option>
          <option value="xformers">xformers</option>
          <option value="flash">flash</option>
        </select>
      </div>
    </div>
    <div class="sdi-warn" style="display:none"></div>
    <div class="sdi-actions">
      <button type="button" class="sdi-btn primary" id="sdi-run">${en ? "Generate" : "生成"}</button>
      <button type="button" class="sdi-btn" id="sdi-stop">${en ? "Stop" : "停止"}</button>
      <button type="button" class="sdi-btn" id="sdi-refresh">${en ? "Refresh LoRAs" : "刷新列表"}</button>
    </div>
  </div>
  <div class="sdi-gallery">
    <div class="sdi-empty">${en ? "Results will appear here." : "生成结果会显示在这里。"}</div>
  </div>
</div>`;
    document.body.appendChild(el);
    el.querySelector(".sdi-close").addEventListener("click", closeOverlay);
    el.querySelector("#sdi-run").addEventListener("click", onRun);
    el.querySelector("#sdi-stop").addEventListener("click", onStop);
    el.querySelector("#sdi-refresh").addEventListener("click", () => refreshStatus(true));
    el.querySelector("#sdi-lora").addEventListener("change", () => autoFillFromSelectedLora(true));
  }

  function setField(id, val, force) {
    const root = document.getElementById(OVERLAY_ID);
    if (!root || val == null || val === "") return;
    const el = root.querySelector("#" + id);
    if (!el) return;
    if (force || el.tagName === "SELECT" || !el.value) el.value = String(val);
  }

  function renderLoraMeta() {
    const root = document.getElementById(OVERLAY_ID);
    if (!root) return;
    const box = root.querySelector("#sdi-meta");
    if (!box) return;
    const en = isEnglish();
    if (!loraInfo) {
      box.className = "sdi-meta";
      box.textContent = "";
      return;
    }
    const bits = [];
    bits.push(en ? `Type: ${loraInfo.family_label}` : `类型：${loraInfo.family_label}`);
    if (loraInfo.network_module) bits.push(loraInfo.network_module);
    if (loraInfo.network_algo) bits.push(`algo=${loraInfo.network_algo}`);
    if (loraInfo.ss_sd_model_name) {
      bits.push((en ? "base " : "底模 ") + loraInfo.ss_sd_model_name);
    }
    if (!loraInfo.has_training_metadata) {
      bits.push(en ? "(no training metadata)" : "（无训练元数据，无法可靠识别）");
    }
    const notes = (loraInfo.suggested && loraInfo.suggested.notes) || [];
    box.className = "sdi-meta" + (loraInfo.supported ? "" : " bad");
    box.textContent = bits.join(" · ") + (notes.length ? " — " + notes.join("；") : "");
  }

  async function autoFillFromSelectedLora(forcePaths) {
    const root = document.getElementById(OVERLAY_ID);
    if (!root) return;
    const sel = root.querySelector("#sdi-lora");
    const path = sel && sel.value;
    if (!path) {
      loraInfo = null;
      renderLoraMeta();
      renderStatus();
      return;
    }
    const seq = ++loraInfoSeq;
    try {
      const json = await api("GET", "/api/infer/lora-info?path=" + encodeURIComponent(path));
      if (seq !== loraInfoSeq) return;
      if (!json || json.status !== "success") {
        loraInfo = null;
        renderLoraMeta();
        renderStatus();
        return;
      }
      loraInfo = json.data || null;
      const sug = (loraInfo && loraInfo.suggested) || {};
      setField("sdi-dit", sug.dit, forcePaths);
      setField("sdi-vae", sug.vae, forcePaths);
      setField("sdi-te", sug.text_encoder, forcePaths);
      renderLoraMeta();
      renderStatus();
    } catch (_err) {
      if (seq !== loraInfoSeq) return;
      loraInfo = null;
      renderLoraMeta();
    }
  }

  function applyDefaults(data) {
    const root = document.getElementById(OVERLAY_ID);
    if (!root || !data) return;
    const d = data.defaults || {};
    setField("sdi-dit", d.dit, false);
    setField("sdi-vae", d.vae, false);
    setField("sdi-te", d.text_encoder, false);
    setField("sdi-w", d.width, false);
    setField("sdi-h", d.height, false);
    setField("sdi-steps", d.steps, false);
    setField("sdi-cfg", d.cfg, false);
    setField("sdi-fs", d.flow_shift, false);
    setField("sdi-sch", d.scheduler, false);
    setField("sdi-ss", d.sampler, false);
    setField("sdi-mult", 1, false);
    if (!root.querySelector("#sdi-prompt").value) {
      root.querySelector("#sdi-prompt").value = "1girl, solo";
    }

    const sel = root.querySelector("#sdi-lora");
    const prev = sel.value;
    const loras = data.recent_loras || [];
    sel.innerHTML = loras.length
      ? loras.map((l) => {
          const miss = l.missing ? " (missing)" : "";
          return `<option value="${escapeHtml(l.path)}"${l.missing ? " disabled" : ""}>${escapeHtml(l.name)}${miss}</option>`;
        }).join("")
      : `<option value="">${isEnglish() ? "(no recent LoRA under output/)" : "（output/ 下暂无近期 LoRA）"}</option>`;
    if (prev && [...sel.options].some((o) => o.value === prev)) sel.value = prev;
    autoFillFromSelectedLora(true);
  }

  function renderStatus() {
    const root = document.getElementById(OVERLAY_ID);
    if (!root || !status) return;
    const en = isEnglish();
    const stateEl = root.querySelector(".sdi-state");
    const warn = root.querySelector(".sdi-warn");
    const runBtn = root.querySelector("#sdi-run");
    const stopBtn = root.querySelector("#sdi-stop");

    let line = en ? "Idle" : "空闲";
    if (status.busy_training) line = en ? "Training occupies GPU" : "训练占用 GPU";
    else if (status.busy_infer) line = en ? "Inferring…" : "推理中…";
    stateEl.textContent = line;

    if (status.busy_training) {
      warn.style.display = "";
      warn.textContent = en
        ? "Quick infer is disabled while training is using the GPU."
        : "训练正在占用 GPU，快速推理已禁用。请等训练结束后再试。";
      runBtn.disabled = true;
    } else if (loraInfo && loraInfo.supported === false) {
      warn.style.display = "";
      warn.textContent = loraInfo.warning
        || (en ? "Quick infer currently supports Anima LoRAs only." : "快速推理目前仅支持 Anima LoRA。");
      runBtn.disabled = true;
    } else {
      warn.style.display = "none";
      runBtn.disabled = !!status.busy_infer;
    }
    stopBtn.disabled = !status.busy_infer;
    renderNavState();
  }

  async function refreshImages() {
    const root = document.getElementById(OVERLAY_ID);
    if (!root || !lastTaskId) return;
    try {
      const json = await api("GET", `/api/infer/images/${encodeURIComponent(lastTaskId)}`);
      if (!json || json.status !== "success") return;
      const names = (json.data && json.data.images) || [];
      const gallery = root.querySelector(".sdi-gallery");
      if (!names.length) {
        if (json.data && json.data.error) {
          gallery.innerHTML = `<div class="sdi-empty">${escapeHtml(json.data.error)}</div>`;
        }
        return;
      }
      gallery.innerHTML = names
        .map((n) => `<img src="/api/infer/image/${encodeURIComponent(lastTaskId)}/${encodeURIComponent(n)}?t=${Date.now()}" alt="${escapeHtml(n)}" />`)
        .join("");
      if (json.data && json.data.error) {
        toast(json.data.error, false);
      }
    } catch (_err) {
      /* ignore poll errors */
    }
  }

  async function refreshStatus(forceApply) {
    try {
      const json = await api("GET", "/api/infer/status");
      if (!json || json.status !== "success") return;
      status = json.data || {};
      if (status.task_id) lastTaskId = status.task_id;
      if (forceApply || overlayOpen) applyDefaults(status);
      if (overlayOpen) {
        renderStatus();
        if (lastTaskId) await refreshImages();
      } else {
        renderNavState();
      }
    } catch (_err) {
      /* ignore */
    }
  }

  async function onRun() {
    if (status && status.busy_training) {
      toast(isEnglish() ? "Disabled while training." : "训练占用 GPU，无法推理。", false);
      return;
    }
    if (loraInfo && loraInfo.supported === false) {
      toast(loraInfo.warning || (isEnglish() ? "Anima LoRA only." : "仅支持 Anima LoRA。"), false);
      return;
    }
    const body = formValues();
    if (!body.lora_path) {
      toast(isEnglish() ? "Select a LoRA first." : "请先选择 LoRA。", false);
      return;
    }
    try {
      const json = await api("POST", "/api/infer/run", body);
      if (!json) return;
      if (json.message) toast(json.message, json.status === "success");
      if (json.status === "success" && json.data) {
        lastTaskId = json.data.task_id;
        const gallery = document.querySelector(`#${OVERLAY_ID} .sdi-gallery`);
        if (gallery) {
          gallery.innerHTML = `<div class="sdi-empty">${isEnglish() ? "Generating…" : "生成中…"}</div>`;
        }
      }
      await refreshStatus(false);
    } catch (err) {
      toast((isEnglish() ? "Infer request failed: " : "推理请求失败：") + err, false);
    }
  }

  async function onStop() {
    try {
      const json = await api("POST", "/api/infer/terminate", { task_id: lastTaskId });
      if (json && json.message) toast(json.message, json.status === "success");
      await refreshStatus(false);
    } catch (err) {
      toast(String(err), false);
    }
  }

  function openOverlay() {
    injectStyle();
    ensureOverlay();
    const el = document.getElementById(OVERLAY_ID);
    el.style.display = "flex";
    overlayOpen = true;
    if (location.hash !== "#sd-infer") {
      try { history.replaceState(null, "", "#sd-infer"); } catch (_e) { location.hash = "#sd-infer"; }
    }
    refreshStatus(true);
  }

  function closeOverlay() {
    const el = document.getElementById(OVERLAY_ID);
    if (el) el.style.display = "none";
    overlayOpen = false;
    if (location.hash === "#sd-infer") {
      try { history.replaceState(null, "", location.pathname + location.search); } catch (_e) { /* ignore */ }
    }
  }

  function boot() {
    injectStyle();
    ensureNav();
    refreshStatus(false);
    if (location.hash === "#sd-infer") openOverlay();
    window.addEventListener("hashchange", () => {
      if (location.hash === "#sd-infer") openOverlay();
    });
    pollTimer = setInterval(() => {
      ensureNav();
      refreshStatus(false);
    }, POLL_MS);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }
})();
