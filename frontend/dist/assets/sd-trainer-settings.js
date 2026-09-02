/**
 * Trainer settings page: persist disk-preflight / Hugging Face token to the
 * backend so /api/run and the training queue see the same values.
 *
 * Loaded no-cache from sd-trainer-brand.js on every page. Must keep working
 * after SPA navigation — users usually open Home first, then click
 * 「训练器设置」 (`/other/settings.md`). An early pathname return would leave
 * the switch display-only.
 */
(function () {
  if (window.__sdTrainerSettingsLoaded) return;
  window.__sdTrainerSettingsLoaded = true;

  var API = "/api/trainer-settings";
  var AUTOSAVE_KEY = "configs-settings-autosave";
  var HISTORY_KEY = "configs-settings";
  var KEYS = [
    "disk_preflight_enabled",
    "tensorboard_url",
    "huggingface_token",
    "huggingface_repo_id",
    "huggingface_path_in_repo",
    "huggingface_repo_visibility",
    "huggingface_repo_type",
    "async_upload",
    "save_state_to_huggingface",
  ];
  var lastSent = "";
  var hydrating = false;
  var observedPath = "";
  var syncTimer = null;
  var origFetch = window.fetch;

  function isSettingsPage() {
    return /\/other\/settings(\.html|\.md)?\/?$/.test(location.pathname || "");
  }

  function parseStore(raw) {
    if (!raw) return {};
    try {
      var parsed = JSON.parse(raw);
      if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) return parsed;
      if (Array.isArray(parsed) && parsed[0] && parsed[0].config) return parsed[0].config;
    } catch (e) {
      /* ignore */
    }
    return {};
  }

  function boolValue(value, fallback) {
    if (value === true || value === false) return value;
    if (value === 1 || value === 0) return value === 1;
    if (value == null || value === "") return fallback;
    var text = String(value).trim().toLowerCase();
    if (text === "1" || text === "true" || text === "yes" || text === "on") return true;
    if (text === "0" || text === "false" || text === "no" || text === "off") return false;
    return fallback;
  }

  function normalizePayload(raw, fallbackPreflight) {
    var src = raw && typeof raw === "object" ? raw : {};
    var out = {};
    for (var i = 0; i < KEYS.length; i++) {
      var key = KEYS[i];
      if (src[key] !== undefined && src[key] !== null) out[key] = src[key];
    }
    out.disk_preflight_enabled = boolValue(out.disk_preflight_enabled, fallbackPreflight);
    if (out.async_upload !== undefined) out.async_upload = boolValue(out.async_upload, false);
    if (out.save_state_to_huggingface !== undefined) {
      out.save_state_to_huggingface = boolValue(out.save_state_to_huggingface, false);
    }
    return out;
  }

  function findItem(field) {
    return Array.from(document.querySelectorAll(".example-container .k-schema-item")).find(function (row) {
      var title = row.querySelector("h3");
      var text = ((title && title.textContent) || "").replace(/\s+/g, "");
      return text.indexOf(field) !== -1;
    });
  }

  function readSwitchOn(row) {
    if (!row) return null;
    var input = row.querySelector('input[type="checkbox"]');
    if (input) return !!input.checked;
    var sw = row.querySelector(".el-switch");
    return sw ? sw.classList.contains("is-checked") : null;
  }

  function setSwitchOn(row, on) {
    if (!row) return;
    if (readSwitchOn(row) === on) return;
    var sw = row.querySelector(".el-switch");
    if (sw) sw.click();
  }

  function setInputValue(row, value) {
    if (!row) return;
    var input = row.querySelector("input.el-input__inner, textarea, input");
    if (!input) return;
    var next = value == null ? "" : String(value);
    if (input.value === next) return;
    var proto = input.tagName === "TEXTAREA" ? window.HTMLTextAreaElement.prototype : window.HTMLInputElement.prototype;
    var desc = Object.getOwnPropertyDescriptor(proto, "value");
    if (desc && desc.set) desc.set.call(input, next);
    else input.value = next;
    input.dispatchEvent(new Event("input", { bubbles: true }));
    input.dispatchEvent(new Event("change", { bubbles: true }));
  }

  function findFormModelRef() {
    var el = document.querySelector(".example-container");
    if (!el) return null;
    var inst = el.__vueParentComponent;
    while (inst) {
      var setup = inst.setupState || {};
      var names = Object.keys(setup);
      for (var i = 0; i < names.length; i++) {
        var val = setup[names[i]];
        var inner = val && typeof val === "object" && "value" in val ? val.value : null;
        if (!inner || typeof inner !== "object" || Array.isArray(inner)) continue;
        if (
          Object.prototype.hasOwnProperty.call(inner, "disk_preflight_enabled") ||
          Object.prototype.hasOwnProperty.call(inner, "tensorboard_url") ||
          Object.prototype.hasOwnProperty.call(inner, "huggingface_token") ||
          Object.prototype.hasOwnProperty.call(inner, "huggingface_repo_id")
        ) {
          return val;
        }
      }
      inst = inst.parent;
    }
    return null;
  }

  function applyToForm(payload) {
    hydrating = true;
    try {
      var modelRef = findFormModelRef();
      if (modelRef && modelRef.value && typeof modelRef.value === "object") {
        modelRef.value = Object.assign({}, modelRef.value, payload);
      }
      setSwitchOn(findItem("disk_preflight_enabled"), payload.disk_preflight_enabled !== false);
      setSwitchOn(findItem("async_upload"), !!payload.async_upload);
      setSwitchOn(findItem("save_state_to_huggingface"), !!payload.save_state_to_huggingface);
      ["tensorboard_url", "huggingface_token", "huggingface_repo_id", "huggingface_path_in_repo"].forEach(function (key) {
        if (payload[key] != null) setInputValue(findItem(key), payload[key]);
      });
      localStorage.setItem(AUTOSAVE_KEY, JSON.stringify(payload));
    } finally {
      window.setTimeout(function () {
        hydrating = false;
      }, 0);
    }
  }

  function readFromForm() {
    var modelRef = findFormModelRef();
    var model = modelRef && modelRef.value && typeof modelRef.value === "object" ? modelRef.value : {};
    var local = parseStore(localStorage.getItem(AUTOSAVE_KEY));
    var merged = Object.assign({}, local, model);
    var preflight = readSwitchOn(findItem("disk_preflight_enabled"));
    if (preflight !== null) merged.disk_preflight_enabled = preflight;
    var asyncOn = readSwitchOn(findItem("async_upload"));
    if (asyncOn !== null) merged.async_upload = asyncOn;
    var stateOn = readSwitchOn(findItem("save_state_to_huggingface"));
    if (stateOn !== null) merged.save_state_to_huggingface = stateOn;
    return normalizePayload(merged, true);
  }

  function putSettings(payload) {
    var normalized = normalizePayload(payload, true);
    var body = JSON.stringify(normalized);
    if (body === lastSent) {
      return Promise.resolve({
        status: "success",
        message: "训练器设置已保存",
        data: normalized,
      });
    }
    lastSent = body;
    return origFetch(API, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: body,
    }).then(function (res) {
      return res.json().catch(function () {
        return { status: "fail" };
      });
    }).then(function (json) {
      if (!json || json.status !== "success") lastSent = "";
      return json;
    });
  }

  function hydrateFromServer() {
    if (!isSettingsPage()) return;
    origFetch(API)
      .then(function (res) {
        return res.json();
      })
      .then(function (json) {
        if (!json || json.status !== "success" || !json.data) return;
        // Server file is the source of truth. Stale configs-settings-autosave
        // often stores disk_preflight_enabled=false because an unbound switch
        // renders off while the backend default is still on.
        var payload = normalizePayload(json.data, true);
        lastSent = JSON.stringify(payload);
        var tries = 0;
        function apply() {
          tries += 1;
          applyToForm(payload);
          var row = findItem("disk_preflight_enabled");
          var want = payload.disk_preflight_enabled !== false;
          var ok = row && readSwitchOn(row) === want;
          if (!ok && tries < 12) window.setTimeout(apply, 150);
        }
        apply();
      })
      .catch(function () {
        /* backend offline */
      });
  }

  function relabelTrainButton() {
    if (!isSettingsPage()) return;
    var buttons = document.querySelectorAll(".el-button, button");
    for (var i = 0; i < buttons.length; i++) {
      var el = buttons[i];
      var text = (el.textContent || "").replace(/\s+/g, "");
      if (
        text.indexOf("开始训练") >= 0 ||
        text.indexOf("Starttraining") >= 0 ||
        text.indexOf("StartTraining") >= 0
      ) {
        el.textContent = "保存训练器设置";
      }
    }
  }

  window.fetch = function (url, opts) {
    var href = typeof url === "string" ? url : url && url.url;
    if (isSettingsPage() && href && href.indexOf("/api/run") >= 0) {
      var body = opts && opts.body ? opts.body : JSON.stringify(readFromForm());
      var payload;
      try {
        payload = typeof body === "string" ? JSON.parse(body) : body;
      } catch (e) {
        payload = readFromForm();
      }
      var form = readFromForm();
      payload = Object.assign({}, form, normalizePayload(payload, form.disk_preflight_enabled));
      return putSettings(payload).then(function (json) {
        var ok = json && json.status === "success";
        var message = ok ? "训练器设置已保存" : (json && json.message) || "保存训练器设置失败";
        return new Response(
          JSON.stringify({
            status: ok ? "success" : "fail",
            message: message,
            data: { queue_message: message },
          }),
          { headers: { "Content-Type": "application/json" } }
        );
      });
    }
    return origFetch.apply(this, arguments);
  };

  function scheduleSync() {
    if (!isSettingsPage() || hydrating) return;
    if (syncTimer) clearTimeout(syncTimer);
    syncTimer = setTimeout(function () {
      putSettings(readFromForm());
    }, 400);
  }

  function onPageMaybeChanged() {
    var path = location.pathname || "";
    var pathChanged = path !== observedPath;
    observedPath = path;
    if (!isSettingsPage()) return;
    relabelTrainButton();
    if (pathChanged) hydrateFromServer();
  }

  window.addEventListener("storage", function (ev) {
    if (ev.key === AUTOSAVE_KEY || ev.key === HISTORY_KEY) scheduleSync();
  });
  document.addEventListener("change", scheduleSync, true);
  document.addEventListener("input", scheduleSync, true);
  document.addEventListener("click", function (ev) {
    if (!isSettingsPage() || hydrating) return;
    if (ev.target && ev.target.closest && ev.target.closest(".el-switch")) {
      scheduleSync();
    }
  }, true);

  function boot() {
    var app = document.getElementById("app") || document.body;
    new MutationObserver(function () {
      onPageMaybeChanged();
    }).observe(app, { childList: true, subtree: true });
    onPageMaybeChanged();
  }
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }
})();
