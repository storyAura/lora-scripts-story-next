/**
 * Trainer settings page: persist disk-preflight / Hugging Face token to the
 * backend so /api/run and the training queue see the same values.
 *
 * Loaded no-cache from sd-trainer-brand.js.
 */
(function () {
  var PATH_RE = /\/other\/settings(?:\.html)?(?:$|[?#])/;
  if (!PATH_RE.test(location.pathname + location.search)) return;

  var API = "/api/trainer-settings";
  var AUTOSAVE_KEY = "configs-settings-autosave";
  var HISTORY_KEY = "configs-settings";
  var lastSent = "";

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

  function currentLocal() {
    return parseStore(localStorage.getItem(AUTOSAVE_KEY));
  }

  function mergeSettings(server, local) {
    var out = {};
    var src = server && typeof server === "object" ? server : {};
    var loc = local && typeof local === "object" ? local : {};
    var keys = [
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
    for (var i = 0; i < keys.length; i++) {
      var key = keys[i];
      if (Object.prototype.hasOwnProperty.call(loc, key) && loc[key] !== undefined && loc[key] !== "") {
        out[key] = loc[key];
      } else if (Object.prototype.hasOwnProperty.call(src, key)) {
        out[key] = src[key];
      }
    }
    if (out.disk_preflight_enabled === undefined) out.disk_preflight_enabled = true;
    return Object.assign({}, src, loc, out);
  }

  function putSettings(payload) {
    var body = JSON.stringify(payload || {});
    if (body === lastSent) return Promise.resolve();
    lastSent = body;
    return fetch(API, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: body,
    }).then(function (res) {
      return res.json().catch(function () {
        return { status: "fail" };
      });
    });
  }

  function relabelTrainButton() {
    var buttons = document.querySelectorAll(".el-button, button");
    for (var i = 0; i < buttons.length; i++) {
      var el = buttons[i];
      var text = (el.textContent || "").replace(/\s+/g, "");
      if (text.indexOf("开始训练") >= 0 || text.indexOf("Starttraining") >= 0 || text.indexOf("StartTraining") >= 0) {
        el.textContent = "保存训练器设置";
      }
    }
  }

  var origFetch = window.fetch;
  window.fetch = function (url, opts) {
    var href = typeof url === "string" ? url : url && url.url;
    if (href && href.indexOf("/api/run") >= 0) {
      var body = opts && opts.body ? opts.body : JSON.stringify(currentLocal());
      var payload;
      try {
        payload = typeof body === "string" ? JSON.parse(body) : body;
      } catch (e) {
        payload = currentLocal();
      }
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

  fetch(API)
    .then(function (res) {
      return res.json();
    })
    .then(function (json) {
      if (!json || json.status !== "success" || !json.data) return;
      var merged = mergeSettings(json.data, currentLocal());
      localStorage.setItem(AUTOSAVE_KEY, JSON.stringify(merged));
      lastSent = JSON.stringify(merged);
    })
    .catch(function () {
      /* backend offline */
    });

  var timer = null;
  function scheduleSync() {
    if (timer) clearTimeout(timer);
    timer = setTimeout(function () {
      putSettings(currentLocal());
    }, 400);
  }

  window.addEventListener("storage", function (ev) {
    if (ev.key === AUTOSAVE_KEY || ev.key === HISTORY_KEY) scheduleSync();
  });
  document.addEventListener("change", scheduleSync, true);
  document.addEventListener("input", scheduleSync, true);

  function boot() {
    relabelTrainButton();
    var app = document.getElementById("app") || document.body;
    new MutationObserver(relabelTrainButton).observe(app, { childList: true, subtree: true });
  }
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }
})();
