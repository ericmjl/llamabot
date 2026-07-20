/**
 * Append text without wiping existing child nodes (unlike `el.textContent +=`).
 */
function appendText(el, text) {
  el.appendChild(document.createTextNode(text));
}

/**
 * After HTMX swaps in a .stream-target, open EventSource for /sse/{id}.
 */
function attachSseStreams(root) {
  const scope = root && root.querySelectorAll ? root : document;
  const candidates = root && root.matches && root.matches("[data-sse-id]")
    ? [root]
    : root.querySelectorAll
      ? [...root.querySelectorAll("[data-sse-id]")]
      : [];

  candidates.forEach((el) => {
    if (!el || el.dataset.sseAttached === "1") return;
    const sid = el.dataset.sseId;
    if (!sid) return;
    el.dataset.sseAttached = "1";
    el.classList.add("streaming");

    const label = document.createElement("div");
    label.className = "streaming-label";
    label.textContent = "AsyncAgentBot (SSE)";
    el.insertBefore(label, el.firstChild);

    const body = document.createElement("div");
    body.className = "stream-body";
    el.appendChild(body);

    const es = new EventSource(`/sse/${sid}`);
    let streamFailed = false;
    es.addEventListener("status", (ev) => {
      const line = document.createElement("div");
      line.className = "sse-status";
      line.textContent = ev.data;
      body.appendChild(line);
    });
    es.addEventListener("message", (ev) => {
      appendText(body, ev.data);
    });
    es.addEventListener("stream_error", (ev) => {
      streamFailed = true;
      el.classList.remove("streaming");
      label.textContent = "Error";
      const err = document.createElement("div");
      err.className = "sse-error";
      err.textContent = ev.data || "(unknown error)";
      body.appendChild(err);
      es.close();
    });
    es.addEventListener("done", () => {
      el.classList.remove("streaming");
      if (!streamFailed) label.textContent = "AsyncAgentBot";
      es.close();
    });
    es.addEventListener("error", () => {
      if (es.readyState === EventSource.CLOSED) return;
      streamFailed = true;
      el.classList.remove("streaming");
      label.textContent = "Connection error";
      if (body.childNodes.length === 0) {
        appendText(body, "(SSE connection failed)");
      }
      es.close();
    });
  });
}

function initAsyncAgentbotHtmxDemo() {
  attachSseStreams(document.getElementById("chat"));
  document.body.addEventListener("htmx:afterSwap", (evt) => {
    const t = evt.detail && evt.detail.target;
    if (t) attachSseStreams(t);
  });
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", initAsyncAgentbotHtmxDemo);
} else {
  initAsyncAgentbotHtmxDemo();
}
