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
    label.textContent = "Model (tool stream)";
    el.insertBefore(label, el.firstChild);

    const body = document.createElement("div");
    body.className = "stream-body";
    el.appendChild(body);

    const es = new EventSource(`/sse/${sid}`);
    es.addEventListener("message", (ev) => {
      body.textContent += ev.data;
    });
    es.addEventListener("done", () => {
      el.classList.remove("streaming");
      label.textContent = "Model";
      es.close();
    });
    es.addEventListener("error", () => {
      el.classList.remove("streaming");
      label.textContent = "Error";
      if (!body.textContent) body.textContent = "(stream failed)";
      es.close();
    });
  });
}

function initAsyncToolbotHtmxDemo() {
  attachSseStreams(document.getElementById("chat"));
  document.body.addEventListener("htmx:afterSwap", (evt) => {
    const t = evt.detail && evt.detail.target;
    if (t) attachSseStreams(t);
  });
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", initAsyncToolbotHtmxDemo);
} else {
  initAsyncToolbotHtmxDemo();
}
