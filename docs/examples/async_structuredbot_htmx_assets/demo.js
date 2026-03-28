/**
 * async_structuredbot_htmx_demo — client-side logic.
 *
 * After HTMX swaps in _stream_panel.html, openStreamPanel() connects an
 * EventSource to /sse/{id}.  Each "token" event appends to the JSON preview;
 * on "done" the accumulated JSON is parsed and used to populate the profile
 * form fields on the right.
 */

/** Fill the read-only profile form from a parsed JobProfile object.
 *
 * @param {Record<string, unknown>} data - Parsed JSON from the bot.
 */
function populateForm(data) {
  const simpleFields = [
    "full_name",
    "email",
    "phone",
    "years_of_experience",
    "desired_role",
    "professional_summary",
  ];

  simpleFields.forEach((key) => {
    const el = document.getElementById(`f-${key}`);
    if (el && data[key] !== undefined && data[key] !== null) {
      el.value = String(data[key]);
    }
  });

  // Render skills as pill tags.
  const skillsEl = document.getElementById("f-top_skills");
  if (skillsEl && Array.isArray(data.top_skills)) {
    skillsEl.innerHTML = data.top_skills
      .map(
        (s) =>
          `<span class="skill-tag">${s
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")}</span>`
      )
      .join("");
  }
}

/** Open an SSE connection for the stream panel injected by HTMX.
 *
 * @param {HTMLElement} panel - The #stream-panel element with data-sse-id.
 */
function openStreamPanel(panel) {
  const sid = panel.dataset.sseId;
  if (!sid || panel.dataset.sseAttached === "1") return;
  panel.dataset.sseAttached = "1";

  const preview = document.getElementById("json-preview");
  const statusEl = document.getElementById("stream-status");
  let accumulated = "";

  const es = new EventSource(`/sse/${sid}`);

  es.addEventListener("token", (ev) => {
    accumulated += ev.data;
    if (preview) preview.textContent = accumulated;
  });

  es.addEventListener("done", () => {
    es.close();
    if (statusEl) {
      statusEl.textContent = "Done";
      const dots = panel.querySelector(".dots");
      if (dots) dots.style.display = "none";
    }
    try {
      const data = JSON.parse(accumulated);
      populateForm(data);
    } catch (err) {
      if (statusEl) statusEl.textContent = "Parse error — see console";
      console.error("Failed to parse structured JSON:", err, accumulated);
    }
  });

  es.addEventListener("error", () => {
    es.close();
    if (statusEl) statusEl.textContent = "Stream error";
    const dots = panel.querySelector(".dots");
    if (dots) dots.style.display = "none";
  });
}

function initDemo() {
  // Watch for HTMX swapping in the stream panel after form submission.
  document.body.addEventListener("htmx:afterSwap", (evt) => {
    const panel = document.getElementById("stream-panel");
    if (panel) openStreamPanel(panel);
  });
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", initDemo);
} else {
  initDemo();
}
