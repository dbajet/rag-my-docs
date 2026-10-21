"use strict";

const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => Array.from(root.querySelectorAll(selector));

const STORAGE_PREFIX = "rag_my_docs-";
const STORAGE_VIEW = "rag_my_docs-view";
// The model a new connection of each kind starts with, when the endpoint offers it.
const CONNECTION_DEFAULTS = {
  claude: { model: "claude-opus-5-5" },
  openai: { model: "" },
};
const KIND_LABELS = { claude: "Cloud · Claude", openai: "OpenAI-compatible" };
// The views a person can be on; the choice is remembered on this device.
const VIEWS = ["ask", "documents", "inspect", "settings"];
const STORAGE_INSPECTED = "rag_my_docs-inspected";
// How often a view showing a document in the pipeline asks again for its status.
const POLL_MILLISECONDS = 3000;
const BUSY_STATUSES = ["pending", "processing"];
const INDEX_BUSY_STATUSES = ["pending", "indexing"];
const INDEX_LABELS = {
  not_indexed: "Not indexed",
  pending: "Index waiting",
  indexing: "Indexing…",
  indexed: "Indexed",
  failed: "Index failed",
};
const STATUS_LABELS = {
  pending: "Waiting",
  processing: "Processing…",
  completed: "Processed",
  failed: "Failed",
  no_text: "No text",
  unsupported: "Not processed",
};
const SOURCE_LABELS = {
  bookmark: "from bookmarks",
  numbered: "numbered heading",
  layout: "from layout",
  fallback: "no heading found",
};
const DEFAULT_VIEW = "ask";
const LINK_PATH = "/auth/link";
const MEGABYTE = 1024 * 1024;
// Questions from the test set, offered to fill the question box.
const EXAMPLE_QUESTIONS = [
  "What is the 2025 standard deduction for a single filer?",
  "What is the standard mileage rate for business use of a car in 2025?",
  "When is a vacation home I rent out considered used as a home?",
  "How do I renew a U.S. passport?",
];
const GENERIC_ERROR = "Something went wrong. Try again.";
const KIND_OF_TYPE = {
  "application/pdf": { icon: "pdf", label: "PDF" },
  "image/png": { icon: "image", label: "PNG image" },
  "image/jpeg": { icon: "image", label: "JPEG image" },
  "image/webp": { icon: "image", label: "WebP image" },
  "image/gif": { icon: "image", label: "GIF image" },
  "text/plain": { icon: "text", label: "Text" },
  "text/markdown": { icon: "text", label: "Markdown" },
  "text/csv": { icon: "text", label: "CSV" },
};

// The one module-scope mutable object.
const state = {
  view: "loading",
  user: null,
  documents: [],
  documentsLoaded: false,
  filter: "",
  modal: null,
  inspectedId: null,
  inspection: null,
  blocks: null,
  selectedSection: null,
  tab: "chunks",
  pollTimer: null,
  focusedChunk: null,
  answer: null,
  evaluationCases: null,
  isEvaluating: false,
  llm: null,
};

class ApiError extends Error {
  constructor(message, status) {
    super(message);
    this.status = status;
  }
}

// --- small helpers -------------------------------------------------------------

function stored(key) {
  try {
    return localStorage.getItem(key);
  } catch (error) {
    return null;
  }
}

function store(key, value) {
  try {
    localStorage.setItem(key, value);
  } catch (error) {
    // Private windows may refuse storage: the choice still holds for this page.
  }
}

function forgetStored() {
  try {
    Object.keys(localStorage)
      .filter((key) => key.startsWith(STORAGE_PREFIX))
      .forEach((key) => localStorage.removeItem(key));
  } catch (error) {
    // Nothing stored, nothing to forget.
  }
}

// Text set through textContent, never innerHTML: passages and answers are
// documents' and models' words. Icons are the one innerHTML, from constants.
function element(tag, attributes, children) {
  const result = document.createElement(tag);
  Object.entries(attributes).forEach(([name, value]) => {
    if (value === false || value === null || value === undefined) {
      return;
    }
    if (name === "text") {
      result.textContent = String(value);
    } else if (name === "icon") {
      result.innerHTML = ICONS[value] || "";
      result.setAttribute("aria-hidden", "true");
    } else if (value === true) {
      result.setAttribute(name, "");
    } else {
      result.setAttribute(name, String(value));
    }
  });
  children.forEach((child) => result.append(child));
  return result;
}

function iconLabelled(icon, label) {
  return [element("span", { icon }, []), element("span", { text: label }, [])];
}

function renderIcons(root) {
  $$("[data-icon]", root).forEach((target) => {
    // Constants from icons.js only, never user input.
    target.innerHTML = ICONS[target.dataset.icon] || "";
  });
}

function sizeText(bytes) {
  const size = Number(bytes);
  if (size < 1024) {
    return `${size} B`;
  }
  if (size < MEGABYTE) {
    return `${(size / 1024).toFixed(1)} KB`;
  }
  return `${(size / MEGABYTE).toFixed(1)} MB`;
}

function dateText(iso) {
  const date = new Date(String(iso));
  if (Number.isNaN(date.getTime())) {
    return "";
  }
  return date.toLocaleDateString(undefined, { year: "numeric", month: "short", day: "numeric" });
}

function status(message, kind) {
  const target = $("#status");
  target.textContent = message;
  target.dataset.kind = kind;
  target.hidden = message === "";
}

function failed(error) {
  status(error instanceof ApiError ? error.message : GENERIC_ERROR, "error");
}

// --- the API -------------------------------------------------------------------

function detailOf(data, fallback) {
  return data !== null && typeof data === "object" && typeof data.detail === "string" ? data.detail : fallback;
}

// The single fetch wrapper: JSON in and out, the server's
// `detail` as the error, 401 back to the sign-in, and the button that sent the
// request disabled until it answers, so a double click submits once.
async function api(path, options) {
  const button = options.button || null;
  if (button !== null) {
    button.disabled = true;
    button.setAttribute("aria-busy", "true");
  }
  try {
    const headers = { Accept: "application/json", ...(options.headers || {}) };
    let body;
    if (options.json !== undefined) {
      headers["Content-Type"] = "application/json";
      body = JSON.stringify(options.json);
    }
    let response;
    try {
      response = await fetch(path, { method: options.method || "GET", headers, body, credentials: "same-origin" });
    } catch (error) {
      throw new ApiError("The server could not be reached. Check your connection.", 0);
    }
    const text = await response.text();
    let data = null;
    try {
      data = text === "" ? null : JSON.parse(text);
    } catch (error) {
      data = null;
    }
    if (response.status === 401 && options.isSignInCheck !== true && path.startsWith("/api/auth/") === false) {
      signedOutLocally();
    }
    if (response.ok === false) {
      throw new ApiError(detailOf(data, GENERIC_ERROR), response.status);
    }
    return data;
  } finally {
    if (button !== null) {
      button.disabled = false;
      button.removeAttribute("aria-busy");
    }
  }
}

// --- views ---------------------------------------------------------------------

function storedView() {
  const saved = stored(STORAGE_VIEW);
  const inspected = Number(stored(STORAGE_INSPECTED));
  if (saved === "inspect") {
    if (Number.isInteger(inspected) === false || inspected <= 0) {
      return DEFAULT_VIEW;
    }
    state.inspectedId = inspected;
  }
  if (VIEWS.includes(saved) === false) {
    return DEFAULT_VIEW;
  }
  return saved;
}

function shown(view) {
  state.view = view;
  $$("[data-view]").forEach((section) => {
    section.hidden = section.dataset.view !== view;
  });
  // Inspecting a publication is part of the publications.
  const navView = view === "inspect" ? "documents" : view;
  $$("[data-nav]").forEach((button) => {
    if (button.dataset.nav === navView) {
      button.setAttribute("aria-current", "page");
    } else {
      button.removeAttribute("aria-current");
    }
  });
}

function opened(view) {
  status("", "");
  pollStopped();
  shown(view);
  if (VIEWS.includes(view)) {
    store(STORAGE_VIEW, view);
  }
  const heading = $(`[data-view="${view}"] h1`);
  if (heading !== null) {
    heading.setAttribute("tabindex", "-1");
    heading.focus({ preventScroll: true });
  }
  if (view === "documents") {
    loadDocuments();
  } else if (view === "settings") {
    loadConnections();
  } else if (view === "inspect") {
    loadInspection();
  } else if (view === "ask") {
    loadEvaluation();
  }
}

// --- polling ------------------------------------------------------------------------

function pollScheduled(action) {
  pollStopped();
  state.pollTimer = window.setTimeout(action, POLL_MILLISECONDS);
}

function pollStopped() {
  if (state.pollTimer !== null) {
    window.clearTimeout(state.pollTimer);
    state.pollTimer = null;
  }
}

function renderChrome() {
  const isSignedIn = state.user !== null;
  $("#app-nav").hidden = isSignedIn === false;
  $("#sign-out").hidden = isSignedIn === false;
  $("#settings-open").hidden = isSignedIn === false;
}

// --- signing in ----------------------------------------------------------------

async function loadAccount() {
  shown("loading");
  try {
    state.user = await api("/api/account", { isSignInCheck: true });
  } catch (error) {
    state.user = null;
    renderChrome();
    shown("sign-in");
    if (error instanceof ApiError && error.status !== 401) {
      failed(error);
    }
    return;
  }
  renderChrome();
  opened(storedView());
}

function signedOutLocally() {
  // The next person on this device sees none of the previous one's data.
  state.user = null;
  state.documents = [];
  state.documentsLoaded = false;
  state.filter = "";
  state.inspectedId = null;
  state.inspection = null;
  state.blocks = null;
  state.selectedSection = null;
  state.focusedChunk = null;
  state.answer = null;
  state.evaluationCases = null;
  state.llm = null;
  $("#llm-connection-list").replaceChildren();
  $("#search-query").value = "";
  $("#answer-panel").hidden = true;
  $("#evaluation-results").replaceChildren();
  $("#evaluation-summary").textContent = "";
  pollStopped();
  forgetStored();
  Theme.applied();
  $("#document-filter").value = "";
  $("#document-list").replaceChildren();
  closeModal();
  renderChrome();
  shown("sign-in");
}

async function signOut(event) {
  try {
    await api("/api/auth/sign-out", { method: "POST", button: event.currentTarget });
  } catch (error) {
    // The session is forgotten here either way; the server drops it at expiry.
  }
  signedOutLocally();
  status("You are signed out.", "success");
}

async function linkRequested(event) {
  event.preventDefault();
  const input = $("#sign-in-email");
  const email = input.value.trim();
  if (email === "" || input.checkValidity() === false) {
    status("Enter your email address.", "error");
    input.focus();
    return;
  }
  status("", "");
  try {
    const answer = await api("/api/auth/link-request", { method: "POST", json: { email }, button: $("#link-submit") });
    const notice = $("#link-sent");
    notice.textContent = `${answer.message} Check your inbox.`;
    notice.hidden = false;
  } catch (error) {
    failed(error);
  }
}

async function linkConfirmed(event) {
  const token = new URLSearchParams(window.location.search).get("token") || "";
  try {
    await api("/api/auth/link", { method: "POST", json: { token }, button: event.currentTarget });
  } catch (error) {
    failed(error);
    $("#link-confirm").hidden = true;
    $("#link-back").hidden = false;
    return;
  }
  // The token is spent: it leaves the address bar and the history.
  window.history.replaceState(null, "", "/");
  await loadAccount();
}

function linkAbandoned() {
  window.history.replaceState(null, "", "/");
  status("", "");
  shown("sign-in");
  $("#sign-in-email").focus();
}

// --- publications ----------------------------------------------------------------
// Loaded by the server when it starts: the list follows them through download,
// processing and embedding.

async function loadDocuments() {
  $("#documents-loading").hidden = state.documentsLoaded;
  try {
    const answer = await api("/api/documents", {});
    state.documents = answer.documents;
    state.documentsLoaded = true;
  } catch (error) {
    failed(error);
  } finally {
    $("#documents-loading").hidden = true;
  }
  renderDocuments();
  // An empty list is the corpus still downloading.
  if (state.view === "documents" && (state.documents.length === 0 || state.documents.some(isInPipeline))) {
    pollScheduled(loadDocuments);
  }
}

function isInPipeline(documentItem) {
  return BUSY_STATUSES.includes(documentItem.processing_status) || INDEX_BUSY_STATUSES.includes(documentItem.index_status);
}

function documentShown(documentItem) {
  const kind = KIND_OF_TYPE[documentItem.media_type] || { icon: "file", label: "File" };
  const meta = `${kind.label} · ${sizeText(documentItem.size_bytes)} · ${dateText(documentItem.created_at)}`;
  const download = element(
    "a",
    {
      class: "button quiet",
      href: `/api/documents/${Number(documentItem.document_id)}/content`,
      download: documentItem.name,
      "aria-label": `Download ${documentItem.name}`,
      title: "Download",
    },
    [element("span", { icon: "download" }, [])],
  );
  const actions = [download];
  const isPdf = documentItem.media_type === "application/pdf";
  if (isPdf) {
    const inspect = element(
      "button",
      { type: "button", class: "button quiet", "aria-label": `Inspect the processing of ${documentItem.name}`, title: "Inspect" },
      [element("span", { icon: "layers" }, [])],
    );
    inspect.addEventListener("click", () => inspected(documentItem.document_id));
    actions.unshift(inspect);
  }
  const metaLine = element("p", { class: "document-meta", text: meta }, []);
  if (isPdf) {
    metaLine.append(" ", statusChip(documentItem.processing_status));
    if (documentItem.processing_status === "completed") {
      metaLine.append(" ", indexChip(documentItem.index_status));
      metaLine.append(
        element("span", { class: "document-counts", text: ` ${counted(documentItem.chunk_count, "chunk")} · ${documentItem.indexed_count} indexed` }, []),
      );
    }
  }
  return element("li", { class: "document" }, [
    element("span", { class: "document-kind", icon: kind.icon }, []),
    element("div", { class: "document-text" }, [
      element("p", { class: "document-name", text: documentItem.name }, []),
      metaLine,
    ]),
    element("div", { class: "document-actions" }, actions),
  ]);
}

function renderDocuments() {
  const needle = state.filter.trim().toLowerCase();
  const visible = state.documents.filter((item) => needle === "" || item.name.toLowerCase().includes(needle));
  $("#document-list").replaceChildren(...visible.map(documentShown));
  const total = state.documents.reduce((sum, item) => sum + Number(item.size_bytes), 0);
  const count = state.documents.length;
  $("#documents-summary").textContent = count === 0 ? "" : `${counted(count, "publication")} · ${sizeText(total)}`;
  const empty = $("#documents-empty");
  if (state.documentsLoaded === false) {
    empty.hidden = true;
  } else if (count === 0) {
    empty.textContent = "The publications are being downloaded; this list refreshes on its own.";
    empty.hidden = false;
  } else if (visible.length === 0) {
    empty.textContent = "No publication matches this filter.";
    empty.hidden = false;
  } else {
    empty.hidden = true;
  }
}

// --- inspection ----------------------------------------------------------------
// A learning view: what the pipeline read from a PDF, the sections it found, and
// every chunk (passage) it cut, so a wrong section boundary or an awkward split is easy to see.

function statusChip(processingStatus) {
  return element("span", { class: `chip status-${processingStatus}`, text: STATUS_LABELS[processingStatus] || processingStatus }, []);
}

function indexChip(indexStatus) {
  return element("span", { class: `chip index-${indexStatus}`, text: INDEX_LABELS[indexStatus] || indexStatus }, []);
}

function inspected(documentId) {
  state.focusedChunk = null;
  state.inspectedId = Number(documentId);
  state.inspection = null;
  state.blocks = null;
  state.selectedSection = null;
  state.tab = "chunks";
  store(STORAGE_INSPECTED, String(state.inspectedId));
  opened("inspect");
}

async function loadInspection() {
  renderTab();
  try {
    state.inspection = await api(`/api/documents/${state.inspectedId}/processing`, {});
  } catch (error) {
    failed(error);
    if (error instanceof ApiError && error.status === 404) {
      opened("documents");
    }
    return;
  }
  renderInspection();
  focusedChunkShown();
  const info = state.inspection.document;
  if (BUSY_STATUSES.includes(info.status) || INDEX_BUSY_STATUSES.includes(info.index_status)) {
    pollScheduled(loadInspection);
  } else if (state.tab === "extraction") {
    loadExtraction();
  }
}

function renderInspection() {
  const { document: info, sections, chunks } = state.inspection;
  const report = info.report || {};
  $("#inspect-title").textContent = info.name;
  $("#inspect-status").replaceWith(Object.assign(statusChip(info.status), { id: "inspect-status" }));
  const parts = [];
  if (info.page_count !== null) {
    parts.push(`${info.page_count} page${info.page_count === 1 ? "" : "s"}`);
  }
  if (info.status === "completed") {
    parts.push(`${sections.length} section${sections.length === 1 ? "" : "s"}`, `${chunks.length} chunk${chunks.length === 1 ? "" : "s"}`);
  }
  if (info.processed_at !== "") {
    parts.push(`processed ${dateText(info.processed_at)}`);
  }
  $("#inspect-meta").textContent = parts.join(" · ");
  $("#inspect-strategy").textContent = info.chunking_version === "" ? "" : `Chunking: ${info.chunking_version}`;
  $("#inspect-stale").hidden = info.is_stale !== true;
  const errorLine = $("#inspect-error");
  errorLine.textContent = info.error;
  errorLine.hidden = info.error === "";
  $("#reprocess").disabled = BUSY_STATUSES.includes(info.status) || info.media_type !== "application/pdf";
  renderReport(report);
  renderIndexing(info);
  renderSectionTree();
  renderChunks();
}

function renderIndexing(info) {
  $("#inspect-index-status").replaceWith(Object.assign(indexChip(info.index_status), { id: "inspect-index-status" }));
  const parts = [`${info.indexed_count} of ${counted(info.chunk_count, "chunk")} embedded`];
  if (info.index_model !== "") {
    parts.push(info.index_model);
  }
  const report = info.index_report || {};
  if (typeof report.duration_ms === "number") {
    parts.push(`last run: ${report.embedded} embedded, ${report.reused} kept, ${report.duration_ms} ms`);
  }
  if (info.indexed_at !== "") {
    parts.push(dateText(info.indexed_at));
  }
  $("#inspect-index-meta").textContent = parts.join(" · ");
  const errorLine = $("#inspect-index-error");
  errorLine.textContent = info.index_error;
  errorLine.hidden = info.index_error === "";
  const isLocked = info.status !== "completed" || INDEX_BUSY_STATUSES.includes(info.index_status);
  $("#index-missing").disabled = isLocked;
  $("#reindex").disabled = isLocked;
}

async function reindexed(button, rebuild) {
  try {
    await api(`/api/documents/${state.inspectedId}/index`, { method: "POST", json: { rebuild }, button });
  } catch (error) {
    failed(error);
    return;
  }
  status(rebuild ? "Every chunk will be embedded again." : "The document is queued for indexing.", "success");
  await loadInspection();
}

function renderReport(report) {
  const facts = [];
  const pagesText = (pages) => pages.join(", ");
  if (Array.isArray(report.pages_without_text) && report.pages_without_text.length > 0) {
    facts.push(`Pages without text: ${pagesText(report.pages_without_text)}`);
  }
  if (Array.isArray(report.pages_probably_scanned) && report.pages_probably_scanned.length > 0) {
    facts.push(`Probably scanned (no OCR yet): ${pagesText(report.pages_probably_scanned)}`);
  }
  if (typeof report.blocks === "number") {
    facts.push(`${counted(report.blocks, "block")} read`, counted(report.headings, "heading"), counted(report.tables, "table"));
    facts.push(`${counted(report.headers_and_footers, "header/footer line")} left out`, counted(report.bookmarks, "bookmark"));
  }
  if (typeof report.section_source === "string" && report.section_source !== "") {
    facts.push(`Sections: ${SOURCE_LABELS[report.section_source] || report.section_source}`);
  }
  $("#inspect-report").replaceChildren(...facts.map((fact) => element("li", { text: fact }, [])));
}

function counted(count, word) {
  return `${count} ${word}${Number(count) === 1 ? "" : "s"}`;
}

function pagesText(first, last) {
  return first === last ? `p. ${first}` : `p. ${first}–${last}`;
}

function sectionTitle(section) {
  return section.title === "" ? `Untitled section (${pagesText(section.start_page, section.end_page)})` : section.title;
}

function renderSectionTree() {
  const { sections, chunks } = state.inspection;
  const empty = $("#inspect-empty");
  if (sections.length === 0) {
    empty.textContent = BUSY_STATUSES.includes(state.inspection.document.status)
      ? "The document is being processed; this view refreshes on its own."
      : "Nothing to show: the document has no processed text.";
    empty.hidden = false;
  } else {
    empty.hidden = true;
  }
  const byParent = new Map();
  sections.forEach((section) => {
    const key = section.parent_id === null ? "root" : String(section.parent_id);
    byParent.set(key, [...(byParent.get(key) || []), section]);
  });
  const branch = (key) =>
    (byParent.get(key) || []).map((section) => {
      const button = element(
        "button",
        {
          type: "button",
          class: "section-button",
          "aria-pressed": state.selectedSection === section.section_id ? "true" : "false",
        },
        [
          element("span", { class: "section-title", text: sectionTitle(section) }, []),
          element(
            "span",
            {
              class: "section-meta",
              text: `${pagesText(section.start_page, section.end_page)} · ${counted(section.chunk_count, "chunk")} · ${SOURCE_LABELS[section.source] || section.source}`,
            },
            [],
          ),
        ],
      );
      button.addEventListener("click", () => {
        state.selectedSection = state.selectedSection === section.section_id ? null : section.section_id;
        renderSectionTree();
        renderChunks();
      });
      const children = branch(String(section.section_id));
      return element("li", {}, [button, ...(children.length > 0 ? [element("ul", {}, children)] : [])]);
    });
  $("#section-tree").replaceChildren(...branch("root"));
  $("#chunk-filter-clear").hidden = state.selectedSection === null;
  const titles = new Map(sections.map((section) => [section.section_id, sectionTitle(section)]));
  const shownChunks = chunks.filter((chunk) => state.selectedSection === null || chunk.section_id === state.selectedSection);
  $("#chunk-heading").textContent =
    state.selectedSection === null ? `Chunks (${chunks.length})` : `Chunks of “${titles.get(state.selectedSection)}” (${shownChunks.length})`;
}

function renderChunks() {
  const { sections, chunks } = state.inspection;
  const titles = new Map(sections.map((section) => [section.section_id, sectionTitle(section)]));
  const shownChunks = chunks.filter((chunk) => state.selectedSection === null || chunk.section_id === state.selectedSection);
  $("#chunk-list").replaceChildren(
    ...shownChunks.map((chunk) => {
      const pages = pagesText(chunk.start_page, chunk.end_page);
      const prefix = chunk.text.slice(0, chunk.prefix_chars);
      const own = chunk.text.slice(chunk.prefix_chars);
      const body = element("pre", { class: `chunk-text ${chunk.kind === "table" ? "is-table" : ""}` }, []);
      if (prefix !== "") {
        body.append(element("span", { class: "overlap", title: chunk.kind === "table" ? "Repeated header row" : "Overlap with the previous chunk", text: prefix }, []));
      }
      body.append(document.createTextNode(own));
      const isFocused = state.focusedChunk === chunk.chunk_id;
      return element("li", { class: `chunk ${isFocused ? "is-focused" : ""}`, "data-chunk": chunk.chunk_id }, [
        element("p", { class: "chunk-meta" }, [
          element("strong", { text: `#${chunk.position + 1}` }, []),
          document.createTextNode(
            ` · ${pages} · ${chunk.token_count} tokens · ${chunk.kind} · ${chunk.embedding_model === "" ? "not embedded" : "embedded"} · `,
          ),
          element("span", { class: "muted", text: titles.get(chunk.section_id) || "" }, []),
        ]),
        body,
      ]);
    }),
  );
}

function focusedChunkShown() {
  if (state.focusedChunk === null) {
    return;
  }
  const card = $(`[data-chunk="${state.focusedChunk}"]`);
  if (card !== null) {
    card.scrollIntoView({ block: "center" });
  }
}

async function loadExtraction() {
  if (state.blocks !== null) {
    renderBlocks();
    return;
  }
  try {
    const answer = await api(`/api/documents/${state.inspectedId}/extraction`, {});
    state.blocks = answer.blocks;
  } catch (error) {
    failed(error);
    return;
  }
  renderBlocks();
}

function renderBlocks() {
  const blocks = state.blocks || [];
  const list = $("#block-list");
  if (blocks.length === 0) {
    list.replaceChildren(element("li", { class: "muted", text: "No extracted text." }, []));
    return;
  }
  list.replaceChildren(
    ...blocks.map((block) =>
      element("li", { class: `block kind-${block.kind}` }, [
        element("p", { class: "block-meta", text: `p. ${block.page} · ${block.kind} · block ${block.block_id}` }, []),
        element("pre", { class: "block-text", text: block.text }, []),
      ]),
    ),
  );
}

function tabSelected(tab) {
  state.tab = tab;
  renderTab();
  if (tab === "extraction" && state.inspection !== null && BUSY_STATUSES.includes(state.inspection.document.status) === false) {
    loadExtraction();
  }
}

function renderTab() {
  [["chunks", "#tab-chunks", "#panel-chunks"], ["extraction", "#tab-extraction", "#panel-extraction"]].forEach(([name, tab, panel]) => {
    const isSelected = state.tab === name;
    $(tab).setAttribute("aria-selected", isSelected ? "true" : "false");
    $(tab).setAttribute("tabindex", isSelected ? "0" : "-1");
    $(panel).hidden = isSelected === false;
  });
}

async function reprocessed(event) {
  try {
    await api(`/api/documents/${state.inspectedId}/processing`, { method: "POST", button: event.currentTarget });
  } catch (error) {
    failed(error);
    return;
  }
  state.blocks = null;
  state.selectedSection = null;
  status("The document is queued for processing.", "success");
  await loadInspection();
}

// Sized through the CSSOM: the CSP refuses inline style attributes, not this.
function scoreFill(score) {
  const result = element("span", { class: "score-fill" }, []);
  result.style.width = `${Math.round(score * 100)}%`;
  return result;
}

function searchHitShown(hit) {
  const score = Math.max(0, Math.min(1, hit.similarity));
  const body = element("pre", { class: "chunk-text" }, []);
  if (hit.prefix_chars > 0) {
    body.append(element("span", { class: "overlap", title: "Overlap with the previous chunk", text: hit.text.slice(0, hit.prefix_chars) }, []));
  }
  body.append(document.createTextNode(hit.text.slice(hit.prefix_chars)));
  const inspect = element("button", { type: "button", class: "button small" }, iconLabelled("layers", "Show in the document"));
  inspect.addEventListener("click", () => {
    inspected(hit.document_id);
    state.focusedChunk = hit.chunk_id;
  });
  const download = element(
    "a",
    { class: "button small quiet", href: `/api/documents/${Number(hit.document_id)}/content`, download: hit.document_name },
    iconLabelled("download", "Download"),
  );
  return element("li", { class: "search-hit" }, [
    element("div", { class: "hit-head" }, [
      element("span", { class: "hit-rank", text: `#${hit.rank}` }, []),
      element("span", { class: "hit-score", title: "Cosine similarity: 1 means the same direction" }, [
        element("span", { class: "score-bar" }, [scoreFill(score)]),
        element("span", { text: hit.similarity.toFixed(3) }, []),
      ]),
    ]),
    element("p", { class: "chunk-meta" }, [
      element("strong", { text: hit.document_name }, []),
      document.createTextNode(` · ${hit.section_path} · ${pagesText(hit.start_page, hit.end_page)} · chunk #${hit.chunk_position + 1}`),
    ]),
    body,
    element("div", { class: "hit-actions" }, [inspect, download]),
  ]);
}

async function indexedAll(event) {
  try {
    const answer = await api("/api/index/all", { method: "POST", button: event.currentTarget });
    status(answer.queued === 0 ? "Every processed document is already indexed." : `${counted(answer.queued, "document")} queued for indexing.`, "success");
  } catch (error) {
    failed(error);
    return;
  }
  state.documentsLoaded = false;
}

// --- language models ---------------------------------------------------------
// Connections to Claude or to any OpenAI-compatible endpoint, each with a light that
// says whether it answers now, and the connection that answers.

async function loadConnections() {
  try {
    state.llm = await api("/api/llm/connections", {});
  } catch (error) {
    failed(error);
    return;
  }
  renderConnections();
  state.llm.connections.forEach((connection) => connectionChecked(connection.connection_id));
}

function renderConnections() {
  const connections = state.llm.connections;
  $("#llm-empty").hidden = connections.length > 0;
  $("#llm-connection-list").replaceChildren(...connections.map(connectionShown));
  const chosen = state.llm.routing.answers;
  $("#llm-route-answers").replaceChildren(
    element("option", { value: "", text: "None: answers are off", selected: chosen === null }, []),
    ...connections.map((connection) =>
      element("option", { value: connection.connection_id, text: connection.name, selected: connection.connection_id === chosen }, []),
    ),
  );
}

function connectionShown(connection) {
  const test = element("button", { type: "button", class: "button small quiet" }, iconLabelled("refresh", "Test"));
  test.addEventListener("click", () => connectionChecked(connection.connection_id));
  const edit = element("button", { type: "button", class: "button small quiet" }, iconLabelled("pencil", "Edit"));
  edit.addEventListener("click", () => connectionAsked(connection));
  const remove = element("button", { type: "button", class: "button small quiet danger-text" }, iconLabelled("trash", "Delete"));
  remove.addEventListener("click", () => connectionRemovalAsked(connection));
  const facts = [`Model: ${connection.model}`];
  if (connection.kind === "openai") {
    facts.unshift(connection.base_url);
  }
  return element("li", { class: "connection", "data-connection": connection.connection_id }, [
    element("span", { class: "light is-checking", role: "img", "aria-label": "Checking…", title: "Checking…" }, []),
    element("div", { class: "connection-body" }, [
      element("p", { class: "connection-name" }, [
        element("strong", { text: connection.name }, []),
        document.createTextNode(" "),
        element("span", { class: "chip", text: KIND_LABELS[connection.kind] || connection.kind }, []),
      ]),
      element("p", { class: "muted small", text: facts.join(" · ") }, []),
      element("p", { class: "small connection-detail", text: "Checking…" }, []),
      element("div", { class: "connection-actions" }, [test, edit, remove]),
    ]),
  ]);
}

async function connectionChecked(connectionId) {
  const item = $(`[data-connection="${Number(connectionId)}"]`);
  if (item === null) {
    return;
  }
  lightShown(item, "is-checking", "Checking…");
  let health;
  try {
    health = await api(`/api/llm/connections/${Number(connectionId)}/health`, {});
  } catch (error) {
    lightShown(item, "is-down", error.message);
    return;
  }
  lightShown(item, health.is_ok ? "is-up" : "is-down", `${health.detail} (${Math.round(health.elapsed_ms)} ms)`);
}

function lightShown(item, kind, detail) {
  const light = $(".light", item);
  light.className = `light ${kind}`;
  const label = kind === "is-up" ? "Connected" : kind === "is-down" ? "Not connected" : "Checking…";
  light.setAttribute("aria-label", label);
  light.setAttribute("title", detail);
  $(".connection-detail", item).textContent = detail;
}

// Two tabs: the endpoint first (kind, address, key), then its model, chosen from the
// list the endpoint answered with rather than typed.
function connectionAsked(connection) {
  const isNew = connection === null;
  const kind = isNew ? "claude" : connection.kind;
  const values = isNew ? { name: "", base_url: "", ...CONNECTION_DEFAULTS.claude, disable_thinking: true } : connection;
  const kindSelect = element("select", { name: "kind" }, [
    element("option", { value: "claude", text: "Claude (cloud)", selected: kind === "claude" }, []),
    element("option", { value: "openai", text: "OpenAI-compatible (vLLM, or hosted)", selected: kind === "openai" }, []),
  ]);
  const urlInput = element("input", { type: "url", name: "base_url", value: values.base_url, placeholder: "https://llm.example.com/v1" }, []);
  const urlField = element("div", { class: "kind-openai" }, field("Address of the API, ending in /v1", urlInput));
  const keyHint = isNew || connection.has_key === false ? "API key" : "API key (leave empty to keep the stored one)";
  const keyInput = element("input", { type: "password", name: "api_key", autocomplete: "off" }, []);
  const thinking = element("div", { class: "kind-openai" }, [
    checkbox("Ask the model not to think first (faster; vLLM reasoning models)", "disable_thinking", values.disable_thinking),
  ]);
  const modelSelect = element("select", { name: "model" }, []);
  const offered = element("p", { class: "muted small" }, []);
  const tabs = {
    endpoint: element("button", { type: "button", role: "tab", id: "connection-tab-endpoint", "aria-controls": "connection-panel-endpoint", text: "1. Connection" }, []),
    models: element("button", { type: "button", role: "tab", id: "connection-tab-models", "aria-controls": "connection-panel-models", text: "2. Model" }, []),
  };
  const panels = {
    endpoint: element("div", { role: "tabpanel", id: "connection-panel-endpoint", "aria-labelledby": "connection-tab-endpoint" }, [
      ...field("Kind", kindSelect),
      ...field("Name", element("input", { type: "text", name: "name", maxlength: 80, value: values.name }, [])),
      urlField,
      ...field(keyHint, keyInput),
      element("p", { class: "muted small", text: "Connect to fetch the models this key can use." }, []),
    ]),
    models: element("div", { role: "tabpanel", id: "connection-panel-models", "aria-labelledby": "connection-tab-models" }, [
      offered,
      ...field("Model", modelSelect),
      thinking,
    ]),
  };
  const dialog = { tab: "endpoint", models: null };
  const tabShown = (tab) => {
    dialog.tab = tab;
    Object.entries(tabs).forEach(([name, button]) => {
      const isChosen = name === tab;
      button.setAttribute("aria-selected", String(isChosen));
      button.tabIndex = isChosen ? 0 : -1;
      panels[name].hidden = isChosen === false;
    });
    state.modal.submit.button.textContent = tab === "endpoint" ? "Connect" : isNew ? "Add" : "Save";
  };
  // A changed endpoint may offer other models: they are fetched again.
  const endpointChanged = () => {
    dialog.models = null;
    tabs.models.disabled = true;
    const isOpenai = kindSelect.value === "openai";
    urlField.hidden = isOpenai === false;
    // Disabled while hidden: an address left from another kind must not block the form.
    urlInput.disabled = isOpenai === false;
    thinking.hidden = isOpenai === false;
  };
  const connected = async (button) => {
    const listed = await api("/api/llm/models", {
      method: "POST",
      json: {
        connection_id: isNew ? 0 : Number(connection.connection_id),
        kind: kindSelect.value,
        base_url: urlInput.value,
        api_key: keyInput.value,
      },
      button,
    });
    if (listed.models.length === 0) {
      throw new ApiError("Connected, but the endpoint offers no model.", 0);
    }
    const preferred = kindSelect.value === kind ? values : CONNECTION_DEFAULTS[kindSelect.value];
    modelOptions(modelSelect, listed.models, preferred.model);
    offered.textContent = `Connected: ${listed.models.length} ${listed.models.length === 1 ? "model is" : "models are"} offered.`;
    dialog.models = listed.models;
    tabs.models.disabled = false;
    tabShown("models");
    tabs.models.focus();
  };
  const saved = async (button) => {
    const form = $("#modal-form");
    const json = {
      kind: kindSelect.value,
      name: form.elements.name.value,
      base_url: urlInput.value,
      api_key: keyInput.value,
      model: modelSelect.value,
      disable_thinking: form.elements.disable_thinking.checked,
    };
    const path = isNew ? "/api/llm/connections" : `/api/llm/connections/${Number(connection.connection_id)}`;
    const answer = await api(path, { method: isNew ? "POST" : "PUT", json, button });
    closeModal();
    status(`${answer.name} is saved.`, "success");
    await loadConnections();
  };
  [kindSelect, urlInput, keyInput].forEach((input) => input.addEventListener("input", endpointChanged));
  Object.entries(tabs).forEach(([name, button]) => {
    button.addEventListener("click", () => tabShown(name));
    // Arrow keys move between the tabs, as a tablist is expected to behave.
    button.addEventListener("keydown", (event) => {
      const other = name === "endpoint" ? "models" : "endpoint";
      if ((event.key === "ArrowRight" || event.key === "ArrowLeft") && tabs[other].disabled === false) {
        event.preventDefault();
        tabShown(other);
        tabs[other].focus();
      }
    });
  });
  openModal({
    title: isNew ? "Add a connection" : `Edit ${connection.name}`,
    isTall: true,
    content: [element("div", { class: "tabs", role: "tablist", "aria-label": "Connection steps" }, [tabs.endpoint, tabs.models]), panels.endpoint, panels.models],
    actions: [
      { label: "Cancel", kind: "button", onClick: closeModal },
      {
        label: "Connect",
        kind: "button primary",
        isSubmit: true,
        onClick: (button) => (dialog.tab === "endpoint" ? connected(button) : saved(button)),
      },
    ],
  });
  endpointChanged();
  tabShown("endpoint");
  kindSelect.focus();
}

// The endpoint's models as options; the one preferred is chosen when offered.
function modelOptions(select, models, preferred) {
  const chosen = models.includes(preferred) ? preferred : models[0];
  select.replaceChildren(...models.map((model) => element("option", { value: model, text: model, selected: model === chosen }, [])));
}

function connectionRemovalAsked(connection) {
  openModal({
    title: "Delete this connection?",
    content: [
      element("p", { class: "document-name", text: connection.name }, []),
      element("p", { class: "muted", text: "If it answers, answers are off until you choose another." }, []),
    ],
    actions: [
      { label: "Cancel", kind: "button", onClick: closeModal },
      {
        label: "Delete",
        kind: "button danger",
        onClick: async (button) => {
          await api(`/api/llm/connections/${Number(connection.connection_id)}`, { method: "DELETE", button });
          closeModal();
          status(`${connection.name} was deleted.`, "success");
          await loadConnections();
        },
      },
    ],
  });
}

async function routingSaved(event) {
  event.preventDefault();
  const value = $("#llm-route-answers").value;
  try {
    await api("/api/llm/routing", {
      method: "PUT",
      json: { answers: value === "" ? null : Number(value) },
      button: $("#llm-routing-save"),
    });
  } catch (error) {
    failed(error);
    return;
  }
  status("The connection that answers is saved.", "success");
}

// --- answers ---------------------------------------------------------------------
// Retrieval-augmented generation, every step shown: the passages retrieved and their
// similarity, the exact prompt, and the answer citing the passages it used.

function examplesShown() {
  $("#example-list").replaceChildren(
    ...EXAMPLE_QUESTIONS.map((question) => {
      const button = element("button", { type: "button", class: "button small quiet", text: question }, []);
      button.addEventListener("click", () => {
        $("#search-query").value = question;
        $("#search-query").focus();
      });
      return element("li", {}, [button]);
    }),
  );
}

async function answered(event) {
  event.preventDefault();
  const question = $("#search-query").value.trim();
  if (question === "") {
    status("Type a question to answer.", "error");
    $("#search-query").focus();
    return;
  }
  status("", "");
  const panel = $("#answer-panel");
  panel.hidden = false;
  $("#answer-found").replaceWith(Object.assign(element("span", { class: "chip", text: "Answering…" }, []), { id: "answer-found" }));
  $("#answer-text").textContent = "Retrieving passages, then asking the model. This takes a few seconds.";
  $("#answer-citations").replaceChildren();
  $("#answer-meta").textContent = "";
  $("#answer-passages").replaceChildren();
  $("#answer-system").textContent = "";
  $("#answer-prompt").textContent = "";
  try {
    state.answer = await api("/api/answers", {
      method: "POST",
      json: { question, top_k: Number($("#search-top-k").value) },
      button: $("#answer-submit"),
    });
  } catch (error) {
    panel.hidden = true;
    failed(error);
    return;
  }
  renderAnswer(state.answer);
}

function renderAnswer(result) {
  const answer = result.answer;
  const found = element(
    "span",
    { class: `chip ${answer.is_found ? "answer-found" : "answer-missing"}`, text: answer.is_found ? "Found" : "Not in the publications" },
    [],
  );
  $("#answer-found").replaceWith(Object.assign(found, { id: "answer-found" }));
  $("#answer-text").textContent = answer.text;
  $("#answer-citations").replaceChildren(
    ...answer.citations.map((citation) => element("li", { text: `[${citation.passage}] ${citation.document}, p. ${citation.page}` }, [])),
  );
  const model = result.model === "" ? "no model called" : `${result.model} answered in ${secondsText(result.answer_ms)}`;
  $("#answer-meta").textContent = `Question embedded in ${result.embedding_ms} ms · vector search ${result.search_ms} ms · ${model}`;
  $("#answer-passages").replaceChildren(...result.passages.map(searchHitShown));
  $("#answer-no-model").hidden = result.model !== "";
  $("#answer-system").textContent = result.system;
  $("#answer-prompt").textContent = result.prompt;
}

function secondsText(milliseconds) {
  return `${(milliseconds / 1000).toFixed(1)} s`;
}

// --- the test set ---------------------------------------------------------------
// Each question is answered for real and judged: did a passage come from a publication
// that holds the answer, and does the answer hold the expected facts. One request per
// question: each takes as long as an answer.

async function loadEvaluation() {
  if (state.evaluationCases !== null) {
    return;
  }
  try {
    const answer = await api("/api/evaluation/cases", {});
    state.evaluationCases = answer.cases;
  } catch (error) {
    failed(error);
    return;
  }
  $("#evaluation-results").replaceChildren(...state.evaluationCases.map(evaluationCaseShown));
}

function evaluationCaseShown(item) {
  const expected = item.expects_answer ? `expects: ${item.facts.join(", ")}` : "expects: not in the publications";
  return element("li", { class: "evaluation-row" }, [
    element("p", { class: "evaluation-question" }, [element("strong", { text: item.question }, [])]),
    element("p", { class: "muted small", text: expected }, []),
  ]);
}

async function evaluationRun(event) {
  const cases = state.evaluationCases;
  if (cases === null || state.isEvaluating) {
    return;
  }
  const button = event.currentTarget;
  const topK = Number($("#search-top-k").value);
  state.isEvaluating = true;
  button.disabled = true;
  const rows = cases.map((item) => element("li", { class: "evaluation-row", text: `${item.question} — waiting` }, []));
  $("#evaluation-results").replaceChildren(...rows);
  const totals = { retrieval: 0, answers: 0, done: 0 };
  for (let index = 0; index < cases.length; index += 1) {
    rows[index].textContent = `${cases[index].question} — running…`;
    try {
      const outcome = await api("/api/evaluation/run", { method: "POST", json: { index, top_k: topK } });
      totals.retrieval += outcome.is_retrieval_right ? 1 : 0;
      totals.answers += outcome.is_answer_right ? 1 : 0;
      const shown = evaluationRowShown(outcome);
      rows[index].replaceWith(shown);
      rows[index] = shown;
    } catch (error) {
      rows[index].textContent = `${cases[index].question} — ${error.message}`;
    }
    totals.done += 1;
    $("#evaluation-summary").textContent =
      `${totals.done} of ${cases.length} run with ${counted(topK, "passage")} · retrieval right ${totals.retrieval} · answers right ${totals.answers}`;
  }
  state.isEvaluating = false;
  button.disabled = false;
}

function evaluationRowShown(outcome) {
  const result = outcome.result;
  const misses = [];
  if (outcome.missing_documents.length > 0) {
    misses.push(`no passage from: ${outcome.missing_documents.join(", ")}`);
  }
  if (outcome.missing_facts.length > 0) {
    misses.push(`missing: ${outcome.missing_facts.join(", ")}`);
  }
  const best = result.passages.length > 0 ? `best passage ${result.passages[0].similarity.toFixed(3)}` : "no passage";
  return element("li", { class: "evaluation-row" }, [
    element("p", { class: "evaluation-question" }, [
      element("span", { class: `chip ${outcome.is_retrieval_right ? "answer-found" : "answer-missing"}`, text: outcome.is_retrieval_right ? "Retrieval ✓" : "Retrieval ✗" }, []),
      document.createTextNode(" "),
      element("span", { class: `chip ${outcome.is_answer_right ? "answer-found" : "answer-missing"}`, text: outcome.is_answer_right ? "Answer ✓" : "Answer ✗" }, []),
      document.createTextNode(" "),
      element("strong", { text: outcome.question }, []),
    ]),
    element("p", { class: "small", text: result.answer.text }, []),
    element("p", { class: "muted small", text: [...misses, best, `${secondsText(result.embedding_ms + result.search_ms + result.answer_ms)} in all`].join(" · ") }, []),
  ]);
}

// --- form fields -----------------------------------------------------------------

function field(label, input) {
  const id = `modal-field-${input.getAttribute("name")}`;
  input.setAttribute("id", id);
  return [element("label", { for: id, text: label }, []), input];
}

function checkbox(label, name, isChecked) {
  return element("label", { class: "check" }, [
    element("input", { type: "checkbox", name, checked: isChecked }, []),
    element("span", { text: label }, []),
  ]);
}

// --- the modal -----------------------------------------------------------------

// One in-app dialog for every question (never alert/confirm/prompt).
// Its frame is fixed by the stylesheet; showModal() keeps focus inside and Escape
// closes it; focus returns to whatever opened it.
function openModal(spec) {
  const dialog = $("#modal");
  state.modal = { returnFocus: document.activeElement, submit: null };
  $("#modal-title").textContent = spec.title;
  // A taller frame for a form that would otherwise scroll on a laptop screen.
  dialog.classList.toggle("is-tall", spec.isTall === true);
  $("#modal-body").replaceChildren(...spec.content);
  $("#modal-error").hidden = true;
  $("#modal-actions").replaceChildren(
    ...spec.actions.map((action) => {
      const button = element("button", { type: action.isSubmit === true ? "submit" : "button", class: action.kind, text: action.label }, []);
      const run = async () => {
        $("#modal-error").hidden = true;
        try {
          await action.onClick(button);
        } catch (error) {
          const message = $("#modal-error");
          message.textContent = error instanceof ApiError ? error.message : GENERIC_ERROR;
          message.hidden = false;
        }
      };
      if (action.isSubmit === true) {
        state.modal.submit = { run, button };
      } else {
        button.addEventListener("click", run);
      }
      return button;
    }),
  );
  dialog.showModal();
  const first = $("#modal-body input") || $("#modal-actions button");
  if (first !== null) {
    first.focus();
  }
}

function closeModal() {
  const dialog = $("#modal");
  if (dialog.open) {
    dialog.close();
  }
}

function modalSubmitted(event) {
  event.preventDefault();
  if (state.modal === null || state.modal.submit === null) {
    return;
  }
  const form = $("#modal-form");
  if (form.reportValidity() === false) {
    return;
  }
  state.modal.submit.run();
}

function modalClosed() {
  // The close event arrives a task after close(): a dialog re-opened in between
  // already holds new content, which this late event must not clear.
  if ($("#modal").open) {
    return;
  }
  const returnFocus = state.modal === null ? null : state.modal.returnFocus;
  state.modal = null;
  $("#modal-body").replaceChildren();
  $("#modal-actions").replaceChildren();
  if (returnFocus !== null && typeof returnFocus.focus === "function" && document.contains(returnFocus)) {
    returnFocus.focus();
  }
}

// --- wiring --------------------------------------------------------------------

function bound() {
  $$("[data-nav]").forEach((button) => {
    button.addEventListener("click", () => opened(button.dataset.nav));
  });
  $("#sign-out").addEventListener("click", signOut);
  $("#link-form").addEventListener("submit", linkRequested);
  $("#link-confirm").addEventListener("click", linkConfirmed);
  $("#link-back").addEventListener("click", linkAbandoned);
  $("#document-filter").addEventListener("input", (event) => {
    state.filter = event.target.value;
    renderDocuments();
  });
  $("#llm-add").addEventListener("click", () => connectionAsked(null));
  $("#llm-routing-form").addEventListener("submit", routingSaved);
  $("#inspect-back").addEventListener("click", () => opened("documents"));
  $("#index-missing").addEventListener("click", (event) => reindexed(event.currentTarget, false));
  $("#reindex").addEventListener("click", (event) => reindexed(event.currentTarget, true));
  $("#search-form").addEventListener("submit", answered);
  $("#evaluation-run").addEventListener("click", evaluationRun);
  $("#index-all").addEventListener("click", indexedAll);
  $("#reprocess").addEventListener("click", reprocessed);
  $("#tab-chunks").addEventListener("click", () => tabSelected("chunks"));
  $("#tab-extraction").addEventListener("click", () => tabSelected("extraction"));
  $$('[data-view="inspect"] [role="tab"]').forEach((tab) => {
    // Arrow keys move between tabs, as a tablist is expected to behave.
    tab.addEventListener("keydown", (event) => {
      if (event.key === "ArrowRight" || event.key === "ArrowLeft") {
        const next = state.tab === "chunks" ? "extraction" : "chunks";
        tabSelected(next);
        $(next === "chunks" ? "#tab-chunks" : "#tab-extraction").focus();
      }
    });
  });
  $("#chunk-filter-clear").addEventListener("click", () => {
    state.selectedSection = null;
    renderSectionTree();
    renderChunks();
  });
  $("#modal-form").addEventListener("submit", modalSubmitted);
  $("#modal").addEventListener("close", modalClosed);
  $("#version-button").addEventListener("click", (event) => {
    const button = event.currentTarget;
    button.setAttribute("aria-expanded", button.getAttribute("aria-expanded") === "true" ? "false" : "true");
  });
}

function workerRegistered() {
  if ("serviceWorker" in navigator) {
    navigator.serviceWorker.register("/sw.js").catch(() => {
      // Without a worker the app still works; it just cannot be installed.
    });
  }
}

async function booted() {
  renderIcons(document);
  examplesShown();
  bound();
  workerRegistered();
  if (window.location.pathname === LINK_PATH) {
    shown("link");
    return;
  }
  await loadAccount();
}

document.addEventListener("DOMContentLoaded", booted);
