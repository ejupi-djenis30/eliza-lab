import { ElizaEngine, MAX_INPUT_CHARS } from "./engine.mjs?v=3.0.0-1";
import { loadOpenSetBundle } from "./open-set-engine.mjs?v=3.0.0-1";
import { loadSelectionAudit } from "./selection-audit.mjs?v=1.0.0-1";

const MAX_TRANSCRIPT_MESSAGES = 80;
const form = document.querySelector("[data-form]");
const input = form?.elements.namedItem("message");
const transcript = document.querySelector("[data-transcript]");
const resetButton = document.querySelector("[data-reset]");
const ruleOutput = document.querySelector("[data-rule]");
const decisionOutput = document.querySelector("[data-decision]");
const confidenceOutput = document.querySelector("[data-confidence]");
const marginOutput = document.querySelector("[data-margin]");
const evidenceOutput = document.querySelector("[data-evidence]");
const modelStatus = document.querySelector("[data-model-status]");
const traceStatus = document.querySelector("[data-trace-status]");
const labShell = document.querySelector(".lab-shell");
const gatedControls = document.querySelectorAll("[data-model-gated]");
const v3Status = document.querySelector("[data-v3-status]");
const selectionStatus = document.querySelector("[data-selection-status]");
const mobileMenu = document.querySelector(".mobile-menu");

mobileMenu?.querySelectorAll("a").forEach((link) => {
  link.addEventListener("click", () => {
    mobileMenu.removeAttribute("open");
  });
});

document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && mobileMenu?.hasAttribute("open")) {
    mobileMenu.removeAttribute("open");
    mobileMenu.querySelector("summary")?.focus();
  }
});

let engine = null;
let activeModel = null;

const setModelReady = () => {
  labShell?.setAttribute("aria-busy", "false");
  gatedControls.forEach((control) => {
    if (control instanceof HTMLButtonElement || control instanceof HTMLInputElement) {
      control.disabled = false;
    }
  });
};

const setModelFailed = () => {
  labShell?.setAttribute("aria-busy", "false");
  gatedControls.forEach((control) => {
    if (control instanceof HTMLButtonElement || control instanceof HTMLInputElement) {
      control.disabled = true;
    }
  });
};

const initializeModel = async () => {
  try {
    activeModel = await loadOpenSetBundle("./data/open-set-v3");
    engine = new ElizaEngine(activeModel);
    renderOpenSetReport(activeModel.metrics);
    if (modelStatus) modelStatus.textContent = `ML ${activeModel.version} VERIFIED`;
    setModelReady();
  } catch {
    activeModel = null;
    engine = null;
    if (modelStatus) modelStatus.textContent = "V3 VERIFICATION FAILED — INTERACTION DISABLED";
    if (v3Status) {
      v3Status.textContent = "V3 VERIFICATION FAILED — NO INFERENCE AVAILABLE";
      v3Status.dataset.state = "error";
    }
    setModelFailed();
  }
};

const finiteMetric = (value, label) => {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new TypeError(`${label} is not finite`);
  }
  return value;
};

const setV3Text = (selector, value) => {
  const target = document.querySelector(selector);
  if (target) target.textContent = value;
};

const renderOpenSetReport = (report) => {
  const counts = report.partition_counts;
  const id = report.id_test;
  const bootstrap = report.bootstrap_95;
  const ood = report.ood_test;
  const contrast = report.contrast_test;
  for (const [partition, suffix] of [
    ["train", "grouped rows"],
    ["development", "rows for model selection and thresholds"],
    ["calibration", "rows for temperature"],
    ["id-test", "untouched rows"],
  ]) {
    const count = counts[partition];
    setV3Text(`[data-v3-count="${partition}"]`, `${count} ${suffix}`);
  }
  const accuracy = finiteMetric(id.accuracy, "ID accuracy");
  const macroF1 = finiteMetric(id.macro_f1, "ID macro F1");
  const coverage = finiteMetric(id.coverage, "ID coverage");
  const nll = finiteMetric(id.calibration.negative_log_likelihood, "ID NLL");
  const auroc = finiteMetric(ood.discrimination.auroc, "OOD AUROC");
  const fpr95 = finiteMetric(ood.discrimination.fpr_at_95_tpr, "OOD FPR95");
  const oodCoverage = finiteMetric(ood.coverage, "OOD coverage");
  const contrastPairAccuracy = finiteMetric(
    contrast.pair_accuracy,
    "contrast pair accuracy",
  );
  const baselineDelta = finiteMetric(
    report.baselines.learned_minus_unigram_macro_f1,
    "learned-minus-unigram macro F1",
  );
  setV3Text('[data-v3-metric="accuracy"]', `${(accuracy * 100).toFixed(1)}%`);
  setV3Text('[data-v3-metric="macro-f1"]', macroF1.toFixed(3));
  setV3Text(
    '[data-v3-metric="coverage"]',
    `${Math.round(coverage * id.example_count)} / ${id.example_count}`,
  );
  setV3Text('[data-v3-metric="nll"]', nll.toFixed(3));
  setV3Text('[data-v3-metric="ood-auroc"]', auroc.toFixed(3));
  setV3Text('[data-v3-metric="ood-fpr95"]', fpr95.toFixed(3));
  setV3Text('[data-v3-metric="ood-coverage"]', `${(oodCoverage * 100).toFixed(1)}%`);
  setV3Text(
    '[data-v3-metric="contrast-pair-accuracy"]',
    `${Math.round(contrastPairAccuracy * contrast.pair_count)} / ${contrast.pair_count}`,
  );
  setV3Text(
    '[data-v3-metric="baseline-delta"]',
    `${baselineDelta >= 0 ? "+" : ""}${baselineDelta.toFixed(3)}`,
  );
  setV3Text(
    '[data-v3-interval="accuracy"]',
    `95% bootstrap: ${(finiteMetric(bootstrap.id_accuracy.lower_95, "accuracy lower") * 100).toFixed(1)}–${(finiteMetric(bootstrap.id_accuracy.upper_95, "accuracy upper") * 100).toFixed(1)}%`,
  );
  setV3Text(
    '[data-v3-interval="macro-f1"]',
    `95% bootstrap: ${finiteMetric(bootstrap.id_macro_f1.lower_95, "macro F1 lower").toFixed(3)}–${finiteMetric(bootstrap.id_macro_f1.upper_95, "macro F1 upper").toFixed(3)}`,
  );
  setV3Text(
    '[data-v3-interval="ood-auroc"]',
    `95% bootstrap: ${finiteMetric(bootstrap.ood_auroc.lower_95, "AUROC lower").toFixed(3)}–${finiteMetric(bootstrap.ood_auroc.upper_95, "AUROC upper").toFixed(3)}`,
  );
  if (v3Status) {
    v3Status.textContent = `VERIFIED MODEL ${report.model_version} / ${bootstrap.resamples} BOOTSTRAP RESAMPLES`;
    v3Status.dataset.state = "ready";
  }
};

const renderSelectionAudit = (report) => {
  const metrics = report.metrics;
  const bootstrap = report.bootstrap_95;
  const mostSelected = [...report.candidate_stability].sort(
    (left, right) =>
      right.selected_folds - left.selected_folds ||
      left.candidate_id.localeCompare(right.candidate_id),
  )[0];
  setV3Text(
    '[data-selection-metric="accuracy"]',
    `${(finiteMetric(metrics.accuracy, "selection accuracy") * 100).toFixed(1)}%`,
  );
  setV3Text(
    '[data-selection-metric="macro-f1"]',
    finiteMetric(metrics.macro_f1, "selection macro F1").toFixed(3),
  );
  setV3Text(
    '[data-selection-metric="interval"]',
    `${(finiteMetric(bootstrap.accuracy.lower_95, "selection lower") * 100).toFixed(1)}–${(finiteMetric(bootstrap.accuracy.upper_95, "selection upper") * 100).toFixed(1)}%`,
  );
  setV3Text(
    '[data-selection-metric="candidate"]',
    `${mostSelected.selected_folds} / ${report.outer_folds}`,
  );
  if (selectionStatus) {
    selectionStatus.textContent = `${report.model_fits} FITS / ${report.pool.family_count} FAMILIES / COMPLETE OOF LEDGER VERIFIED`;
    selectionStatus.dataset.state = "ready";
  }
};

const initializeSelectionAudit = async () => {
  try {
    renderSelectionAudit(await loadSelectionAudit("./data/selection-stability-v1.json"));
  } catch {
    if (selectionStatus) {
      selectionStatus.textContent = "SELECTION AUDIT VERIFICATION FAILED";
      selectionStatus.dataset.state = "error";
    }
  }
};

const boundedCharacters = (value) => {
  let output = "";
  let count = 0;
  for (const character of value) {
    if (count === MAX_INPUT_CHARS) return { text: output, truncated: true };
    output += character;
    count += 1;
  }
  return { text: output, truncated: false };
};

const displayBounded = (value) => {
  const bounded = boundedCharacters(value);
  return bounded.truncated ? `${bounded.text}…` : bounded.text;
};

const trimTranscript = () => {
  while (transcript && transcript.childElementCount > MAX_TRANSCRIPT_MESSAGES) {
    transcript.firstElementChild?.remove();
  }
};

const appendMessage = (author, text, turn, className, alert = false) => {
  const article = document.createElement("article");
  article.className = `message ${className}`;
  article.dataset.turn = String(turn);
  if (alert) article.setAttribute("role", "alert");
  const label = document.createElement("span");
  label.textContent = `${author} / ${String(turn).padStart(2, "0")}`;
  const paragraph = document.createElement("p");
  paragraph.textContent = text;
  article.append(label, paragraph);
  transcript?.append(article);
};

const scrollTranscript = () => {
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  transcript?.scrollTo({ top: transcript.scrollHeight, behavior: reduceMotion ? "auto" : "smooth" });
};

const runPrompt = (value) => {
  if (!activeModel || !engine) return;
  const prompt = String(value ?? "").trim();
  if (!prompt) return;

  const reply = engine.respond(prompt);
  appendMessage("YOU", displayBounded(prompt), reply.turn, "message-user");
  appendMessage(
    "ELIZA",
    reply.text,
    reply.turn,
    reply.rule === "safety-boundary" ? "message-system message-safety" : "message-system",
    reply.rule === "safety-boundary",
  );
  trimTranscript();
  scrollTranscript();
  if (ruleOutput) ruleOutput.textContent = reply.rule;
  if (decisionOutput) {
    decisionOutput.textContent = reply.model
      ? `${reply.model.label} / ${reply.model.accepted ? "accepted" : "abstained"}`
      : reply.keyword ?? "hard boundary";
  }
  if (confidenceOutput) {
    confidenceOutput.textContent = reply.model
      ? `${(reply.model.confidence * 100).toFixed(1)}%`
      : "—";
  }
  if (marginOutput) {
    marginOutput.textContent = reply.model ? reply.model.margin.toFixed(3) : "—";
  }
  if (evidenceOutput) {
    evidenceOutput.textContent = reply.model?.topFeatures.length
      ? reply.model.topFeatures
          .slice(0, 3)
          .map(({ feature, contribution }) => `${feature} (${contribution.toFixed(3)})`)
          .join(" · ")
      : "—";
  }
  if (traceStatus) traceStatus.textContent = `TURN ${String(reply.turn).padStart(2, "0")}`;
};

form?.addEventListener("submit", (event) => {
  event.preventDefault();
  if (!(input instanceof HTMLInputElement)) return;
  runPrompt(input.value);
  input.value = "";
  input.focus();
});

input?.addEventListener("input", () => {
  if (!(input instanceof HTMLInputElement)) return;
  const bounded = boundedCharacters(input.value);
  if (!bounded.truncated) return;
  input.value = bounded.text;
  if (traceStatus) traceStatus.textContent = "INPUT LIMIT";
});

document.querySelectorAll("[data-sample]").forEach((button) => {
  button.addEventListener("click", () => {
    if (!(button instanceof HTMLButtonElement)) return;
    runPrompt(button.dataset.sample ?? "");
    input?.focus();
  });
});

resetButton?.addEventListener("click", () => {
  if (!activeModel) return;
  engine = new ElizaEngine(activeModel);
  transcript?.replaceChildren();
  appendMessage("ELIZA", "Hello. What is on your mind today?", 0, "message-system");
  if (ruleOutput) ruleOutput.textContent = "waiting-for-input";
  if (decisionOutput) decisionOutput.textContent = "—";
  if (confidenceOutput) confidenceOutput.textContent = "—";
  if (marginOutput) marginOutput.textContent = "—";
  if (evidenceOutput) evidenceOutput.textContent = "—";
  if (traceStatus) traceStatus.textContent = "CLEARED";
  input?.focus();
});

void initializeModel();
void initializeSelectionAudit();
