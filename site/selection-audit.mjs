export const EXPECTED_SELECTION_AUDIT_SHA256 =
  "e6c0db0ecb0db89f9162549657e86986ae5945feacc0e1efd00b8d88e1ce507a";

const MAX_REPORT_BYTES = 1_048_576;
const TOLERANCE = 3e-9;

const invariant = (condition, message) => {
  if (!condition) throw new TypeError(`Selection audit: ${message}`);
};

const finite = (value, label) => {
  invariant(typeof value === "number" && Number.isFinite(value), `${label} must be finite`);
  return value;
};

const close = (actual, expected, label) => {
  invariant(
    Math.abs(finite(actual, `${label} actual`) - finite(expected, `${label} expected`)) <= TOLERANCE,
    `${label} does not reconstruct`,
  );
};

const sha256 = async (bytes, crypto) => {
  invariant(crypto?.subtle, "Web Crypto is unavailable");
  const digest = await crypto.subtle.digest("SHA-256", bytes);
  return [...new Uint8Array(digest)]
    .map((byte) => byte.toString(16).padStart(2, "0"))
    .join("");
};

const summarize = (rows, labels) => {
  const labelIndex = new Map(labels.map((label, index) => [label, index]));
  const confusion = labels.map(() => labels.map(() => 0));
  let correct = 0;
  let nll = 0;
  let brier = 0;
  for (const row of rows) {
    invariant(labelIndex.has(row.actual_label), `unknown actual label ${row.actual_label}`);
    invariant(labelIndex.has(row.predicted_label), `unknown predicted label ${row.predicted_label}`);
    invariant(row.correct === (row.actual_label === row.predicted_label), `incorrect row flag ${row.id}`);
    const actual = labelIndex.get(row.actual_label);
    const predicted = labelIndex.get(row.predicted_label);
    confusion[actual][predicted] += 1;
    if (actual === predicted) correct += 1;
    let probabilitySum = 0;
    for (const label of labels) {
      const probability = finite(row.probabilities?.[label], `${row.id}/${label} probability`);
      invariant(probability >= 0 && probability <= 1, `${row.id}/${label} probability is out of range`);
      probabilitySum += probability;
      brier += (probability - Number(label === row.actual_label)) ** 2;
    }
    close(probabilitySum, 1, `${row.id} probability sum`);
    nll -= Math.log(Math.max(row.probabilities[row.actual_label], 1e-15));
  }
  const perClass = labels.map((label, index) => {
    const support = confusion[index].reduce((sum, count) => sum + count, 0);
    const predicted = confusion.reduce((sum, row) => sum + row[index], 0);
    const truePositive = confusion[index][index];
    const precision = predicted === 0 ? 0 : truePositive / predicted;
    const recall = support === 0 ? 0 : truePositive / support;
    const f1 = precision + recall === 0 ? 0 : (2 * precision * recall) / (precision + recall);
    return { label, support, predicted, true_positive: truePositive, precision, recall, f1 };
  });
  return {
    example_count: rows.length,
    accuracy: correct / rows.length,
    macro_f1: perClass.reduce((sum, value) => sum + value.f1, 0) / perClass.length,
    negative_log_likelihood: nll / rows.length,
    multiclass_brier: brier / rows.length,
    confusion_matrix: confusion,
    per_class: perClass,
  };
};

const verifySummary = (actual, expected, label) => {
  invariant(actual.example_count === expected.example_count, `${label} example count differs`);
  invariant(
    JSON.stringify(actual.confusion_matrix) === JSON.stringify(expected.confusion_matrix),
    `${label} confusion matrix differs`,
  );
  for (const metric of [
    "accuracy",
    "macro_f1",
    "negative_log_likelihood",
    "multiclass_brier",
  ]) {
    close(actual[metric], finite(expected[metric], `${label}/${metric}`), `${label}/${metric}`);
  }
  invariant(actual.per_class.length === expected.per_class.length, `${label} class count differs`);
  actual.per_class.forEach((value, index) => {
    const recorded = expected.per_class[index];
    invariant(
      value.label === recorded.label &&
        value.support === recorded.support &&
        value.predicted === recorded.predicted &&
        value.true_positive === recorded.true_positive,
      `${label}/${value.label} counts differ`,
    );
    for (const metric of ["precision", "recall", "f1"]) {
      close(value[metric], recorded[metric], `${label}/${value.label}/${metric}`);
    }
  });
};

export const verifySelectionAudit = (report) => {
  invariant(report?.schema_version === 1, "unsupported schema");
  invariant(report.audit_kind === "eliza-selection-stability-audit", "unexpected audit kind");
  invariant(
    report.algorithm === "nested-group-stratified-selection-stability-v1",
    "unexpected algorithm",
  );
  invariant(report.pool?.example_count === 385, "selection pool must contain 385 rows");
  invariant(report.pool.family_count === 77, "selection pool must contain 77 families");
  invariant(report.pool.label_count === 7, "selection pool must contain seven labels");
  invariant(report.pool.families_per_label === 11, "selection pool must contain eleven families per label");
  invariant(report.pool.examples_per_family === 5, "selection families must contain five rows");
  invariant(
    report.training?.optimizer ===
      "deterministic-full-batch-multinomial-logistic-regression" &&
      report.training.seed === report.pool.audit_seed &&
      report.training.epochs === 600 &&
      report.training.learning_rate === 0.8 &&
      report.training.word_ngram_min === 1 &&
      report.training.word_ngram_max === 2 &&
      report.training.char_ngram_min === 3 &&
      report.training.char_ngram_max === 5 &&
      report.training.min_document_frequency === 1 &&
      report.training.macro_f1_tolerance === 0.005,
    "training configuration differs",
  );
  invariant(report.outer_folds === 11 && report.inner_folds === 5, "nested fold contract differs");
  invariant(report.model_fits === 506, "model-fit ledger differs");
  invariant(Array.isArray(report.grid) && report.grid.length === 9, "grid must contain nine candidates");
  const gridIds = report.grid.map(({ candidate_id }) => candidate_id);
  invariant(new Set(gridIds).size === 9, "candidate IDs must be unique");
  invariant(Array.isArray(report.folds) && report.folds.length === 11, "outer fold report differs");
  invariant(Array.isArray(report.ledger) && report.ledger.length === 385, "OOF ledger differs");

  const ids = new Set();
  const families = new Set();
  const familiesByLabel = new Map();
  for (const row of report.ledger) {
    invariant(typeof row.id === "string" && !ids.has(row.id), "OOF IDs must be unique");
    invariant(typeof row.family_id === "string", `${row.id} has no family`);
    invariant(Number.isInteger(row.outer_fold) && row.outer_fold >= 1 && row.outer_fold <= 11, `${row.id} has an invalid fold`);
    invariant(!("text" in row) && !("prompt" in row), `${row.id} leaks source text`);
    ids.add(row.id);
    families.add(row.family_id);
    if (!familiesByLabel.has(row.actual_label)) familiesByLabel.set(row.actual_label, new Set());
    familiesByLabel.get(row.actual_label).add(row.family_id);
  }
  invariant(families.size === 77, "ledger family count differs");
  invariant(
    familiesByLabel.size === 7 &&
      [...familiesByLabel.values()].every((members) => members.size === 11),
    "families are not balanced by label",
  );

  const labels = report.metrics?.labels;
  invariant(Array.isArray(labels) && labels.length === 7, "metric labels differ");
  verifySummary(summarize(report.ledger, labels), report.metrics, "complete OOF ledger");

  const selectedCounts = new Map(gridIds.map((candidateId) => [candidateId, 0]));
  const ranksByCandidate = new Map(gridIds.map((candidateId) => [candidateId, []]));
  const heldFamilies = new Set();
  for (const fold of report.folds) {
    invariant(
      Array.isArray(fold.outer_holdout_families) && fold.outer_holdout_families.length === 7,
      `fold ${fold.fold} outer families differ`,
    );
    fold.outer_holdout_families.forEach((family) => {
      invariant(!heldFamilies.has(family), `family ${family} is held out twice`);
      heldFamilies.add(family);
    });
    const innerFamilies = fold.inner_holdout_families?.flat() ?? [];
    invariant(
      fold.inner_holdout_families?.length === 5 &&
        fold.inner_holdout_families.every((familiesInFold) => familiesInFold.length === 14) &&
        innerFamilies.length === 70 &&
        new Set(innerFamilies).size === 70 &&
        innerFamilies.every((family) => !fold.outer_holdout_families.includes(family)) &&
        innerFamilies.every((family) => families.has(family)),
      `fold ${fold.fold} inner family assignment differs`,
    );
    const innerCandidateIds = fold.inner_candidates?.map(({ candidate_id }) => candidate_id) ?? [];
    const innerRanks = fold.inner_candidates?.map(({ rank }) => rank) ?? [];
    invariant(
      innerCandidateIds.length === 9 &&
        JSON.stringify([...innerCandidateIds].sort()) === JSON.stringify([...gridIds].sort()) &&
        JSON.stringify([...innerRanks].sort((left, right) => left - right)) ===
          JSON.stringify([1, 2, 3, 4, 5, 6, 7, 8, 9]),
      `fold ${fold.fold} candidate ranking differs`,
    );
    fold.inner_candidates.forEach(({ candidate_id: candidateId, rank }) => {
      ranksByCandidate.get(candidateId).push(rank);
    });
    invariant(selectedCounts.has(fold.selected_candidate_id), `fold ${fold.fold} selected an unknown candidate`);
    selectedCounts.set(fold.selected_candidate_id, selectedCounts.get(fold.selected_candidate_id) + 1);
    const rows = report.ledger.filter(({ outer_fold }) => outer_fold === fold.fold);
    invariant(rows.length === 35, `fold ${fold.fold} must own 35 OOF rows`);
    invariant(
      rows.every(({ selected_candidate_id: candidateId }) => candidateId === fold.selected_candidate_id),
      `fold ${fold.fold} ledger candidate differs`,
    );
    invariant(
      JSON.stringify([...new Set(rows.map(({ family_id: familyId }) => familyId))].sort()) ===
        JSON.stringify([...fold.outer_holdout_families].sort()),
      `fold ${fold.fold} ledger families differ`,
    );
    const foldLabelCounts = new Map(labels.map((label) => [label, 0]));
    rows.forEach(({ actual_label: label }) => foldLabelCounts.set(label, foldLabelCounts.get(label) + 1));
    invariant(
      [...foldLabelCounts.values()].every((count) => count === 5),
      `fold ${fold.fold} is not label-balanced`,
    );
    const summary = summarize(rows, labels);
    for (const metric of [
      "accuracy",
      "macro_f1",
      "negative_log_likelihood",
      "multiclass_brier",
    ]) {
      close(summary[metric], fold[`outer_${metric}`], `fold ${fold.fold}/${metric}`);
    }
  }
  invariant(heldFamilies.size === 77, "outer folds do not cover every family exactly once");

  invariant(report.candidate_stability?.length === 9, "candidate stability table differs");
  for (const candidate of report.candidate_stability) {
    invariant(
      selectedCounts.get(candidate.candidate_id) === candidate.selected_folds,
      `${candidate.candidate_id} selection count differs`,
    );
    close(
      candidate.selection_rate,
      candidate.selected_folds / report.outer_folds,
      `${candidate.candidate_id} selection rate`,
    );
    const ranks = ranksByCandidate.get(candidate.candidate_id);
    invariant(ranks?.length === report.outer_folds, `${candidate.candidate_id} rank ledger differs`);
    const meanRank = ranks.reduce((sum, rank) => sum + rank, 0) / ranks.length;
    const rankVariance =
      ranks.reduce((sum, rank) => sum + (rank - meanRank) ** 2, 0) / ranks.length;
    close(candidate.mean_rank, meanRank, `${candidate.candidate_id} mean rank`);
    close(
      candidate.rank_standard_deviation,
      Math.sqrt(rankVariance),
      `${candidate.candidate_id} rank deviation`,
    );
  }
  for (const metric of [
    "accuracy",
    "macro_f1",
    "negative_log_likelihood",
    "multiclass_brier",
  ]) {
    const interval = report.bootstrap_95?.[metric];
    close(interval?.value, report.metrics[metric], `${metric} bootstrap point`);
    invariant(
      finite(interval.lower_95, `${metric} lower`) <= interval.value &&
        interval.value <= finite(interval.upper_95, `${metric} upper`),
      `${metric} point is outside its interval`,
    );
  }
  invariant(report.bootstrap_95.resamples === 1_000, "bootstrap resample count differs");
  const expectedBootstrapSeed = Number(
    (BigInt(report.pool.audit_seed) ^ 0xbb67ae8584caa73bn) & ((1n << 53n) - 1n),
  );
  invariant(
    Number.isSafeInteger(report.bootstrap_95.seed) &&
      report.bootstrap_95.seed === expectedBootstrapSeed,
    "bootstrap seed derivation differs",
  );
  return report;
};

export const loadSelectionAudit = async (
  url,
  { fetch: fetchImplementation = globalThis.fetch, crypto = globalThis.crypto } = {},
) => {
  invariant(typeof fetchImplementation === "function", "fetch is unavailable");
  const response = await fetchImplementation(url, { cache: "no-store" });
  invariant(response?.ok, "report request failed");
  const declaredLength = Number(response.headers?.get?.("content-length"));
  invariant(
    !Number.isFinite(declaredLength) || declaredLength <= MAX_REPORT_BYTES,
    "declared report is too large",
  );
  const bytes = await response.arrayBuffer();
  invariant(bytes.byteLength <= MAX_REPORT_BYTES, "report is too large");
  invariant(
    (await sha256(bytes, crypto)) === EXPECTED_SELECTION_AUDIT_SHA256,
    "report digest does not match the release trust root",
  );
  return verifySelectionAudit(JSON.parse(new TextDecoder().decode(bytes)));
};
