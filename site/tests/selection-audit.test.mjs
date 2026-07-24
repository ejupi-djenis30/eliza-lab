import assert from "node:assert/strict";
import { webcrypto } from "node:crypto";
import { readFile } from "node:fs/promises";
import test from "node:test";

import {
  EXPECTED_SELECTION_AUDIT_SHA256,
  loadSelectionAudit,
  verifySelectionAudit,
} from "../selection-audit.mjs";

const reportBytes = await readFile(
  new URL("../../reports/selection-stability-v1.json", import.meta.url),
);
const report = JSON.parse(reportBytes);
const workflows = await Promise.all(
  ["ci.yml", "pages.yml", "release.yml"].map(async (name) => [
    name,
    await readFile(new URL(`../../.github/workflows/${name}`, import.meta.url), "utf8"),
  ]),
);

const clone = (value) => structuredClone(value);

test("the pre-test audit reconstructs its complete OOF evidence", () => {
  const verified = verifySelectionAudit(clone(report));
  assert.equal(verified.metrics.example_count, 385);
  assert.equal(verified.pool.family_count, 77);
  assert.equal(verified.outer_folds, 11);
  assert.equal(verified.inner_folds, 5);
  assert.equal(verified.grid.length, 9);
  assert.equal(verified.model_fits, 506);
});

test("the browser trust root pins the canonical report bytes", async () => {
  const actual = Buffer.from(
    await webcrypto.subtle.digest("SHA-256", reportBytes),
  ).toString("hex");
  assert.equal(actual, EXPECTED_SELECTION_AUDIT_SHA256);
  const loaded = await loadSelectionAudit("https://static.invalid/selection.json", {
    crypto: webcrypto,
    fetch: async () => new Response(reportBytes),
  });
  assert.deepEqual(loaded, report);
});

test("forged aggregates, ledgers and selection counts fail closed", () => {
  const forgedMetric = clone(report);
  forgedMetric.metrics.accuracy += 0.01;
  assert.throws(() => verifySelectionAudit(forgedMetric), /does not reconstruct/u);

  const duplicate = clone(report);
  duplicate.ledger[1].id = duplicate.ledger[0].id;
  assert.throws(() => verifySelectionAudit(duplicate), /OOF IDs must be unique/u);

  const forgedSelection = clone(report);
  forgedSelection.candidate_stability[0].selected_folds += 1;
  assert.throws(() => verifySelectionAudit(forgedSelection), /selection count differs/u);

  const forgedBootstrapSeed = clone(report);
  forgedBootstrapSeed.bootstrap_95.seed += 1;
  assert.throws(() => verifySelectionAudit(forgedBootstrapSeed), /seed derivation differs/u);
});

test("a report changed in transit fails its pinned digest", async () => {
  const tampered = Buffer.from(reportBytes);
  tampered[tampered.length - 2] ^= 1;
  await assert.rejects(
    loadSelectionAudit("https://static.invalid/selection.json", {
      crypto: webcrypto,
      fetch: async () => new Response(tampered),
    }),
    /digest does not match/u,
  );
});

test("CI, Pages and release reproduce the complete audit before use", () => {
  for (const [name, workflow] of workflows) {
    const audit = workflow.indexOf("selection audit");
    const comparison = workflow.indexOf("cmp reports/selection-stability-v1.json", audit);
    assert.ok(audit >= 0, `${name} must rerun the nested audit`);
    assert.ok(comparison > audit, `${name} must compare the regenerated report bytes`);
  }
  const pages = workflows.find(([name]) => name === "pages.yml")[1];
  assert.match(
    pages,
    /cp reports\/selection-stability-v1\.json pages-dist\/data\/selection-stability-v1\.json/u,
  );
});
