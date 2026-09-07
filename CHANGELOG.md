# Changelog

## Unreleased

- Make batch inference stop immediately on oversized input instead of draining an
  unbounded line. Count blank lines toward physical-line and total-byte budgets, reject
  duplicate IDs, and report physical error locations without echoing submitted values.
- Refresh the pinned official RustSec database while preserving the 14-day freshness
  limit and zero-warning audit policy.

## 1.6.0 — 2026-07-30

- Add `doctor --json`, a prompt-free self-test that verifies the embedded open-set bundle,
  compiles the classifier and exercises its inference and safety boundaries without reading user
  input.
- Complete the move to the ELIZA Lab repository and Pages URL while keeping exact verification
  instructions for immutable releases signed under the former repository identity.
- Rework the mobile navigation, narrow-screen evidence layout and public touch targets so the full
  technical presentation remains usable without a desktop viewport.
- Refresh the pinned RustSec advisory database while retaining the fail-closed 14-day freshness and
  zero-warning release policy.

## 1.5.0 — 2026-07-24

- Add a typed `SelectionAuditPool` containing only the 385 train and development rows. Run eleven
  outer family holdouts and five inner group-stratified folds across all nine declared candidates,
  producing one out-of-family prediction for every row without exposing calibration, ID-test, OOD
  or contrast inputs to the audit API.
- Persist a canonical 506-fit selection-stability report with the complete OOF probability ledger,
  candidate ranks and selection frequencies, per-class metrics, confusion matrix and 1,000
  deterministic family-cluster bootstrap intervals.
- Report the actual, lower pre-test result—`0.626` accuracy and `0.626` macro F1—and the movement in
  candidate choice instead of presenting only the stronger frozen ID-test result.
- Pin and semantically reconstruct the audit in the browser, add a responsive evidence section to
  Pages, and reproduce the report byte-for-byte in CI, Pages and release quality.
- Write the report atomically with persisted-byte verification, reject input/output path aliases,
  and make final-test and OOD flags explicit CLI errors.

## 1.4.0 — 2026-07-23

- Add a local metamorphic robustness audit for the verified open-set classifier. Measure label,
  acceptance and routed-decision agreement together with confidence drift and normalized
  Jensen–Shannon divergence across four formatting invariants and three controlled typo stresses.
- Stream JSONL behind row, physical-line, per-line and total-byte caps, or consume a non-forgeable
  verified bundle to reconstruct the frozen 70-row ID-test. Keep reports and parser failures free
  of identifiers, prompts, transformed text, submitted fields and row-level predictions.
- Enforce exact formatting invariants from unrounded in-process evidence by default, fail closed
  when only a serialized report is available and let release operators opt into explicit
  typographic thresholds without feeding post-test observations back into model selection.
- Generate the frozen-ID robustness report in CI, Pages and release quality, reconstruct every
  derivable family aggregate, then bind each dashboard value and graph width to the verified report
  before deployment or publication.

## 1.3.0 — 2026-07-22

- Add an experimental open-set v3 path with explicit group IDs and group-disjoint train,
  development, calibration and ID-test partitions. Keep OOD-development separate from OOD-test and
  encode the no-test-leakage boundary in typed APIs.
- Fit a real temperature scale on calibration rows, select confidence and probability-margin
  thresholds only from development plus OOD-development, then report ECE, multiclass Brier, NLL,
  risk-coverage, AURC, OOD AUROC/AUPR/FPR@95TPR and 1,000 deterministic cluster-bootstrap
  intervals: ID families are resampled within labels and OOD examples by broader domain.
- Add SHA-256-linked v3 model, operating-policy, metrics and split-plan artifacts with verify and
  in-memory reproduce commands. Compile and index a verified model once for bounded JSONL batch
  inference.
- Canonicalize persisted metrics to a documented nine-decimal reporting precision, pin LF bytes
  for digest-bound files, and smoke-test both embedded and external bundles on every native release
  target so Windows and macOS builds preserve byte-identical provenance.
- Replace top-class-only evidence in v3 with contrastive top-versus-runner-up contributions whose
  bias and feature terms are tested against the exact logit margin. Present the wider intervals and
  weak OOD FPR on Pages instead of upgrading the project claim.

## 1.2.0 — 2026-07-22

- Replace the cosmetic rule-only framing with a real, local intent-classification pipeline:
  validated TSV data, a deterministic stratified split, training-only TF-IDF features,
  multinomial logistic regression, L2 regularization, versioned JSON weights, and reproducible
  train/evaluate/infer commands.
- Add uncertainty calibration that uses training data plus a separate unlabeled OOD fixture while
  leaving the 21-row holdout untouched until final evaluation. Record probabilities, margins,
  feature contributions, per-class metrics, a confusion matrix, and every holdout decision.
- Embed model `1.0.0` and the synthetic fixtures in native CLI builds, retain the deterministic
  rule mode, and enforce input and non-clinical safety boundaries before learned inference.
- Run the same model in Rust and the browser with a shared parity fixture. If the static model
  cannot load or validate, the site identifies its deterministic rule fallback explicitly.
- Add a model card, dataset contract, architecture guide, OOD limitations, byte-reproducibility
  tests, strict model-validation tests, safe output-path handling, and generated-report checks.
- Rebuild the Pages presentation around code-rendered pipeline geometry and restrained report
  copy that states the small holdout, 14/21 raw accuracy, 0.661 macro-F1, and 7/21 coverage without
  making general NLP or clinical claims.

## 1.1.2 — 2026-07-20

- Reissue the verified builds from the repository's privacy-safe history. The ELIZA engine and
  interface are unchanged; only commit attribution and release provenance changed.
- Keep the retired immutable tags unavailable instead of reusing them, so the release advances to
  `v1.1.2`.

## 1.1.1 — 2026-07-20

- Wait for GitHub's release listing to expose a newly created draft before uploading any asset.
- Keep delayed draft discovery bounded and fail closed without mutating an undiscoverable draft.
- Cover both eventual-consistency recovery and retry exhaustion with deterministic publisher tests.

## 1.1.0 — 2026-07-20

- Add a tag-gated release pipeline for Linux x64, Windows x64, macOS Intel, and Apple Silicon with
  native CLI smoke tests and platform-appropriate archives.
- Verify Cargo/tag/commit parity, local and remote artifact inventory, SHA-256 checksums, dependency
  evidence, SPDX 2.3 output, and GitHub build provenance before a release can be published.
- Resume only an exact contract-bearing draft, verify the uploaded remote bytes, and recheck both
  tag and inventory before the draft-to-published transition.
- Run the complete packaging contract on pull requests and manual workflow runs without publishing.
- Pin the RustSec scanner and advisory database, and carry its no-warning result into the verified
  release inventory.
- Require a pushed tag at the current default-branch tip for initial authorization, a dated
  changelog section, safe file snapshots, immutable releases, and idempotent rerun verification.
- Independently verify every GitHub attestation identity before publication and confirm the final
  release is both immutable and latest.
- Discover interrupted drafts through GitHub's authenticated paginated release listing and reject
  duplicate or foreign release state before mutation.
- License the project under MIT and authorize GitHub release publication only while the Cargo SPDX
  expression, repository license file, and versioned policy agree.
- Enforce a maximum age for the pinned RustSec database and check that its recorded commit time
  matches the fetched official commit.
- Pin CI and Pages runners and toolchains, and keep Cargo checks locked to the committed dependency
  graph.
- Replace substring safety checks with deterministic word-boundary phrase matching.
- Expand explicit safety exit phrases while documenting false positives and false negatives.
- Align Rust and browser tokenization, apostrophe handling and pronoun reflection.
- Add a shared Rust/JavaScript parity corpus.
- Bound CLI line reads, browser transcript growth and turn-counter overflow.
- Add a distinct accessible safety response and clearer input/session limits.
- Pin CI and Pages actions to reviewed commit SHAs and add weekly Dependabot coverage.

## 1.0.0 — 2026-07-19

- Rebuild the legacy Telegram experiment as a local-only Rust engine and static learning tool.
- Remove the database, remote model, accounts and therapeutic framing.
