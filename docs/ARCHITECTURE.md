# Architecture

ELIZA Lab is a local, deterministic open-set text-classification system. The v3 bundle is the
default inference path in the CLI and browser. The original v1 artifact remains available only
through the explicit `--legacy-v1` compatibility flag.

## Components

1. `src/open_set.rs` owns the v3 dataset contracts, typed partitions, model selection, training,
   calibration, abstention policy, evaluation, baselines, bootstrap, artifact verification and
   compiled inference.
2. `src/open_set/selection_audit.rs` owns the train-plus-development capability, nested
   family-disjoint cross-validation, OOF ledger and family-cluster uncertainty report. Final-test
   and OOD types are not part of this module's public input.
3. `src/robustness.rs` owns bounded JSONL robustness input, deterministic metamorphic
   transformations, aggregate stability metrics and optional release gates. Caller-provided cases
   use a compiled model; the frozen ID-test path accepts and consumes only a `VerifiedBundle`.
   Neither path can mutate or select a model.
4. `src/lib.rs` owns bounded dialogue behaviour. Empty or oversized input and explicit safety-stop
   phrases are handled before learned inference.
5. `src/main.rs` exposes v3 training, selection stability, verification, reproduction, batch
   inference, aggregate robustness auditing and interactive commands. Legacy inference must be
   requested explicitly.
6. `site/open-set-engine.mjs` verifies the same five-file bundle, reproduces its prediction
   ledgers and runs inference in the browser. A missing trust root, digest mismatch or semantic
   mismatch disables the interface.
7. `site/selection-audit.mjs` pins the canonical audit digest and reconstructs the complete OOF
   confusion matrix, probability losses, fold metrics, candidate ranks and selection counts.
8. `site/app.js` renders only successfully verified evidence. Prompts stay in the tab.

## Experiment flow

```text
validated grouped corpus
    → deterministic family-disjoint SplitPlan
        ├── TrainingPartition → TF-IDF + multinomial logistic regression
        ├── DevelopmentPartition → fixed 3 × 3 model grid
        │                           macro-F1 epsilon, simpler candidate first
        ├── CalibrationPartition → temperature scaling
        ├── DevelopmentPartition + OodDevelopmentPartition
        │       → fixed 7 × 7 confidence/margin grid
        ├── ID-test → final in-domain metrics + majority/unigram baselines
        ├── OOD-test → aggregate and per-stratum open-set metrics
        └── ContrastTestPartition → paired anti-shortcut metrics

family/domain cluster bootstrap
    → 95% intervals
        → model.json + policy.json + metrics.json + split-plan.json
            → SHA-256 manifest
```

The role-specific Rust types are capabilities, not labels on arbitrary slices:

- the fitter accepts only `TrainingPartition`;
- candidate selection accepts training and development capabilities;
- temperature calibration accepts only `CalibrationPartition`;
- threshold selection accepts development and OOD-development capabilities;
- contrast evaluation accepts only `ContrastTestPartition`;
- no final test is present in any selector signature.

The development partition serves both candidate selection and threshold selection. That can still
create selection optimism, so the limitation is recorded in the artifact.

## Pre-test selection audit

The audit is a parallel evidence path. `SelectionAuditPool::from_dataset` follows the frozen
four-way split rule but retains only train and development: 385 rows, 77 families and eleven
families per label. Its fingerprint derives the audit seed. The capability contains no
calibration, ID-test, OOD or contrast field.

Eleven outer folds each hold out one family per label. For each outer fold, five inner folds divide
the remaining ten families per label evenly and evaluate the complete nine-candidate grid. The
selected candidate is refit on the outer training population and predicts the 35 outer rows once.
The final 385-row OOF ledger therefore contains no prediction from a model trained on that row's
family.

The canonical report records both outer and inner family assignments, training configuration,
candidate metrics and ranks, selection frequencies, the OOF probability ledger, confusion matrix,
per-class metrics and 1,000 family-cluster bootstrap intervals. CI, Pages and release quality rerun
all 506 fits in release mode and compare the resulting bytes with the checked-in report. This audit
measures selection stability only; it neither changes nor reopens the frozen v3 test results.

## Model and policy

The vectorizer extracts word uni- and bigrams and character 3- to 5-grams, fits smoothed IDF on
training only, keeps a development-selected feature budget and L2-normalizes sparse vectors. A
deterministic full-batch multinomial logistic regression learns seven class scores.

The fixed model grid is declared in source before the final run. Candidates within `0.005`
macro-F1 are treated as practically tied; the comparator then prefers fewer features, stronger L2
regularization, accuracy, lower NLL and lower Brier score, in that order.

Temperature scaling uses calibration only. A coarse, fixed 49-point confidence/margin grid then
chooses the highest-coverage development operating point that reaches the minimum selective
accuracy and maximum OOD-development coverage constraints.

An empty feature vector always abstains. Accepted predictions expose all class probabilities and a
contrastive feature explanation. Bias difference plus every feature contribution must reconstruct
the exact top-two logit margin.

## Post-training robustness audit

The robustness audit is outside the experiment-selection graph. It streams caller-provided JSONL
through a verified compiled model, applies a fixed set of feature-equivalent formatting
transformations and controlled single-edit typo stresses, then emits aggregate-only stability
metrics. It does not retain input rows or add results to the signed v3 bundle.

The CLI can stream caller-provided cases or reconstruct the frozen ID-test directly from a
semantically verified bundle. `VerifiedBundle` has private fields, and the ID-test audit consumes
that capability while compiling the runtime internally. Arbitrary parsed cases can therefore
produce only the `provided-cases` provenance. The frozen ID-test mode is a regression diagnostic
only; no audit result is reachable from model fitting, calibration or policy selection.

Formatting transformations must reconstruct the same features, so their default release gate is
exact on unrounded in-memory measurements. The deterministic JSON report is quantized to nine
decimal places, but serialized reports do not carry raw gate evidence and cannot be gated after
deserialization. Typographic thresholds are opt-in and must be declared by the caller; they cannot
retroactively influence the frozen model or policy. Normalized Jensen–Shannon divergence measures
probability drift, while routed-decision agreement distinguishes accepted label changes from two
stable abstentions. CI, Pages and release quality generate a fresh frozen-ID report, reconstruct
its family aggregates and bind the site metrics to it before deployment or publication. See
[ROBUSTNESS.md](ROBUSTNESS.md).

## Evaluation

The final ID ledger records every prediction. Metrics include accuracy, macro F1, confusion matrix,
per-class precision/recall/F1, calibration NLL/Brier/ECE, selective coverage and AURC.

Two deterministic training-only baselines are evaluated on the same ID-test rows:

- alphabetical-tie-broken majority class;
- Laplace-smoothed multinomial unigram Naive Bayes.

The report includes learned-minus-unigram deltas and adds an explicit limitation if the learned
model does not beat the unigram baseline on both accuracy and macro F1.

OOD metrics include coverage, AUROC, AUPR with ID as positive and FPR at 95% TPR, both in aggregate
and for semantic, capability and noise strata. ID intervals resample held-out families within each
label. OOD intervals resample broader domains. A separate fourteen-pair contrast test reports
whether small meaning-changing edits cause the prediction to change and whether both sides of each
pair are classified correctly.

## Artifact trust and reproduction

The bundle contains exactly five regular files:

```text
manifest.json
metrics.json
model.json
policy.json
split-plan.json
```

Rust verification checks strict schemas, bounded sizes, provenance, model/policy consistency,
prediction ledgers, contrast-pair summaries, baseline reconstruction and SHA-256 digests. `bundle reproduce`
reruns the deterministic experiment and compares all four payload digests.

The browser carries the expected manifest digest as its release trust root. It verifies all payload
digests, source-row fingerprints, prediction ledgers, calibration summaries, threshold
observations, baseline results and OOD strata before enabling controls. Verification failure is a
hard stop, not a silent fallback presented as ML. Neither runtime accepts a contrast report that
cannot be rebuilt row by row from the frozen split plan.

## Failure boundaries

- Invalid or overlapping TSV populations stop before training.
- Dataset similarity review and bootstrap workloads have explicit upper bounds.
- Unknown JSON fields, unsupported versions, non-finite values, oversized files, symlinks and
  unexpected bundle files are rejected.
- Bundle output never replaces an unrelated non-empty directory.
- CLI and browser inference are local and do not persist prompts.
- Robustness reports are aggregate-only and omit IDs, prompts, transformed text and row-level
  predictions. The reader bounds row count, physical lines, per-line bytes, total bytes and
  identifier memory, and schema failures do not expose parser details or submitted fields.
- The safety phrase list is an exit condition, not crisis detection.
- The classifier is not suitable for clinical, safety, employment or other decisions about people.
