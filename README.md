<p align="center">
  <img src="site/assets/eliza-lab-lockup.svg" width="720" alt="ELIZA Lab — local machine learning you can inspect" />
</p>

# ELIZA Lab

[![CI](https://github.com/ejupi-djenis30/eliza-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/ejupi-djenis30/eliza-lab/actions/workflows/ci.yml)

> Train a real intent classifier locally, reproduce its evaluation, and inspect every decision.

This repository began as a Telegram “psychologist” bot. That framing was misleading, and the
implementation stored sensitive conversations in an insecure database. The project has been
rebuilt as **ELIZA Lab**: an educational Rust machine-learning pipeline and browser lab with no
accounts, prompt submission, transcript storage, diagnosis, or therapeutic claims.

[Open the interactive lab](https://ejupi-djenis30.github.io/eliza-lab/) ·
[Model card](docs/MODEL_CARD.md) · [Dataset contract](docs/DATASET.md) ·
[Architecture](docs/ARCHITECTURE.md) · [Release recovery](docs/RELEASE_RECOVERY.md) ·
[Safety model](SECURITY.md)

## What makes it useful

- **A complete learning pipeline:** strict TSV validation, a deterministic stratified split,
  training-only vocabulary fitting, TF-IDF features, multinomial logistic regression, evaluation,
  uncertainty calibration, serialization, and inference live in the repository.
- **Inspectable predictions:** JSON and browser traces expose all class probabilities, confidence,
  top-two margin, and the strongest positive feature contributions.
- **Reproducible artifacts:** a declared nine-decimal reporting precision makes the complete v3
  bundle byte-identical across supported release targets. The report still records every final
  prediction, while model and policy precision remain unchanged.
- **Real abstention:** calibration, policy selection, ID-test and OOD-test have different roles.
  Inputs with weak evidence abstain instead of being presented as confident classifications.
- **Private by construction:** CLI and browser inference are local. Prompts are not stored or sent
  to a model service.
- **Hard non-clinical boundary:** input limits and explicit safety-stop phrases run before ML
  inference. The stop is not presented as crisis detection.

## Open-set protocol v3

The checked-in [`artifacts/eliza-open-set-v3`](artifacts/eliza-open-set-v3) bundle is the primary
model path. The older v1 artifact remains available only through the explicit `--legacy-v1` flag.
V3 makes the evaluation boundary concrete:

- semantically related rows stay together through explicit `group_id` values;
- train, development, probability calibration and ID-test are four separate partitions;
- OOD-development selects the abstention thresholds, while a different OOD-test fixture measures
  the frozen policy afterward;
- a separate paired contrast test probes meaning-changing edits only after every choice is frozen.

Temperature scaling sees only calibration. A fixed, coarse confidence/margin grid sees only
development plus OOD-development. Role-specific Rust types keep both final tests out of every
selection API, and the bundle records those inputs directly.

The supervised fixture has 525 rows in 105 equal five-prompt families. Each OOD population has 36
rows in twelve multi-prompt families, six broader domains and balanced semantic, capability and
noise strata. The contrast fixture adds 28 rows in fourteen label-changing pairs. A training-only
majority baseline and Laplace unigram Naive Bayes baseline are reconstructed from the split plan
during verification.

The bundle separates `model.json`, `policy.json`, `metrics.json` and `split-plan.json`. A final
manifest links their exact bytes with SHA-256. Build, verify and reproduce it with:

```bash
cargo run --locked -- train-v3 --output target/open-set-v3
cargo run --locked -- bundle verify --bundle artifacts/eliza-open-set-v3
cargo run --locked -- bundle reproduce --bundle artifacts/eliza-open-set-v3
```

`train-v3` replaces only an empty destination or a bundle that already passes the complete v3
verification contract. It will not repurpose an unrelated non-empty directory.

## Pre-test selection stability

The frozen bundle still uses one development partition for candidate selection and threshold
selection. The separate [`selection-stability-v1.json`](reports/selection-stability-v1.json) report
measures how stable the model-grid choice is before any final test is available:

- its `SelectionAuditPool` contains only the 315 training and 70 development rows;
- eleven outer folds each hold out one whole family per label;
- five inner group-stratified folds compare all nine declared candidates;
- every one of the 385 rows receives exactly one prediction from a model that did not see its
  family;
- the constructor uses the fixed supervised split only to isolate train + development; calibration
  and ID-test are then discarded, while OOD and contrast data are not inputs or CLI options.

The actual out-of-fold result is `0.626` accuracy and `0.626` macro F1. The family-clustered 95%
intervals are `[0.571, 0.686]` and `[0.573, 0.682]`. Candidate choice also moves across folds: the
most frequently selected configuration wins six of eleven, not all eleven. This is intentionally
less flattering than the final ID-test score and is the reason the report exists.
These are raw candidate-label metrics before temperature calibration or abstention-policy
selection, not coverage estimates.

Reproduce the canonical 506-fit audit locally:

```bash
cargo run --release --locked -- selection audit \
  --output reports/selection-stability-v1.json
```

The command derives its audit seed from the selection-pool fingerprint and writes the report
atomically, then reads the persisted bytes back before returning their SHA-256.

Run bounded batch inference without validating or indexing the vocabulary again for every row:

```bash
printf '%s\n' '{"id":"sample-1","text":"Today I feel calm"}' \
  | cargo run --locked -- infer-batch --bundle artifacts/eliza-open-set-v3
```

Every v3 prediction includes calibrated probabilities and a contrastive explanation for the top
class against the runner-up. The explanation records its bias delta and feature sum, and tests
verify that they reconstruct the exact top-two logit margin.

Audit prediction stability locally without writing prompts or row-level results:

```bash
printf '%s\n' \
  '{"id":"fictional-01","text":"I intend to test one concrete next step"}' \
  '{"id":"fictional-02","text":"Today I feel calm about the outcome"}' \
  | cargo run --locked -- robustness audit
```

The audit measures decision agreement and normalized Jensen–Shannon divergence under four
feature-equivalent formatting changes and three controlled typographic edits. Formatting
invariants fail closed on unrounded measurements by default; typo thresholds are opt-in because
controlled noise is not a guaranteed paraphrase. Reports are aggregate-only and exclude IDs,
prompts, transformed text and row-level predictions. See the
[robustness contract](docs/ROBUSTNESS.md).

Run the same aggregate regression audit over the verified bundle's frozen 70-row ID-test without
materializing another input file:

```bash
cargo run --locked -- robustness audit --bundle-id-test \
  > target/robustness-id-test-report.json
node scripts/verify-robustness-report.mjs target/robustness-id-test-report.json
```

The verifier reconstructs every family aggregate and binds each dashboard value to that report,
the checked bundle manifest and policy, and the frozen ID-test ledger. CI, Pages and release
quality run the same check before deployment or publication.

The verified browser build is published with the project at
[`ejupi-djenis30.github.io/eliza-lab`](https://ejupi-djenis30.github.io/eliza-lab/). Its deployment
is built from the same reviewed runtime files and evidence bundle checked by CI.

## Install a verified build

Download the archive for your system from the
[latest release](https://github.com/ejupi-djenis30/eliza-lab/releases/latest):

| Platform | Release asset |
| --- | --- |
| Linux x64 | `eliza-lab-v<version>-linux-x86_64.tar.gz` |
| Windows x64 | `eliza-lab-v<version>-windows-x86_64.zip` |
| macOS Apple Silicon | `eliza-lab-v<version>-macos-aarch64.tar.gz` |
| macOS Intel | `eliza-lab-v<version>-macos-x86_64.tar.gz` |

Compare it with the matching `.sha256` file or `SHA256SUMS`, then verify its GitHub attestation.
Release `v1.5.0` was signed before the repository moved to `ejupi-djenis30/eliza-lab`, so its
immutable provenance retains the former repository identity:

```bash
gh attestation verify <downloaded-archive> --repo ejupi-djenis30/PsychologistRustBot
```

GitHub redirects that repository path to ELIZA Lab. Releases created after the rename use
`--repo ejupi-djenis30/eliza-lab`.

Extract the archive and run the included `eliza-lab` executable. Open-set bundle model `3.0.0`, the
legacy `1.0.0` compatibility artifact and synthetic fixtures are embedded, so inference,
verification and retraining need no separate model download.
The application version (`1.6.0`) and bundled model versions are intentionally independent.

## Run it

Verify a build from the current source tree before giving it any input:

```bash
cargo run --release --locked -- doctor --json
```

The self-test reads no prompt, file, environment variable or standard input. It verifies the
embedded bundle's SHA-256 inventory and semantic contract, compiles the runtime, checks the
probability simplex and contrastive explanation, then proves that featureless input, oversized
input and the narrow safety-stop path do not expose a learned decision. This is an operational
check, not another evaluation set, and none of its fixed probes enter training or policy
selection.

Inspect one learned prediction:

```bash
cargo run --locked -- infer --json "Today I feel calm"
```

Start an uncertainty-aware local session:

```bash
cargo run --locked -- chat
```

Rebuild a v3 bundle from the embedded synthetic fixtures:

```bash
cargo run --locked -- train-v3 --output target/open-set-v3
cargo run --locked -- bundle verify --bundle target/open-set-v3
```

The CLI refuses collisions between inputs and outputs. Legacy model inference is deliberately
separate from the default path:

```bash
cargo run --locked -- infer --legacy-v1 --model models/eliza-intent-v1.json --json "Today I feel calm"
```

## Honest evaluation boundary

The corpus, OOD fixtures and split are synthetic by design. The exact released values live in the
cryptographically linked [`metrics.json`](artifacts/eliza-open-set-v3/metrics.json), and the browser
recomputes their ledgers before showing them. Read the [model card](docs/MODEL_CARD.md) before
interpreting accuracy, calibration or OOD discrimination.

An earlier prerelease run was discarded after review found partition-correlated family sizes and
singleton OOD families. The current protocol equalizes family support before splitting, keeps
broader OOD domains disjoint, reports per-stratum metrics and compares the learned model with a
training-only unigram baseline. These controls make the experiment more credible; they do not make
it a production language model.

## Verify it

```bash
cargo fmt --check
cargo clippy --all-targets --locked -- -D warnings
cargo test --all --locked
cargo run --locked -- bundle verify --bundle artifacts/eliza-open-set-v3
cargo run --locked -- bundle reproduce --bundle artifacts/eliza-open-set-v3
cargo run --release --locked -- selection audit \
  --output target/selection-stability-v1.json
printf '%s\n' '{"id":"fictional-01","text":"I intend to test one next step"}' \
  | cargo run --locked -- robustness audit
cargo run --locked -- robustness audit --bundle-id-test \
  > target/robustness-id-test-report.json
node scripts/verify-robustness-report.mjs target/robustness-id-test-report.json
node --test site/tests/*.test.mjs
node site/scripts/validate-site.mjs
node --test scripts/tests/*.test.mjs
node scripts/release-contract.mjs verify
node scripts/release-contract.mjs audit-policy
```

## Release verification

The release workflow exercises the same contract on pull requests and manual runs before any tag
exists. It builds and smoke-tests locked Rust binaries for Linux x64, Windows x64, macOS Intel, and
macOS Apple Silicon. Unix binaries ship in `.tar.gz` archives that preserve their executable bit;
Windows ships as `.zip`. The workflow checks every archive name, SHA-256 digest, source commit,
dependency record, SPDX 2.3 entry, and RustSec result as one release set. The RustSec policy pins
both `cargo-audit` and the advisory database commit, denies every warning, and allows no ignored
advisories. Pull requests and manual runs cannot request OIDC attestations or publish a GitHub
Release, even when a manual run starts from a tag ref.

Workflow policy tests parse CI, Pages, and release YAML as explicit structures. They reject anchors,
aliases, tags, duplicate keys, hidden permission overrides, local actions, and any remote Action that
is not pinned to a full commit SHA.

GitHub release publication is authorized under the MIT License. The versioned policy in
`.github/release-policy.json`, the SPDX expression in `Cargo.toml`, and the repository-root
[`LICENSE`](LICENSE) file must agree before a tag run can call the release API. The crate remains
private to this repository (`publish = false`); the supported distribution channel is the verified
GitHub Release assembled by the workflow.

A release can only be published from a `v*` tag pushed for the version in `Cargo.toml`, with a dated
section for that version in `CHANGELOG.md` and no pending text under `Unreleased`. For example,
version `1.6.0` accepts `v1.6.0` and rejects every other tag. The workflow assembles all four native
archives from verified file-descriptor snapshots, creates a consolidated `SHA256SUMS` file covering
every release asset, and adds GitHub provenance attestations. The publish job independently verifies
each attestation against this repository, workflow, tag ref, and source commit before it can touch a
release.

The first authorization requires the pushed tag to resolve to the current remote default-branch tip.
The publisher then creates an exact contract-bearing draft and waits, for a bounded time, until
GitHub's paginated release listing exposes that same draft. It does not upload assets before the draft
is uniquely visible. If a run stops during upload, a later run finds the draft and resumes it without
rebuilding or guessing which commit it represents. Duplicate drafts or foreign contract metadata
stop the run before mutation. Recovery and immutable reruns use GitHub's compare API to prove that the
exact release commit is still identical to or an ancestor of the current default branch; a divergent
commit is rejected without touching the draft. The release `target_commitish` must be that exact commit.
The manual recovery path also binds the draft ID to one exact receipt in the exact failed publish
job's GitHub Actions log. It does not infer draft causality from `release.created_at`, which is
tag-or-commit metadata rather than draft-creation time. See the
[release recovery contract](docs/RELEASE_RECOVERY.md).
The publisher rechecks the protected tag, branch ancestry, draft contract, and complete remote name,
size, and SHA-256 inventory before promotion. Publication is the irreversible boundary: GitHub must
report the result as both immutable and latest. A rerun only verifies an already-published release and
never rewrites it.

The repository protects `refs/tags/v*` against updates and deletions with a GitHub ruleset and has
immutable releases enabled. The workflow revalidates the tag and requires GitHub to mark the final
release immutable.
Prerelease and build-metadata versions remain rejected until a separate prerelease policy exists.

The RustSec database commit and its commit time are pinned together. CI enforces a 14-day freshness
window on every change and once a week, so a reproducible audit cannot quietly become an outdated
audit. Updating the pin requires reviewing the new official RustSec commit and recording its commit
epoch in `.github/rustsec-audit-policy.json`.

Verify files from `v1.5.0` and earlier with:

```bash
sha256sum -c SHA256SUMS --ignore-missing
gh attestation verify <downloaded-file> -R ejupi-djenis30/PsychologistRustBot
```

For releases created after the repository rename, use
`gh attestation verify <downloaded-file> -R ejupi-djenis30/eliza-lab`.

To test a proposed tag without creating one, start the **Release** workflow manually and provide
the tag in `release_tag`, or run:

```bash
node scripts/release-contract.mjs verify --tag v1.6.0
```

## Architecture

```text
src/open_set.rs             v3 data contracts, typed splits, training, evaluation, bundles, inference
src/open_set/selection_audit.rs typed pre-test pool, nested group CV, OOF ledger and intervals
src/robustness.rs           bounded aggregate-only metamorphic robustness audit and release gates
src/diagnostics.rs          prompt-free embedded bundle and inference self-test
src/ml.rs                   explicit legacy-v1 compatibility implementation
src/lib.rs                  hard boundaries and dialogue routing
src/main.rs                 train / selection audit / evaluate / infer / chat CLI
fixtures/                   supervised, OOD, contrast, and parity corpora
models/                     versioned learned artifact
reports/                    generated split, calibration, metrics, and selection-stability evidence
artifacts/eliza-open-set-v3 SHA-256-linked model, policy, metrics, and split plan
site/open-set-engine.mjs    trust-root verification and browser v3 inference
site/selection-audit.mjs    pinned digest and semantic reconstruction of the OOF report
docs/                       model card, dataset contract, architecture
```

Rust and JavaScript run the same versioned model against
[`fixtures/ml-parity.tsv`](fixtures/ml-parity.tsv). The original rule fallback retains its separate
[`fixtures/parity.tsv`](fixtures/parity.tsv) contract. See [the architecture document](docs/ARCHITECTURE.md)
for data flow and failure behavior.

The safety phrase list is only an exit condition for the experiment. It can miss urgent language
and can match benign discussion of a phrase. Never use this project to assess a person or decide
whether help is needed. See [the safety and privacy model](SECURITY.md).

## Provenance

ELIZA Lab grew out of a small 2023 experiment. Ejupi Labs and the project contributors rebuilt it
around a safer premise, removing Telegram, MySQL, OpenAI, and the PHP administration panel instead
of preserving unsafe behaviour for compatibility.

## License

ELIZA Lab is available under the [MIT License](LICENSE).
