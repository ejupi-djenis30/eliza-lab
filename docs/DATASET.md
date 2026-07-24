# Dataset contract

ELIZA Lab v3 uses four synthetic, English-only fixtures. They are designed to exercise the
pipeline without collecting conversations or importing personal data. They are not evidence of
population-level language understanding.

## Supervised corpus

[`fixtures/intents-v3.tsv`](../fixtures/intents-v3.tsv) has this exact UTF-8 TSV schema:

```text
id<TAB>group_id<TAB>label<TAB>text
```

The frozen corpus contains 525 prompts: seven labels, fifteen paraphrase families per label and
five prompts per family. A family belongs to exactly one label. Family support must be uniform and
at least three prompts; this prevents the row count itself from revealing which families will land
in training or evaluation.

| Label | Meaning inside this dataset |
| --- | --- |
| `greeting` | An opening or salutation |
| `feeling` | A direct statement of an emotional state |
| `reason` | An explanation, cause or consequence |
| `ownership` | A statement about the speaker's own object, plan or context |
| `question` | A direct request for an answer or explanation |
| `goal` | An intention, objective or next step |
| `observation` | A descriptive statement without another listed intent |

IDs, normalized texts and model-feature identities must be unique. The parser also rejects empty
fields, unknown labels, oversized text, groups that cross labels and datasets that cannot populate
all four partitions.

### Near-duplicate gate

The audit uses the exact feature families consumed by the model: normalized word uni- and bigrams
plus character 3-, 4- and 5-grams. It rejects:

- any cross-family pair with raw feature Jaccard similarity greater than or equal to `0.30`;
- any same-label cross-family pair with residual Jaccard similarity greater than or equal to
  `0.25`, after removing features present in at least three families of that label.

The search is prefix-indexed. It does not enumerate every possible pair, and it fails closed after
500,000 candidate pairs or 5,000,000 posting comparisons. Tests cover both the rejection rule and
the workload boundary.

## Four-way supervised split

The split hashes whole families within each label. For fifteen families per label it assigns nine
families to training and two each to development, calibration and ID-test.

| Partition | Families | Rows | May influence |
| --- | ---: | ---: | --- |
| Train | 63 | 315 | Vocabulary, IDF and classifier weights |
| Development | 14 | 70 | Candidate selection and abstention thresholds |
| Calibration | 14 | 70 | Temperature scaling only |
| ID-test | 14 | 70 | Final in-domain metrics only |

Every label appears in every partition. No family crosses a partition. The seed and every row
assignment are stored in `split-plan.json`; the plan's SHA-256 is linked from the model, policy,
metrics and bundle manifest.

## Selection-audit pool

The `selection audit` command reconstructs a capability from only the train and development
assignments:

| Audit population | Families | Rows | Use |
| --- | ---: | ---: | --- |
| Train + development | 77 | 385 | Nested candidate-stability audit |
| Calibration + ID-test | 28 | 140 | Parsed to reproduce the split, then discarded |
| Both OOD populations + contrast test | 24 OOD families + 14 pairs | 100 | Not audit inputs |

There are eleven available families per label. The outer loop uses eleven folds, each holding out
one family per label. The ten remaining families per label divide evenly across five inner folds.
Each inner holdout therefore contains fourteen families and 70 rows; each outer holdout contains
seven families and 35 rows.

The pool fingerprint covers only the 385 included rows. Its first 52 bits derive the JSON-safe audit
seed. The report records outer assignments, every inner-candidate metric, the candidate chosen per
outer fold and one out-of-fold probability vector per row. A label-stratified bootstrap resamples
whole families, not paraphrases. Altering a calibration, final-test, OOD or contrast row cannot
become an audit input.

## OOD populations

[`fixtures/ood-dev-v3.tsv`](../fixtures/ood-dev-v3.tsv) and
[`fixtures/ood-test-v3.tsv`](../fixtures/ood-test-v3.tsv) use:

```text
id<TAB>family_id<TAB>domain_group<TAB>stratum<TAB>text
```

Each population contains 36 prompts, twelve families, six broader domains and three prompts per
family. The three strata are balanced at twelve rows each:

- `semantic`: coherent requests outside the seven intent definitions;
- `capability`: requests that would require an external side effect or unavailable tool;
- `noise`: corrupted, synthetic or incomplete input.

Each broader domain contains two families and belongs to one stratum. Development and test use
different families and different broader domains. OOD-development may select the abstention
policy. OOD-test is evaluated only after model, temperature and thresholds are frozen.

The bootstrap resamples OOD-test by broader domain, not by sentence. Per-stratum coverage and
discrimination metrics are reported separately so a good aggregate cannot hide one weak category.

## Paired contrast test

[`fixtures/contrast-test-v3.tsv`](../fixtures/contrast-test-v3.tsv) uses:

```text
id<TAB>pair_id<TAB>variant<TAB>label<TAB>text
```

It contains 28 prompts in fourteen two-row pairs. Each pair has one `a` and one `b` variant with
different intent labels. The surrounding wording is deliberately similar while the decisive
meaning changes. All seven labels appear exactly four times. This makes a separate anti-shortcut
check: the report measures row accuracy, macro F1, pair accuracy, prediction-flip rate and
abstention coverage.

The contrast test is never passed to model selection, fitting, calibration or threshold selection.
It is opened only after the model and policy are frozen. It is still a small, synthetic,
source-authored test rather than an external benchmark.

## Editing and release discipline

A fixture edit invalidates its SHA-256, split, weights and metrics. Freeze the supervised, both OOD
fixtures and the contrast test first, record their hashes, derive a fresh deterministic seed, and
then run the final evaluation once. Do not rewrite prompts, choose a new seed or tune the model
after inspecting the frozen ID-test, OOD-test or contrast-test results.

### Final v3 freeze

The final source fixtures were frozen before the release evaluation with these raw-file and
canonical model-row SHA-256 digests:

| Fixture | Raw file SHA-256 | Canonical row fingerprint |
| --- | --- | --- |
| `intents-v3.tsv` | `6c025bd19fe196273107f82c60a1d59efec939c6ab0e2b7fefe45a1243dae82a` | `efc3d4a3f38d8bf3f81026b72bbd5394de779e654d793bae638fed48485f269c` |
| `ood-dev-v3.tsv` | `dbc6525969a9a4512c75324a24a67db4d0618b061114114ab0e04c5f7c744b9a` | `71d012f7ee16acb666bcc333473148705c8554c839042ad604744e234b7614a9` |
| `ood-test-v3.tsv` | `8ff19c105423c08ff81fa702874ada69824d8798d7b786db5dcec1f30d36743a` | `ff9cea872481cf69944fa98406727a1c4c09120d1e0a38fbab6d2150a734a00c` |
| `contrast-test-v3.tsv` | `90c413ab970e5f1d9e0d2ea1c85f2a542a1bcd2319a4fde4ce97b9fc745d8a56` | `8f4427267cdc4aa58ff3a2e64968f0c6f1784edb98e5e81cbfb6f250fdb4fa3f` |

The seed is derived mechanically. In the table order above, concatenate the domain separator
`eliza-open-set-v3-final-seed`, a newline, each lowercase raw digest followed by a newline, hash
that UTF-8 string with SHA-256, interpret the first eight digest bytes as an unsigned big-endian
integer, and reduce it modulo `2^53`. The seed-material digest is
`a70e5d2d9fa2c31386d1adaa2a2bb9163f8641f740ab719928bdd8a7d8e1aae3`; the resulting experiment
seed is `4043100207104787`. It was fixed without observing final ID, OOD or contrast metrics.
