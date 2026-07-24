use super::*;
use std::path::Path;

pub const DEFAULT_SELECTION_AUDIT_BOOTSTRAP_RESAMPLES: usize = 1_000;
const SELECTION_AUDIT_SCHEMA_VERSION: u32 = 1;
const SELECTION_AUDIT_KIND: &str = "eliza-selection-stability-audit";
const SELECTION_AUDIT_ALGORITHM: &str = "nested-group-stratified-selection-stability-v1";
const INNER_FOLDS: usize = 5;
const MAX_AUDIT_MODEL_FITS: usize = 1_024;
const MAX_AUDIT_BOOTSTRAP_ROWS: usize = 5_000_000;
const BOOTSTRAP_SEED_DOMAIN: u64 = 0xbb67_ae85_84ca_a73b;
const JSON_SAFE_INTEGER_MASK: u64 = (1_u64 << 53) - 1;

/// A capability containing only the train and development families used for model selection.
///
/// The constructor rebuilds the supervised four-way split, then discards calibration and ID-test.
/// OOD and contrast examples are never constructor inputs. Once built, this type is the only data
/// capability accepted by the fit, selection and scoring audit.
#[derive(Debug, Clone)]
pub struct SelectionAuditPool {
    examples: Vec<GroupedExample>,
    labels: Vec<String>,
    families_by_label: BTreeMap<String, Vec<String>>,
    pool_sha256: String,
    partition_seed: u64,
    audit_seed: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct SelectionAuditConfig {
    pub training: OpenSetTrainingConfig,
    pub bootstrap_resamples: usize,
}

impl Default for SelectionAuditConfig {
    fn default() -> Self {
        Self {
            training: OpenSetTrainingConfig::default(),
            bootstrap_resamples: DEFAULT_SELECTION_AUDIT_BOOTSTRAP_RESAMPLES,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct SelectionAuditPoolSummary {
    pub pool_sha256: String,
    pub partition_seed: u64,
    pub audit_seed: u64,
    pub seed_derivation: String,
    pub example_count: usize,
    pub family_count: usize,
    pub label_count: usize,
    pub families_per_label: usize,
    pub examples_per_family: usize,
    pub included_roles: Vec<String>,
    pub structurally_unavailable_roles: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct SelectionAuditCandidate {
    pub candidate_id: String,
    pub max_features: usize,
    pub l2_penalty: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct SelectionAuditTraining {
    pub optimizer: String,
    pub seed: u64,
    pub epochs: usize,
    pub learning_rate: f64,
    pub word_ngram_min: usize,
    pub word_ngram_max: usize,
    pub char_ngram_min: usize,
    pub char_ngram_max: usize,
    pub min_document_frequency: usize,
    pub macro_f1_tolerance: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct SelectionAuditCandidateResult {
    pub candidate_id: String,
    pub rank: usize,
    pub accuracy: f64,
    pub macro_f1: f64,
    pub negative_log_likelihood: f64,
    pub multiclass_brier: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct SelectionAuditFold {
    pub fold: usize,
    pub outer_holdout_families: Vec<String>,
    pub outer_training_examples: usize,
    pub outer_holdout_examples: usize,
    pub inner_folds: usize,
    pub inner_holdout_families: Vec<Vec<String>>,
    pub inner_candidates: Vec<SelectionAuditCandidateResult>,
    pub selected_candidate_id: String,
    pub outer_accuracy: f64,
    pub outer_macro_f1: f64,
    pub outer_negative_log_likelihood: f64,
    pub outer_multiclass_brier: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct SelectionAuditCandidateStability {
    pub candidate_id: String,
    pub selected_folds: usize,
    pub selection_rate: f64,
    pub mean_rank: f64,
    pub rank_standard_deviation: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct SelectionAuditLedgerRow {
    pub id: String,
    pub family_id: String,
    pub actual_label: String,
    pub predicted_label: String,
    pub correct: bool,
    pub probabilities: BTreeMap<String, f64>,
    pub outer_fold: usize,
    pub selected_candidate_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct SelectionAuditMetrics {
    pub example_count: usize,
    pub accuracy: f64,
    pub macro_f1: f64,
    pub negative_log_likelihood: f64,
    pub multiclass_brier: f64,
    pub labels: Vec<String>,
    pub confusion_matrix: Vec<Vec<usize>>,
    pub per_class: Vec<PerClassMetrics>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct SelectionAuditBootstrap {
    pub strategy: String,
    pub seed: u64,
    pub resamples: usize,
    pub confidence_level: f64,
    pub accuracy: MetricEstimate,
    pub macro_f1: MetricEstimate,
    pub negative_log_likelihood: MetricEstimate,
    pub multiclass_brier: MetricEstimate,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct SelectionAuditReport {
    pub schema_version: u32,
    pub audit_kind: String,
    pub algorithm: String,
    pub pool: SelectionAuditPoolSummary,
    pub training: SelectionAuditTraining,
    pub grid: Vec<SelectionAuditCandidate>,
    pub outer_folds: usize,
    pub inner_folds: usize,
    pub model_fits: usize,
    pub folds: Vec<SelectionAuditFold>,
    pub candidate_stability: Vec<SelectionAuditCandidateStability>,
    pub metrics: SelectionAuditMetrics,
    pub bootstrap_95: SelectionAuditBootstrap,
    pub ledger: Vec<SelectionAuditLedgerRow>,
    pub limitations: Vec<String>,
}

impl SelectionAuditPool {
    pub fn from_dataset(dataset: &GroupedDataset, partition_seed: u64) -> Result<Self, MlError> {
        if partition_seed > 9_007_199_254_740_991 {
            return Err(MlError::InvalidConfiguration(
                "the partition seed exceeds the JSON-safe integer ceiling".into(),
            ));
        }
        dataset.validate_partition_support()?;
        let mut grouped: BTreeMap<String, BTreeMap<String, Vec<GroupedExample>>> = BTreeMap::new();
        for example in dataset.examples() {
            grouped
                .entry(example.label.clone())
                .or_default()
                .entry(example.group_id.clone())
                .or_default()
                .push(example.clone());
        }

        let mut examples = Vec::new();
        let mut families_by_label = BTreeMap::new();
        for (label, groups) in grouped {
            let mut groups = groups.into_iter().collect::<Vec<_>>();
            groups.sort_by(|(left, _), (right, _)| {
                group_split_hash(&label, left, partition_seed)
                    .cmp(&group_split_hash(&label, right, partition_seed))
                    .then_with(|| left.cmp(right))
            });
            let evaluation_groups = evaluation_group_quota(groups.len());
            let selection_start = evaluation_groups.checked_mul(2).ok_or_else(|| {
                MlError::InvalidConfiguration("selection-pool boundary overflowed".into())
            })?;
            let mut selected_families = Vec::new();
            for (index, (family_id, mut family_examples)) in groups.into_iter().enumerate() {
                if index < selection_start {
                    continue;
                }
                family_examples.sort_by(|left, right| left.id.cmp(&right.id));
                selected_families.push(family_id);
                examples.extend(family_examples);
            }
            selected_families.sort();
            families_by_label.insert(label, selected_families);
        }
        examples.sort_by(|left, right| left.id.cmp(&right.id));
        let labels = families_by_label.keys().cloned().collect::<Vec<_>>();
        let family_counts = families_by_label
            .values()
            .map(Vec::len)
            .collect::<BTreeSet<_>>();
        if labels.len() < 2
            || family_counts.len() != 1
            || family_counts.first().copied().unwrap_or_default() < 2
        {
            return Err(MlError::InvalidDataset(
                "the selection pool must contain the same number of families for every label"
                    .into(),
            ));
        }
        let family_sizes = examples
            .iter()
            .fold(BTreeMap::<&str, usize>::new(), |mut counts, example| {
                *counts.entry(&example.group_id).or_default() += 1;
                counts
            })
            .into_values()
            .collect::<BTreeSet<_>>();
        if family_sizes.len() != 1 || family_sizes.first().copied().unwrap_or_default() == 0 {
            return Err(MlError::InvalidDataset(
                "selection-pool families must have a uniform, non-zero size".into(),
            ));
        }
        let pool_sha256 = selection_pool_fingerprint(&examples);
        let audit_seed = u64::from_str_radix(&pool_sha256[..13], 16).map_err(|_| {
            MlError::InvalidDataset("selection-pool fingerprint cannot derive a seed".into())
        })?;
        Ok(Self {
            examples,
            labels,
            families_by_label,
            pool_sha256,
            partition_seed,
            audit_seed,
        })
    }

    pub fn examples(&self) -> &[GroupedExample] {
        &self.examples
    }

    pub fn pool_sha256(&self) -> &str {
        &self.pool_sha256
    }

    pub fn audit_seed(&self) -> u64 {
        self.audit_seed
    }

    fn summary(&self) -> SelectionAuditPoolSummary {
        let family_count = self
            .examples
            .iter()
            .map(|example| example.group_id.as_str())
            .collect::<HashSet<_>>()
            .len();
        SelectionAuditPoolSummary {
            pool_sha256: self.pool_sha256.clone(),
            partition_seed: self.partition_seed,
            audit_seed: self.audit_seed,
            seed_derivation: "first-52-bits-of-selection-pool-sha256".into(),
            example_count: self.examples.len(),
            family_count,
            label_count: self.labels.len(),
            families_per_label: self.families_by_label.values().next().map_or(0, Vec::len),
            examples_per_family: self.examples.len() / family_count,
            included_roles: vec!["train".into(), "development".into()],
            structurally_unavailable_roles: vec![
                "calibration".into(),
                "id-test".into(),
                "ood-development".into(),
                "ood-test".into(),
                "contrast-test".into(),
            ],
        }
    }
}

impl SelectionAuditConfig {
    fn validate(&self, pool: &SelectionAuditPool) -> Result<(), MlError> {
        self.training.validate()?;
        if !(100..=20_000).contains(&self.bootstrap_resamples) {
            return Err(MlError::InvalidConfiguration(
                "selection-audit bootstrap resamples must be between 100 and 20000".into(),
            ));
        }
        let families_per_label = pool.families_by_label.values().next().map_or(0, Vec::len);
        if families_per_label < INNER_FOLDS + 1 {
            return Err(MlError::InvalidDataset(format!(
                "selection audit requires at least {} families per label",
                INNER_FOLDS + 1
            )));
        }
        let candidate_count = self
            .training
            .development_selection
            .max_features_candidates
            .len()
            .checked_mul(
                self.training
                    .development_selection
                    .l2_penalty_candidates
                    .len(),
            )
            .ok_or_else(|| {
                MlError::InvalidConfiguration("selection-audit grid size overflowed".into())
            })?;
        let candidate_ids = candidate_grid(&self.training)
            .into_iter()
            .map(|candidate| candidate.candidate_id)
            .collect::<HashSet<_>>();
        if candidate_ids.len() != candidate_count {
            return Err(MlError::InvalidConfiguration(
                "selection-audit candidate IDs collide at report precision".into(),
            ));
        }
        let fits = families_per_label
            .checked_mul(candidate_count * INNER_FOLDS + 1)
            .ok_or_else(|| {
                MlError::InvalidConfiguration("selection-audit workload overflowed".into())
            })?;
        if fits > MAX_AUDIT_MODEL_FITS {
            return Err(MlError::InvalidConfiguration(format!(
                "selection audit requires {fits} model fits; limit is {MAX_AUDIT_MODEL_FITS}"
            )));
        }
        let sampled_rows = pool
            .examples
            .len()
            .checked_mul(self.bootstrap_resamples)
            .ok_or_else(|| {
                MlError::InvalidConfiguration(
                    "selection-audit bootstrap workload overflowed".into(),
                )
            })?;
        if sampled_rows > MAX_AUDIT_BOOTSTRAP_ROWS {
            return Err(MlError::InvalidConfiguration(format!(
                "selection-audit bootstrap exceeds {MAX_AUDIT_BOOTSTRAP_ROWS} sampled rows"
            )));
        }
        Ok(())
    }
}

fn selection_pool_fingerprint(examples: &[GroupedExample]) -> String {
    let mut rows = examples
        .iter()
        .map(|example| {
            format!(
                "{}\t{}\t{}\t{}",
                example.id,
                example.group_id,
                example.label,
                feature_identity(&example.text)
            )
        })
        .collect::<Vec<_>>();
    rows.sort();
    sha256_hex(rows.join("\n").as_bytes())
}

fn candidate_grid(config: &OpenSetTrainingConfig) -> Vec<SelectionAuditCandidate> {
    let mut grid = Vec::new();
    for max_features in &config.development_selection.max_features_candidates {
        for l2_penalty in &config.development_selection.l2_penalty_candidates {
            grid.push(SelectionAuditCandidate {
                candidate_id: candidate_id(*max_features, *l2_penalty),
                max_features: *max_features,
                l2_penalty: *l2_penalty,
            });
        }
    }
    grid
}

fn training_summary(config: &OpenSetTrainingConfig, audit_seed: u64) -> SelectionAuditTraining {
    SelectionAuditTraining {
        optimizer: "deterministic-full-batch-multinomial-logistic-regression".into(),
        seed: audit_seed,
        epochs: config.epochs,
        learning_rate: config.learning_rate,
        word_ngram_min: config.vectorizer.word_ngram_min,
        word_ngram_max: config.vectorizer.word_ngram_max,
        char_ngram_min: config.vectorizer.char_ngram_min,
        char_ngram_max: config.vectorizer.char_ngram_max,
        min_document_frequency: config.vectorizer.min_document_frequency,
        macro_f1_tolerance: config.development_selection.macro_f1_tolerance,
    }
}

fn candidate_id(max_features: usize, l2_penalty: f64) -> String {
    format!("features-{max_features}-l2-{l2_penalty:.6}")
}

fn examples_for_families(
    pool: &SelectionAuditPool,
    families: &HashSet<&str>,
    include: bool,
) -> Vec<GroupedExample> {
    let mut examples = pool
        .examples
        .iter()
        .filter(|example| families.contains(example.group_id.as_str()) == include)
        .cloned()
        .collect::<Vec<_>>();
    examples.sort_by(|left, right| left.id.cmp(&right.id));
    examples
}

fn outer_holdouts(pool: &SelectionAuditPool) -> Vec<Vec<String>> {
    let fold_count = pool.families_by_label.values().next().map_or(0, Vec::len);
    let mut ordered_by_label = BTreeMap::new();
    for (label, families) in &pool.families_by_label {
        let mut families = families.clone();
        families.sort_by(|left, right| {
            group_split_hash(label, left, pool.audit_seed)
                .cmp(&group_split_hash(label, right, pool.audit_seed))
                .then_with(|| left.cmp(right))
        });
        ordered_by_label.insert(label, families);
    }
    (0..fold_count)
        .map(|fold| {
            let mut holdout = ordered_by_label
                .values()
                .map(|families| families[fold].clone())
                .collect::<Vec<_>>();
            holdout.sort();
            holdout
        })
        .collect()
}

fn inner_holdouts(
    pool: &SelectionAuditPool,
    outer_families: &HashSet<&str>,
    outer_fold: usize,
) -> Vec<Vec<String>> {
    let mut folds = vec![Vec::new(); INNER_FOLDS];
    for (label, families) in &pool.families_by_label {
        let mut remaining = families
            .iter()
            .filter(|family| !outer_families.contains(family.as_str()))
            .cloned()
            .collect::<Vec<_>>();
        let fold_seed = pool.audit_seed ^ ((outer_fold as u64 + 1) * 0x9e37_79b9);
        remaining.sort_by(|left, right| {
            group_split_hash(label, left, fold_seed)
                .cmp(&group_split_hash(label, right, fold_seed))
                .then_with(|| left.cmp(right))
        });
        for (index, family) in remaining.into_iter().enumerate() {
            folds[index % INNER_FOLDS].push(family);
        }
    }
    for fold in &mut folds {
        fold.sort();
    }
    folds
}

fn training_partition<'a>(
    examples: &'a [GroupedExample],
    pool_sha256: &'a str,
    plan_sha256: String,
) -> TrainingPartition<'a> {
    TrainingPartition {
        examples,
        dataset_sha256: pool_sha256,
        split_plan_sha256: plan_sha256,
    }
}

fn candidate_training_config(
    base: &OpenSetTrainingConfig,
    candidate: &SelectionAuditCandidate,
    audit_seed: u64,
) -> OpenSetTrainingConfig {
    let mut config = base.clone();
    config.seed = audit_seed;
    config.vectorizer.max_features = candidate.max_features;
    config.l2_penalty = candidate.l2_penalty;
    config
}

fn evaluate_candidate_inner(
    pool: &SelectionAuditPool,
    outer_families: &HashSet<&str>,
    inner_folds: &[Vec<String>],
    candidate: &SelectionAuditCandidate,
    config: &SelectionAuditConfig,
    plan_sha256: &str,
) -> Result<DevelopmentCandidateMetrics, MlError> {
    let mut predictions = Vec::new();
    for inner_holdout in inner_folds {
        let mut excluded = outer_families.clone();
        excluded.extend(inner_holdout.iter().map(String::as_str));
        let train = examples_for_families(pool, &excluded, false);
        let holdout = examples_for_families(
            pool,
            &inner_holdout.iter().map(String::as_str).collect(),
            true,
        );
        let partition = training_partition(&train, &pool.pool_sha256, plan_sha256.to_owned());
        let model = fit_model(
            &partition,
            candidate_training_config(&config.training, candidate, pool.audit_seed),
        )?;
        predictions.extend(
            evaluate_development_candidate(&model, DevelopmentPartition(&holdout))?.predictions,
        );
    }
    predictions.sort_by(|left, right| left.id.cmp(&right.id));
    let summary = summarize_id_predictions(&pool.labels, predictions);
    Ok(DevelopmentCandidateMetrics {
        max_features: candidate.max_features,
        l2_penalty: candidate.l2_penalty,
        accuracy: summary.accuracy,
        macro_f1: summary.macro_f1,
        negative_log_likelihood: summary.calibration.negative_log_likelihood,
        multiclass_brier: summary.calibration.multiclass_brier,
    })
}

fn rank_candidates(metrics: &[DevelopmentCandidateMetrics]) -> Vec<usize> {
    let mut indices = (0..metrics.len()).collect::<Vec<_>>();
    indices.sort_by(|left, right| {
        metrics[*right]
            .macro_f1
            .total_cmp(&metrics[*left].macro_f1)
            .then_with(|| {
                metrics[*left]
                    .max_features
                    .cmp(&metrics[*right].max_features)
            })
            .then_with(|| {
                metrics[*right]
                    .l2_penalty
                    .total_cmp(&metrics[*left].l2_penalty)
            })
            .then_with(|| metrics[*right].accuracy.total_cmp(&metrics[*left].accuracy))
            .then_with(|| {
                metrics[*left]
                    .negative_log_likelihood
                    .total_cmp(&metrics[*right].negative_log_likelihood)
            })
            .then_with(|| {
                metrics[*left]
                    .multiclass_brier
                    .total_cmp(&metrics[*right].multiclass_brier)
            })
            .then_with(|| left.cmp(right))
    });
    let mut ranks = vec![0; metrics.len()];
    for (rank, index) in indices.into_iter().enumerate() {
        ranks[index] = rank + 1;
    }
    ranks
}

fn select_candidate(metrics: &[DevelopmentCandidateMetrics], tolerance: f64) -> usize {
    let mut selected = 0;
    for candidate in 1..metrics.len() {
        if development_candidate_is_better(&metrics[candidate], &metrics[selected], tolerance) {
            selected = candidate;
        }
    }
    selected
}

fn report_metrics(summary: &IdEvaluationV3) -> SelectionAuditMetrics {
    SelectionAuditMetrics {
        example_count: summary.example_count,
        accuracy: quantize(summary.accuracy),
        macro_f1: quantize(summary.macro_f1),
        negative_log_likelihood: quantize(summary.calibration.negative_log_likelihood),
        multiclass_brier: quantize(summary.calibration.multiclass_brier),
        labels: summary.labels.clone(),
        confusion_matrix: summary.confusion_matrix.clone(),
        per_class: summary.per_class.clone(),
    }
}

fn selection_bootstrap(
    pool: &SelectionAuditPool,
    summary: &IdEvaluationV3,
    resamples: usize,
) -> Result<SelectionAuditBootstrap, MlError> {
    let family_by_id = pool
        .examples
        .iter()
        .map(|example| (example.id.as_str(), example.group_id.as_str()))
        .collect::<HashMap<_, _>>();
    let mut by_label_family: BTreeMap<&str, BTreeMap<&str, Vec<&EvaluatedOpenSetPrediction>>> =
        BTreeMap::new();
    for prediction in &summary.predictions {
        let family = family_by_id.get(prediction.id.as_str()).ok_or_else(|| {
            MlError::InvalidDataset("an OOF prediction is missing its selection-pool family".into())
        })?;
        by_label_family
            .entry(&prediction.actual_label)
            .or_default()
            .entry(*family)
            .or_default()
            .push(prediction);
    }
    if by_label_family.len() != pool.labels.len() {
        return Err(MlError::InvalidDataset(
            "selection bootstrap requires every label".into(),
        ));
    }
    let bootstrap_seed = (pool.audit_seed ^ BOOTSTRAP_SEED_DOMAIN) & JSON_SAFE_INTEGER_MASK;
    let mut rng = DeterministicRng::new(bootstrap_seed);
    let mut accuracies = Vec::with_capacity(resamples);
    let mut macro_f1s = Vec::with_capacity(resamples);
    let mut nlls = Vec::with_capacity(resamples);
    let mut briers = Vec::with_capacity(resamples);
    for _ in 0..resamples {
        let mut sampled = Vec::with_capacity(pool.examples.len());
        for families in by_label_family.values() {
            let families = families.values().collect::<Vec<_>>();
            for _ in 0..families.len() {
                let family = families[rng.index(families.len())];
                sampled.extend(family.iter().map(|prediction| (*prediction).clone()));
            }
        }
        let sample_summary = summarize_id_predictions(&pool.labels, sampled);
        accuracies.push(sample_summary.accuracy);
        macro_f1s.push(sample_summary.macro_f1);
        nlls.push(sample_summary.calibration.negative_log_likelihood);
        briers.push(sample_summary.calibration.multiclass_brier);
    }
    Ok(SelectionAuditBootstrap {
        strategy: "label-stratified-family-cluster-percentile-v1".into(),
        seed: bootstrap_seed,
        resamples,
        confidence_level: 0.95,
        accuracy: estimate(summary.accuracy, accuracies),
        macro_f1: estimate(summary.macro_f1, macro_f1s),
        negative_log_likelihood: estimate(summary.calibration.negative_log_likelihood, nlls),
        multiclass_brier: estimate(summary.calibration.multiclass_brier, briers),
    })
}

fn validate_report(report: &SelectionAuditReport) -> Result<(), MlError> {
    if report.schema_version != SELECTION_AUDIT_SCHEMA_VERSION
        || report.audit_kind != SELECTION_AUDIT_KIND
        || report.algorithm != SELECTION_AUDIT_ALGORITHM
        || report.outer_folds != report.pool.families_per_label
        || report.inner_folds != INNER_FOLDS
        || report.folds.len() != report.outer_folds
        || report.ledger.len() != report.pool.example_count
        || report.metrics.example_count != report.pool.example_count
        || report.training.seed != report.pool.audit_seed
        || report.training.optimizer != "deterministic-full-batch-multinomial-logistic-regression"
    {
        return Err(MlError::InvalidModel(
            "selection-audit report violates its identity or count contract".into(),
        ));
    }
    let ids = report
        .ledger
        .iter()
        .map(|row| row.id.as_str())
        .collect::<HashSet<_>>();
    let families = report
        .ledger
        .iter()
        .map(|row| row.family_id.as_str())
        .collect::<HashSet<_>>();
    if ids.len() != report.ledger.len()
        || families.len() != report.pool.family_count
        || report.folds.iter().any(|fold| {
            fold.inner_candidates.len() != report.grid.len()
                || fold.inner_holdout_families.len() != INNER_FOLDS
        })
        || report
            .candidate_stability
            .iter()
            .map(|candidate| candidate.selected_folds)
            .sum::<usize>()
            != report.outer_folds
    {
        return Err(MlError::InvalidModel(
            "selection-audit ledger, grid or selection counts are inconsistent".into(),
        ));
    }
    Ok(())
}

pub fn run_selection_stability_audit(
    pool: &SelectionAuditPool,
    config: SelectionAuditConfig,
) -> Result<SelectionAuditReport, MlError> {
    config.validate(pool)?;
    let grid = candidate_grid(&config.training);
    let holdouts = outer_holdouts(pool);
    let model_fits = holdouts.len() * (grid.len() * INNER_FOLDS + 1);
    let plan_sha256 = sha256_hex(
        format!(
            "{SELECTION_AUDIT_ALGORITHM}\n{}\n{}",
            pool.pool_sha256, pool.audit_seed
        )
        .as_bytes(),
    );
    let family_by_id = pool
        .examples
        .iter()
        .map(|example| (example.id.as_str(), example.group_id.as_str()))
        .collect::<HashMap<_, _>>();
    let mut all_predictions = Vec::with_capacity(pool.examples.len());
    let mut folds = Vec::with_capacity(holdouts.len());
    let mut selected_counts = vec![0usize; grid.len()];
    let mut ranks_by_candidate = vec![Vec::new(); grid.len()];
    let mut fold_and_candidate_by_id = HashMap::<String, (usize, String)>::new();

    for (outer_index, outer_holdout) in holdouts.iter().enumerate() {
        let outer_families = outer_holdout
            .iter()
            .map(String::as_str)
            .collect::<HashSet<_>>();
        let inner = inner_holdouts(pool, &outer_families, outer_index);
        let mut candidate_metrics = Vec::with_capacity(grid.len());
        for candidate in &grid {
            candidate_metrics.push(evaluate_candidate_inner(
                pool,
                &outer_families,
                &inner,
                candidate,
                &config,
                &plan_sha256,
            )?);
        }
        let ranks = rank_candidates(&candidate_metrics);
        for (index, rank) in ranks.iter().copied().enumerate() {
            ranks_by_candidate[index].push(rank as f64);
        }
        let selected = select_candidate(
            &candidate_metrics,
            config.training.development_selection.macro_f1_tolerance,
        );
        selected_counts[selected] += 1;

        let outer_train = examples_for_families(pool, &outer_families, false);
        let outer_test = examples_for_families(pool, &outer_families, true);
        let partition = training_partition(&outer_train, &pool.pool_sha256, plan_sha256.clone());
        let model = fit_model(
            &partition,
            candidate_training_config(&config.training, &grid[selected], pool.audit_seed),
        )?;
        let evaluation = evaluate_development_candidate(&model, DevelopmentPartition(&outer_test))?;
        for prediction in &evaluation.predictions {
            fold_and_candidate_by_id.insert(
                prediction.id.clone(),
                (outer_index + 1, grid[selected].candidate_id.clone()),
            );
        }
        all_predictions.extend(evaluation.predictions.clone());
        let ranked_candidates = candidate_metrics
            .iter()
            .enumerate()
            .map(|(index, metrics)| SelectionAuditCandidateResult {
                candidate_id: grid[index].candidate_id.clone(),
                rank: ranks[index],
                accuracy: quantize(metrics.accuracy),
                macro_f1: quantize(metrics.macro_f1),
                negative_log_likelihood: quantize(metrics.negative_log_likelihood),
                multiclass_brier: quantize(metrics.multiclass_brier),
            })
            .collect();
        folds.push(SelectionAuditFold {
            fold: outer_index + 1,
            outer_holdout_families: outer_holdout.clone(),
            outer_training_examples: outer_train.len(),
            outer_holdout_examples: outer_test.len(),
            inner_folds: INNER_FOLDS,
            inner_holdout_families: inner,
            inner_candidates: ranked_candidates,
            selected_candidate_id: grid[selected].candidate_id.clone(),
            outer_accuracy: quantize(evaluation.accuracy),
            outer_macro_f1: quantize(evaluation.macro_f1),
            outer_negative_log_likelihood: quantize(evaluation.calibration.negative_log_likelihood),
            outer_multiclass_brier: quantize(evaluation.calibration.multiclass_brier),
        });
    }

    all_predictions.sort_by(|left, right| left.id.cmp(&right.id));
    let summary = summarize_id_predictions(&pool.labels, all_predictions.clone());
    let ledger = all_predictions
        .iter()
        .map(|prediction| {
            let family_id = family_by_id
                .get(prediction.id.as_str())
                .expect("validated pool prediction has a family");
            let (outer_fold, selected_candidate_id) = fold_and_candidate_by_id
                .get(&prediction.id)
                .expect("validated pool prediction has a fold");
            SelectionAuditLedgerRow {
                id: prediction.id.clone(),
                family_id: (*family_id).to_owned(),
                actual_label: prediction.actual_label.clone(),
                predicted_label: prediction.predicted_label.clone(),
                correct: prediction.correct,
                probabilities: prediction.probabilities.clone(),
                outer_fold: *outer_fold,
                selected_candidate_id: selected_candidate_id.clone(),
            }
        })
        .collect::<Vec<_>>();
    let candidate_stability = grid
        .iter()
        .enumerate()
        .map(|(index, candidate)| {
            let ranks = &ranks_by_candidate[index];
            let mean = ranks.iter().sum::<f64>() / ranks.len() as f64;
            let variance =
                ranks.iter().map(|rank| (rank - mean).powi(2)).sum::<f64>() / ranks.len() as f64;
            SelectionAuditCandidateStability {
                candidate_id: candidate.candidate_id.clone(),
                selected_folds: selected_counts[index],
                selection_rate: quantize(selected_counts[index] as f64 / holdouts.len() as f64),
                mean_rank: quantize(mean),
                rank_standard_deviation: quantize(variance.sqrt()),
            }
        })
        .collect::<Vec<_>>();
    let bootstrap_95 = selection_bootstrap(pool, &summary, config.bootstrap_resamples)?;
    let report = SelectionAuditReport {
        schema_version: SELECTION_AUDIT_SCHEMA_VERSION,
        audit_kind: SELECTION_AUDIT_KIND.into(),
        algorithm: SELECTION_AUDIT_ALGORITHM.into(),
        pool: pool.summary(),
        training: training_summary(&config.training, pool.audit_seed),
        grid,
        outer_folds: holdouts.len(),
        inner_folds: INNER_FOLDS,
        model_fits,
        folds,
        candidate_stability,
        metrics: report_metrics(&summary),
        bootstrap_95,
        ledger,
        limitations: vec![
            "The audit uses synthetic, English-only examples; it measures selection stability, not real-world validity.".into(),
            "Nested group cross-validation reduces development reuse but cannot remove bias in the synthetic data design.".into(),
            "Candidate predictions are compared before temperature calibration or abstention-policy selection; these are closed-set label metrics, not decision-coverage estimates.".into(),
            "Pool construction parses the supervised corpus to reproduce the fixed four-way group split, then discards calibration and ID-test; fitting, inner selection and outer scoring receive only train and development. OOD and contrast data are not inputs.".into(),
            "Families, not individual paraphrases, are the independent resampling unit; 77 families still produce uncertain intervals.".into(),
            "Candidate rankings are diagnostic. They do not authorize clinical, safety, employment or other decisions about people.".into(),
        ],
    };
    validate_report(&report)?;
    Ok(report)
}

pub fn selection_audit_report_bytes(report: &SelectionAuditReport) -> Result<Vec<u8>, MlError> {
    validate_report(report)?;
    let mut value = serde_json::to_value(report)?;
    quantize_json_floats(&mut value)?;
    canonical_json(&value)
}

fn quantize_json_floats(value: &mut serde_json::Value) -> Result<(), MlError> {
    match value {
        serde_json::Value::Number(number) if number.is_f64() => {
            let raw = number.as_f64().ok_or_else(|| {
                MlError::InvalidModel("selection-audit metric is not representable as f64".into())
            })?;
            let quantized = (raw * METRIC_REPORTING_SCALE).round() / METRIC_REPORTING_SCALE;
            *number = serde_json::Number::from_f64(quantized).ok_or_else(|| {
                MlError::InvalidModel("selection-audit metric is not a finite JSON number".into())
            })?;
        }
        serde_json::Value::Array(values) => {
            for value in values {
                quantize_json_floats(value)?;
            }
        }
        serde_json::Value::Object(values) => {
            for value in values.values_mut() {
                quantize_json_floats(value)?;
            }
        }
        _ => {}
    }
    Ok(())
}

pub fn write_selection_audit_report(
    path: impl AsRef<Path>,
    report: &SelectionAuditReport,
) -> Result<String, MlError> {
    let bytes = selection_audit_report_bytes(report)?;
    write_atomic(path.as_ref(), &bytes)?;
    let persisted = fs::read(path.as_ref())?;
    if persisted != bytes {
        return Err(MlError::Io(std::io::Error::other(
            "selection-audit report failed persisted-byte verification",
        )));
    }
    Ok(sha256_hex(&bytes))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fast_config() -> SelectionAuditConfig {
        let mut config = SelectionAuditConfig::default();
        config.training.epochs = 1;
        config.bootstrap_resamples = 100;
        config
    }

    #[test]
    fn pool_contains_only_train_and_development_families() {
        let dataset = GroupedDataset::bundled().unwrap();
        let pool =
            SelectionAuditPool::from_dataset(&dataset, OpenSetTrainingConfig::default().seed)
                .unwrap();
        assert_eq!(pool.examples.len(), 385);
        assert_eq!(pool.labels.len(), 7);
        assert!(pool
            .families_by_label
            .values()
            .all(|families| families.len() == 11));
        assert_eq!(
            pool.examples
                .iter()
                .map(|example| example.group_id.as_str())
                .collect::<HashSet<_>>()
                .len(),
            77
        );

        let plan = SplitPlan::build(
            &dataset,
            &OpenSetOodDataset::bundled_development().unwrap(),
            &OpenSetOodDataset::bundled_test().unwrap(),
            &OpenSetContrastDataset::bundled_test().unwrap(),
            OpenSetTrainingConfig::default().seed,
        )
        .unwrap();
        let expected_ids = plan
            .train
            .iter()
            .chain(&plan.development)
            .map(|example| example.id.as_str())
            .collect::<BTreeSet<_>>();
        let actual_ids = pool
            .examples
            .iter()
            .map(|example| example.id.as_str())
            .collect::<BTreeSet<_>>();
        assert_eq!(actual_ids, expected_ids);
        assert!(plan
            .calibration
            .iter()
            .chain(&plan.id_test)
            .all(|example| !actual_ids.contains(example.id.as_str())));
    }

    #[test]
    fn nested_plan_holds_out_every_family_once_without_leakage() {
        let dataset = GroupedDataset::bundled().unwrap();
        let pool =
            SelectionAuditPool::from_dataset(&dataset, OpenSetTrainingConfig::default().seed)
                .unwrap();
        let outer = outer_holdouts(&pool);
        let all_outer = outer
            .iter()
            .flatten()
            .map(String::as_str)
            .collect::<Vec<_>>();
        assert_eq!(all_outer.len(), 77);
        assert_eq!(all_outer.iter().copied().collect::<HashSet<_>>().len(), 77);
        for (outer_index, holdout) in outer.iter().enumerate() {
            assert_eq!(holdout.len(), 7);
            let held = holdout.iter().map(String::as_str).collect::<HashSet<_>>();
            let inner = inner_holdouts(&pool, &held, outer_index);
            assert_eq!(inner.len(), INNER_FOLDS);
            let all_inner = inner
                .iter()
                .flatten()
                .map(String::as_str)
                .collect::<Vec<_>>();
            assert_eq!(all_inner.len(), 70);
            assert_eq!(all_inner.iter().copied().collect::<HashSet<_>>().len(), 70);
            assert!(all_inner.iter().all(|family| !held.contains(family)));
            assert!(inner.iter().all(|fold| fold.len() == 14));
        }
    }

    #[test]
    fn report_candidate_ids_must_be_unambiguous() {
        let dataset = GroupedDataset::bundled().unwrap();
        let pool =
            SelectionAuditPool::from_dataset(&dataset, OpenSetTrainingConfig::default().seed)
                .unwrap();
        let mut config = fast_config();
        config.training.l2_penalty = 0.000_100_1;
        config.training.development_selection.l2_penalty_candidates =
            vec![0.000_100_1, 0.000_100_2];
        assert!(matches!(
            config.validate(&pool),
            Err(MlError::InvalidConfiguration(message))
                if message.contains("candidate IDs collide")
        ));
    }

    #[test]
    fn excluded_partition_text_cannot_change_the_selection_pool() {
        let dataset = GroupedDataset::bundled().unwrap();
        let seed = OpenSetTrainingConfig::default().seed;
        let original = SelectionAuditPool::from_dataset(&dataset, seed).unwrap();
        let included_ids = original
            .examples()
            .iter()
            .map(|example| example.id.as_str())
            .collect::<HashSet<_>>();
        let mut changed = false;
        let rewritten = include_str!("../../fixtures/intents-v3.tsv")
            .lines()
            .map(|line| {
                let mut fields = line.split('\t').collect::<Vec<_>>();
                if !changed
                    && fields.len() == 4
                    && fields[0] != "id"
                    && !included_ids.contains(fields[0])
                {
                    fields[3] = "Quarantined evaluation wording changed zxqv final partition only";
                    changed = true;
                    fields.join("\t")
                } else {
                    line.to_owned()
                }
            })
            .collect::<Vec<_>>()
            .join("\n");
        assert!(changed);
        let modified = GroupedDataset::from_tsv(&rewritten).unwrap();
        let rebuilt = SelectionAuditPool::from_dataset(&modified, seed).unwrap();
        assert_eq!(rebuilt.pool_sha256(), original.pool_sha256());
        assert_eq!(rebuilt.examples(), original.examples());
    }

    #[test]
    fn audit_is_byte_deterministic_and_tests_all_nine_candidates() {
        let dataset = GroupedDataset::bundled().unwrap();
        let pool =
            SelectionAuditPool::from_dataset(&dataset, OpenSetTrainingConfig::default().seed)
                .unwrap();
        let first = run_selection_stability_audit(&pool, fast_config()).unwrap();
        let second = run_selection_stability_audit(&pool, fast_config()).unwrap();
        assert_eq!(
            selection_audit_report_bytes(&first).unwrap(),
            selection_audit_report_bytes(&second).unwrap()
        );
        assert_eq!(first.grid.len(), 9);
        assert_eq!(first.model_fits, 506);
        assert!(first
            .folds
            .iter()
            .all(|fold| fold.inner_candidates.len() == 9));
        assert_eq!(
            first
                .ledger
                .iter()
                .map(|row| row.id.as_str())
                .collect::<HashSet<_>>()
                .len(),
            385
        );
    }
}
