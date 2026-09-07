//! Leak-resistant, local open-set intent classification.
//!
//! This module is the version-three experimental path. It deliberately keeps the legacy model and
//! CLI stable while adding group-aware data partitions, probability calibration, independent OOD
//! evaluation, cryptographically linked artifacts, and a compiled inference representation.

use crate::ml::{MlError, VectorizerConfig};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fmt::Write as _;
use std::fs;
use std::io::{BufRead, Read, Write};
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use unicode_normalization::UnicodeNormalization;

mod selection_audit;
pub use selection_audit::{
    run_selection_stability_audit, selection_audit_report_bytes, write_selection_audit_report,
    SelectionAuditConfig, SelectionAuditPool, SelectionAuditReport,
    DEFAULT_SELECTION_AUDIT_BOOTSTRAP_RESAMPLES,
};

pub const OPEN_SET_SCHEMA_VERSION: u32 = 3;
pub const OPEN_SET_MODEL_VERSION: &str = "3.0.0";
pub const OPEN_SET_MODEL_KIND: &str = "eliza-open-set-linear";
pub const OPEN_SET_BUNDLE_VERSION: &str = "3.0.0";
pub const DEFAULT_BOOTSTRAP_RESAMPLES: usize = 1_000;
const BUNDLE_INVENTORY: [&str; 5] = [
    "manifest.json",
    "metrics.json",
    "model.json",
    "policy.json",
    "split-plan.json",
];
const MAX_EXAMPLES: usize = 100_000;
const MAX_GROUPS_PER_LABEL: usize = 20_000;
const MAX_CLASSES: usize = 256;
const MAX_JSON_BYTES: u64 = 64 * 1024 * 1024;
const MAX_JSONL_BYTES: usize = crate::MAX_INPUT_CHARS * 4 + 16_384;
const MAX_BOOTSTRAP_SAMPLED_ROWS: usize = 5_000_000;
const MAX_PARAMETER_MAGNITUDE: f64 = 1_000_000.0;
const MAX_IDF: f64 = 64.0;
const MIN_PARAPHRASES_PER_FAMILY: usize = 3;
const COMMON_FEATURE_FAMILY_FREQUENCY: usize = 3;
const MAX_RAW_CROSS_FAMILY_JACCARD: f64 = 0.30;
const MAX_RESIDUAL_CROSS_FAMILY_JACCARD: f64 = 0.25;
const MAX_SIMILARITY_CANDIDATE_PAIRS: usize = 500_000;
const MAX_SIMILARITY_PAIR_INSERT_ATTEMPTS: usize = 5_000_000;
const METRIC_REPORTING_SCALE: f64 = 1_000_000_000.0;
const METRIC_REPORTING_UNIT: f64 = 1.0 / METRIC_REPORTING_SCALE;
const METRIC_COMPARISON_TOLERANCE: f64 = 2.0 * METRIC_REPORTING_UNIT;
static BUNDLE_TEMP_SEQUENCE: AtomicUsize = AtomicUsize::new(0);

#[derive(Debug, Clone, Copy)]
struct SimilarityCandidateBudget {
    maximum_candidate_pairs: usize,
    maximum_pair_insert_attempts: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GroupedExample {
    pub id: String,
    pub group_id: String,
    pub label: String,
    pub text: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GroupedDataset {
    examples: Vec<GroupedExample>,
}

impl GroupedDataset {
    pub fn bundled() -> Result<Self, MlError> {
        Self::from_tsv(include_str!("../fixtures/intents-v3.tsv"))
    }

    pub fn read(path: impl AsRef<Path>) -> Result<Self, MlError> {
        let path = path.as_ref();
        reject_oversized_file(path, MAX_JSON_BYTES, "grouped dataset")?;
        Self::from_tsv(&fs::read_to_string(path)?)
    }

    pub fn from_tsv(input: &str) -> Result<Self, MlError> {
        let mut lines = input.lines();
        let header = lines
            .next()
            .map(str::trim_end)
            .ok_or_else(|| MlError::InvalidDataset("the grouped dataset is empty".into()))?;
        if header != "id\tgroup_id\tlabel\ttext" {
            return Err(MlError::InvalidDataset(
                "the grouped header must be exactly `id\\tgroup_id\\tlabel\\ttext`".into(),
            ));
        }

        let mut examples = Vec::new();
        let mut ids = HashSet::new();
        let mut feature_identities = HashSet::new();
        let mut group_labels: HashMap<String, String> = HashMap::new();
        for (offset, raw_line) in lines.enumerate() {
            let line_number = offset + 2;
            let line = raw_line.trim_end_matches('\r');
            if line.trim().is_empty() || line.trim_start().starts_with('#') {
                continue;
            }
            let fields = line.split('\t').collect::<Vec<_>>();
            if fields.len() != 4 {
                return Err(MlError::InvalidDataset(format!(
                    "grouped line {line_number} must contain four tab-separated fields"
                )));
            }
            let id = fields[0].trim();
            let group_id = fields[1].trim();
            let label = fields[2].trim();
            let text = fields[3].trim();
            validate_identifier(id, "example id", line_number)?;
            validate_identifier(group_id, "group id", line_number)?;
            validate_label(label, line_number)?;
            validate_text(text, line_number, "grouped")?;
            if !ids.insert(id.to_owned()) {
                return Err(MlError::InvalidDataset(format!(
                    "duplicate grouped example id `{id}`"
                )));
            }
            if !feature_identities.insert(feature_identity(text)) {
                return Err(MlError::InvalidDataset(format!(
                    "grouped line {line_number} duplicates a feature-equivalent text"
                )));
            }
            match group_labels.get(group_id) {
                Some(existing) if existing != label => {
                    return Err(MlError::InvalidDataset(format!(
                        "group `{group_id}` crosses labels `{existing}` and `{label}`"
                    )))
                }
                Some(_) => {}
                None => {
                    group_labels.insert(group_id.to_owned(), label.to_owned());
                }
            }
            examples.push(GroupedExample {
                id: id.to_owned(),
                group_id: group_id.to_owned(),
                label: label.to_owned(),
                text: text.to_owned(),
            });
            if examples.len() > MAX_EXAMPLES {
                return Err(MlError::InvalidDataset(format!(
                    "the grouped dataset exceeds {MAX_EXAMPLES} examples"
                )));
            }
        }
        if examples.is_empty() {
            return Err(MlError::InvalidDataset(
                "the grouped dataset contains no examples".into(),
            ));
        }
        let dataset = Self { examples };
        dataset.validate_partition_support()?;
        dataset.validate_family_contract()?;
        Ok(dataset)
    }

    pub fn examples(&self) -> &[GroupedExample] {
        &self.examples
    }

    pub fn labels(&self) -> Vec<String> {
        self.examples
            .iter()
            .map(|example| example.label.clone())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect()
    }

    pub fn fingerprint_sha256(&self) -> String {
        let mut rows = self
            .examples
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

    fn validate_partition_support(&self) -> Result<(), MlError> {
        let mut groups_by_label: BTreeMap<&str, HashSet<&str>> = BTreeMap::new();
        for example in &self.examples {
            groups_by_label
                .entry(&example.label)
                .or_default()
                .insert(&example.group_id);
        }
        if groups_by_label.len() < 2 || groups_by_label.len() > MAX_CLASSES {
            return Err(MlError::InvalidDataset(format!(
                "the grouped dataset must contain between 2 and {MAX_CLASSES} labels"
            )));
        }
        for (label, groups) in groups_by_label {
            if groups.len() < 4 {
                return Err(MlError::InvalidDataset(format!(
                    "label `{label}` has {} groups; at least four are required",
                    groups.len()
                )));
            }
            if groups.len() > MAX_GROUPS_PER_LABEL {
                return Err(MlError::InvalidDataset(format!(
                    "label `{label}` exceeds the group limit"
                )));
            }
        }
        Ok(())
    }

    fn validate_family_contract(&self) -> Result<(), MlError> {
        let mut family_sizes: BTreeMap<&str, usize> = BTreeMap::new();
        for example in &self.examples {
            *family_sizes.entry(&example.group_id).or_insert(0) += 1;
        }
        let expected_size = family_sizes.values().next().copied().unwrap_or(0);
        if expected_size < MIN_PARAPHRASES_PER_FAMILY
            || family_sizes.values().any(|size| *size != expected_size)
        {
            return Err(MlError::InvalidDataset(format!(
                "every paraphrase family must contain the same number of examples and at least {MIN_PARAPHRASES_PER_FAMILY}"
            )));
        }
        validate_cross_family_similarity(&self.examples)
    }
}

fn validate_cross_family_similarity(examples: &[GroupedExample]) -> Result<(), MlError> {
    let config = VectorizerConfig::default();
    let feature_sets = examples
        .iter()
        .map(|example| {
            extract_terms(&example.text, &config)
                .into_iter()
                .collect::<HashSet<_>>()
        })
        .collect::<Vec<_>>();
    let mut group_features_by_label: BTreeMap<&str, BTreeMap<&str, HashSet<&str>>> =
        BTreeMap::new();
    for (example, features) in examples.iter().zip(&feature_sets) {
        let group_features = group_features_by_label
            .entry(&example.label)
            .or_default()
            .entry(&example.group_id)
            .or_default();
        group_features.extend(features.iter().map(String::as_str));
    }
    let mut common_features_by_label: BTreeMap<&str, HashSet<&str>> = BTreeMap::new();
    for (label, groups) in &group_features_by_label {
        let mut family_frequency: HashMap<&str, usize> = HashMap::new();
        for features in groups.values() {
            for feature in features {
                *family_frequency.entry(feature).or_insert(0) += 1;
            }
        }
        common_features_by_label.insert(
            label,
            family_frequency
                .into_iter()
                .filter_map(|(feature, count)| {
                    (count >= COMMON_FEATURE_FAMILY_FREQUENCY).then_some(feature)
                })
                .collect(),
        );
    }
    let residual_feature_sets = examples
        .iter()
        .zip(&feature_sets)
        .map(|(example, features)| {
            let common = &common_features_by_label[example.label.as_str()];
            features
                .iter()
                .filter(|feature| !common.contains(feature.as_str()))
                .cloned()
                .collect::<HashSet<_>>()
        })
        .collect::<Vec<_>>();
    let mut candidate_pairs = HashSet::new();
    let mut pair_insert_attempts = 0usize;
    let budget = SimilarityCandidateBudget {
        maximum_candidate_pairs: MAX_SIMILARITY_CANDIDATE_PAIRS,
        maximum_pair_insert_attempts: MAX_SIMILARITY_PAIR_INSERT_ATTEMPTS,
    };
    collect_similarity_candidates(
        examples,
        &feature_sets,
        MAX_RAW_CROSS_FAMILY_JACCARD,
        false,
        &mut candidate_pairs,
        &mut pair_insert_attempts,
        budget,
    )?;
    collect_similarity_candidates(
        examples,
        &residual_feature_sets,
        MAX_RESIDUAL_CROSS_FAMILY_JACCARD,
        true,
        &mut candidate_pairs,
        &mut pair_insert_attempts,
        budget,
    )?;
    let mut candidate_pairs = candidate_pairs.into_iter().collect::<Vec<_>>();
    candidate_pairs.sort_unstable();
    for (left_index, right_index) in candidate_pairs {
        let left = &examples[left_index];
        let right = &examples[right_index];
        let raw_intersection = feature_sets[left_index]
            .intersection(&feature_sets[right_index])
            .count();
        let raw_union =
            feature_sets[left_index].len() + feature_sets[right_index].len() - raw_intersection;
        let raw_similarity = if raw_union == 0 {
            0.0
        } else {
            raw_intersection as f64 / raw_union as f64
        };
        if raw_similarity >= MAX_RAW_CROSS_FAMILY_JACCARD {
            return Err(MlError::InvalidDataset(format!(
                    "examples `{}` ({}) and `{}` ({}) cross paraphrase families with raw feature Jaccard {raw_similarity:.6}; maximum is below {MAX_RAW_CROSS_FAMILY_JACCARD:.2}",
                    left.id, left.group_id, right.id, right.group_id
                )));
        }
        if left.label != right.label {
            continue;
        }
        let left_residual = &residual_feature_sets[left_index];
        let right_residual = &residual_feature_sets[right_index];
        let intersection = left_residual.intersection(right_residual).count();
        let union = left_residual.len() + right_residual.len() - intersection;
        let similarity = if union == 0 {
            0.0
        } else {
            intersection as f64 / union as f64
        };
        if similarity >= MAX_RESIDUAL_CROSS_FAMILY_JACCARD {
            return Err(MlError::InvalidDataset(format!(
                    "examples `{}` ({}) and `{}` ({}) cross paraphrase families with residual feature Jaccard {similarity:.6}; maximum is below {MAX_RESIDUAL_CROSS_FAMILY_JACCARD:.2}",
                    left.id, left.group_id, right.id, right.group_id
                )));
        }
    }
    Ok(())
}

fn collect_similarity_candidates(
    examples: &[GroupedExample],
    feature_sets: &[HashSet<String>],
    threshold: f64,
    same_label_only: bool,
    candidates: &mut HashSet<(usize, usize)>,
    pair_insert_attempts: &mut usize,
    budget: SimilarityCandidateBudget,
) -> Result<(), MlError> {
    let mut document_frequency: HashMap<&str, usize> = HashMap::new();
    for features in feature_sets {
        for feature in features {
            *document_frequency.entry(feature).or_insert(0) += 1;
        }
    }
    let ordered_features = feature_sets
        .iter()
        .map(|features| {
            let mut ordered = features.iter().map(String::as_str).collect::<Vec<_>>();
            ordered.sort_unstable_by(|left, right| {
                document_frequency[left]
                    .cmp(&document_frequency[right])
                    .then_with(|| left.cmp(right))
            });
            ordered
        })
        .collect::<Vec<_>>();
    let mut postings: HashMap<&str, Vec<usize>> = HashMap::new();
    for (index, features) in ordered_features.iter().enumerate() {
        if features.is_empty() {
            continue;
        }
        let required_overlap = (threshold * features.len() as f64).ceil() as usize;
        let prefix_length = (features.len() - required_overlap + 1).min(features.len());
        for feature in features.iter().take(prefix_length) {
            if let Some(previous) = postings.get(feature) {
                for other_index in previous {
                    if examples[*other_index].group_id == examples[index].group_id
                        || (same_label_only
                            && examples[*other_index].label != examples[index].label)
                    {
                        continue;
                    }
                    let smaller = feature_sets[*other_index]
                        .len()
                        .min(feature_sets[index].len());
                    let larger = feature_sets[*other_index]
                        .len()
                        .max(feature_sets[index].len());
                    if larger == 0 || smaller as f64 / (larger as f64) < threshold {
                        continue;
                    }
                    *pair_insert_attempts += 1;
                    if *pair_insert_attempts > budget.maximum_pair_insert_attempts {
                        return Err(MlError::InvalidDataset(
                            "the paraphrase similarity review exceeds its bounded comparison budget"
                                .into(),
                        ));
                    }
                    candidates.insert((*other_index, index));
                    if candidates.len() > budget.maximum_candidate_pairs {
                        return Err(MlError::InvalidDataset(
                            "the paraphrase similarity review exceeds its bounded candidate budget"
                                .into(),
                        ));
                    }
                }
            }
            postings.entry(feature).or_default().push(index);
        }
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "kebab-case")]
pub enum OodStratum {
    Semantic,
    Capability,
    Noise,
}

impl OodStratum {
    fn parse(value: &str, line_number: usize) -> Result<Self, MlError> {
        match value {
            "semantic" => Ok(Self::Semantic),
            "capability" => Ok(Self::Capability),
            "noise" => Ok(Self::Noise),
            _ => Err(MlError::InvalidDataset(format!(
                "OOD line {line_number} has unsupported stratum `{value}`"
            ))),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Semantic => "semantic",
            Self::Capability => "capability",
            Self::Noise => "noise",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpenSetOodExample {
    pub id: String,
    pub family_id: String,
    pub domain_group: String,
    pub stratum: OodStratum,
    pub text: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpenSetOodDataset {
    examples: Vec<OpenSetOodExample>,
}

impl OpenSetOodDataset {
    pub fn bundled_development() -> Result<Self, MlError> {
        Self::from_tsv(include_str!("../fixtures/ood-dev-v3.tsv"))
    }

    pub fn bundled_test() -> Result<Self, MlError> {
        Self::from_tsv(include_str!("../fixtures/ood-test-v3.tsv"))
    }

    pub fn read(path: impl AsRef<Path>) -> Result<Self, MlError> {
        let path = path.as_ref();
        reject_oversized_file(path, MAX_JSON_BYTES, "OOD dataset")?;
        Self::from_tsv(&fs::read_to_string(path)?)
    }

    pub fn from_tsv(input: &str) -> Result<Self, MlError> {
        let mut lines = input.lines();
        let header = lines
            .next()
            .map(str::trim_end)
            .ok_or_else(|| MlError::InvalidDataset("the OOD dataset is empty".into()))?;
        if header != "id\tfamily_id\tdomain_group\tstratum\ttext" {
            return Err(MlError::InvalidDataset(
                "the OOD header must be exactly `id\\tfamily_id\\tdomain_group\\tstratum\\ttext`"
                    .into(),
            ));
        }
        let mut examples = Vec::new();
        let mut ids = HashSet::new();
        let mut texts = HashSet::new();
        for (offset, raw_line) in lines.enumerate() {
            let line_number = offset + 2;
            let line = raw_line.trim_end_matches('\r');
            if line.trim().is_empty() || line.trim_start().starts_with('#') {
                continue;
            }
            let fields = line.split('\t').collect::<Vec<_>>();
            if fields.len() != 5 {
                return Err(MlError::InvalidDataset(format!(
                    "OOD line {line_number} must contain five tab-separated fields"
                )));
            }
            let id = fields[0].trim();
            let family_id = fields[1].trim();
            let domain_group = fields[2].trim();
            let stratum = OodStratum::parse(fields[3].trim(), line_number)?;
            let text = fields[4].trim();
            validate_identifier(id, "OOD id", line_number)?;
            validate_identifier(family_id, "OOD family id", line_number)?;
            validate_identifier(domain_group, "OOD domain group", line_number)?;
            validate_text(text, line_number, "OOD")?;
            if !ids.insert(id.to_owned()) {
                return Err(MlError::InvalidDataset(format!("duplicate OOD id `{id}`")));
            }
            if !texts.insert(feature_identity(text)) {
                return Err(MlError::InvalidDataset(format!(
                    "OOD line {line_number} duplicates a feature-equivalent text"
                )));
            }
            examples.push(OpenSetOodExample {
                id: id.to_owned(),
                family_id: family_id.to_owned(),
                domain_group: domain_group.to_owned(),
                stratum,
                text: text.to_owned(),
            });
            if examples.len() > MAX_EXAMPLES {
                return Err(MlError::InvalidDataset(format!(
                    "the OOD dataset exceeds {MAX_EXAMPLES} examples"
                )));
            }
        }
        if examples.is_empty() {
            return Err(MlError::InvalidDataset(
                "the OOD dataset contains no examples".into(),
            ));
        }
        let dataset = Self { examples };
        dataset.validate_contract()?;
        Ok(dataset)
    }

    pub fn examples(&self) -> &[OpenSetOodExample] {
        &self.examples
    }

    pub fn fingerprint_sha256(&self) -> String {
        let mut rows = self
            .examples
            .iter()
            .map(|example| {
                format!(
                    "{}\t{}\t{}\t{}\t{}",
                    example.id,
                    example.family_id,
                    example.domain_group,
                    example.stratum.as_str(),
                    feature_identity(&example.text)
                )
            })
            .collect::<Vec<_>>();
        rows.sort();
        sha256_hex(rows.join("\n").as_bytes())
    }

    fn validate_contract(&self) -> Result<(), MlError> {
        let mut family_ownership: BTreeMap<&str, (&str, OodStratum, usize)> = BTreeMap::new();
        let mut domain_ownership: BTreeMap<&str, (OodStratum, BTreeSet<&str>)> = BTreeMap::new();
        let mut rows_by_stratum: BTreeMap<OodStratum, usize> = BTreeMap::new();
        let mut domains_by_stratum: BTreeMap<OodStratum, BTreeSet<&str>> = BTreeMap::new();
        for example in &self.examples {
            match family_ownership.entry(&example.family_id) {
                std::collections::btree_map::Entry::Vacant(entry) => {
                    entry.insert((&example.domain_group, example.stratum, 1));
                }
                std::collections::btree_map::Entry::Occupied(mut entry) => {
                    let (domain, stratum, count) = entry.get_mut();
                    if *domain != example.domain_group || *stratum != example.stratum {
                        return Err(MlError::InvalidDataset(format!(
                            "OOD family `{}` crosses domain groups or strata",
                            example.family_id
                        )));
                    }
                    *count += 1;
                }
            }
            match domain_ownership.entry(&example.domain_group) {
                std::collections::btree_map::Entry::Vacant(entry) => {
                    entry.insert((
                        example.stratum,
                        BTreeSet::from([example.family_id.as_str()]),
                    ));
                }
                std::collections::btree_map::Entry::Occupied(mut entry) => {
                    let (stratum, families) = entry.get_mut();
                    if *stratum != example.stratum {
                        return Err(MlError::InvalidDataset(format!(
                            "OOD domain `{}` crosses strata",
                            example.domain_group
                        )));
                    }
                    families.insert(&example.family_id);
                }
            }
            *rows_by_stratum.entry(example.stratum).or_insert(0) += 1;
            domains_by_stratum
                .entry(example.stratum)
                .or_default()
                .insert(&example.domain_group);
        }
        let expected_family_size = family_ownership
            .values()
            .next()
            .map(|(_, _, count)| *count)
            .unwrap_or(0);
        if expected_family_size < MIN_PARAPHRASES_PER_FAMILY
            || family_ownership
                .values()
                .any(|(_, _, count)| *count != expected_family_size)
            || domain_ownership
                .values()
                .any(|(_, families)| families.len() < 2)
        {
            return Err(MlError::InvalidDataset(
                "OOD families must have equal multi-prompt support and every domain must contain at least two families"
                    .into(),
            ));
        }
        let strata = [
            OodStratum::Semantic,
            OodStratum::Capability,
            OodStratum::Noise,
        ];
        let expected_rows = rows_by_stratum.get(&strata[0]).copied().unwrap_or(0);
        let expected_domains = domains_by_stratum
            .get(&strata[0])
            .map(BTreeSet::len)
            .unwrap_or(0);
        if expected_rows == 0
            || expected_domains == 0
            || strata.iter().any(|stratum| {
                rows_by_stratum.get(stratum).copied() != Some(expected_rows)
                    || domains_by_stratum.get(stratum).map(BTreeSet::len) != Some(expected_domains)
            })
        {
            return Err(MlError::InvalidDataset(
                "OOD semantic, capability, and noise strata must have equal row and domain support"
                    .into(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "lowercase")]
pub enum ContrastVariant {
    A,
    B,
}

impl ContrastVariant {
    fn parse(value: &str, line_number: usize) -> Result<Self, MlError> {
        match value {
            "a" => Ok(Self::A),
            "b" => Ok(Self::B),
            _ => Err(MlError::InvalidDataset(format!(
                "contrast line {line_number} has unsupported variant `{value}`"
            ))),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::A => "a",
            Self::B => "b",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpenSetContrastExample {
    pub id: String,
    pub pair_id: String,
    pub variant: ContrastVariant,
    pub label: String,
    pub text: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpenSetContrastDataset {
    examples: Vec<OpenSetContrastExample>,
}

impl OpenSetContrastDataset {
    pub fn bundled_test() -> Result<Self, MlError> {
        Self::from_tsv(include_str!("../fixtures/contrast-test-v3.tsv"))
    }

    pub fn read(path: impl AsRef<Path>) -> Result<Self, MlError> {
        let path = path.as_ref();
        reject_oversized_file(path, MAX_JSON_BYTES, "contrast dataset")?;
        Self::from_tsv(&fs::read_to_string(path)?)
    }

    pub fn from_tsv(input: &str) -> Result<Self, MlError> {
        let mut lines = input.lines();
        let header = lines
            .next()
            .map(str::trim_end)
            .ok_or_else(|| MlError::InvalidDataset("the contrast dataset is empty".into()))?;
        if header != "id\tpair_id\tvariant\tlabel\ttext" {
            return Err(MlError::InvalidDataset(
                "the contrast header must be exactly `id\\tpair_id\\tvariant\\tlabel\\ttext`"
                    .into(),
            ));
        }
        let mut examples = Vec::new();
        let mut ids = HashSet::new();
        let mut feature_identities = HashSet::new();
        for (offset, raw_line) in lines.enumerate() {
            let line_number = offset + 2;
            let line = raw_line.trim_end_matches('\r');
            if line.trim().is_empty() || line.trim_start().starts_with('#') {
                continue;
            }
            let fields = line.split('\t').collect::<Vec<_>>();
            if fields.len() != 5 {
                return Err(MlError::InvalidDataset(format!(
                    "contrast line {line_number} must contain five tab-separated fields"
                )));
            }
            let id = fields[0].trim();
            let pair_id = fields[1].trim();
            let variant = ContrastVariant::parse(fields[2].trim(), line_number)?;
            let label = fields[3].trim();
            let text = fields[4].trim();
            validate_identifier(id, "contrast id", line_number)?;
            validate_identifier(pair_id, "contrast pair id", line_number)?;
            validate_label(label, line_number)?;
            validate_text(text, line_number, "contrast")?;
            if !ids.insert(id.to_owned()) {
                return Err(MlError::InvalidDataset(format!(
                    "duplicate contrast id `{id}`"
                )));
            }
            if !feature_identities.insert(feature_identity(text)) {
                return Err(MlError::InvalidDataset(format!(
                    "contrast line {line_number} duplicates a feature-equivalent text"
                )));
            }
            examples.push(OpenSetContrastExample {
                id: id.to_owned(),
                pair_id: pair_id.to_owned(),
                variant,
                label: label.to_owned(),
                text: text.to_owned(),
            });
            if examples.len() > MAX_EXAMPLES {
                return Err(MlError::InvalidDataset(format!(
                    "the contrast dataset exceeds {MAX_EXAMPLES} examples"
                )));
            }
        }
        let dataset = Self { examples };
        dataset.validate_contract()?;
        Ok(dataset)
    }

    pub fn examples(&self) -> &[OpenSetContrastExample] {
        &self.examples
    }

    pub fn fingerprint_sha256(&self) -> String {
        let mut rows = self
            .examples
            .iter()
            .map(|example| {
                format!(
                    "{}\t{}\t{}\t{}\t{}",
                    example.id,
                    example.pair_id,
                    example.variant.as_str(),
                    example.label,
                    feature_identity(&example.text)
                )
            })
            .collect::<Vec<_>>();
        rows.sort();
        sha256_hex(rows.join("\n").as_bytes())
    }

    fn validate_contract(&self) -> Result<(), MlError> {
        let mut pairs: BTreeMap<&str, (BTreeSet<ContrastVariant>, BTreeSet<&str>)> =
            BTreeMap::new();
        let mut label_counts: BTreeMap<&str, usize> = BTreeMap::new();
        for example in &self.examples {
            let (variants, labels) = pairs.entry(&example.pair_id).or_default();
            if !variants.insert(example.variant) {
                return Err(MlError::InvalidDataset(format!(
                    "contrast pair `{}` repeats a variant",
                    example.pair_id
                )));
            }
            labels.insert(&example.label);
            *label_counts.entry(&example.label).or_insert(0) += 1;
        }
        if pairs.is_empty()
            || pairs
                .values()
                .any(|(variants, labels)| variants.len() != 2 || labels.len() != 2)
        {
            return Err(MlError::InvalidDataset(
                "every contrast pair must contain variants a and b with different labels".into(),
            ));
        }
        let expected_label_support = label_counts.values().next().copied().unwrap_or(0);
        if label_counts.len() < 2
            || expected_label_support < 2
            || label_counts
                .values()
                .any(|count| *count != expected_label_support)
        {
            return Err(MlError::InvalidDataset(
                "the contrast set must be multi-intent with equal label support".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "kebab-case")]
pub enum PartitionKind {
    Train,
    Development,
    Calibration,
    IdTest,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum SelectionDataRole {
    Train,
    Development,
    OodDevelopment,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct SplitAssignment {
    pub id: String,
    pub group_id: String,
    pub label: String,
    pub text: String,
    pub partition: PartitionKind,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct OodPlanRow {
    pub id: String,
    pub family_id: String,
    pub domain_group: String,
    pub stratum: OodStratum,
    pub text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ContrastPlanRow {
    pub id: String,
    pub pair_id: String,
    pub variant: ContrastVariant,
    pub label: String,
    pub text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct SplitPlanManifest {
    pub schema_version: u32,
    pub strategy: String,
    pub seed: u64,
    pub dataset_sha256: String,
    pub assignments: Vec<SplitAssignment>,
    pub ood_development: Vec<OodPlanRow>,
    pub ood_test: Vec<OodPlanRow>,
    pub contrast_test: Vec<ContrastPlanRow>,
}

impl SplitPlanManifest {
    fn validate_contract(&self) -> Result<(), MlError> {
        if self.schema_version != OPEN_SET_SCHEMA_VERSION
            || self.strategy != "group-stratified-scaled-four-way-v3"
            || self.seed > 9_007_199_254_740_991
            || !valid_sha256(&self.dataset_sha256)
            || self.assignments.is_empty()
            || self.assignments.len() > MAX_EXAMPLES
            || self.ood_development.is_empty()
            || self.ood_development.len() > MAX_EXAMPLES
            || self.ood_test.is_empty()
            || self.ood_test.len() > MAX_EXAMPLES
            || self.contrast_test.is_empty()
            || self.contrast_test.len() > MAX_EXAMPLES
        {
            return Err(MlError::InvalidDataset(
                "the split-plan manifest has an invalid identity or size".into(),
            ));
        }

        let mut ids = HashSet::new();
        let mut group_ownership: HashMap<&str, (&str, PartitionKind)> = HashMap::new();
        let mut groups_by_label: BTreeMap<&str, BTreeSet<&str>> = BTreeMap::new();
        let mut labels_by_partition: BTreeMap<PartitionKind, BTreeSet<&str>> = BTreeMap::new();
        let mut feature_identities = HashSet::new();
        let mut previous_id: Option<&str> = None;
        for (index, assignment) in self.assignments.iter().enumerate() {
            validate_identifier(&assignment.id, "split assignment id", index + 1)?;
            validate_identifier(&assignment.group_id, "split assignment group id", index + 1)?;
            validate_label(&assignment.label, index + 1)?;
            validate_text(&assignment.text, index + 1, "split assignment")?;
            if previous_id.is_some_and(|previous| previous >= assignment.id.as_str())
                || !ids.insert(assignment.id.as_str())
            {
                return Err(MlError::InvalidDataset(
                    "split assignments must have unique, ascending ids".into(),
                ));
            }
            previous_id = Some(&assignment.id);
            if !feature_identities.insert(feature_identity(&assignment.text)) {
                return Err(MlError::InvalidDataset(
                    "split assignments contain feature-equivalent texts".into(),
                ));
            }
            match group_ownership.insert(
                assignment.group_id.as_str(),
                (assignment.label.as_str(), assignment.partition),
            ) {
                Some((label, partition))
                    if label != assignment.label || partition != assignment.partition =>
                {
                    return Err(MlError::InvalidDataset(format!(
                        "split group `{}` crosses labels or partitions",
                        assignment.group_id
                    )))
                }
                _ => {}
            }
            labels_by_partition
                .entry(assignment.partition)
                .or_default()
                .insert(&assignment.label);
            groups_by_label
                .entry(&assignment.label)
                .or_default()
                .insert(&assignment.group_id);
        }
        let expected_labels = labels_by_partition
            .get(&PartitionKind::Train)
            .cloned()
            .unwrap_or_default();
        if expected_labels.len() < 2
            || [
                PartitionKind::Train,
                PartitionKind::Development,
                PartitionKind::Calibration,
                PartitionKind::IdTest,
            ]
            .iter()
            .any(|partition| labels_by_partition.get(partition) != Some(&expected_labels))
        {
            return Err(MlError::InvalidDataset(
                "every split partition must contain the same two or more labels".into(),
            ));
        }
        for (label, groups) in groups_by_label {
            if groups.len() < 4 || groups.len() > MAX_GROUPS_PER_LABEL {
                return Err(MlError::InvalidDataset(format!(
                    "split label `{label}` has an invalid group count"
                )));
            }
            let mut ordered_groups = groups.into_iter().collect::<Vec<_>>();
            ordered_groups.sort_by(|left, right| {
                group_split_hash(label, left, self.seed)
                    .cmp(&group_split_hash(label, right, self.seed))
                    .then_with(|| left.cmp(right))
            });
            let evaluation_groups = evaluation_group_quota(ordered_groups.len());
            for (group_index, group_id) in ordered_groups.into_iter().enumerate() {
                let expected_partition = if group_index < evaluation_groups {
                    PartitionKind::IdTest
                } else if group_index < evaluation_groups * 2 {
                    PartitionKind::Calibration
                } else if group_index < evaluation_groups * 3 {
                    PartitionKind::Development
                } else {
                    PartitionKind::Train
                };
                if group_ownership[group_id].1 != expected_partition {
                    return Err(MlError::InvalidDataset(format!(
                        "split group `{group_id}` does not match the declared strategy"
                    )));
                }
            }
        }
        validate_plan_ood_rows(
            &self.ood_development,
            &self.ood_test,
            &ids,
            &group_ownership,
            &mut feature_identities,
        )?;
        validate_plan_contrast_rows(
            &self.contrast_test,
            &ids,
            &group_ownership,
            &self.ood_development,
            &self.ood_test,
            &mut feature_identities,
        )?;
        Ok(())
    }
}

fn validate_plan_contrast_rows(
    rows: &[ContrastPlanRow],
    supervised_ids: &HashSet<&str>,
    supervised_groups: &HashMap<&str, (&str, PartitionKind)>,
    ood_development: &[OodPlanRow],
    ood_test: &[OodPlanRow],
    feature_identities: &mut HashSet<String>,
) -> Result<(), MlError> {
    let ood_ids = ood_development
        .iter()
        .chain(ood_test)
        .map(|row| row.id.as_str())
        .collect::<HashSet<_>>();
    let ood_groups = ood_development
        .iter()
        .chain(ood_test)
        .flat_map(|row| [row.family_id.as_str(), row.domain_group.as_str()])
        .collect::<HashSet<_>>();
    let mut previous_id: Option<&str> = None;
    let mut ids = HashSet::new();
    let dataset = OpenSetContrastDataset {
        examples: rows
            .iter()
            .enumerate()
            .map(|(index, row)| {
                validate_identifier(&row.id, "contrast plan id", index + 1)?;
                validate_identifier(&row.pair_id, "contrast plan pair id", index + 1)?;
                validate_label(&row.label, index + 1)?;
                validate_text(&row.text, index + 1, "contrast plan")?;
                if previous_id.is_some_and(|previous| previous >= row.id.as_str())
                    || supervised_ids.contains(row.id.as_str())
                    || ood_ids.contains(row.id.as_str())
                    || supervised_groups.contains_key(row.pair_id.as_str())
                    || ood_groups.contains(row.pair_id.as_str())
                    || !ids.insert(row.id.as_str())
                    || !feature_identities.insert(feature_identity(&row.text))
                {
                    return Err(MlError::InvalidDataset(
                        "contrast-test contains an overlapping id, pair, or feature-equivalent text"
                            .into(),
                    ));
                }
                previous_id = Some(&row.id);
                Ok(OpenSetContrastExample {
                    id: row.id.clone(),
                    pair_id: row.pair_id.clone(),
                    variant: row.variant,
                    label: row.label.clone(),
                    text: row.text.clone(),
                })
            })
            .collect::<Result<Vec<_>, MlError>>()?,
    };
    dataset.validate_contract()
}

fn validate_plan_ood_rows(
    development: &[OodPlanRow],
    test: &[OodPlanRow],
    supervised_ids: &HashSet<&str>,
    supervised_groups: &HashMap<&str, (&str, PartitionKind)>,
    feature_identities: &mut HashSet<String>,
) -> Result<(), MlError> {
    let mut ood_ids = HashSet::new();
    let mut development_families = HashSet::new();
    let mut test_families = HashSet::new();
    let mut development_domains = HashSet::new();
    let mut test_domains = HashSet::new();
    for (population, rows, families, domains) in [
        (
            "OOD development",
            development,
            &mut development_families,
            &mut development_domains,
        ),
        ("OOD test", test, &mut test_families, &mut test_domains),
    ] {
        let mut previous_id: Option<&str> = None;
        for (index, row) in rows.iter().enumerate() {
            validate_identifier(&row.id, "OOD plan id", index + 1)?;
            validate_identifier(&row.family_id, "OOD plan family id", index + 1)?;
            validate_identifier(&row.domain_group, "OOD plan domain group", index + 1)?;
            validate_text(&row.text, index + 1, population)?;
            if previous_id.is_some_and(|previous| previous >= row.id.as_str())
                || supervised_ids.contains(row.id.as_str())
                || !ood_ids.insert(row.id.as_str())
                || supervised_groups.contains_key(row.family_id.as_str())
                || supervised_groups.contains_key(row.domain_group.as_str())
                || !feature_identities.insert(feature_identity(&row.text))
            {
                return Err(MlError::InvalidDataset(format!(
                    "{population} contains an overlapping id, family, or feature-equivalent text"
                )));
            }
            previous_id = Some(&row.id);
            families.insert(row.family_id.as_str());
            domains.insert(row.domain_group.as_str());
        }
    }
    if development_families
        .iter()
        .any(|family| test_families.contains(family))
    {
        return Err(MlError::InvalidDataset(
            "OOD development and OOD test must be disjoint by domain family".into(),
        ));
    }
    if development_domains
        .iter()
        .any(|domain| test_domains.contains(domain))
    {
        return Err(MlError::InvalidDataset(
            "OOD development and OOD test must be disjoint by broader domain group".into(),
        ));
    }
    let as_dataset = |rows: &[OodPlanRow]| OpenSetOodDataset {
        examples: rows
            .iter()
            .map(|row| OpenSetOodExample {
                id: row.id.clone(),
                family_id: row.family_id.clone(),
                domain_group: row.domain_group.clone(),
                stratum: row.stratum,
                text: row.text.clone(),
            })
            .collect(),
    };
    as_dataset(development).validate_contract()?;
    as_dataset(test).validate_contract()?;
    Ok(())
}

#[derive(Debug, Clone)]
pub struct SplitPlan {
    manifest: SplitPlanManifest,
    train: Vec<GroupedExample>,
    development: Vec<GroupedExample>,
    calibration: Vec<GroupedExample>,
    id_test: Vec<GroupedExample>,
}

struct TrainingPartition<'a> {
    examples: &'a [GroupedExample],
    dataset_sha256: &'a str,
    split_plan_sha256: String,
}

#[derive(Clone, Copy)]
struct DevelopmentPartition<'a>(&'a [GroupedExample]);

#[derive(Clone, Copy)]
struct CalibrationPartition<'a>(&'a [GroupedExample]);

#[derive(Clone, Copy)]
struct OodDevelopmentPartition<'a>(&'a OpenSetOodDataset);

#[derive(Clone, Copy)]
struct ContrastTestPartition<'a>(&'a OpenSetContrastDataset);

impl<'a> TrainingPartition<'a> {
    fn examples(&self) -> &'a [GroupedExample] {
        self.examples
    }
}

impl<'a> DevelopmentPartition<'a> {
    fn examples(self) -> &'a [GroupedExample] {
        self.0
    }
}

impl<'a> CalibrationPartition<'a> {
    fn examples(self) -> &'a [GroupedExample] {
        self.0
    }
}

impl<'a> OodDevelopmentPartition<'a> {
    fn dataset(self) -> &'a OpenSetOodDataset {
        self.0
    }
}

impl<'a> ContrastTestPartition<'a> {
    fn dataset(self) -> &'a OpenSetContrastDataset {
        self.0
    }
}

fn plan_ood_rows(dataset: &OpenSetOodDataset) -> Vec<OodPlanRow> {
    let mut rows = dataset
        .examples()
        .iter()
        .map(|example| OodPlanRow {
            id: example.id.clone(),
            family_id: example.family_id.clone(),
            domain_group: example.domain_group.clone(),
            stratum: example.stratum,
            text: example.text.clone(),
        })
        .collect::<Vec<_>>();
    rows.sort_by(|left, right| left.id.cmp(&right.id));
    rows
}

fn plan_contrast_rows(dataset: &OpenSetContrastDataset) -> Vec<ContrastPlanRow> {
    let mut rows = dataset
        .examples()
        .iter()
        .map(|example| ContrastPlanRow {
            id: example.id.clone(),
            pair_id: example.pair_id.clone(),
            variant: example.variant,
            label: example.label.clone(),
            text: example.text.clone(),
        })
        .collect::<Vec<_>>();
    rows.sort_by(|left, right| left.id.cmp(&right.id));
    rows
}

impl SplitPlan {
    pub fn build(
        dataset: &GroupedDataset,
        ood_development: &OpenSetOodDataset,
        ood_test: &OpenSetOodDataset,
        contrast_test: &OpenSetContrastDataset,
        seed: u64,
    ) -> Result<Self, MlError> {
        dataset.validate_partition_support()?;
        reject_cross_dataset_overlap(dataset, ood_development, ood_test, contrast_test)?;
        let mut grouped: BTreeMap<String, BTreeMap<String, Vec<GroupedExample>>> = BTreeMap::new();
        for example in dataset.examples() {
            grouped
                .entry(example.label.clone())
                .or_default()
                .entry(example.group_id.clone())
                .or_default()
                .push(example.clone());
        }

        let mut assignments = Vec::with_capacity(dataset.examples().len());
        let mut train = Vec::new();
        let mut development = Vec::new();
        let mut calibration = Vec::new();
        let mut id_test = Vec::new();
        for (label, groups) in grouped {
            let mut groups = groups.into_iter().collect::<Vec<_>>();
            groups.sort_by(|(left, _), (right, _)| {
                group_split_hash(&label, left, seed)
                    .cmp(&group_split_hash(&label, right, seed))
                    .then_with(|| left.cmp(right))
            });
            if groups.len() < 4 {
                return Err(MlError::InvalidDataset(format!(
                    "label `{label}` cannot populate four group-disjoint partitions"
                )));
            }
            let evaluation_groups = evaluation_group_quota(groups.len());
            for (group_index, (_, mut examples)) in groups.into_iter().enumerate() {
                examples.sort_by(|left, right| left.id.cmp(&right.id));
                let partition = if group_index < evaluation_groups {
                    PartitionKind::IdTest
                } else if group_index < evaluation_groups * 2 {
                    PartitionKind::Calibration
                } else if group_index < evaluation_groups * 3 {
                    PartitionKind::Development
                } else {
                    PartitionKind::Train
                };
                for example in examples {
                    assignments.push(SplitAssignment {
                        id: example.id.clone(),
                        group_id: example.group_id.clone(),
                        label: example.label.clone(),
                        text: example.text.clone(),
                        partition,
                    });
                    match partition {
                        PartitionKind::Train => train.push(example),
                        PartitionKind::Development => development.push(example),
                        PartitionKind::Calibration => calibration.push(example),
                        PartitionKind::IdTest => id_test.push(example),
                    }
                }
            }
        }
        assignments.sort_by(|left, right| left.id.cmp(&right.id));
        for partition in [&mut train, &mut development, &mut calibration, &mut id_test] {
            partition.sort_by(|left, right| left.id.cmp(&right.id));
        }
        let plan = Self {
            manifest: SplitPlanManifest {
                schema_version: OPEN_SET_SCHEMA_VERSION,
                strategy: "group-stratified-scaled-four-way-v3".into(),
                seed,
                dataset_sha256: dataset.fingerprint_sha256(),
                assignments,
                ood_development: plan_ood_rows(ood_development),
                ood_test: plan_ood_rows(ood_test),
                contrast_test: plan_contrast_rows(contrast_test),
            },
            train,
            development,
            calibration,
            id_test,
        };
        plan.validate()?;
        Ok(plan)
    }

    pub fn train(&self) -> &[GroupedExample] {
        &self.train
    }

    pub fn development(&self) -> &[GroupedExample] {
        &self.development
    }

    pub fn calibration(&self) -> &[GroupedExample] {
        &self.calibration
    }

    pub fn id_test(&self) -> &[GroupedExample] {
        &self.id_test
    }

    pub fn manifest(&self) -> &SplitPlanManifest {
        &self.manifest
    }

    pub fn manifest_sha256(&self) -> Result<String, MlError> {
        Ok(sha256_hex(&canonical_json(&self.manifest)?))
    }

    fn training_partition(&self) -> Result<TrainingPartition<'_>, MlError> {
        Ok(TrainingPartition {
            examples: &self.train,
            dataset_sha256: &self.manifest.dataset_sha256,
            split_plan_sha256: self.manifest_sha256()?,
        })
    }

    fn development_partition(&self) -> DevelopmentPartition<'_> {
        DevelopmentPartition(&self.development)
    }

    fn calibration_partition(&self) -> CalibrationPartition<'_> {
        CalibrationPartition(&self.calibration)
    }

    fn validate(&self) -> Result<(), MlError> {
        self.manifest.validate_contract()?;
        let partitions = [
            (PartitionKind::Train, self.train()),
            (PartitionKind::Development, self.development()),
            (PartitionKind::Calibration, self.calibration()),
            (PartitionKind::IdTest, self.id_test()),
        ];
        let mut seen_ids = HashSet::new();
        let mut expected_assignments = Vec::with_capacity(self.manifest.assignments.len());
        let mut group_partition: HashMap<&str, PartitionKind> = HashMap::new();
        let mut labels_by_partition: BTreeMap<PartitionKind, BTreeSet<&str>> = BTreeMap::new();
        for (kind, examples) in partitions {
            if examples.is_empty() {
                return Err(MlError::InvalidDataset(format!(
                    "partition {kind:?} is empty"
                )));
            }
            for example in examples {
                if !seen_ids.insert(example.id.as_str()) {
                    return Err(MlError::InvalidDataset(format!(
                        "example `{}` crosses partitions",
                        example.id
                    )));
                }
                if let Some(previous) = group_partition.insert(&example.group_id, kind) {
                    if previous != kind {
                        return Err(MlError::InvalidDataset(format!(
                            "group `{}` crosses {previous:?} and {kind:?}",
                            example.group_id
                        )));
                    }
                }
                labels_by_partition
                    .entry(kind)
                    .or_default()
                    .insert(&example.label);
                expected_assignments.push(SplitAssignment {
                    id: example.id.clone(),
                    group_id: example.group_id.clone(),
                    label: example.label.clone(),
                    text: example.text.clone(),
                    partition: kind,
                });
            }
        }
        let expected_labels = labels_by_partition
            .get(&PartitionKind::Train)
            .cloned()
            .unwrap_or_default();
        if labels_by_partition
            .values()
            .any(|labels| *labels != expected_labels)
        {
            return Err(MlError::InvalidDataset(
                "all four partitions must contain the same labels".into(),
            ));
        }
        expected_assignments.sort_by(|left, right| left.id.cmp(&right.id));
        if self.manifest.assignments != expected_assignments {
            return Err(MlError::InvalidDataset(
                "split assignments do not exactly match the partition examples".into(),
            ));
        }
        Ok(())
    }
}

fn evaluation_group_quota(group_count: usize) -> usize {
    ((group_count + 5) / 10).max(1).min((group_count - 1) / 3)
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct DevelopmentSelectionConfig {
    pub max_features_candidates: Vec<usize>,
    pub l2_penalty_candidates: Vec<f64>,
    pub macro_f1_tolerance: f64,
}

impl Default for DevelopmentSelectionConfig {
    fn default() -> Self {
        Self {
            max_features_candidates: vec![512, 1_024, 2_048],
            l2_penalty_candidates: vec![0.0001, 0.0005, 0.002],
            macro_f1_tolerance: 0.005,
        }
    }
}

impl DevelopmentSelectionConfig {
    fn validate(&self) -> Result<(), MlError> {
        if self.max_features_candidates.is_empty()
            || self.max_features_candidates.len() > 8
            || self.l2_penalty_candidates.is_empty()
            || self.l2_penalty_candidates.len() > 8
            || self
                .max_features_candidates
                .windows(2)
                .any(|pair| pair[0] >= pair[1])
            || self
                .l2_penalty_candidates
                .windows(2)
                .any(|pair| pair[0] >= pair[1])
            || self
                .max_features_candidates
                .iter()
                .any(|candidate| !(32..=100_000).contains(candidate))
            || self
                .l2_penalty_candidates
                .iter()
                .any(|candidate| !candidate.is_finite() || !(0.0..=1.0).contains(candidate))
            || !self.macro_f1_tolerance.is_finite()
            || !(0.0..=0.05).contains(&self.macro_f1_tolerance)
        {
            return Err(MlError::InvalidConfiguration(
                "the development-only model-selection grid is invalid".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct OpenSetTrainingConfig {
    pub seed: u64,
    pub epochs: usize,
    pub learning_rate: f64,
    pub l2_penalty: f64,
    pub vectorizer: VectorizerConfig,
    pub development_selection: DevelopmentSelectionConfig,
}

impl Default for OpenSetTrainingConfig {
    fn default() -> Self {
        Self {
            seed: 4_043_100_207_104_787,
            epochs: 600,
            learning_rate: 0.8,
            l2_penalty: 0.0005,
            vectorizer: VectorizerConfig::default(),
            development_selection: DevelopmentSelectionConfig::default(),
        }
    }
}

impl OpenSetTrainingConfig {
    fn validate(&self) -> Result<(), MlError> {
        if self.seed > 9_007_199_254_740_991 {
            return Err(MlError::InvalidConfiguration(
                "the open-set seed exceeds the JSON-safe integer ceiling".into(),
            ));
        }
        if !(1..=10_000).contains(&self.epochs) {
            return Err(MlError::InvalidConfiguration(
                "open-set epochs must be between 1 and 10000".into(),
            ));
        }
        if !self.learning_rate.is_finite() || !(0.000_001..=10.0).contains(&self.learning_rate) {
            return Err(MlError::InvalidConfiguration(
                "open-set learning_rate must be finite and between 0.000001 and 10".into(),
            ));
        }
        if !self.l2_penalty.is_finite() || !(0.0..=1.0).contains(&self.l2_penalty) {
            return Err(MlError::InvalidConfiguration(
                "open-set l2_penalty must be finite and between 0 and 1".into(),
            ));
        }
        validate_vectorizer_config(&self.vectorizer)?;
        self.development_selection.validate()?;
        if !self
            .development_selection
            .max_features_candidates
            .contains(&self.vectorizer.max_features)
            || !self
                .development_selection
                .l2_penalty_candidates
                .contains(&self.l2_penalty)
        {
            return Err(MlError::InvalidConfiguration(
                "the serialized training configuration must identify one candidate from its development grid"
                    .into(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct OpenSetVectorizer {
    pub config: VectorizerConfig,
    pub vocabulary: Vec<String>,
    pub inverse_document_frequency: Vec<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct OpenSetModelV3 {
    pub schema_version: u32,
    pub model_kind: String,
    pub model_version: String,
    pub dataset_sha256: String,
    pub split_plan_sha256: String,
    pub training_config: OpenSetTrainingConfig,
    pub labels: Vec<String>,
    pub vectorizer: OpenSetVectorizer,
    pub weights: Vec<Vec<f64>>,
    pub biases: Vec<f64>,
}

impl OpenSetModelV3 {
    pub fn validate(&self) -> Result<(), MlError> {
        if self.schema_version != OPEN_SET_SCHEMA_VERSION
            || self.model_kind != OPEN_SET_MODEL_KIND
            || self.model_version != OPEN_SET_MODEL_VERSION
            || !valid_sha256(&self.dataset_sha256)
            || !valid_sha256(&self.split_plan_sha256)
        {
            return Err(MlError::InvalidModel(
                "the open-set model identity is invalid".into(),
            ));
        }
        self.training_config.validate()?;
        self.vectorizer.validate()?;
        if self.training_config.vectorizer != self.vectorizer.config {
            return Err(MlError::InvalidModel(
                "open-set training and serialized vectorizer configs differ".into(),
            ));
        }
        if self.labels.len() < 2
            || self.labels.len() > MAX_CLASSES
            || self.labels.len() != self.weights.len()
            || self.labels.len() != self.biases.len()
            || self.labels.iter().collect::<HashSet<_>>().len() != self.labels.len()
        {
            return Err(MlError::InvalidModel(
                "open-set labels, weights, and biases are not aligned".into(),
            ));
        }
        for label in &self.labels {
            if label.is_empty()
                || !label
                    .chars()
                    .all(|character| character.is_ascii_lowercase() || character == '-')
            {
                return Err(MlError::InvalidModel(
                    "the open-set model contains an invalid label".into(),
                ));
            }
        }
        for (row, bias) in self.weights.iter().zip(&self.biases) {
            if row.len() != self.vectorizer.vocabulary.len()
                || row
                    .iter()
                    .any(|weight| !weight.is_finite() || weight.abs() > MAX_PARAMETER_MAGNITUDE)
                || !bias.is_finite()
                || bias.abs() > MAX_PARAMETER_MAGNITUDE
            {
                return Err(MlError::InvalidModel(
                    "open-set parameters must be finite, bounded, and rectangular".into(),
                ));
            }
        }
        Ok(())
    }
}

impl OpenSetVectorizer {
    fn fit(examples: &[GroupedExample], config: VectorizerConfig) -> Result<Self, MlError> {
        validate_vectorizer_config(&config)?;
        let mut document_frequency: HashMap<String, usize> = HashMap::new();
        for example in examples {
            for term in extract_terms(&example.text, &config)
                .into_iter()
                .collect::<HashSet<_>>()
            {
                *document_frequency.entry(term).or_insert(0) += 1;
            }
        }
        let mut candidates = document_frequency
            .into_iter()
            .filter(|(_, count)| *count >= config.min_document_frequency)
            .collect::<Vec<_>>();
        candidates.sort_by(|(left_term, left_count), (right_term, right_count)| {
            right_count
                .cmp(left_count)
                .then_with(|| left_term.cmp(right_term))
        });
        candidates.truncate(config.max_features);
        if candidates.is_empty() {
            return Err(MlError::InvalidDataset(
                "the open-set training partition produced an empty vocabulary".into(),
            ));
        }
        let document_count = examples.len() as f64;
        let (vocabulary, inverse_document_frequency) = candidates
            .into_iter()
            .map(|(term, count)| {
                let idf = quantize(((1.0 + document_count) / (1.0 + count as f64)).ln() + 1.0);
                (term, idf)
            })
            .unzip();
        Ok(Self {
            config,
            vocabulary,
            inverse_document_frequency,
        })
    }

    fn validate(&self) -> Result<(), MlError> {
        validate_vectorizer_config(&self.config).map_err(|error| {
            MlError::InvalidModel(format!(
                "the open-set vectorizer config is invalid: {error}"
            ))
        })?;
        if self.vocabulary.is_empty()
            || self.vocabulary.len() != self.inverse_document_frequency.len()
            || self.vocabulary.len() > self.config.max_features
            || self.vocabulary.iter().collect::<HashSet<_>>().len() != self.vocabulary.len()
            || self
                .inverse_document_frequency
                .iter()
                .any(|value| !value.is_finite() || !(1.0..=MAX_IDF).contains(value))
        {
            return Err(MlError::InvalidModel(
                "the open-set vectorizer violates its shape contract".into(),
            ));
        }
        Ok(())
    }
}

fn fit_model(
    training: &TrainingPartition<'_>,
    config: OpenSetTrainingConfig,
) -> Result<OpenSetModelV3, MlError> {
    config.validate()?;
    let labels = training
        .examples()
        .iter()
        .map(|example| example.label.clone())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    if labels.len() < 2 {
        return Err(MlError::InvalidDataset(
            "open-set training requires at least two labels".into(),
        ));
    }
    let label_index = labels
        .iter()
        .enumerate()
        .map(|(index, label)| (label.as_str(), index))
        .collect::<HashMap<_, _>>();
    let vectorizer = OpenSetVectorizer::fit(training.examples(), config.vectorizer.clone())?;
    let feature_index = build_feature_index(&vectorizer.vocabulary);
    let features = training
        .examples()
        .iter()
        .map(|example| transform(&vectorizer, &feature_index, &example.text))
        .collect::<Vec<_>>();
    let targets = training
        .examples()
        .iter()
        .map(|example| label_index[example.label.as_str()])
        .collect::<Vec<_>>();
    let mut weights = vec![vec![0.0; vectorizer.vocabulary.len()]; labels.len()];
    let mut biases = vec![0.0; labels.len()];
    let sample_count = training.examples().len() as f64;
    for epoch in 0..config.epochs {
        let mut weight_gradient = vec![vec![0.0; vectorizer.vocabulary.len()]; labels.len()];
        let mut bias_gradient = vec![0.0; labels.len()];
        for (row, target) in features.iter().zip(&targets) {
            let probabilities = softmax(&logits_for(row, &weights, &biases), 1.0);
            for class in 0..labels.len() {
                let error = probabilities[class] - usize::from(class == *target) as f64;
                bias_gradient[class] += error;
                for (feature, value) in row {
                    weight_gradient[class][*feature] += error * value;
                }
            }
        }
        let learning_rate = config.learning_rate / (1.0 + epoch as f64 * 0.025).sqrt();
        for class in 0..labels.len() {
            biases[class] =
                quantize(biases[class] - learning_rate * bias_gradient[class] / sample_count);
            for feature in 0..vectorizer.vocabulary.len() {
                let gradient = weight_gradient[class][feature] / sample_count
                    + config.l2_penalty * weights[class][feature];
                weights[class][feature] =
                    quantize(weights[class][feature] - learning_rate * gradient);
            }
        }
    }
    let model = OpenSetModelV3 {
        schema_version: OPEN_SET_SCHEMA_VERSION,
        model_kind: OPEN_SET_MODEL_KIND.into(),
        model_version: OPEN_SET_MODEL_VERSION.into(),
        dataset_sha256: training.dataset_sha256.to_owned(),
        split_plan_sha256: training.split_plan_sha256.clone(),
        training_config: config,
        labels,
        vectorizer,
        weights,
        biases,
    };
    model.validate()?;
    Ok(model)
}

fn development_candidate_is_better(
    candidate: &DevelopmentCandidateMetrics,
    current: &DevelopmentCandidateMetrics,
    macro_f1_tolerance: f64,
) -> bool {
    let macro_f1_difference = candidate.macro_f1 - current.macro_f1;
    if macro_f1_difference > macro_f1_tolerance {
        return true;
    }
    if macro_f1_difference < -macro_f1_tolerance {
        return false;
    }
    current
        .max_features
        .cmp(&candidate.max_features)
        .then_with(|| candidate.l2_penalty.total_cmp(&current.l2_penalty))
        .then_with(|| candidate.accuracy.total_cmp(&current.accuracy))
        .then_with(|| {
            current
                .negative_log_likelihood
                .total_cmp(&candidate.negative_log_likelihood)
        })
        .then_with(|| {
            current
                .multiclass_brier
                .total_cmp(&candidate.multiclass_brier)
        })
        .is_gt()
}

fn evaluate_development_candidate(
    model: &OpenSetModelV3,
    development: DevelopmentPartition<'_>,
) -> Result<IdEvaluationV3, MlError> {
    let examples = development.examples();
    let policy = OpenSetPolicyV3 {
        schema_version: OPEN_SET_SCHEMA_VERSION,
        model_version: model.model_version.clone(),
        dataset_sha256: model.dataset_sha256.clone(),
        split_plan_sha256: model.split_plan_sha256.clone(),
        temperature: 1.0,
        minimum_confidence: 0.0,
        minimum_probability_margin: 0.0,
        temperature_source: "calibration-partition-temperature-scaling-v3".into(),
        threshold_source: "fixed-development-plus-ood-development-grid-v3".into(),
        calibration_example_count: 1,
        development_example_count: examples.len(),
        ood_development_example_count: 1,
    };
    evaluate_id(&CompiledModel::new(model.clone(), policy)?, examples)
}

fn select_model_on_development(
    training: &TrainingPartition<'_>,
    development: DevelopmentPartition<'_>,
    base_config: OpenSetTrainingConfig,
) -> Result<(OpenSetModelV3, DevelopmentSelectionReport), MlError> {
    base_config.validate()?;
    let mut candidates = Vec::new();
    let mut selected: Option<(usize, OpenSetModelV3)> = None;
    for max_features in &base_config.development_selection.max_features_candidates {
        for l2_penalty in &base_config.development_selection.l2_penalty_candidates {
            let mut candidate_config = base_config.clone();
            candidate_config.vectorizer.max_features = *max_features;
            candidate_config.l2_penalty = *l2_penalty;
            let model = fit_model(training, candidate_config)?;
            let evaluation = evaluate_development_candidate(&model, development)?;
            let metrics = DevelopmentCandidateMetrics {
                max_features: *max_features,
                l2_penalty: *l2_penalty,
                accuracy: evaluation.accuracy,
                macro_f1: evaluation.macro_f1,
                negative_log_likelihood: evaluation.calibration.negative_log_likelihood,
                multiclass_brier: evaluation.calibration.multiclass_brier,
            };
            let candidate_index = candidates.len();
            let replace = match selected.as_ref() {
                Some((selected_index, _)) => development_candidate_is_better(
                    &metrics,
                    &candidates[*selected_index],
                    base_config.development_selection.macro_f1_tolerance,
                ),
                None => true,
            };
            candidates.push(metrics);
            if replace {
                selected = Some((candidate_index, model));
            }
        }
    }
    let (selected_index, model) = selected.ok_or_else(|| {
        MlError::InvalidConfiguration("the development-only model-selection grid is empty".into())
    })?;
    let family_count = |examples: &[GroupedExample]| {
        examples
            .iter()
            .map(|example| example.group_id.as_str())
            .collect::<HashSet<_>>()
            .len()
    };
    let report = DevelopmentSelectionReport {
        strategy: "train-fit-development-f1-epsilon-parsimony-accuracy-nll-brier-v3".into(),
        seed: base_config.seed,
        macro_f1_tolerance: base_config.development_selection.macro_f1_tolerance,
        training_example_count: training.examples().len(),
        training_family_count: family_count(training.examples()),
        development_example_count: development.examples().len(),
        development_family_count: family_count(development.examples()),
        candidates,
        selected_index,
        inputs: vec![SelectionDataRole::Train, SelectionDataRole::Development],
    };
    Ok((model, report))
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct OpenSetPolicyV3 {
    pub schema_version: u32,
    pub model_version: String,
    pub dataset_sha256: String,
    pub split_plan_sha256: String,
    pub temperature: f64,
    pub minimum_confidence: f64,
    pub minimum_probability_margin: f64,
    pub temperature_source: String,
    pub threshold_source: String,
    pub calibration_example_count: usize,
    pub development_example_count: usize,
    pub ood_development_example_count: usize,
}

impl OpenSetPolicyV3 {
    fn validate_against(&self, model: &OpenSetModelV3) -> Result<(), MlError> {
        if self.schema_version != OPEN_SET_SCHEMA_VERSION
            || self.model_version != model.model_version
            || self.dataset_sha256 != model.dataset_sha256
            || self.split_plan_sha256 != model.split_plan_sha256
            || self.temperature_source != "calibration-partition-temperature-scaling-v3"
            || self.threshold_source != "fixed-development-plus-ood-development-grid-v3"
        {
            return Err(MlError::InvalidModel(
                "the open-set policy does not match its model and provenance".into(),
            ));
        }
        if !self.temperature.is_finite() || !(0.05..=20.0).contains(&self.temperature) {
            return Err(MlError::InvalidModel(
                "the open-set temperature must be finite and between 0.05 and 20".into(),
            ));
        }
        if !self.minimum_confidence.is_finite()
            || !(0.0..=1.0).contains(&self.minimum_confidence)
            || !self.minimum_probability_margin.is_finite()
            || !(0.0..=1.0).contains(&self.minimum_probability_margin)
            || self.calibration_example_count == 0
            || self.development_example_count == 0
            || self.ood_development_example_count == 0
        {
            return Err(MlError::InvalidModel(
                "the open-set operating policy contains invalid thresholds or provenance counts"
                    .into(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ContrastiveContribution {
    pub feature: String,
    pub value: f64,
    pub top_weight: f64,
    pub runner_up_weight: f64,
    pub contribution: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ContrastiveExplanation {
    pub top_label: String,
    pub runner_up_label: String,
    pub bias_difference: f64,
    pub feature_contribution_sum: f64,
    pub reconstructed_logit_margin: f64,
    pub top_contributions: Vec<ContrastiveContribution>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct OpenSetPrediction {
    pub label: String,
    pub runner_up_label: String,
    pub accepted: bool,
    pub confidence: f64,
    pub probability_margin: f64,
    pub logit_margin: f64,
    pub probabilities: BTreeMap<String, f64>,
    pub explanation: ContrastiveExplanation,
}

#[derive(Debug, Clone)]
pub struct CompiledModel {
    model: OpenSetModelV3,
    policy: OpenSetPolicyV3,
    feature_index: HashMap<String, usize>,
}

impl CompiledModel {
    pub fn new(model: OpenSetModelV3, policy: OpenSetPolicyV3) -> Result<Self, MlError> {
        model.validate()?;
        policy.validate_against(&model)?;
        let feature_index = build_feature_index(&model.vectorizer.vocabulary);
        Ok(Self {
            model,
            policy,
            feature_index,
        })
    }

    pub fn model(&self) -> &OpenSetModelV3 {
        &self.model
    }

    pub fn policy(&self) -> &OpenSetPolicyV3 {
        &self.policy
    }

    pub fn predict(&self, text: &str) -> OpenSetPrediction {
        let bounded_text = if text.chars().nth(crate::MAX_INPUT_CHARS).is_some() {
            ""
        } else {
            text
        };
        let features = transform(&self.model.vectorizer, &self.feature_index, bounded_text);
        let logits = logits_for(&features, &self.model.weights, &self.model.biases);
        prediction_from_scores(&self.model, &self.policy, &features, &logits)
    }

    pub fn predict_batch<'a, I>(&self, inputs: I) -> Vec<OpenSetPrediction>
    where
        I: IntoIterator<Item = &'a str>,
    {
        inputs.into_iter().map(|text| self.predict(text)).collect()
    }
}

fn prediction_from_scores(
    model: &OpenSetModelV3,
    policy: &OpenSetPolicyV3,
    features: &[(usize, f64)],
    logits: &[f64],
) -> OpenSetPrediction {
    let probabilities = softmax(logits, policy.temperature);
    let mut ranking = probabilities
        .iter()
        .copied()
        .enumerate()
        .collect::<Vec<_>>();
    ranking.sort_by(|(left_index, left), (right_index, right)| {
        right
            .total_cmp(left)
            .then_with(|| left_index.cmp(right_index))
    });
    let (top_index, confidence) = ranking[0];
    let (runner_up_index, runner_up_probability) = ranking[1];
    let probability_margin = confidence - runner_up_probability;
    let logit_margin = logits[top_index] - logits[runner_up_index];
    let accepted = !features.is_empty()
        && confidence >= policy.minimum_confidence
        && probability_margin >= policy.minimum_probability_margin;
    let bias_difference = model.biases[top_index] - model.biases[runner_up_index];
    let mut all_contributions = features
        .iter()
        .map(|(feature, value)| {
            let top_weight = model.weights[top_index][*feature];
            let runner_up_weight = model.weights[runner_up_index][*feature];
            ContrastiveContribution {
                feature: model.vectorizer.vocabulary[*feature].clone(),
                value: *value,
                top_weight,
                runner_up_weight,
                contribution: *value * (top_weight - runner_up_weight),
            }
        })
        .collect::<Vec<_>>();
    let feature_contribution_sum = all_contributions
        .iter()
        .map(|contribution| contribution.contribution)
        .sum::<f64>();
    all_contributions.sort_by(|left, right| {
        right
            .contribution
            .abs()
            .total_cmp(&left.contribution.abs())
            .then_with(|| left.feature.cmp(&right.feature))
    });
    all_contributions.truncate(8);
    OpenSetPrediction {
        label: model.labels[top_index].clone(),
        runner_up_label: model.labels[runner_up_index].clone(),
        accepted,
        confidence,
        probability_margin,
        logit_margin,
        probabilities: model.labels.iter().cloned().zip(probabilities).collect(),
        explanation: ContrastiveExplanation {
            top_label: model.labels[top_index].clone(),
            runner_up_label: model.labels[runner_up_index].clone(),
            bias_difference,
            feature_contribution_sum,
            reconstructed_logit_margin: bias_difference + feature_contribution_sum,
            top_contributions: all_contributions,
        },
    }
}

fn calibrate_temperature(
    model: &OpenSetModelV3,
    calibration: CalibrationPartition<'_>,
) -> Result<f64, MlError> {
    let examples = calibration.examples();
    if examples.is_empty() {
        return Err(MlError::InvalidDataset(
            "temperature scaling requires a non-empty calibration partition".into(),
        ));
    }
    let feature_index = build_feature_index(&model.vectorizer.vocabulary);
    let label_index = model
        .labels
        .iter()
        .enumerate()
        .map(|(index, label)| (label.as_str(), index))
        .collect::<HashMap<_, _>>();
    let scored = examples
        .iter()
        .map(|example| {
            let target = label_index
                .get(example.label.as_str())
                .copied()
                .ok_or_else(|| {
                    MlError::InvalidDataset(format!(
                        "calibration label `{}` is absent from the model",
                        example.label
                    ))
                })?;
            let features = transform(&model.vectorizer, &feature_index, &example.text);
            Ok((logits_for(&features, &model.weights, &model.biases), target))
        })
        .collect::<Result<Vec<_>, MlError>>()?;

    let loss = |log_temperature: f64| {
        let temperature = log_temperature.exp();
        scored
            .iter()
            .map(|(logits, target)| -softmax(logits, temperature)[*target].max(1e-15).ln())
            .sum::<f64>()
            / scored.len() as f64
    };
    let mut left = 0.05_f64.ln();
    let mut right = 20.0_f64.ln();
    let golden = (5.0_f64.sqrt() - 1.0) / 2.0;
    let mut middle_left = right - golden * (right - left);
    let mut middle_right = left + golden * (right - left);
    let mut loss_left = loss(middle_left);
    let mut loss_right = loss(middle_right);
    for _ in 0..96 {
        if loss_left <= loss_right {
            right = middle_right;
            middle_right = middle_left;
            loss_right = loss_left;
            middle_left = right - golden * (right - left);
            loss_left = loss(middle_left);
        } else {
            left = middle_left;
            middle_left = middle_right;
            loss_left = loss_right;
            middle_right = left + golden * (right - left);
            loss_right = loss(middle_right);
        }
    }
    Ok(quantize(((left + right) / 2.0).exp().clamp(0.05, 20.0)))
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ThresholdSelection {
    pub strategy: String,
    pub evaluated_candidate_count: usize,
    pub feasible_candidate_count: usize,
    pub development_example_count: usize,
    pub ood_development_example_count: usize,
    pub minimum_development_selective_accuracy: f64,
    pub maximum_ood_development_coverage: f64,
    pub selected_confidence: f64,
    pub selected_probability_margin: f64,
    pub observed_development_coverage: f64,
    pub observed_development_selective_accuracy: f64,
    pub observed_ood_development_coverage: f64,
    pub inputs: Vec<SelectionDataRole>,
}

fn select_thresholds(
    model: &OpenSetModelV3,
    temperature: f64,
    development: DevelopmentPartition<'_>,
    ood_development: OodDevelopmentPartition<'_>,
) -> Result<ThresholdSelection, MlError> {
    let development = development.examples();
    let ood_development = ood_development.dataset();
    if development.is_empty() || ood_development.examples().is_empty() {
        return Err(MlError::InvalidDataset(
            "threshold selection requires development and OOD-development examples".into(),
        ));
    }
    let temporary_policy = OpenSetPolicyV3 {
        schema_version: OPEN_SET_SCHEMA_VERSION,
        model_version: model.model_version.clone(),
        dataset_sha256: model.dataset_sha256.clone(),
        split_plan_sha256: model.split_plan_sha256.clone(),
        temperature,
        minimum_confidence: 0.0,
        minimum_probability_margin: 0.0,
        temperature_source: "calibration-partition-temperature-scaling-v3".into(),
        threshold_source: "fixed-development-plus-ood-development-grid-v3".into(),
        calibration_example_count: 1,
        development_example_count: development.len(),
        ood_development_example_count: ood_development.examples().len(),
    };
    let runtime = CompiledModel::new(model.clone(), temporary_policy)?;
    let development_scores = development
        .iter()
        .map(|example| {
            let prediction = runtime.predict(&example.text);
            (
                prediction.accepted,
                prediction.label == example.label,
                prediction.confidence,
                prediction.probability_margin,
            )
        })
        .collect::<Vec<_>>();
    let ood_scores = ood_development
        .examples()
        .iter()
        .map(|example| {
            let prediction = runtime.predict(&example.text);
            (
                prediction.accepted,
                prediction.confidence,
                prediction.probability_margin,
            )
        })
        .collect::<Vec<_>>();

    const MIN_SELECTIVE_ACCURACY: f64 = 0.75;
    const MAX_OOD_COVERAGE: f64 = 0.10;
    #[derive(Clone, Copy)]
    struct Candidate {
        confidence: f64,
        margin: f64,
        development_coverage: f64,
        development_selective_accuracy: f64,
        ood_coverage: f64,
    }
    const CONFIDENCE_CANDIDATES: [f64; 7] = [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95];
    const MARGIN_CANDIDATES: [f64; 7] = [0.0, 0.20, 0.40, 0.60, 0.70, 0.80, 0.90];
    let mut best = None;
    let mut feasible_candidate_count = 0usize;
    for confidence in CONFIDENCE_CANDIDATES {
        for margin in MARGIN_CANDIDATES {
            let accepted = development_scores
                .iter()
                .filter(|(has_features, _, observed_confidence, observed_margin)| {
                    *has_features
                        && *observed_confidence >= confidence
                        && *observed_margin >= margin
                })
                .collect::<Vec<_>>();
            if accepted.is_empty() {
                continue;
            }
            let development_selective_accuracy = accepted
                .iter()
                .filter(|(_, correct, _, _)| *correct)
                .count() as f64
                / accepted.len() as f64;
            if development_selective_accuracy < MIN_SELECTIVE_ACCURACY {
                continue;
            }
            let ood_accepted = ood_scores
                .iter()
                .filter(|(has_features, observed_confidence, observed_margin)| {
                    *has_features
                        && *observed_confidence >= confidence
                        && *observed_margin >= margin
                })
                .count();
            let ood_coverage = ood_accepted as f64 / ood_scores.len() as f64;
            if ood_coverage > MAX_OOD_COVERAGE {
                continue;
            }
            feasible_candidate_count += 1;
            let candidate = Candidate {
                confidence,
                margin,
                development_coverage: accepted.len() as f64 / development_scores.len() as f64,
                development_selective_accuracy,
                ood_coverage,
            };
            let is_better = best.map_or(true, |current: Candidate| {
                candidate.development_coverage > current.development_coverage
                    || (candidate.development_coverage == current.development_coverage
                        && candidate.development_selective_accuracy
                            > current.development_selective_accuracy)
                    || (candidate.development_coverage == current.development_coverage
                        && candidate.development_selective_accuracy
                            == current.development_selective_accuracy
                        && candidate.ood_coverage < current.ood_coverage)
                    || (candidate.development_coverage == current.development_coverage
                        && candidate.development_selective_accuracy
                            == current.development_selective_accuracy
                        && candidate.ood_coverage == current.ood_coverage
                        && candidate.margin > current.margin)
                    || (candidate.development_coverage == current.development_coverage
                        && candidate.development_selective_accuracy
                            == current.development_selective_accuracy
                        && candidate.ood_coverage == current.ood_coverage
                        && candidate.margin == current.margin
                        && candidate.confidence > current.confidence)
            });
            if is_better {
                best = Some(candidate);
            }
        }
    }
    let best = best.ok_or_else(|| {
        MlError::InvalidConfiguration(
            "no development/OOD-development threshold satisfies the locked v3 policy".into(),
        )
    })?;
    Ok(ThresholdSelection {
        strategy: "fixed-development-plus-ood-development-grid-v3".into(),
        evaluated_candidate_count: CONFIDENCE_CANDIDATES.len() * MARGIN_CANDIDATES.len(),
        feasible_candidate_count,
        development_example_count: development.len(),
        ood_development_example_count: ood_development.examples().len(),
        minimum_development_selective_accuracy: MIN_SELECTIVE_ACCURACY,
        maximum_ood_development_coverage: MAX_OOD_COVERAGE,
        selected_confidence: best.confidence,
        selected_probability_margin: best.margin,
        observed_development_coverage: best.development_coverage,
        observed_development_selective_accuracy: best.development_selective_accuracy,
        observed_ood_development_coverage: best.ood_coverage,
        inputs: vec![
            SelectionDataRole::Development,
            SelectionDataRole::OodDevelopment,
        ],
    })
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct MetricEstimate {
    pub value: f64,
    pub lower_95: f64,
    pub upper_95: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct CalibrationMetrics {
    pub negative_log_likelihood: f64,
    pub multiclass_brier: f64,
    pub expected_calibration_error: f64,
    pub ece_bins: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct DevelopmentCandidateMetrics {
    pub max_features: usize,
    pub l2_penalty: f64,
    pub accuracy: f64,
    pub macro_f1: f64,
    pub negative_log_likelihood: f64,
    pub multiclass_brier: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct DevelopmentSelectionReport {
    pub strategy: String,
    pub seed: u64,
    pub macro_f1_tolerance: f64,
    pub training_example_count: usize,
    pub training_family_count: usize,
    pub development_example_count: usize,
    pub development_family_count: usize,
    pub candidates: Vec<DevelopmentCandidateMetrics>,
    pub selected_index: usize,
    pub inputs: Vec<SelectionDataRole>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct RiskCoveragePoint {
    pub accepted: usize,
    pub coverage: f64,
    pub risk: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct EvaluatedOpenSetPrediction {
    pub id: String,
    pub actual_label: String,
    pub predicted_label: String,
    pub correct: bool,
    pub accepted: bool,
    pub confidence: f64,
    pub probability_margin: f64,
    pub probabilities: BTreeMap<String, f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PerClassMetrics {
    pub label: String,
    pub support: usize,
    pub predicted: usize,
    pub true_positive: usize,
    pub precision: f64,
    pub recall: f64,
    pub f1: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct BaselineEvaluation {
    pub accuracy: f64,
    pub macro_f1: f64,
    pub confusion_matrix: Vec<Vec<usize>>,
    pub per_class: Vec<PerClassMetrics>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct BaselineReport {
    pub strategy: String,
    pub inputs: Vec<SelectionDataRole>,
    pub evaluation_partition: String,
    pub training_example_count: usize,
    pub training_family_count: usize,
    pub majority_label: String,
    pub majority: BaselineEvaluation,
    pub unigram_naive_bayes: BaselineEvaluation,
    pub learned_minus_unigram_accuracy: f64,
    pub learned_minus_unigram_macro_f1: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct IdEvaluationV3 {
    pub example_count: usize,
    pub accuracy: f64,
    pub macro_f1: f64,
    pub labels: Vec<String>,
    pub confusion_matrix: Vec<Vec<usize>>,
    pub per_class: Vec<PerClassMetrics>,
    pub coverage: f64,
    pub selective_accuracy: Option<f64>,
    pub calibration: CalibrationMetrics,
    pub aurc: f64,
    pub risk_coverage_curve: Vec<RiskCoveragePoint>,
    pub predictions: Vec<EvaluatedOpenSetPrediction>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct OodEvaluatedPrediction {
    pub id: String,
    pub family_id: String,
    pub domain_group: String,
    pub stratum: OodStratum,
    pub predicted_label: String,
    pub accepted: bool,
    pub confidence: f64,
    pub probability_margin: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct OodDiscriminationMetrics {
    pub auroc: f64,
    pub aupr_in_domain: f64,
    pub fpr_at_95_tpr: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct OodEvaluationV3 {
    pub example_count: usize,
    pub accepted_examples: usize,
    pub coverage: f64,
    pub discrimination: OodDiscriminationMetrics,
    pub by_stratum: BTreeMap<String, OodStratumEvaluation>,
    pub predictions: Vec<OodEvaluatedPrediction>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct OodStratumEvaluation {
    pub example_count: usize,
    pub accepted_examples: usize,
    pub coverage: f64,
    pub discrimination: OodDiscriminationMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ContrastEvaluatedPrediction {
    pub id: String,
    pub pair_id: String,
    pub variant: ContrastVariant,
    pub actual_label: String,
    pub predicted_label: String,
    pub correct: bool,
    pub accepted: bool,
    pub confidence: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ContrastEvaluationV3 {
    pub example_count: usize,
    pub pair_count: usize,
    pub accuracy: f64,
    pub macro_f1: f64,
    pub pair_accuracy: f64,
    pub prediction_flip_rate: f64,
    pub coverage: f64,
    pub confusion_matrix: Vec<Vec<usize>>,
    pub per_class: Vec<PerClassMetrics>,
    pub predictions: Vec<ContrastEvaluatedPrediction>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct BootstrapReport {
    pub strategy: String,
    pub seed: u64,
    pub resamples: usize,
    pub confidence_level: f64,
    pub id_accuracy: MetricEstimate,
    pub id_macro_f1: MetricEstimate,
    pub id_negative_log_likelihood: MetricEstimate,
    pub id_multiclass_brier: MetricEstimate,
    pub id_expected_calibration_error: MetricEstimate,
    pub id_aurc: MetricEstimate,
    pub ood_auroc: MetricEstimate,
    pub ood_aupr_in_domain: MetricEstimate,
    pub ood_fpr_at_95_tpr: MetricEstimate,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct OpenSetMetricsV3 {
    pub schema_version: u32,
    pub model_version: String,
    pub dataset_sha256: String,
    pub split_plan_sha256: String,
    pub ood_development_sha256: String,
    pub ood_test_sha256: String,
    pub contrast_test_sha256: String,
    pub partition_counts: BTreeMap<String, usize>,
    pub partition_family_counts: BTreeMap<String, usize>,
    pub paraphrases_per_family: usize,
    pub ood_domain_counts: BTreeMap<String, usize>,
    pub ood_stratum_counts: BTreeMap<String, BTreeMap<String, usize>>,
    pub development_selection: DevelopmentSelectionReport,
    pub threshold_selection: ThresholdSelection,
    pub uncalibrated_calibration_partition: CalibrationMetrics,
    pub calibrated_calibration_partition: CalibrationMetrics,
    pub id_test: IdEvaluationV3,
    pub baselines: BaselineReport,
    pub contrast_test: ContrastEvaluationV3,
    pub ood_test: OodEvaluationV3,
    pub bootstrap_95: BootstrapReport,
    pub limitations: Vec<String>,
}

fn evaluate_id(
    runtime: &CompiledModel,
    examples: &[GroupedExample],
) -> Result<IdEvaluationV3, MlError> {
    if examples.is_empty() {
        return Err(MlError::InvalidDataset(
            "ID evaluation requires at least one example".into(),
        ));
    }
    let predictions = examples
        .iter()
        .map(|example| {
            let prediction = runtime.predict(&example.text);
            EvaluatedOpenSetPrediction {
                id: example.id.clone(),
                actual_label: example.label.clone(),
                predicted_label: prediction.label.clone(),
                correct: prediction.label == example.label,
                accepted: prediction.accepted,
                confidence: prediction.confidence,
                probability_margin: prediction.probability_margin,
                probabilities: prediction.probabilities,
            }
        })
        .collect::<Vec<_>>();
    Ok(summarize_id_predictions(&runtime.model.labels, predictions))
}

fn summarize_id_predictions(
    labels: &[String],
    predictions: Vec<EvaluatedOpenSetPrediction>,
) -> IdEvaluationV3 {
    let example_count = predictions.len();
    let correct = predictions
        .iter()
        .filter(|prediction| prediction.correct)
        .count();
    let accepted = predictions
        .iter()
        .filter(|prediction| prediction.accepted)
        .collect::<Vec<_>>();
    let accepted_correct = accepted
        .iter()
        .filter(|prediction| prediction.correct)
        .count();
    let label_index = labels
        .iter()
        .enumerate()
        .map(|(index, label)| (label.as_str(), index))
        .collect::<HashMap<_, _>>();
    let mut confusion_matrix = vec![vec![0usize; labels.len()]; labels.len()];
    for prediction in &predictions {
        if let (Some(actual), Some(predicted)) = (
            label_index.get(prediction.actual_label.as_str()),
            label_index.get(prediction.predicted_label.as_str()),
        ) {
            confusion_matrix[*actual][*predicted] += 1;
        }
    }
    let per_class = labels
        .iter()
        .enumerate()
        .map(|(index, label)| {
            let support = confusion_matrix[index].iter().sum::<usize>();
            let predicted = confusion_matrix.iter().map(|row| row[index]).sum::<usize>();
            let true_positive = confusion_matrix[index][index];
            let precision = safe_ratio(true_positive as f64, predicted as f64);
            let recall = safe_ratio(true_positive as f64, support as f64);
            PerClassMetrics {
                label: label.clone(),
                support,
                predicted,
                true_positive,
                precision,
                recall,
                f1: safe_ratio(2.0 * precision * recall, precision + recall),
            }
        })
        .collect::<Vec<_>>();
    let calibration = calibration_metrics(&predictions, 10);
    let (risk_coverage_curve, aurc) = risk_coverage(&predictions);
    IdEvaluationV3 {
        example_count,
        accuracy: correct as f64 / example_count as f64,
        macro_f1: per_class.iter().map(|metrics| metrics.f1).sum::<f64>() / per_class.len() as f64,
        labels: labels.to_vec(),
        confusion_matrix,
        per_class,
        coverage: accepted.len() as f64 / example_count as f64,
        selective_accuracy: (!accepted.is_empty())
            .then_some(accepted_correct as f64 / accepted.len() as f64),
        calibration,
        aurc,
        risk_coverage_curve,
        predictions,
    }
}

fn summarize_baseline_predictions(
    labels: &[String],
    actual_and_predicted: &[(String, String)],
) -> BaselineEvaluation {
    let label_index = labels
        .iter()
        .enumerate()
        .map(|(index, label)| (label.as_str(), index))
        .collect::<HashMap<_, _>>();
    let mut confusion_matrix = vec![vec![0usize; labels.len()]; labels.len()];
    for (actual, predicted) in actual_and_predicted {
        confusion_matrix[label_index[actual.as_str()]][label_index[predicted.as_str()]] += 1;
    }
    let per_class = labels
        .iter()
        .enumerate()
        .map(|(index, label)| {
            let support = confusion_matrix[index].iter().sum::<usize>();
            let predicted = confusion_matrix.iter().map(|row| row[index]).sum::<usize>();
            let true_positive = confusion_matrix[index][index];
            let precision = safe_ratio(true_positive as f64, predicted as f64);
            let recall = safe_ratio(true_positive as f64, support as f64);
            PerClassMetrics {
                label: label.clone(),
                support,
                predicted,
                true_positive,
                precision,
                recall,
                f1: safe_ratio(2.0 * precision * recall, precision + recall),
            }
        })
        .collect::<Vec<_>>();
    let correct = actual_and_predicted
        .iter()
        .filter(|(actual, predicted)| actual == predicted)
        .count();
    BaselineEvaluation {
        accuracy: correct as f64 / actual_and_predicted.len() as f64,
        macro_f1: per_class.iter().map(|metrics| metrics.f1).sum::<f64>() / per_class.len() as f64,
        confusion_matrix,
        per_class,
    }
}

fn evaluate_baselines(
    training: &TrainingPartition<'_>,
    id_test: &[GroupedExample],
    labels: &[String],
    learned_accuracy: f64,
    learned_macro_f1: f64,
) -> BaselineReport {
    let mut document_counts: BTreeMap<&str, usize> = BTreeMap::new();
    let mut token_counts: BTreeMap<&str, HashMap<String, usize>> = BTreeMap::new();
    let mut token_totals: BTreeMap<&str, usize> = BTreeMap::new();
    let mut vocabulary = BTreeSet::new();
    for example in training.examples() {
        *document_counts.entry(&example.label).or_insert(0) += 1;
        for token in tokenize(&example.text) {
            vocabulary.insert(token.clone());
            *token_counts
                .entry(&example.label)
                .or_default()
                .entry(token)
                .or_insert(0) += 1;
            *token_totals.entry(&example.label).or_insert(0) += 1;
        }
    }
    let majority_label = labels
        .iter()
        .min_by(|left, right| {
            document_counts[right.as_str()]
                .cmp(&document_counts[left.as_str()])
                .then_with(|| left.cmp(right))
        })
        .expect("the fitted model always has labels")
        .clone();
    let majority_predictions = id_test
        .iter()
        .map(|example| (example.label.clone(), majority_label.clone()))
        .collect::<Vec<_>>();
    let training_count = training.examples().len() as f64;
    let vocabulary_size = vocabulary.len() as f64;
    let unigram_predictions = id_test
        .iter()
        .map(|example| {
            let tokens = tokenize(&example.text);
            let mut best_label = labels[0].as_str();
            let mut best_score = f64::NEG_INFINITY;
            for label in labels {
                let label_documents = document_counts[label.as_str()] as f64;
                let mut score =
                    ((label_documents + 1.0) / (training_count + labels.len() as f64)).ln();
                let denominator = token_totals[label.as_str()] as f64 + vocabulary_size;
                for token in &tokens {
                    if vocabulary.contains(token) {
                        let count = token_counts[label.as_str()]
                            .get(token)
                            .copied()
                            .unwrap_or(0) as f64;
                        score += ((count + 1.0) / denominator).ln();
                    }
                }
                if score > best_score {
                    best_score = score;
                    best_label = label;
                }
            }
            (example.label.clone(), best_label.to_owned())
        })
        .collect::<Vec<_>>();
    let majority = summarize_baseline_predictions(labels, &majority_predictions);
    let unigram_naive_bayes = summarize_baseline_predictions(labels, &unigram_predictions);
    BaselineReport {
        strategy: "training-only-majority-and-laplace-unigram-naive-bayes-v3".into(),
        inputs: vec![SelectionDataRole::Train],
        evaluation_partition: "id-test".into(),
        training_example_count: training.examples().len(),
        training_family_count: training
            .examples()
            .iter()
            .map(|example| example.group_id.as_str())
            .collect::<HashSet<_>>()
            .len(),
        majority_label,
        majority,
        learned_minus_unigram_accuracy: learned_accuracy - unigram_naive_bayes.accuracy,
        learned_minus_unigram_macro_f1: learned_macro_f1 - unigram_naive_bayes.macro_f1,
        unigram_naive_bayes,
    }
}

fn evaluate_contrast(
    runtime: &CompiledModel,
    contrast: ContrastTestPartition<'_>,
) -> Result<ContrastEvaluationV3, MlError> {
    let dataset = contrast.dataset();
    dataset.validate_contract()?;
    let predictions = dataset
        .examples()
        .iter()
        .map(|example| {
            let prediction = runtime.predict(&example.text);
            ContrastEvaluatedPrediction {
                id: example.id.clone(),
                pair_id: example.pair_id.clone(),
                variant: example.variant,
                actual_label: example.label.clone(),
                correct: prediction.label == example.label,
                predicted_label: prediction.label,
                accepted: prediction.accepted,
                confidence: prediction.confidence,
            }
        })
        .collect::<Vec<_>>();
    let actual_and_predicted = predictions
        .iter()
        .map(|prediction| {
            (
                prediction.actual_label.clone(),
                prediction.predicted_label.clone(),
            )
        })
        .collect::<Vec<_>>();
    let summary = summarize_baseline_predictions(&runtime.model.labels, &actual_and_predicted);
    let mut by_pair: BTreeMap<&str, Vec<&ContrastEvaluatedPrediction>> = BTreeMap::new();
    for prediction in &predictions {
        by_pair
            .entry(&prediction.pair_id)
            .or_default()
            .push(prediction);
    }
    if by_pair.values().any(|members| members.len() != 2) {
        return Err(MlError::InvalidDataset(
            "contrast evaluation requires exactly two predictions per pair".into(),
        ));
    }
    let correct_pairs = by_pair
        .values()
        .filter(|members| members.iter().all(|prediction| prediction.correct))
        .count();
    let flipped_pairs = by_pair
        .values()
        .filter(|members| members[0].predicted_label != members[1].predicted_label)
        .count();
    let accepted = predictions
        .iter()
        .filter(|prediction| prediction.accepted)
        .count();
    Ok(ContrastEvaluationV3 {
        example_count: predictions.len(),
        pair_count: by_pair.len(),
        accuracy: summary.accuracy,
        macro_f1: summary.macro_f1,
        pair_accuracy: correct_pairs as f64 / by_pair.len() as f64,
        prediction_flip_rate: flipped_pairs as f64 / by_pair.len() as f64,
        coverage: accepted as f64 / predictions.len() as f64,
        confusion_matrix: summary.confusion_matrix,
        per_class: summary.per_class,
        predictions,
    })
}

fn calibration_metrics(
    predictions: &[EvaluatedOpenSetPrediction],
    bins: usize,
) -> CalibrationMetrics {
    let count = predictions.len() as f64;
    let negative_log_likelihood = predictions
        .iter()
        .map(|prediction| {
            -prediction.probabilities[&prediction.actual_label]
                .max(1e-15)
                .ln()
        })
        .sum::<f64>()
        / count;
    let multiclass_brier = predictions
        .iter()
        .map(|prediction| {
            prediction
                .probabilities
                .iter()
                .map(|(label, probability)| {
                    let target = usize::from(label == &prediction.actual_label) as f64;
                    (probability - target).powi(2)
                })
                .sum::<f64>()
        })
        .sum::<f64>()
        / count;
    let mut expected_calibration_error = 0.0;
    for bin in 0..bins {
        let lower = bin as f64 / bins as f64;
        let upper = (bin + 1) as f64 / bins as f64;
        let members = predictions
            .iter()
            .filter(|prediction| {
                prediction.confidence >= lower
                    && (prediction.confidence < upper
                        || (bin + 1 == bins && prediction.confidence <= upper))
            })
            .collect::<Vec<_>>();
        if members.is_empty() {
            continue;
        }
        let accuracy = members
            .iter()
            .filter(|prediction| prediction.correct)
            .count() as f64
            / members.len() as f64;
        let confidence = members
            .iter()
            .map(|prediction| prediction.confidence)
            .sum::<f64>()
            / members.len() as f64;
        expected_calibration_error += members.len() as f64 / count * (accuracy - confidence).abs();
    }
    CalibrationMetrics {
        negative_log_likelihood,
        multiclass_brier,
        expected_calibration_error,
        ece_bins: bins,
    }
}

fn risk_coverage(predictions: &[EvaluatedOpenSetPrediction]) -> (Vec<RiskCoveragePoint>, f64) {
    let mut ranked = predictions.iter().collect::<Vec<_>>();
    ranked.sort_by(|left, right| {
        right
            .confidence
            .total_cmp(&left.confidence)
            .then_with(|| left.id.cmp(&right.id))
    });
    let mut correct = 0usize;
    let mut index = 0usize;
    let mut aurc = 0.0;
    let mut curve = Vec::with_capacity(ranked.len());
    while index < ranked.len() {
        let confidence = ranked[index].confidence;
        let start = index;
        while index < ranked.len() && ranked[index].confidence.total_cmp(&confidence).is_eq() {
            correct += usize::from(ranked[index].correct);
            index += 1;
        }
        let accepted = index;
        let risk = 1.0 - correct as f64 / accepted as f64;
        aurc += risk * (index - start) as f64 / ranked.len() as f64;
        curve.push(RiskCoveragePoint {
            accepted,
            coverage: accepted as f64 / ranked.len() as f64,
            risk,
        });
    }
    (curve, aurc)
}

fn evaluate_ood(
    runtime: &CompiledModel,
    id_evaluation: &IdEvaluationV3,
    ood: &OpenSetOodDataset,
) -> Result<OodEvaluationV3, MlError> {
    if ood.examples().is_empty() {
        return Err(MlError::InvalidDataset(
            "OOD evaluation requires at least one example".into(),
        ));
    }
    let predictions = ood
        .examples()
        .iter()
        .map(|example| {
            let prediction = runtime.predict(&example.text);
            OodEvaluatedPrediction {
                id: example.id.clone(),
                family_id: example.family_id.clone(),
                domain_group: example.domain_group.clone(),
                stratum: example.stratum,
                predicted_label: prediction.label,
                accepted: prediction.accepted,
                confidence: prediction.confidence,
                probability_margin: prediction.probability_margin,
            }
        })
        .collect::<Vec<_>>();
    let accepted_examples = predictions
        .iter()
        .filter(|prediction| prediction.accepted)
        .count();
    let id_scores = id_evaluation
        .predictions
        .iter()
        .map(|prediction| prediction.confidence)
        .collect::<Vec<_>>();
    let ood_scores = predictions
        .iter()
        .map(|prediction| prediction.confidence)
        .collect::<Vec<_>>();
    let mut by_stratum = BTreeMap::new();
    for stratum in [
        OodStratum::Semantic,
        OodStratum::Capability,
        OodStratum::Noise,
    ] {
        let members = predictions
            .iter()
            .filter(|prediction| prediction.stratum == stratum)
            .collect::<Vec<_>>();
        if members.is_empty() {
            return Err(MlError::InvalidDataset(format!(
                "OOD evaluation is missing the {} stratum",
                stratum.as_str()
            )));
        }
        let accepted = members
            .iter()
            .filter(|prediction| prediction.accepted)
            .count();
        let scores = members
            .iter()
            .map(|prediction| prediction.confidence)
            .collect::<Vec<_>>();
        by_stratum.insert(
            stratum.as_str().to_owned(),
            OodStratumEvaluation {
                example_count: members.len(),
                accepted_examples: accepted,
                coverage: accepted as f64 / members.len() as f64,
                discrimination: discrimination_metrics(&id_scores, &scores),
            },
        );
    }
    Ok(OodEvaluationV3 {
        example_count: predictions.len(),
        accepted_examples,
        coverage: accepted_examples as f64 / predictions.len() as f64,
        discrimination: discrimination_metrics(&id_scores, &ood_scores),
        by_stratum,
        predictions,
    })
}

fn discrimination_metrics(id_scores: &[f64], ood_scores: &[f64]) -> OodDiscriminationMetrics {
    let mut ranked = id_scores
        .iter()
        .copied()
        .map(|score| (score, true))
        .chain(ood_scores.iter().copied().map(|score| (score, false)))
        .collect::<Vec<_>>();
    ranked.sort_by(|(left_score, _), (right_score, _)| right_score.total_cmp(left_score));
    let mut true_positives = 0usize;
    let mut false_positives = 0usize;
    let mut auroc_wins = 0.0;
    let mut aupr_in_domain = 0.0;
    let mut index = 0usize;
    while index < ranked.len() {
        let score = ranked[index].0;
        let mut group_positives = 0usize;
        let mut group_negatives = 0usize;
        while index < ranked.len() && ranked[index].0.total_cmp(&score).is_eq() {
            if ranked[index].1 {
                group_positives += 1;
            } else {
                group_negatives += 1;
            }
            index += 1;
        }
        let lower_scoring_negatives = ood_scores.len() - false_positives - group_negatives;
        auroc_wins += (group_positives * lower_scoring_negatives) as f64
            + 0.5 * (group_positives * group_negatives) as f64;
        true_positives += group_positives;
        false_positives += group_negatives;
        if group_positives > 0 {
            let precision = true_positives as f64 / (true_positives + false_positives) as f64;
            let recall_increment = group_positives as f64 / id_scores.len() as f64;
            aupr_in_domain += precision * recall_increment;
        }
    }
    let auroc = auroc_wins / (id_scores.len() * ood_scores.len()) as f64;

    let mut sorted_id = id_scores.to_vec();
    sorted_id.sort_by(|left, right| right.total_cmp(left));
    let target_rank = ((0.95 * sorted_id.len() as f64).ceil() as usize)
        .saturating_sub(1)
        .min(sorted_id.len() - 1);
    let threshold = sorted_id[target_rank];
    let fpr_at_95_tpr = ood_scores
        .iter()
        .filter(|score| **score >= threshold)
        .count() as f64
        / ood_scores.len() as f64;
    OodDiscriminationMetrics {
        auroc,
        aupr_in_domain,
        fpr_at_95_tpr,
    }
}

fn bootstrap_report(
    labels: &[String],
    id: &IdEvaluationV3,
    id_examples: &[GroupedExample],
    ood: &OodEvaluationV3,
    ood_examples: &OpenSetOodDataset,
    seed: u64,
    resamples: usize,
) -> Result<BootstrapReport, MlError> {
    if !(100..=20_000).contains(&resamples) {
        return Err(MlError::InvalidConfiguration(
            "bootstrap resamples must be between 100 and 20000".into(),
        ));
    }
    let id_families = id_examples
        .iter()
        .map(|example| (example.id.as_str(), example.group_id.as_str()))
        .collect::<HashMap<_, _>>();
    let ood_domains = ood_examples
        .examples()
        .iter()
        .map(|example| (example.id.as_str(), example.domain_group.as_str()))
        .collect::<HashMap<_, _>>();
    let mut by_label_family: BTreeMap<&str, BTreeMap<&str, Vec<&EvaluatedOpenSetPrediction>>> =
        BTreeMap::new();
    for prediction in &id.predictions {
        let family = id_families.get(prediction.id.as_str()).ok_or_else(|| {
            MlError::InvalidDataset(
                "an ID-test prediction is missing its family for cluster bootstrap".into(),
            )
        })?;
        by_label_family
            .entry(&prediction.actual_label)
            .or_default()
            .entry(*family)
            .or_default()
            .push(prediction);
    }
    let mut by_ood_domain: BTreeMap<&str, Vec<&OodEvaluatedPrediction>> = BTreeMap::new();
    for prediction in &ood.predictions {
        let domain = ood_domains.get(prediction.id.as_str()).ok_or_else(|| {
            MlError::InvalidDataset(
                "an OOD-test prediction is missing its domain for cluster bootstrap".into(),
            )
        })?;
        by_ood_domain.entry(*domain).or_default().push(prediction);
    }
    if by_label_family.len() != labels.len() || by_ood_domain.is_empty() {
        return Err(MlError::InvalidDataset(
            "cluster bootstrap requires every ID label and at least one OOD domain".into(),
        ));
    }
    let maximum_id_rows = by_label_family
        .values()
        .map(|families| families.len() * families.values().map(Vec::len).max().unwrap_or_default())
        .sum::<usize>();
    let maximum_ood_rows = by_ood_domain.len()
        * by_ood_domain
            .values()
            .map(Vec::len)
            .max()
            .unwrap_or_default();
    let sampled_rows = maximum_id_rows
        .checked_add(maximum_ood_rows)
        .and_then(|population| population.checked_mul(resamples))
        .ok_or_else(|| {
            MlError::InvalidConfiguration("bootstrap workload overflows its size boundary".into())
        })?;
    if sampled_rows > MAX_BOOTSTRAP_SAMPLED_ROWS {
        return Err(MlError::InvalidConfiguration(format!(
            "bootstrap workload exceeds {MAX_BOOTSTRAP_SAMPLED_ROWS} sampled rows"
        )));
    }
    let mut rng = DeterministicRng::new(seed ^ 0x6a09_e667_f3bc_c909);
    let mut id_accuracy = Vec::with_capacity(resamples);
    let mut id_macro_f1 = Vec::with_capacity(resamples);
    let mut id_nll = Vec::with_capacity(resamples);
    let mut id_brier = Vec::with_capacity(resamples);
    let mut id_ece = Vec::with_capacity(resamples);
    let mut id_aurc = Vec::with_capacity(resamples);
    let mut ood_auroc = Vec::with_capacity(resamples);
    let mut ood_aupr = Vec::with_capacity(resamples);
    let mut ood_fpr95 = Vec::with_capacity(resamples);
    for _ in 0..resamples {
        let mut sampled_id = Vec::with_capacity(maximum_id_rows);
        for families in by_label_family.values() {
            let families = families.values().collect::<Vec<_>>();
            for _ in 0..families.len() {
                let selected = families[rng.index(families.len())];
                sampled_id.extend(selected.iter().map(|prediction| (*prediction).clone()));
            }
        }
        let summary = summarize_id_predictions(labels, sampled_id);
        let ood_domains = by_ood_domain.values().collect::<Vec<_>>();
        let mut sampled_ood = Vec::with_capacity(maximum_ood_rows);
        for _ in 0..ood_domains.len() {
            let selected = ood_domains[rng.index(ood_domains.len())];
            sampled_ood.extend(selected.iter().map(|prediction| (*prediction).clone()));
        }
        let id_scores = summary
            .predictions
            .iter()
            .map(|prediction| prediction.confidence)
            .collect::<Vec<_>>();
        let ood_scores = sampled_ood
            .iter()
            .map(|prediction| prediction.confidence)
            .collect::<Vec<_>>();
        let discrimination = discrimination_metrics(&id_scores, &ood_scores);
        id_accuracy.push(summary.accuracy);
        id_macro_f1.push(summary.macro_f1);
        id_nll.push(summary.calibration.negative_log_likelihood);
        id_brier.push(summary.calibration.multiclass_brier);
        id_ece.push(summary.calibration.expected_calibration_error);
        id_aurc.push(summary.aurc);
        ood_auroc.push(discrimination.auroc);
        ood_aupr.push(discrimination.aupr_in_domain);
        ood_fpr95.push(discrimination.fpr_at_95_tpr);
    }
    Ok(BootstrapReport {
        strategy: "label-stratified-id-family-and-ood-domain-cluster-percentile-v3".into(),
        seed,
        resamples,
        confidence_level: 0.95,
        id_accuracy: estimate(id.accuracy, id_accuracy),
        id_macro_f1: estimate(id.macro_f1, id_macro_f1),
        id_negative_log_likelihood: estimate(id.calibration.negative_log_likelihood, id_nll),
        id_multiclass_brier: estimate(id.calibration.multiclass_brier, id_brier),
        id_expected_calibration_error: estimate(id.calibration.expected_calibration_error, id_ece),
        id_aurc: estimate(id.aurc, id_aurc),
        ood_auroc: estimate(ood.discrimination.auroc, ood_auroc),
        ood_aupr_in_domain: estimate(ood.discrimination.aupr_in_domain, ood_aupr),
        ood_fpr_at_95_tpr: estimate(ood.discrimination.fpr_at_95_tpr, ood_fpr95),
    })
}

fn estimate(value: f64, mut samples: Vec<f64>) -> MetricEstimate {
    samples.sort_by(f64::total_cmp);
    let lower = ((samples.len() - 1) as f64 * 0.025).floor() as usize;
    let upper = ((samples.len() - 1) as f64 * 0.975).ceil() as usize;
    MetricEstimate {
        value,
        lower_95: samples[lower],
        upper_95: samples[upper.min(samples.len() - 1)],
    }
}

fn calibration_partition_metrics(
    model: &OpenSetModelV3,
    calibration: CalibrationPartition<'_>,
    temperature: f64,
) -> Result<CalibrationMetrics, MlError> {
    let examples = calibration.examples();
    let policy = OpenSetPolicyV3 {
        schema_version: OPEN_SET_SCHEMA_VERSION,
        model_version: model.model_version.clone(),
        dataset_sha256: model.dataset_sha256.clone(),
        split_plan_sha256: model.split_plan_sha256.clone(),
        temperature,
        minimum_confidence: 0.0,
        minimum_probability_margin: 0.0,
        temperature_source: "calibration-partition-temperature-scaling-v3".into(),
        threshold_source: "fixed-development-plus-ood-development-grid-v3".into(),
        calibration_example_count: examples.len(),
        development_example_count: 1,
        ood_development_example_count: 1,
    };
    let runtime = CompiledModel::new(model.clone(), policy)?;
    Ok(evaluate_id(&runtime, examples)?.calibration)
}

#[derive(Debug, Clone)]
pub struct OpenSetExperimentResult {
    pub model: OpenSetModelV3,
    pub policy: OpenSetPolicyV3,
    pub metrics: OpenSetMetricsV3,
    pub split_plan: SplitPlanManifest,
}

pub fn run_open_set_experiment(
    dataset: &GroupedDataset,
    ood_development: &OpenSetOodDataset,
    ood_test: &OpenSetOodDataset,
    contrast_test: &OpenSetContrastDataset,
    config: OpenSetTrainingConfig,
    bootstrap_resamples: usize,
) -> Result<OpenSetExperimentResult, MlError> {
    reject_cross_dataset_overlap(dataset, ood_development, ood_test, contrast_test)?;
    let plan = SplitPlan::build(
        dataset,
        ood_development,
        ood_test,
        contrast_test,
        config.seed,
    )?;
    let training = plan.training_partition()?;
    let development = plan.development_partition();
    let calibration = plan.calibration_partition();
    // These role-specific views are compile-time capabilities: the selector cannot access
    // calibration, ID-test, or either OOD population through its parameters.
    let (model, development_selection) =
        select_model_on_development(&training, development, config.clone())?;

    // The calibrator accepts only the typed calibration capability.
    let temperature = calibrate_temperature(&model, calibration)?;
    let uncalibrated_calibration_partition =
        calibration_partition_metrics(&model, calibration, 1.0)?;
    let calibrated_calibration_partition =
        calibration_partition_metrics(&model, calibration, temperature)?;

    // Threshold selection accepts only typed ID-development and OOD-development capabilities.
    let threshold_selection = select_thresholds(
        &model,
        temperature,
        development,
        OodDevelopmentPartition(ood_development),
    )?;
    let policy = OpenSetPolicyV3 {
        schema_version: OPEN_SET_SCHEMA_VERSION,
        model_version: model.model_version.clone(),
        dataset_sha256: model.dataset_sha256.clone(),
        split_plan_sha256: model.split_plan_sha256.clone(),
        temperature,
        minimum_confidence: threshold_selection.selected_confidence,
        minimum_probability_margin: threshold_selection.selected_probability_margin,
        temperature_source: "calibration-partition-temperature-scaling-v3".into(),
        threshold_source: "fixed-development-plus-ood-development-grid-v3".into(),
        calibration_example_count: plan.calibration().len(),
        development_example_count: plan.development().len(),
        ood_development_example_count: ood_development.examples().len(),
    };
    let runtime = CompiledModel::new(model.clone(), policy.clone())?;

    // Only after the fitted model and operating policy are frozen do we evaluate the three tests.
    let id_test = evaluate_id(&runtime, plan.id_test())?;
    let baselines = evaluate_baselines(
        &training,
        plan.id_test(),
        &model.labels,
        id_test.accuracy,
        id_test.macro_f1,
    );
    let contrast_test_evaluation =
        evaluate_contrast(&runtime, ContrastTestPartition(contrast_test))?;
    let ood_test_evaluation = evaluate_ood(&runtime, &id_test, ood_test)?;
    let bootstrap_95 = bootstrap_report(
        &model.labels,
        &id_test,
        plan.id_test(),
        &ood_test_evaluation,
        ood_test,
        config.seed,
        bootstrap_resamples,
    )?;
    let partition_counts = BTreeMap::from([
        ("train".into(), plan.train().len()),
        ("development".into(), plan.development().len()),
        ("calibration".into(), plan.calibration().len()),
        ("id-test".into(), plan.id_test().len()),
        ("ood-development".into(), ood_development.examples().len()),
        ("ood-test".into(), ood_test.examples().len()),
        ("contrast-test".into(), contrast_test.examples().len()),
    ]);
    let family_count = |partition: PartitionKind| {
        plan.manifest
            .assignments
            .iter()
            .filter(|assignment| assignment.partition == partition)
            .map(|assignment| assignment.group_id.as_str())
            .collect::<HashSet<_>>()
            .len()
    };
    let partition_family_counts = BTreeMap::from([
        ("train".into(), family_count(PartitionKind::Train)),
        (
            "development".into(),
            family_count(PartitionKind::Development),
        ),
        (
            "calibration".into(),
            family_count(PartitionKind::Calibration),
        ),
        ("id-test".into(), family_count(PartitionKind::IdTest)),
        (
            "ood-development".into(),
            ood_development
                .examples()
                .iter()
                .map(|example| example.family_id.as_str())
                .collect::<HashSet<_>>()
                .len(),
        ),
        (
            "ood-test".into(),
            ood_test
                .examples()
                .iter()
                .map(|example| example.family_id.as_str())
                .collect::<HashSet<_>>()
                .len(),
        ),
        (
            "contrast-test".into(),
            contrast_test
                .examples()
                .iter()
                .map(|example| example.pair_id.as_str())
                .collect::<HashSet<_>>()
                .len(),
        ),
    ]);
    let supervised_family_count = dataset
        .examples()
        .iter()
        .map(|example| example.group_id.as_str())
        .collect::<HashSet<_>>()
        .len();
    let paraphrases_per_family = dataset.examples().len() / supervised_family_count;
    let ood_domain_counts = BTreeMap::from([
        (
            "ood-development".into(),
            ood_development
                .examples()
                .iter()
                .map(|example| example.domain_group.as_str())
                .collect::<HashSet<_>>()
                .len(),
        ),
        (
            "ood-test".into(),
            ood_test
                .examples()
                .iter()
                .map(|example| example.domain_group.as_str())
                .collect::<HashSet<_>>()
                .len(),
        ),
    ]);
    let stratum_counts = |dataset: &OpenSetOodDataset| {
        let mut counts = BTreeMap::new();
        for example in dataset.examples() {
            *counts
                .entry(example.stratum.as_str().to_owned())
                .or_insert(0) += 1;
        }
        counts
    };
    let ood_stratum_counts = BTreeMap::from([
        ("ood-development".into(), stratum_counts(ood_development)),
        ("ood-test".into(), stratum_counts(ood_test)),
    ]);
    let mut limitations = vec![
        "All current examples are synthetic and English-only.".into(),
        "The four-way ID split is family-disjoint but synthetic and still modest; confidence intervals remain wide.".into(),
        "The same development partition selects the model candidate and the abstention policy, so development-selection optimism remains possible.".into(),
        "OOD discrimination is measured only on balanced synthetic semantic, capability, and noise strata; broader domains can behave differently.".into(),
        "ID intervals cluster-resample held-out families and OOD intervals cluster-resample broader domains, but both have few independent clusters.".into(),
        "A training-only unigram baseline is reported because surface-label shortcuts remain a material benchmark risk.".into(),
        "The anti-shortcut contrast test is small, synthetic, and source-declared before the final run; it is not an external benchmark.".into(),
        "This classifier is not suitable for clinical, safety, employment, or other decisions about people.".into(),
    ];
    if baselines.learned_minus_unigram_accuracy <= 0.0
        || baselines.learned_minus_unigram_macro_f1 <= 0.0
    {
        limitations.push(
            "On the frozen ID test, the learned model does not beat the training-only unigram baseline on both accuracy and macro F1."
                .into(),
        );
    }
    let metrics = OpenSetMetricsV3 {
        schema_version: OPEN_SET_SCHEMA_VERSION,
        model_version: OPEN_SET_MODEL_VERSION.into(),
        dataset_sha256: dataset.fingerprint_sha256(),
        split_plan_sha256: model.split_plan_sha256.clone(),
        ood_development_sha256: ood_development.fingerprint_sha256(),
        ood_test_sha256: ood_test.fingerprint_sha256(),
        contrast_test_sha256: contrast_test.fingerprint_sha256(),
        partition_counts,
        partition_family_counts,
        paraphrases_per_family,
        ood_domain_counts,
        ood_stratum_counts,
        development_selection,
        threshold_selection,
        uncalibrated_calibration_partition,
        calibrated_calibration_partition,
        id_test,
        baselines,
        contrast_test: contrast_test_evaluation,
        ood_test: ood_test_evaluation,
        bootstrap_95,
        limitations,
    };
    Ok(OpenSetExperimentResult {
        model,
        policy,
        metrics,
        split_plan: plan.manifest,
    })
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct BundleManifestV3 {
    pub schema_version: u32,
    pub bundle_kind: String,
    pub bundle_version: String,
    pub model_version: String,
    pub dataset_sha256: String,
    pub split_plan_sha256: String,
    pub files: BTreeMap<String, String>,
}

#[derive(Debug, Clone)]
pub struct VerifiedBundle {
    manifest: BundleManifestV3,
    model: OpenSetModelV3,
    policy: OpenSetPolicyV3,
    metrics: OpenSetMetricsV3,
    split_plan: SplitPlanManifest,
}

impl VerifiedBundle {
    pub fn manifest(&self) -> &BundleManifestV3 {
        &self.manifest
    }

    pub fn model(&self) -> &OpenSetModelV3 {
        &self.model
    }

    pub fn policy(&self) -> &OpenSetPolicyV3 {
        &self.policy
    }

    pub fn metrics(&self) -> &OpenSetMetricsV3 {
        &self.metrics
    }

    pub fn split_plan(&self) -> &SplitPlanManifest {
        &self.split_plan
    }

    pub fn compile(self) -> Result<CompiledModel, MlError> {
        CompiledModel::new(self.model, self.policy)
    }
}

pub fn write_bundle(
    directory: impl AsRef<Path>,
    result: &OpenSetExperimentResult,
) -> Result<BundleManifestV3, MlError> {
    let directory = directory.as_ref();
    validate_bundle_destination(directory)?;
    let payloads = BTreeMap::from([
        (
            "metrics.json".to_string(),
            canonical_metrics_json(&result.metrics)?,
        ),
        ("model.json".to_string(), canonical_json(&result.model)?),
        ("policy.json".to_string(), canonical_json(&result.policy)?),
        (
            "split-plan.json".to_string(),
            canonical_json(&result.split_plan)?,
        ),
    ]);
    let parent = directory.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent)?;
    let name = directory
        .file_name()
        .ok_or_else(|| {
            MlError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "bundle destination has no directory name",
            ))
        })?
        .to_string_lossy();
    let sequence = BUNDLE_TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    let staging = parent.join(format!(".{name}.staging-{}-{sequence}", std::process::id()));
    let backup = parent.join(format!(".{name}.backup-{}-{sequence}", std::process::id()));
    fs::create_dir(&staging)?;
    let files = payloads
        .iter()
        .map(|(name, bytes)| (name.clone(), sha256_hex(bytes)))
        .collect::<BTreeMap<_, _>>();
    let manifest = BundleManifestV3 {
        schema_version: OPEN_SET_SCHEMA_VERSION,
        bundle_kind: "eliza-open-set-bundle".into(),
        bundle_version: OPEN_SET_BUNDLE_VERSION.into(),
        model_version: result.model.model_version.clone(),
        dataset_sha256: result.model.dataset_sha256.clone(),
        split_plan_sha256: result.model.split_plan_sha256.clone(),
        files,
    };

    for (name, bytes) in payloads {
        if let Err(error) = write_atomic(&staging.join(name), &bytes) {
            let _ = fs::remove_dir_all(&staging);
            return Err(error);
        }
    }
    // The manifest is installed last. A partially written directory therefore never verifies.
    if let Err(error) = write_atomic(&staging.join("manifest.json"), &canonical_json(&manifest)?) {
        let _ = fs::remove_dir_all(&staging);
        return Err(error);
    }
    if let Err(error) = verify_bundle_contract(&staging) {
        let _ = fs::remove_dir_all(&staging);
        return Err(error);
    }

    let had_original = directory.exists();
    if had_original {
        if let Err(error) = fs::rename(directory, &backup) {
            let _ = fs::remove_dir_all(&staging);
            return Err(MlError::Io(error));
        }
    }
    if let Err(error) = fs::rename(&staging, directory) {
        let restore_error = if had_original {
            fs::rename(&backup, directory).err()
        } else {
            None
        };
        let _ = fs::remove_dir_all(&staging);
        let message = restore_error.map_or_else(
            || format!("bundle installation failed and the prior bundle was restored: {error}"),
            |restore| {
                format!(
                    "bundle installation failed: {error}; restoring the prior bundle also failed: {restore}"
                )
            },
        );
        return Err(MlError::Io(std::io::Error::other(message)));
    }
    if let Err(error) = verify_bundle_contract(directory) {
        let displacement_error = fs::rename(directory, &staging).err();
        let restore_error = if had_original && displacement_error.is_none() {
            fs::rename(&backup, directory).err()
        } else {
            None
        };
        if displacement_error.is_none() && restore_error.is_none() {
            let _ = fs::remove_dir_all(&staging);
        }
        let message = match (displacement_error, restore_error) {
            (None, None) if had_original => {
                format!("installed bundle failed verification and the prior bundle was restored: {error}")
            }
            (None, None) => format!("installed bundle failed verification and was removed: {error}"),
            (Some(displacement), _) => format!(
                "installed bundle failed verification: {error}; moving it aside also failed: {displacement}"
            ),
            (None, Some(restore)) => format!(
                "installed bundle failed verification: {error}; restoring the prior bundle also failed: {restore}"
            ),
        };
        return Err(MlError::Io(std::io::Error::other(message)));
    }
    if had_original {
        fs::remove_dir_all(&backup)?;
    }
    Ok(manifest)
}

pub fn verify_bundle(directory: impl AsRef<Path>) -> Result<VerifiedBundle, MlError> {
    load_bundle(directory.as_ref(), true)
}

fn verify_bundle_contract(directory: &Path) -> Result<VerifiedBundle, MlError> {
    load_bundle(directory, false)
}

fn load_bundle(directory: &Path, reproduce_semantics: bool) -> Result<VerifiedBundle, MlError> {
    validate_bundle_inventory(directory)?;
    let manifest_path = directory.join("manifest.json");
    let manifest: BundleManifestV3 = read_bounded_json(&manifest_path, "bundle manifest")?;
    validate_manifest(&manifest)?;
    for (name, expected_sha256) in &manifest.files {
        if !valid_sha256(expected_sha256) {
            return Err(MlError::InvalidModel(format!(
                "bundle file `{name}` has a malformed SHA-256"
            )));
        }
        let path = directory.join(name);
        reject_symlink_or_non_file(&path, name)?;
        reject_oversized_file(&path, MAX_JSON_BYTES, name)?;
        let observed = sha256_hex(&fs::read(&path)?);
        if &observed != expected_sha256 {
            return Err(MlError::InvalidModel(format!(
                "bundle file `{name}` failed SHA-256 verification"
            )));
        }
    }
    let model: OpenSetModelV3 = read_bounded_json(&directory.join("model.json"), "model")?;
    let policy: OpenSetPolicyV3 = read_bounded_json(&directory.join("policy.json"), "policy")?;
    let metrics: OpenSetMetricsV3 = read_bounded_json(&directory.join("metrics.json"), "metrics")?;
    let split_plan: SplitPlanManifest =
        read_bounded_json(&directory.join("split-plan.json"), "split plan")?;
    if reproduce_semantics {
        validate_bundle_artifacts(&manifest, &model, &policy, &metrics, &split_plan)?;
    } else {
        validate_bundle_artifact_contract(&manifest, &model, &policy, &metrics, &split_plan)?;
    }
    Ok(VerifiedBundle {
        manifest,
        model,
        policy,
        metrics,
        split_plan,
    })
}

fn validate_bundle_destination(directory: &Path) -> Result<(), MlError> {
    let metadata = match fs::symlink_metadata(directory) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(error) => return Err(MlError::Io(error)),
    };
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        return Err(MlError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!(
                "bundle destination {} must be a real directory",
                directory.display()
            ),
        )));
    }
    if fs::read_dir(directory)?.next().transpose()?.is_some() {
        verify_bundle(directory).map_err(|error| {
            MlError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!(
                    "refusing to replace non-empty directory {} because it is not a verified v3 bundle: {error}",
                    directory.display()
                ),
            ))
        })?;
    }
    Ok(())
}

fn validate_bundle_inventory(directory: &Path) -> Result<(), MlError> {
    let metadata = fs::symlink_metadata(directory)?;
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        return Err(MlError::InvalidModel(
            "the bundle root must be a real directory".into(),
        ));
    }
    let mut observed = BTreeSet::new();
    for entry in fs::read_dir(directory)? {
        let entry = entry?;
        let name = entry.file_name().into_string().map_err(|_| {
            MlError::InvalidModel("the bundle contains a non-UTF-8 file name".into())
        })?;
        observed.insert(name);
    }
    let expected = BUNDLE_INVENTORY
        .into_iter()
        .map(str::to_owned)
        .collect::<BTreeSet<_>>();
    if observed != expected {
        return Err(MlError::InvalidModel(
            "the bundle inventory must contain exactly the five v3 files".into(),
        ));
    }
    Ok(())
}

pub fn embedded_bundle() -> Result<VerifiedBundle, MlError> {
    let bytes = BTreeMap::from([
        (
            "metrics.json",
            include_bytes!("../artifacts/eliza-open-set-v3/metrics.json").as_slice(),
        ),
        (
            "model.json",
            include_bytes!("../artifacts/eliza-open-set-v3/model.json").as_slice(),
        ),
        (
            "policy.json",
            include_bytes!("../artifacts/eliza-open-set-v3/policy.json").as_slice(),
        ),
        (
            "split-plan.json",
            include_bytes!("../artifacts/eliza-open-set-v3/split-plan.json").as_slice(),
        ),
    ]);
    let manifest: BundleManifestV3 = serde_json::from_slice(include_bytes!(
        "../artifacts/eliza-open-set-v3/manifest.json"
    ))?;
    validate_manifest(&manifest)?;
    for (name, content) in &bytes {
        if sha256_hex(content) != manifest.files[*name] {
            return Err(MlError::InvalidModel(format!(
                "embedded bundle file `{name}` failed SHA-256 verification"
            )));
        }
    }
    let model: OpenSetModelV3 = serde_json::from_slice(bytes["model.json"])?;
    let policy: OpenSetPolicyV3 = serde_json::from_slice(bytes["policy.json"])?;
    let metrics: OpenSetMetricsV3 = serde_json::from_slice(bytes["metrics.json"])?;
    let split_plan: SplitPlanManifest = serde_json::from_slice(bytes["split-plan.json"])?;
    validate_bundle_artifact_contract(&manifest, &model, &policy, &metrics, &split_plan)?;
    Ok(VerifiedBundle {
        manifest,
        model,
        policy,
        metrics,
        split_plan,
    })
}

fn validate_manifest(manifest: &BundleManifestV3) -> Result<(), MlError> {
    if manifest.schema_version != OPEN_SET_SCHEMA_VERSION
        || manifest.bundle_kind != "eliza-open-set-bundle"
        || manifest.bundle_version != OPEN_SET_BUNDLE_VERSION
        || manifest.model_version != OPEN_SET_MODEL_VERSION
        || !valid_sha256(&manifest.dataset_sha256)
        || !valid_sha256(&manifest.split_plan_sha256)
        || manifest.files.keys().cloned().collect::<BTreeSet<_>>()
            != BTreeSet::from([
                "metrics.json".to_string(),
                "model.json".to_string(),
                "policy.json".to_string(),
                "split-plan.json".to_string(),
            ])
    {
        return Err(MlError::InvalidModel(
            "the bundle manifest violates the v3 contract".into(),
        ));
    }
    Ok(())
}

fn validate_bundle_artifacts(
    manifest: &BundleManifestV3,
    model: &OpenSetModelV3,
    policy: &OpenSetPolicyV3,
    metrics: &OpenSetMetricsV3,
    split_plan: &SplitPlanManifest,
) -> Result<(), MlError> {
    validate_bundle_artifact_contract(manifest, model, policy, metrics, split_plan)?;
    let (dataset, ood_development, ood_test, contrast_test) = datasets_from_plan(split_plan)?;
    let reproduced = run_open_set_experiment(
        &dataset,
        &ood_development,
        &ood_test,
        &contrast_test,
        model.training_config.clone(),
        metrics.bootstrap_95.resamples,
    )?;
    for (matches, artifact) in [
        (reproduced.model == *model, "model"),
        (reproduced.policy == *policy, "policy"),
        (reproduced.split_plan == *split_plan, "experiment plan"),
    ] {
        if !matches {
            return Err(MlError::InvalidModel(format!(
                "bundle semantic verification did not reproduce its {artifact}"
            )));
        }
    }
    if !semantic_json_equal(&reproduced.metrics, metrics)? {
        return Err(MlError::InvalidModel(
            "bundle semantic verification did not reproduce its metrics".into(),
        ));
    }
    Ok(())
}

fn validate_bundle_artifact_contract(
    manifest: &BundleManifestV3,
    model: &OpenSetModelV3,
    policy: &OpenSetPolicyV3,
    metrics: &OpenSetMetricsV3,
    split_plan: &SplitPlanManifest,
) -> Result<(), MlError> {
    model.validate()?;
    policy.validate_against(model)?;
    split_plan.validate_contract()?;
    if manifest.dataset_sha256 != model.dataset_sha256
        || manifest.split_plan_sha256 != model.split_plan_sha256
        || metrics.schema_version != OPEN_SET_SCHEMA_VERSION
        || metrics.model_version != model.model_version
        || metrics.dataset_sha256 != model.dataset_sha256
        || metrics.split_plan_sha256 != model.split_plan_sha256
        || split_plan.schema_version != OPEN_SET_SCHEMA_VERSION
        || split_plan.dataset_sha256 != model.dataset_sha256
        || split_plan.seed != model.training_config.seed
        || sha256_hex(&canonical_json(split_plan)?) != model.split_plan_sha256
    {
        return Err(MlError::InvalidModel(
            "bundle artifacts disagree about their model or data provenance".into(),
        ));
    }
    validate_metrics_contract(metrics, model, policy, split_plan)?;
    Ok(())
}

fn semantic_json_equal<T: Serialize>(left: &T, right: &T) -> Result<bool, MlError> {
    fn values_match(left: &serde_json::Value, right: &serde_json::Value) -> bool {
        match (left, right) {
            (serde_json::Value::Number(left), serde_json::Value::Number(right)) => {
                match (left.as_f64(), right.as_f64()) {
                    (Some(left), Some(right)) => approximately_equal(left, right),
                    _ => left == right,
                }
            }
            (serde_json::Value::Array(left), serde_json::Value::Array(right)) => {
                left.len() == right.len()
                    && left
                        .iter()
                        .zip(right)
                        .all(|(left, right)| values_match(left, right))
            }
            (serde_json::Value::Object(left), serde_json::Value::Object(right)) => {
                left.len() == right.len()
                    && left.iter().all(|(key, left)| {
                        right
                            .get(key)
                            .is_some_and(|right| values_match(left, right))
                    })
            }
            _ => left == right,
        }
    }
    Ok(values_match(
        &serde_json::to_value(left)?,
        &serde_json::to_value(right)?,
    ))
}

fn datasets_from_plan(
    plan: &SplitPlanManifest,
) -> Result<
    (
        GroupedDataset,
        OpenSetOodDataset,
        OpenSetOodDataset,
        OpenSetContrastDataset,
    ),
    MlError,
> {
    let dataset = GroupedDataset {
        examples: plan
            .assignments
            .iter()
            .map(|assignment| GroupedExample {
                id: assignment.id.clone(),
                group_id: assignment.group_id.clone(),
                label: assignment.label.clone(),
                text: assignment.text.clone(),
            })
            .collect(),
    };
    dataset.validate_partition_support()?;
    if dataset.fingerprint_sha256() != plan.dataset_sha256 {
        return Err(MlError::InvalidModel(
            "the experiment plan dataset rows do not match its fingerprint".into(),
        ));
    }
    let to_dataset = |rows: &[OodPlanRow]| OpenSetOodDataset {
        examples: rows
            .iter()
            .map(|row| OpenSetOodExample {
                id: row.id.clone(),
                family_id: row.family_id.clone(),
                domain_group: row.domain_group.clone(),
                stratum: row.stratum,
                text: row.text.clone(),
            })
            .collect(),
    };
    let ood_development = to_dataset(&plan.ood_development);
    let ood_test = to_dataset(&plan.ood_test);
    let contrast_test = OpenSetContrastDataset {
        examples: plan
            .contrast_test
            .iter()
            .map(|row| OpenSetContrastExample {
                id: row.id.clone(),
                pair_id: row.pair_id.clone(),
                variant: row.variant,
                label: row.label.clone(),
                text: row.text.clone(),
            })
            .collect(),
    };
    reject_cross_dataset_overlap(&dataset, &ood_development, &ood_test, &contrast_test)?;
    Ok((dataset, ood_development, ood_test, contrast_test))
}

fn validate_metrics_contract(
    metrics: &OpenSetMetricsV3,
    model: &OpenSetModelV3,
    policy: &OpenSetPolicyV3,
    split_plan: &SplitPlanManifest,
) -> Result<(), MlError> {
    let mut split_counts = BTreeMap::from([
        (PartitionKind::Train, 0usize),
        (PartitionKind::Development, 0usize),
        (PartitionKind::Calibration, 0usize),
        (PartitionKind::IdTest, 0usize),
    ]);
    for assignment in &split_plan.assignments {
        *split_counts
            .get_mut(&assignment.partition)
            .expect("all typed partitions are initialized") += 1;
    }
    let supervised_family_count = |partition: PartitionKind| {
        split_plan
            .assignments
            .iter()
            .filter(|assignment| assignment.partition == partition)
            .map(|assignment| assignment.group_id.as_str())
            .collect::<HashSet<_>>()
            .len()
    };
    let expected_counts = BTreeMap::from([
        (
            "calibration".into(),
            split_counts[&PartitionKind::Calibration],
        ),
        (
            "development".into(),
            split_counts[&PartitionKind::Development],
        ),
        ("id-test".into(), split_counts[&PartitionKind::IdTest]),
        ("ood-development".into(), split_plan.ood_development.len()),
        ("ood-test".into(), split_plan.ood_test.len()),
        ("contrast-test".into(), split_plan.contrast_test.len()),
        ("train".into(), split_counts[&PartitionKind::Train]),
    ]);
    let expected_family_counts = BTreeMap::from([
        (
            "calibration".into(),
            supervised_family_count(PartitionKind::Calibration),
        ),
        (
            "development".into(),
            supervised_family_count(PartitionKind::Development),
        ),
        (
            "id-test".into(),
            supervised_family_count(PartitionKind::IdTest),
        ),
        (
            "ood-development".into(),
            split_plan
                .ood_development
                .iter()
                .map(|row| row.family_id.as_str())
                .collect::<HashSet<_>>()
                .len(),
        ),
        (
            "ood-test".into(),
            split_plan
                .ood_test
                .iter()
                .map(|row| row.family_id.as_str())
                .collect::<HashSet<_>>()
                .len(),
        ),
        (
            "contrast-test".into(),
            split_plan
                .contrast_test
                .iter()
                .map(|row| row.pair_id.as_str())
                .collect::<HashSet<_>>()
                .len(),
        ),
        (
            "train".into(),
            supervised_family_count(PartitionKind::Train),
        ),
    ]);
    let supervised_family_sizes = split_plan.assignments.iter().fold(
        BTreeMap::<&str, usize>::new(),
        |mut counts, assignment| {
            *counts.entry(assignment.group_id.as_str()).or_insert(0) += 1;
            counts
        },
    );
    let expected_paraphrases_per_family = supervised_family_sizes
        .values()
        .next()
        .copied()
        .unwrap_or(0);
    let expected_domain_counts = BTreeMap::from([
        (
            "ood-development".into(),
            split_plan
                .ood_development
                .iter()
                .map(|row| row.domain_group.as_str())
                .collect::<HashSet<_>>()
                .len(),
        ),
        (
            "ood-test".into(),
            split_plan
                .ood_test
                .iter()
                .map(|row| row.domain_group.as_str())
                .collect::<HashSet<_>>()
                .len(),
        ),
    ]);
    let stratum_counts = |rows: &[OodPlanRow]| {
        let mut counts = BTreeMap::new();
        for row in rows {
            *counts.entry(row.stratum.as_str().to_owned()).or_insert(0) += 1;
        }
        counts
    };
    let expected_stratum_counts = BTreeMap::from([
        (
            "ood-development".into(),
            stratum_counts(&split_plan.ood_development),
        ),
        ("ood-test".into(), stratum_counts(&split_plan.ood_test)),
    ]);
    let thresholds = &metrics.threshold_selection;
    if metrics.partition_counts != expected_counts
        || metrics.partition_family_counts != expected_family_counts
        || expected_paraphrases_per_family < MIN_PARAPHRASES_PER_FAMILY
        || supervised_family_sizes
            .values()
            .any(|size| *size != expected_paraphrases_per_family)
        || metrics.paraphrases_per_family != expected_paraphrases_per_family
        || metrics.ood_domain_counts != expected_domain_counts
        || metrics.ood_stratum_counts != expected_stratum_counts
        || policy.calibration_example_count != split_counts[&PartitionKind::Calibration]
        || policy.development_example_count != split_counts[&PartitionKind::Development]
        || thresholds.strategy != "fixed-development-plus-ood-development-grid-v3"
        || thresholds.evaluated_candidate_count != 49
        || thresholds.feasible_candidate_count == 0
        || thresholds.feasible_candidate_count > thresholds.evaluated_candidate_count
        || thresholds.development_example_count != policy.development_example_count
        || thresholds.ood_development_example_count != policy.ood_development_example_count
        || thresholds.selected_confidence != policy.minimum_confidence
        || thresholds.selected_probability_margin != policy.minimum_probability_margin
        || thresholds.inputs
            != [
                SelectionDataRole::Development,
                SelectionDataRole::OodDevelopment,
            ]
        || thresholds.minimum_development_selective_accuracy != 0.75
        || thresholds.maximum_ood_development_coverage != 0.10
        || !valid_unit_interval(thresholds.observed_development_coverage)
        || !valid_unit_interval(thresholds.observed_development_selective_accuracy)
        || !valid_unit_interval(thresholds.observed_ood_development_coverage)
        || !valid_sha256(&metrics.ood_development_sha256)
        || !valid_sha256(&metrics.ood_test_sha256)
        || !valid_sha256(&metrics.contrast_test_sha256)
        || metrics.ood_development_sha256 == metrics.ood_test_sha256
        || metrics.contrast_test_sha256 == metrics.ood_development_sha256
        || metrics.contrast_test_sha256 == metrics.ood_test_sha256
        || metrics.ood_development_sha256 == metrics.dataset_sha256
        || metrics.ood_test_sha256 == metrics.dataset_sha256
        || metrics.contrast_test_sha256 == metrics.dataset_sha256
    {
        return Err(MlError::InvalidModel(
            "the v3 metrics disagree with the split or frozen policy".into(),
        ));
    }
    let selection = &metrics.development_selection;
    let expected_candidates = model
        .training_config
        .development_selection
        .max_features_candidates
        .iter()
        .flat_map(|max_features| {
            model
                .training_config
                .development_selection
                .l2_penalty_candidates
                .iter()
                .map(move |l2_penalty| (*max_features, *l2_penalty))
        })
        .collect::<Vec<_>>();
    if selection.strategy != "train-fit-development-f1-epsilon-parsimony-accuracy-nll-brier-v3"
        || selection.seed != model.training_config.seed
        || selection.macro_f1_tolerance
            != model
                .training_config
                .development_selection
                .macro_f1_tolerance
        || selection.training_example_count != split_counts[&PartitionKind::Train]
        || selection.training_family_count != supervised_family_count(PartitionKind::Train)
        || selection.development_example_count != split_counts[&PartitionKind::Development]
        || selection.development_family_count != supervised_family_count(PartitionKind::Development)
        || selection.inputs != [SelectionDataRole::Train, SelectionDataRole::Development]
        || selection.candidates.len() != expected_candidates.len()
        || selection.selected_index >= selection.candidates.len()
        || selection
            .candidates
            .iter()
            .zip(&expected_candidates)
            .any(|(candidate, expected)| {
                (candidate.max_features, candidate.l2_penalty) != *expected
                    || !valid_unit_interval(candidate.accuracy)
                    || !valid_unit_interval(candidate.macro_f1)
                    || !candidate.negative_log_likelihood.is_finite()
                    || candidate.negative_log_likelihood < 0.0
                    || !candidate.multiclass_brier.is_finite()
                    || !(0.0..=2.0).contains(&candidate.multiclass_brier)
            })
    {
        return Err(MlError::InvalidModel(
            "the development-only model-selection audit is invalid".into(),
        ));
    }
    let selected_candidate = &selection.candidates[selection.selected_index];
    let reproduced_selected_index =
        selection
            .candidates
            .iter()
            .enumerate()
            .skip(1)
            .fold(0usize, |best, (index, candidate)| {
                if development_candidate_is_better(
                    candidate,
                    &selection.candidates[best],
                    selection.macro_f1_tolerance,
                ) {
                    index
                } else {
                    best
                }
            });
    if selection.selected_index != reproduced_selected_index
        || selected_candidate.max_features != model.training_config.vectorizer.max_features
        || selected_candidate.l2_penalty != model.training_config.l2_penalty
    {
        return Err(MlError::InvalidModel(
            "the fitted model does not match the recorded development-only selection".into(),
        ));
    }
    validate_calibration_metrics(&metrics.uncalibrated_calibration_partition)?;
    validate_calibration_metrics(&metrics.calibrated_calibration_partition)?;
    if metrics
        .calibrated_calibration_partition
        .negative_log_likelihood
        > metrics
            .uncalibrated_calibration_partition
            .negative_log_likelihood
            + 1e-12
    {
        return Err(MlError::InvalidModel(
            "the recorded temperature increases calibration NLL".into(),
        ));
    }

    let label_set = model
        .labels
        .iter()
        .map(String::as_str)
        .collect::<BTreeSet<_>>();
    let split_label_set = split_plan
        .assignments
        .iter()
        .map(|assignment| assignment.label.as_str())
        .collect::<BTreeSet<_>>();
    let id_assignments = split_plan
        .assignments
        .iter()
        .filter(|assignment| assignment.partition == PartitionKind::IdTest)
        .map(|assignment| (assignment.id.as_str(), assignment))
        .collect::<BTreeMap<_, _>>();
    if split_label_set != label_set || id_assignments.len() != metrics.id_test.example_count {
        return Err(MlError::InvalidModel(
            "the model labels or ID-test ledger disagree with the split plan".into(),
        ));
    }
    let runtime = CompiledModel::new(model.clone(), policy.clone())?;
    let mut id_ids = HashSet::new();
    for prediction in &metrics.id_test.predictions {
        let assignment = id_assignments.get(prediction.id.as_str()).ok_or_else(|| {
            MlError::InvalidModel(
                "the ID-test prediction ledger disagrees with the split plan".into(),
            )
        })?;
        if assignment.label != prediction.actual_label {
            return Err(MlError::InvalidModel(
                "the ID-test prediction ledger disagrees with the split plan".into(),
            ));
        }
        let reproduced_prediction = runtime.predict(&assignment.text);
        let probability_labels = prediction
            .probabilities
            .keys()
            .map(String::as_str)
            .collect::<BTreeSet<_>>();
        let probability_sum = prediction.probabilities.values().sum::<f64>();
        let mut ranking = model
            .labels
            .iter()
            .enumerate()
            .map(|(index, label)| (index, label, prediction.probabilities[label]))
            .collect::<Vec<_>>();
        ranking.sort_by(|(left_index, _, left), (right_index, _, right)| {
            right
                .total_cmp(left)
                .then_with(|| left_index.cmp(right_index))
        });
        let expected_margin = ranking[0].2 - ranking[1].2;
        if !id_ids.insert(prediction.id.as_str())
            || !label_set.contains(prediction.actual_label.as_str())
            || !label_set.contains(prediction.predicted_label.as_str())
            || prediction.predicted_label.as_str() != ranking[0].1.as_str()
            || prediction.predicted_label != reproduced_prediction.label
            || prediction.correct != (prediction.actual_label == prediction.predicted_label)
            || prediction.accepted != reproduced_prediction.accepted
            || probability_labels != label_set
            || prediction
                .probabilities
                .values()
                .any(|value| !valid_unit_interval(*value))
            || (probability_sum - 1.0).abs() > model.labels.len() as f64 * METRIC_REPORTING_UNIT
            || !approximately_equal(
                prediction.confidence,
                prediction.probabilities[&prediction.predicted_label],
            )
            || !approximately_equal(prediction.probability_margin, expected_margin)
            || !approximately_equal(prediction.confidence, reproduced_prediction.confidence)
            || !approximately_equal(
                prediction.probability_margin,
                reproduced_prediction.probability_margin,
            )
            || prediction.probabilities.len() != reproduced_prediction.probabilities.len()
            || prediction.probabilities.iter().any(|(label, probability)| {
                match reproduced_prediction.probabilities.get(label) {
                    Some(expected) => !approximately_equal(*probability, *expected),
                    None => true,
                }
            })
        {
            return Err(MlError::InvalidModel(
                "the ID-test prediction ledger is internally inconsistent".into(),
            ));
        }
    }
    let reproduced_id =
        summarize_id_predictions(&model.labels, metrics.id_test.predictions.clone());
    let risk_curve_matches = metrics.id_test.risk_coverage_curve.len()
        == reproduced_id.risk_coverage_curve.len()
        && metrics
            .id_test
            .risk_coverage_curve
            .iter()
            .zip(&reproduced_id.risk_coverage_curve)
            .all(|(recorded, reproduced)| {
                recorded.accepted == reproduced.accepted
                    && approximately_equal(recorded.coverage, reproduced.coverage)
                    && approximately_equal(recorded.risk, reproduced.risk)
            });
    if metrics.id_test.example_count != reproduced_id.example_count
        || !approximately_equal(metrics.id_test.accuracy, reproduced_id.accuracy)
        || !approximately_equal(metrics.id_test.macro_f1, reproduced_id.macro_f1)
        || metrics.id_test.labels != reproduced_id.labels
        || metrics.id_test.confusion_matrix != reproduced_id.confusion_matrix
        || !semantic_json_equal(&metrics.id_test.per_class, &reproduced_id.per_class)?
        || !approximately_equal(metrics.id_test.coverage, reproduced_id.coverage)
        || !optional_metric_matches(
            metrics.id_test.selective_accuracy,
            reproduced_id.selective_accuracy,
        )
        || !calibration_metrics_match(&metrics.id_test.calibration, &reproduced_id.calibration)
        || !approximately_equal(metrics.id_test.aurc, reproduced_id.aurc)
        || !risk_curve_matches
        || metrics.id_test.predictions != reproduced_id.predictions
        || metrics.id_test.example_count != split_counts[&PartitionKind::IdTest]
    {
        return Err(MlError::InvalidModel(
            "the ID-test summary does not match its prediction ledger".into(),
        ));
    }

    let reconstructed_training = split_plan
        .assignments
        .iter()
        .filter(|assignment| assignment.partition == PartitionKind::Train)
        .map(|assignment| GroupedExample {
            id: assignment.id.clone(),
            group_id: assignment.group_id.clone(),
            label: assignment.label.clone(),
            text: assignment.text.clone(),
        })
        .collect::<Vec<_>>();
    let reconstructed_id_test = split_plan
        .assignments
        .iter()
        .filter(|assignment| assignment.partition == PartitionKind::IdTest)
        .map(|assignment| GroupedExample {
            id: assignment.id.clone(),
            group_id: assignment.group_id.clone(),
            label: assignment.label.clone(),
            text: assignment.text.clone(),
        })
        .collect::<Vec<_>>();
    let baseline_training = TrainingPartition {
        examples: &reconstructed_training,
        dataset_sha256: &split_plan.dataset_sha256,
        split_plan_sha256: model.split_plan_sha256.clone(),
    };
    let reproduced_baselines = evaluate_baselines(
        &baseline_training,
        &reconstructed_id_test,
        &model.labels,
        reproduced_id.accuracy,
        reproduced_id.macro_f1,
    );
    if !semantic_json_equal(&metrics.baselines, &reproduced_baselines)? {
        return Err(MlError::InvalidModel(
            "the training-only baseline report cannot be reproduced from the experiment plan"
                .into(),
        ));
    }

    let reconstructed_ood_development = OpenSetOodDataset {
        examples: split_plan
            .ood_development
            .iter()
            .map(|row| OpenSetOodExample {
                id: row.id.clone(),
                family_id: row.family_id.clone(),
                domain_group: row.domain_group.clone(),
                stratum: row.stratum,
                text: row.text.clone(),
            })
            .collect(),
    };
    let reconstructed_ood_test = OpenSetOodDataset {
        examples: split_plan
            .ood_test
            .iter()
            .map(|row| OpenSetOodExample {
                id: row.id.clone(),
                family_id: row.family_id.clone(),
                domain_group: row.domain_group.clone(),
                stratum: row.stratum,
                text: row.text.clone(),
            })
            .collect(),
    };
    reconstructed_ood_development.validate_contract()?;
    reconstructed_ood_test.validate_contract()?;
    if reconstructed_ood_development.fingerprint_sha256() != metrics.ood_development_sha256
        || reconstructed_ood_test.fingerprint_sha256() != metrics.ood_test_sha256
    {
        return Err(MlError::InvalidModel(
            "the OOD populations disagree with their recorded provenance".into(),
        ));
    }

    let reconstructed_contrast = OpenSetContrastDataset {
        examples: split_plan
            .contrast_test
            .iter()
            .map(|row| OpenSetContrastExample {
                id: row.id.clone(),
                pair_id: row.pair_id.clone(),
                variant: row.variant,
                label: row.label.clone(),
                text: row.text.clone(),
            })
            .collect(),
    };
    reconstructed_contrast.validate_contract()?;
    let contrast_labels = reconstructed_contrast
        .examples()
        .iter()
        .map(|example| example.label.as_str())
        .collect::<BTreeSet<_>>();
    if contrast_labels != label_set
        || reconstructed_contrast.fingerprint_sha256() != metrics.contrast_test_sha256
    {
        return Err(MlError::InvalidModel(
            "the contrast-test population disagrees with the model labels or recorded provenance"
                .into(),
        ));
    }
    let reproduced_contrast =
        evaluate_contrast(&runtime, ContrastTestPartition(&reconstructed_contrast))?;
    if !semantic_json_equal(&metrics.contrast_test, &reproduced_contrast)? {
        return Err(MlError::InvalidModel(
            "the contrast-test report cannot be reproduced from the experiment plan".into(),
        ));
    }

    let mut ood_ids = HashSet::new();
    let ood_rows = split_plan
        .ood_test
        .iter()
        .map(|row| (row.id.as_str(), row))
        .collect::<BTreeMap<_, _>>();
    for prediction in &metrics.ood_test.predictions {
        let row = ood_rows.get(prediction.id.as_str()).ok_or_else(|| {
            MlError::InvalidModel(
                "the OOD-test prediction ledger disagrees with the experiment plan".into(),
            )
        })?;
        let reproduced_prediction = runtime.predict(&row.text);
        if id_ids.contains(prediction.id.as_str())
            || !ood_ids.insert(prediction.id.as_str())
            || prediction.family_id != row.family_id
            || prediction.domain_group != row.domain_group
            || prediction.stratum != row.stratum
            || !label_set.contains(prediction.predicted_label.as_str())
            || prediction.predicted_label != reproduced_prediction.label
            || prediction.accepted != reproduced_prediction.accepted
            || !valid_unit_interval(prediction.confidence)
            || !valid_unit_interval(prediction.probability_margin)
            || !approximately_equal(prediction.confidence, reproduced_prediction.confidence)
            || !approximately_equal(
                prediction.probability_margin,
                reproduced_prediction.probability_margin,
            )
        {
            return Err(MlError::InvalidModel(
                "the OOD-test prediction ledger is internally inconsistent".into(),
            ));
        }
    }
    let accepted_ood = metrics
        .ood_test
        .predictions
        .iter()
        .filter(|prediction| prediction.accepted)
        .count();
    let id_scores = metrics
        .id_test
        .predictions
        .iter()
        .map(|prediction| prediction.confidence)
        .collect::<Vec<_>>();
    let ood_scores = metrics
        .ood_test
        .predictions
        .iter()
        .map(|prediction| prediction.confidence)
        .collect::<Vec<_>>();
    let mut reproduced_by_stratum = BTreeMap::new();
    for stratum in [
        OodStratum::Semantic,
        OodStratum::Capability,
        OodStratum::Noise,
    ] {
        let members = metrics
            .ood_test
            .predictions
            .iter()
            .filter(|prediction| prediction.stratum == stratum)
            .collect::<Vec<_>>();
        let accepted = members
            .iter()
            .filter(|prediction| prediction.accepted)
            .count();
        let scores = members
            .iter()
            .map(|prediction| prediction.confidence)
            .collect::<Vec<_>>();
        reproduced_by_stratum.insert(
            stratum.as_str().to_owned(),
            OodStratumEvaluation {
                example_count: members.len(),
                accepted_examples: accepted,
                coverage: safe_ratio(accepted as f64, members.len() as f64),
                discrimination: discrimination_metrics(&id_scores, &scores),
            },
        );
    }
    if metrics.ood_test.example_count == 0
        || metrics.ood_test.example_count != metrics.ood_test.predictions.len()
        || metrics.ood_test.example_count != split_plan.ood_test.len()
        || metrics.ood_test.accepted_examples != accepted_ood
        || !approximately_equal(
            metrics.ood_test.coverage,
            accepted_ood as f64 / metrics.ood_test.example_count as f64,
        )
        || !discrimination_metrics_match(
            &metrics.ood_test.discrimination,
            &discrimination_metrics(&id_scores, &ood_scores),
        )
        || metrics.ood_test.by_stratum.len() != reproduced_by_stratum.len()
        || metrics
            .ood_test
            .by_stratum
            .iter()
            .any(|(stratum, recorded)| {
                reproduced_by_stratum
                    .get(stratum)
                    .map_or(true, |reproduced| {
                        recorded.example_count != reproduced.example_count
                            || recorded.accepted_examples != reproduced.accepted_examples
                            || !approximately_equal(recorded.coverage, reproduced.coverage)
                            || !discrimination_metrics_match(
                                &recorded.discrimination,
                                &reproduced.discrimination,
                            )
                    })
            })
    {
        return Err(MlError::InvalidModel(
            "the OOD-test summary does not match its prediction ledger".into(),
        ));
    }

    let bootstrap = &metrics.bootstrap_95;
    if bootstrap.strategy != "label-stratified-id-family-and-ood-domain-cluster-percentile-v3"
        || bootstrap.seed != model.training_config.seed
        || !(100..=20_000).contains(&bootstrap.resamples)
        || bootstrap.confidence_level != 0.95
        || metrics.limitations.is_empty()
    {
        return Err(MlError::InvalidModel(
            "the bootstrap report has invalid provenance".into(),
        ));
    }
    for (estimate, value, minimum, maximum) in [
        (&bootstrap.id_accuracy, metrics.id_test.accuracy, 0.0, 1.0),
        (&bootstrap.id_macro_f1, metrics.id_test.macro_f1, 0.0, 1.0),
        (
            &bootstrap.id_negative_log_likelihood,
            metrics.id_test.calibration.negative_log_likelihood,
            0.0,
            f64::INFINITY,
        ),
        (
            &bootstrap.id_multiclass_brier,
            metrics.id_test.calibration.multiclass_brier,
            0.0,
            2.0,
        ),
        (
            &bootstrap.id_expected_calibration_error,
            metrics.id_test.calibration.expected_calibration_error,
            0.0,
            1.0,
        ),
        (&bootstrap.id_aurc, metrics.id_test.aurc, 0.0, 1.0),
        (
            &bootstrap.ood_auroc,
            metrics.ood_test.discrimination.auroc,
            0.0,
            1.0,
        ),
        (
            &bootstrap.ood_aupr_in_domain,
            metrics.ood_test.discrimination.aupr_in_domain,
            0.0,
            1.0,
        ),
        (
            &bootstrap.ood_fpr_at_95_tpr,
            metrics.ood_test.discrimination.fpr_at_95_tpr,
            0.0,
            1.0,
        ),
    ] {
        if !approximately_equal(estimate.value, value)
            || !estimate.value.is_finite()
            || !estimate.lower_95.is_finite()
            || !estimate.upper_95.is_finite()
            || estimate.lower_95 < minimum
            || estimate.upper_95 > maximum
            || estimate.lower_95 > estimate.upper_95
        {
            return Err(MlError::InvalidModel(
                "a bootstrap estimate violates its metric contract".into(),
            ));
        }
    }
    Ok(())
}

fn validate_calibration_metrics(metrics: &CalibrationMetrics) -> Result<(), MlError> {
    if !metrics.negative_log_likelihood.is_finite()
        || metrics.negative_log_likelihood < 0.0
        || !metrics.multiclass_brier.is_finite()
        || !(0.0..=2.0).contains(&metrics.multiclass_brier)
        || !valid_unit_interval(metrics.expected_calibration_error)
        || metrics.ece_bins == 0
    {
        return Err(MlError::InvalidModel(
            "calibration metrics are outside their supported range".into(),
        ));
    }
    Ok(())
}

fn valid_unit_interval(value: f64) -> bool {
    value.is_finite() && (0.0..=1.0).contains(&value)
}

fn approximately_equal(left: f64, right: f64) -> bool {
    (left - right).abs() <= METRIC_COMPARISON_TOLERANCE
}

fn optional_metric_matches(left: Option<f64>, right: Option<f64>) -> bool {
    match (left, right) {
        (Some(left), Some(right)) => approximately_equal(left, right),
        (None, None) => true,
        _ => false,
    }
}

fn calibration_metrics_match(left: &CalibrationMetrics, right: &CalibrationMetrics) -> bool {
    approximately_equal(left.negative_log_likelihood, right.negative_log_likelihood)
        && approximately_equal(left.multiclass_brier, right.multiclass_brier)
        && approximately_equal(
            left.expected_calibration_error,
            right.expected_calibration_error,
        )
        && left.ece_bins == right.ece_bins
}

fn discrimination_metrics_match(
    left: &OodDiscriminationMetrics,
    right: &OodDiscriminationMetrics,
) -> bool {
    approximately_equal(left.auroc, right.auroc)
        && approximately_equal(left.aupr_in_domain, right.aupr_in_domain)
        && approximately_equal(left.fpr_at_95_tpr, right.fpr_at_95_tpr)
}

pub fn reproduce_bundle(
    directory: impl AsRef<Path>,
    dataset: &GroupedDataset,
    ood_development: &OpenSetOodDataset,
    ood_test: &OpenSetOodDataset,
    contrast_test: &OpenSetContrastDataset,
) -> Result<(), MlError> {
    let verified = verify_bundle(directory)?;
    let resamples = verified.metrics.bootstrap_95.resamples;
    let reproduced = run_open_set_experiment(
        dataset,
        ood_development,
        ood_test,
        contrast_test,
        verified.model.training_config.clone(),
        resamples,
    )?;
    let expected_payloads = BTreeMap::from([
        ("metrics.json", canonical_metrics_json(&reproduced.metrics)?),
        ("model.json", canonical_json(&reproduced.model)?),
        ("policy.json", canonical_json(&reproduced.policy)?),
        ("split-plan.json", canonical_json(&reproduced.split_plan)?),
    ]);
    for (name, bytes) in expected_payloads {
        let expected =
            verified.manifest.files.get(name).ok_or_else(|| {
                MlError::InvalidModel(format!("bundle manifest is missing `{name}`"))
            })?;
        if sha256_hex(&bytes) != *expected {
            return Err(MlError::InvalidModel(format!(
                "reproduced `{name}` does not match the verified bundle"
            )));
        }
    }
    Ok(())
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct BatchInput {
    id: String,
    text: String,
}

#[derive(Debug, Serialize)]
struct BatchOutput<'a> {
    id: &'a str,
    prediction: OpenSetPrediction,
}

#[derive(Clone, Copy)]
struct BatchLimits {
    maximum_rows: usize,
    maximum_physical_lines: usize,
    maximum_line_bytes: usize,
    maximum_total_bytes: usize,
}

const DEFAULT_BATCH_LIMITS: BatchLimits = BatchLimits {
    maximum_rows: MAX_EXAMPLES,
    maximum_physical_lines: MAX_EXAMPLES,
    maximum_line_bytes: MAX_JSONL_BYTES,
    maximum_total_bytes: 64 * 1024 * 1024,
};

/// Streams bounded JSONL predictions. Errors stop immediately, without draining the input.
/// A writer can contain a valid prefix after failure; callers must check the result before
/// treating its contents as a complete batch. Diagnostics never include submitted values.
pub fn predict_jsonl(
    runtime: &CompiledModel,
    reader: &mut impl BufRead,
    writer: &mut impl Write,
) -> Result<usize, MlError> {
    predict_jsonl_with_limits(runtime, reader, writer, DEFAULT_BATCH_LIMITS)
}

fn predict_jsonl_with_limits(
    runtime: &CompiledModel,
    reader: &mut impl BufRead,
    writer: &mut impl Write,
    limits: BatchLimits,
) -> Result<usize, MlError> {
    let mut count = 0usize;
    let mut physical_lines = 0usize;
    let mut total_bytes = 0usize;
    let mut ids = HashSet::new();
    let mut line = String::new();
    loop {
        line.clear();
        // Probe at most one byte beyond either remaining budget. Never drain a
        // rejected line: stdin may be an unending stream with no next newline.
        let remaining_bytes = limits.maximum_total_bytes.saturating_sub(total_bytes);
        let read_limit = limits.maximum_line_bytes.min(remaining_bytes) + 1;
        let bytes = reader.take(read_limit as u64).read_line(&mut line)?;
        if bytes == 0 {
            break;
        }
        physical_lines += 1;
        if physical_lines > limits.maximum_physical_lines {
            return Err(MlError::InvalidDataset(format!(
                "JSONL input exceeds {} physical lines",
                limits.maximum_physical_lines
            )));
        }
        if bytes > limits.maximum_line_bytes {
            return Err(MlError::InvalidDataset(format!(
                "JSONL line {physical_lines} exceeds the input boundary"
            )));
        }
        if bytes > remaining_bytes {
            return Err(MlError::InvalidDataset(format!(
                "JSONL input exceeds the {} byte total boundary",
                limits.maximum_total_bytes
            )));
        }
        total_bytes += bytes;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if count >= limits.maximum_rows {
            return Err(MlError::InvalidDataset(format!(
                "JSONL input exceeds {} rows",
                limits.maximum_rows
            )));
        }
        let input: BatchInput = serde_json::from_str(trimmed).map_err(|_| {
            MlError::InvalidDataset(format!(
                "JSONL line {physical_lines} does not match the required object schema"
            ))
        })?;
        validate_identifier(&input.id, "batch id", physical_lines)?;
        validate_text(&input.text, physical_lines, "batch")?;
        if !ids.insert(input.id.clone()) {
            return Err(MlError::InvalidDataset(format!(
                "JSONL line {physical_lines} repeats a batch id"
            )));
        }
        serde_json::to_writer(
            &mut *writer,
            &BatchOutput {
                id: &input.id,
                prediction: runtime.predict(&input.text),
            },
        )?;
        writer.write_all(b"\n")?;
        count += 1;
    }
    Ok(count)
}

fn reject_cross_dataset_overlap(
    dataset: &GroupedDataset,
    ood_development: &OpenSetOodDataset,
    ood_test: &OpenSetOodDataset,
    contrast_test: &OpenSetContrastDataset,
) -> Result<(), MlError> {
    dataset.validate_family_contract()?;
    ood_development.validate_contract()?;
    ood_test.validate_contract()?;
    contrast_test.validate_contract()?;
    let supervised_ids = dataset
        .examples()
        .iter()
        .map(|example| example.id.as_str())
        .collect::<HashSet<_>>();
    let supervised_groups = dataset
        .examples()
        .iter()
        .map(|example| example.group_id.as_str())
        .collect::<HashSet<_>>();
    let supervised_texts = dataset
        .examples()
        .iter()
        .map(|example| feature_identity(&example.text))
        .collect::<HashSet<_>>();
    let mut ood_ids = HashSet::new();
    let mut ood_texts = HashSet::new();
    let development_families = ood_development
        .examples()
        .iter()
        .map(|example| example.family_id.as_str())
        .collect::<HashSet<_>>();
    let test_families = ood_test
        .examples()
        .iter()
        .map(|example| example.family_id.as_str())
        .collect::<HashSet<_>>();
    let development_domains = ood_development
        .examples()
        .iter()
        .map(|example| example.domain_group.as_str())
        .collect::<HashSet<_>>();
    let test_domains = ood_test
        .examples()
        .iter()
        .map(|example| example.domain_group.as_str())
        .collect::<HashSet<_>>();
    if development_families
        .iter()
        .any(|family| test_families.contains(family))
    {
        return Err(MlError::InvalidDataset(
            "OOD development and OOD test overlap by domain family".into(),
        ));
    }
    if development_domains
        .iter()
        .any(|domain| test_domains.contains(domain))
    {
        return Err(MlError::InvalidDataset(
            "OOD development and OOD test overlap by broader domain group".into(),
        ));
    }
    for (name, ood) in [("OOD development", ood_development), ("OOD test", ood_test)] {
        for example in ood.examples() {
            let normalized = feature_identity(&example.text);
            if supervised_ids.contains(example.id.as_str())
                || supervised_groups.contains(example.family_id.as_str())
                || supervised_groups.contains(example.domain_group.as_str())
                || supervised_texts.contains(&normalized)
                || !ood_ids.insert(example.id.as_str())
                || !ood_texts.insert(normalized)
            {
                return Err(MlError::InvalidDataset(format!(
                    "{name} example `{}` overlaps another experimental population",
                    example.id
                )));
            }
        }
    }
    let ood_families_and_domains = ood_development
        .examples()
        .iter()
        .chain(ood_test.examples())
        .flat_map(|example| [example.family_id.as_str(), example.domain_group.as_str()])
        .collect::<HashSet<_>>();
    for example in contrast_test.examples() {
        let normalized = feature_identity(&example.text);
        if supervised_ids.contains(example.id.as_str())
            || supervised_groups.contains(example.pair_id.as_str())
            || ood_ids.contains(example.id.as_str())
            || ood_families_and_domains.contains(example.pair_id.as_str())
            || supervised_texts.contains(&normalized)
            || ood_texts.contains(&normalized)
        {
            return Err(MlError::InvalidDataset(format!(
                "contrast-test example `{}` overlaps another experimental population",
                example.id
            )));
        }
    }
    Ok(())
}

fn validate_identifier(value: &str, name: &str, line: usize) -> Result<(), MlError> {
    if value.is_empty()
        || value.len() > 128
        || !value
            .chars()
            .all(|character| character.is_ascii_alphanumeric() || matches!(character, '-' | '_'))
    {
        return Err(MlError::InvalidDataset(format!(
            "line {line} has an invalid {name}"
        )));
    }
    Ok(())
}

fn validate_label(value: &str, line: usize) -> Result<(), MlError> {
    if value.is_empty()
        || value.len() > 64
        || !value
            .chars()
            .all(|character| character.is_ascii_lowercase() || character == '-')
    {
        return Err(MlError::InvalidDataset(format!(
            "line {line} has an invalid label"
        )));
    }
    Ok(())
}

fn validate_text(value: &str, line: usize, context: &str) -> Result<(), MlError> {
    if value.trim().is_empty() || value.chars().count() > crate::MAX_INPUT_CHARS {
        return Err(MlError::InvalidDataset(format!(
            "{context} line {line} has empty or oversized text"
        )));
    }
    Ok(())
}

fn validate_vectorizer_config(config: &VectorizerConfig) -> Result<(), MlError> {
    if config.word_ngram_min == 0
        || config.word_ngram_min > config.word_ngram_max
        || config.word_ngram_max > 3
        || config.char_ngram_min < 2
        || config.char_ngram_min > config.char_ngram_max
        || config.char_ngram_max > 6
        || config.min_document_frequency == 0
        || config.min_document_frequency > 1_000_000
        || !(32..=100_000).contains(&config.max_features)
    {
        return Err(MlError::InvalidConfiguration(
            "the open-set vectorizer configuration is outside its supported bounds".into(),
        ));
    }
    Ok(())
}

fn normalize_text(value: &str) -> String {
    let lowercase = value
        .nfkc()
        .flat_map(char::to_lowercase)
        .map(|character| match character {
            '’' | '‘' => '\'',
            _ => character,
        })
        .collect::<String>();
    lowercase.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn feature_identity(value: &str) -> String {
    tokenize(value).join(" ")
}

fn tokenize(value: &str) -> Vec<String> {
    normalize_text(value)
        .split(|character: char| !character.is_alphanumeric() && character != '\'')
        .map(|word| word.trim_matches('\''))
        .filter(|word| !word.is_empty())
        .map(str::to_owned)
        .collect()
}

fn extract_terms(text: &str, config: &VectorizerConfig) -> Vec<String> {
    let tokens = tokenize(text);
    let mut terms = Vec::new();
    for size in config.word_ngram_min..=config.word_ngram_max {
        for window in tokens.windows(size) {
            terms.push(format!("w{size}:{}", window.join("_")));
        }
    }
    let normalized = format!("^{}$", tokens.join(" "));
    let characters = normalized.chars().collect::<Vec<_>>();
    for size in config.char_ngram_min..=config.char_ngram_max {
        for window in characters.windows(size) {
            terms.push(format!("c{size}:{}", window.iter().collect::<String>()));
        }
    }
    terms
}

fn build_feature_index(vocabulary: &[String]) -> HashMap<String, usize> {
    vocabulary
        .iter()
        .cloned()
        .enumerate()
        .map(|(index, feature)| (feature, index))
        .collect()
}

fn transform(
    vectorizer: &OpenSetVectorizer,
    index: &HashMap<String, usize>,
    text: &str,
) -> Vec<(usize, f64)> {
    let mut counts: HashMap<usize, usize> = HashMap::new();
    for term in extract_terms(text, &vectorizer.config) {
        if let Some(position) = index.get(&term) {
            *counts.entry(*position).or_insert(0) += 1;
        }
    }
    let mut values = counts
        .into_iter()
        .map(|(position, count)| {
            (
                position,
                (1.0 + (count as f64).ln()) * vectorizer.inverse_document_frequency[position],
            )
        })
        .collect::<Vec<_>>();
    values.sort_by_key(|(position, _)| *position);
    let norm = values
        .iter()
        .map(|(_, value)| value * value)
        .sum::<f64>()
        .sqrt();
    if norm > 0.0 {
        for (_, value) in &mut values {
            *value /= norm;
        }
    }
    values
}

fn logits_for(features: &[(usize, f64)], weights: &[Vec<f64>], biases: &[f64]) -> Vec<f64> {
    weights
        .iter()
        .zip(biases)
        .map(|(row, bias)| {
            *bias
                + features
                    .iter()
                    .map(|(feature, value)| row[*feature] * value)
                    .sum::<f64>()
        })
        .collect()
}

fn softmax(logits: &[f64], temperature: f64) -> Vec<f64> {
    let scaled = logits
        .iter()
        .map(|logit| *logit / temperature)
        .collect::<Vec<_>>();
    let maximum = scaled.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut exponentials = scaled
        .iter()
        .map(|logit| (*logit - maximum).exp())
        .collect::<Vec<_>>();
    let total = exponentials.iter().sum::<f64>();
    for probability in &mut exponentials {
        *probability /= total;
    }
    exponentials
}

fn quantize(value: f64) -> f64 {
    (value * 1_000_000_000_000.0).round() / 1_000_000_000_000.0
}

fn safe_ratio(numerator: f64, denominator: f64) -> f64 {
    if denominator == 0.0 {
        0.0
    } else {
        numerator / denominator
    }
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut output = String::with_capacity(digest.len() * 2);
    for byte in digest {
        write!(output, "{byte:02x}").expect("writing to a String cannot fail");
    }
    output
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || matches!(byte, b'a'..=b'f'))
}

fn canonical_json<T: Serialize>(value: &T) -> Result<Vec<u8>, MlError> {
    Ok(format!("{}\n", serde_json::to_string_pretty(value)?).into_bytes())
}

fn canonical_metrics_json(metrics: &OpenSetMetricsV3) -> Result<Vec<u8>, MlError> {
    fn quantize_floats(value: &mut serde_json::Value) -> Result<(), MlError> {
        match value {
            serde_json::Value::Number(number) if number.is_f64() => {
                let value = number.as_f64().ok_or_else(|| {
                    MlError::InvalidModel("a metric is not representable as f64".into())
                })?;
                let quantized = (value * METRIC_REPORTING_SCALE).round() / METRIC_REPORTING_SCALE;
                *number = serde_json::Number::from_f64(quantized).ok_or_else(|| {
                    MlError::InvalidModel("a metric is not a finite JSON number".into())
                })?;
            }
            serde_json::Value::Array(values) => {
                for value in values {
                    quantize_floats(value)?;
                }
            }
            serde_json::Value::Object(values) => {
                for value in values.values_mut() {
                    quantize_floats(value)?;
                }
            }
            _ => {}
        }
        Ok(())
    }

    let mut value = serde_json::to_value(metrics)?;
    quantize_floats(&mut value)?;
    canonical_json(&value)
}

fn group_split_hash(label: &str, group_id: &str, seed: u64) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64 ^ seed;
    for byte in label
        .as_bytes()
        .iter()
        .chain([0x1f].iter())
        .chain(group_id.as_bytes())
    {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

struct DeterministicRng {
    state: u64,
}

impl DeterministicRng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }

    fn index(&mut self, length: usize) -> usize {
        (self.next_u64() % length as u64) as usize
    }
}

fn reject_oversized_file(path: &Path, maximum: u64, context: &str) -> Result<(), MlError> {
    let metadata = fs::symlink_metadata(path)?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(MlError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("{context} {} must be a regular file", path.display()),
        )));
    }
    if metadata.len() > maximum {
        return Err(MlError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("{context} exceeds the {maximum}-byte boundary"),
        )));
    }
    Ok(())
}

fn reject_symlink_or_non_file(path: &Path, name: &str) -> Result<(), MlError> {
    let metadata = fs::symlink_metadata(path)?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(MlError::InvalidModel(format!(
            "bundle file `{name}` must be a regular file"
        )));
    }
    Ok(())
}

fn read_bounded_json<T: for<'de> Deserialize<'de>>(
    path: &Path,
    context: &str,
) -> Result<T, MlError> {
    reject_oversized_file(path, MAX_JSON_BYTES, context)?;
    Ok(serde_json::from_slice(&fs::read(path)?)?)
}

fn write_atomic(path: &Path, bytes: &[u8]) -> Result<(), MlError> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent)?;
    if path.exists() && !path.is_file() {
        return Err(MlError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("artifact destination {} is not a file", path.display()),
        )));
    }
    let sequence = BUNDLE_TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    let file_name = path
        .file_name()
        .ok_or_else(|| {
            MlError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "artifact destination has no file name",
            ))
        })?
        .to_string_lossy();
    let temporary = parent.join(format!(
        ".{file_name}.tmp-{}-{sequence}",
        std::process::id()
    ));
    let result = (|| -> Result<(), std::io::Error> {
        let mut file = fs::OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&temporary)?;
        file.write_all(bytes)?;
        file.flush()?;
        file.sync_all()?;
        if path.exists() {
            fs::remove_file(path)?;
        }
        fs::rename(&temporary, path)?;
        Ok(())
    })();
    if let Err(error) = result {
        let _ = fs::remove_file(&temporary);
        return Err(MlError::Io(error));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::sync::OnceLock;

    struct TestDirectory(PathBuf);

    impl TestDirectory {
        fn new(name: &str) -> Self {
            let sequence = BUNDLE_TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "eliza-open-set-{name}-{}-{sequence}",
                std::process::id()
            ));
            fs::create_dir_all(&path).unwrap();
            Self(path)
        }
    }

    impl Drop for TestDirectory {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn bundled_experiment(resamples: usize) -> OpenSetExperimentResult {
        static RESULT: OnceLock<OpenSetExperimentResult> = OnceLock::new();
        assert_eq!(resamples, 100);
        RESULT
            .get_or_init(|| {
                let mut config = OpenSetTrainingConfig::default();
                config.development_selection.max_features_candidates =
                    vec![config.vectorizer.max_features];
                config.development_selection.l2_penalty_candidates = vec![config.l2_penalty];
                run_open_set_experiment(
                    &GroupedDataset::bundled().unwrap(),
                    &OpenSetOodDataset::bundled_development().unwrap(),
                    &OpenSetOodDataset::bundled_test().unwrap(),
                    &OpenSetContrastDataset::bundled_test().unwrap(),
                    config,
                    resamples,
                )
                .unwrap()
            })
            .clone()
    }

    #[test]
    fn split_plan_is_group_disjoint_complete_and_deterministic() {
        let dataset = GroupedDataset::bundled().unwrap();
        let ood_development = OpenSetOodDataset::bundled_development().unwrap();
        let ood_test = OpenSetOodDataset::bundled_test().unwrap();
        let contrast = OpenSetContrastDataset::bundled_test().unwrap();
        let left =
            SplitPlan::build(&dataset, &ood_development, &ood_test, &contrast, 20_260_722).unwrap();
        let right =
            SplitPlan::build(&dataset, &ood_development, &ood_test, &contrast, 20_260_722).unwrap();
        assert_eq!(left.manifest, right.manifest);
        assert_eq!(left.manifest.assignments.len(), dataset.examples().len());

        let mut group_partitions = HashMap::new();
        for assignment in &left.manifest.assignments {
            if let Some(previous) =
                group_partitions.insert(&assignment.group_id, assignment.partition)
            {
                assert_eq!(previous, assignment.partition);
            }
        }
        assert_eq!(left.train().len(), 315);
        assert_eq!(left.development().len(), 70);
        assert_eq!(left.calibration().len(), 70);
        assert_eq!(left.id_test().len(), 70);
        assert_eq!(left.manifest.contrast_test.len(), 28);
        assert_eq!(
            left.manifest
                .contrast_test
                .iter()
                .map(|row| row.pair_id.as_str())
                .collect::<HashSet<_>>()
                .len(),
            14
        );
    }

    #[test]
    fn split_group_quotas_scale_without_emptying_training() {
        assert_eq!(evaluation_group_quota(4), 1);
        assert_eq!(evaluation_group_quota(8), 1);
        assert_eq!(evaluation_group_quota(15), 2);
        assert_eq!(evaluation_group_quota(100), 10);
        assert_eq!(evaluation_group_quota(20_000), 2_000);
    }

    #[test]
    fn split_manifest_recomputes_the_declared_hash_and_quota_strategy() {
        let dataset = GroupedDataset::bundled().unwrap();
        let plan = SplitPlan::build(
            &dataset,
            &OpenSetOodDataset::bundled_development().unwrap(),
            &OpenSetOodDataset::bundled_test().unwrap(),
            &OpenSetContrastDataset::bundled_test().unwrap(),
            20_260_722,
        )
        .unwrap();
        let mut manifest = plan.manifest.clone();
        let moved_group = manifest
            .assignments
            .iter()
            .find(|assignment| assignment.partition == PartitionKind::Train)
            .unwrap()
            .group_id
            .clone();
        for assignment in &mut manifest.assignments {
            if assignment.group_id == moved_group {
                assignment.partition = PartitionKind::IdTest;
            }
        }
        let error = manifest.validate_contract().unwrap_err();
        assert!(error.to_string().contains("declared strategy"));
    }

    #[test]
    fn independent_ood_populations_cannot_overlap() {
        let dataset = GroupedDataset::bundled().unwrap();
        let development = OpenSetOodDataset::bundled_development().unwrap();
        let mut test = OpenSetOodDataset::bundled_test().unwrap();
        test.examples[0].text = development.examples()[0].text.clone();
        let contrast = OpenSetContrastDataset::bundled_test().unwrap();
        assert!(reject_cross_dataset_overlap(&dataset, &development, &test, &contrast).is_err());

        let mut same_family = OpenSetOodDataset::bundled_test().unwrap();
        let original_family = same_family.examples[0].family_id.clone();
        let overlapping_family = development.examples()[0].family_id.clone();
        for example in &mut same_family.examples {
            if example.family_id == original_family {
                example.family_id.clone_from(&overlapping_family);
            }
        }
        same_family.validate_contract().unwrap();
        let error = reject_cross_dataset_overlap(&dataset, &development, &same_family, &contrast)
            .unwrap_err();
        assert!(error.to_string().contains("domain family"));
    }

    #[test]
    fn feature_equivalent_texts_are_rejected_before_splitting() {
        let input = "id\tgroup_id\tlabel\ttext\n\
                     a1\ta-1\talpha\tHello there\n\
                     a2\ta-2\talpha\tHello, there\n\
                     a3\ta-3\talpha\tAlpha three\n\
                     a4\ta-4\talpha\tAlpha four\n\
                     b1\tb-1\tbeta\tBeta one\n\
                     b2\tb-2\tbeta\tBeta two\n\
                     b3\tb-3\tbeta\tBeta three\n\
                     b4\tb-4\tbeta\tBeta four\n";
        let error = GroupedDataset::from_tsv(input).unwrap_err();
        assert!(error.to_string().contains("feature-equivalent"));
    }

    #[test]
    fn supervised_family_contract_rejects_uneven_support() {
        let examples = [
            ("a1", "family-a", "first alpha prompt"),
            ("a2", "family-a", "second alpha prompt"),
            ("a3", "family-a", "third alpha prompt"),
            ("b1", "family-b", "first beta prompt"),
            ("b2", "family-b", "second beta prompt"),
            ("b3", "family-b", "third beta prompt"),
            ("b4", "family-b", "fourth beta prompt"),
        ]
        .into_iter()
        .map(|(id, group_id, text)| GroupedExample {
            id: id.into(),
            group_id: group_id.into(),
            label: "label".into(),
            text: text.into(),
        })
        .collect::<Vec<_>>();
        let error = GroupedDataset { examples }
            .validate_family_contract()
            .unwrap_err();
        assert!(error.to_string().contains("same number of examples"));
    }

    #[test]
    fn supervised_family_contract_rejects_raw_near_duplicates() {
        let examples = [
            ("a1", "family-a", "alpha beta gamma delta"),
            ("a2", "family-a", "a remote sentence about copper"),
            ("a3", "family-a", "another unrelated sentence about glass"),
            ("b1", "family-b", "alpha beta gamma delta epsilon"),
            ("b2", "family-b", "a separate phrase concerning forests"),
            ("b3", "family-b", "an independent phrase concerning rivers"),
        ]
        .into_iter()
        .map(|(id, group_id, text)| GroupedExample {
            id: id.into(),
            group_id: group_id.into(),
            label: "label".into(),
            text: text.into(),
        })
        .collect::<Vec<_>>();
        let error = GroupedDataset { examples }
            .validate_family_contract()
            .unwrap_err();
        assert!(error.to_string().contains("raw feature Jaccard"));
    }

    #[test]
    fn similarity_candidate_index_fails_closed_at_its_budget() {
        let examples = (0..4)
            .map(|index| GroupedExample {
                id: format!("id-{index}"),
                group_id: format!("family-{index}"),
                label: "label".into(),
                text: "unused".into(),
            })
            .collect::<Vec<_>>();
        let feature_sets = (0..4)
            .map(|_| HashSet::from(["shared".to_owned()]))
            .collect::<Vec<_>>();
        let mut candidates = HashSet::new();
        let mut attempts = 0;
        let error = collect_similarity_candidates(
            &examples,
            &feature_sets,
            0.30,
            false,
            &mut candidates,
            &mut attempts,
            SimilarityCandidateBudget {
                maximum_candidate_pairs: 2,
                maximum_pair_insert_attempts: 10,
            },
        )
        .unwrap_err();
        assert!(error.to_string().contains("bounded candidate budget"));
    }

    #[test]
    fn epsilon_selection_prefers_the_simpler_candidate() {
        let current = DevelopmentCandidateMetrics {
            max_features: 2_048,
            l2_penalty: 0.0001,
            accuracy: 0.90,
            macro_f1: 0.900,
            negative_log_likelihood: 0.30,
            multiclass_brier: 0.20,
        };
        let simpler = DevelopmentCandidateMetrics {
            max_features: 512,
            l2_penalty: 0.002,
            accuracy: 0.89,
            macro_f1: 0.896,
            negative_log_likelihood: 0.32,
            multiclass_brier: 0.22,
        };
        assert!(development_candidate_is_better(&simpler, &current, 0.005));

        let materially_worse = DevelopmentCandidateMetrics {
            macro_f1: 0.894,
            ..simpler
        };
        assert!(!development_candidate_is_better(
            &materially_worse,
            &current,
            0.005
        ));
    }

    #[test]
    fn baselines_are_deterministic_and_train_only() {
        let training_examples = [
            ("a1", "family-a1", "alpha", "red apple"),
            ("a2", "family-a2", "alpha", "scarlet apple"),
            ("b1", "family-b1", "beta", "blue ocean"),
            ("b2", "family-b2", "beta", "navy ocean"),
        ]
        .into_iter()
        .map(|(id, group_id, label, text)| GroupedExample {
            id: id.into(),
            group_id: group_id.into(),
            label: label.into(),
            text: text.into(),
        })
        .collect::<Vec<_>>();
        let id_test = [
            ("ta", "test-a", "alpha", "red fruit"),
            ("tb", "test-b", "beta", "blue water"),
        ]
        .into_iter()
        .map(|(id, group_id, label, text)| GroupedExample {
            id: id.into(),
            group_id: group_id.into(),
            label: label.into(),
            text: text.into(),
        })
        .collect::<Vec<_>>();
        let dataset_hash = "0".repeat(64);
        let training = TrainingPartition {
            examples: &training_examples,
            dataset_sha256: &dataset_hash,
            split_plan_sha256: "1".repeat(64),
        };
        let labels = vec!["alpha".into(), "beta".into()];
        let first = evaluate_baselines(&training, &id_test, &labels, 1.0, 1.0);
        let second = evaluate_baselines(&training, &id_test, &labels, 1.0, 1.0);
        assert_eq!(first, second);
        assert_eq!(first.inputs, [SelectionDataRole::Train]);
        assert_eq!(first.majority_label, "alpha");
        assert_eq!(first.majority.accuracy, 0.5);
        assert_eq!(first.unigram_naive_bayes.accuracy, 1.0);
        assert_eq!(first.learned_minus_unigram_accuracy, 0.0);
    }

    #[test]
    fn ood_contract_requires_balanced_multi_prompt_strata() {
        let mut development = OpenSetOodDataset::bundled_development().unwrap();
        development.examples[0].family_id = "singleton-family".into();
        let error = development.validate_contract().unwrap_err();
        assert!(error.to_string().contains("equal multi-prompt support"));
    }

    #[test]
    fn contrast_contract_requires_label_changing_balanced_pairs() {
        let mut contrast = OpenSetContrastDataset::bundled_test().unwrap();
        contrast.examples[1].label = contrast.examples[0].label.clone();
        let error = contrast.validate_contract().unwrap_err();
        assert!(error.to_string().contains("different labels"));
    }

    #[test]
    fn temperature_scaling_does_not_increase_calibration_nll() {
        let result = bundled_experiment(100);
        assert!(
            result
                .metrics
                .calibrated_calibration_partition
                .negative_log_likelihood
                <= result
                    .metrics
                    .uncalibrated_calibration_partition
                    .negative_log_likelihood
                    + 1e-12
        );
        assert_eq!(
            result.metrics.threshold_selection.inputs,
            [
                SelectionDataRole::Development,
                SelectionDataRole::OodDevelopment
            ]
        );
    }

    #[test]
    fn threshold_search_cannot_turn_an_oov_row_into_coverage() {
        let result = bundled_experiment(100);
        let runtime = CompiledModel::new(result.model.clone(), result.policy.clone()).unwrap();
        let oov_prediction = runtime.predict("🪐");
        assert!(!oov_prediction.accepted);
        let development = vec![GroupedExample {
            id: "oov-development".into(),
            group_id: "oov-development-group".into(),
            label: oov_prediction.label,
            text: "🪐".into(),
        }];
        assert!(select_thresholds(
            &result.model,
            result.policy.temperature,
            DevelopmentPartition(&development),
            OodDevelopmentPartition(&OpenSetOodDataset::bundled_development().unwrap()),
        )
        .is_err());
    }

    #[test]
    fn contrastive_explanation_reconstructs_the_top_two_logit_margin() {
        let result = bundled_experiment(100);
        let runtime = CompiledModel::new(result.model, result.policy).unwrap();
        for text in [
            "Today I feel calm",
            "I want to finish the prototype",
            "Calculate an orbital period",
        ] {
            let prediction = runtime.predict(text);
            assert!(
                (prediction.logit_margin - prediction.explanation.reconstructed_logit_margin).abs()
                    <= 1e-10,
                "{text}: {} != {}",
                prediction.logit_margin,
                prediction.explanation.reconstructed_logit_margin
            );
            assert_eq!(prediction.label, prediction.explanation.top_label);
            assert_eq!(
                prediction.runner_up_label,
                prediction.explanation.runner_up_label
            );
        }
    }

    #[test]
    fn compiled_batch_matches_individual_prediction() {
        let result = bundled_experiment(100);
        let runtime = CompiledModel::new(result.model, result.policy).unwrap();
        let inputs = [
            "Hello there",
            "My plan needs review",
            "Balance this reaction",
        ];
        let batch = runtime.predict_batch(inputs.iter().copied());
        let individual = inputs
            .iter()
            .map(|input| runtime.predict(input))
            .collect::<Vec<_>>();
        assert_eq!(batch, individual);
    }

    #[test]
    fn embedded_bundle_is_digest_verified_and_compilable() {
        let verified = embedded_bundle().unwrap();
        assert_eq!(verified.manifest.model_version, OPEN_SET_MODEL_VERSION);
        let prediction = verified.compile().unwrap().predict("Today I feel calm");
        assert!(prediction.confidence.is_finite());
    }

    #[test]
    fn bootstrap_intervals_are_deterministic_and_contain_point_estimates() {
        let first = bundled_experiment(100);
        let second = bundled_experiment(100);
        assert_eq!(first.metrics.bootstrap_95, second.metrics.bootstrap_95);
        for estimate in [
            &first.metrics.bootstrap_95.id_accuracy,
            &first.metrics.bootstrap_95.id_macro_f1,
            &first.metrics.bootstrap_95.id_negative_log_likelihood,
            &first.metrics.bootstrap_95.id_multiclass_brier,
            &first.metrics.bootstrap_95.id_expected_calibration_error,
            &first.metrics.bootstrap_95.id_aurc,
            &first.metrics.bootstrap_95.ood_auroc,
            &first.metrics.bootstrap_95.ood_aupr_in_domain,
            &first.metrics.bootstrap_95.ood_fpr_at_95_tpr,
        ] {
            assert!(estimate.lower_95 <= estimate.value);
            assert!(estimate.value <= estimate.upper_95);
        }
    }

    #[test]
    fn persisted_metrics_have_a_cross_platform_reporting_precision() {
        let result = bundled_experiment(100);
        let persisted: OpenSetMetricsV3 =
            serde_json::from_slice(&canonical_metrics_json(&result.metrics).unwrap()).unwrap();
        validate_metrics_contract(
            &persisted,
            &result.model,
            &result.policy,
            &result.split_plan,
        )
        .unwrap();

        let mut metrics = result.metrics;
        metrics.id_test.calibration.negative_log_likelihood = 0.25;
        let baseline = canonical_metrics_json(&metrics).unwrap();
        let mut within_tolerance = metrics.clone();
        within_tolerance.id_test.calibration.negative_log_likelihood += 0.4e-9;
        assert_eq!(
            baseline,
            canonical_metrics_json(&within_tolerance).unwrap(),
            "sub-nanometric libm drift must not alter the persisted report"
        );

        let mut material_change = metrics;
        material_change.id_test.calibration.negative_log_likelihood += 1.1e-9;
        assert_ne!(
            baseline,
            canonical_metrics_json(&material_change).unwrap(),
            "a change beyond the declared reporting precision must remain visible"
        );
    }

    #[test]
    fn discrimination_metrics_handle_ties_by_score_group() {
        let metrics = discrimination_metrics(&[0.9, 0.5], &[0.5, 0.1]);
        assert!((metrics.auroc - 0.875).abs() <= 1e-12);
        assert!((metrics.aupr_in_domain - (5.0 / 6.0)).abs() <= 1e-12);
        assert!((metrics.fpr_at_95_tpr - 0.5).abs() <= 1e-12);
    }

    #[test]
    fn risk_coverage_groups_confidence_ties_and_ignores_id_order() {
        let make = |id: &str, confidence: f64, correct: bool| EvaluatedOpenSetPrediction {
            id: id.into(),
            actual_label: "a".into(),
            predicted_label: if correct { "a".into() } else { "b".into() },
            correct,
            accepted: true,
            confidence,
            probability_margin: 0.0,
            probabilities: BTreeMap::new(),
        };
        let left = vec![
            make("one", 0.9, true),
            make("alpha", 0.5, true),
            make("omega", 0.5, false),
            make("four", 0.1, false),
        ];
        let right = vec![
            make("renamed-one", 0.9, true),
            make("zulu", 0.5, true),
            make("able", 0.5, false),
            make("renamed-four", 0.1, false),
        ];
        let (left_curve, left_aurc) = risk_coverage(&left);
        let (right_curve, right_aurc) = risk_coverage(&right);
        assert_eq!(left_curve, right_curve);
        assert_eq!(
            left_curve
                .iter()
                .map(|point| point.accepted)
                .collect::<Vec<_>>(),
            vec![1, 3, 4]
        );
        assert!((left_aurc - 7.0 / 24.0).abs() <= 1e-12);
        assert_eq!(left_aurc, right_aurc);
    }

    #[test]
    fn model_validation_rejects_overflow_scale_parameters() {
        let result = bundled_experiment(100);
        let mut huge_weight = result.model.clone();
        huge_weight.weights[0][0] = f64::MAX;
        assert!(huge_weight.validate().is_err());

        let mut huge_bias = result.model.clone();
        huge_bias.biases[0] = f64::MAX;
        assert!(huge_bias.validate().is_err());

        let mut huge_idf = result.model;
        huge_idf.vectorizer.inverse_document_frequency[0] = f64::MAX;
        assert!(huge_idf.validate().is_err());
    }

    #[test]
    fn compiled_prediction_bounds_direct_library_inputs() {
        let result = bundled_experiment(100);
        let runtime = CompiledModel::new(result.model, result.policy).unwrap();
        let prediction = runtime.predict(&"x".repeat(crate::MAX_INPUT_CHARS + 1));
        assert!(!prediction.accepted);
        assert!(prediction.explanation.top_contributions.is_empty());
    }

    #[test]
    fn metrics_reject_ids_shared_by_id_and_ood_tests() {
        let result = bundled_experiment(100);
        let mut metrics = result.metrics;
        metrics.ood_test.predictions[0].id = metrics.id_test.predictions[0].id.clone();
        let error =
            validate_metrics_contract(&metrics, &result.model, &result.policy, &result.split_plan)
                .unwrap_err();
        assert!(error.to_string().contains("OOD-test prediction ledger"));
    }

    #[test]
    fn bundle_verification_detects_tampering_and_reproduction_matches() {
        let directory = TestDirectory::new("bundle");
        let result = bundled_experiment(100);
        write_bundle(&directory.0, &result).unwrap();
        write_bundle(&directory.0, &result).unwrap();
        let verified = verify_bundle(&directory.0).unwrap();
        assert_eq!(verified.model, result.model);
        reproduce_bundle(
            &directory.0,
            &GroupedDataset::bundled().unwrap(),
            &OpenSetOodDataset::bundled_development().unwrap(),
            &OpenSetOodDataset::bundled_test().unwrap(),
            &OpenSetContrastDataset::bundled_test().unwrap(),
        )
        .unwrap();

        let model_path = directory.0.join("model.json");
        let mut bytes = fs::read(&model_path).unwrap();
        let middle = bytes.len() / 2;
        bytes[middle] ^= 1;
        fs::write(model_path, bytes).unwrap();
        assert!(verify_bundle(&directory.0).is_err());
    }

    #[test]
    fn checked_in_v3_bundle_passes_full_semantic_verification() {
        let bundle = Path::new(env!("CARGO_MANIFEST_DIR")).join("artifacts/eliza-open-set-v3");
        let verified = verify_bundle(bundle).unwrap();
        assert_eq!(verified.model.training_config.seed, 4_043_100_207_104_787);
        assert_eq!(
            verified.metrics.bootstrap_95.resamples,
            DEFAULT_BOOTSTRAP_RESAMPLES
        );
        assert_eq!(verified.metrics.contrast_test.example_count, 28);
        assert_eq!(verified.metrics.contrast_test.pair_count, 14);
    }

    #[test]
    fn bundle_writer_never_replaces_an_unrelated_non_empty_directory() {
        let root = TestDirectory::new("bundle-destination-guard");
        let unrelated = root.0.join("unrelated");
        fs::create_dir(&unrelated).unwrap();
        let sentinel = unrelated.join("keep-me.txt");
        fs::write(&sentinel, b"important").unwrap();
        let result = bundled_experiment(100);

        let error = write_bundle(&unrelated, &result).unwrap_err();
        assert!(error.to_string().contains("refusing to replace"));
        assert_eq!(fs::read(&sentinel).unwrap(), b"important");

        let bundle = root.0.join("bundle");
        write_bundle(&bundle, &result).unwrap();
        let unexpected = bundle.join("unexpected.txt");
        fs::write(&unexpected, b"not part of the contract").unwrap();
        assert!(verify_bundle(&bundle).is_err());
        assert!(write_bundle(&bundle, &result).is_err());
        assert!(unexpected.exists());
    }

    #[test]
    fn bundle_verification_rejects_self_rehashed_semantic_mismatches() {
        let root = TestDirectory::new("bundle-semantic-tamper");
        let result = bundled_experiment(100);
        let policy_bundle = root.0.join("policy-bundle");
        write_bundle(&policy_bundle, &result).unwrap();

        let policy_path = policy_bundle.join("policy.json");
        let mut policy: OpenSetPolicyV3 =
            serde_json::from_slice(&fs::read(&policy_path).unwrap()).unwrap();
        policy.minimum_confidence -= 0.01;
        let policy_bytes = canonical_json(&policy).unwrap();
        fs::write(&policy_path, &policy_bytes).unwrap();

        let manifest_path = policy_bundle.join("manifest.json");
        let mut manifest: BundleManifestV3 =
            serde_json::from_slice(&fs::read(&manifest_path).unwrap()).unwrap();
        manifest
            .files
            .insert("policy.json".into(), sha256_hex(&policy_bytes));
        fs::write(&manifest_path, canonical_json(&manifest).unwrap()).unwrap();

        let error = verify_bundle(&policy_bundle).unwrap_err();
        assert!(error.to_string().contains("frozen policy"));

        let ledger_bundle = root.0.join("ledger-bundle");
        write_bundle(&ledger_bundle, &result).unwrap();
        let metrics_path = ledger_bundle.join("metrics.json");
        let mut metrics: OpenSetMetricsV3 =
            serde_json::from_slice(&fs::read(&metrics_path).unwrap()).unwrap();
        let replacement_label = result
            .model
            .labels
            .iter()
            .find(|label| **label != metrics.id_test.predictions[0].actual_label)
            .unwrap()
            .clone();
        metrics.id_test.predictions[0].actual_label = replacement_label;
        metrics.id_test.predictions[0].correct = metrics.id_test.predictions[0].actual_label
            == metrics.id_test.predictions[0].predicted_label;
        let metrics_bytes = canonical_metrics_json(&metrics).unwrap();
        fs::write(&metrics_path, &metrics_bytes).unwrap();
        let ledger_manifest_path = ledger_bundle.join("manifest.json");
        let mut ledger_manifest: BundleManifestV3 =
            serde_json::from_slice(&fs::read(&ledger_manifest_path).unwrap()).unwrap();
        ledger_manifest
            .files
            .insert("metrics.json".into(), sha256_hex(&metrics_bytes));
        fs::write(
            &ledger_manifest_path,
            canonical_json(&ledger_manifest).unwrap(),
        )
        .unwrap();
        let error = verify_bundle(&ledger_bundle).unwrap_err();
        assert!(error.to_string().contains("split plan"));

        let acceptance_bundle = root.0.join("acceptance-bundle");
        write_bundle(&acceptance_bundle, &result).unwrap();
        let metrics_path = acceptance_bundle.join("metrics.json");
        let mut metrics: OpenSetMetricsV3 =
            serde_json::from_slice(&fs::read(&metrics_path).unwrap()).unwrap();
        let accepted = metrics
            .id_test
            .predictions
            .iter_mut()
            .find(|prediction| prediction.accepted)
            .expect("the fixture must cover at least one ID-test row");
        accepted.accepted = false;
        let accepted_count = metrics
            .id_test
            .predictions
            .iter()
            .filter(|prediction| prediction.accepted)
            .count();
        metrics.id_test.coverage = accepted_count as f64 / metrics.id_test.example_count as f64;
        metrics.id_test.selective_accuracy = Some(
            metrics
                .id_test
                .predictions
                .iter()
                .filter(|prediction| prediction.accepted && prediction.correct)
                .count() as f64
                / accepted_count as f64,
        );
        let metrics_bytes = canonical_metrics_json(&metrics).unwrap();
        fs::write(&metrics_path, &metrics_bytes).unwrap();
        let manifest_path = acceptance_bundle.join("manifest.json");
        let mut manifest: BundleManifestV3 =
            serde_json::from_slice(&fs::read(&manifest_path).unwrap()).unwrap();
        manifest
            .files
            .insert("metrics.json".into(), sha256_hex(&metrics_bytes));
        fs::write(&manifest_path, canonical_json(&manifest).unwrap()).unwrap();
        let error = verify_bundle(&acceptance_bundle).unwrap_err();
        assert!(error.to_string().contains("internally inconsistent"));
    }

    #[test]
    fn jsonl_batch_is_bounded_and_preserves_ids() {
        let result = bundled_experiment(100);
        let runtime = CompiledModel::new(result.model, result.policy).unwrap();
        let input = b"{\"id\":\"row-1\",\"text\":\"Today I feel calm\"}\n{\"id\":\"row-2\",\"text\":\"Hello there\"}\n";
        let mut output = Vec::new();
        let count = predict_jsonl(&runtime, &mut &input[..], &mut output).unwrap();
        assert_eq!(count, 2);
        let lines = String::from_utf8(output).unwrap();
        assert!(lines.contains("\"id\":\"row-1\""));
        assert!(lines.contains("\"id\":\"row-2\""));

        let oversized = format!(
            "{{\"id\":\"row-3\",\"text\":\"{}\"}}\n",
            "x".repeat(MAX_JSONL_BYTES)
        );
        let error =
            predict_jsonl(&runtime, &mut oversized.as_bytes(), &mut Vec::new()).unwrap_err();
        assert!(error.to_string().contains("exceeds the input boundary"));
    }

    #[test]
    fn jsonl_batch_stops_at_byte_limits_without_draining() {
        let runtime = embedded_bundle().unwrap().compile().unwrap();
        let limits = BatchLimits {
            maximum_line_bytes: 32,
            maximum_total_bytes: 100,
            ..DEFAULT_BATCH_LIMITS
        };
        let mut reader = std::io::Cursor::new(vec![b'x'; 10_000]);
        let mut output = Vec::new();
        let error =
            predict_jsonl_with_limits(&runtime, &mut reader, &mut output, limits).unwrap_err();
        assert!(error
            .to_string()
            .contains("line 1 exceeds the input boundary"));
        assert_eq!(reader.position(), 33);
        assert!(output.is_empty());

        // Many short blank lines must also exhaust a total-byte budget; only one
        // extra byte is consumed to establish that the stream is too large.
        let mut reader = std::io::Cursor::new(b" \n".repeat(100));
        let limits = BatchLimits {
            maximum_total_bytes: 10,
            ..limits
        };
        let error =
            predict_jsonl_with_limits(&runtime, &mut reader, &mut output, limits).unwrap_err();
        assert!(error.to_string().contains("10 byte total boundary"));
        assert_eq!(reader.position(), 11);
        assert!(output.is_empty());
    }

    #[test]
    fn jsonl_batch_counts_blank_lines_and_accepts_exact_limits() {
        let runtime = embedded_bundle().unwrap().compile().unwrap();
        let mut reader = std::io::Cursor::new(b"\n".repeat(MAX_EXAMPLES + 20));
        let error = predict_jsonl(&runtime, &mut reader, &mut Vec::new()).unwrap_err();
        assert!(error.to_string().contains("100000 physical lines"));
        assert_eq!(reader.position(), (MAX_EXAMPLES + 1) as u64);

        let input = b" \r\n{\"id\":\"row-1\",\"text\":\"Today I feel calm\"}";
        let limits = BatchLimits {
            maximum_rows: 1,
            maximum_physical_lines: 2,
            maximum_line_bytes: input.len() - 3,
            maximum_total_bytes: input.len(),
        };
        let mut output = Vec::new();
        let count =
            predict_jsonl_with_limits(&runtime, &mut &input[..], &mut output, limits).unwrap();
        assert_eq!(count, 1);
        let value: serde_json::Value = serde_json::from_slice(&output).unwrap();
        assert_eq!(value["id"], "row-1");
    }

    #[test]
    fn jsonl_batch_reports_physical_lines_without_echoing_invalid_input() {
        let runtime = embedded_bundle().unwrap().compile().unwrap();
        for invalid in [
            r#"{"id":"row-1","text":"private prompt","private field":1}"#,
            r#"{"id":"row-1","text":"private prompt","text":"duplicate"}"#,
            r#"{"id":"row-1","text":{"private prompt":true}}"#,
            r#"{"id":"row-1","text":"private prompt""#,
        ] {
            let input = format!("\n\r\n{invalid}\n");
            let mut output = Vec::new();
            let error = predict_jsonl(&runtime, &mut input.as_bytes(), &mut output).unwrap_err();
            let diagnostic = error.to_string();
            assert!(diagnostic.contains("line 3 does not match the required object schema"));
            assert!(!diagnostic.contains("private"));
            assert!(!diagnostic.contains("row-1"));
            assert!(output.is_empty());
        }
    }

    #[test]
    fn jsonl_batch_rejects_duplicate_ids_and_leaves_only_a_valid_prefix() {
        let runtime = embedded_bundle().unwrap().compile().unwrap();
        let row = "{\"id\":\"row-1\",\"text\":\"Today I feel calm\"}\n";
        let input = format!("{row}\n{row}");
        let mut output = Vec::new();
        let error = predict_jsonl(&runtime, &mut input.as_bytes(), &mut output).unwrap_err();
        assert!(error.to_string().contains("line 3 repeats a batch id"));
        assert!(!error.to_string().contains("row-1"));
        assert_eq!(output.iter().filter(|byte| **byte == b'\n').count(), 1);

        let mut output = Vec::new();
        let limits = BatchLimits {
            maximum_rows: 1,
            ..DEFAULT_BATCH_LIMITS
        };
        let error = predict_jsonl_with_limits(&runtime, &mut input.as_bytes(), &mut output, limits)
            .unwrap_err();
        assert!(error.to_string().contains("exceeds 1 rows"));
        assert_eq!(output.iter().filter(|byte| **byte == b'\n').count(), 1);
    }
}
