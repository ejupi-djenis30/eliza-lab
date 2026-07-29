//! Deterministic, prompt-free diagnostics for a built ELIZA Lab binary.
//!
//! The diagnostic path verifies the exact embedded bundle, compiles it once, and exercises
//! inference invariants without accepting user data. It is deliberately separate from model
//! evaluation: the fixed probes do not tune, score, or modify the model.

use crate::ml::MlError;
use crate::open_set::{embedded_bundle, CompiledModel};
use crate::{ElizaEngine, MAX_INPUT_CHARS};
use serde::Serialize;
use std::collections::BTreeMap;

const REPORT_KIND: &str = "eliza-embedded-self-test";
const SCHEMA_VERSION: u32 = 1;
const PROBABILITY_TOLERANCE: f64 = 1e-12;
const EXPLANATION_TOLERANCE: f64 = 1e-9;
const FICTIONAL_INFERENCE_PROBE: &str = "Today I feel calm about the outcome";
const SAFETY_BOUNDARY_PROBE: &str = "I want to die";

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct DiagnosticCheck {
    pub name: String,
    pub status: String,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct DiagnosticBundle {
    pub bundle_version: String,
    pub model_version: String,
    pub dataset_sha256: String,
    pub split_plan_sha256: String,
    pub payload_sha256: BTreeMap<String, String>,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct DiagnosticReport {
    pub schema_version: u32,
    pub report_kind: String,
    pub status: String,
    pub application_version: String,
    pub bundle: DiagnosticBundle,
    pub checks: Vec<DiagnosticCheck>,
}

fn passed(name: &str) -> DiagnosticCheck {
    DiagnosticCheck {
        name: name.into(),
        status: "pass".into(),
    }
}

fn diagnostic_failure(message: impl Into<String>) -> MlError {
    MlError::InvalidModel(format!("embedded self-test failed: {}", message.into()))
}

fn verify_probability_simplex(runtime: &CompiledModel) -> Result<(), MlError> {
    let prediction = runtime.predict(FICTIONAL_INFERENCE_PROBE);
    if prediction.probabilities.len() < 2
        || prediction
            .probabilities
            .values()
            .any(|value| !value.is_finite() || !(0.0..=1.0).contains(value))
    {
        return Err(diagnostic_failure(
            "inference returned an invalid probability vector",
        ));
    }
    let total = prediction.probabilities.values().sum::<f64>();
    if (total - 1.0).abs() > PROBABILITY_TOLERANCE {
        return Err(diagnostic_failure(
            "inference probabilities do not sum to one",
        ));
    }
    Ok(())
}

fn verify_contrastive_explanation(runtime: &CompiledModel) -> Result<(), MlError> {
    let prediction = runtime.predict(FICTIONAL_INFERENCE_PROBE);
    if prediction.label != prediction.explanation.top_label
        || prediction.runner_up_label != prediction.explanation.runner_up_label
        || (prediction.logit_margin - prediction.explanation.reconstructed_logit_margin).abs()
            > EXPLANATION_TOLERANCE
    {
        return Err(diagnostic_failure(
            "contrastive evidence does not reconstruct the top-two decision",
        ));
    }
    Ok(())
}

fn verify_abstention_path(runtime: &CompiledModel) -> Result<(), MlError> {
    let prediction = runtime.predict("");
    if prediction.accepted || !prediction.explanation.top_contributions.is_empty() {
        return Err(diagnostic_failure(
            "featureless input did not take the abstention path",
        ));
    }
    Ok(())
}

fn verify_input_and_safety_boundaries(runtime: &CompiledModel) -> Result<(), MlError> {
    let mut engine = ElizaEngine::new();
    let oversized = "x".repeat(MAX_INPUT_CHARS + 1);
    let oversized_reply = engine.respond_with_open_set(&oversized, runtime);
    if oversized_reply.rule_id != "input-boundary" || oversized_reply.model_trace.is_some() {
        return Err(diagnostic_failure(
            "the input boundary did not stop learned inference",
        ));
    }

    let safety_reply = engine.respond_with_open_set(SAFETY_BOUNDARY_PROBE, runtime);
    if safety_reply.rule_id != "safety-boundary" || safety_reply.model_trace.is_some() {
        return Err(diagnostic_failure(
            "the safety boundary did not stop learned inference",
        ));
    }
    Ok(())
}

/// Verifies the embedded artifact and inference path without reading prompts, files, or stdin.
pub fn run_embedded_self_test() -> Result<DiagnosticReport, MlError> {
    let verified = embedded_bundle()?;
    let manifest = verified.manifest().clone();
    let runtime = verified.compile()?;

    verify_probability_simplex(&runtime)?;
    verify_contrastive_explanation(&runtime)?;
    verify_abstention_path(&runtime)?;
    verify_input_and_safety_boundaries(&runtime)?;

    Ok(DiagnosticReport {
        schema_version: SCHEMA_VERSION,
        report_kind: REPORT_KIND.into(),
        status: "pass".into(),
        application_version: env!("CARGO_PKG_VERSION").into(),
        bundle: DiagnosticBundle {
            bundle_version: manifest.bundle_version,
            model_version: manifest.model_version,
            dataset_sha256: manifest.dataset_sha256,
            split_plan_sha256: manifest.split_plan_sha256,
            payload_sha256: manifest.files,
        },
        checks: vec![
            passed("embedded-sha256-inventory"),
            passed("semantic-artifact-contract"),
            passed("compiled-inference-runtime"),
            passed("probability-simplex"),
            passed("contrastive-explanation"),
            passed("featureless-input-abstention"),
            passed("input-and-safety-boundaries-before-ml"),
        ],
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedded_self_test_is_deterministic_and_contains_no_probe_text() {
        let first = run_embedded_self_test().unwrap();
        let second = run_embedded_self_test().unwrap();
        assert_eq!(first, second);
        assert_eq!(first.schema_version, 1);
        assert_eq!(first.report_kind, "eliza-embedded-self-test");
        assert_eq!(first.status, "pass");
        assert_eq!(first.bundle.model_version, "3.0.0");
        assert_eq!(first.bundle.payload_sha256.len(), 4);
        assert!(first.checks.iter().all(|check| check.status == "pass"));

        let serialized = serde_json::to_string(&first).unwrap();
        assert!(!serialized.contains(FICTIONAL_INFERENCE_PROBE));
        assert!(!serialized.contains(SAFETY_BOUNDARY_PROBE));
        assert!(!serialized.contains(&"x".repeat(MAX_INPUT_CHARS)));
    }
}
