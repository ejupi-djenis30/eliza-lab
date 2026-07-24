use serde_json::Value;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};

static TEMP_SEQUENCE: AtomicUsize = AtomicUsize::new(0);

struct TestDirectory(PathBuf);

impl TestDirectory {
    fn new() -> Self {
        let sequence = TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "eliza-lab-cli-test-{}-{sequence}",
            std::process::id()
        ));
        fs::create_dir_all(&path).unwrap();
        Self(path)
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TestDirectory {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn binary() -> &'static str {
    env!("CARGO_BIN_EXE_eliza-lab")
}

fn project_path(path: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(path)
}

fn run_train(directory: &TestDirectory, model: &str, report: &str) -> Output {
    Command::new(binary())
        .current_dir(directory.path())
        .args([
            "train",
            "--dataset",
            project_path("fixtures/intents-v1.tsv").to_str().unwrap(),
            "--ood",
            project_path("fixtures/ood-v1.tsv").to_str().unwrap(),
            "--output",
            model,
            "--report",
            report,
        ])
        .output()
        .unwrap()
}

#[test]
fn legacy_once_keeps_its_stable_trace_contract() {
    let output = Command::new(binary())
        .args(["--once", "I feel calm today"])
        .output()
        .unwrap();

    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("rule=feeling-reflection turn=1"));
    assert!(!stdout.contains("boundary="));
}

#[test]
fn current_directory_outputs_are_byte_reproducible() {
    let directory = TestDirectory::new();
    let first = run_train(&directory, "model-a.json", "report-a.json");
    assert!(
        first.status.success(),
        "{}",
        String::from_utf8_lossy(&first.stderr)
    );
    let second = run_train(&directory, "model-b.json", "report-b.json");
    assert!(
        second.status.success(),
        "{}",
        String::from_utf8_lossy(&second.stderr)
    );

    assert_eq!(
        fs::read(directory.path().join("model-a.json")).unwrap(),
        fs::read(directory.path().join("model-b.json")).unwrap()
    );
    assert_eq!(
        fs::read(directory.path().join("report-a.json")).unwrap(),
        fs::read(directory.path().join("report-b.json")).unwrap()
    );
}

#[test]
fn training_refuses_to_overwrite_an_input_or_alias_its_outputs() {
    let dataset = project_path("fixtures/intents-v1.tsv");
    let original = fs::read(&dataset).unwrap();
    let output = Command::new(binary())
        .args([
            "train",
            "--dataset",
            dataset.to_str().unwrap(),
            "--ood",
            project_path("fixtures/ood-v1.tsv").to_str().unwrap(),
            "--output",
            dataset.to_str().unwrap(),
            "--report",
            dataset.to_str().unwrap(),
        ])
        .output()
        .unwrap();

    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("collides"));
    assert_eq!(fs::read(dataset).unwrap(), original);
}

#[test]
fn training_rejects_threshold_options_instead_of_silently_overriding_them() {
    let directory = TestDirectory::new();
    let dataset = project_path("fixtures/intents-v1.tsv");
    let ood = project_path("fixtures/ood-v1.tsv");

    for (index, option) in ["--minimum-confidence", "--minimum-margin"]
        .into_iter()
        .enumerate()
    {
        let model = format!("model-{index}.json");
        let report = format!("report-{index}.json");
        let output = Command::new(binary())
            .current_dir(directory.path())
            .args([
                "train",
                "--dataset",
                dataset.to_str().unwrap(),
                "--ood",
                ood.to_str().unwrap(),
                "--output",
                &model,
                "--report",
                &report,
                option,
                "0.50",
            ])
            .output()
            .unwrap();

        assert!(!output.status.success());
        assert!(String::from_utf8_lossy(&output.stderr)
            .contains(&format!("unknown train option `{option}`")));
        assert!(!directory.path().join(model).exists());
        assert!(!directory.path().join(report).exists());
    }
}

#[test]
fn primary_inference_json_uses_v3_and_keeps_evidence_and_hard_safety_boundary() {
    let inferred = Command::new(binary())
        .args(["infer", "--json", "Hello, I want to make a concrete plan"])
        .output()
        .unwrap();
    assert!(
        inferred.status.success(),
        "{}",
        String::from_utf8_lossy(&inferred.stderr)
    );
    let inference: Value = serde_json::from_slice(&inferred.stdout).unwrap();
    assert!(matches!(
        inference["boundary"].as_str(),
        Some("ml-intent" | "ml-abstain")
    ));
    assert_eq!(inference["model"]["version"], "3.0.0");
    assert!(inference["model"]["accepted"].is_boolean());
    assert!(inference["model"]["confidence"].is_number());
    assert!(inference["model"]["margin"].is_number());
    assert_eq!(
        inference["model"]["probabilities"]
            .as_object()
            .unwrap()
            .len(),
        7
    );
    let contribution = &inference["model"]["top_features"][0];
    assert!(contribution["feature"].is_string());
    assert!(contribution["value"].is_number());
    assert!(contribution["weight"].is_number());
    assert!(contribution["contribution"].is_number());

    let safety = Command::new(binary())
        .args(["infer", "--json", "I want to die"])
        .output()
        .unwrap();
    assert!(safety.status.success());
    let safety: Value = serde_json::from_slice(&safety.stdout).unwrap();
    assert_eq!(safety["boundary"], "safety-boundary");
    assert!(safety["model"].is_null());
}

#[test]
fn legacy_inference_requires_an_explicit_mode_and_rejects_conflicting_sources() {
    let model = project_path("models/eliza-intent-v1.json");
    let bundle = project_path("artifacts/eliza-open-set-v3");

    let implicit_legacy = Command::new(binary())
        .args([
            "infer",
            "--model",
            model.to_str().unwrap(),
            "--json",
            "Today I feel calm",
        ])
        .output()
        .unwrap();
    assert!(!implicit_legacy.status.success());
    assert!(String::from_utf8_lossy(&implicit_legacy.stderr)
        .contains("--model requires the explicit --legacy-v1 mode"));

    let explicit_legacy = Command::new(binary())
        .args([
            "infer",
            "--legacy-v1",
            "--model",
            model.to_str().unwrap(),
            "--json",
            "Today I feel calm",
        ])
        .output()
        .unwrap();
    assert!(explicit_legacy.status.success());
    let inference: Value = serde_json::from_slice(&explicit_legacy.stdout).unwrap();
    assert_eq!(inference["model"]["version"], "1.0.0");

    let conflicting = Command::new(binary())
        .args([
            "infer",
            "--legacy-v1",
            "--bundle",
            bundle.to_str().unwrap(),
            "prompt",
        ])
        .output()
        .unwrap();
    assert!(!conflicting.status.success());
    assert!(String::from_utf8_lossy(&conflicting.stderr)
        .contains("--bundle cannot be combined with --legacy-v1"));
}

#[test]
fn bundle_help_succeeds_and_verify_rejects_reproduction_only_inputs() {
    for arguments in [vec!["bundle", "--help"], vec!["bundle", "help"]] {
        let output = Command::new(binary()).args(arguments).output().unwrap();
        assert!(output.status.success());
        assert!(String::from_utf8_lossy(&output.stdout).contains("bundle verify"));
    }

    let output = Command::new(binary())
        .args([
            "bundle",
            "verify",
            "--dataset",
            project_path("fixtures/intents-v3.tsv").to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("unknown bundle option `--dataset`"));
}

#[test]
fn selection_audit_cannot_accept_final_test_or_ood_inputs() {
    for option in [
        "--calibration",
        "--id-test",
        "--ood-development",
        "--ood-test",
        "--contrast-test",
    ] {
        let output = Command::new(binary())
            .args(["selection", "audit", option, "forbidden.tsv"])
            .output()
            .unwrap();
        assert!(!output.status.success());
        assert!(
            String::from_utf8_lossy(&output.stderr)
                .contains(&format!("unknown selection audit option `{option}`")),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
}

#[test]
fn selection_audit_refuses_to_overwrite_its_dataset() {
    let dataset = project_path("fixtures/intents-v3.tsv");
    let before = fs::read(&dataset).unwrap();
    let output = Command::new(binary())
        .args([
            "selection",
            "audit",
            "--dataset",
            dataset.to_str().unwrap(),
            "--output",
            dataset.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("collides"));
    assert_eq!(fs::read(&dataset).unwrap(), before);
}

#[test]
fn open_set_bundle_reproduces_and_serves_bounded_jsonl_predictions() {
    let directory = TestDirectory::new();
    let bundle = directory.path().join("bundle-v3");
    let trained = Command::new(binary())
        .current_dir(directory.path())
        .args([
            "train-v3",
            "--output",
            bundle.to_str().unwrap(),
            "--bootstrap-resamples",
            "100",
        ])
        .output()
        .unwrap();
    assert!(
        trained.status.success(),
        "{}",
        String::from_utf8_lossy(&trained.stderr)
    );
    assert!(String::from_utf8_lossy(&trained.stdout).contains("70 ID-test"));

    for operation in ["verify", "reproduce"] {
        let output = Command::new(binary())
            .current_dir(directory.path())
            .args(["bundle", operation, "--bundle", bundle.to_str().unwrap()])
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{operation}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    let mut child = Command::new(binary())
        .current_dir(directory.path())
        .args(["infer-batch", "--bundle", bundle.to_str().unwrap()])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(b"{\"id\":\"case-1\",\"text\":\"Today I feel calm\"}\n")
        .unwrap();
    let output = child.wait_with_output().unwrap();
    assert!(output.status.success());
    let prediction: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(prediction["id"], "case-1");
    assert!(prediction["prediction"]["confidence"].is_number());
    assert_eq!(
        prediction["prediction"]["explanation"]["top_label"],
        prediction["prediction"]["label"]
    );
}

#[test]
fn robustness_audit_is_aggregate_local_and_deterministic() {
    let input = concat!(
        "{\"id\":\"audit-1\",\"text\":\"Hello, I intend to make a concrete plan\"}\n",
        "{\"id\":\"audit-2\",\"text\":\"Today I feel uncertain about the outcome\"}\n"
    );
    let run = || {
        let mut child = Command::new(binary())
            .args(["robustness", "audit"])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        child
            .stdin
            .take()
            .unwrap()
            .write_all(input.as_bytes())
            .unwrap();
        child.wait_with_output().unwrap()
    };

    let first = run();
    let second = run();
    assert!(
        first.status.success(),
        "{}",
        String::from_utf8_lossy(&first.stderr)
    );
    assert_eq!(first.stdout, second.stdout);
    let report: Value = serde_json::from_slice(&first.stdout).unwrap();
    assert_eq!(report["schema_version"], 1);
    assert_eq!(report["report_kind"], "eliza-metamorphic-robustness");
    assert_eq!(report["suite_version"], "1.0.0");
    assert_eq!(report["population"], "jsonl");
    assert_eq!(report["model_version"], "3.0.0");
    assert_eq!(report["input_count"], 2);
    assert_eq!(report["formatting"]["decision_agreement"], 1.0);
    assert_eq!(
        report["formatting"]["maximum_normalized_js_divergence"],
        0.0
    );
    assert_eq!(report["perturbations"].as_array().unwrap().len(), 7);
    let serialized = String::from_utf8(first.stdout).unwrap();
    assert!(!serialized.contains("audit-1"));
    assert!(!serialized.contains("concrete plan"));
}

#[test]
fn robustness_audit_rejects_invalid_policy_before_reading_input() {
    let output = Command::new(binary())
        .args([
            "robustness",
            "audit",
            "--minimum-typographic-decision-agreement",
            "1.01",
        ])
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("invalid robustness policy"));
}

#[test]
fn robustness_audit_sanitizes_json_schema_failures() {
    let input = concat!(
        "{\"id\":\"private-id\",\"text\":\"invented private prompt\",",
        "\"private_token\":\"secret-value\"}\n"
    );
    let mut child = Command::new(binary())
        .args(["robustness", "audit"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(input.as_bytes())
        .unwrap();
    let output = child.wait_with_output().unwrap();
    assert!(!output.status.success());
    assert!(output.stdout.is_empty());
    let error = String::from_utf8_lossy(&output.stderr);
    assert!(error.contains("line 1 does not match the required object schema"));
    for private_fragment in [
        "private-id",
        "invented private prompt",
        "private_token",
        "secret-value",
        "unknown field",
    ] {
        assert!(
            !error.contains(private_fragment),
            "schema error exposed `{private_fragment}`: {error}"
        );
    }
}

#[test]
fn robustness_audit_can_regress_the_verified_bundle_id_test_without_stdin() {
    let output = Command::new(binary())
        .args(["robustness", "audit", "--bundle-id-test"])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let report: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(report["input_count"], 70);
    assert_eq!(report["population"], "bundle-id-test");
    assert_eq!(report["baseline_accepted"], 44);
    assert_eq!(report["evaluated_variants"], 490);
    assert_eq!(report["formatting"]["decision_agreement"], 1.0);
    assert_eq!(
        report["formatting"]["maximum_normalized_js_divergence"],
        0.0
    );
}
