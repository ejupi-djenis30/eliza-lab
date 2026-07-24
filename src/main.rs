use eliza_lab::ml::{
    write_training_artifacts, Dataset, EvaluationMetrics, IntentModel, MlError, OodDataset,
    OodMetrics, TrainingConfig,
};
use eliza_lab::open_set::{
    embedded_bundle, predict_jsonl, reproduce_bundle, run_open_set_experiment,
    run_selection_stability_audit, verify_bundle, write_bundle, write_selection_audit_report,
    CompiledModel, GroupedDataset, OpenSetContrastDataset, OpenSetOodDataset,
    OpenSetTrainingConfig, SelectionAuditConfig, SelectionAuditPool, DEFAULT_BOOTSTRAP_RESAMPLES,
};
use eliza_lab::robustness::{audit_bundle_id_test, audit_jsonl, RobustnessGate};
use eliza_lab::{ElizaEngine, Reply, MAX_INPUT_CHARS};
use serde::Serialize;
use std::env;
use std::error::Error;
use std::fmt;
use std::io::{self, BufRead, Write};
use std::path::{Component, Path, PathBuf};

const DEFAULT_DATASET: &str = "fixtures/intents-v1.tsv";
const DEFAULT_OOD_DATASET: &str = "fixtures/ood-v1.tsv";
const DEFAULT_MODEL: &str = "models/eliza-intent-v1.json";
const DEFAULT_REPORT: &str = "reports/eliza-intent-v1.json";
const DEFAULT_V3_DATASET: &str = "fixtures/intents-v3.tsv";
const DEFAULT_V3_OOD_DEVELOPMENT: &str = "fixtures/ood-dev-v3.tsv";
const DEFAULT_V3_OOD_TEST: &str = "fixtures/ood-test-v3.tsv";
const DEFAULT_V3_CONTRAST_TEST: &str = "fixtures/contrast-test-v3.tsv";
const DEFAULT_V3_BUNDLE: &str = "artifacts/eliza-open-set-v3";
const DEFAULT_SELECTION_AUDIT_REPORT: &str = "reports/selection-stability-v1.json";
const MAX_INPUT_BYTES: usize = MAX_INPUT_CHARS * 4;
const APP_VERSION: &str = env!("CARGO_PKG_VERSION");

enum InputLine {
    Text(String),
    TooLong,
}

enum DialogueModel {
    OpenSet(Box<CompiledModel>),
    Legacy(Box<IntentModel>),
}

#[derive(Debug)]
struct CliError(String);

impl fmt::Display for CliError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl Error for CliError {}

#[derive(Serialize)]
struct InferenceOutput<'a> {
    response: &'a str,
    boundary: &'a str,
    turn: usize,
    model: Option<InferenceTrace<'a>>,
}

#[derive(Serialize)]
struct InferenceTrace<'a> {
    version: &'a str,
    label: &'a str,
    accepted: bool,
    confidence: f64,
    margin: f64,
    probabilities: &'a std::collections::BTreeMap<String, f64>,
    top_features: &'a [eliza_lab::ml::FeatureContribution],
}

#[derive(Serialize)]
struct EvaluationOutput {
    dataset_fingerprint: String,
    holdout: EvaluationMetrics,
    out_of_domain: OodMetrics,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error}");
        std::process::exit(2);
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    let arguments = env::args().skip(1).collect::<Vec<_>>();
    match arguments.first().map(String::as_str) {
        Some("train") => train_command(&arguments[1..]),
        Some("train-v3") => train_v3_command(&arguments[1..]),
        Some("selection") => selection_command(&arguments[1..]),
        Some("evaluate") => evaluate_command(&arguments[1..]),
        Some("infer") => infer_command(&arguments[1..]),
        Some("infer-batch") => infer_batch_command(&arguments[1..]),
        Some("bundle") => bundle_command(&arguments[1..]),
        Some("robustness") => robustness_command(&arguments[1..]),
        Some("chat") => chat_command(&arguments[1..]),
        Some("dataset") => dataset_command(&arguments[1..]),
        Some("--once") => legacy_once(&arguments[1..]),
        Some("--help" | "-h" | "help") => {
            print_help();
            Ok(())
        }
        Some(command) => Err(CliError(format!(
            "unknown command `{command}`; run `eliza-lab --help`"
        ))
        .into()),
        None => interactive(None),
    }
}

fn selection_command(arguments: &[String]) -> Result<(), Box<dyn Error>> {
    let operation = arguments
        .first()
        .map(String::as_str)
        .ok_or_else(|| CliError("selection requires `audit`".into()))?;
    if matches!(operation, "--help" | "-h" | "help") {
        print_selection_help();
        return Ok(());
    }
    if operation != "audit" {
        return Err(CliError(format!(
            "unknown selection operation `{operation}`; expected `audit`"
        ))
        .into());
    }
    let mut dataset_path = None;
    let mut output_path = PathBuf::from(DEFAULT_SELECTION_AUDIT_REPORT);
    let mut config = SelectionAuditConfig::default();
    let mut index = 1;
    while index < arguments.len() {
        match arguments[index].as_str() {
            "--dataset" => dataset_path = Some(PathBuf::from(option_value(arguments, &mut index)?)),
            "--output" => output_path = PathBuf::from(option_value(arguments, &mut index)?),
            "--bootstrap-resamples" => {
                config.bootstrap_resamples =
                    parse_option(arguments, &mut index, "bootstrap resamples")?
            }
            "--help" | "-h" => {
                print_selection_help();
                return Ok(());
            }
            option => {
                return Err(CliError(format!("unknown selection audit option `{option}`")).into())
            }
        }
        index += 1;
    }
    if let Some(dataset_path) = dataset_path.as_deref() {
        ensure_distinct_paths(&[
            ("selection dataset", dataset_path),
            ("selection report", &output_path),
        ])?;
    }
    let dataset = match dataset_path.as_deref() {
        Some(path) => GroupedDataset::read(path)?,
        None => GroupedDataset::bundled()?,
    };
    let pool = SelectionAuditPool::from_dataset(&dataset, config.training.seed)?;
    let report = run_selection_stability_audit(&pool, config)?;
    let digest = write_selection_audit_report(&output_path, &report)?;
    println!("report       {}", output_path.display());
    println!("sha256       {digest}");
    println!(
        "pool         {} examples / {} families / {} labels",
        report.pool.example_count, report.pool.family_count, report.pool.label_count
    );
    println!(
        "nested CV    {} outer folds / {} inner folds / {} model fits",
        report.outer_folds, report.inner_folds, report.model_fits
    );
    println!(
        "OOF          accuracy={:.4} macro-f1={:.4} nll={:.4} brier={:.4}",
        report.metrics.accuracy,
        report.metrics.macro_f1,
        report.metrics.negative_log_likelihood,
        report.metrics.multiclass_brier
    );
    Ok(())
}

fn train_v3_command(arguments: &[String]) -> Result<(), Box<dyn Error>> {
    let mut dataset_path = None;
    let mut ood_development_path = None;
    let mut ood_test_path = None;
    let mut contrast_test_path = None;
    let mut output_path = PathBuf::from(DEFAULT_V3_BUNDLE);
    let mut bootstrap_resamples = DEFAULT_BOOTSTRAP_RESAMPLES;
    let mut config = OpenSetTrainingConfig::default();
    let mut index = 0;
    while index < arguments.len() {
        match arguments[index].as_str() {
            "--dataset" => dataset_path = Some(PathBuf::from(option_value(arguments, &mut index)?)),
            "--ood-development" => {
                ood_development_path = Some(PathBuf::from(option_value(arguments, &mut index)?))
            }
            "--ood-test" => {
                ood_test_path = Some(PathBuf::from(option_value(arguments, &mut index)?))
            }
            "--contrast-test" => {
                contrast_test_path = Some(PathBuf::from(option_value(arguments, &mut index)?))
            }
            "--output" => output_path = PathBuf::from(option_value(arguments, &mut index)?),
            "--seed" => config.seed = parse_option(arguments, &mut index, "seed")?,
            "--epochs" => config.epochs = parse_option(arguments, &mut index, "epochs")?,
            "--learning-rate" => {
                config.learning_rate = parse_option(arguments, &mut index, "learning rate")?
            }
            "--l2" => {
                config.l2_penalty = parse_option(arguments, &mut index, "L2 penalty")?;
                config.development_selection.l2_penalty_candidates = vec![config.l2_penalty];
            }
            "--max-features" => {
                config.vectorizer.max_features =
                    parse_option(arguments, &mut index, "max features")?;
                config.development_selection.max_features_candidates =
                    vec![config.vectorizer.max_features];
            }
            "--bootstrap-resamples" => {
                bootstrap_resamples = parse_option(arguments, &mut index, "bootstrap resamples")?
            }
            "--help" | "-h" => {
                print_train_v3_help();
                return Ok(());
            }
            option => return Err(CliError(format!("unknown train-v3 option `{option}`")).into()),
        }
        index += 1;
    }

    let dataset = match dataset_path.as_deref() {
        Some(path) => GroupedDataset::read(path)?,
        None => GroupedDataset::bundled()?,
    };
    let ood_development = match ood_development_path.as_deref() {
        Some(path) => OpenSetOodDataset::read(path)?,
        None => OpenSetOodDataset::bundled_development()?,
    };
    let ood_test = match ood_test_path.as_deref() {
        Some(path) => OpenSetOodDataset::read(path)?,
        None => OpenSetOodDataset::bundled_test()?,
    };
    let contrast_test = match contrast_test_path.as_deref() {
        Some(path) => OpenSetContrastDataset::read(path)?,
        None => OpenSetContrastDataset::bundled_test()?,
    };
    let result = run_open_set_experiment(
        &dataset,
        &ood_development,
        &ood_test,
        &contrast_test,
        config,
        bootstrap_resamples,
    )?;
    let manifest = write_bundle(&output_path, &result)?;
    println!("bundle       {}", output_path.display());
    println!("model        {}", result.model.model_version);
    println!("dataset      {}", manifest.dataset_sha256);
    println!("split plan   {}", manifest.split_plan_sha256);
    println!(
        "partitions   {} train / {} development / {} calibration / {} ID-test",
        result.metrics.partition_counts["train"],
        result.metrics.partition_counts["development"],
        result.metrics.partition_counts["calibration"],
        result.metrics.partition_counts["id-test"]
    );
    println!("temperature  {:.6}", result.policy.temperature);
    println!(
        "selection    {}/{} candidates on development only; max-features={} l2={:.6}",
        result.metrics.development_selection.selected_index + 1,
        result.metrics.development_selection.candidates.len(),
        result.model.training_config.vectorizer.max_features,
        result.model.training_config.l2_penalty,
    );
    println!(
        "thresholds   confidence={:.2} probability-margin={:.2} (development + OOD-development only)",
        result.policy.minimum_confidence, result.policy.minimum_probability_margin
    );
    println!(
        "ID-test      n={} accuracy={:.4} macro-f1={:.4} coverage={:.4} aurc={:.4}",
        result.metrics.id_test.example_count,
        result.metrics.id_test.accuracy,
        result.metrics.id_test.macro_f1,
        result.metrics.id_test.coverage,
        result.metrics.id_test.aurc
    );
    println!(
        "OOD-test     n={} accepted={} auroc={:.4} aupr-in={:.4} fpr95={:.4}",
        result.metrics.ood_test.example_count,
        result.metrics.ood_test.accepted_examples,
        result.metrics.ood_test.discrimination.auroc,
        result.metrics.ood_test.discrimination.aupr_in_domain,
        result.metrics.ood_test.discrimination.fpr_at_95_tpr
    );
    Ok(())
}

fn bundle_command(arguments: &[String]) -> Result<(), Box<dyn Error>> {
    let operation = arguments
        .first()
        .map(String::as_str)
        .ok_or_else(|| CliError("bundle requires `verify` or `reproduce`".into()))?;
    if matches!(operation, "--help" | "-h" | "help") {
        print_bundle_help();
        return Ok(());
    }
    if !matches!(operation, "verify" | "reproduce") {
        return Err(CliError(format!(
            "unknown bundle operation `{operation}`; expected `verify` or `reproduce`"
        ))
        .into());
    }
    let mut bundle_path = PathBuf::from(DEFAULT_V3_BUNDLE);
    let mut dataset_path = None;
    let mut ood_development_path = None;
    let mut ood_test_path = None;
    let mut contrast_test_path = None;
    let mut index = 1;
    while index < arguments.len() {
        match arguments[index].as_str() {
            "--bundle" => bundle_path = PathBuf::from(option_value(arguments, &mut index)?),
            "--dataset" if operation == "reproduce" => {
                dataset_path = Some(PathBuf::from(option_value(arguments, &mut index)?))
            }
            "--ood-development" if operation == "reproduce" => {
                ood_development_path = Some(PathBuf::from(option_value(arguments, &mut index)?))
            }
            "--ood-test" if operation == "reproduce" => {
                ood_test_path = Some(PathBuf::from(option_value(arguments, &mut index)?))
            }
            "--contrast-test" if operation == "reproduce" => {
                contrast_test_path = Some(PathBuf::from(option_value(arguments, &mut index)?))
            }
            "--help" | "-h" => {
                print_bundle_help();
                return Ok(());
            }
            option => return Err(CliError(format!("unknown bundle option `{option}`")).into()),
        }
        index += 1;
    }
    match operation {
        "verify" => {
            let verified = verify_bundle(&bundle_path)?;
            println!("semantically verified {}", bundle_path.display());
            println!("model        {}", verified.model().model_version);
            println!("dataset      {}", verified.manifest().dataset_sha256);
            println!(
                "payloads     {} (plus manifest)",
                verified.manifest().files.len()
            );
        }
        "reproduce" => {
            let dataset = match dataset_path.as_deref() {
                Some(path) => GroupedDataset::read(path)?,
                None => GroupedDataset::bundled()?,
            };
            let ood_development = match ood_development_path.as_deref() {
                Some(path) => OpenSetOodDataset::read(path)?,
                None => OpenSetOodDataset::bundled_development()?,
            };
            let ood_test = match ood_test_path.as_deref() {
                Some(path) => OpenSetOodDataset::read(path)?,
                None => OpenSetOodDataset::bundled_test()?,
            };
            let contrast_test = match contrast_test_path.as_deref() {
                Some(path) => OpenSetContrastDataset::read(path)?,
                None => OpenSetContrastDataset::bundled_test()?,
            };
            reproduce_bundle(
                &bundle_path,
                &dataset,
                &ood_development,
                &ood_test,
                &contrast_test,
            )?;
            println!("byte-identical reproduction {}", bundle_path.display());
        }
        _ => unreachable!("the operation was validated before option parsing"),
    }
    Ok(())
}

fn infer_batch_command(arguments: &[String]) -> Result<(), Box<dyn Error>> {
    let mut bundle_path = None;
    let mut index = 0;
    while index < arguments.len() {
        match arguments[index].as_str() {
            "--bundle" => bundle_path = Some(PathBuf::from(option_value(arguments, &mut index)?)),
            "--help" | "-h" => {
                println!(
                    "Usage: eliza-lab infer-batch [--bundle PATH]\n\
                     Reads bounded JSONL objects with `id` and `text` from stdin and writes one prediction per line.\n\
                     Without --bundle, the verified v3 bundle embedded in the release binary is used."
                );
                return Ok(());
            }
            option => return Err(CliError(format!("unknown infer-batch option `{option}`")).into()),
        }
        index += 1;
    }
    let runtime = match bundle_path.as_deref() {
        Some(path) => verify_bundle(path)?.compile()?,
        None => embedded_bundle()?.compile()?,
    };
    let stdin = io::stdin();
    let stdout = io::stdout();
    let count = predict_jsonl(&runtime, &mut stdin.lock(), &mut stdout.lock())?;
    eprintln!(
        "processed {count} rows with model {}",
        runtime.model().model_version
    );
    Ok(())
}

fn robustness_command(arguments: &[String]) -> Result<(), Box<dyn Error>> {
    let operation = arguments
        .first()
        .map(String::as_str)
        .ok_or_else(|| CliError("robustness requires `audit`".into()))?;
    if matches!(operation, "--help" | "-h" | "help") {
        print_robustness_help();
        return Ok(());
    }
    if operation != "audit" {
        return Err(CliError(format!(
            "unknown robustness operation `{operation}`; expected `audit`"
        ))
        .into());
    }

    let mut bundle_path = None;
    let mut bundle_id_test = false;
    let mut gate = RobustnessGate::default();
    let mut index = 1;
    while index < arguments.len() {
        match arguments[index].as_str() {
            "--bundle" => bundle_path = Some(PathBuf::from(option_value(arguments, &mut index)?)),
            "--bundle-id-test" => bundle_id_test = true,
            "--minimum-formatting-label-agreement" => {
                gate.minimum_formatting_label_agreement =
                    parse_option(arguments, &mut index, "formatting label agreement")?
            }
            "--minimum-formatting-decision-agreement" => {
                gate.minimum_formatting_decision_agreement =
                    parse_option(arguments, &mut index, "formatting decision agreement")?
            }
            "--maximum-formatting-js-divergence" => {
                gate.maximum_formatting_js_divergence =
                    parse_option(arguments, &mut index, "formatting JS divergence")?
            }
            "--minimum-typographic-label-agreement" => {
                gate.minimum_typographic_label_agreement = Some(parse_option(
                    arguments,
                    &mut index,
                    "typographic label agreement",
                )?)
            }
            "--minimum-typographic-decision-agreement" => {
                gate.minimum_typographic_decision_agreement = Some(parse_option(
                    arguments,
                    &mut index,
                    "typographic decision agreement",
                )?)
            }
            "--maximum-typographic-js-divergence" => {
                gate.maximum_typographic_js_divergence = Some(parse_option(
                    arguments,
                    &mut index,
                    "typographic JS divergence",
                )?)
            }
            "--help" | "-h" => {
                print_robustness_help();
                return Ok(());
            }
            option => {
                return Err(CliError(format!("unknown robustness audit option `{option}`")).into())
            }
        }
        index += 1;
    }
    gate.validate()?;

    let verified = match bundle_path.as_deref() {
        Some(path) => verify_bundle(path)?,
        None => embedded_bundle()?,
    };
    let report = if bundle_id_test {
        audit_bundle_id_test(verified)?
    } else {
        let runtime = verified.compile()?;
        {
            let stdin = io::stdin();
            audit_jsonl(&runtime, &mut stdin.lock())?
        }
    };
    println!("{}", serde_json::to_string_pretty(&report)?);
    io::stdout().flush()?;
    gate.enforce(&report)?;
    eprintln!(
        "audited {} inputs and {} deterministic variants with model {}",
        report.input_count, report.evaluated_variants, report.model_version
    );
    Ok(())
}

fn train_command(arguments: &[String]) -> Result<(), Box<dyn Error>> {
    let mut dataset_path = None;
    let mut output_path = PathBuf::from(DEFAULT_MODEL);
    let mut report_path = PathBuf::from(DEFAULT_REPORT);
    let mut ood_path = None;
    let mut config = TrainingConfig::default();
    let mut index = 0;
    while index < arguments.len() {
        match arguments[index].as_str() {
            "--dataset" => dataset_path = Some(PathBuf::from(option_value(arguments, &mut index)?)),
            "--output" => output_path = PathBuf::from(option_value(arguments, &mut index)?),
            "--report" => report_path = PathBuf::from(option_value(arguments, &mut index)?),
            "--ood" => ood_path = Some(PathBuf::from(option_value(arguments, &mut index)?)),
            "--seed" => config.seed = parse_option(arguments, &mut index, "seed")?,
            "--epochs" => config.epochs = parse_option(arguments, &mut index, "epochs")?,
            "--learning-rate" => {
                config.learning_rate = parse_option(arguments, &mut index, "learning rate")?
            }
            "--l2" => config.l2_penalty = parse_option(arguments, &mut index, "L2 penalty")?,
            "--holdout" => {
                config.holdout_fraction = parse_option(arguments, &mut index, "holdout fraction")?
            }
            "--max-features" => {
                config.vectorizer.max_features =
                    parse_option(arguments, &mut index, "max features")?
            }
            "--help" | "-h" => {
                print_train_help();
                return Ok(());
            }
            option => return Err(CliError(format!("unknown train option `{option}`")).into()),
        }
        index += 1;
    }

    let mut checked_paths = vec![
        ("model output", output_path.as_path()),
        ("report output", report_path.as_path()),
    ];
    if let Some(path) = dataset_path.as_deref() {
        checked_paths.push(("dataset", path));
    }
    if let Some(path) = ood_path.as_deref() {
        checked_paths.push(("OOD dataset", path));
    }
    ensure_distinct_paths(&checked_paths)?;

    let dataset = load_dataset(dataset_path.as_deref())?;
    let ood_dataset = load_ood_dataset(ood_path.as_deref())?;
    let (mut model, mut report) = IntentModel::train(&dataset, config)?;
    let split = dataset.stratified_split(
        model.training_config.holdout_fraction,
        model.training_config.seed,
    )?;
    let calibration = model.calibrate_thresholds(&split, &ood_dataset, 0.0, 0.98)?;
    report.training_metrics = model.evaluate(split.training_examples())?;
    report.holdout_metrics = model.evaluate(split.holdout_examples())?;
    report.calibration = Some(calibration);
    report.ood_metrics = Some(model.evaluate_ood(ood_dataset.examples())?);
    write_training_artifacts(&model, &output_path, &report, &report_path)?;

    println!("model       {}", output_path.display());
    println!("report      {}", report_path.display());
    println!("fingerprint {}", report.dataset_fingerprint);
    println!(
        "split        {} train / {} holdout (seed {})",
        report.training_example_ids.len(),
        report.holdout_example_ids.len(),
        report.seed
    );
    println!("features     {}", report.vocabulary_size);
    println!(
        "thresholds   confidence={:.2} margin={:.2} (calibrated without holdout)",
        model.training_config.thresholds.minimum_confidence,
        model.training_config.thresholds.minimum_margin,
    );
    print_metrics("train", &report.training_metrics);
    print_metrics("holdout", &report.holdout_metrics);
    print_ood_metrics(
        "out-of-domain",
        report
            .ood_metrics
            .as_ref()
            .expect("the CLI always records OOD evaluation"),
    );
    Ok(())
}

fn evaluate_command(arguments: &[String]) -> Result<(), Box<dyn Error>> {
    let mut dataset_path = None;
    let mut model_path = None;
    let mut ood_path = None;
    let mut json = false;
    let mut index = 0;
    while index < arguments.len() {
        match arguments[index].as_str() {
            "--dataset" => dataset_path = Some(PathBuf::from(option_value(arguments, &mut index)?)),
            "--model" => model_path = Some(PathBuf::from(option_value(arguments, &mut index)?)),
            "--ood" => ood_path = Some(PathBuf::from(option_value(arguments, &mut index)?)),
            "--json" => json = true,
            "--help" | "-h" => {
                println!(
                    "Usage: eliza-lab evaluate [--dataset PATH] [--model PATH] [--ood PATH] [--json]\n\
                     Evaluates the deterministic holdout and a separate unlabeled OOD corpus."
                );
                return Ok(());
            }
            option => return Err(CliError(format!("unknown evaluate option `{option}`")).into()),
        }
        index += 1;
    }

    let dataset = load_dataset(dataset_path.as_deref())?;
    let model = load_model(model_path.as_deref())?;
    verify_fingerprint(&dataset, &model)?;
    let split = dataset.stratified_split(
        model.training_config.holdout_fraction,
        model.training_config.seed,
    )?;
    let metrics = model.evaluate(split.holdout_examples())?;
    let ood_dataset = load_ood_dataset(ood_path.as_deref())?;
    let ood_metrics = model.evaluate_ood(ood_dataset.examples())?;
    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&EvaluationOutput {
                dataset_fingerprint: model.dataset_fingerprint.clone(),
                holdout: metrics,
                out_of_domain: ood_metrics,
            })?
        );
    } else {
        println!(
            "model       {}",
            model_path
                .as_deref()
                .map(|path| path.display().to_string())
                .unwrap_or_else(|| "<embedded eliza-intent-v1>".into())
        );
        println!("fingerprint {}", model.dataset_fingerprint);
        println!("holdout ids {}", split.holdout_examples().len());
        print_metrics("holdout", &metrics);
        print_confusion(&metrics.labels, &metrics.confusion_matrix);
        print_ood_metrics("out-of-domain", &ood_metrics);
    }
    Ok(())
}

fn infer_command(arguments: &[String]) -> Result<(), Box<dyn Error>> {
    let mut bundle_path = None;
    let mut model_path = None;
    let mut legacy_v1 = false;
    let mut json = false;
    let mut prompt = Vec::new();
    let mut index = 0;
    while index < arguments.len() {
        match arguments[index].as_str() {
            "--bundle" => bundle_path = Some(PathBuf::from(option_value(arguments, &mut index)?)),
            "--model" => model_path = Some(PathBuf::from(option_value(arguments, &mut index)?)),
            "--legacy-v1" => legacy_v1 = true,
            "--json" => json = true,
            "--help" | "-h" => {
                println!(
                    "Usage: eliza-lab infer [--bundle PATH] [--json] <fictional prompt>\n\
                     eliza-lab infer --legacy-v1 [--model PATH] [--json] <fictional prompt>\n\
                     The semantically verified open-set bundle is the default. --model is available only with --legacy-v1.\n\
                     Safety and input boundaries run before learned inference."
                );
                return Ok(());
            }
            value if value.starts_with('-') => {
                return Err(CliError(format!("unknown infer option `{value}`")).into())
            }
            value => prompt.push(value.to_owned()),
        }
        index += 1;
    }
    if prompt.is_empty() {
        return Err(CliError("infer requires a fictional prompt".into()).into());
    }
    if legacy_v1 && bundle_path.is_some() {
        return Err(CliError("--bundle cannot be combined with --legacy-v1".into()).into());
    }
    if !legacy_v1 && model_path.is_some() {
        return Err(CliError("--model requires the explicit --legacy-v1 mode".into()).into());
    }
    let mut engine = ElizaEngine::new();
    let prompt = prompt.join(" ");
    let reply = if legacy_v1 {
        let model = load_model(model_path.as_deref())?;
        engine.respond_with_model(&prompt, &model)
    } else {
        let model = match bundle_path.as_deref() {
            Some(path) => verify_bundle(path)?.compile()?,
            None => embedded_bundle()?.compile()?,
        };
        engine.respond_with_open_set(&prompt, &model)
    };
    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&inference_output(&reply))?
        );
    } else {
        print_reply(&reply);
    }
    Ok(())
}

fn chat_command(arguments: &[String]) -> Result<(), Box<dyn Error>> {
    let mut bundle_path = None;
    let mut model_path = None;
    let mut legacy_v1 = false;
    let mut index = 0;
    while index < arguments.len() {
        match arguments[index].as_str() {
            "--bundle" => bundle_path = Some(PathBuf::from(option_value(arguments, &mut index)?)),
            "--model" => model_path = Some(PathBuf::from(option_value(arguments, &mut index)?)),
            "--legacy-v1" => legacy_v1 = true,
            "--help" | "-h" => {
                println!(
                    "Usage: eliza-lab chat [--bundle PATH]\n\
                     eliza-lab chat --legacy-v1 [--model PATH]\n\
                     The semantically verified open-set bundle is the default. --model is available only with --legacy-v1."
                );
                return Ok(());
            }
            option => return Err(CliError(format!("unknown chat option `{option}`")).into()),
        }
        index += 1;
    }
    if legacy_v1 && bundle_path.is_some() {
        return Err(CliError("--bundle cannot be combined with --legacy-v1".into()).into());
    }
    if !legacy_v1 && model_path.is_some() {
        return Err(CliError("--model requires the explicit --legacy-v1 mode".into()).into());
    }
    let model = if legacy_v1 {
        DialogueModel::Legacy(Box::new(load_model(model_path.as_deref())?))
    } else {
        DialogueModel::OpenSet(Box::new(match bundle_path.as_deref() {
            Some(path) => verify_bundle(path)?.compile()?,
            None => embedded_bundle()?.compile()?,
        }))
    };
    interactive(Some(model))
}

fn dataset_command(arguments: &[String]) -> Result<(), Box<dyn Error>> {
    let mut dataset_path = None;
    let mut index = 0;
    while index < arguments.len() {
        match arguments[index].as_str() {
            "check" => {}
            "--dataset" => dataset_path = Some(PathBuf::from(option_value(arguments, &mut index)?)),
            "--help" | "-h" => {
                println!("Usage: eliza-lab dataset check [--dataset PATH]");
                return Ok(());
            }
            option => return Err(CliError(format!("unknown dataset option `{option}`")).into()),
        }
        index += 1;
    }
    let dataset = load_dataset(dataset_path.as_deref())?;
    println!(
        "dataset     {}",
        dataset_path
            .as_deref()
            .map(|path| path.display().to_string())
            .unwrap_or_else(|| "<embedded intents-v1>".into())
    );
    println!("fingerprint {}", dataset.fingerprint());
    println!("examples    {}", dataset.examples().len());
    for (label, count) in dataset.class_counts() {
        println!("class        {label}: {count}");
    }
    Ok(())
}

fn legacy_once(arguments: &[String]) -> Result<(), Box<dyn Error>> {
    let mut engine = ElizaEngine::new();
    let reply = engine.respond(&arguments.join(" "));
    println!("{}", reply.text);
    println!("rule={} turn={}", reply.rule_id, reply.turn);
    Ok(())
}

fn interactive(model: Option<DialogueModel>) -> Result<(), Box<dyn Error>> {
    println!("ELIZA Lab — local dialogue classification you can inspect");
    println!("Educational software, not therapy or medical advice. No transcript is stored.");
    println!(
        "Mode: {}. Type /quit to leave.\n",
        if model.is_some() {
            "learned intent model with deterministic abstention"
        } else {
            "deterministic rule engine"
        }
    );

    let stdin = io::stdin();
    let mut stdin = stdin.lock();
    let mut engine = ElizaEngine::new();
    let mut stdout = io::stdout().lock();
    write!(stdout, "you > ")?;
    stdout.flush()?;

    while let Some(line) = read_bounded_line(&mut stdin)? {
        let input = match line {
            InputLine::Text(input) => input,
            InputLine::TooLong => "x".repeat(MAX_INPUT_CHARS + 1),
        };
        if matches!(input.trim(), "/quit" | "/exit") {
            writeln!(stdout, "eliza > Goodbye.")?;
            break;
        }

        let response = match &model {
            Some(DialogueModel::OpenSet(model)) => engine.respond_with_open_set(&input, model),
            Some(DialogueModel::Legacy(model)) => engine.respond_with_model(&input, model),
            None => engine.respond(&input),
        };
        writeln!(stdout, "eliza > {}", response.text)?;
        write_trace(&mut stdout, &response)?;
        write!(stdout, "you > ")?;
        stdout.flush()?;
    }
    Ok(())
}

fn read_bounded_line(reader: &mut impl BufRead) -> io::Result<Option<InputLine>> {
    let mut bytes = Vec::with_capacity(MAX_INPUT_BYTES.min(256));
    let mut discarded = false;
    let mut saw_input = false;

    loop {
        let buffer = reader.fill_buf()?;
        if buffer.is_empty() {
            if !saw_input {
                return Ok(None);
            }
            break;
        }
        saw_input = true;
        let newline = buffer.iter().position(|byte| *byte == b'\n');
        let content_length = newline.unwrap_or(buffer.len());
        let remaining = (MAX_INPUT_BYTES + 2).saturating_sub(bytes.len());
        let copy_length = content_length.min(remaining);
        bytes.extend_from_slice(&buffer[..copy_length]);
        discarded |= content_length > remaining;

        let consumed = newline.map_or(buffer.len(), |position| position + 1);
        reader.consume(consumed);
        if newline.is_some() {
            break;
        }
    }

    if bytes.last() == Some(&b'\r') {
        bytes.pop();
    }
    if discarded || bytes.len() > MAX_INPUT_BYTES {
        return Ok(Some(InputLine::TooLong));
    }
    let text = String::from_utf8(bytes)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
    Ok(Some(InputLine::Text(text)))
}

fn inference_output(reply: &Reply) -> InferenceOutput<'_> {
    InferenceOutput {
        response: &reply.text,
        boundary: reply.rule_id,
        turn: reply.turn,
        model: reply.model_trace.as_ref().map(|trace| InferenceTrace {
            version: &trace.model_version,
            label: &trace.label,
            accepted: trace.accepted,
            confidence: trace.confidence,
            margin: trace.margin,
            probabilities: &trace.probabilities,
            top_features: &trace.top_features,
        }),
    }
}

fn print_reply(reply: &Reply) {
    println!("{}", reply.text);
    println!("boundary={} turn={}", reply.rule_id, reply.turn);
    if let Some(trace) = &reply.model_trace {
        println!(
            "model={} label={} accepted={} confidence={:.4} margin={:.4}",
            trace.model_version, trace.label, trace.accepted, trace.confidence, trace.margin
        );
        if !trace.top_features.is_empty() {
            println!(
                "features={}",
                trace
                    .top_features
                    .iter()
                    .map(|feature| format!("{}:{:.4}", feature.feature, feature.contribution))
                    .collect::<Vec<_>>()
                    .join(", ")
            );
        }
    }
}

fn write_trace(writer: &mut impl Write, reply: &Reply) -> io::Result<()> {
    writeln!(writer, "trace > {} (turn {})", reply.rule_id, reply.turn)?;
    if let Some(trace) = &reply.model_trace {
        writeln!(
            writer,
            "model > {} / accepted={} / confidence={:.3} / margin={:.3}",
            trace.label, trace.accepted, trace.confidence, trace.margin
        )?;
    }
    Ok(())
}

fn print_metrics(name: &str, metrics: &eliza_lab::ml::EvaluationMetrics) {
    println!(
        "{name:<12} n={} accuracy={:.4} macro_f1={:.4} log_loss={:.4} coverage={:.4} selective_accuracy={}",
        metrics.example_count,
        metrics.accuracy,
        metrics.macro_f1,
        metrics.log_loss,
        metrics.coverage,
        metrics
            .selective_accuracy
            .map(|value| format!("{value:.4}"))
            .unwrap_or_else(|| "n/a".into())
    );
}

fn print_confusion(labels: &[String], matrix: &[Vec<usize>]) {
    println!("confusion matrix: rows=actual, columns=predicted");
    println!("labels: {}", labels.join(", "));
    for (label, row) in labels.iter().zip(matrix) {
        println!("{label:<12} {row:?}");
    }
}

fn print_ood_metrics(name: &str, metrics: &OodMetrics) {
    println!(
        "{name:<12} n={} accepted={} rejected={} coverage={:.4} abstention={:.4} mean_confidence={:.4}",
        metrics.example_count,
        metrics.accepted_examples,
        metrics.rejected_examples,
        metrics.coverage,
        metrics.abstention_rate,
        metrics.mean_confidence,
    );
}

fn verify_fingerprint(dataset: &Dataset, model: &IntentModel) -> Result<(), MlError> {
    let observed = dataset.fingerprint();
    if observed != model.dataset_fingerprint {
        return Err(MlError::InvalidDataset(format!(
            "fingerprint mismatch: model expects {}, dataset is {observed}",
            model.dataset_fingerprint
        )));
    }
    Ok(())
}

fn load_model(path: Option<&Path>) -> Result<IntentModel, MlError> {
    match path {
        Some(path) => IntentModel::read(path),
        None => IntentModel::bundled(),
    }
}

fn load_dataset(path: Option<&Path>) -> Result<Dataset, MlError> {
    match path {
        Some(path) => Dataset::read(path),
        None => Dataset::bundled(),
    }
}

fn load_ood_dataset(path: Option<&Path>) -> Result<OodDataset, MlError> {
    match path {
        Some(path) => OodDataset::read(path),
        None => OodDataset::bundled(),
    }
}

fn ensure_distinct_paths(paths: &[(&str, &Path)]) -> Result<(), CliError> {
    let mut seen = std::collections::HashMap::new();
    for (name, path) in paths {
        let key = collision_key(path)?;
        if let Some(previous) = seen.insert(key, *name) {
            return Err(CliError(format!(
                "{name} collides with {previous}; input and output paths must be distinct"
            )));
        }
    }
    Ok(())
}

fn collision_key(path: &Path) -> Result<String, CliError> {
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        env::current_dir()
            .map_err(|error| CliError(format!("cannot resolve current directory: {error}")))?
            .join(path)
    };

    let existing_ancestor = absolute
        .ancestors()
        .find(|ancestor| ancestor.exists())
        .ok_or_else(|| {
            CliError(format!(
                "cannot find an existing ancestor for {}",
                path.display()
            ))
        })?;
    let unresolved_suffix = absolute.strip_prefix(existing_ancestor).map_err(|error| {
        CliError(format!(
            "cannot resolve the suffix of {}: {error}",
            path.display()
        ))
    })?;
    let canonical_ancestor = existing_ancestor.canonicalize().map_err(|error| {
        CliError(format!(
            "cannot resolve existing ancestor {}: {error}",
            existing_ancestor.display()
        ))
    })?;
    let resolved = canonical_ancestor.join(unresolved_suffix);
    let mut normalized = PathBuf::new();
    for component in resolved.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                normalized.pop();
            }
            _ => normalized.push(component.as_os_str()),
        }
    }
    let key = normalized.to_string_lossy().into_owned();
    Ok(if cfg!(windows) {
        key.to_lowercase()
    } else {
        key
    })
}

fn option_value(arguments: &[String], index: &mut usize) -> Result<String, CliError> {
    let option = arguments[*index].clone();
    *index += 1;
    arguments
        .get(*index)
        .cloned()
        .ok_or_else(|| CliError(format!("{option} requires a value")))
}

fn parse_option<T>(
    arguments: &[String],
    index: &mut usize,
    description: &str,
) -> Result<T, CliError>
where
    T: std::str::FromStr,
{
    let value = option_value(arguments, index)?;
    value
        .parse()
        .map_err(|_| CliError(format!("invalid {description} `{value}`")))
}

fn print_help() {
    println!(
        "ELIZA Lab {APP_VERSION}\n\
         Local, explainable intent classification with a non-clinical dialogue shell.\n\n\
         USAGE\n\
           eliza-lab train [options]\n\
           eliza-lab train-v3 [options]\n\
           eliza-lab selection audit [options]\n\
           eliza-lab evaluate [options]\n\
           eliza-lab infer [options] <fictional prompt>\n\
           eliza-lab infer-batch [--bundle PATH] < input.jsonl\n\
           eliza-lab bundle <verify|reproduce> [options]\n\
           eliza-lab robustness audit [options] [< input.jsonl]\n\
           eliza-lab chat [--bundle PATH]\n\
           eliza-lab dataset check [--dataset PATH]\n\
           eliza-lab --once <prompt>       Legacy rule-only response\n\n\
         Run a command with --help for details. With no command, the rule-only shell starts."
    );
}

fn print_selection_help() {
    println!(
        "Usage: eliza-lab selection audit [options]\n\
         Runs nested group cross-validation on the train + development selection pool only.\n\
         Calibration, ID-test, OOD and contrast partitions are not accepted by this command.\n\
         --dataset PATH              Grouped TSV input (default embedded {DEFAULT_V3_DATASET})\n\
         --output PATH               Canonical JSON report (default {DEFAULT_SELECTION_AUDIT_REPORT})\n\
         --bootstrap-resamples N     Family-clustered 95% interval resamples"
    );
}

fn print_train_v3_help() {
    println!(
        "Usage: eliza-lab train-v3 [options]\n\
         --dataset PATH              Grouped TSV input (default embedded {DEFAULT_V3_DATASET})\n\
         --ood-development PATH      OOD data used only for threshold selection (default embedded {DEFAULT_V3_OOD_DEVELOPMENT})\n\
         --ood-test PATH             Independent OOD evaluation data (default embedded {DEFAULT_V3_OOD_TEST})\n\
         --contrast-test PATH        Frozen paired anti-shortcut test (default embedded {DEFAULT_V3_CONTRAST_TEST})\n\
         --output DIRECTORY          Verified artifact bundle (default {DEFAULT_V3_BUNDLE})\n\
         --seed INTEGER              Deterministic split and bootstrap seed\n\
         --epochs INTEGER            Full-batch gradient steps\n\
         --learning-rate FLOAT       Initial learning rate\n\
         --l2 FLOAT                  Fix the L2 penalty instead of the recorded default grid\n\
         --max-features INTEGER      Fix the vocabulary cap instead of the recorded default grid\n\
         --bootstrap-resamples N     Label-stratified 95% interval resamples"
    );
}

fn print_bundle_help() {
    println!(
        "Usage: eliza-lab bundle verify [--bundle PATH]\n\
         eliza-lab bundle reproduce [--bundle PATH] [data options]\n\
         --bundle PATH               Bundle directory (default {DEFAULT_V3_BUNDLE})\n\
         --dataset PATH              Grouped dataset used only by reproduce\n\
         --ood-development PATH      OOD-development data used only by reproduce\n\
         --ood-test PATH             OOD-test data used only by reproduce\n\
         --contrast-test PATH        Contrast-test data used only by reproduce\n\
         verify recomputes the experiment from the plan embedded in the bundle.\n\
         reproduce requires the source fixtures and proves byte-identical output."
    );
}

fn print_robustness_help() {
    println!(
        "Usage: eliza-lab robustness audit [options] [< input.jsonl]\n\
         Reads bounded JSONL objects with `id` and `text`; emits aggregate metrics without prompts or identifiers.\n\
         --bundle PATH                                  Verified v3 bundle (default embedded bundle)\n\
         --bundle-id-test                               Audit the verified bundle's frozen ID-test instead of stdin\n\
         --minimum-formatting-label-agreement FLOAT    Default 1.0; top labels must remain invariant\n\
         --minimum-formatting-decision-agreement FLOAT Default 1.0; formatting is a preprocessing invariant\n\
         --maximum-formatting-js-divergence FLOAT      Default 0.0; normalized Jensen-Shannon divergence\n\
         --minimum-typographic-label-agreement FLOAT   Optional release threshold for typo stress tests\n\
         --minimum-typographic-decision-agreement FLOAT Optional release threshold for typo stress tests\n\
         --maximum-typographic-js-divergence FLOAT      Optional release threshold for typo stress tests\n\
         The report is written before a selected gate failure so CI can retain the evidence."
    );
}

fn print_train_help() {
    println!(
        "Usage: eliza-lab train [options]\n\
         --dataset PATH              TSV input (default embedded {DEFAULT_DATASET})\n\
         --output PATH               Versioned model JSON (default {DEFAULT_MODEL})\n\
         --report PATH               Split and metric report (default {DEFAULT_REPORT})\n\
         --ood PATH                  OOD fixture (default embedded {DEFAULT_OOD_DATASET})\n\
         --seed INTEGER              Deterministic split seed\n\
         --epochs INTEGER            Full-batch gradient steps\n\
         --learning-rate FLOAT       Initial learning rate\n\
         --l2 FLOAT                  L2 penalty\n\
         --holdout FLOAT             Per-class holdout fraction\n\
         --max-features INTEGER      Vocabulary cap"
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[cfg(unix)]
    fn symlink_directory(source: &Path, destination: &Path) -> std::io::Result<()> {
        std::os::unix::fs::symlink(source, destination)
    }

    #[cfg(windows)]
    fn symlink_directory(source: &Path, destination: &Path) -> std::io::Result<()> {
        std::os::windows::fs::symlink_dir(source, destination)
    }

    #[test]
    fn bounded_reader_drains_an_oversized_line_before_the_next_prompt() {
        let input = format!("{}\nnext\n", "x".repeat(MAX_INPUT_BYTES + 20));
        let mut reader = Cursor::new(input.into_bytes());

        assert!(matches!(
            read_bounded_line(&mut reader).unwrap(),
            Some(InputLine::TooLong)
        ));
        match read_bounded_line(&mut reader).unwrap() {
            Some(InputLine::Text(value)) => assert_eq!(value, "next"),
            _ => panic!("the next bounded line should remain readable"),
        }
    }

    #[test]
    fn bounded_reader_accepts_a_maximum_size_utf8_line() {
        let input = format!("{}\r\n", "🙂".repeat(MAX_INPUT_CHARS));
        let mut reader = Cursor::new(input.into_bytes());

        match read_bounded_line(&mut reader).unwrap() {
            Some(InputLine::Text(value)) => assert_eq!(value.chars().count(), MAX_INPUT_CHARS),
            _ => panic!("a maximum-size UTF-8 line should be accepted"),
        }
    }

    #[test]
    fn fingerprint_guard_rejects_a_different_dataset() {
        let dataset = Dataset::read(std::path::Path::new(DEFAULT_DATASET)).unwrap();
        let (mut model, _) = IntentModel::train(&dataset, TrainingConfig::default()).unwrap();
        model.dataset_fingerprint = "fnv1a64:0000000000000000".into();

        assert!(verify_fingerprint(&dataset, &model).is_err());
    }

    #[test]
    fn collision_keys_resolve_symlinked_parents_for_future_outputs() {
        let directory = std::env::temp_dir().join(format!(
            "eliza-collision-key-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let real_parent = directory.join("real");
        let alias_parent = directory.join("alias");
        std::fs::create_dir_all(&real_parent).unwrap();
        if let Err(error) = symlink_directory(&real_parent, &alias_parent) {
            if error.kind() == std::io::ErrorKind::PermissionDenied {
                std::fs::remove_dir_all(directory).unwrap();
                return;
            }
            panic!("failed to create test directory symlink: {error}");
        }
        if std::fs::read_dir(&alias_parent).is_err() {
            #[cfg(unix)]
            std::fs::remove_file(&alias_parent).unwrap();
            #[cfg(windows)]
            std::fs::remove_dir(&alias_parent).unwrap();
            std::fs::remove_dir_all(directory).unwrap();
            return;
        }

        let real_output = real_parent.join("future/model.json");
        let aliased_output = alias_parent.join("future/model.json");
        assert_eq!(
            collision_key(&real_output).unwrap(),
            collision_key(&aliased_output).unwrap()
        );
        assert!(ensure_distinct_paths(&[
            ("model output", &real_output),
            ("report output", &aliased_output),
        ])
        .is_err());

        #[cfg(unix)]
        std::fs::remove_file(&alias_parent).unwrap();
        #[cfg(windows)]
        std::fs::remove_dir(&alias_parent).unwrap();
        std::fs::remove_dir_all(directory).unwrap();
    }
}
