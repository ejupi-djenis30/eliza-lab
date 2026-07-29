import assert from "node:assert/strict";
import { webcrypto } from "node:crypto";
import { access, readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import path from "node:path";

import {
  EXPECTED_BUNDLE_MANIFEST_SHA256,
  loadOpenSetBundle,
} from "../open-set-engine.mjs";
import {
  EXPECTED_SELECTION_AUDIT_SHA256,
  verifySelectionAudit,
} from "../selection-audit.mjs";

const siteRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const repositoryRoot = path.resolve(siteRoot, "..");
const html = await readFile(path.join(siteRoot, "index.html"), "utf8");
const app = await readFile(path.join(siteRoot, "app.js"), "utf8");
const openSetEngine = await readFile(path.join(siteRoot, "open-set-engine.mjs"), "utf8");
const selectionAuditEngine = await readFile(path.join(siteRoot, "selection-audit.mjs"), "utf8");
const styles = await readFile(path.join(siteRoot, "styles.css"), "utf8");
const notFound = await readFile(path.join(siteRoot, "404.html"), "utf8");
const robots = await readFile(path.join(siteRoot, "robots.txt"), "utf8");
const sitemap = await readFile(path.join(siteRoot, "sitemap.xml"), "utf8");
const socialPreviewSource = await readFile(
  path.join(siteRoot, "assets/social-preview-source.svg"),
  "utf8",
);
const v3BundleRoot = path.join(repositoryRoot, "artifacts/eliza-open-set-v3");
const v3Inventory = ["manifest.json", "metrics.json", "model.json", "policy.json", "split-plan.json"];
const v3Bytes = new Map(
  await Promise.all(
    v3Inventory.map(async (name) => [name, await readFile(path.join(v3BundleRoot, name))]),
  ),
);
const v3ManifestBytes = v3Bytes.get("manifest.json");
const v3MetricsBytes = v3Bytes.get("metrics.json");
const v3PolicyBytes = v3Bytes.get("policy.json");
const v3Manifest = JSON.parse(v3ManifestBytes);
const v3Metrics = JSON.parse(v3MetricsBytes);
const v3Policy = JSON.parse(v3PolicyBytes);
const selectionAuditBytes = await readFile(
  path.join(repositoryRoot, "reports/selection-stability-v1.json"),
);
const selectionAudit = verifySelectionAudit(JSON.parse(selectionAuditBytes));
const verifiedBrowserBundle = await loadOpenSetBundle("https://static.invalid/open-set-v3", {
  crypto: webcrypto,
  fetch: async (url) => {
    const name = new URL(String(url)).pathname.split("/").at(-1);
    const bytes = v3Bytes.get(name);
    return bytes ? new Response(bytes) : new Response("missing", { status: 404 });
  },
});
const expectedCsp = "default-src 'none'; script-src 'self'; style-src 'self'; img-src 'self'; media-src 'none'; connect-src 'self'; worker-src 'none'; manifest-src 'none'; object-src 'none'; base-uri 'none'; form-action 'none'";
const canonicalUrl = "https://ejupi-djenis30.github.io/eliza-lab/";
const socialPreviewUrl = `${canonicalUrl}assets/social-preview.png`;

assert.match(html, /<html lang="en">/);
assert.match(html, /<title>[^<]+<\/title>/);
assert.match(html, /name="description"/);
assert.ok(html.includes(`<link rel="canonical" href="${canonicalUrl}" />`));
assert.ok(html.includes(`property="og:url" content="${canonicalUrl}"`));
assert.match(html, /<meta name="referrer" content="no-referrer" \/>/);
assert.match(html, /http-equiv="Content-Security-Policy"/);
assert.ok(html.includes(`content="${expectedCsp}"`), "The CSP must allow only the static model fetch");
assert.doesNotMatch(html, /frame-ancestors/, "A meta CSP cannot enforce frame-ancestors");
assert.ok(html.includes(`property="og:image" content="${socialPreviewUrl}"`));
assert.match(html, /property="og:image:width" content="1200"/);
assert.match(html, /property="og:image:height" content="675"/);
assert.match(html, /property="og:image:alt"/);
assert.match(html, /name="twitter:card" content="summary_large_image"/);
assert.ok(html.includes(`name="twitter:image" content="${socialPreviewUrl}"`));
assert.match(html, /name="twitter:image:alt"/);
assert.match(html, /type="module" src="app\.js(?:\?[^"\s]+)?"/);
assert.match(app, /\.\/engine\.mjs\?v=[^"\s]+/);
assert.match(app, /\.\/open-set-engine\.mjs\?v=[^"\s]+/);
assert.match(app, /\.\/selection-audit\.mjs\?v=[^"\s]+/);
assert.doesNotMatch(app, /\.\/ml-engine\.mjs/);
assert.doesNotMatch(html, /(?:src|href)="\//, "Assets must remain relative for project Pages");
assert.ok(
  html.includes(
    '<a href="https://github.com/ejupi-djenis30/eliza-lab">ELIZA Lab contributors ↗</a>',
  ),
  "The footer must use collective project attribution.",
);
assert.doesNotMatch(html, /Djenis\s+Ejupi/iu, "The public site must not expose a personal byline.");

const skipLink = '<a class="skip-link" href="#main-content">Skip to content</a>';
assert.ok(html.includes(skipLink), "The site must expose a skip link.");
assert.ok(
  html.includes('<main id="main-content" tabindex="-1">'),
  "The skip-link target must be the focusable main landmark.",
);
assert.ok(
  html.indexOf(skipLink) < html.indexOf('<header class="site-header">'),
  "The skip link must appear before the repeated header.",
);

assert.match(app, /"message-user"/, "The app must identify user messages");
assert.match(app, /MAX_TRANSCRIPT_MESSAGES = 80/, "The transcript must remain bounded");
assert.match(app, /boundedCharacters\(input\.value\)/, "The input must use the Unicode-aware limit");
assert.doesNotMatch(
  html,
  /\bmaxlength=/,
  "Native maxlength counts UTF-16 units; the code-point-aware JavaScript guard owns the limit",
);
assert.match(app, /for \(const character of value\)/, "The input limit must count code points");
assert.match(app, /message-safety/, "Safety exits must have a distinct accessible message");
assert.doesNotMatch(app, /eliza-intent-v1\.json/, "The live app must not load the legacy v1 model");
assert.match(app, /loadOpenSetBundle\("\.\/data\/open-set-v3"\)/, "The app must load the full v3 protocol bundle");
assert.match(app, /V3 VERIFICATION FAILED — INTERACTION DISABLED/, "A verification failure must be visible");
assert.doesNotMatch(app, /RULE FALLBACK/, "Verification failure must not activate an unverified fallback");
assert.match(openSetEngine, /subtle\.digest\("SHA-256"/, "The bundle loader must use Web Crypto SHA-256");
for (const file of v3Inventory) {
  assert.match(openSetEngine, new RegExp(file.replace(".", "\\.")), `The bundle loader must include ${file}`);
}
assert.match(app, /topFeatures/, "The browser trace must expose feature contributions");
assert.match(app, /loadSelectionAudit\("\.\/data\/selection-stability-v1\.json"\)/);
assert.match(html, /The less flattering result/);
assert.match(html, /data-selection-metric="accuracy"/);
assert.equal(selectionAudit.pool.example_count, 385);
assert.equal(selectionAudit.model_fits, 506);
assert.equal(
  Buffer.from(await webcrypto.subtle.digest("SHA-256", selectionAuditBytes)).toString("hex"),
  EXPECTED_SELECTION_AUDIT_SHA256,
  "The browser trust root must pin the selection-audit report bytes",
);
assert.match(selectionAuditEngine, /complete OOF ledger/u);
assert.match(html, /class="lab-shell" aria-busy="true"/);
assert.match(html, /role="status" aria-live="polite" data-model-status/);
assert.match(html, /data-model-gated disabled/);
assert.match(app, /aria-busy", "false"/);
assert.match(app, /try\s*\{[\s\S]*setModelReady\(\);[\s\S]*\}\s*catch\s*\{[\s\S]*setModelFailed\(\);/s);
assert.doesNotMatch(app, /finally\s*\{\s*setModelReady\(\)/s);
assert.match(app, /activeModel = null;\s*engine = null;/s);
assert.match(app, /control\.disabled = true/);
assert.match(styles, /\.message-user\b/, "User messages must have a matching style");
assert.match(styles, /\.message-safety\b/, "Safety messages must have a matching style");
assert.match(styles, /\.pipeline-map\s*\{/, "The pipeline diagram must be code-built");
assert.match(styles, /\.v3-protocol\s*\{/, "The v3 protocol diagram must be code-built");
assert.match(styles, /\.skip-link\s*\{/, "The skip link must have a visible style");
assert.match(styles, /\.skip-link:focus-visible\s*\{/, "The skip link needs a keyboard-focus state");
assert.match(notFound, /<meta name="robots" content="noindex(?:,\s*nofollow)?"/);
assert.ok(notFound.includes(`href="${canonicalUrl}"`), "The 404 page must return to the canonical lab");
assert.ok(robots.includes(`Sitemap: ${canonicalUrl}sitemap.xml`));
assert.ok(sitemap.includes(`<loc>${canonicalUrl}</loc>`));

assert.equal(v3Manifest.schema_version, 3);
assert.equal(v3Manifest.bundle_kind, "eliza-open-set-bundle");
assert.equal(v3Manifest.bundle_version, "3.0.0");
assert.equal(v3Manifest.model_version, "3.0.0");
assert.equal(
  Buffer.from(await webcrypto.subtle.digest("SHA-256", v3ManifestBytes)).toString("hex"),
  EXPECTED_BUNDLE_MANIFEST_SHA256,
  "The browser trust root must pin the checked-in manifest bytes",
);
assert.equal(v3Metrics.schema_version, 3);
assert.equal(v3Metrics.dataset_sha256, v3Manifest.dataset_sha256);
assert.equal(v3Metrics.split_plan_sha256, v3Manifest.split_plan_sha256);
assert.equal(verifiedBrowserBundle.version, v3Manifest.model_version);
assert.deepEqual(verifiedBrowserBundle.metrics, v3Metrics);
assert.deepEqual(v3Metrics.partition_counts, {
  calibration: 70,
  development: 70,
  "id-test": 70,
  "contrast-test": 28,
  "ood-development": 36,
  "ood-test": 36,
  train: 315,
});
assert.deepEqual(v3Metrics.partition_family_counts, {
  calibration: 14,
  development: 14,
  "id-test": 14,
  "contrast-test": 14,
  "ood-development": 12,
  "ood-test": 12,
  train: 63,
});
assert.equal(v3Metrics.paraphrases_per_family, 5);
assert.deepEqual(v3Metrics.ood_domain_counts, {
  "ood-development": 6,
  "ood-test": 6,
});
assert.deepEqual(v3Metrics.ood_stratum_counts, {
  "ood-development": { capability: 12, noise: 12, semantic: 12 },
  "ood-test": { capability: 12, noise: 12, semantic: 12 },
});
assert.equal(v3Metrics.development_selection.candidates.length, 9);
assert.equal(v3Metrics.development_selection.training_example_count, 315);
assert.equal(v3Metrics.development_selection.training_family_count, 63);
assert.equal(v3Metrics.development_selection.development_example_count, 70);
assert.equal(v3Metrics.development_selection.development_family_count, 14);
assert.deepEqual(v3Metrics.development_selection.inputs, ["train", "development"]);
assert.equal(v3Metrics.development_selection.macro_f1_tolerance, 0.005);
assert.deepEqual(v3Metrics.threshold_selection.inputs, ["development", "ood-development"]);
assert.equal(v3Metrics.threshold_selection.evaluated_candidate_count, 49);
assert.ok(v3Metrics.threshold_selection.feasible_candidate_count > 0);
assert.equal(v3Policy.temperature_source, "calibration-partition-temperature-scaling-v3");
assert.equal(v3Policy.threshold_source, "fixed-development-plus-ood-development-grid-v3");
assert.equal(v3Metrics.bootstrap_95.resamples, 1_000);
assert.equal(v3Metrics.bootstrap_95.strategy, "label-stratified-id-family-and-ood-domain-cluster-percentile-v3");
assert.equal(v3Metrics.id_test.example_count, 70);
assert.equal(v3Metrics.ood_test.example_count, 36);
assert.equal(v3Metrics.contrast_test.example_count, 28);
assert.equal(v3Metrics.contrast_test.pair_count, 14);
assert.equal(v3Metrics.contrast_test.predictions.length, 28);
assert.match(html, /Contrast pair accuracy/);
assert.match(app, /contrast\.pair_accuracy/);
assert.equal(v3Metrics.baselines.strategy, "training-only-majority-and-laplace-unigram-naive-bayes-v3");
assert.deepEqual(v3Metrics.baselines.inputs, ["train"]);
assert.deepEqual(Object.keys(v3Metrics.ood_test.by_stratum).sort(), ["capability", "noise", "semantic"]);
assert.ok(v3Metrics.id_test.accuracy >= v3Metrics.bootstrap_95.id_accuracy.lower_95);
assert.ok(v3Metrics.id_test.accuracy <= v3Metrics.bootstrap_95.id_accuracy.upper_95);
assert.ok(v3Metrics.ood_test.discrimination.auroc >= v3Metrics.bootstrap_95.ood_auroc.lower_95);
assert.ok(v3Metrics.ood_test.discrimination.auroc <= v3Metrics.bootstrap_95.ood_auroc.upper_95);
assert.match(html, /No test row chooses a weight, temperature or threshold\./);
assert.match(html, /majority and unigram baselines/i);
assert.doesNotMatch(html, /data-report-metric=/u, "The primary page must not expose the legacy v1 report");
assert.doesNotMatch(html, /SOFTMAX V1|112 synthetic examples|441 synthetic rows/iu);
assert.doesNotMatch(
  html,
  /deterministic fallback/iu,
  "The v3 presentation must describe visible abstention rather than legacy fallback framing",
);

for (const file of [
  "app.js",
  "engine.mjs",
  "ml-engine.mjs",
  "open-set-engine.mjs",
  "selection-audit.mjs",
  "styles.css",
  "assets/eliza-lab-mark.svg",
  "assets/eliza-lab-lockup.svg",
  "assets/social-preview.png",
  "assets/social-preview-source.svg",
]) {
  await access(path.join(siteRoot, file));
}

const socialPreview = await readFile(path.join(siteRoot, "assets/social-preview.png"));
assert.equal(socialPreview.subarray(0, 8).toString("hex"), "89504e470d0a1a0a", "Social preview must be PNG");
assert.equal(socialPreview.readUInt32BE(16), 1_200, "Social preview width must be 1200 pixels");
assert.equal(socialPreview.readUInt32BE(20), 675, "Social preview height must be 675 pixels");
assert.match(socialPreviewSource, /FIT[\s\S]*SELECT[\s\S]*CALIBRATE[\s\S]*FREEZE[\s\S]*TEST/u);
assert.match(socialPreviewSource, /ID-TEST ACCURACY/u);
assert.match(socialPreviewSource, /OOD AUROC/u);
assert.match(socialPreviewSource, /CONTRAST TEST/u);
assert.doesNotMatch(
  socialPreviewSource,
  /RESPONSE TRACE|feeling-reflection|my → your|RULE FALLBACK/iu,
  "The social preview must describe v3 instead of the legacy rule demo",
);

const stageIndex = process.argv.indexOf("--stage");
if (stageIndex >= 0) {
  const stageArgument = process.argv[stageIndex + 1];
  assert.ok(stageArgument, "--stage requires a directory");
  const stage = path.resolve(repositoryRoot, stageArgument);
  assert.equal(
    await readFile(path.join(stage, "open-set-engine.mjs"), "utf8"),
    openSetEngine,
    "The deployed browser verifier must match the checked-in module",
  );
  for (const file of v3Inventory) {
    assert.deepEqual(
      await readFile(path.join(stage, "data/open-set-v3", file)),
      await readFile(path.join(v3BundleRoot, file)),
      `The deployed v3 ${file} must be byte-identical to the checked-in bundle`,
    );
  }
  assert.deepEqual(
    await readFile(path.join(stage, "data/selection-stability-v1.json")),
    selectionAuditBytes,
    "The deployed selection audit must be byte-identical to the checked-in report",
  );
  for (const file of [
    "404.html",
    "app.js",
    "engine.mjs",
    "index.html",
    "open-set-engine.mjs",
    "robots.txt",
    "selection-audit.mjs",
    "sitemap.xml",
    "styles.css",
  ]) {
    assert.deepEqual(
      await readFile(path.join(stage, file)),
      await readFile(path.join(siteRoot, file)),
      `The staged ${file} must be byte-identical to its reviewed source`,
    );
  }
}

console.log("ELIZA Lab site and v3 bundle validation passed.");
