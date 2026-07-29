import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

import {
  assertPinnedActionReferences,
  assertReleaseCandidateGate,
  assertReleasePermissions,
  assertReleaseRecoveryPermissions,
  parseWorkflowYaml,
} from "../workflow-policy.mjs";

const repositoryRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");
const workflow = readFileSync(path.join(repositoryRoot, ".github", "workflows", "release.yml"), "utf8");
const recoveryWorkflow = readFileSync(path.join(repositoryRoot, ".github", "workflows", "release-recovery.yml"), "utf8");
const continuousIntegration = readFileSync(path.join(repositoryRoot, ".github", "workflows", "ci.yml"), "utf8");
const pages = readFileSync(path.join(repositoryRoot, ".github", "workflows", "pages.yml"), "utf8");
const gitAttributes = readFileSync(path.join(repositoryRoot, ".gitattributes"), "utf8");
const workflowFixtures = path.join(repositoryRoot, "scripts", "tests", "fixtures", "workflows");
const hiddenJobWriteAll = readFileSync(path.join(workflowFixtures, "hidden-job-write-all.yml"), "utf8");
const hiddenStepsFloatingAction = readFileSync(path.join(workflowFixtures, "hidden-steps-floating-action.yml"), "utf8");
const trustedTagCondition = "github.event_name == 'push' && github.ref_type == 'tag' && startsWith(github.ref_name, 'v')";

function jobBlock(name) {
  const match = workflow.match(new RegExp(`^  ${name}:\\r?\\n([\\s\\S]*?)(?=^  [a-z][a-z0-9_-]*:\\r?$|(?![\\s\\S]))`, "mu"));
  assert.ok(match, `release workflow is missing the ${name} job`);
  return match[1];
}

function replaceOnce(source, before, after) {
  const index = source.indexOf(before);
  assert.ok(index >= 0, `mutation source is missing ${JSON.stringify(before)}`);
  assert.equal(source.indexOf(before, index + before.length), -1, `mutation source repeats ${JSON.stringify(before)}`);
  return source.slice(0, index) + after + source.slice(index + before.length);
}

test("OIDC attestation and publication require a push event for a tag", () => {
  for (const job of ["attest", "publish"]) {
    assert.match(jobBlock(job), new RegExp(`^    if: ${trustedTagCondition.replace(/[.*+?^${}()|[\]\\]/gu, "\\$&")}\\r?$`, "mu"));
  }

  const mayMutateRelease = (eventName, refType, refName) => eventName === "push" && refType === "tag" && refName.startsWith("v");
  assert.equal(mayMutateRelease("workflow_dispatch", "tag", "v1.1.0"), false);
  assert.equal(mayMutateRelease("push", "branch", "v1.1.0"), false);
  assert.equal(mayMutateRelease("push", "tag", "v1.1.0"), true);
});

test("the pinned RustSec gate is causal for every release mutation", () => {
  const quality = jobBlock("quality");
  assert.match(quality, /audit_rust_version=.*rustVersion/u);
  assert.match(quality, /rustup toolchain install "\$audit_rust_version" --profile minimal/u);
  assert.match(quality, /cargo "\+\$audit_rust_version" install cargo-audit --version "\$audit_version" --locked/u);
  assert.match(quality, /--no-fetch/u);
  assert.match(quality, /--deny warnings/u);
  assert.match(quality, /--json/u);
  assert.match(quality, /audit-policy/u);
  assert.match(quality, /--database-epoch "\$actual_database_epoch"/u);
  assert.match(quality, /rustsec-audit-policy\.json/u);
  assert.match(quality, /rustsec-audit\.json/u);
  assert.match(jobBlock("build"), /^    needs: quality\r?$/mu);
  assert.match(jobBlock("assemble"), /^    needs: \[quality, build\]\r?$/mu);
  const releaseCandidateGate = jobBlock("release_candidate_gate");
  assert.match(releaseCandidateGate, /^    if: always\(\)\r?$/mu);
  assert.match(releaseCandidateGate, /^    needs: \[quality, build, assemble\]\r?$/mu);
  for (const result of ["QUALITY_RESULT", "BUILD_RESULT", "ASSEMBLE_RESULT"]) {
    assert.match(releaseCandidateGate, new RegExp(`\\[\\[ "\\$${result}" == "success" \\]\\]`, "u"));
  }
  assert.match(jobBlock("attest"), /^    needs: release_candidate_gate\r?$/mu);
  assert.match(jobBlock("publish"), /^    needs: \[release_candidate_gate, attest\]\r?$/mu);
});

test("a tag cannot reach publication until repository licensing is explicitly approved", () => {
  const quality = jobBlock("quality");
  const policyIndex = quality.indexOf("name: Enforce explicit license approval before a tag can publish");
  const testIndex = quality.indexOf("name: Test release tooling");
  assert.ok(policyIndex >= 0 && testIndex > policyIndex);
  assert.match(
    quality,
    new RegExp(`if: ${trustedTagCondition.replace(/[.*+?^${}()|[\]\\]/gu, "\\$&")}`, "u"),
  );
  assert.match(quality, /node scripts\/release-policy\.mjs verify/u);
  const parsedWorkflow = parseWorkflowYaml(workflow, "release workflow");
  const pullRequest = parsedWorkflow.on?.pull_request;
  assert.deepEqual(Object.keys(pullRequest), ["branches"]);
  assert.deepEqual(pullRequest.branches, ["master", "main"]);
  assert.equal(Object.hasOwn(pullRequest, "paths"), false);
});

test("release tooling and permissions stay pinned and least-privilege", () => {
  assertReleasePermissions(workflow, "release workflow");
  assertReleaseCandidateGate(workflow, "release workflow");
  for (const [label, source] of [
    ["release workflow", workflow],
    ["CI workflow", continuousIntegration],
    ["Pages workflow", pages],
  ]) {
    assertPinnedActionReferences(source, label);
  }
  assert.equal((workflow.match(/^          node-version: 22\.23\.1\r?$/gmu) || []).length, 4);
  assert.doesNotMatch(workflow, /ubuntu-latest|windows-latest|macos-latest/u);
  assert.match(
    workflow,
    /^  group: \$\{\{ startsWith\(github\.ref, 'refs\/tags\/'\) && 'eliza-release-publication' \|\| format\('release-\{0\}-\{1\}', github\.workflow, github\.ref\) \}\}\r?$/mu,
  );

  const publish = jobBlock("publish");
  assert.match(publish, /GH_CLI_VERSION: "2\.94\.0"/u);
  assert.match(publish, /GH_CLI_SHA256: "a757f1ba6db18f4de8cbadb244843a5f89bc75b5e7c6fc127d2bd77fbd12ed62"/u);
  assert.match(publish, /sha256sum --check --strict/u);
});

test("release recovery is an explicit least-privilege default-branch operation over the original artifact", () => {
  const parsed = assertReleaseRecoveryPermissions(recoveryWorkflow, "release recovery workflow");
  assertPinnedActionReferences(recoveryWorkflow, "release recovery workflow");
  assert.deepEqual(Object.keys(parsed.on), ["workflow_dispatch"]);
  assert.deepEqual(
    Object.keys(parsed.on.workflow_dispatch.inputs),
    ["release_tag", "release_commit", "source_run_id", "draft_release_id"],
  );
  for (const input of Object.values(parsed.on.workflow_dispatch.inputs)) {
    assert.equal(input.required, "true");
    assert.equal(input.type, "string");
  }
  assert.equal(parsed.concurrency.group, "eliza-release-publication");
  assert.equal(parsed.concurrency["cancel-in-progress"], "false");
  assert.equal(
    parsed.jobs.recovery.if,
    "github.ref == format('refs/heads/{0}', github.event.repository.default_branch)",
  );
  assert.equal(parsed.jobs.recovery["runs-on"], "ubuntu-22.04");
  assert.equal(parsed.jobs.recovery["timeout-minutes"], "10");
  assert.equal(parsed.jobs.recovery.permissions.actions, "read");
  assert.match(
    recoveryWorkflow,
    /Authorization and mutation intentionally share one job[\s\S]*unsigned job outputs or a newly uploaded intermediary/u,
  );
  assert.match(
    recoveryWorkflow,
    /actions:read retrieves the exact validated Publish job log and the original inventory/u,
  );

  assert.match(recoveryWorkflow, /name: Download original verified release inventory/u);
  assert.match(recoveryWorkflow, /git -C release-source rev-parse 'HEAD\^\{commit\}'/u);
  assert.match(recoveryWorkflow, /name: verified-release-assets/u);
  assert.match(recoveryWorkflow, /run-id: \$\{\{ inputs\.source_run_id \}\}/u);
  assert.match(recoveryWorkflow, /github-token: \$\{\{ github\.token \}\}/u);
  assert.match(recoveryWorkflow, /--signer-digest "\$\{RECOVERY_COMMIT\}"/u);
  assert.match(recoveryWorkflow, /--source-digest "\$\{RECOVERY_COMMIT\}"/u);
  assert.match(recoveryWorkflow, /--source-ref "refs\/tags\/\$\{RECOVERY_TAG\}"/u);
  assert.match(recoveryWorkflow, /--deny-self-hosted-runners/u);
  assert.match(recoveryWorkflow, /--format json > recovery-attestation\.json/u);
  assert.match(recoveryWorkflow, /release-publish\.mjs recover-empty-draft/u);
  assert.match(recoveryWorkflow, /--manifest release-source\/Cargo\.toml/u);
  assert.match(recoveryWorkflow, /--policy release-source\/\.github\/release-policy\.json/u);
  assert.doesNotMatch(recoveryWorkflow, /cargo build|verify-assets|actions\/upload-artifact|gh release upload/u);
  assert.match(recoveryWorkflow, /GH_CLI_VERSION: "2\.94\.0"/u);
  assert.match(recoveryWorkflow, /GH_CLI_SHA256: "a757f1ba6db18f4de8cbadb244843a5f89bc75b5e7c6fc127d2bd77fbd12ed62"/u);
});

test("ordinary release workflow_dispatch remains unable to attest or publish", () => {
  const parsed = parseWorkflowYaml(workflow, "release workflow");
  assert.ok(parsed.on.workflow_dispatch);
  for (const job of ["attest", "publish"]) {
    assert.equal(parsed.jobs[job].if, trustedTagCondition);
  }
  assert.doesNotMatch(jobBlock("quality"), /contents: write|attestations: write/u);
  assert.doesNotMatch(jobBlock("build"), /contents: write|attestations: write/u);
  assert.doesNotMatch(jobBlock("assemble"), /contents: write|attestations: write/u);
});

test("every native release binary proves the V3 model contract before packaging", () => {
  const build = jobBlock("build");
  const buildIndex = build.indexOf("name: Build locked release binary");
  const smokeIndex = build.indexOf("name: Smoke-test built CLI");
  const packageIndex = build.indexOf("name: Package binary with checksum and provenance metadata");
  assert.ok(buildIndex >= 0 && smokeIndex > buildIndex && packageIndex > smokeIndex);
  assert.match(build, /node scripts\/release-contract\.mjs smoke/u);
  assert.match(build, /--bundle artifacts\/eliza-open-set-v3/u);
  assert.match(build, /--legacy-model models\/eliza-intent-v1\.json/u);
  assert.match(build, /node scripts\/release-contract\.mjs smoke-archive/u);

  const releaseContract = readFileSync(path.join(repositoryRoot, "scripts", "release-contract.mjs"), "utf8");
  for (const required of [
    '["infer", "--json", "Hello, I want to make a concrete plan"]',
    '["infer", "--bundle", resolvedBundle, "--json"',
    '["bundle", "verify", "--bundle", resolvedBundle]',
    '["bundle", "reproduce", "--bundle", resolvedBundle]',
    'trace?.model?.version === "3.0.0"',
    'typeof trace.model.accepted === "boolean"',
    'Number.isFinite(trace.model.confidence)',
    'Number.isFinite(trace.model.margin)',
    '"--legacy-v1"',
  ]) {
    assert.ok(releaseContract.includes(required), `release smoke is missing ${required}`);
  }
  assert.doesNotMatch(releaseContract, /spawnSync\(resolvedBinary, \["--once"/u);
  assert.doesNotMatch(releaseContract, /rule=feeling-reflection/u);
});

test("release quality regenerates and verifies the robustness report before publication", () => {
  const quality = jobBlock("quality");
  const reproduction = quality.indexOf("name: Reproduce every embedded ML artifact");
  const audit = quality.indexOf(
    "cargo run --locked -- robustness audit --bundle-id-test",
    reproduction,
  );
  const reportPath = "target/release-robustness-id-test-report.json";
  const output = quality.indexOf(reportPath, audit);
  const verifier = quality.indexOf("node scripts/verify-robustness-report.mjs", output);
  const verifiedInput = quality.indexOf(reportPath, verifier);
  const nextGate = quality.indexOf(
    "name: Audit locked Rust dependencies against pinned RustSec data",
    verifier,
  );

  assert.ok(reproduction >= 0, "release quality must reproduce the ML artifacts");
  assert.ok(audit > reproduction, "release quality must generate the robustness report");
  assert.ok(output > audit, "release quality must write the generated report");
  assert.ok(verifier > output, "release quality must verify after generation");
  assert.ok(verifiedInput > verifier, "release quality must verify the generated report path");
  assert.ok(nextGate > verifiedInput, "robustness verification must remain inside the quality gate");
  assert.match(jobBlock("build"), /^    needs: quality\r?$/mu);
});

test("digest-bound fixtures keep LF bytes on every release runner", () => {
  for (const pattern of [
    "/artifacts/eliza-open-set-v3/*.json text eol=lf",
    "/models/*.json text eol=lf",
    "/reports/*.json text eol=lf",
    "/fixtures/*.tsv text eol=lf",
  ]) {
    assert.ok(gitAttributes.includes(pattern), `missing cross-platform byte contract: ${pattern}`);
  }
});

test("the release candidate gate rejects every fail-open or contract drift mutation", () => {
  const newline = workflow.includes("\r\n") ? "\r\n" : "\n";
  const mutations = [
    [
      "hardcoded successful dependency result",
      "          ASSEMBLE_RESULT: ${{ needs.assemble.result }}",
      "          ASSEMBLE_RESULT: success",
      /ASSEMBLE_RESULT must be/u,
    ],
    [
      "job-level continue-on-error",
      `    name: Release candidate gate${newline}    if: always()${newline}    needs: [quality, build, assemble]${newline}    runs-on: ubuntu-22.04${newline}    timeout-minutes: 2`,
      `    name: Release candidate gate${newline}    if: always()${newline}    needs: [quality, build, assemble]${newline}    runs-on: ubuntu-22.04${newline}    timeout-minutes: 2${newline}    continue-on-error: true`,
      /must not define continue-on-error/u,
    ],
    [
      "step-level continue-on-error",
      `          QUALITY_RESULT: \${{ needs.quality.result }}${newline}        shell: bash`,
      `          QUALITY_RESULT: \${{ needs.quality.result }}${newline}        shell: bash${newline}        continue-on-error: true`,
      /step must not define continue-on-error/u,
    ],
    [
      "unexpected job key",
      `    name: Release candidate gate${newline}    if: always()`,
      `    name: Release candidate gate${newline}    environment: release${newline}    if: always()`,
      /release_candidate_gate must contain exactly/u,
    ],
    [
      "unexpected step key",
      "      - name: Require every release candidate stage to pass",
      `      - name: Require every release candidate stage to pass${newline}        id: hidden-bypass`,
      /release_candidate_gate step must contain exactly/u,
    ],
    [
      "shell fail-open suffix",
      '          [[ "$QUALITY_RESULT" == "success" ]]',
      '          [[ "$QUALITY_RESULT" == "success" ]] || true',
      /run body must stay exact/u,
    ],
    [
      "condition drift",
      "    if: always()",
      "    if: success()",
      /if must be always/u,
    ],
    [
      "dependency drift",
      "    needs: [quality, build, assemble]",
      "    needs: [assemble]",
      /needs must be exactly/u,
    ],
    [
      "run body drift",
      '          [[ "$BUILD_RESULT" == "success" ]]',
      `          [[ "$BUILD_RESULT" == "success" ]]${newline}          echo bypass`,
      /run body must stay exact/u,
    ],
    [
      "extra step",
      `          [[ "$ASSEMBLE_RESULT" == "success" ]]${newline}${newline}  attest:`,
      `          [[ "$ASSEMBLE_RESULT" == "success" ]]${newline}      - name: Hidden bypass${newline}        run: exit 0${newline}${newline}  attest:`,
      /must contain exactly one step/u,
    ],
  ];

  for (const [name, before, after, expected] of mutations) {
    const fixture = replaceOnce(workflow, before, after);
    assert.throws(() => assertReleaseCandidateGate(fixture, `${name} fixture`), expected);
  }
});

test("YAML line separators cannot hide fail-open release gate keys behind comments", () => {
  const canonicalWorkflow = workflow.replaceAll("\r\n", "\n");
  const separators = [
    ["LF", "\n"],
    ["CRLF", "\r\n"],
    ["CR", "\r"],
    ["NEL", "\u0085"],
    ["LS", "\u2028"],
    ["PS", "\u2029"],
  ];

  for (const [name, separator] of separators) {
    const separatedWorkflow = canonicalWorkflow.replaceAll("\n", separator);
    assert.doesNotThrow(
      () => assertReleaseCandidateGate(separatedWorkflow, `${name} canonical fixture`),
      `${name} must be parsed as a YAML line separator`,
    );

    const jobFixture = replaceOnce(
      canonicalWorkflow,
      "    if: always()",
      `    if: always() #${separator}    continue-on-error: true`,
    );
    assert.throws(
      () => assertReleaseCandidateGate(jobFixture, `${name} job separator fixture`),
      /release_candidate_gate must not define continue-on-error/u,
    );

    const stepFixture = replaceOnce(
      canonicalWorkflow,
      "        shell: bash\n        run: |\n          set -euo pipefail\n          [[ \"$QUALITY_RESULT\" == \"success\" ]]",
      `        shell: bash #${separator}        continue-on-error: true\n        run: |\n          set -euo pipefail\n          [[ \"$QUALITY_RESULT\" == \"success\" ]]`,
    );
    assert.throws(
      () => assertReleaseCandidateGate(stepFixture, `${name} step separator fixture`),
      /release_candidate_gate step must not define continue-on-error/u,
    );

    const duplicateKeyFixture = replaceOnce(
      canonicalWorkflow,
      "    if: always()",
      `    if: always() #${separator}    if: success()`,
    );
    assert.throws(
      () => parseWorkflowYaml(duplicateKeyFixture, `${name} duplicate key fixture`),
      /duplicate mapping key if/u,
    );
  }
});

const validPermissionsFixture = `permissions:
  contents: read
jobs:
  quality:
    runs-on: ubuntu-22.04
  build:
    runs-on: ubuntu-22.04
  assemble:
    runs-on: ubuntu-22.04
  release_candidate_gate:
    runs-on: ubuntu-22.04
  attest:
    permissions:
      attestations: write
      contents: read
      id-token: write
  publish:
    permissions:
      attestations: read
      contents: write
`;

test("structured permission policy rejects scalar, extra, duplicate, overridden, and misplaced permissions", () => {
  assert.doesNotThrow(() => assertReleasePermissions(validPermissionsFixture, "valid fixture"));

  for (const scalar of ["write-all", "read-all"]) {
    const fixture = validPermissionsFixture.replace("permissions:\n  contents: read", `permissions: ${scalar}`);
    assert.throws(() => assertReleasePermissions(fixture, `${scalar} fixture`), /top-level permissions must be a mapping/u);
  }

  const extraPermission = validPermissionsFixture.replace("  contents: read\njobs:", "  contents: read\n  issues: write\njobs:");
  assert.throws(() => assertReleasePermissions(extraPermission, "extra fixture"), /must contain exactly contents/u);

  const duplicatePermission = validPermissionsFixture.replace("  contents: read\njobs:", "  contents: read\n  contents: write\njobs:");
  assert.throws(() => assertReleasePermissions(duplicatePermission, "duplicate fixture"), /duplicate mapping key contents/u);

  for (const jobName of ["quality", "build", "assemble", "release_candidate_gate"]) {
    const override = validPermissionsFixture.replace(
      `  ${jobName}:\n    runs-on: ubuntu-22.04`,
      `  ${jobName}:\n    permissions:\n      contents: write\n    runs-on: ubuntu-22.04`,
    );
    assert.throws(() => assertReleasePermissions(override, `${jobName} override fixture`), /must inherit/u);
  }

  const movedUnderEnvironment = validPermissionsFixture.replace(
    "  publish:\n    permissions:\n      attestations: read\n      contents: write",
    "  publish:\n    env:\n      attestations: read\n      contents: write",
  );
  assert.throws(() => assertReleasePermissions(movedUnderEnvironment, "misplaced fixture"), /publish permissions must be a mapping/u);

  for (const [jobName, unexpectedPermission] of [
    ["attest", "      packages: write\n"],
    ["publish", "      id-token: write\n"],
  ]) {
    const fixture = validPermissionsFixture.replace(
      `  ${jobName}:\n    permissions:\n`,
      `  ${jobName}:\n    permissions:\n${unexpectedPermission}`,
    );
    assert.throws(() => assertReleasePermissions(fixture, `${jobName} extra fixture`), /must contain exactly/u);
  }

  const duplicatePublishPermission = validPermissionsFixture.replace(
    "      attestations: read\n      contents: write",
    "      attestations: read\n      contents: write\n      contents: read",
  );
  assert.throws(
    () => assertReleasePermissions(duplicatePublishPermission, "publish duplicate fixture"),
    /duplicate mapping key contents/u,
  );

  assert.throws(
    () => assertReleasePermissions(hiddenJobWriteAll, "hidden job fixture"),
    /YAML anchor, alias, or tag/u,
  );
});

test("structured action policy rejects symbolic, version, dynamic, and local action references", () => {
  const pinned = "actions/checkout@9c091bb21b7c1c1d1991bb908d89e4e9dddfe3e0";
  const fixture = (reference) => `jobs:
  verify:
    steps:
      # uses: actions/checkout@main
      - run: |
          uses: actions/checkout@main
      - uses: ${reference}
`;

  assert.deepEqual(assertPinnedActionReferences(fixture(pinned), "pinned fixture"), [pinned]);
  for (const reference of ["actions/checkout@main", "actions/checkout@v7", "${{ matrix.action }}"]) {
    assert.throws(
      () => assertPinnedActionReferences(fixture(reference), `${reference} fixture`),
      /valid remote GitHub owner\/repository@lowercase-40-character-commit/u,
    );
  }

  const commit = "9c091bb21b7c1c1d1991bb908d89e4e9dddfe3e0";
  for (const reference of [
    "./.github/actions/local",
    "../outside/action",
    ".github/actions@" + commit,
    "actions/.github@" + commit,
    "actions/checkout/subdirectory@" + commit,
    "docker://alpine:3.23",
    "-invalid/repository@" + commit,
    "invalid-/repository@" + commit,
    "invalid--owner/repository@" + commit,
  ]) {
    assert.throws(
      () => assertPinnedActionReferences(fixture(reference), `${reference} fixture`),
      /local actions are forbidden/u,
    );
  }

  const flowMappingFixture = `jobs:
  verify:
    steps: [{ uses: actions/checkout@main }]
`;
  assert.throws(
    () => assertPinnedActionReferences(flowMappingFixture, "flow mapping fixture"),
    /unsupported flow mapping/u,
  );

  assert.throws(
    () => assertPinnedActionReferences(hiddenStepsFloatingAction, "hidden steps fixture"),
    /YAML anchor, alias, or tag/u,
  );
});

test("policy-managed jobs, steps, and permission blocks stay structurally typed", () => {
  const commit = "9c091bb21b7c1c1d1991bb908d89e4e9dddfe3e0";
  for (const [name, source, expected] of [
    ["jobs sequence", `jobs:\n  - verify\n`, /jobs must be a mapping/u],
    ["job scalar", `jobs:\n  verify: scalar\n`, /job verify must be a mapping/u],
    ["steps mapping", `jobs:\n  verify:\n    steps:\n      uses: actions\/checkout@${commit}\n`, /steps must be a sequence/u],
    ["step scalar", `jobs:\n  verify:\n    steps:\n      - echo fixture\n`, /step 1 must be a mapping/u],
    ["job permission scalar", `jobs:\n  verify:\n    permissions: write-all\n    steps:\n      - uses: actions\/checkout@${commit}\n`, /permissions must be a mapping/u],
  ]) {
    assert.throws(() => assertPinnedActionReferences(source, `${name} fixture`), expected);
  }

  for (const metadata of ["&label Release", "*label", "!!str Release", "!custom Release"]) {
    assert.throws(
      () => parseWorkflowYaml(`name: ${metadata}\njobs:\n  verify:\n    runs-on: ubuntu-22.04\n`, `${metadata} fixture`),
      /YAML anchor, alias, or tag/u,
    );
  }
  for (const flowMetadata of ["[*label]", "[value, &label other]", "[!!str value]"]) {
    assert.throws(
      () => parseWorkflowYaml(`name: Release\non:\n  push:\n    branches: ${flowMetadata}\njobs:\n  verify:\n    runs-on: ubuntu-22.04\n`, `${flowMetadata} fixture`),
      /YAML anchor, alias, or tag/u,
    );
  }
});

test("publication independently verifies every attestation identity", () => {
  const publish = jobBlock("publish");
  const install = publish.indexOf("name: Install verified GitHub CLI");
  const verification = publish.indexOf("name: Verify release attestations before publication");
  const mutation = publish.indexOf("node scripts/release-publish.mjs publish");
  assert.ok(install >= 0 && verification > install && mutation > verification);
  for (const binding of [
    '--repo "${GITHUB_REPOSITORY}"',
    '--signer-workflow "${GITHUB_REPOSITORY}/.github/workflows/release.yml"',
    '--source-digest "${GITHUB_SHA}"',
    '--source-ref "${GITHUB_REF}"',
    '--predicate-type "https://slsa.dev/provenance/v1"',
    '--cert-oidc-issuer "https://token.actions.githubusercontent.com"',
    "--deny-self-hosted-runners",
  ]) {
    assert.ok(publish.includes(binding), `missing attestation identity binding: ${binding}`);
  }
  assert.match(publish, /find release-assets -maxdepth 1 -type f -print0 \| sort -z/u);
});

test("CI and Pages pin runners and toolchains instead of floating production inputs", () => {
  for (const source of [continuousIntegration, pages]) {
    assert.doesNotMatch(source, /ubuntu-latest|windows-latest|macos-latest/u);
  }
  assert.doesNotMatch(continuousIntegration, /^\s*toolchain:\s*stable\s*$/mu);
  assert.doesNotMatch(continuousIntegration, /^\s*node-version:\s*22\s*$/mu);
  assert.match(continuousIntegration, /toolchain: 1\.81\.0/u);
  assert.match(continuousIntegration, /node-version: 22\.23\.1/u);
  assert.match(continuousIntegration, /cargo clippy --all-targets --locked -- -D warnings/u);
  assert.match(continuousIntegration, /node scripts\/release-contract\.mjs audit-policy/u);
  assert.match(continuousIntegration, /cron: "15 6 \* \* 1"/u);
  assert.match(pages, /node-version: 22\.23\.1/u);
  assert.match(pages, /include-hidden-files: true/u);
});
