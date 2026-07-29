import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const repositoryRoot = new URL("../../", import.meta.url);

test("public project surfaces use collective attribution", async () => {
  const [license, readme, site] = await Promise.all([
    readFile(new URL("LICENSE", repositoryRoot), "utf8"),
    readFile(new URL("README.md", repositoryRoot), "utf8"),
    readFile(new URL("site/index.html", repositoryRoot), "utf8"),
  ]);

  for (const [label, content] of [
    ["LICENSE", license],
    ["README.md", readme],
    ["site/index.html", site],
  ]) {
    assert.doesNotMatch(content, /Djenis\s+Ejupi/iu, `${label} must not expose a personal byline`);
  }

  assert.match(
    license,
    /^Copyright \(c\) 2026 Ejupi Labs and ELIZA Lab contributors$/mu,
    "The license must identify the collective copyright holders.",
  );
  assert.match(readme, /Ejupi Labs and the project contributors rebuilt it/u);
  assert.match(
    readme,
    /v1\.5\.0[\s\S]*--repo ejupi-djenis30\/PsychologistRustBot[\s\S]*after the rename[\s\S]*--repo ejupi-djenis30\/eliza-lab/u,
    "The README must distinguish immutable pre-rename provenance from future releases.",
  );
  assert.match(site, />ELIZA Lab contributors ↗<\/a>/u);
  assert.doesNotMatch(
    site,
    /href="https:\/\/github\.com\/ejupi-djenis30">/u,
    "The retired personal-profile footer link must not return.",
  );
});
