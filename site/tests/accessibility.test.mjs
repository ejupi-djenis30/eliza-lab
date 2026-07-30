import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const siteRoot = new URL("../", import.meta.url);

test("the skip link precedes the header and targets the focusable main landmark", async () => {
  const html = await readFile(new URL("index.html", siteRoot), "utf8");
  const skipLink = '<a class="skip-link" href="#main-content">Skip to content</a>';

  assert.ok(html.includes(skipLink));
  assert.ok(html.includes('<main id="main-content" tabindex="-1">'));
  assert.ok(html.indexOf(skipLink) < html.indexOf('<header class="site-header">'));
});

test("the skip link exposes a visible keyboard state and a usable target size", async () => {
  const styles = await readFile(new URL("styles.css", siteRoot), "utf8");

  assert.match(styles, /\.skip-link\s*\{[^}]*min-height:\s*2\.75rem;/s);
  assert.match(styles, /\.skip-link:focus-visible\s*\{[^}]*transform:\s*translateY\(0\);/s);
});

test("mobile navigation uses a native disclosure with a labelled landmark", async () => {
  const html = await readFile(new URL("index.html", siteRoot), "utf8");

  assert.match(
    html,
    /<details class="mobile-menu">\s*<summary>Menu<\/summary>\s*<nav aria-label="Mobile navigation">/,
  );
});

test("the prompt limit is not delegated to UTF-16 maxlength", async () => {
  const html = await readFile(new URL("index.html", siteRoot), "utf8");
  const app = await readFile(new URL("app.js", siteRoot), "utf8");

  assert.doesNotMatch(html, /\bmaxlength=/);
  assert.match(app, /for \(const character of value\)/);
  assert.match(app, /count === MAX_INPUT_CHARS/);
});

test("the lab enables controls only after a verified model loads", async () => {
  const html = await readFile(new URL("index.html", siteRoot), "utf8");
  const app = await readFile(new URL("app.js", siteRoot), "utf8");

  assert.match(html, /class="lab-shell" aria-busy="true"/);
  assert.match(html, /role="status" aria-live="polite" data-model-status/);
  assert.ok((html.match(/data-model-gated disabled/gu) ?? []).length >= 5);
  assert.match(html, /name="message"\s+data-model-gated\s+disabled/s);
  assert.match(app, /try\s*\{[\s\S]*setModelReady\(\);[\s\S]*\}\s*catch\s*\{[\s\S]*setModelFailed\(\);/s);
  assert.doesNotMatch(app, /finally\s*\{\s*setModelReady\(\)/s);
  assert.match(app, /activeModel = null;\s*engine = null;/s);
  assert.match(app, /control\.disabled = false/);
  assert.match(app, /control\.disabled = true/);
  assert.match(app, /if \(!activeModel \|\| !engine\) return;/);
});

test("the app versions every transitive local module", async () => {
  const html = await readFile(new URL("index.html", siteRoot), "utf8");
  const app = await readFile(new URL("app.js", siteRoot), "utf8");

  assert.match(html, /src="app\.js\?v=[^"\s]+"/);
  assert.match(app, /from "\.\/engine\.mjs\?v=[^"\s]+"/);
  assert.match(app, /from "\.\/open-set-engine\.mjs\?v=[^"\s]+"/);
  assert.doesNotMatch(app, /from "\.\/ml-engine\.mjs/);
});
