import { cp, copyFile, mkdir, rm } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const repositoryRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const siteRoot = path.join(repositoryRoot, "site");
const outputRoot = path.join(repositoryRoot, "pages-dist");

if (
  path.dirname(outputRoot) !== repositoryRoot
  || path.basename(outputRoot) !== "pages-dist"
) {
  throw new Error("The staged site must remain in the repository-local pages-dist directory");
}

const runtimeFiles = [
  "404.html",
  "app.js",
  "engine.mjs",
  "index.html",
  "open-set-engine.mjs",
  "robots.txt",
  "selection-audit.mjs",
  "sitemap.xml",
  "styles.css",
];

await rm(outputRoot, { recursive: true, force: true });
await mkdir(path.join(outputRoot, "data"), { recursive: true });

await Promise.all(
  runtimeFiles.map((file) => copyFile(path.join(siteRoot, file), path.join(outputRoot, file))),
);
await cp(path.join(siteRoot, "assets"), path.join(outputRoot, "assets"), { recursive: true });
await cp(
  path.join(repositoryRoot, "artifacts", "eliza-open-set-v3"),
  path.join(outputRoot, "data", "open-set-v3"),
  { recursive: true },
);
await copyFile(
  path.join(repositoryRoot, "reports", "selection-stability-v1.json"),
  path.join(outputRoot, "data", "selection-stability-v1.json"),
);

console.log(`Staged ${runtimeFiles.length} reviewed runtime files and the verified v3 evidence.`);
