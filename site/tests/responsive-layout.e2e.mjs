import assert from "node:assert/strict";
import { createReadStream } from "node:fs";
import { stat } from "node:fs/promises";
import { createServer } from "node:http";
import { extname, normalize, relative, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";

import { chromium } from "playwright";

const repositoryRoot = resolve(fileURLToPath(new URL("../../", import.meta.url)));
const siteRoot = resolve(repositoryRoot, "site");
const artifactRoot = resolve(repositoryRoot, "artifacts/eliza-open-set-v3");
const selectionReport = resolve(repositoryRoot, "reports/selection-stability-v1.json");
const mountPath = "/eliza-lab";
const widths = [320, 375, 620, 621, 960, 961, 1440];
const navigationTargets = [
  "#experiment",
  "#method",
  "#open-set-v3",
  "#selection-stability",
  "#safety",
];
const measuredSelectors = [
  ".site-header",
  ".brand",
  ".source-link",
  ".hero",
  ".lab-shell",
  ".method-grid",
  ".pipeline-map",
  ".v3-protocol",
  ".report-grid",
  ".selection-audit-grid",
  ".safety",
  "footer",
];

const contentTypes = new Map([
  [".css", "text/css; charset=utf-8"],
  [".html", "text/html; charset=utf-8"],
  [".js", "text/javascript; charset=utf-8"],
  [".json", "application/json; charset=utf-8"],
  [".mjs", "text/javascript; charset=utf-8"],
  [".png", "image/png"],
  [".svg", "image/svg+xml"],
]);

function containedPath(root, requestedPath) {
  const candidate = resolve(root, `.${normalize(requestedPath)}`);
  const withinRoot = relative(root, candidate);
  if (withinRoot === "" || withinRoot === ".." || withinRoot.startsWith(`..${sep}`)) return null;
  return candidate;
}

function resolveRequestPath(requestUrl) {
  let pathname;
  try {
    pathname = decodeURIComponent(new URL(requestUrl ?? "/", "http://127.0.0.1").pathname);
  } catch {
    return null;
  }
  if (pathname !== mountPath && !pathname.startsWith(`${mountPath}/`)) return null;

  const mountedPath = pathname.slice(mountPath.length) || "/";
  const requestedPath = mountedPath.endsWith("/") ? `${mountedPath}index.html` : mountedPath;
  const artifactPrefix = "/data/open-set-v3/";
  if (requestedPath.startsWith(artifactPrefix)) {
    return containedPath(artifactRoot, requestedPath.slice(artifactPrefix.length - 1));
  }
  if (requestedPath === "/data/selection-stability-v1.json") return selectionReport;
  return containedPath(siteRoot, requestedPath);
}

function startStaticServer() {
  const server = createServer(async (request, response) => {
    const path = resolveRequestPath(request.url);
    if (!path) {
      response.writeHead(403).end("Forbidden");
      return;
    }
    try {
      const metadata = await stat(path);
      if (!metadata.isFile()) throw new Error("Not a file");
      response.writeHead(200, {
        "Cache-Control": "no-store",
        "Content-Length": metadata.size,
        "Content-Type": contentTypes.get(extname(path).toLowerCase()) ?? "application/octet-stream",
      });
      createReadStream(path).pipe(response);
    } catch {
      response.writeHead(404).end("Not found");
    }
  });
  return new Promise((resolveServer, reject) => {
    server.once("error", reject);
    server.listen(0, "127.0.0.1", () => resolveServer(server));
  });
}

function closeServer(server) {
  return new Promise((resolveClose, reject) => {
    server.close((error) => (error ? reject(error) : resolveClose()));
  });
}

const server = await startStaticServer();
const address = server.address();
assert(address && typeof address !== "string", "Static test server did not expose a TCP port");
const baseUrl = `http://127.0.0.1:${address.port}${mountPath}/`;
const browser = await chromium.launch({ headless: true });

try {
  const page = await browser.newPage({ viewport: { width: widths[0], height: 900 } });
  const runtimeErrors = [];
  page.on("pageerror", (error) => runtimeErrors.push(`pageerror: ${error.message}`));
  page.on("console", (message) => {
    if (message.type() === "error") runtimeErrors.push(`console: ${message.text()}`);
  });
  const cssResponsePromise = page.waitForResponse((response) =>
    response.url().endsWith(`${mountPath}/styles.css?v=1.5.0`),
  );
  const navigationResponse = await page.goto(baseUrl, { waitUntil: "networkidle" });
  const cssResponse = await cssResponsePromise;

  assert.equal(navigationResponse?.status(), 200, "Site document must load successfully");
  assert.equal(cssResponse.status(), 200, "Site stylesheet must load successfully");
  assert.match(cssResponse.headers()["content-type"] ?? "", /^text\/css\b/);
  assert.equal(await page.locator(".site-header nav a").count(), 5);
  await page.waitForFunction(
    () =>
      document.querySelector(".lab-shell")?.getAttribute("aria-busy") === "false" &&
      document.querySelector("[data-model-status]")?.textContent?.includes("VERIFIED") &&
      document.querySelector("[data-selection-status]")?.dataset.state === "ready",
  );
  assert.match(
    (await page.locator("[data-model-status]").textContent()) ?? "",
    /^ML .+ VERIFIED$/,
    "The responsive gate must exercise a verified, ready model",
  );
  assert.deepEqual(runtimeErrors, [], "The site must initialize without browser runtime errors");
  await page.evaluate(() => {
    document.documentElement.style.scrollBehavior = "auto";
  });

  for (const width of widths) {
    await page.setViewportSize({ width, height: 900 });
    await page.evaluate(
      () => new Promise((resolveFrame) => requestAnimationFrame(() => requestAnimationFrame(resolveFrame))),
    );

    const geometry = await page.evaluate((selectors) => {
      const viewportWidth = document.documentElement.clientWidth;
      const boxes = selectors.map((selector) => {
        const element = document.querySelector(selector);
        if (!element) return { selector, missing: true };
        const bounds = element.getBoundingClientRect();
        return {
          selector,
          missing: false,
          left: bounds.left,
          right: bounds.right,
          width: bounds.width,
          height: bounds.height,
        };
      });
      const header = document.querySelector(".site-header")?.getBoundingClientRect();
      const brand = document.querySelector(".brand")?.getBoundingClientRect();
      const source = document.querySelector(".source-link")?.getBoundingClientRect();
      const navLinks = Array.from(document.querySelectorAll(".site-header nav a"), (element) => {
        const bounds = element.getBoundingClientRect();
        return { width: bounds.width, height: bounds.height };
      });
      const protocolHeights = Array.from(document.querySelectorAll(".v3-protocol article"), (element) =>
        element.getBoundingClientRect().height,
      );
      return {
        viewportWidth,
        documentWidth: document.documentElement.scrollWidth,
        bodyWidth: document.body.scrollWidth,
        boxes,
        headerHeight: header?.height ?? 0,
        brandTop: brand?.top ?? 0,
        brandBottom: brand?.bottom ?? 0,
        sourceTop: source?.top ?? 0,
        sourceBottom: source?.bottom ?? 0,
        navLinks,
        protocolHeights,
      };
    }, measuredSelectors);

    assert.equal(geometry.documentWidth, geometry.viewportWidth, `${width}px: document overflow`);
    assert.equal(geometry.bodyWidth, geometry.viewportWidth, `${width}px: body overflow`);
    for (const box of geometry.boxes) {
      assert.equal(box.missing, false, `${width}px: missing ${box.selector}`);
      assert(box.width > 0 && box.height > 0, `${width}px: ${box.selector} has no geometry`);
      assert(box.left >= -1, `${width}px: ${box.selector} crosses the left edge`);
      assert(box.right <= geometry.viewportWidth + 1, `${width}px: ${box.selector} crosses the right edge`);
    }

    if (width <= 960) {
      assert.equal(geometry.navLinks.length, 5, `${width}px: all navigation targets must remain visible`);
      for (const [index, link] of geometry.navLinks.entries()) {
        assert(link.width > 0, `${width}px: navigation target ${index + 1} has no width`);
        assert(link.height >= 44, `${width}px: navigation target ${index + 1} is under 44px`);
      }
      assert(
        Math.max(geometry.brandTop, geometry.sourceTop) <=
          Math.min(geometry.brandBottom, geometry.sourceBottom),
        `${width}px: source must share the first header row with the brand`,
      );

      for (const target of navigationTargets) {
        const anchorState = await page.evaluate((selector) => {
          const destination = document.querySelector(selector);
          destination?.scrollIntoView({ block: "start" });
          const headerBounds = document.querySelector(".site-header")?.getBoundingClientRect();
          const targetBounds = destination?.getBoundingClientRect();
          return {
            headerBottom: headerBounds?.bottom ?? 0,
            targetTop: targetBounds?.top ?? -1,
          };
        }, target);
        assert(
          anchorState.targetTop >= anchorState.headerBottom - 1,
          `${width}px: ${target} begins under the sticky header (${anchorState.targetTop} < ${anchorState.headerBottom})`,
        );
      }
    }

    if (width <= 620) {
      assert.equal(geometry.protocolHeights.length, 4, `${width}px: protocol must contain four cards`);
      assert(
        Math.max(...geometry.protocolHeights) - Math.min(...geometry.protocolHeights) <= 1,
        `${width}px: protocol cards must share a common height`,
      );
    }
  }

  console.log(`Responsive site validation passed at ${widths.length} viewport widths.`);
} finally {
  await browser.close();
  await closeServer(server);
}
