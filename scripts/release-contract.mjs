import { createHash } from "node:crypto";
import { spawnSync } from "node:child_process";
import {
  appendFileSync,
  chmodSync,
  closeSync,
  existsSync,
  fstatSync,
  lstatSync,
  mkdtempSync,
  mkdirSync,
  openSync,
  readFileSync,
  readdirSync,
  renameSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { isDeepStrictEqual } from "node:util";
import { deflateRawSync, gunzipSync, gzipSync, inflateRawSync } from "node:zlib";

const scriptDirectory = path.dirname(fileURLToPath(import.meta.url));
const repositoryRoot = path.resolve(scriptDirectory, "..");
const defaultManifestPath = path.join(repositoryRoot, "Cargo.toml");
const defaultLockPath = path.join(repositoryRoot, "Cargo.lock");
const defaultAuditPolicyPath = path.join(repositoryRoot, ".github", "rustsec-audit-policy.json");
const defaultOpenSetBundlePath = path.join(repositoryRoot, "artifacts", "eliza-open-set-v3");
const defaultLegacyModelPath = path.join(repositoryRoot, "models", "eliza-intent-v1.json");
const packageName = "eliza-lab";

export const SUPPORTED_TARGETS = Object.freeze({
  "x86_64-unknown-linux-gnu": Object.freeze({ os: "linux", architecture: "x86_64", binary: packageName, archive: "tar.gz" }),
  "x86_64-pc-windows-msvc": Object.freeze({ os: "windows", architecture: "x86_64", binary: `${packageName}.exe`, archive: "zip" }),
  "x86_64-apple-darwin": Object.freeze({ os: "macos", architecture: "x86_64", binary: packageName, archive: "tar.gz" }),
  "aarch64-apple-darwin": Object.freeze({ os: "macos", architecture: "aarch64", binary: packageName, archive: "tar.gz" }),
});

const stableSemverPattern = /^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$/;
const gitCommitPattern = /^[0-9a-f]{40}$/u;

function invariant(condition, message) {
  if (!condition) {
    throw new Error(message);
  }
}

export function parseCargoPackage(manifest) {
  let section = "";
  const packageFields = {};

  for (const rawLine of manifest.split(/\r?\n/u)) {
    const line = rawLine.trim();
    const sectionMatch = line.match(/^\[([^\]]+)\](?:\s*#.*)?$/u);
    if (sectionMatch) {
      section = sectionMatch[1];
      continue;
    }
    if (section !== "package") {
      continue;
    }
    const fieldMatch = line.match(/^([A-Za-z0-9_-]+)\s*=\s*"([^"]*)"(?:\s*#.*)?$/u);
    if (fieldMatch) {
      packageFields[fieldMatch[1]] = fieldMatch[2];
    }
  }

  invariant(packageFields.name, "Cargo.toml is missing [package].name");
  invariant(packageFields.version, "Cargo.toml is missing [package].version");
  invariant(
    stableSemverPattern.test(packageFields.version),
    `Cargo package version must be stable SemVer without prerelease or build metadata: ${packageFields.version}`,
  );
  return Object.freeze({
    name: packageFields.name,
    version: packageFields.version,
    license: packageFields.license || null,
  });
}

export function readCargoPackage(manifestPath = defaultManifestPath) {
  return parseCargoPackage(readFileSync(manifestPath, "utf8"));
}

export function expectedTag(version) {
  invariant(stableSemverPattern.test(version), `Cannot create a release tag from non-stable SemVer: ${version}`);
  return `v${version}`;
}

export function verifyVersionTag(version, tag = "") {
  const expected = expectedTag(version);
  if (tag === "") {
    return expected;
  }
  invariant(tag === expected, `Release tag ${JSON.stringify(tag)} does not match Cargo.toml version ${version}; expected ${expected}`);
  return expected;
}

function stripHtmlComments(line, state) {
  let visible = "";
  let cursor = 0;
  while (cursor < line.length) {
    if (state.inComment) {
      const end = line.indexOf("-->", cursor);
      if (end === -1) {
        return visible;
      }
      state.inComment = false;
      cursor = end + 3;
      continue;
    }
    const start = line.indexOf("<!--", cursor);
    if (start === -1) {
      return visible + line.slice(cursor);
    }
    visible += line.slice(cursor, start);
    state.inComment = true;
    cursor = start + 4;
  }
  return visible;
}

function visibleMarkdownLines(markdown) {
  const lines = [];
  const commentState = { inComment: false };
  let fence = null;
  for (const rawLine of markdown.split(/\r?\n/u)) {
    if (fence) {
      const closing = rawLine.match(/^\s{0,3}(`{3,}|~{3,})[\t ]*$/u)?.[1];
      if (closing && closing[0] === fence.character && closing.length >= fence.length) {
        fence = null;
      }
      lines.push("");
      continue;
    }

    const visible = stripHtmlComments(rawLine, commentState);
    const opening = visible.match(/^\s{0,3}(`{3,}|~{3,})/u)?.[1];
    if (opening) {
      fence = { character: opening[0], length: opening.length };
      lines.push("");
      continue;
    }
    lines.push(visible);
  }
  invariant(!commentState.inComment, "CHANGELOG.md contains an unterminated HTML comment");
  invariant(!fence, "CHANGELOG.md contains an unterminated fenced code block");
  return lines;
}

export function parseChangelogRelease(changelog, version, { requireUnreleasedEmpty = false } = {}) {
  invariant(stableSemverPattern.test(version), `Cannot parse a changelog for non-stable SemVer: ${version}`);
  const sections = [];
  let currentSection = null;
  for (const line of visibleMarkdownLines(changelog)) {
    const heading = line.match(/^##[\t ]+(.+?)[\t ]*$/u)?.[1];
    if (heading) {
      if (currentSection) {
        sections.push(currentSection);
      }
      currentSection = { heading, lines: [] };
    } else if (currentSection) {
      currentSection.lines.push(line);
    }
  }
  if (currentSection) {
    sections.push(currentSection);
  }

  const unreleasedSections = sections.filter((section) => section.heading === "Unreleased");
  invariant(unreleasedSections.length === 1, "CHANGELOG.md must contain exactly one visible ## Unreleased section");
  if (requireUnreleasedEmpty) {
    invariant(
      unreleasedSections[0].lines.every((line) => line.trim() === ""),
      `CHANGELOG.md still contains unreleased changes for ${version}`,
    );
  }

  const candidates = sections.filter((section) => section.heading === version || section.heading.startsWith(`${version} `));
  invariant(candidates.length === 1, `CHANGELOG.md must contain exactly one visible, dated section for ${version}`);
  const headingMatch = candidates[0].heading.match(/^(\d+\.\d+\.\d+)[\t ]+(?:—|-)[\t ]+(\d{4}-\d{2}-\d{2})$/u);
  invariant(headingMatch && headingMatch[1] === version, `CHANGELOG.md section ${version} must include an ISO date`);
  const releaseDate = headingMatch[2];
  const parsedDate = new Date(`${releaseDate}T00:00:00Z`);
  invariant(
    !Number.isNaN(parsedDate.getTime()) && parsedDate.toISOString().slice(0, 10) === releaseDate,
    `CHANGELOG.md section ${version} has an invalid date`,
  );
  const noteLines = candidates[0].lines.filter((line) => line.trim() !== "");
  invariant(noteLines.some((line) => /^\s*[-*+][\t ]+\S/u.test(line)), `CHANGELOG.md section ${version} has no release notes`);
  return Object.freeze({ version, releaseDate, noteLines: Object.freeze([...noteLines]) });
}

export function artifactName(version, target) {
  const targetDetails = SUPPORTED_TARGETS[target];
  invariant(targetDetails, `Unsupported release target: ${target}`);
  return `${packageName}-v${version}-${targetDetails.os}-${targetDetails.architecture}.${targetDetails.archive}`;
}

function sha256Buffer(bytes) {
  return createHash("sha256").update(bytes).digest("hex");
}

function sameFileIdentity(left, right) {
  return left.dev === right.dev && left.ino === right.ino;
}

function sameFileSnapshot(left, right) {
  return sameFileIdentity(left, right)
    && left.mode === right.mode
    && left.size === right.size
    && left.mtimeNs === right.mtimeNs
    && left.ctimeNs === right.ctimeNs;
}

export function readRegularFileSnapshot(filePath, label = path.basename(filePath)) {
  const resolvedPath = path.resolve(filePath);
  const descriptor = openSync(resolvedPath, "r");
  try {
    const opened = fstatSync(descriptor, { bigint: true });
    invariant(opened.isFile(), `${label} is not a regular file: ${resolvedPath}`);
    invariant(opened.size > 0n, `${label} is empty: ${resolvedPath}`);

    const linked = lstatSync(resolvedPath, { bigint: true });
    invariant(linked.isFile() && !linked.isSymbolicLink(), `${label} must not be a symbolic link: ${resolvedPath}`);
    invariant(sameFileIdentity(opened, linked), `${label} changed while it was opened: ${resolvedPath}`);

    const bytes = readFileSync(descriptor);
    const afterRead = fstatSync(descriptor, { bigint: true });
    const currentLink = lstatSync(resolvedPath, { bigint: true });
    invariant(sameFileSnapshot(opened, afterRead), `${label} changed while it was read: ${resolvedPath}`);
    invariant(currentLink.isFile() && !currentLink.isSymbolicLink(), `${label} changed into a symbolic link: ${resolvedPath}`);
    invariant(sameFileIdentity(opened, currentLink), `${label} was replaced while it was read: ${resolvedPath}`);
    invariant(BigInt(bytes.length) === opened.size, `${label} size changed while it was read: ${resolvedPath}`);
    return bytes;
  } finally {
    closeSync(descriptor);
  }
}

function writeJson(filePath, value) {
  mkdirSync(path.dirname(filePath), { recursive: true });
  const temporaryPath = `${filePath}.${process.pid}.tmp`;
  writeFileSync(temporaryPath, `${JSON.stringify(value, null, 2)}\n`, "utf8");
  renameSync(temporaryPath, filePath);
}

function ensureEmptyDirectory(directoryPath) {
  mkdirSync(directoryPath, { recursive: true });
  const stats = lstatSync(directoryPath);
  invariant(stats.isDirectory() && !stats.isSymbolicLink(), `Output path must be a regular directory: ${directoryPath}`);
  invariant(readdirSync(directoryPath).length === 0, `Output directory must be empty: ${directoryPath}`);
}

export function buildReleaseContract(manifestPath = defaultManifestPath, tag = "", sourceCommit = "") {
  const cargoPackage = readCargoPackage(manifestPath);
  invariant(cargoPackage.name === packageName, `Expected Cargo package ${packageName}, found ${cargoPackage.name}`);
  invariant(sourceCommit === "" || gitCommitPattern.test(sourceCommit), "Release source commit must be a lowercase 40-character Git SHA");
  const expected = verifyVersionTag(cargoPackage.version, tag);
  const changelogPath = path.join(path.dirname(path.resolve(manifestPath)), "CHANGELOG.md");
  const changelogBytes = readRegularFileSnapshot(changelogPath, "CHANGELOG.md");
  const changelogRelease = parseChangelogRelease(changelogBytes.toString("utf8"), cargoPackage.version, {
    requireUnreleasedEmpty: tag !== "",
  });
  const targets = Object.keys(SUPPORTED_TARGETS);

  return Object.freeze({
    schemaVersion: 2,
    package: packageName,
    version: cargoPackage.version,
    releaseDate: changelogRelease.releaseDate,
    changelogSha256: sha256Buffer(changelogBytes),
    expectedTag: expected,
    validatedTag: tag || null,
    sourceCommit: sourceCommit || null,
    targets,
    artifacts: targets.map((target) => artifactName(cargoPackage.version, target)),
  });
}

function writeTarString(header, offset, length, value) {
  const bytes = Buffer.from(value, "utf8");
  invariant(bytes.length <= length, `Tar header value is too long: ${value}`);
  bytes.copy(header, offset);
}

function writeTarOctal(header, offset, length, value) {
  const octal = value.toString(8).padStart(length - 1, "0");
  invariant(octal.length === length - 1, `Tar numeric value is too large: ${value}`);
  header.write(octal, offset, length - 1, "ascii");
  header[offset + length - 1] = 0;
}

function createTarGzArchive(bytes, binaryName, destination) {
  const header = Buffer.alloc(512);
  writeTarString(header, 0, 100, binaryName);
  writeTarOctal(header, 100, 8, 0o755);
  writeTarOctal(header, 108, 8, 0);
  writeTarOctal(header, 116, 8, 0);
  writeTarOctal(header, 124, 12, bytes.length);
  writeTarOctal(header, 136, 12, 0);
  header.fill(0x20, 148, 156);
  header[156] = "0".charCodeAt(0);
  writeTarString(header, 257, 6, "ustar");
  writeTarString(header, 263, 2, "00");
  writeTarString(header, 265, 32, "root");
  writeTarString(header, 297, 32, "root");
  const checksum = header.reduce((sum, value) => sum + value, 0);
  header.write(checksum.toString(8).padStart(6, "0"), 148, 6, "ascii");
  header[154] = 0;
  header[155] = 0x20;
  const padding = Buffer.alloc((512 - (bytes.length % 512)) % 512);
  const tar = Buffer.concat([header, bytes, padding, Buffer.alloc(1_024)]);
  const archive = gzipSync(tar, { level: 9, mtime: 0 });
  archive[9] = 3;
  writeFileSync(destination, archive);
}

function crc32(bytes) {
  let crc = 0xffffffff;
  for (const byte of bytes) {
    crc ^= byte;
    for (let bit = 0; bit < 8; bit += 1) {
      crc = (crc >>> 1) ^ (0xedb88320 & -(crc & 1));
    }
  }
  return (crc ^ 0xffffffff) >>> 0;
}

function createZipArchive(bytes, binaryName, destination) {
  const name = Buffer.from(binaryName, "utf8");
  const compressed = deflateRawSync(bytes, { level: 9 });
  const digest = crc32(bytes);
  const flags = 0x0800;
  const method = 8;
  const dosDate = (1 << 5) | 1;

  const localHeader = Buffer.alloc(30);
  localHeader.writeUInt32LE(0x04034b50, 0);
  localHeader.writeUInt16LE(20, 4);
  localHeader.writeUInt16LE(flags, 6);
  localHeader.writeUInt16LE(method, 8);
  localHeader.writeUInt16LE(0, 10);
  localHeader.writeUInt16LE(dosDate, 12);
  localHeader.writeUInt32LE(digest, 14);
  localHeader.writeUInt32LE(compressed.length, 18);
  localHeader.writeUInt32LE(bytes.length, 22);
  localHeader.writeUInt16LE(name.length, 26);
  localHeader.writeUInt16LE(0, 28);
  const localRecord = Buffer.concat([localHeader, name, compressed]);

  const centralHeader = Buffer.alloc(46);
  centralHeader.writeUInt32LE(0x02014b50, 0);
  centralHeader.writeUInt16LE((3 << 8) | 20, 4);
  centralHeader.writeUInt16LE(20, 6);
  centralHeader.writeUInt16LE(flags, 8);
  centralHeader.writeUInt16LE(method, 10);
  centralHeader.writeUInt16LE(0, 12);
  centralHeader.writeUInt16LE(dosDate, 14);
  centralHeader.writeUInt32LE(digest, 16);
  centralHeader.writeUInt32LE(compressed.length, 20);
  centralHeader.writeUInt32LE(bytes.length, 24);
  centralHeader.writeUInt16LE(name.length, 28);
  centralHeader.writeUInt16LE(0, 30);
  centralHeader.writeUInt16LE(0, 32);
  centralHeader.writeUInt16LE(0, 34);
  centralHeader.writeUInt16LE(0, 36);
  centralHeader.writeUInt32LE((0o100755 << 16) >>> 0, 38);
  centralHeader.writeUInt32LE(0, 42);
  const centralRecord = Buffer.concat([centralHeader, name]);

  const end = Buffer.alloc(22);
  end.writeUInt32LE(0x06054b50, 0);
  end.writeUInt16LE(0, 4);
  end.writeUInt16LE(0, 6);
  end.writeUInt16LE(1, 8);
  end.writeUInt16LE(1, 10);
  end.writeUInt32LE(centralRecord.length, 12);
  end.writeUInt32LE(localRecord.length, 16);
  end.writeUInt16LE(0, 20);
  writeFileSync(destination, Buffer.concat([localRecord, centralRecord, end]));
}

export function packageBinary({
  target,
  source,
  outputDirectory,
  commit,
  manifestPath = defaultManifestPath,
}) {
  invariant(gitCommitPattern.test(commit), "The source commit must be a lowercase 40-character Git SHA");
  const contract = buildReleaseContract(manifestPath);
  invariant(contract.targets.includes(target), `Target is not part of the release contract: ${target}`);
  const sourcePath = path.resolve(source);
  const sourceBytes = readRegularFileSnapshot(sourcePath, "Built binary");

  const destinationDirectory = path.resolve(outputDirectory);
  ensureEmptyDirectory(destinationDirectory);
  const artifact = artifactName(contract.version, target);
  const destination = path.join(destinationDirectory, artifact);
  const targetDetails = SUPPORTED_TARGETS[target];
  if (targetDetails.archive === "zip") {
    createZipArchive(sourceBytes, targetDetails.binary, destination);
  } else {
    createTarGzArchive(sourceBytes, targetDetails.binary, destination);
  }

  verifyArchive(destination, target, sourceBytes);

  const digest = sha256Buffer(readRegularFileSnapshot(destination, `Packaged archive ${artifact}`));
  const checksumName = `${artifact}.sha256`;
  const manifestName = `${artifact}.json`;
  writeFileSync(path.join(destinationDirectory, checksumName), `${digest}  ${artifact}\n`, "utf8");
  writeJson(path.join(destinationDirectory, manifestName), {
    schemaVersion: 1,
    package: packageName,
    version: contract.version,
    tag: contract.expectedTag,
    target,
    os: SUPPORTED_TARGETS[target].os,
    architecture: SUPPORTED_TARGETS[target].architecture,
    archiveFormat: SUPPORTED_TARGETS[target].archive,
    packagedBinary: SUPPORTED_TARGETS[target].binary,
    artifact,
    sha256: digest,
    sourceCommit: commit,
  });

  return Object.freeze({ artifact, checksum: checksumName, manifest: manifestName, sha256: digest });
}

function readNullTerminatedString(bytes) {
  const terminator = bytes.indexOf(0);
  return bytes.subarray(0, terminator === -1 ? bytes.length : terminator).toString("utf8");
}

function readTarOctal(bytes, label) {
  const text = readNullTerminatedString(bytes).trim();
  invariant(/^[0-7]+$/u.test(text), `Tar ${label} is not valid octal`);
  return Number.parseInt(text, 8);
}

function readTarGzBinary(archiveBytes) {
  const tar = gunzipSync(archiveBytes);
  invariant(tar.length >= 1_536, "Tar archive is too short");
  const header = tar.subarray(0, 512);
  invariant(readNullTerminatedString(header.subarray(257, 263)) === "ustar", "Tar archive is not USTAR");
  const storedChecksum = readTarOctal(header.subarray(148, 156), "checksum");
  const checksumHeader = Buffer.from(header);
  checksumHeader.fill(0x20, 148, 156);
  invariant(checksumHeader.reduce((sum, value) => sum + value, 0) === storedChecksum, "Tar header checksum does not match");
  invariant(header[156] === "0".charCodeAt(0) || header[156] === 0, "Tar archive entry is not a regular file");
  const name = readNullTerminatedString(header.subarray(0, 100));
  const mode = readTarOctal(header.subarray(100, 108), "mode");
  const size = readTarOctal(header.subarray(124, 136), "size");
  const dataEnd = 512 + size;
  invariant(dataEnd <= tar.length, "Tar archive entry exceeds the archive length");
  const paddedEnd = 512 + Math.ceil(size / 512) * 512;
  invariant(tar.length === paddedEnd + 1_024, "Tar archive must contain exactly one file and two end blocks");
  invariant(tar.subarray(dataEnd).every((value) => value === 0), "Tar archive contains non-zero trailing data");
  return Object.freeze({ name, mode, bytes: Buffer.from(tar.subarray(512, dataEnd)) });
}

function readZipBinary(archiveBytes) {
  invariant(archiveBytes.length >= 98, "ZIP archive is too short");
  invariant(archiveBytes.readUInt32LE(0) === 0x04034b50, "ZIP local header signature is invalid");
  const flags = archiveBytes.readUInt16LE(6);
  const method = archiveBytes.readUInt16LE(8);
  const storedCrc = archiveBytes.readUInt32LE(14);
  const compressedSize = archiveBytes.readUInt32LE(18);
  const uncompressedSize = archiveBytes.readUInt32LE(22);
  const localNameLength = archiveBytes.readUInt16LE(26);
  const localExtraLength = archiveBytes.readUInt16LE(28);
  invariant((flags & 0x0008) === 0, "ZIP data descriptors are not allowed");
  invariant(method === 8, "ZIP archive must use Deflate compression");
  const localNameStart = 30;
  const localNameEnd = localNameStart + localNameLength;
  const compressedStart = localNameEnd + localExtraLength;
  const compressedEnd = compressedStart + compressedSize;
  invariant(compressedEnd + 46 + 22 <= archiveBytes.length, "ZIP entry exceeds the archive length");
  const name = archiveBytes.subarray(localNameStart, localNameEnd).toString("utf8");
  const bytes = inflateRawSync(archiveBytes.subarray(compressedStart, compressedEnd));
  invariant(bytes.length === uncompressedSize, "ZIP uncompressed size does not match");
  invariant(crc32(bytes) === storedCrc, "ZIP CRC-32 does not match");

  const centralOffset = compressedEnd;
  invariant(archiveBytes.readUInt32LE(centralOffset) === 0x02014b50, "ZIP central directory signature is invalid");
  const centralNameLength = archiveBytes.readUInt16LE(centralOffset + 28);
  const centralExtraLength = archiveBytes.readUInt16LE(centralOffset + 30);
  const centralCommentLength = archiveBytes.readUInt16LE(centralOffset + 32);
  const centralNameStart = centralOffset + 46;
  const centralEnd = centralNameStart + centralNameLength + centralExtraLength + centralCommentLength;
  invariant(archiveBytes.subarray(centralNameStart, centralNameStart + centralNameLength).toString("utf8") === name, "ZIP filename differs between headers");
  invariant(archiveBytes.readUInt32LE(centralOffset + 16) === storedCrc, "ZIP CRC differs between headers");
  invariant(archiveBytes.readUInt32LE(centralOffset + 20) === compressedSize, "ZIP compressed size differs between headers");
  invariant(archiveBytes.readUInt32LE(centralOffset + 24) === uncompressedSize, "ZIP size differs between headers");
  invariant(archiveBytes.readUInt32LE(centralOffset + 42) === 0, "ZIP entry has an invalid local-header offset");
  const mode = archiveBytes.readUInt32LE(centralOffset + 38) >>> 16;

  invariant(archiveBytes.readUInt32LE(centralEnd) === 0x06054b50, "ZIP end-of-directory signature is invalid");
  invariant(archiveBytes.readUInt16LE(centralEnd + 8) === 1 && archiveBytes.readUInt16LE(centralEnd + 10) === 1, "ZIP archive must contain exactly one entry");
  invariant(archiveBytes.readUInt32LE(centralEnd + 12) === centralEnd - centralOffset, "ZIP central directory size is invalid");
  invariant(archiveBytes.readUInt32LE(centralEnd + 16) === centralOffset, "ZIP central directory offset is invalid");
  const commentLength = archiveBytes.readUInt16LE(centralEnd + 20);
  invariant(centralEnd + 22 + commentLength === archiveBytes.length, "ZIP archive has unexpected trailing data");
  return Object.freeze({ name, mode, bytes });
}

function readArchiveBinaryBytes(archiveBytes, target, archiveLabel) {
  const targetDetails = SUPPORTED_TARGETS[target];
  invariant(targetDetails, `Unsupported release target: ${target}`);
  const entry = targetDetails.archive === "zip" ? readZipBinary(archiveBytes) : readTarGzBinary(archiveBytes);
  invariant(entry.name === targetDetails.binary, `Archive ${archiveLabel} must contain only ${targetDetails.binary}`);
  invariant((entry.mode & 0o111) !== 0, `Archive ${archiveLabel} did not preserve an executable binary mode`);
  return entry.bytes;
}

function readArchiveBinary(archivePath, target) {
  const resolvedPath = path.resolve(archivePath);
  const archiveBytes = readRegularFileSnapshot(resolvedPath, `Archive ${path.basename(resolvedPath)}`);
  return readArchiveBinaryBytes(archiveBytes, target, path.basename(resolvedPath));
}

function verifyArchiveBytes(archiveBytes, target, archiveLabel, expectedBytes) {
  const archivedBytes = readArchiveBinaryBytes(archiveBytes, target, archiveLabel);
  if (expectedBytes) {
    invariant(archivedBytes.equals(expectedBytes), `Archive ${archiveLabel} changed the built binary bytes`);
  }
}

function verifyArchive(archivePath, target, expectedBytes) {
  const resolvedPath = path.resolve(archivePath);
  const archiveBytes = readRegularFileSnapshot(resolvedPath, `Archive ${path.basename(resolvedPath)}`);
  verifyArchiveBytes(archiveBytes, target, path.basename(resolvedPath), expectedBytes);
}

function spdxIdentifier(packageId, name) {
  const safeName = name.replace(/[^A-Za-z0-9.-]+/gu, "-");
  const suffix = createHash("sha256").update(packageId).digest("hex").slice(0, 12);
  return `SPDXRef-Package-${safeName}-${suffix}`;
}

function spdxCreationTime() {
  const epoch = process.env.SOURCE_DATE_EPOCH;
  if (epoch !== undefined && epoch !== "") {
    invariant(/^\d+$/u.test(epoch), "SOURCE_DATE_EPOCH must contain whole seconds");
    return new Date(Number(epoch) * 1_000).toISOString().replace(".000Z", "Z");
  }
  return new Date().toISOString().replace(/\.\d{3}Z$/u, "Z");
}

export function generateSpdx(metadata, {
  manifestPath = defaultManifestPath,
  lockPath = defaultLockPath,
  lockBytes,
} = {}) {
  const contract = buildReleaseContract(manifestPath);
  invariant(metadata && Array.isArray(metadata.packages), "Cargo metadata must contain a packages array");
  invariant(metadata.resolve && Array.isArray(metadata.resolve.nodes), "Cargo metadata must contain a resolved dependency graph");
  const rootPackage = metadata.packages.find((entry) => entry.id === metadata.resolve.root);
  invariant(rootPackage, "Cargo metadata does not identify the workspace root package");
  invariant(rootPackage.name === packageName, `Cargo metadata root package must be ${packageName}`);
  invariant(rootPackage.version === contract.version, "Cargo metadata version does not match Cargo.toml");

  const identifiers = new Map(metadata.packages.map((entry) => [entry.id, spdxIdentifier(entry.id, entry.name)]));
  const packages = metadata.packages
    .map((entry) => ({
      SPDXID: identifiers.get(entry.id),
      name: entry.name,
      versionInfo: entry.version,
      downloadLocation: entry.source ? `https://crates.io/crates/${encodeURIComponent(entry.name)}/${encodeURIComponent(entry.version)}` : "NOASSERTION",
      filesAnalyzed: false,
      licenseConcluded: "NOASSERTION",
      licenseDeclared: entry.license || "NOASSERTION",
      copyrightText: "NOASSERTION",
      externalRefs: [
        {
          referenceCategory: "PACKAGE-MANAGER",
          referenceType: "purl",
          referenceLocator: `pkg:cargo/${encodeURIComponent(entry.name)}@${encodeURIComponent(entry.version)}`,
        },
      ],
    }))
    .sort((left, right) => left.SPDXID.localeCompare(right.SPDXID));

  const relationships = [
    {
      spdxElementId: "SPDXRef-DOCUMENT",
      relationshipType: "DESCRIBES",
      relatedSpdxElement: identifiers.get(rootPackage.id),
    },
  ];
  for (const node of metadata.resolve.nodes) {
    const sourceIdentifier = identifiers.get(node.id);
    invariant(sourceIdentifier, `Resolved Cargo package is missing metadata: ${node.id}`);
    for (const dependencyId of node.dependencies || []) {
      const dependencyIdentifier = identifiers.get(dependencyId);
      invariant(dependencyIdentifier, `Cargo dependency is missing metadata: ${dependencyId}`);
      relationships.push({
        spdxElementId: sourceIdentifier,
        relationshipType: "DEPENDS_ON",
        relatedSpdxElement: dependencyIdentifier,
      });
    }
  }
  relationships.sort((left, right) => JSON.stringify(left).localeCompare(JSON.stringify(right)));

  const resolvedLockBytes = lockBytes || readRegularFileSnapshot(lockPath, "Cargo.lock");
  const lockDigest = sha256Buffer(resolvedLockBytes);
  return {
    spdxVersion: "SPDX-2.3",
    dataLicense: "CC0-1.0",
    SPDXID: "SPDXRef-DOCUMENT",
    name: `${packageName}-${contract.version}`,
    documentNamespace: `https://github.com/ejupi-djenis30/eliza-lab/releases/download/${contract.expectedTag}/${packageName}-${contract.version}-spdx-${lockDigest.slice(0, 16)}`,
    creationInfo: {
      created: spdxCreationTime(),
      creators: ["Tool: ELIZA-Lab-release-contract"],
    },
    packages,
    relationships,
  };
}

function runBinary(binaryPath, argumentsList, label, timeout = 10_000) {
  const result = spawnSync(binaryPath, argumentsList, {
    encoding: "utf8",
    timeout,
    windowsHide: true,
  });
  invariant(!result.error, `${label} could not be executed: ${result.error?.message}`);
  invariant(result.status === 0, `${label} exited with status ${result.status}: ${result.stderr}`);
  return result;
}

export function smokeBinary(
  binaryPath,
  bundlePath = defaultOpenSetBundlePath,
  legacyModelPath = defaultLegacyModelPath,
) {
  const resolvedBinary = path.resolve(binaryPath);
  invariant(existsSync(resolvedBinary), `Cannot smoke-test missing binary: ${resolvedBinary}`);
  const resolvedBundle = path.resolve(bundlePath);
  const resolvedLegacyModel = path.resolve(legacyModelPath);
  invariant(existsSync(resolvedBundle), `Cannot verify missing V3 bundle: ${resolvedBundle}`);
  invariant(existsSync(resolvedLegacyModel), `Cannot smoke-test missing legacy model: ${resolvedLegacyModel}`);

  const embeddedInference = runBinary(
    resolvedBinary,
    ["infer", "--json", "Hello, I want to make a concrete plan"],
    "Embedded V3 inference smoke test",
    120_000,
  );
  const inference = runBinary(
    resolvedBinary,
    ["infer", "--bundle", resolvedBundle, "--json", "Hello, I want to make a concrete plan"],
    "Primary V3 inference smoke test",
    120_000,
  );
  const traces = [];
  for (const [label, result] of [
    ["Embedded V3 inference", embeddedInference],
    ["External V3 bundle inference", inference],
  ]) {
    try {
      traces.push(JSON.parse(result.stdout));
    } catch (error) {
      throw new Error(`${label} did not return JSON: ${error.message}`);
    }
  }
  for (const trace of traces) {
    invariant(trace?.model?.version === "3.0.0", "V3 inference did not use model 3.0.0");
    invariant(
      trace.boundary === "ml-intent" || trace.boundary === "ml-abstain",
      "V3 inference did not expose the learned-policy boundary",
    );
    invariant(typeof trace.model.accepted === "boolean", "V3 inference omitted its accept/abstain decision");
    invariant(Number.isFinite(trace.model.confidence), "V3 inference omitted policy confidence");
    invariant(Number.isFinite(trace.model.margin), "V3 inference omitted policy probability margin");
    invariant(
      trace.model.probabilities && Object.keys(trace.model.probabilities).length === 7,
      "V3 inference omitted its seven-class probability trace",
    );
    invariant(Array.isArray(trace.model.top_features), "V3 inference omitted feature evidence");
  }
  invariant(
    isDeepStrictEqual(traces[0], traces[1]),
    "Embedded and external V3 bundle inference produced different traces",
  );

  const verified = runBinary(
    resolvedBinary,
    ["bundle", "verify", "--bundle", resolvedBundle],
    "V3 semantic bundle verification",
    180_000,
  );
  invariant(verified.stdout.includes("semantically verified"), "Built binary did not report semantic bundle verification");
  invariant(verified.stdout.includes("model        3.0.0"), "Built binary verified an unexpected model version");

  const reproduced = runBinary(
    resolvedBinary,
    ["bundle", "reproduce", "--bundle", resolvedBundle],
    "V3 byte-identical bundle reproduction",
    300_000,
  );
  invariant(reproduced.stdout.includes("byte-identical reproduction"), "Built binary did not reproduce the V3 bundle");

  const legacy = runBinary(
    resolvedBinary,
    [
      "infer",
      "--legacy-v1",
      "--model",
      resolvedLegacyModel,
      "--json",
      "Today I feel calm",
    ],
    "Explicit legacy V1 compatibility smoke test",
  );
  let legacyTrace;
  try {
    legacyTrace = JSON.parse(legacy.stdout);
  } catch (error) {
    throw new Error(`Explicit legacy V1 inference did not return JSON: ${error.message}`);
  }
  invariant(legacyTrace?.model?.version === "1.0.0", "Explicit legacy inference did not use model 1.0.0");
}

export function smokeArchive(
  archivePath,
  target,
  bundlePath = defaultOpenSetBundlePath,
  legacyModelPath = defaultLegacyModelPath,
) {
  const targetDetails = SUPPORTED_TARGETS[target];
  invariant(targetDetails, `Unsupported release target: ${target}`);
  const resolvedArchive = path.resolve(archivePath);
  const binaryBytes = readArchiveBinary(resolvedArchive, target);
  const extractionDirectory = mkdtempSync(path.join(tmpdir(), "eliza-lab-smoke-"));
  try {
    const binaryPath = path.join(extractionDirectory, targetDetails.binary);
    writeFileSync(binaryPath, binaryBytes);
    if (targetDetails.os !== "windows") {
      chmodSync(binaryPath, 0o755);
    }
    smokeBinary(binaryPath, bundlePath, legacyModelPath);
  } finally {
    // Windows can retain a short-lived executable image lock after CreateProcess exits. Hosted
    // runners clear RUNNER_TEMP; failing a verified package because that cleanup races the OS is
    // worse than leaving one isolated temporary directory for runner teardown.
    if (process.platform !== "win32") {
      rmSync(extractionDirectory, { recursive: true, force: true, maxRetries: 5, retryDelay: 100 });
    }
  }
}

function readJson(filePath, label) {
  return parseJsonBytes(readFileSync(filePath), label);
}

function parseJsonBytes(bytes, label) {
  try {
    return JSON.parse(bytes.toString("utf8"));
  } catch (error) {
    throw new Error(`${label} is not valid JSON: ${error.message}`);
  }
}

function readExactDirectorySnapshots(directoryPath, expectedNames, label) {
  const before = lstatSync(directoryPath, { bigint: true });
  invariant(before.isDirectory() && !before.isSymbolicLink(), `${label} must be a regular directory: ${directoryPath}`);
  const actualNames = readdirSync(directoryPath).sort();
  invariant(
    isDeepStrictEqual(actualNames, expectedNames),
    `${label} inventory differs from the release contract. Expected ${expectedNames.join(", ")}; found ${actualNames.join(", ")}`,
  );
  const snapshots = new Map();
  for (const name of expectedNames) {
    snapshots.set(name, readRegularFileSnapshot(path.join(directoryPath, name), `${label} file ${name}`));
  }
  const after = lstatSync(directoryPath, { bigint: true });
  invariant(sameFileSnapshot(before, after), `${label} directory changed while it was read: ${directoryPath}`);
  return snapshots;
}

function writeVerifiedSnapshot(destination, bytes) {
  const descriptor = openSync(destination, "wx", 0o644);
  try {
    writeFileSync(descriptor, bytes);
  } finally {
    closeSync(descriptor);
  }
  const written = readRegularFileSnapshot(destination, `Assembled release asset ${path.basename(destination)}`);
  invariant(written.equals(bytes), `Assembled release asset changed while it was written: ${path.basename(destination)}`);
}

function cargoLockPackages(lockContents) {
  const packages = [];
  for (const block of lockContents.split("[[package]]").slice(1)) {
    const name = block.match(/^\s*name\s*=\s*"([^"]+)"/mu)?.[1];
    const version = block.match(/^\s*version\s*=\s*"([^"]+)"/mu)?.[1];
    invariant(name && version, "Cargo.lock contains a package without a name or version");
    packages.push(`${name}@${version}`);
  }
  invariant(packages.length > 0, "Cargo.lock does not contain any packages");
  return packages.sort();
}

function validateAuditPolicy(policy) {
  invariant(
    isDeepStrictEqual(Object.keys(policy).sort(), ["allowStaleDatabase", "database", "deny", "ignoredAdvisories", "schemaVersion", "tool"]),
    "RustSec audit policy contains unexpected or missing fields",
  );
  invariant(policy.schemaVersion === 1, "RustSec audit policy has an unsupported schema version");
  invariant(
    isDeepStrictEqual(Object.keys(policy.tool || {}).sort(), ["name", "rustVersion", "version"]),
    "RustSec audit tool policy contains unexpected fields",
  );
  invariant(
    policy.tool?.name === "cargo-audit" && stableSemverPattern.test(policy.tool?.version || ""),
    "RustSec audit policy must pin a stable cargo-audit version",
  );
  invariant(stableSemverPattern.test(policy.tool?.rustVersion || ""), "RustSec audit policy must pin the scanner Rust version");
  invariant(
    isDeepStrictEqual(Object.keys(policy.database || {}).sort(), ["commit", "commitEpoch", "maximumAgeDays", "url"]),
    "RustSec advisory database policy contains unexpected fields",
  );
  invariant(
    policy.database?.url === "https://github.com/RustSec/advisory-db.git"
      && gitCommitPattern.test(policy.database?.commit || ""),
    "RustSec audit policy must pin the official advisory database to a commit",
  );
  invariant(
    Number.isSafeInteger(policy.database?.commitEpoch) && policy.database.commitEpoch > 0,
    "RustSec audit policy must record the pinned database commit time",
  );
  invariant(
    Number.isSafeInteger(policy.database?.maximumAgeDays)
      && policy.database.maximumAgeDays >= 1
      && policy.database.maximumAgeDays <= 30,
    "RustSec audit policy must define a database freshness window from 1 to 30 days",
  );
  invariant(isDeepStrictEqual(policy.deny, ["warnings"]), "RustSec audit policy must deny all audit warnings");
  invariant(isDeepStrictEqual(policy.ignoredAdvisories, []), "RustSec audit policy must not silently ignore advisories");
  invariant(policy.allowStaleDatabase === false, "RustSec audit policy must reject stale advisory data");
  return policy;
}

function validateAuditDatabaseFreshness(policy, {
  databaseCommit,
  databaseCommitEpoch,
  nowEpochSeconds = Math.floor(Date.now() / 1_000),
}) {
  invariant(databaseCommit === policy.database.commit, "RustSec advisory database commit does not match policy");
  invariant(
    Number.isSafeInteger(databaseCommitEpoch) && databaseCommitEpoch === policy.database.commitEpoch,
    "RustSec advisory database commit time does not match policy",
  );
  invariant(Number.isSafeInteger(nowEpochSeconds) && nowEpochSeconds > 0, "RustSec policy evaluation time is invalid");
  const databaseAgeSeconds = nowEpochSeconds - databaseCommitEpoch;
  invariant(databaseAgeSeconds >= -300, "RustSec advisory database commit time is unexpectedly in the future");
  invariant(
    databaseAgeSeconds <= policy.database.maximumAgeDays * 86_400,
    `RustSec advisory database is older than the allowed ${policy.database.maximumAgeDays} days`,
  );
}

export function verifyAuditDatabasePolicy({
  policyPath = defaultAuditPolicyPath,
  databaseCommit,
  databaseCommitEpoch,
  nowEpochSeconds = Math.floor(Date.now() / 1_000),
} = {}) {
  const policyBytes = readRegularFileSnapshot(policyPath, "RustSec audit policy");
  const policy = validateAuditPolicy(parseJsonBytes(policyBytes, "RustSec audit policy"));
  validateAuditDatabaseFreshness(policy, {
    databaseCommit: databaseCommit ?? policy.database.commit,
    databaseCommitEpoch: databaseCommitEpoch ?? policy.database.commitEpoch,
    nowEpochSeconds,
  });
  return Object.freeze({ ...policy, database: Object.freeze({ ...policy.database }) });
}

function validateAuditReport(report, dependencyCount) {
  invariant(
    Number.isSafeInteger(report?.database?.["advisory-count"]) && report.database["advisory-count"] > 0,
    "RustSec audit report does not identify a populated advisory database",
  );
  invariant(report?.lockfile?.["dependency-count"] === dependencyCount, "RustSec audit report does not cover the complete Cargo.lock");
  invariant(isDeepStrictEqual(report?.settings?.ignore, []), "RustSec audit report unexpectedly ignores advisories");
  invariant(isDeepStrictEqual(report?.settings?.target_arch, []), "RustSec audit report unexpectedly filters target architectures");
  invariant(isDeepStrictEqual(report?.settings?.target_os, []), "RustSec audit report unexpectedly filters target operating systems");
  invariant(report?.settings?.severity === null, "RustSec audit report unexpectedly filters vulnerability severity");
  invariant(
    isDeepStrictEqual(report?.settings?.informational_warnings, ["unmaintained", "unsound", "notice"]),
    "RustSec audit report does not include every informational warning category",
  );
  invariant(
    report?.vulnerabilities?.found === false
      && report.vulnerabilities.count === 0
      && isDeepStrictEqual(report.vulnerabilities.list, []),
    "RustSec audit report contains vulnerable dependencies",
  );
  invariant(
    report?.warnings && Object.keys(report.warnings).length === 0,
    "RustSec audit report contains denied dependency warnings",
  );
  return report;
}

export function buildAuditEvidence({
  report,
  toolVersion,
  databaseCommit,
  databaseCommitEpoch,
  policyPath = defaultAuditPolicyPath,
  lockPath = defaultLockPath,
  nowEpochSeconds = Math.floor(Date.now() / 1_000),
}) {
  const policyBytes = readRegularFileSnapshot(policyPath, "RustSec audit policy");
  const policy = validateAuditPolicy(parseJsonBytes(policyBytes, "RustSec audit policy"));
  validateAuditDatabaseFreshness(policy, {
    databaseCommit,
    databaseCommitEpoch,
    nowEpochSeconds,
  });
  const lockBytes = readRegularFileSnapshot(lockPath, "Cargo.lock");
  const dependencyCount = cargoLockPackages(lockBytes.toString("utf8")).length;
  invariant(toolVersion === policy.tool.version, `cargo-audit ${toolVersion} does not match policy version ${policy.tool.version}`);
  validateAuditReport(report, dependencyCount);
  return Object.freeze({
    schemaVersion: 2,
    tool: Object.freeze({ ...policy.tool }),
    database: Object.freeze({ ...policy.database }),
    deny: Object.freeze([...policy.deny]),
    ignoredAdvisories: Object.freeze([...policy.ignoredAdvisories]),
    allowStaleDatabase: policy.allowStaleDatabase,
    policySha256: sha256Buffer(policyBytes),
    lockfileSha256: sha256Buffer(lockBytes),
    dependencyCount,
    report,
  });
}

function expectedEvidenceFiles(contract) {
  return [
    "Cargo.lock",
    "cargo-metadata.json",
    "cargo-tree.txt",
    "release-contract.json",
    "rustsec-audit-policy.json",
    "rustsec-audit.json",
    `${packageName}-v${contract.version}.spdx.json`,
  ].sort();
}

export function expectedReleaseFileNames(manifestPath = defaultManifestPath) {
  const contract = buildReleaseContract(manifestPath);
  const platformFiles = contract.targets.flatMap((target) => {
    const artifact = artifactName(contract.version, target);
    return [artifact, `${artifact}.sha256`, `${artifact}.json`];
  });
  return [...platformFiles, ...expectedEvidenceFiles(contract), "SHA256SUMS"].sort();
}

function validateEvidence({ evidenceSnapshots, contract, manifestPath }) {
  const evidenceFiles = expectedEvidenceFiles(contract);
  invariant(isDeepStrictEqual([...evidenceSnapshots.keys()].sort(), evidenceFiles), "Supply-chain evidence snapshots are incomplete");

  const releaseContract = parseJsonBytes(evidenceSnapshots.get("release-contract.json"), "release-contract.json");
  invariant(isDeepStrictEqual(releaseContract, contract), "Release contract evidence does not exactly match this source, tag, and commit");

  const sourceLockPath = path.join(path.dirname(path.resolve(manifestPath)), "Cargo.lock");
  const sourceLock = readRegularFileSnapshot(sourceLockPath, "Source Cargo.lock");
  const evidenceLock = evidenceSnapshots.get("Cargo.lock");
  invariant(sourceLock.equals(evidenceLock), "Cargo.lock evidence differs from the checked-out source");

  const metadata = parseJsonBytes(evidenceSnapshots.get("cargo-metadata.json"), "cargo-metadata.json");
  invariant(Array.isArray(metadata.packages), "Cargo metadata evidence is missing packages");
  const metadataPackages = metadata.packages.map((entry) => `${entry.name}@${entry.version}`).sort();
  invariant(
    isDeepStrictEqual(metadataPackages, cargoLockPackages(evidenceLock.toString("utf8"))),
    "Cargo metadata packages do not match Cargo.lock",
  );

  const tree = evidenceSnapshots.get("cargo-tree.txt").toString("utf8");
  invariant(tree.startsWith(`${packageName} v${contract.version}`), "Cargo dependency tree does not begin with the release package and version");
  invariant(!tree.includes("\0"), "Cargo dependency tree contains invalid NUL data");

  const sourcePolicyBytes = readRegularFileSnapshot(defaultAuditPolicyPath, "Source RustSec audit policy");
  const evidencePolicyBytes = evidenceSnapshots.get("rustsec-audit-policy.json");
  invariant(sourcePolicyBytes.equals(evidencePolicyBytes), "RustSec audit policy evidence differs from the checked-out source");
  const auditEvidence = parseJsonBytes(evidenceSnapshots.get("rustsec-audit.json"), "rustsec-audit.json");
  const expectedAuditEvidence = buildAuditEvidence({
    report: auditEvidence.report,
    toolVersion: auditEvidence.tool?.version,
    databaseCommit: auditEvidence.database?.commit,
    databaseCommitEpoch: auditEvidence.database?.commitEpoch,
    policyPath: defaultAuditPolicyPath,
    lockPath: sourceLockPath,
  });
  invariant(isDeepStrictEqual(auditEvidence, expectedAuditEvidence), "RustSec audit evidence does not match the pinned policy and Cargo.lock");

  const sbomName = `${packageName}-v${contract.version}.spdx.json`;
  const sbom = parseJsonBytes(evidenceSnapshots.get(sbomName), sbomName);
  const expectedSbom = generateSpdx(metadata, { manifestPath, lockBytes: evidenceLock });
  const { creationInfo: actualCreation, ...actualSbomBody } = sbom;
  const { creationInfo: expectedCreation, ...expectedSbomBody } = expectedSbom;
  invariant(isDeepStrictEqual(actualSbomBody, expectedSbomBody), "Release SBOM content does not match Cargo metadata and Cargo.lock");
  invariant(
    actualCreation
      && /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$/u.test(actualCreation.created)
      && isDeepStrictEqual(actualCreation.creators, expectedCreation.creators),
    "Release SBOM creation metadata is invalid",
  );

  return evidenceFiles;
}

export function verifyAndAssembleAssets({
  inputDirectory,
  evidenceDirectory,
  outputDirectory,
  expectedCommit,
  validatedTag = "",
  manifestPath = defaultManifestPath,
}) {
  invariant(gitCommitPattern.test(expectedCommit), "Expected commit must be a lowercase 40-character Git SHA");
  const contract = buildReleaseContract(manifestPath, validatedTag, expectedCommit);
  const input = path.resolve(inputDirectory);
  const evidence = path.resolve(evidenceDirectory);
  const output = path.resolve(outputDirectory);

  const expectedBinaryFiles = contract.targets.flatMap((target) => {
    const artifact = artifactName(contract.version, target);
    return [artifact, `${artifact}.sha256`, `${artifact}.json`];
  }).sort();
  const inputSnapshots = readExactDirectorySnapshots(input, expectedBinaryFiles, "Binary artifact");
  const evidenceFiles = expectedEvidenceFiles(contract);
  const evidenceSnapshots = readExactDirectorySnapshots(evidence, evidenceFiles, "Supply-chain evidence");

  const sourceCommits = new Set();
  for (const target of contract.targets) {
    const artifact = artifactName(contract.version, target);
    const artifactBytes = inputSnapshots.get(artifact);
    verifyArchiveBytes(artifactBytes, target, artifact);
    const digest = sha256Buffer(artifactBytes);
    const checksum = inputSnapshots.get(`${artifact}.sha256`).toString("utf8");
    invariant(checksum === `${digest}  ${artifact}\n`, `Checksum file does not match ${artifact}`);
    const artifactManifest = parseJsonBytes(inputSnapshots.get(`${artifact}.json`), `${artifact}.json`);
    const targetDetails = SUPPORTED_TARGETS[target];
    const expectedManifest = {
      schemaVersion: 1,
      package: packageName,
      version: contract.version,
      tag: contract.expectedTag,
      target,
      os: targetDetails.os,
      architecture: targetDetails.architecture,
      archiveFormat: targetDetails.archive,
      packagedBinary: targetDetails.binary,
      artifact,
      sha256: digest,
      sourceCommit: expectedCommit,
    };
    invariant(isDeepStrictEqual(artifactManifest, expectedManifest), `Artifact manifest does not exactly match ${artifact}`);
    sourceCommits.add(artifactManifest.sourceCommit);
  }
  invariant(sourceCommits.size === 1, "Release binaries were not built from one source commit");
  invariant([...sourceCommits][0] === expectedCommit, "Release binaries do not match the expected workflow commit");

  validateEvidence({ evidenceSnapshots, contract, manifestPath });

  ensureEmptyDirectory(output);
  const verifiedSnapshots = new Map();
  for (const file of expectedBinaryFiles) {
    verifiedSnapshots.set(file, inputSnapshots.get(file));
  }
  for (const file of evidenceFiles) {
    verifiedSnapshots.set(file, evidenceSnapshots.get(file));
  }
  const checksummedFiles = [...expectedBinaryFiles, ...evidenceFiles].sort();
  for (const file of checksummedFiles) {
    writeVerifiedSnapshot(path.join(output, file), verifiedSnapshots.get(file));
  }
  const consolidatedChecksums = checksummedFiles.map((file) => `${sha256Buffer(verifiedSnapshots.get(file))}  ${file}`);
  const checksumBytes = Buffer.from(`${consolidatedChecksums.join("\n")}\n`, "utf8");
  writeVerifiedSnapshot(path.join(output, "SHA256SUMS"), checksumBytes);

  const expectedOutputFiles = expectedReleaseFileNames(manifestPath);
  const outputSnapshots = readExactDirectorySnapshots(output, expectedOutputFiles, "Assembled release");
  for (const [file, bytes] of verifiedSnapshots) {
    invariant(outputSnapshots.get(file).equals(bytes), `Assembled release asset differs from its verified snapshot: ${file}`);
  }
  invariant(outputSnapshots.get("SHA256SUMS").equals(checksumBytes), "Assembled SHA256SUMS changed after creation");

  return Object.freeze({
    version: contract.version,
    tag: contract.expectedTag,
    sourceCommit: [...sourceCommits][0],
    files: expectedOutputFiles,
  });
}

function parseOptions(argumentsList) {
  const options = {};
  for (let index = 0; index < argumentsList.length; index += 2) {
    const option = argumentsList[index];
    invariant(option?.startsWith("--"), `Expected an option, received ${JSON.stringify(option)}`);
    invariant(index + 1 < argumentsList.length, `Missing value for ${option}`);
    const key = option.slice(2);
    invariant(!(key in options), `Option was provided more than once: ${option}`);
    options[key] = argumentsList[index + 1];
  }
  return options;
}

function requireOption(options, name) {
  invariant(options[name] !== undefined && options[name] !== "", `Missing required option --${name}`);
  return options[name];
}

function rejectUnknownOptions(options, allowed) {
  for (const option of Object.keys(options)) {
    invariant(allowed.includes(option), `Unknown option --${option}`);
  }
}

function writeGitHubOutputs(values) {
  const outputPath = process.env.GITHUB_OUTPUT;
  if (!outputPath) {
    return;
  }
  for (const [key, value] of Object.entries(values)) {
    invariant(!String(value).includes("\n"), `GitHub output ${key} cannot contain a newline`);
    appendFileSync(outputPath, `${key}=${value}\n`, "utf8");
  }
}

function runCli() {
  const [command, ...rawOptions] = process.argv.slice(2);
  const options = parseOptions(rawOptions);

  if (command === "verify") {
    rejectUnknownOptions(options, ["tag", "output", "expected-commit", "actual-commit"]);
    const expectedCommit = options["expected-commit"] || "";
    const actualCommit = options["actual-commit"] || "";
    invariant(
      (expectedCommit === "" && actualCommit === "") || (expectedCommit !== "" && actualCommit !== ""),
      "--expected-commit and --actual-commit must be provided together",
    );
    if (expectedCommit) {
      invariant(gitCommitPattern.test(expectedCommit), "Expected commit must be a lowercase 40-character Git SHA");
      invariant(gitCommitPattern.test(actualCommit), "Actual tag commit must be a lowercase 40-character Git SHA");
      invariant(actualCommit === expectedCommit, `Tag resolves to ${actualCommit}, not expected workflow commit ${expectedCommit}`);
    }
    const contract = buildReleaseContract(defaultManifestPath, options.tag || "", expectedCommit);
    if (options.output) {
      writeJson(path.resolve(options.output), contract);
    }
    writeGitHubOutputs({ version: contract.version, expected_tag: contract.expectedTag });
    console.log(`Release contract valid: ${contract.package} ${contract.version} (${contract.expectedTag})`);
    return;
  }

  if (command === "package") {
    rejectUnknownOptions(options, ["target", "source", "output", "commit"]);
    const packaged = packageBinary({
      target: requireOption(options, "target"),
      source: requireOption(options, "source"),
      outputDirectory: requireOption(options, "output"),
      commit: requireOption(options, "commit"),
    });
    writeGitHubOutputs(packaged);
    console.log(`Packaged ${packaged.artifact} (${packaged.sha256})`);
    return;
  }

  if (command === "sbom") {
    rejectUnknownOptions(options, ["metadata", "output"]);
    const metadataPath = path.resolve(requireOption(options, "metadata"));
    const outputPath = path.resolve(requireOption(options, "output"));
    const metadata = readJson(metadataPath, "Cargo metadata");
    writeJson(outputPath, generateSpdx(metadata));
    console.log(`Wrote SPDX 2.3 SBOM to ${outputPath}`);
    return;
  }

  if (command === "audit-evidence") {
    rejectUnknownOptions(options, ["report", "output", "tool-version", "database-commit", "database-epoch", "policy", "lock"]);
    const reportPath = path.resolve(requireOption(options, "report"));
    const outputPath = path.resolve(requireOption(options, "output"));
    const evidence = buildAuditEvidence({
      report: readJson(reportPath, "cargo-audit report"),
      toolVersion: requireOption(options, "tool-version"),
      databaseCommit: requireOption(options, "database-commit"),
      databaseCommitEpoch: Number(requireOption(options, "database-epoch")),
      policyPath: options.policy ? path.resolve(options.policy) : defaultAuditPolicyPath,
      lockPath: options.lock ? path.resolve(options.lock) : defaultLockPath,
    });
    writeJson(outputPath, evidence);
    console.log(`Wrote pinned RustSec audit evidence to ${outputPath}`);
    return;
  }

  if (command === "audit-policy") {
    rejectUnknownOptions(options, ["database-commit", "database-epoch", "policy"]);
    invariant(
      Boolean(options["database-commit"]) === Boolean(options["database-epoch"]),
      "--database-commit and --database-epoch must be provided together",
    );
    const policy = verifyAuditDatabasePolicy({
      policyPath: options.policy ? path.resolve(options.policy) : defaultAuditPolicyPath,
      databaseCommit: options["database-commit"],
      databaseCommitEpoch: options["database-epoch"] ? Number(options["database-epoch"]) : undefined,
    });
    console.log(
      `RustSec advisory database policy is fresh: ${policy.database.commit} (maximum ${policy.database.maximumAgeDays} days).`,
    );
    return;
  }

  if (command === "smoke") {
    rejectUnknownOptions(options, ["binary", "bundle", "legacy-model"]);
    smokeBinary(
      requireOption(options, "binary"),
      requireOption(options, "bundle"),
      requireOption(options, "legacy-model"),
    );
    console.log("Built CLI V3 verification, reproduction, inference, and explicit V1 compatibility checks passed.");
    return;
  }

  if (command === "smoke-archive") {
    rejectUnknownOptions(options, ["archive", "target", "bundle", "legacy-model"]);
    smokeArchive(
      requireOption(options, "archive"),
      requireOption(options, "target"),
      requireOption(options, "bundle"),
      requireOption(options, "legacy-model"),
    );
    console.log("Packaged CLI V3 verification, reproduction, inference, and explicit V1 compatibility checks passed.");
    return;
  }

  if (command === "verify-assets") {
    rejectUnknownOptions(options, ["input", "evidence", "output", "expected-commit", "tag"]);
    const inventory = verifyAndAssembleAssets({
      inputDirectory: requireOption(options, "input"),
      evidenceDirectory: requireOption(options, "evidence"),
      outputDirectory: requireOption(options, "output"),
      expectedCommit: requireOption(options, "expected-commit"),
      validatedTag: options.tag || "",
    });
    writeGitHubOutputs({ version: inventory.version, expected_tag: inventory.tag });
    console.log(`Verified ${inventory.files.length} release assets from ${inventory.sourceCommit}.`);
    return;
  }

  throw new Error("Usage: release-contract.mjs <verify|package|sbom|audit-policy|audit-evidence|smoke|smoke-archive|verify-assets> [options]");
}

if (process.argv[1] && import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href) {
  try {
    runCli();
  } catch (error) {
    console.error(`release-contract: ${error.message}`);
    process.exitCode = 1;
  }
}
