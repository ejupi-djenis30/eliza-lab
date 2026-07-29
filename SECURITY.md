# Safety and privacy model

## Supported versions

Safety and security fixes target the current default branch. Earlier Git revisions document the
legacy prototype and are not maintained releases.

ELIZA Lab is an educational machine-learning and dialogue-system demonstration. It is not a
psychologist, therapist, medical device, crisis service, or source of diagnosis.

## Data handling

- The dialogue engine stores only a saturating numeric turn counter; model weights are read-only.
- The command-line application does not write transcripts or analytics.
- Open-set batch inference reads bounded JSONL, emits one result at a time, and does not retain a
  transcript. Bundle verification rejects symlinks, oversized files and SHA-256 mismatches before
  inference.
- Robustness auditing reads JSONL locally with limits of 100,000 rows, 100,000 physical lines,
  18,432 bytes per line and 64 MiB in total. It validates unique IDs in bounded memory but never
  serializes IDs, prompts, transformed text or row-level predictions. Oversized lines abort
  immediately, and schema errors omit submitted field names, values and parser diagnostics.
- The GitHub Pages demo downloads the versioned model as a same-origin static asset, then performs
  feature extraction and inference in the tab without submitting prompts.
- There are no accounts, cookies, databases, API keys, or remote models.
- Prompts are capped at 512 Unicode characters. Batch inference drains oversized lines without
  retaining them; the robustness reader stops at its per-line boundary instead of draining an
  unbounded remainder. The browser keeps at most 40 visible turns.

## Safety boundary

Before learned inference, the engine matches a small, explicit set of safety phrases only to stop the simulation and direct
the visitor toward immediate human help. Phrase matching uses word boundaries, but it still has
both false positives and false negatives. It does not infer intent, assess risk, monitor a person,
or guarantee that urgent language will be recognized. It must not be used as a safety tool.

If someone may be in immediate danger, contact local emergency services or a trusted person.

The public page asks visitors to use harmless fictional prompts. Do not enter real health or
personal information even though the application is local-only.

## Reporting

Use [GitHub private vulnerability reporting](https://github.com/ejupi-djenis30/eliza-lab/security/advisories/new),
or email `info@ejupilabs.com`. The deployed Page publishes the same primary contact at
[`/.well-known/security.txt`](https://ejupi-djenis30.github.io/eliza-lab/.well-known/security.txt).
Do not include real conversation transcripts, credentials, or personal data in a public issue.
