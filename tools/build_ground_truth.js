const fs = require("fs");
const path = require("path");

// Reuse the repo's NVIDIA v1 OCR pipeline (api/solve.js). This keeps behavior
// consistent with the app and avoids adding new OCR dependencies.
const solveHandler = require(path.join("..", "api", "solve.js"));

const ROOT = path.join(__dirname, "..");
const SCREENSHOTS_DIR = path.join(ROOT, "images screenshots");
const OUT_PATH = path.join(ROOT, "ground_truth.json");
const ERR_PATH = path.join(ROOT, "ground_truth.errors.json");

function loadDotEnv(filePath = path.join(ROOT, ".env")) {
  if (!fs.existsSync(filePath)) return;
  const text = fs.readFileSync(filePath, "utf8");
  for (const rawLine of text.split(/\r?\n/)) {
    const line = rawLine.trim();
    if (!line || line.startsWith("#")) continue;
    const idx = line.indexOf("=");
    if (idx === -1) continue;
    const key = line.slice(0, idx).trim();
    const value = line.slice(idx + 1).trim();
    if (key && !process.env[key]) {
      process.env[key] = value;
    }
  }
}

function fileToDataUrl(filePath) {
  const ext = path.extname(filePath).toLowerCase();
  const mime =
    ext === ".png"
      ? "image/png"
      : ext === ".jpg" || ext === ".jpeg"
        ? "image/jpeg"
        : "application/octet-stream";
  const buf = fs.readFileSync(filePath);
  return `data:${mime};base64,${buf.toString("base64")}`;
}

async function runSolveHandler({
  imageDataUrl,
  gridSize,
  preprocessedImageDataUrl,
  preprocessedDetectedGridSize,
  skipPreprocess,
}) {
  let resolveJson;
  let rejectJson;
  const p = new Promise((resolve, reject) => {
    resolveJson = resolve;
    rejectJson = reject;
  });

  const req = {
    method: "POST",
    body: {
      imageDataUrl,
      gridSize,
      preprocessedImageDataUrl,
      preprocessedDetectedGridSize,
      skipPreprocess,
    },
  };
  const res = {
    statusCode: 200,
    status(code) {
      this.statusCode = code;
      return this;
    },
    json(payload) {
      resolveJson({ status: this.statusCode, payload });
    },
  };

  try {
    // Handler is async, but writes via res.json().
    await solveHandler(req, res);
  } catch (err) {
    rejectJson(err);
  }
  return p;
}

function isImageFile(name) {
  const ext = path.extname(name).toLowerCase();
  return ext === ".jpg" || ext === ".jpeg" || ext === ".png";
}

function isValidSquareGrid(v) {
  if (!Array.isArray(v) || !v.length) return false;
  const n = v.length;
  if (!(n === 4 || n === 5)) return false;
  return v.every((row) => Array.isArray(row) && row.length === n);
}

function readJsonIfExists(filePath, fallback) {
  if (!fs.existsSync(filePath)) return fallback;
  try {
    return JSON.parse(fs.readFileSync(filePath, "utf8"));
  } catch (_) {
    return fallback;
  }
}

function parseArgs(argv) {
  const args = {
    concurrency: 4,
    force: false,
    retryFailedOnly: false,
  };

  for (let i = 2; i < argv.length; i++) {
    const a = String(argv[i] || "");
    if (a === "--force") {
      args.force = true;
      continue;
    }
    if (a === "--retry-failed-only") {
      args.retryFailedOnly = true;
      continue;
    }
    if (a.startsWith("--concurrency=")) {
      const v = Number(a.slice("--concurrency=".length));
      if (Number.isInteger(v) && v > 0) args.concurrency = v;
      continue;
    }
    if (a === "--concurrency" && i + 1 < argv.length) {
      const v = Number(argv[i + 1]);
      if (Number.isInteger(v) && v > 0) {
        args.concurrency = v;
        i += 1;
      }
    }
  }
  return args;
}

async function main() {
  const args = parseArgs(process.argv);
  loadDotEnv();

  if (!fs.existsSync(SCREENSHOTS_DIR)) {
    throw new Error(`Screenshots directory not found: ${SCREENSHOTS_DIR}`);
  }

  const files = fs
    .readdirSync(SCREENSHOTS_DIR)
    .filter(isImageFile)
    .sort((a, b) => a.localeCompare(b));

  if (!files.length) {
    throw new Error(`No image files found in: ${SCREENSHOTS_DIR}`);
  }

  const existingOut = readJsonIfExists(OUT_PATH, {});
  const existingErrors = readJsonIfExists(ERR_PATH, {});
  const out = { ...existingOut };
  const errors = { ...existingErrors };

  let targets = files;
  if (!args.force) {
    targets = targets.filter((name) => !isValidSquareGrid(out[name]));
  }
  if (args.retryFailedOnly) {
    targets = targets.filter((name) => Object.prototype.hasOwnProperty.call(errors, name));
  }

  const total = targets.length;
  if (total === 0) {
    process.stdout.write("No pending files. ground_truth.json already has all 5x5 boards.\n");
    return;
  }

  const concurrency = Math.max(1, Math.min(args.concurrency, total));
  process.stdout.write(
    `Processing ${total}/${files.length} file(s) with concurrency=${concurrency}` +
      `${args.force ? " (force)" : " (resume mode)"}\n`
  );

  let nextIndex = 0;
  let done = 0;

  async function processOne(name, ordinal) {
    const fullPath = path.join(SCREENSHOTS_DIR, name);
    process.stdout.write(`[${ordinal}/${total}] OCR ${name} ... `);

    try {
      const imageDataUrl = fileToDataUrl(fullPath);
      let payload = null;
      let lastErr = null;

      // Primary: prefer 5x5.
      // Fallback: auto-detect and accept whichever square board size is returned.
      for (const gridSizeHint of [5, null]) {
        try {
          const prep = await solveHandler.preprocessBoardImage(
            imageDataUrl,
            gridSizeHint === null ? null : gridSizeHint
          );
          const { status, payload: p } = await runSolveHandler({
            imageDataUrl,
            gridSize: gridSizeHint === null ? undefined : gridSizeHint,
            preprocessedImageDataUrl: prep.croppedDataUrl,
            preprocessedDetectedGridSize: prep.detectedGridSize,
            skipPreprocess: true,
          });
          if (status !== 200) {
            throw new Error(p?.error || `HTTP ${status}`);
          }
          const n = p?.board?.grid_size;
          const grid = p?.board?.grid;
          if (!(n === 4 || n === 5)) {
            throw new Error(`Expected 4x4 or 5x5 but got ${n}x${n}`);
          }
          if (!isValidSquareGrid(grid)) {
            throw new Error("Invalid grid shape.");
          }
          payload = p;
          break;
        } catch (e) {
          lastErr = e;
        }
      }

      if (!payload) throw lastErr || new Error("OCR failed");

      out[name] = payload.board.grid;
      delete errors[name];
      process.stdout.write("ok\n");
    } catch (err) {
      errors[name] = err?.message || String(err);
      process.stdout.write(`fail (${errors[name]})\n`);
    } finally {
      done += 1;
      if (done % 3 === 0 || done === total) {
        fs.writeFileSync(OUT_PATH, JSON.stringify(out, null, 2), "utf8");
        fs.writeFileSync(ERR_PATH, JSON.stringify(errors, null, 2), "utf8");
      }
      // Small pacing to reduce chance of rate limiting while still parallel.
      await new Promise((r) => setTimeout(r, 80));
    }
  }

  const workers = Array.from({ length: concurrency }, async () => {
    while (true) {
      const idx = nextIndex;
      nextIndex += 1;
      if (idx >= total) return;
      await processOne(targets[idx], idx + 1);
    }
  });

  await Promise.all(workers);

  fs.writeFileSync(OUT_PATH, JSON.stringify(out, null, 2), "utf8");

  const errKeys = Object.keys(errors);
  if (errKeys.length) {
    fs.writeFileSync(ERR_PATH, JSON.stringify(errors, null, 2), "utf8");
    process.stdout.write(
      `\nWrote ${Object.keys(out).length} boards to ground_truth.json; ${errKeys.length} failures in ground_truth.errors.json\n`
    );
    process.exitCode = 2;
    return;
  }

  process.stdout.write(`\nWrote ${Object.keys(out).length} boards to ground_truth.json\n`);
}

main().catch((err) => {
  console.error(err?.stack || String(err));
  process.exitCode = 1;
});
