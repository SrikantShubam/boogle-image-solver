const fs = require("fs");
const path = require("path");
const os = require("os");
const { spawnSync } = require("child_process");

let sharpInstance = null;
function getSharp() {
  if (sharpInstance) return sharpInstance;
  // Load lazily so deployment-time native module issues return JSON from handler.
  // Top-level require failures bypass the handler entirely and Vercel returns HTML/plain text.
  sharpInstance = require("sharp");
  return sharpInstance;
}

const MIN_WORD_LEN = 3;
const PRIMARY_MODEL = "mistralai/mistral-large-3-675b-instruct-2512";
const FALLBACK_MODEL = "google/gemma-3-27b-it";
const MODEL_CHAIN = [PRIMARY_MODEL, FALLBACK_MODEL];
const NVIDIA_URL = "https://integrate.api.nvidia.com/v1/chat/completions";
const MODEL_TIMEOUT_MS = 25000;
const MODEL_MAX_TOKENS = 220;

let WORD_CACHE = null;
let TRIE_CACHE = null;
let MAX_WORD_LEN = 0;

function loadWords() {
  if (WORD_CACHE) return WORD_CACHE;
  const wordsPath = path.join(process.cwd(), "words.txt");
  const text = fs.readFileSync(wordsPath, "utf-8");
  const set = new Set();
  for (const line of text.split(/\r?\n/)) {
    const w = line.trim().toUpperCase();
    if (w.length >= MIN_WORD_LEN && /^[A-Z]+$/.test(w)) {
      set.add(w);
    }
  }
  WORD_CACHE = set;
  MAX_WORD_LEN = 0;
  for (const w of set) {
    if (w.length > MAX_WORD_LEN) MAX_WORD_LEN = w.length;
  }
  return WORD_CACHE;
}

function buildTrie(words) {
  const root = { c: Object.create(null), e: false };
  for (const word of words) {
    let node = root;
    for (const ch of word) {
      if (!node.c[ch]) node.c[ch] = { c: Object.create(null), e: false };
      node = node.c[ch];
    }
    node.e = true;
  }
  return root;
}

function ensureTrie() {
  if (!TRIE_CACHE) {
    TRIE_CACHE = buildTrie(loadWords());
  }
  return TRIE_CACHE;
}

function normalizeToken(v) {
  const s = String(v || "")
    .toUpperCase()
    .replace(/[^A-Z]/g, "");
  if (!s) return null;
  return s.slice(0, 2);
}

function buildNeighbors(size) {
  const neighbors = Array.from({ length: size * size }, () => []);
  for (let r = 0; r < size; r++) {
    for (let c = 0; c < size; c++) {
      const idx = r * size + c;
      for (let dr = -1; dr <= 1; dr++) {
        for (let dc = -1; dc <= 1; dc++) {
          if (dr === 0 && dc === 0) continue;
          const nr = r + dr;
          const nc = c + dc;
          if (nr >= 0 && nr < size && nc >= 0 && nc < size) {
            neighbors[idx].push(nr * size + nc);
          }
        }
      }
    }
  }
  return neighbors;
}

function advanceTrie(node, token) {
  let cur = node;
  for (const ch of token) {
    cur = cur.c[ch];
    if (!cur) return null;
  }
  return cur;
}

function solveExactTrieDFS(grid) {
  const size = grid.length;
  const tiles = grid.flat();
  const neighbors = buildNeighbors(size);
  const trieRoot = ensureTrie();
  const found = new Set();

  function dfs(idx, node, mask, path) {
    const token = tiles[idx];
    const next = advanceTrie(node, token);
    if (!next) return;

    const word = path + token;
    if (next.e && word.length >= MIN_WORD_LEN) {
      found.add(word);
    }
    if (word.length >= MAX_WORD_LEN) return;

    const newMask = mask | (1n << BigInt(idx));
    for (const nb of neighbors[idx]) {
      if (newMask & (1n << BigInt(nb))) continue;
      if (advanceTrie(next, tiles[nb])) {
        dfs(nb, next, newMask, word);
      }
    }
  }

  for (let i = 0; i < tiles.length; i++) {
    if (advanceTrie(trieRoot, tiles[i])) {
      dfs(i, trieRoot, 0n, "");
    }
  }
  return found;
}

function parseJsonFromText(s) {
  if (!s) return null;
  const text = String(s).trim();
  try {
    return JSON.parse(text);
  } catch (_) {}

  const start = text.indexOf("{");
  const end = text.lastIndexOf("}");
  if (start >= 0 && end > start) {
    const slice = text.slice(start, end + 1);
    return JSON.parse(slice);
  }
  return null;
}

function normalizeRequestedGridSize(gridSize) {
  if (gridSize === 4 || gridSize === "4" || gridSize === "4x4") return 4;
  if (gridSize === 5 || gridSize === "5" || gridSize === "5x5") return 5;
  return null;
}

async function callNvidiaModel({ imageDataUrl, apiKey, model, detectedGridSize }) {
  const rowTemplate = (n) => `[${Array(n).fill('"?"').join(",")}]`;
  const fixedSizePrompt = detectedGridSize
    ? `This image shows a ${detectedGridSize}x${detectedGridSize} Boggle board with ${detectedGridSize * detectedGridSize} white circular tiles arranged in a grid.`
    : `This image shows either a 4x4 or 5x5 Boggle board made of white circular tiles arranged in a square grid. You must detect whether it is 4x4 or 5x5 before reading the letters.`;
  const schemaHint = detectedGridSize
    ? `{"grid_size":${detectedGridSize},"grid":[${Array(detectedGridSize).fill(rowTemplate(detectedGridSize)).join(",")}],"bonus":{"row":0,"col":0}}`
    : `{"grid_size":4_or_5,"grid":[["?","?","?","?"],["?","?","?","?"],["?","?","?","?"],["?","?","?","?"]],"bonus":{"row":0,"col":0}}\nIf the board is 5x5, return five rows with five entries each instead.`;
  const prompt = `${fixedSizePrompt} Read only the text inside the rounded white circles. The text is black. A tile may contain either one uppercase letter like A, B, S or a mixed-case two-letter tile like Qu, Th, He. If a tile visibly contains two letters, return both letters exactly as seen and do not collapse them to one letter. Read the tiles row by row, left to right, top to bottom. Return ONLY a JSON object, no markdown:\n${schemaHint}\nSet bonus to the row/col of the cyan tile, or null if none. Do not force 4x4 if the board is actually 5x5.`;

  const body = {
    model,
    messages: [
      {
        role: "user",
        content: [
          { type: "text", text: prompt },
          { type: "image_url", image_url: { url: imageDataUrl } },
        ],
      },
    ],
    max_tokens: MODEL_MAX_TOKENS,
    temperature: 0.0,
    top_p: 1.0,
    frequency_penalty: 0.0,
    presence_penalty: 0.0,
    stream: false,
  };

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), MODEL_TIMEOUT_MS);
  let res;
  try {
    res = await fetch(NVIDIA_URL, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${apiKey}`,
        Accept: "application/json",
      },
      body: JSON.stringify(body),
      signal: controller.signal,
    });
  } catch (err) {
    if (err.name === "AbortError") {
      throw new Error(`Model OCR timed out after ${MODEL_TIMEOUT_MS}ms.`);
    }
    throw err;
  } finally {
    clearTimeout(timeout);
  }

  if (!res.ok) {
    const t = await res.text();
    throw new Error(`Model OCR request failed: ${res.status} ${t}`);
  }
  const data = await res.json();
  const content = data?.choices?.[0]?.message?.content;
  const text = Array.isArray(content)
    ? content.map((x) => x.text || "").join("")
    : String(content || "");
  const parsed = parseJsonFromText(text);
  if (!parsed) {
    throw new Error("Could not parse JSON board from model response.");
  }
  return parsed;
}

function parseDataUrl(imageDataUrl) {
  const match = String(imageDataUrl).match(/^data:(.+?);base64,(.+)$/);
  if (!match) {
    throw new Error("imageDataUrl must be a base64 data URL.");
  }
  return {
    mimeType: match[1],
    buffer: Buffer.from(match[2], "base64"),
  };
}

async function preprocessBoardImage(imageDataUrl, requestedGridSize = null) {
  const sharp = getSharp();
  const { mimeType, buffer } = parseDataUrl(imageDataUrl);
  const meta = await sharp(buffer).metadata();
  const width = meta.width;
  const height = meta.height;

  if (!width || !height) {
    throw new Error("Could not read image metadata.");
  }

  let left = Math.max(0, Math.floor(width * 0.06));
  let top = Math.max(0, Math.floor(height * 0.25));
  let cropWidth = Math.min(width - left, Math.floor(width * 0.88));
  let cropHeight = Math.min(height - top, Math.floor(height * 0.58));
  let detectedGridSize = null;
  let detectedPoints = null;

  const helperPython = path.join(process.cwd(), ".venv", "Scripts", "python.exe");
  const helperScript = path.join(process.cwd(), "detect_board_bbox.py");
  // Prefer ML-robust preprocessing when available. Even if the caller requests a fixed
  // grid size (e.g. 5x5), the bbox detector can still improve cropping quality.
  if (fs.existsSync(helperPython) && fs.existsSync(helperScript)) {
    const ext = mimeType.includes("png") ? ".png" : ".jpg";
    const tmpPath = path.join(
      os.tmpdir(),
      `boggle-board-${Date.now()}-${Math.random().toString(36).slice(2)}${ext}`
    );
    fs.writeFileSync(tmpPath, buffer);
    try {
      const helperArgs = [helperScript, tmpPath];
      if (requestedGridSize === 4 || requestedGridSize === 5) {
        helperArgs.push(String(requestedGridSize));
      }
      const proc = spawnSync(helperPython, helperArgs, {
        cwd: process.cwd(),
        encoding: "utf8",
        timeout: 15000,
      });
      if (proc.status === 0 && proc.stdout) {
        const parsed = JSON.parse(proc.stdout.trim());
        // Only trust the detector's bbox/points when it agrees with the requested size.
        // If it disagrees, we fall back to the heuristic crop; applying a mismatched bbox
        // can clip the board and lead to wrong grid_size predictions downstream.
        if (!requestedGridSize || parsed.grid_size === requestedGridSize) {
          left = parsed.left;
          top = parsed.top;
          cropWidth = parsed.width;
          cropHeight = parsed.height;
          detectedGridSize = parsed.grid_size;
          detectedPoints = Array.isArray(parsed.points) ? parsed.points : null;
        }
      }
    } catch (_) {
      // Fall back to heuristic crop.
    } finally {
      try {
        fs.unlinkSync(tmpPath);
      } catch (_) {}
    }
  }

  let cropped;
  if (detectedGridSize && detectedPoints && detectedPoints.length === detectedGridSize * detectedGridSize) {
    const cellSize = 160;
    const pad = 18;
    const canvasSize = detectedGridSize * cellSize + (detectedGridSize + 1) * pad;
    const composites = [];

    for (let idx = 0; idx < detectedPoints.length; idx++) {
      const p = detectedPoints[idx];
      const row = Math.floor(idx / detectedGridSize);
      const col = idx % detectedGridSize;
      const radius = Math.max(24, Math.floor(p.r * 1.05));
      const tileLeft = Math.max(0, p.x - radius);
      const tileTop = Math.max(0, p.y - radius);
      const tileSize = Math.max(1, Math.min(radius * 2, width - tileLeft, height - tileTop));

      const tileBufferBase = await sharp(buffer)
        .extract({
          left: tileLeft,
          top: tileTop,
          width: tileSize,
          height: tileSize,
        })
        .resize(cellSize, cellSize)
        .normalize()
        .sharpen()
        .ensureAlpha()
        .png()
        .toBuffer();

      // Keep only inside the circular tile region so non-board text does not
      // leak into OCR when screenshots contain multiple grid-like regions.
      const circleR = Math.floor(cellSize * 0.46);
      const circleC = Math.floor(cellSize / 2);
      const circleMask = Buffer.from(
        `<svg width="${cellSize}" height="${cellSize}" xmlns="http://www.w3.org/2000/svg"><rect width="100%" height="100%" fill="black"/><circle cx="${circleC}" cy="${circleC}" r="${circleR}" fill="white"/></svg>`
      );
      const tileBuffer = await sharp(tileBufferBase)
        .composite([{ input: circleMask, blend: "dest-in" }])
        .png()
        .toBuffer();

      composites.push({
        input: tileBuffer,
        left: pad + col * (cellSize + pad),
        top: pad + row * (cellSize + pad),
      });
    }

    cropped = await sharp({
      create: {
        width: canvasSize,
        height: canvasSize,
        channels: 4,
        background: { r: 245, g: 247, b: 250, alpha: 1 },
      },
    })
      .composite(composites)
      .jpeg({ quality: 90 })
      .toBuffer();
  } else {
    cropped = await sharp(buffer)
      .extract({
        left,
        top,
        width: Math.max(1, cropWidth),
        height: Math.max(1, cropHeight),
      })
      .resize({
        width: 900,
        fit: "inside",
        withoutEnlargement: false,
      })
      .normalize()
      .sharpen()
      .jpeg({ quality: 90 })
      .toBuffer();
  }

  return {
    croppedDataUrl: `data:image/jpeg;base64,${cropped.toString("base64")}`,
    detectedGridSize: requestedGridSize || detectedGridSize,
  };
}

async function extractBoardWithFallback({
  imageDataUrl,
  apiKey,
  requestedGridSize,
  preprocessedImageDataUrl = null,
  preprocessedDetectedGridSize = null,
}) {
  const errors = [];
  let croppedDataUrl = preprocessedImageDataUrl;
  let detectedGridSize = preprocessedDetectedGridSize;
  if (!croppedDataUrl) {
    const prep = await preprocessBoardImage(imageDataUrl, requestedGridSize);
    croppedDataUrl = prep.croppedDataUrl;
    detectedGridSize = prep.detectedGridSize;
  }

  for (const model of MODEL_CHAIN) {
    try {
      const parsed = await callNvidiaModel({
        imageDataUrl: croppedDataUrl,
        apiKey,
        model,
        detectedGridSize,
      });
      // Treat schema/normalization issues as a model failure so we can fall back
      // to the next model rather than failing the whole request.
      const board = sanitizeBoard(parsed);
      return { board, modelUsed: model };
    } catch (err) {
      errors.push(`${model}: ${err.message}`);
    }
  }
  throw new Error(`All model attempts failed. ${errors.join(" | ")}`);
}

function sanitizeBoard(parsed) {
  const n = Number(parsed?.grid_size);
  if (!(n === 4 || n === 5)) {
    throw new Error("grid_size must be 4 or 5.");
  }

  if (!Array.isArray(parsed.grid) || parsed.grid.length !== n) {
    throw new Error("grid shape mismatch.");
  }

  const grid = [];
  for (let r = 0; r < n; r++) {
    const row = parsed.grid[r];
    if (!Array.isArray(row) || row.length !== n) {
      throw new Error("grid row shape mismatch.");
    }
    const out = [];
    for (let c = 0; c < n; c++) {
      const tok = normalizeToken(row[c]);
      if (!tok) throw new Error(`Invalid token at (${r},${c}).`);
      out.push(tok);
    }
    grid.push(out);
  }

  let bonus = null;
  if (parsed.bonus && Number.isInteger(parsed.bonus.row) && Number.isInteger(parsed.bonus.col)) {
    const br = parsed.bonus.row;
    const bc = parsed.bonus.col;
    if (br >= 0 && br < n && bc >= 0 && bc < n) {
      bonus = { row: br, col: bc };
    }
  }

  return { grid_size: n, grid, bonus };
}

function scoreWord(word) {
  return Math.max(0, word.length - 2);
}

async function handler(req, res) {
  if (req.method !== "POST") {
    return res.status(405).json({ error: "Method not allowed" });
  }

  try {
    const {
      imageDataUrl,
      apiKey: clientApiKey,
      gridSize,
      preprocessedImageDataUrl,
      preprocessedDetectedGridSize,
      skipPreprocess,
    } = req.body || {};
    if (!imageDataUrl || typeof imageDataUrl !== "string") {
      return res.status(400).json({ error: "imageDataUrl is required." });
    }

    const apiKey = process.env.NVIDIA_API_KEY || clientApiKey;
    if (!apiKey) {
      return res
        .status(400)
        .json({ error: "Missing API key. Set NVIDIA_API_KEY or provide apiKey in request." });
    }

    const requestedGridSize = normalizeRequestedGridSize(gridSize);
    const gridMode = requestedGridSize ? `${requestedGridSize}x${requestedGridSize}` : "auto";

    loadWords();
    ensureTrie();

    const { board, modelUsed } = await extractBoardWithFallback({
      imageDataUrl,
      apiKey,
      requestedGridSize,
      preprocessedImageDataUrl:
        skipPreprocess && typeof preprocessedImageDataUrl === "string"
          ? preprocessedImageDataUrl
          : null,
      preprocessedDetectedGridSize:
        skipPreprocess && Number.isInteger(preprocessedDetectedGridSize)
          ? preprocessedDetectedGridSize
          : null,
    });
    const words = [...solveExactTrieDFS(board.grid)];
    words.sort((a, b) => b.length - a.length || a.localeCompare(b));

    const bonusIndex =
      board.bonus && Number.isInteger(board.bonus.row) && Number.isInteger(board.bonus.col)
        ? board.bonus.row * board.grid_size + board.bonus.col
        : null;

    // Keep scoring lightweight for UI summary.
    let totalScore = 0;
    for (const w of words) {
      totalScore += scoreWord(w);
      if (bonusIndex !== null) {
        // approximate bonus accounting: if bonus cell letter appears, +1 (UI only).
        const bonusToken = board.grid[Math.floor(bonusIndex / board.grid_size)][
          bonusIndex % board.grid_size
        ];
        if (w.includes(bonusToken)) totalScore += 1;
      }
    }

    return res.status(200).json({
      model_used: modelUsed,
      grid_mode: gridMode,
      board,
      count: words.length,
      score_estimate: totalScore,
      words,
    });
  } catch (err) {
    return res.status(500).json({
      error: err.message || "Unknown error",
    });
  }
}

module.exports = handler;
module.exports.preprocessBoardImage = preprocessBoardImage;
