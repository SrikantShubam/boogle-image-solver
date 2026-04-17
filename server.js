const fs = require("fs");
const path = require("path");
const http = require("http");

const solveHandler = require("./api/solve");

const PORT = Number(process.env.PORT || 3000);
const ROOT = process.cwd();

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

function sendJson(res, statusCode, payload) {
  const body = JSON.stringify(payload);
  res.writeHead(statusCode, {
    "Content-Type": "application/json; charset=utf-8",
    "Content-Length": Buffer.byteLength(body),
  });
  res.end(body);
}

function sendFile(res, filePath) {
  if (!fs.existsSync(filePath)) {
    sendJson(res, 404, { error: "Not found" });
    return;
  }

  const ext = path.extname(filePath).toLowerCase();
  const contentType =
    {
      ".html": "text/html; charset=utf-8",
      ".js": "text/javascript; charset=utf-8",
      ".css": "text/css; charset=utf-8",
      ".json": "application/json; charset=utf-8",
      ".png": "image/png",
      ".jpg": "image/jpeg",
      ".jpeg": "image/jpeg",
      ".svg": "image/svg+xml",
      ".ico": "image/x-icon",
      ".txt": "text/plain; charset=utf-8",
    }[ext] || "application/octet-stream";

  res.writeHead(200, { "Content-Type": contentType });
  fs.createReadStream(filePath).pipe(res);
}

function parseRequestBody(req) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    req.on("data", (chunk) => chunks.push(chunk));
    req.on("end", () => {
      if (!chunks.length) {
        resolve({});
        return;
      }
      try {
        const text = Buffer.concat(chunks).toString("utf8");
        resolve(JSON.parse(text));
      } catch (error) {
        reject(error);
      }
    });
    req.on("error", reject);
  });
}

function createApiResponse(res) {
  return {
    status(code) {
      this.statusCode = code;
      return this;
    },
    json(payload) {
      sendJson(res, this.statusCode || 200, payload);
    },
  };
}

loadDotEnv();

const server = http.createServer(async (req, res) => {
  try {
    const url = new URL(req.url, `http://${req.headers.host}`);

    if (req.method === "POST" && url.pathname === "/api/solve") {
      req.body = await parseRequestBody(req);
      const apiRes = createApiResponse(res);
      await solveHandler(req, apiRes);
      return;
    }

    if (req.method === "GET" && url.pathname === "/") {
      sendFile(res, path.join(ROOT, "index.html"));
      return;
    }

    if (req.method === "GET") {
      const filePath = path.join(ROOT, decodeURIComponent(url.pathname.replace(/^\/+/, "")));
      if (filePath.startsWith(ROOT)) {
        sendFile(res, filePath);
        return;
      }
    }

    sendJson(res, 404, { error: "Not found" });
  } catch (error) {
    sendJson(res, 500, { error: error.message || "Server error" });
  }
});

server.listen(PORT, () => {
  console.log(`Boggle demo running at http://localhost:${PORT}`);
});
