// Static file server for the graph viewer. No deps: node's http + fs cover this.
const http = require("http");
const fs = require("fs");
const path = require("path");

const PORT = process.env.PORT || 8787;
const HOST = process.env.HOST || "0.0.0.0";
const PUBLIC_DIR = path.join(__dirname, "public");

const MIME = { ".html": "text/html", ".js": "text/javascript", ".css": "text/css", ".json": "application/json" };

http
  .createServer((req, res) => {
    const reqPath = req.url === "/" ? "/index.html" : req.url;
    const filePath = path.join(PUBLIC_DIR, decodeURIComponent(reqPath.split("?")[0]));
    if (!filePath.startsWith(PUBLIC_DIR)) {
      res.writeHead(403).end("Forbidden");
      return;
    }
    fs.readFile(filePath, (err, data) => {
      if (err) {
        res.writeHead(404).end("Not found");
        return;
      }
      res.writeHead(200, { "Content-Type": MIME[path.extname(filePath)] || "application/octet-stream" });
      res.end(data);
    });
  })
  .listen(PORT, HOST, () => console.log(`Graph viewer at http://${HOST}:${PORT}`));
