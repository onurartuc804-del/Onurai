import express from "express";
import dotenv from "dotenv";
import { GoogleGenAI } from "@google/genai";
import sharp from "sharp";
import ffmpegPath from "ffmpeg-static";
import { spawn } from "node:child_process";
import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import crypto from "node:crypto";

dotenv.config();

const app = express();
const PORT = Number(process.env.PORT || 3000);
const MODEL =
  process.env.GEMINI_CHAT_MODEL || "gemini-3.5-flash-lite";

if (!process.env.GEMINI_API_KEY) {
  console.error("GEMINI_API_KEY .env dosyasında eksik.");
  process.exit(1);
}

const ai = new GoogleGenAI({
  apiKey: process.env.GEMINI_API_KEY
});

app.use(express.json({ limit: "2mb" }));
app.use(express.static("public"));

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function num(value, fallback = 0) {
  const result = Number(value);
  return Number.isFinite(result) ? result : fallback;
}

function safeColor(value, fallback = "#ffffff") {
  if (
    typeof value === "string" &&
    (/^#[0-9a-fA-F]{3,8}$/.test(value) ||
      /^[a-zA-Z]{1,20}$/.test(value))
  ) {
    return value;
  }
  return fallback;
}

function escapeXml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&apos;");
}

function extractJSON(text) {
  const cleaned = String(text || "")
    .replace(/```json/gi, "")
    .replace(/```/g, "")
    .trim();

  const start = cleaned.indexOf("{");
  const end = cleaned.lastIndexOf("}");

  if (start === -1 || end <= start) {
    throw new Error("Geçerli JSON şablonu alınamadı.");
  }

  return JSON.parse(cleaned.slice(start, end + 1));
}

async function generateTemplate(prompt, video = false) {
  const instructions = video
    ? `
Bir animasyon videosu için JSON şablonu üret.
Yalnızca geçerli JSON döndür, açıklama veya Markdown yazma.
Şema:
{
  "type":"video",
  "title":"Sahne",
  "width":512,
  "height":512,
  "duration":5,
  "fps":24,
  "background":"#87CEEB",
  "shapes":[
    {
      "type":"circle",
      "x":200,
      "y":150,
      "radius":35,
      "color":"#ff9900",
      "vx":2,
      "vy":0,
      "rotationSpeed":0
    }
  ]
}
Desteklenen şekiller: rect, circle, ellipse, line, polygon, text.
En fazla 50 şekil kullan.
Süre 1-10 saniye, fps 1-30 arasında olsun.
Hareket için vx, vy ve rotationSpeed kullan.
Koordinatları sahne boyutlarına göre ayarla.
`
    : `
Bir görsel için JSON çizim şablonu üret.
Yalnızca geçerli JSON döndür, açıklama veya Markdown yazma.
Şema:
{
  "width":512,
  "height":512,
  "background":"#ffffff",
  "shapes":[
    {
      "type":"circle",
      "x":200,
      "y":150,
      "radius":40,
      "color":"#ff9900"
    }
  ]
}
Desteklenen şekiller: rect, circle, ellipse, line, polygon, text.
En fazla 50 şekil kullan.
`;

  const response = await ai.models.generateContent({
    model: MODEL,
    contents: `${instructions}\nKullanıcı isteği: ${prompt}`
  });

  return extractJSON(response.text);
}

function shapeToSvg(shape, width, height, time = 0) {
  const type = String(shape.type || "");
  const x = num(shape.x);
  const y = num(shape.y);
  const color = safeColor(shape.color);
  const opacity = clamp(num(shape.opacity, 1), 0, 1);

  const px = clamp(
    x + num(shape.vx) * time,
    -width * 2,
    width * 3
  );

  const py = clamp(
    y + num(shape.vy) * time,
    -height * 2,
    height * 3
  );

  const rotation =
    num(shape.rotation) + num(shape.rotationSpeed) * time;

  const transform = rotation
    ? ` transform="rotate(${rotation} ${px} ${py})"`
    : "";

  const style = `fill="${color}" opacity="${opacity}"`;

  switch (type) {
    case "rect": {
      const w = clamp(num(shape.width, 50), 0, width * 3);
      const h = clamp(num(shape.height, 50), 0, height * 3);

      return `<rect x="${px}" y="${py}" width="${w}" height="${h}" ${style}${transform}/>`;
    }

    case "circle": {
      const radius = clamp(num(shape.radius, 20), 0, width);

      return `<circle cx="${px}" cy="${py}" r="${radius}" ${style}${transform}/>`;
    }

    case "ellipse": {
      const rx = clamp(num(shape.rx, 30), 0, width);
      const ry = clamp(num(shape.ry, 20), 0, height);

      return `<ellipse cx="${px}" cy="${py}" rx="${rx}" ry="${ry}" ${style}${transform}/>`;
    }

    case "line": {
      const x2 = clamp(
        num(shape.x2, px + 50) + num(shape.vx) * time,
        -width * 2,
        width * 3
      );

      const y2 = clamp(
        num(shape.y2, py + 50) + num(shape.vy) * time,
        -height * 2,
        height * 3
      );

      const strokeWidth = clamp(
        num(shape.strokeWidth, 3),
        1,
        50
      );

      return `<line x1="${px}" y1="${py}" x2="${x2}" y2="${y2}" stroke="${color}" stroke-width="${strokeWidth}" opacity="${opacity}"/>`;
    }

    case "polygon": {
      if (!Array.isArray(shape.points)) return "";

      const points = shape.points
        .slice(0, 100)
        .filter(
          (point) =>
            Array.isArray(point) && point.length >= 2
        )
        .map((point) => {
          const xx = clamp(
            num(point[0]) + num(shape.vx) * time,
            -width * 2,
            width * 3
          );

          const yy = clamp(
            num(point[1]) + num(shape.vy) * time,
            -height * 2,
            height * 3
          );

          return `${xx},${yy}`;
        })
        .join(" ");

      return points
        ? `<polygon points="${points}" ${style}${transform}/>`
        : "";
    }

    case "text": {
      const fontSize = clamp(num(shape.fontSize, 24), 8, 120);
      const content = escapeXml(shape.text || "");

      return `<text x="${px}" y="${py}" font-size="${fontSize}" fill="${color}" opacity="${opacity}" font-family="Arial, sans-serif"${transform}>${content}</text>`;
    }

    default:
      return "";
  }
}

function buildSvg(template, time = 0) {
  const width = clamp(
    Math.round(num(template.width, 512)),
    64,
    1024
  );

  const height = clamp(
    Math.round(num(template.height, 512)),
    64,
    1024
  );

  const background = safeColor(
    template.background,
    "#ffffff"
  );

  const shapes = Array.isArray(template.shapes)
    ? template.shapes.slice(0, 50)
    : [];

  const body = shapes
    .map((shape) =>
      shapeToSvg(shape, width, height, time)
    )
    .join("");

  const svg = `
    <svg xmlns="http://www.w3.org/2000/svg"
         width="${width}" height="${height}"
         viewBox="0 0 ${width} ${height}">
      <rect width="100%" height="100%" fill="${background}"/>
      ${body}
    </svg>
  `;

  return { svg, width, height };
}

async function renderPng(template, time = 0) {
  const { svg, width, height } = buildSvg(template, time);

  return sharp(Buffer.from(svg))
    .resize(width, height)
    .png()
    .toBuffer();
}

function runProcess(command, args) {
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, {
      stdio: ["ignore", "ignore", "pipe"]
    });

    let errorOutput = "";

    child.stderr.on("data", (data) => {
      errorOutput += data.toString();

      if (errorOutput.length > 12000) {
        errorOutput = errorOutput.slice(-12000);
      }
    });

    child.on("error", reject);

    child.on("close", (code) => {
      if (code === 0) {
        resolve();
      } else {
        reject(
          new Error(
            errorOutput || `FFmpeg hata kodu: ${code}`
          )
        );
      }
    });
  });
}

async function createVideo(template) {
  if (!ffmpegPath) {
    throw new Error("FFmpeg bulunamadı.");
  }

  const width = clamp(
    Math.round(num(template.width, 512)),
    64,
    720
  );

  const height = clamp(
    Math.round(num(template.height, 512)),
    64,
    720
  );

  const duration = clamp(
    num(template.duration, 5),
    1,
    10
  );

  const fps = clamp(
    Math.round(num(template.fps, 24)),
    1,
    30
  );

  const totalFrames = Math.ceil(duration * fps);

  const tempDir = path.join(
    os.tmpdir(),
    `onur-ai-${crypto.randomUUID()}`
  );

  await fs.mkdir(tempDir, { recursive: true });

  try {
    for (let i = 0; i < totalFrames; i++) {
      const png = await renderPng(
        { ...template, width, height },
        i / fps
      );

      const framePath = path.join(
        tempDir,
        `frame-${String(i).padStart(5, "0")}.png`
      );

      await fs.writeFile(framePath, png);
    }

    const outputPath = path.join(tempDir, "video.mp4");

    await runProcess(ffmpegPath, [
      "-y",
      "-framerate", String(fps),
      "-i", path.join(tempDir, "frame-%05d.png"),
      "-t", String(duration),
      "-c:v", "libx264",
      "-pix_fmt", "yuv420p",
      "-movflags", "+faststart",
      outputPath
    ]);

    return await fs.readFile(outputPath);
  } finally {
    await fs.rm(tempDir, {
      recursive: true,
      force: true
    });
  }
}

app.get("/api/health", (req, res) => {
  res.json({
    ok: true,
    model: MODEL
  });
});

app.post("/api/chat", async (req, res) => {
  try {
    const message = String(
      req.body?.message || ""
    ).trim();

    if (!message) {
      return res.status(400).json({
        error: "Mesaj boş olamaz."
      });
    }

    const response = await ai.models.generateContent({
      model: MODEL,
      contents: message
    });

    res.json({
      reply: response.text || "Yanıt oluşturulamadı."
    });
  } catch (error) {
    console.error("Chat hatası:", error);

    res.status(500).json({
      error: "Sohbet başarısız. API anahtarını ve model adını kontrol et."
    });
  }
});

app.post("/api/image", async (req, res) => {
  try {
    const prompt = String(
      req.body?.prompt || ""
    ).trim();

    if (!prompt) {
      return res.status(400).json({
        error: "Görsel açıklaması gerekli."
      });
    }

    const template = await generateTemplate(prompt, false);
    const png = await renderPng(template);

    res.setHeader("Content-Type", "image/png");
    res.setHeader(
      "Content-Disposition",
      'inline; filename="onur-ai.png"'
    );

    res.send(png);
  } catch (error) {
    console.error("Görsel hatası:", error);

    res.status(500).json({
      error: "Görsel oluşturulamadı. Sunucu günlüklerini kontrol et."
    });
  }
});

app.post("/api/video", async (req, res) => {
  try {
    const prompt = String(
      req.body?.prompt || ""
    ).trim();

    if (!prompt) {
      return res.status(400).json({
        error: "Video açıklaması gerekli."
      });
    }

    const template = await generateTemplate(prompt, true);
    const video = await createVideo(template);

    res.setHeader("Content-Type", "video/mp4");
    res.setHeader(
      "Content-Disposition",
      'inline; filename="onur-ai.mp4"'
    );

    res.send(video);
  } catch (error) {
    console.error("Video hatası:", error);

    res.status(500).json({
      error: "Video oluşturulamadı. Modeli ve FFmpeg kurulumunu kontrol et."
    });
  }
});

app.listen(PORT, () => {
  console.log(`Onur AI çalışıyor: http://localhost:${PORT}`);
});
