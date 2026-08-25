import express from "express";
import cors from "cors";
import helmet from "helmet";
import cookieParser from "cookie-parser";
import rateLimit from "express-rate-limit";
import { config } from "./config.js";
import { migrate } from "./db.js";
import { authRouter } from "./routes/auth.js";
import { dashboardRouter } from "./routes/dashboard.js";
import { normalAbnormalRouter } from "./routes/normalAbnormal.js";
import { scaKoaRouter } from "./routes/scaKoa.js";
import { pdNeuropathyRouter } from "./routes/pdNeuropathy.js";
import { exerciseDetectionRouter } from "./routes/exerciseDetection.js";
import { centralProfileRouter } from "./routes/centralProfile.js";

await migrate();

const app = express();
const allowedOrigins = new Set(config.clientOrigins);

function isAllowedDevOrigin(origin) {
  if (config.nodeEnv === "production") return false;

  try {
    const url = new URL(origin);
    return ["localhost", "127.0.0.1"].includes(url.hostname);
  } catch {
    return false;
  }
}

app.use(helmet({
  contentSecurityPolicy: false,
  crossOriginResourcePolicy: { policy: "cross-origin" }
}));
app.use(express.json({ limit: "8mb" }));
app.use(cookieParser());
app.use(cors({
  origin(origin, callback) {
    if (!origin || allowedOrigins.has(origin) || isAllowedDevOrigin(origin)) {
      return callback(null, true);
    }

    return callback(new Error(`CORS origin not allowed: ${origin}`));
  },
  credentials: true
}));
app.use(rateLimit({ windowMs: 15 * 60 * 1000, limit: 300, standardHeaders: true, legacyHeaders: false }));

app.get("/api/health", (req, res) => res.json({ ok: true, service: "gait-auth-api" }));
app.use("/api/auth", authRouter);
app.use("/api/dashboard", dashboardRouter);
app.use("/api/normal-abnormal", normalAbnormalRouter);
app.use("/api/sca-koa", scaKoaRouter);
app.use("/api/pd-neuropathy", pdNeuropathyRouter);
app.use("/api/exercise-detection", exerciseDetectionRouter);
app.use("/api/central-profile", centralProfileRouter);

app.use((req, res) => res.status(404).json({ message: "Route not found." }));

app.use((err, req, res, next) => {
  console.error(err);
  res.status(500).json({ message: "Unexpected server error." });
});

app.listen(config.port, () => {
  console.log(`API running on port ${config.port}`);
});
