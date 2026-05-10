import express from "express";
import cors from "cors";
import helmet from "helmet";
import cookieParser from "cookie-parser";
import rateLimit from "express-rate-limit";
import { config } from "./config.js";
import { migrate } from "./db.js";
import { authRouter } from "./routes/auth.js";
import { dashboardRouter } from "./routes/dashboard.js";

await migrate();

const app = express();

app.use(helmet({ contentSecurityPolicy: false }));
app.use(express.json({ limit: "8mb" }));
app.use(cookieParser());
app.use(cors({ origin: config.clientOrigin, credentials: true }));
app.use(rateLimit({ windowMs: 15 * 60 * 1000, limit: 300, standardHeaders: true, legacyHeaders: false }));

app.get("/api/health", (req, res) => res.json({ ok: true, service: "gait-auth-api" }));
app.use("/api/auth", authRouter);
app.use("/api/dashboard", dashboardRouter);

app.use((req, res) => res.status(404).json({ message: "Route not found." }));

app.use((err, req, res, next) => {
  console.error(err);
  res.status(500).json({ message: "Unexpected server error." });
});

app.listen(config.port, () => {
  console.log(`API running on port ${config.port}`);
});
