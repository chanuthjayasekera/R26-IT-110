import dotenv from "dotenv";

dotenv.config();

function parseClientOrigins(value) {
  return String(value || "http://localhost:5173")
    .split(",")
    .map((origin) => origin.trim())
    .filter(Boolean);
}

const clientOrigins = parseClientOrigins(process.env.CLIENT_ORIGIN);

export const config = {
  port: Number(process.env.PORT || 5000),
  clientOrigin: clientOrigins[0],
  clientOrigins,
  jwtSecret: process.env.JWT_SECRET || "dev_only_replace_this_secret",
  nodeEnv: process.env.NODE_ENV || "development",
  cookieName: "gait_session",
  pythonPath: process.env.PYTHON_PATH || process.env.PYTHON || (process.platform === "win32" ? "python" : "python3"),
  databaseUrl: process.env.DATABASE_URL || "",
  smtp: {
    host: process.env.SMTP_HOST || "",
    port: Number(process.env.SMTP_PORT || 587),
    secure: String(process.env.SMTP_SECURE || "false").toLowerCase() === "true",
    user: process.env.SMTP_USER || "",
    pass: process.env.SMTP_PASS || "",
    from: process.env.SMTP_FROM || "Gait AI Care <no-reply@gait-ai-care.local>"
  }
};
