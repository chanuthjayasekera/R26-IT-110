import express from "express";
import fs from "fs";
import multer from "multer";
import path from "path";
import { spawn } from "child_process";
import { fileURLToPath } from "url";
import { config } from "../config.js";
import { db } from "../db.js";
import { requireAuth, requireRole } from "../middleware/auth.js";
import { createId } from "../utils/security.js";

const dirname = path.dirname(fileURLToPath(import.meta.url));
const serverRoot = path.resolve(dirname, "..", "..");
const uploadRoot = path.join(serverRoot, "data", "uploads", "normal-abnormal");
const inferenceScript = path.join(serverRoot, "ml", "component1", "run_inference.py");

fs.mkdirSync(uploadRoot, { recursive: true });

function safeFileName(name = "upload") {
  const ext = path.extname(name).toLowerCase();
  const stem = path.basename(name, ext).replace(/[^a-zA-Z0-9_-]/g, "_").slice(0, 80) || "gait_upload";
  return `${stem}${ext}`;
}

const upload = multer({
  storage: multer.diskStorage({
    destination: (req, file, cb) => cb(null, uploadRoot),
    filename: (req, file, cb) => cb(null, `${Date.now()}_${createId("file")}_${safeFileName(file.originalname)}`)
  }),
  limits: { fileSize: 300 * 1024 * 1024 },
  fileFilter: (req, file, cb) => {
    const ext = path.extname(file.originalname).toLowerCase();
    if ([".mp4", ".mov", ".avi", ".mkv", ".csv"].includes(ext)) return cb(null, true);
    return cb(new Error("Upload a gait video or training-safe CSV file."));
  }
});

export const normalAbnormalRouter = express.Router();

function numberOrNull(value) {
  if (value === undefined || value === null || value === "") return null;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function inferInputType(fileName, requestedType) {
  if (requestedType === "csv" || requestedType === "video") return requestedType;
  return path.extname(fileName).toLowerCase() === ".csv" ? "csv" : "video";
}

function runComponent1({ filePath, inputType, direction, fps }) {
  return new Promise((resolve, reject) => {
    const args = [inferenceScript, "--input", filePath, "--input-type", inputType];
    if (direction) args.push("--direction", direction);
    if (fps) args.push("--fps", String(fps));

    const child = spawn(config.pythonPath, args, {
      cwd: path.join(serverRoot, "ml", "component1"),
      windowsHide: true
    });

    let stdout = "";
    let stderr = "";

    child.stdout.on("data", (chunk) => {
      stdout += chunk.toString();
    });

    child.stderr.on("data", (chunk) => {
      stderr += chunk.toString();
    });

    child.on("error", (error) => reject(error));

    child.on("close", (code) => {
      let payload;
      try {
        payload = JSON.parse(stdout.trim());
      } catch {
        return reject(new Error(stderr || stdout || `Component 1 model exited with code ${code}.`));
      }

      if (code !== 0 || !payload.ok) {
        return reject(new Error(payload?.message || stderr || "Component 1 prediction failed."));
      }

      resolve(payload.result);
    });
  });
}

function publicScreening(row) {
  if (!row) return null;
  const isVideo = row.input_type === "video";
  return {
    id: row.id,
    inputType: row.input_type,
    fileName: row.file_name,
    hasVideoPreview: isVideo,
    videoUrl: isVideo ? `/normal-abnormal/screenings/${row.id}/file` : null,
    direction: row.direction,
    fpsUsed: row.fps_used,
    finalLabel: row.final_label,
    finalResult: row.final_result,
    modelSuggestedLabel: row.model_suggested_label,
    modelSuggestedResult: row.model_suggested_result,
    meanProbAbnormal: row.mean_prob_abnormal,
    confidencePercent: row.confidence_percent,
    abnormalRatioThreshold: row.abnormal_ratio_threshold,
    screeningSeverity: row.screening_severity,
    reliabilityLevel: row.reliability_level,
    reliabilityReasons: JSON.parse(row.reliability_reasons || "[]"),
    clinicalNote: row.clinical_note,
    result: JSON.parse(row.raw_result_json || "{}"),
    createdAt: row.created_at
  };
}

function publicClinicalProfile(row, screenings = []) {
  const latest = screenings[0] || null;
  const abnormalScreenings = screenings.filter((item) => item.final_label === 1).length;
  const normalScreenings = screenings.filter((item) => item.final_label === 0).length;

  return {
    id: row?.id || null,
    totalScreenings: screenings.length,
    abnormalScreenings,
    normalScreenings,
    latestResult: latest?.final_result || null,
    latestSeverity: latest?.screening_severity || null,
    latestDirection: latest?.direction || null,
    latestConfidence: latest?.confidence_percent ?? null,
    latestProbability: latest?.mean_prob_abnormal ?? null,
    latestReliability: latest?.reliability_level || null,
    updatedAt: latest?.created_at || row?.updated_at || null
  };
}

function safeUnlinkStoredFile(storedPath) {
  if (!storedPath) return;
  const absolutePath = path.isAbsolute(storedPath)
    ? path.resolve(storedPath)
    : path.resolve(serverRoot, storedPath);

  if (!absolutePath.startsWith(serverRoot)) return;
  fs.promises.unlink(absolutePath).catch(() => {});
}

normalAbnormalRouter.get("/clinical-profile", requireAuth, requireRole("patient"), async (req, res) => {
  const profile = await db.get("SELECT * FROM clinical_profiles WHERE user_id = ?", req.user.id);
  const screenings = await db.all(`
    SELECT * FROM normal_abnormal_screenings
    WHERE user_id = ?
    ORDER BY created_at DESC
    LIMIT 20
  `, req.user.id);

  res.json({
    profile: publicClinicalProfile(profile, screenings),
    screenings: screenings.map(publicScreening)
  });
});

normalAbnormalRouter.get("/screenings/:id/file", requireAuth, requireRole("patient"), async (req, res) => {
  const screening = await db.get(`
    SELECT * FROM normal_abnormal_screenings
    WHERE id = ? AND user_id = ? AND input_type = 'video'
  `, req.params.id, req.user.id);

  if (!screening) return res.status(404).json({ message: "Screening video not found." });

  const absolutePath = path.isAbsolute(screening.file_path)
    ? path.resolve(screening.file_path)
    : path.resolve(serverRoot, screening.file_path);

  if (!absolutePath.startsWith(serverRoot) || !fs.existsSync(absolutePath)) {
    return res.status(404).json({ message: "Screening video file not found." });
  }

  res.setHeader("Content-Disposition", `inline; filename="${safeFileName(screening.file_name)}"`);
  res.sendFile(absolutePath);
});

normalAbnormalRouter.get("/screenings/:id", requireAuth, requireRole("patient"), async (req, res) => {
  const screening = await db.get(`
    SELECT * FROM normal_abnormal_screenings
    WHERE id = ? AND user_id = ?
  `, req.params.id, req.user.id);

  if (!screening) return res.status(404).json({ message: "Screening record not found." });
  res.json({ screening: publicScreening(screening) });
});

async function deleteScreeningRecord(req, res) {
  const screening = await db.get(`
    SELECT * FROM normal_abnormal_screenings
    WHERE id = ? AND user_id = ?
  `, req.params.id, req.user.id);

  if (!screening) return res.status(404).json({ message: "Screening record not found." });

  const now = new Date().toISOString();
  await db.transaction(async () => {
    await db.run("DELETE FROM normal_abnormal_screenings WHERE id = ? AND user_id = ?", req.params.id, req.user.id);
    await db.run("UPDATE clinical_profiles SET updated_at = ? WHERE user_id = ?", now, req.user.id);
  });

  safeUnlinkStoredFile(screening.file_path);
  safeUnlinkStoredFile(screening.csv_path);

  const profile = await db.get("SELECT * FROM clinical_profiles WHERE user_id = ?", req.user.id);
  const latestScreenings = await db.all(`
    SELECT * FROM normal_abnormal_screenings
    WHERE user_id = ?
    ORDER BY created_at DESC
    LIMIT 20
  `, req.user.id);

  res.json({
    message: "Screening record deleted.",
    profile: publicClinicalProfile(profile, latestScreenings),
    screenings: latestScreenings.map(publicScreening)
  });
}

normalAbnormalRouter.delete("/screenings/:id", requireAuth, requireRole("patient"), deleteScreeningRecord);
normalAbnormalRouter.post("/screenings/:id/delete", requireAuth, requireRole("patient"), deleteScreeningRecord);

async function clearScreeningRecords(req, res) {
  const screenings = await db.all(`
    SELECT * FROM normal_abnormal_screenings
    WHERE user_id = ?
  `, req.user.id);

  const now = new Date().toISOString();
  await db.transaction(async () => {
    await db.run("DELETE FROM normal_abnormal_screenings WHERE user_id = ?", req.user.id);
    await db.run("UPDATE clinical_profiles SET updated_at = ? WHERE user_id = ?", now, req.user.id);
  });

  screenings.forEach((screening) => {
    safeUnlinkStoredFile(screening.file_path);
    safeUnlinkStoredFile(screening.csv_path);
  });

  const profile = await db.get("SELECT * FROM clinical_profiles WHERE user_id = ?", req.user.id);

  res.json({
    message: "All screening records cleared.",
    profile: publicClinicalProfile(profile, []),
    screenings: []
  });
}

normalAbnormalRouter.delete("/screenings", requireAuth, requireRole("patient"), clearScreeningRecords);
normalAbnormalRouter.post("/screenings/clear", requireAuth, requireRole("patient"), clearScreeningRecords);

normalAbnormalRouter.post(
  "/screenings",
  requireAuth,
  requireRole("patient"),
  upload.single("gaitFile"),
  async (req, res) => {
    if (!req.file) return res.status(422).json({ message: "Upload a gait video or CSV file." });

    const inputType = inferInputType(req.file.originalname, req.body.inputType);
    const direction = ["L2R", "R2L"].includes(req.body.direction) ? req.body.direction : null;
    const fps = numberOrNull(req.body.fps);

    try {
      const result = await runComponent1({
        filePath: req.file.path,
        inputType,
        direction,
        fps
      });

      const now = new Date().toISOString();
      const profileId = createId("clin");
      const screeningId = createId("nas");

      const saved = await db.transaction(async () => {
        const existingProfile = await db.get("SELECT id FROM clinical_profiles WHERE user_id = ?", req.user.id);
        const finalProfileId = existingProfile?.id || profileId;

        await db.run(`
          INSERT INTO clinical_profiles (
            id, user_id, created_at, updated_at
          ) VALUES (?, ?, ?, ?)
          ON CONFLICT(user_id) DO UPDATE SET
            updated_at = excluded.updated_at
        `, finalProfileId,
          req.user.id,
          now,
          now);

        await db.run(`
          INSERT INTO normal_abnormal_screenings (
            id, user_id, clinical_profile_id, input_type, file_name, file_path,
            csv_path, direction, fps_used, final_label, final_result,
            model_suggested_label, model_suggested_result, mean_prob_abnormal,
            confidence_percent, abnormal_ratio_threshold, screening_severity,
            reliability_level, reliability_reasons, clinical_note, raw_result_json, created_at
          ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        `, screeningId,
          req.user.id,
          finalProfileId,
          inputType,
          req.file.originalname,
          path.relative(serverRoot, req.file.path),
          result.csv_file || null,
          result.direction || direction,
          numberOrNull(result.fps_used || fps),
          result.final_label ?? null,
          result.final_result || null,
          result.model_suggested_label ?? null,
          result.model_suggested_result || null,
          numberOrNull(result.mean_prob_abnormal),
          numberOrNull(result.confidence_percent),
          numberOrNull(result.abnormal_ratio_threshold),
          result.screening_severity || null,
          result.reliability_level || null,
          JSON.stringify(result.reliability_reasons || []),
          result.clinical_note || null,
          JSON.stringify(result),
          now);

        return db.get("SELECT * FROM normal_abnormal_screenings WHERE id = ?", screeningId);
      });
      const profile = await db.get("SELECT * FROM clinical_profiles WHERE user_id = ?", req.user.id);
      const latestScreenings = await db.all(`
        SELECT * FROM normal_abnormal_screenings
        WHERE user_id = ?
        ORDER BY created_at DESC
        LIMIT 20
      `, req.user.id);

      res.status(201).json({
        message: "Normal vs abnormal screening completed and saved.",
        screening: publicScreening(saved),
        profile: publicClinicalProfile(profile, latestScreenings)
      });
    } catch (error) {
      res.status(500).json({
        message: error.message || "Normal vs abnormal detection failed."
      });
    }
  }
);
