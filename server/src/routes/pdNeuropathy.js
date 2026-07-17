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
const uploadRoot = path.join(serverRoot, "data", "uploads", "pd-neuropathy");
const inferenceScript = path.join(serverRoot, "ml", "component3", "run_inference.py");

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

export const pdNeuropathyRouter = express.Router();

function numberOrNull(value) {
  if (value === undefined || value === null || value === "") return null;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function inferInputType(fileName, requestedType) {
  if (requestedType === "csv" || requestedType === "video") return requestedType;
  return path.extname(fileName).toLowerCase() === ".csv" ? "csv" : "video";
}

function normalizeModelKey(value) {
  return value === "neuropathy" ? "neuropathy" : "pd";
}

function modelLabel(modelKey) {
  return modelKey === "neuropathy" ? "Neuropathy" : "PD";
}

function runComponent3({ modelKey, filePath, inputType, direction, fps }) {
  return new Promise((resolve, reject) => {
    const args = [inferenceScript, "--model", modelKey, "--input", filePath, "--input-type", inputType];
    if (direction) args.push("--direction", direction);
    if (fps) args.push("--fps", String(fps));

    const child = spawn(config.pythonPath, args, {
      cwd: path.join(serverRoot, "ml", "component3"),
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
      const jsonLine = stdout.trim().split(/\r?\n/).reverse().find((line) => line.trim().startsWith("{"));
      try {
        payload = JSON.parse(jsonLine || stdout.trim());
      } catch {
        return reject(new Error(stderr || stdout || `Component 3 model exited with code ${code}.`));
      }

      if (code !== 0 || !payload.ok) {
        return reject(new Error(payload?.message || stderr || "Component 3 prediction failed."));
      }

      resolve(payload.result);
    });
  });
}

function summarizeResult(modelKey, result) {
  if (modelKey === "pd") {
    return {
      finalResult: result.final_result || null,
      detected: result.final_detected_as_pd ? 1 : 0,
      tendency: 0,
      probability: numberOrNull(result.median_pd_probability ?? result.aggregated_pd_probability),
      maxProbability: numberOrNull(result.maximum_window_probability),
      positiveWindowCount: numberOrNull(result.positive_window_count_at_threshold),
      positiveWindowRatio: numberOrNull(result.positive_window_ratio_at_threshold),
      reliabilityLevel: result.preprocessing?.status === "ok" ? "High" : result.preprocessing?.status || null,
      reliabilityReasons: result.preprocessing?.notes || [],
      clinicalNote: result.clinical_note || null,
      direction: result.direction || null,
      fpsUsed: numberOrNull(result.fps_used ?? result.fps),
      csvPath: result.csv_file || result.csv || null
    };
  }

  return {
    finalResult: result.final_result || null,
    detected: result.detected ? 1 : 0,
    tendency: result.borderline_tendency ? 1 : 0,
    probability: numberOrNull(result.neuropathic_probability),
    maxProbability: numberOrNull(result.max_probability),
    positiveWindowCount: numberOrNull(result.positive_window_count),
    positiveWindowRatio: numberOrNull(result.positive_window_ratio),
    reliabilityLevel: result.reliability || null,
    reliabilityReasons: result.reliability_reasons || [],
    clinicalNote: result.clinical_note || null,
    direction: result.direction || null,
    fpsUsed: numberOrNull(result.fps_used),
    csvPath: result.csv_file || null
  };
}

function publicScreening(row) {
  if (!row) return null;
  const isVideo = row.input_type === "video";
  return {
    id: row.id,
    modelKey: row.model_key,
    modelLabel: modelLabel(row.model_key),
    inputType: row.input_type,
    fileName: row.file_name,
    hasVideoPreview: isVideo,
    videoUrl: isVideo ? `/pd-neuropathy/screenings/${row.id}/file` : null,
    direction: row.direction,
    fpsUsed: row.fps_used,
    finalResult: row.final_result,
    detected: Boolean(row.detected),
    tendency: Boolean(row.tendency),
    probability: row.probability,
    maxProbability: row.max_probability,
    positiveWindowCount: row.positive_window_count,
    positiveWindowRatio: row.positive_window_ratio,
    reliabilityLevel: row.reliability_level,
    reliabilityReasons: JSON.parse(row.reliability_reasons || "[]"),
    clinicalNote: row.clinical_note,
    result: JSON.parse(row.raw_result_json || "{}"),
    createdAt: row.created_at
  };
}

function publicComponent3Profile(screenings = []) {
  return {
    totalScreenings: screenings.length,
    pdScreenings: screenings.filter((item) => item.model_key === "pd").length,
    neuropathyScreenings: screenings.filter((item) => item.model_key === "neuropathy").length,
    pdDetected: screenings.filter((item) => item.model_key === "pd" && item.detected).length,
    neuropathyDetected: screenings.filter((item) => item.model_key === "neuropathy" && item.detected).length,
    latestResult: screenings[0]?.final_result || null,
    updatedAt: screenings[0]?.created_at || null
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

function latestScreenings(userId) {
  return db.prepare(`
    SELECT * FROM pd_neuropathy_screenings
    WHERE user_id = ?
    ORDER BY created_at DESC
    LIMIT 40
  `).all(userId);
}

pdNeuropathyRouter.get("/clinical-profile", requireAuth, requireRole("patient"), (req, res) => {
  const screenings = latestScreenings(req.user.id);
  res.json({
    profile: publicComponent3Profile(screenings),
    screenings: screenings.map(publicScreening)
  });
});

pdNeuropathyRouter.get("/screenings/:id/file", requireAuth, requireRole("patient"), (req, res) => {
  const screening = db.prepare(`
    SELECT * FROM pd_neuropathy_screenings
    WHERE id = ? AND user_id = ? AND input_type = 'video'
  `).get(req.params.id, req.user.id);

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

pdNeuropathyRouter.get("/screenings/:id", requireAuth, requireRole("patient"), (req, res) => {
  const screening = db.prepare(`
    SELECT * FROM pd_neuropathy_screenings
    WHERE id = ? AND user_id = ?
  `).get(req.params.id, req.user.id);

  if (!screening) return res.status(404).json({ message: "Screening record not found." });
  res.json({ screening: publicScreening(screening) });
});

function deleteScreeningRecord(req, res) {
  const screening = db.prepare(`
    SELECT * FROM pd_neuropathy_screenings
    WHERE id = ? AND user_id = ?
  `).get(req.params.id, req.user.id);

  if (!screening) return res.status(404).json({ message: "Screening record not found." });

  const now = new Date().toISOString();
  db.transaction(() => {
    db.prepare("DELETE FROM pd_neuropathy_screenings WHERE id = ? AND user_id = ?").run(req.params.id, req.user.id);
    db.prepare("UPDATE clinical_profiles SET updated_at = ? WHERE user_id = ?").run(now, req.user.id);
  })();

  safeUnlinkStoredFile(screening.file_path);
  safeUnlinkStoredFile(screening.csv_path);

  const remaining = latestScreenings(req.user.id);
  res.json({
    message: "Component 3 screening record deleted.",
    profile: publicComponent3Profile(remaining),
    screenings: remaining.map(publicScreening)
  });
}

pdNeuropathyRouter.delete("/screenings/:id", requireAuth, requireRole("patient"), deleteScreeningRecord);
pdNeuropathyRouter.post("/screenings/:id/delete", requireAuth, requireRole("patient"), deleteScreeningRecord);

function clearScreeningRecords(req, res) {
  const modelKey = req.query.model ? normalizeModelKey(req.query.model) : null;
  const screenings = modelKey
    ? db.prepare("SELECT * FROM pd_neuropathy_screenings WHERE user_id = ? AND model_key = ?").all(req.user.id, modelKey)
    : db.prepare("SELECT * FROM pd_neuropathy_screenings WHERE user_id = ?").all(req.user.id);

  const now = new Date().toISOString();
  db.transaction(() => {
    if (modelKey) {
      db.prepare("DELETE FROM pd_neuropathy_screenings WHERE user_id = ? AND model_key = ?").run(req.user.id, modelKey);
    } else {
      db.prepare("DELETE FROM pd_neuropathy_screenings WHERE user_id = ?").run(req.user.id);
    }
    db.prepare("UPDATE clinical_profiles SET updated_at = ? WHERE user_id = ?").run(now, req.user.id);
  })();

  screenings.forEach((screening) => {
    safeUnlinkStoredFile(screening.file_path);
    safeUnlinkStoredFile(screening.csv_path);
  });

  const remaining = latestScreenings(req.user.id);
  res.json({
    message: "Component 3 screening records cleared.",
    profile: publicComponent3Profile(remaining),
    screenings: remaining.map(publicScreening)
  });
}

pdNeuropathyRouter.delete("/screenings", requireAuth, requireRole("patient"), clearScreeningRecords);
pdNeuropathyRouter.post("/screenings/clear", requireAuth, requireRole("patient"), clearScreeningRecords);

pdNeuropathyRouter.post(
  "/screenings",
  requireAuth,
  requireRole("patient"),
  upload.single("gaitFile"),
  async (req, res) => {
    if (!req.file) return res.status(422).json({ message: "Upload a gait video or CSV file." });

    const modelKey = normalizeModelKey(req.body.modelKey);
    const inputType = inferInputType(req.file.originalname, req.body.inputType);
    const direction = ["L2R", "R2L"].includes(req.body.direction) ? req.body.direction : null;
    const fps = numberOrNull(req.body.fps);

    try {
      const result = await runComponent3({
        modelKey,
        filePath: req.file.path,
        inputType,
        direction,
        fps
      });
      const summary = summarizeResult(modelKey, result);
      const now = new Date().toISOString();
      const profileId = createId("clin");
      const screeningId = createId("c3s");

      const save = db.transaction(() => {
        const existingProfile = db.prepare("SELECT id FROM clinical_profiles WHERE user_id = ?").get(req.user.id);
        const finalProfileId = existingProfile?.id || profileId;

        db.prepare(`
          INSERT INTO clinical_profiles (
            id, user_id, created_at, updated_at
          ) VALUES (?, ?, ?, ?)
          ON CONFLICT(user_id) DO UPDATE SET
            updated_at = excluded.updated_at
        `).run(finalProfileId, req.user.id, now, now);

        db.prepare(`
          INSERT INTO pd_neuropathy_screenings (
            id, user_id, clinical_profile_id, model_key, input_type, file_name, file_path,
            csv_path, direction, fps_used, final_result, detected, tendency, probability,
            max_probability, positive_window_count, positive_window_ratio, reliability_level,
            reliability_reasons, clinical_note, raw_result_json, created_at
          ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        `).run(
          screeningId,
          req.user.id,
          finalProfileId,
          modelKey,
          inputType,
          req.file.originalname,
          path.relative(serverRoot, req.file.path),
          summary.csvPath,
          summary.direction || direction,
          summary.fpsUsed || fps,
          summary.finalResult,
          summary.detected,
          summary.tendency,
          summary.probability,
          summary.maxProbability,
          summary.positiveWindowCount,
          summary.positiveWindowRatio,
          summary.reliabilityLevel,
          JSON.stringify(summary.reliabilityReasons || []),
          summary.clinicalNote,
          JSON.stringify(result),
          now
        );

        return db.prepare("SELECT * FROM pd_neuropathy_screenings WHERE id = ?").get(screeningId);
      });

      const saved = save();
      const allScreenings = latestScreenings(req.user.id);

      res.status(201).json({
        message: `${modelLabel(modelKey)} screening completed and saved.`,
        screening: publicScreening(saved),
        profile: publicComponent3Profile(allScreenings)
      });
    } catch (error) {
      res.status(500).json({
        message: error.message || `${modelLabel(modelKey)} detection failed.`
      });
    }
  }
);
