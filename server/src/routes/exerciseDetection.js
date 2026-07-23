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
const uploadRoot = path.join(serverRoot, "data", "uploads", "exercise-detection");
const inferenceScript = path.join(serverRoot, "ml", "component4", "run_inference.py");

fs.mkdirSync(uploadRoot, { recursive: true });

const exerciseLabels = {
  gesture3: "Seated left-arm forward raise",
  gesture5: "Seated left-arm lateral raise",
  gesture2: "Seated right-arm forward raise"
};

function safeFileName(name = "upload") {
  const ext = path.extname(name).toLowerCase();
  const stem = path.basename(name, ext).replace(/[^a-zA-Z0-9_-]/g, "_").slice(0, 80) || "exercise_upload";
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
    if ([".mp4", ".mov", ".avi", ".mkv"].includes(ext)) return cb(null, true);
    return cb(new Error("Upload an exercise video file."));
  }
});

export const exerciseDetectionRouter = express.Router();

function numberOrNull(value) {
  if (value === undefined || value === null || value === "") return null;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function normalizeExerciseKey(value) {
  return Object.prototype.hasOwnProperty.call(exerciseLabels, value) ? value : "gesture3";
}

function runExerciseInference({ exerciseKey, filePath }) {
  return new Promise((resolve, reject) => {
    const args = [inferenceScript, "--exercise", exerciseKey, "--input", filePath, "--save-annotated"];
    const child = spawn(config.pythonPath, args, {
      cwd: path.join(serverRoot, "ml", "component4"),
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
        return reject(new Error(stderr || stdout || `Exercise model exited with code ${code}.`));
      }

      if (code !== 0 || !payload.ok) {
        return reject(new Error(payload?.message || stderr || "Exercise detection failed."));
      }

      resolve(payload.result);
    });
  });
}

function publicScreening(row) {
  if (!row) return null;
  return {
    id: row.id,
    exerciseKey: row.exercise_key,
    exerciseLabel: row.exercise_label,
    fileName: row.file_name,
    videoUrl: `/exercise-detection/screenings/${row.id}/file`,
    annotatedVideoUrl: row.annotated_path ? `/exercise-detection/screenings/${row.id}/annotated` : null,
    finalPrediction: row.final_prediction,
    finalLabel: row.final_label,
    isCorrect: row.final_label === 1,
    qualityScore: row.quality_score,
    meanCorrectProbability: row.mean_correct_probability,
    decisionThreshold: row.decision_threshold,
    correctRatio: row.correct_ratio,
    incorrectRatio: row.incorrect_ratio,
    numWindows: row.num_windows,
    validPoseFrames: row.valid_pose_frames,
    reliabilityLevel: row.reliability_level,
    reliabilityReasons: JSON.parse(row.reliability_reasons || "[]"),
    windowReport: JSON.parse(row.window_report_json || "{}"),
    result: JSON.parse(row.raw_result_json || "{}"),
    createdAt: row.created_at
  };
}

function publicProfile(screenings = []) {
  return {
    totalScreenings: screenings.length,
    correctScreenings: screenings.filter((item) => item.final_label === 1).length,
    incorrectScreenings: screenings.filter((item) => item.final_label === 0).length,
    latestResult: screenings[0]?.final_prediction || null,
    updatedAt: screenings[0]?.created_at || null,
    byExercise: Object.fromEntries(
      Object.keys(exerciseLabels).map((key) => [
        key,
        {
          label: exerciseLabels[key],
          total: screenings.filter((item) => item.exercise_key === key).length,
          correct: screenings.filter((item) => item.exercise_key === key && item.final_label === 1).length,
          incorrect: screenings.filter((item) => item.exercise_key === key && item.final_label === 0).length
        }
      ])
    )
  };
}

function latestScreenings(userId) {
  return db.prepare(`
    SELECT * FROM exercise_screenings
    WHERE user_id = ?
    ORDER BY created_at DESC
    LIMIT 60
  `).all(userId);
}

function safeResolveStoredFile(storedPath) {
  if (!storedPath) return null;
  const absolutePath = path.isAbsolute(storedPath)
    ? path.resolve(storedPath)
    : path.resolve(serverRoot, storedPath);
  if (!absolutePath.startsWith(serverRoot) || !fs.existsSync(absolutePath)) return null;
  return absolutePath;
}

function safeUnlinkStoredFile(storedPath) {
  const absolutePath = safeResolveStoredFile(storedPath);
  if (!absolutePath) return;
  fs.promises.unlink(absolutePath).catch(() => {});
}

exerciseDetectionRouter.get("/clinical-profile", requireAuth, requireRole("patient"), (req, res) => {
  const screenings = latestScreenings(req.user.id);
  res.json({
    profile: publicProfile(screenings),
    screenings: screenings.map(publicScreening)
  });
});

exerciseDetectionRouter.get("/screenings/:id", requireAuth, requireRole("patient"), (req, res) => {
  const screening = db.prepare(`
    SELECT * FROM exercise_screenings
    WHERE id = ? AND user_id = ?
  `).get(req.params.id, req.user.id);

  if (!screening) return res.status(404).json({ message: "Exercise screening record not found." });
  res.json({ screening: publicScreening(screening) });
});

function sendStoredVideo(req, res, column) {
  const screening = db.prepare(`
    SELECT * FROM exercise_screenings
    WHERE id = ? AND user_id = ?
  `).get(req.params.id, req.user.id);

  if (!screening) return res.status(404).json({ message: "Exercise video not found." });
  const absolutePath = safeResolveStoredFile(screening[column]);
  if (!absolutePath) return res.status(404).json({ message: "Exercise video file not found." });

  res.setHeader("Cross-Origin-Resource-Policy", "cross-origin");
  res.setHeader("Access-Control-Allow-Origin", config.clientOrigin);
  res.setHeader("Access-Control-Allow-Credentials", "true");
  res.setHeader("Content-Disposition", `inline; filename="${safeFileName(path.basename(absolutePath))}"`);
  res.sendFile(absolutePath);
}

exerciseDetectionRouter.get("/screenings/:id/file", requireAuth, requireRole("patient"), (req, res) => {
  sendStoredVideo(req, res, "file_path");
});

exerciseDetectionRouter.get("/screenings/:id/annotated", requireAuth, requireRole("patient"), (req, res) => {
  sendStoredVideo(req, res, "annotated_path");
});

function deleteScreeningRecord(req, res) {
  const screening = db.prepare(`
    SELECT * FROM exercise_screenings
    WHERE id = ? AND user_id = ?
  `).get(req.params.id, req.user.id);

  if (!screening) return res.status(404).json({ message: "Exercise screening record not found." });

  const now = new Date().toISOString();
  db.transaction(() => {
    db.prepare("DELETE FROM exercise_screenings WHERE id = ? AND user_id = ?").run(req.params.id, req.user.id);
    db.prepare("UPDATE clinical_profiles SET updated_at = ? WHERE user_id = ?").run(now, req.user.id);
  })();

  safeUnlinkStoredFile(screening.file_path);
  safeUnlinkStoredFile(screening.annotated_path);

  const remaining = latestScreenings(req.user.id);
  res.json({
    message: "Exercise screening record deleted.",
    profile: publicProfile(remaining),
    screenings: remaining.map(publicScreening)
  });
}

exerciseDetectionRouter.delete("/screenings/:id", requireAuth, requireRole("patient"), deleteScreeningRecord);
exerciseDetectionRouter.post("/screenings/:id/delete", requireAuth, requireRole("patient"), deleteScreeningRecord);

function clearScreeningRecords(req, res) {
  const exerciseKey = req.query.exercise ? normalizeExerciseKey(req.query.exercise) : null;
  const screenings = exerciseKey
    ? db.prepare("SELECT * FROM exercise_screenings WHERE user_id = ? AND exercise_key = ?").all(req.user.id, exerciseKey)
    : db.prepare("SELECT * FROM exercise_screenings WHERE user_id = ?").all(req.user.id);

  const now = new Date().toISOString();
  db.transaction(() => {
    if (exerciseKey) {
      db.prepare("DELETE FROM exercise_screenings WHERE user_id = ? AND exercise_key = ?").run(req.user.id, exerciseKey);
    } else {
      db.prepare("DELETE FROM exercise_screenings WHERE user_id = ?").run(req.user.id);
    }
    db.prepare("UPDATE clinical_profiles SET updated_at = ? WHERE user_id = ?").run(now, req.user.id);
  })();

  screenings.forEach((screening) => {
    safeUnlinkStoredFile(screening.file_path);
    safeUnlinkStoredFile(screening.annotated_path);
  });

  const remaining = latestScreenings(req.user.id);
  res.json({
    message: "Exercise screening records cleared.",
    profile: publicProfile(remaining),
    screenings: remaining.map(publicScreening)
  });
}

exerciseDetectionRouter.delete("/screenings", requireAuth, requireRole("patient"), clearScreeningRecords);
exerciseDetectionRouter.post("/screenings/clear", requireAuth, requireRole("patient"), clearScreeningRecords);

exerciseDetectionRouter.post(
  "/screenings",
  requireAuth,
  requireRole("patient"),
  upload.single("exerciseFile"),
  async (req, res) => {
    if (!req.file) return res.status(422).json({ message: "Upload an exercise video file." });

    const exerciseKey = normalizeExerciseKey(req.body.exerciseKey);

    try {
      const result = await runExerciseInference({
        exerciseKey,
        filePath: req.file.path
      });

      const now = new Date().toISOString();
      const profileId = createId("clin");
      const screeningId = createId("exs");

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
          INSERT INTO exercise_screenings (
            id, user_id, clinical_profile_id, exercise_key, exercise_label, file_name,
            file_path, annotated_path, final_prediction, final_label, quality_score,
            mean_correct_probability, decision_threshold, correct_ratio, incorrect_ratio,
            num_windows, valid_pose_frames, reliability_level, reliability_reasons,
            window_report_json, raw_result_json, created_at
          ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        `).run(
          screeningId,
          req.user.id,
          finalProfileId,
          exerciseKey,
          result.exercise_label || exerciseLabels[exerciseKey],
          req.file.originalname,
          path.relative(serverRoot, req.file.path),
          result.annotated_video_path || null,
          result.final_prediction || null,
          numberOrNull(result.final_label),
          numberOrNull(result.quality_score),
          numberOrNull(result.mean_correct_probability),
          numberOrNull(result.decision_threshold),
          numberOrNull(result.correct_ratio),
          numberOrNull(result.incorrect_ratio),
          numberOrNull(result.num_windows),
          numberOrNull(result.valid_pose_frames),
          result.reliability || null,
          JSON.stringify(result.reliability_notes || []),
          JSON.stringify(result.window_report || {}),
          JSON.stringify(result),
          now
        );

        return db.prepare("SELECT * FROM exercise_screenings WHERE id = ?").get(screeningId);
      });

      const saved = save();
      const allScreenings = latestScreenings(req.user.id);

      res.status(201).json({
        message: `${exerciseLabels[exerciseKey]} screening completed and saved.`,
        screening: publicScreening(saved),
        profile: publicProfile(allScreenings)
      });
    } catch (error) {
      res.status(500).json({
        message: error.message || "Exercise detection failed."
      });
    }
  }
);
