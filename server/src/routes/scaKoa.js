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
const uploadRoot = path.join(serverRoot, "data", "uploads", "sca-koa");
const inferenceScript = path.join(serverRoot, "ml", "component2", "run_inference.py");

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

export const scaKoaRouter = express.Router();

function numberOrNull(value) {
  if (value === undefined || value === null || value === "") return null;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function inferInputType(fileName, requestedType) {
  if (requestedType === "csv" || requestedType === "video") return requestedType;
  return path.extname(fileName).toLowerCase() === ".csv" ? "csv" : "video";
}

function modelLabel(modelKey) {
  return modelKey === "koa" ? "KOA" : "SCA";
}

function normalizeModelKey(value) {
  return value === "koa" ? "koa" : "sca";
}

function runComponent2({ modelKey, filePath, inputType, direction, fps }) {
  return new Promise((resolve, reject) => {
    const args = [inferenceScript, "--model", modelKey, "--input", filePath, "--input-type", inputType];
    if (direction) args.push("--direction", direction);
    if (fps) args.push("--fps", String(fps));

    const child = spawn(config.pythonPath, args, {
      cwd: path.join(serverRoot, "ml", "component2"),
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
        return reject(new Error(stderr || stdout || `Component 2 model exited with code ${code}.`));
      }

      if (code !== 0 || !payload.ok) {
        return reject(new Error(payload?.message || stderr || "Component 2 prediction failed."));
      }

      resolve(payload.result);
    });
  });
}

function summarizeResult(modelKey, result) {
  if (modelKey === "koa") {
    return {
      finalResult: result.final_koa_result || null,
      detected: result.koa_detected ? 1 : 0,
      tendency: result.koa_borderline_tendency ? 1 : 0,
      probability: numberOrNull(result.koa_probability),
      maxProbability: numberOrNull(result.koa_max_probability),
      positiveWindowCount: numberOrNull(result.koa_positive_window_count),
      positiveWindowRatio: numberOrNull(result.koa_window_ratio_detection),
      patternStrength: result.koa_pattern_strength || null,
      reliabilityLevel: result.reliability_level || null,
      reliabilityReasons: result.reliability_reasons || [],
      clinicalNote: result.clinical_note || null,
      direction: result.direction || null,
      fpsUsed: numberOrNull(result.fps_used),
      csvPath: result.csv_file || null
    };
  }

  return {
    finalResult: result.final_result || null,
    detected: result.sca_detected ? 1 : 0,
    tendency: result.sca_tendency ? 1 : 0,
    probability: numberOrNull(result.mean_probability),
    maxProbability: numberOrNull(result.max_probability),
    positiveWindowCount: numberOrNull(result.positive_count),
    positiveWindowRatio: numberOrNull(result.positive_ratio),
    patternStrength: result.pattern_strength || null,
    reliabilityLevel: result.reliability || null,
    reliabilityReasons: result.reliability_note ? [result.reliability_note] : [],
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
    videoUrl: isVideo ? `/sca-koa/screenings/${row.id}/file` : null,
    direction: row.direction,
    fpsUsed: row.fps_used,
    finalResult: row.final_result,
    detected: Boolean(row.detected),
    tendency: Boolean(row.tendency),
    probability: row.probability,
    maxProbability: row.max_probability,
    positiveWindowCount: row.positive_window_count,
    positiveWindowRatio: row.positive_window_ratio,
    patternStrength: row.pattern_strength,
    reliabilityLevel: row.reliability_level,
    reliabilityReasons: JSON.parse(row.reliability_reasons || "[]"),
    clinicalNote: row.clinical_note,
    instabilityMap: JSON.parse(row.instability_json || "null"),
    result: JSON.parse(row.raw_result_json || "{}"),
    createdAt: row.created_at
  };
}

function relationWeight(relation = "") {
  const text = String(relation).toLowerCase();
  if (["mother", "father", "sister", "brother", "daughter", "son", "child"].some((item) => text.includes(item))) return 32;
  if (["grandmother", "grandfather", "aunt", "uncle", "niece", "nephew", "half"].some((item) => text.includes(item))) return 22;
  if (["cousin", "great"].some((item) => text.includes(item))) return 12;
  return 8;
}

function normalizeArray(value) {
  return Array.isArray(value) ? value : [];
}

function cleanRelative(item = {}) {
  return {
    id: item.id || createId("rel"),
    relation: String(item.relation || "").trim(),
    familySide: String(item.familySide || "unknown").trim(),
    scaStatus: String(item.scaStatus || "unknown").trim(),
    geneticConfirmed: Boolean(item.geneticConfirmed),
    ageOfOnset: item.ageOfOnset ? String(item.ageOfOnset).trim() : "",
    gaitNotes: String(item.gaitNotes || "").trim()
  };
}

function cleanSuspectedRelative(item = {}) {
  return {
    id: item.id || createId("sus"),
    relation: String(item.relation || "").trim(),
    familySide: String(item.familySide || "unknown").trim(),
    gaitPattern: String(item.gaitPattern || "").trim(),
    ageNoticed: item.ageNoticed ? String(item.ageNoticed).trim() : "",
    notes: String(item.notes || "").trim()
  };
}

function buildGeneticAwareness({ answers = {}, relatives = [], suspected = [] }) {
  const cleanedRelatives = normalizeArray(relatives).map(cleanRelative).filter((item) => item.relation || item.scaStatus !== "unknown");
  const cleanedSuspected = normalizeArray(suspected).map(cleanSuspectedRelative).filter((item) => item.relation || item.gaitPattern);
  let score = 0;

  cleanedRelatives.forEach((relative) => {
    const status = relative.scaStatus.toLowerCase();
    if (status === "diagnosed" || status === "positive_genetic_test") score += relationWeight(relative.relation);
    else if (status === "suspected") score += relationWeight(relative.relation) * 0.55;
    if (relative.geneticConfirmed) score += 12;
    const onset = Number(relative.ageOfOnset);
    if (Number.isFinite(onset) && onset > 0 && onset < 45) score += 8;
  });

  cleanedSuspected.forEach((relative) => {
    score += Math.max(6, relationWeight(relative.relation) * 0.35);
  });

  if (answers.knownFamilySca === "yes") score += 16;
  if (answers.familyAbnormalGait === "yes") score += 10;
  if (answers.geneticTesting === "positive") score += 24;
  if (answers.geneticTesting === "negative") score -= 10;

  const awarenessScore = Math.max(0, Math.min(100, Math.round(score)));
  const level = awarenessScore >= 70 ? "High family-history awareness" : awarenessScore >= 35 ? "Moderate family-history awareness" : "Low documented family-history awareness";
  const primaryNotes = [];
  if (cleanedRelatives.some((item) => item.geneticConfirmed)) primaryNotes.push("At least one relative is marked genetically confirmed.");
  if (cleanedRelatives.some((item) => ["mother", "father", "sister", "brother", "son", "daughter"].includes(item.relation.toLowerCase()))) primaryNotes.push("A first-degree relative is included in the family-history chart.");
  if (cleanedSuspected.length > 0) primaryNotes.push("The chart includes relatives with suspected abnormal gait behavior.");
  if (primaryNotes.length === 0) primaryNotes.push("No strong hereditary SCA family-history signal has been entered yet.");

  return {
    awarenessScore,
    level,
    notes: primaryNotes,
    relatives: cleanedRelatives,
    suspected: cleanedSuspected,
    chart: {
      root: {
        id: "patient",
        label: "Patient",
        relation: "Self",
        score: awarenessScore,
        status: level
      },
      diagnosed: cleanedRelatives.map((item) => ({
        id: item.id,
        label: item.relation || "Relative",
        side: item.familySide,
        status: item.scaStatus,
        geneticConfirmed: item.geneticConfirmed,
        ageOfOnset: item.ageOfOnset,
        gaitNotes: item.gaitNotes
      })),
      suspected: cleanedSuspected.map((item) => ({
        id: item.id,
        label: item.relation || "Relative",
        side: item.familySide,
        gaitPattern: item.gaitPattern,
        ageNoticed: item.ageNoticed,
        notes: item.notes
      }))
    }
  };
}

function publicGenetics(row) {
  if (!row) return null;
  return {
    id: row.id,
    screeningId: row.screening_id,
    answers: JSON.parse(row.answers_json || "{}"),
    relatives: JSON.parse(row.relatives_json || "[]"),
    suspected: JSON.parse(row.suspected_json || "[]"),
    awareness: JSON.parse(row.awareness_json || "{}"),
    createdAt: row.created_at,
    updatedAt: row.updated_at
  };
}

function publicComponent2Profile(screenings = []) {
  return {
    totalScreenings: screenings.length,
    scaScreenings: screenings.filter((item) => item.model_key === "sca").length,
    koaScreenings: screenings.filter((item) => item.model_key === "koa").length,
    scaDetected: screenings.filter((item) => item.model_key === "sca" && item.detected).length,
    koaDetected: screenings.filter((item) => item.model_key === "koa" && item.detected).length,
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
    SELECT * FROM sca_koa_screenings
    WHERE user_id = ?
    ORDER BY created_at DESC
    LIMIT 40
  `).all(userId);
}

scaKoaRouter.get("/clinical-profile", requireAuth, requireRole("patient"), (req, res) => {
  const screenings = latestScreenings(req.user.id);
  res.json({
    profile: publicComponent2Profile(screenings),
    screenings: screenings.map(publicScreening)
  });
});

scaKoaRouter.get("/screenings/:id/file", requireAuth, requireRole("patient"), (req, res) => {
  const screening = db.prepare(`
    SELECT * FROM sca_koa_screenings
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

scaKoaRouter.get("/screenings/:id", requireAuth, requireRole("patient"), (req, res) => {
  const screening = db.prepare(`
    SELECT * FROM sca_koa_screenings
    WHERE id = ? AND user_id = ?
  `).get(req.params.id, req.user.id);

  if (!screening) return res.status(404).json({ message: "Screening record not found." });
  res.json({ screening: publicScreening(screening) });
});

scaKoaRouter.get("/screenings/:id/genetics", requireAuth, requireRole("patient"), (req, res) => {
  const screening = db.prepare(`
    SELECT * FROM sca_koa_screenings
    WHERE id = ? AND user_id = ? AND model_key = 'sca'
  `).get(req.params.id, req.user.id);

  if (!screening) return res.status(404).json({ message: "SCA screening record not found." });

  const row = db.prepare(`
    SELECT * FROM sca_genetic_awareness
    WHERE screening_id = ? AND user_id = ?
  `).get(req.params.id, req.user.id);

  res.json({ genetics: publicGenetics(row) });
});

scaKoaRouter.put("/screenings/:id/genetics", requireAuth, requireRole("patient"), (req, res) => {
  const screening = db.prepare(`
    SELECT * FROM sca_koa_screenings
    WHERE id = ? AND user_id = ? AND model_key = 'sca'
  `).get(req.params.id, req.user.id);

  if (!screening) return res.status(404).json({ message: "SCA screening record not found." });
  if (!screening.detected && !screening.tendency) {
    return res.status(422).json({ message: "Genetic awareness chart is available after an SCA detection or borderline SCA tendency." });
  }

  const answers = {
    knownFamilySca: req.body?.answers?.knownFamilySca || "unknown",
    familyAbnormalGait: req.body?.answers?.familyAbnormalGait || "unknown",
    geneticTesting: req.body?.answers?.geneticTesting || "not_tested",
    notes: String(req.body?.answers?.notes || "").trim()
  };
  const relatives = normalizeArray(req.body?.relatives).map(cleanRelative);
  const suspected = normalizeArray(req.body?.suspected).map(cleanSuspectedRelative);
  const awareness = buildGeneticAwareness({ answers, relatives, suspected });
  const now = new Date().toISOString();
  const existing = db.prepare(`
    SELECT id FROM sca_genetic_awareness
    WHERE screening_id = ? AND user_id = ?
  `).get(req.params.id, req.user.id);
  const id = existing?.id || createId("gen");

  db.prepare(`
    INSERT INTO sca_genetic_awareness (
      id, user_id, screening_id, answers_json, relatives_json, suspected_json,
      awareness_json, created_at, updated_at
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(screening_id) DO UPDATE SET
      answers_json = excluded.answers_json,
      relatives_json = excluded.relatives_json,
      suspected_json = excluded.suspected_json,
      awareness_json = excluded.awareness_json,
      updated_at = excluded.updated_at
  `).run(
    id,
    req.user.id,
    req.params.id,
    JSON.stringify(answers),
    JSON.stringify(relatives),
    JSON.stringify(suspected),
    JSON.stringify(awareness),
    now,
    now
  );

  const row = db.prepare(`
    SELECT * FROM sca_genetic_awareness
    WHERE screening_id = ? AND user_id = ?
  `).get(req.params.id, req.user.id);

  res.json({
    message: "SCA genetic awareness chart saved.",
    genetics: publicGenetics(row)
  });
});

function deleteScreeningRecord(req, res) {
  const screening = db.prepare(`
    SELECT * FROM sca_koa_screenings
    WHERE id = ? AND user_id = ?
  `).get(req.params.id, req.user.id);

  if (!screening) return res.status(404).json({ message: "Screening record not found." });

  const now = new Date().toISOString();
  db.transaction(() => {
    db.prepare("DELETE FROM sca_koa_screenings WHERE id = ? AND user_id = ?").run(req.params.id, req.user.id);
    db.prepare("UPDATE clinical_profiles SET updated_at = ? WHERE user_id = ?").run(now, req.user.id);
  })();

  safeUnlinkStoredFile(screening.file_path);
  safeUnlinkStoredFile(screening.csv_path);

  const remaining = latestScreenings(req.user.id);
  res.json({
    message: "Component 2 screening record deleted.",
    profile: publicComponent2Profile(remaining),
    screenings: remaining.map(publicScreening)
  });
}

scaKoaRouter.delete("/screenings/:id", requireAuth, requireRole("patient"), deleteScreeningRecord);
scaKoaRouter.post("/screenings/:id/delete", requireAuth, requireRole("patient"), deleteScreeningRecord);

function clearScreeningRecords(req, res) {
  const modelKey = req.query.model ? normalizeModelKey(req.query.model) : null;
  const screenings = modelKey
    ? db.prepare("SELECT * FROM sca_koa_screenings WHERE user_id = ? AND model_key = ?").all(req.user.id, modelKey)
    : db.prepare("SELECT * FROM sca_koa_screenings WHERE user_id = ?").all(req.user.id);

  const now = new Date().toISOString();
  db.transaction(() => {
    if (modelKey) {
      db.prepare("DELETE FROM sca_koa_screenings WHERE user_id = ? AND model_key = ?").run(req.user.id, modelKey);
    } else {
      db.prepare("DELETE FROM sca_koa_screenings WHERE user_id = ?").run(req.user.id);
    }
    db.prepare("UPDATE clinical_profiles SET updated_at = ? WHERE user_id = ?").run(now, req.user.id);
  })();

  screenings.forEach((screening) => {
    safeUnlinkStoredFile(screening.file_path);
    safeUnlinkStoredFile(screening.csv_path);
  });

  const remaining = latestScreenings(req.user.id);
  res.json({
    message: "Component 2 screening records cleared.",
    profile: publicComponent2Profile(remaining),
    screenings: remaining.map(publicScreening)
  });
}

scaKoaRouter.delete("/screenings", requireAuth, requireRole("patient"), clearScreeningRecords);
scaKoaRouter.post("/screenings/clear", requireAuth, requireRole("patient"), clearScreeningRecords);

scaKoaRouter.post(
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
      const result = await runComponent2({
        modelKey,
        filePath: req.file.path,
        inputType,
        direction,
        fps
      });
      const summary = summarizeResult(modelKey, result);
      const now = new Date().toISOString();
      const profileId = createId("clin");
      const screeningId = createId("c2s");

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
          INSERT INTO sca_koa_screenings (
            id, user_id, clinical_profile_id, model_key, input_type, file_name, file_path,
            csv_path, direction, fps_used, final_result, detected, tendency, probability,
            max_probability, positive_window_count, positive_window_ratio, pattern_strength,
            reliability_level, reliability_reasons, clinical_note, instability_json, raw_result_json, created_at
          ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
          summary.patternStrength,
          summary.reliabilityLevel,
          JSON.stringify(summary.reliabilityReasons || []),
          summary.clinicalNote,
          JSON.stringify(result.instability_map || null),
          JSON.stringify(result),
          now
        );

        return db.prepare("SELECT * FROM sca_koa_screenings WHERE id = ?").get(screeningId);
      });

      const saved = save();
      const allScreenings = latestScreenings(req.user.id);

      res.status(201).json({
        message: `${modelLabel(modelKey)} screening completed and saved.`,
        screening: publicScreening(saved),
        profile: publicComponent2Profile(allScreenings)
      });
    } catch (error) {
      res.status(500).json({
        message: error.message || `${modelLabel(modelKey)} detection failed.`
      });
    }
  }
);
