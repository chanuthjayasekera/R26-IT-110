import express from "express";
import fs from "fs";
import multer from "multer";
import path from "path";
import { fileURLToPath } from "url";
import { db } from "../db.js";
import { requireAuth, requireRole, requireVerifiedProfessional } from "../middleware/auth.js";
import { createId } from "../utils/security.js";

export const centralProfileRouter = express.Router();

const dirname = path.dirname(fileURLToPath(import.meta.url));
const serverRoot = path.resolve(dirname, "..", "..");
const guidanceUploadRoot = path.join(serverRoot, "data", "uploads", "central-guidance");
fs.mkdirSync(guidanceUploadRoot, { recursive: true });

function safeFileName(name = "guidance-attachment") {
  const ext = path.extname(name).toLowerCase();
  const stem = path.basename(name, ext).replace(/[^a-zA-Z0-9_-]/g, "_").slice(0, 80) || "guidance_attachment";
  return `${stem}${ext}`;
}

const allowedGuidanceExtensions = new Set([
  ".mp4",
  ".mov",
  ".avi",
  ".mkv",
  ".webm",
  ".pdf",
  ".doc",
  ".docx",
  ".txt",
  ".rtf",
  ".png",
  ".jpg",
  ".jpeg"
]);

const guidanceUpload = multer({
  storage: multer.diskStorage({
    destination: (req, file, cb) => cb(null, guidanceUploadRoot),
    filename: (req, file, cb) => cb(null, `${Date.now()}_${createId("guide")}_${safeFileName(file.originalname)}`)
  }),
  limits: { fileSize: 300 * 1024 * 1024 },
  fileFilter: (req, file, cb) => {
    const ext = path.extname(file.originalname).toLowerCase();
    if (allowedGuidanceExtensions.has(ext)) return cb(null, true);
    return cb(new Error("Upload a supported guidance attachment: video, PDF, Word, text, or image file."));
  }
});

function uploadGuidanceAttachment(req, res, next) {
  guidanceUpload.single("attachment")(req, res, (err) => {
    if (!err) return next();
    const message = err instanceof multer.MulterError && err.code === "LIMIT_FILE_SIZE"
      ? "Attachment is too large. Upload a file under 300 MB."
      : err.message || "Unable to upload this guidance attachment.";
    return res.status(422).json({ message });
  });
}

function safeUnlinkGuidanceFile(storedPath) {
  if (!storedPath) return;
  const absolutePath = path.resolve(storedPath);
  if (!absolutePath.startsWith(guidanceUploadRoot)) return;
  try {
    if (fs.existsSync(absolutePath)) fs.unlinkSync(absolutePath);
  } catch {
    // Best-effort cleanup; the database row is the source of truth.
  }
}

const sourceLabels = {
  normal_abnormal: "Normal vs Abnormal",
  sca: "SCA",
  koa: "KOA",
  pd: "PD",
  neuropathy: "Neuropathy",
  exercise_gesture2: "Right-arm forward raise",
  exercise_gesture3: "Left-arm forward raise",
  exercise_gesture5: "Left-arm lateral raise"
};

const sourceOrder = [
  "normal_abnormal",
  "sca",
  "koa",
  "pd",
  "neuropathy",
  "exercise_gesture2",
  "exercise_gesture3",
  "exercise_gesture5"
];

function parseJson(value, fallback) {
  try {
    return value ? JSON.parse(value) : fallback;
  } catch {
    return fallback;
  }
}

function percentLike(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return null;
  return Math.round(Number(value) * 100);
}

function probability(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return 0;
  return Math.max(0, Math.min(1, Number(value)));
}

function getSourceType(body) {
  const sourceType = String(body?.sourceType || "").trim();
  return Object.prototype.hasOwnProperty.call(sourceLabels, sourceType) ? sourceType : "";
}

async function getScreeningForSource(userId, sourceType, screeningId) {
  if (sourceType === "normal_abnormal") {
    const row = await db.get(`
      SELECT * FROM normal_abnormal_screenings
      WHERE user_id = ? AND id = ?
    `, userId, screeningId);
    if (!row) return null;
    return {
      sourceType,
      screeningId: row.id,
      title: "Component 1 Screening",
      subtitle: row.file_name,
      resultLabel: row.final_result || "Inconclusive",
      modelLabel: sourceLabels[sourceType],
      createdAt: row.created_at,
      metrics: {
        confidence: row.confidence_percent,
        probability: row.mean_prob_abnormal === null ? null : Math.round(Number(row.mean_prob_abnormal) * 100),
        reliability: row.reliability_level,
        direction: row.direction,
        severity: row.screening_severity
      },
      details: {
        id: row.id,
        fileName: row.file_name,
        inputType: row.input_type,
        finalResult: row.final_result,
        finalLabel: row.final_label,
        meanProbAbnormal: row.mean_prob_abnormal,
        confidencePercent: row.confidence_percent,
        screeningSeverity: row.screening_severity,
        reliabilityLevel: row.reliability_level,
        reliabilityReasons: parseJson(row.reliability_reasons, []),
        clinicalNote: row.clinical_note,
        biometrics: parseJson(row.raw_result_json, {})?.biometrics || null
      }
    };
  }

  if (sourceType === "sca" || sourceType === "koa") {
    const row = await db.get(`
      SELECT * FROM sca_koa_screenings
      WHERE user_id = ? AND id = ? AND model_key = ?
    `, userId, screeningId, sourceType);
    if (!row) return null;
    const genetics = sourceType === "sca"
      ? await db.get("SELECT * FROM sca_genetic_awareness WHERE user_id = ? AND screening_id = ?", userId, screeningId)
      : null;
    return {
      sourceType,
      screeningId: row.id,
      title: `${sourceLabels[sourceType]} Gait Screening`,
      subtitle: row.file_name,
      resultLabel: row.final_result || (row.detected ? "Detected" : row.tendency ? "Borderline" : "Not detected"),
      modelLabel: sourceLabels[sourceType],
      createdAt: row.created_at,
      metrics: {
        probability: percentLike(row.probability),
        maxProbability: percentLike(row.max_probability),
        reliability: row.reliability_level,
        direction: row.direction,
        positiveWindows: row.positive_window_count
      },
      details: {
        id: row.id,
        fileName: row.file_name,
        finalResult: row.final_result,
        detected: Boolean(row.detected),
        tendency: Boolean(row.tendency),
        probability: row.probability,
        maxProbability: row.max_probability,
        reliabilityLevel: row.reliability_level,
        reliabilityReasons: parseJson(row.reliability_reasons, []),
        clinicalNote: row.clinical_note,
        instabilityMap: parseJson(row.instability_json, null),
        genetics: genetics ? {
          answers: parseJson(genetics.answers_json, {}),
          relatives: parseJson(genetics.relatives_json, []),
          suspected: parseJson(genetics.suspected_json, []),
          awareness: parseJson(genetics.awareness_json, {})
        } : null
      }
    };
  }

  if (sourceType === "pd" || sourceType === "neuropathy") {
    const row = await db.get(`
      SELECT * FROM pd_neuropathy_screenings
      WHERE user_id = ? AND id = ? AND model_key = ?
    `, userId, screeningId, sourceType);
    if (!row) return null;
    return {
      sourceType,
      screeningId: row.id,
      title: `${sourceLabels[sourceType]} Gait Screening`,
      subtitle: row.file_name,
      resultLabel: row.final_result || (row.detected ? "Detected" : row.tendency ? "Borderline" : "Not detected"),
      modelLabel: sourceLabels[sourceType],
      createdAt: row.created_at,
      metrics: {
        probability: percentLike(row.probability),
        maxProbability: percentLike(row.max_probability),
        reliability: row.reliability_level,
        direction: row.direction,
        positiveWindows: row.positive_window_count
      },
      details: {
        id: row.id,
        fileName: row.file_name,
        finalResult: row.final_result,
        detected: Boolean(row.detected),
        tendency: Boolean(row.tendency),
        probability: row.probability,
        maxProbability: row.max_probability,
        reliabilityLevel: row.reliability_level,
        reliabilityReasons: parseJson(row.reliability_reasons, []),
        clinicalNote: row.clinical_note
      }
    };
  }

  if (sourceType.startsWith("exercise_")) {
    const exerciseKey = sourceType.replace("exercise_", "");
    const row = await db.get(`
      SELECT * FROM exercise_screenings
      WHERE user_id = ? AND id = ? AND exercise_key = ?
    `, userId, screeningId, exerciseKey);
    if (!row) return null;
    return {
      sourceType,
      screeningId: row.id,
      title: row.exercise_label,
      subtitle: row.file_name,
      resultLabel: row.final_prediction || (row.final_label === 1 ? "Correct posture" : "Incorrect posture"),
      modelLabel: "Exercise Quality",
      createdAt: row.created_at,
      metrics: {
        qualityScore: row.quality_score,
        probability: percentLike(row.mean_correct_probability),
        reliability: row.reliability_level,
        correctRatio: percentLike(row.correct_ratio),
        incorrectRatio: percentLike(row.incorrect_ratio)
      },
      details: {
        id: row.id,
        fileName: row.file_name,
        exerciseKey: row.exercise_key,
        exerciseLabel: row.exercise_label,
        finalPrediction: row.final_prediction,
        finalLabel: row.final_label,
        qualityScore: row.quality_score,
        meanCorrectProbability: row.mean_correct_probability,
        reliabilityLevel: row.reliability_level,
        reliabilityReasons: parseJson(row.reliability_reasons, []),
        windowReport: parseJson(row.window_report_json, {})
      }
    };
  }

  return null;
}

function publicFlag(row) {
  if (!row) return null;
  return {
    id: row.id,
    sourceType: row.source_type,
    sourceLabel: sourceLabels[row.source_type] || row.source_type,
    screeningId: row.screening_id,
    snapshot: parseJson(row.snapshot_json, {}),
    risk: parseJson(row.risk_json, null),
    rehab: parseJson(row.rehab_json, null),
    flaggedAt: row.flagged_at,
    updatedAt: row.updated_at
  };
}

function publicGuidance(row) {
  if (!row) return null;
  return {
    id: row.id,
    patientUserId: row.patient_user_id,
    professionalUserId: row.professional_user_id,
    doctorName: row.doctor_name || null,
    guidanceType: row.guidance_type,
    title: row.title,
    sourceType: row.source_type,
    diseaseFocus: row.disease_focus,
    priority: row.priority,
    payload: parseJson(row.payload_json, {}),
    attachmentName: row.attachment_name,
    attachmentUrl: row.attachment_path ? `/central-profile/guidance/${row.id}/attachment` : null,
    patientViewedAt: row.patient_viewed_at,
    isUnread: !row.patient_viewed_at,
    createdAt: row.created_at,
    updatedAt: row.updated_at
  };
}

function centralSummary(flags) {
  const flaggedTypes = new Set(flags.map((flag) => flag.sourceType));
  return {
    totalFlagged: flags.length,
    totalSlots: sourceOrder.length,
    completedSlots: sourceOrder.filter((sourceType) => flaggedTypes.has(sourceType)).length,
    missingSlots: sourceOrder.filter((sourceType) => !flaggedTypes.has(sourceType)).map((sourceType) => ({
      sourceType,
      sourceLabel: sourceLabels[sourceType]
    }))
  };
}

function normalStatusFromFlag(flag) {
  const snapshot = flag?.snapshot || {};
  const label = String(snapshot.resultLabel || snapshot.details?.finalResult || "").toLowerCase();
  if (snapshot.details?.finalLabel === 1 || label.includes("abnormal")) return "abnormal";
  if (snapshot.details?.finalLabel === 0 || label.includes("normal")) return "normal";
  return "inconclusive";
}

function diseaseMatch(flag, diseaseType) {
  if (!diseaseType || diseaseType === "all") return true;
  if (diseaseType === "exercise") {
    return flag.sourceType?.startsWith("exercise_") && (flag.snapshot?.details?.finalLabel === 0 || String(flag.snapshot?.resultLabel || "").toLowerCase().includes("incorrect"));
  }
  if (flag.sourceType !== diseaseType) return false;
  const details = flag.snapshot?.details || {};
  const text = String(flag.snapshot?.resultLabel || details.finalResult || "").toLowerCase();
  return Boolean(details.detected || details.tendency || (text.includes("detected") && !text.includes("non-")));
}

function publicNormalScreening(row) {
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
    reliabilityReasons: parseJson(row.reliability_reasons, []),
    clinicalNote: row.clinical_note,
    result: parseJson(row.raw_result_json, {}),
    createdAt: row.created_at
  };
}

function publicScaKoaScreening(row) {
  if (!row) return null;
  const isVideo = row.input_type === "video";
  return {
    id: row.id,
    modelKey: row.model_key,
    modelLabel: row.model_key === "koa" ? "KOA" : "SCA",
    inputType: row.input_type,
    fileName: row.file_name,
    hasVideoPreview: isVideo,
    videoUrl: isVideo ? `/sca-koa/screenings/${row.id}/file` : null,
    direction: row.direction,
    fpsUsed: row.fps_used,
    finalResult: row.final_result,
    detected: Boolean(row.detected),
    tendency: Boolean(row.tendency),
    probability: probability(row.probability),
    maxProbability: probability(row.max_probability),
    positiveWindowCount: row.positive_window_count,
    positiveWindowRatio: row.positive_window_ratio,
    patternStrength: row.pattern_strength,
    reliabilityLevel: row.reliability_level,
    reliabilityReasons: parseJson(row.reliability_reasons, []),
    clinicalNote: row.clinical_note,
    instabilityMap: parseJson(row.instability_json, null),
    result: parseJson(row.raw_result_json, {}),
    createdAt: row.created_at
  };
}

function publicPdNeuropathyScreening(row) {
  if (!row) return null;
  const isVideo = row.input_type === "video";
  return {
    id: row.id,
    modelKey: row.model_key,
    modelLabel: row.model_key === "neuropathy" ? "Neuropathy" : "PD",
    inputType: row.input_type,
    fileName: row.file_name,
    hasVideoPreview: isVideo,
    videoUrl: isVideo ? `/pd-neuropathy/screenings/${row.id}/file` : null,
    direction: row.direction,
    fpsUsed: row.fps_used,
    finalResult: row.final_result,
    detected: Boolean(row.detected),
    tendency: Boolean(row.tendency),
    probability: probability(row.probability),
    maxProbability: probability(row.max_probability),
    positiveWindowCount: row.positive_window_count,
    positiveWindowRatio: row.positive_window_ratio,
    reliabilityLevel: row.reliability_level,
    reliabilityReasons: parseJson(row.reliability_reasons, []),
    clinicalNote: row.clinical_note,
    result: parseJson(row.raw_result_json, {}),
    createdAt: row.created_at
  };
}

function publicExerciseScreening(row) {
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
    reliabilityReasons: parseJson(row.reliability_reasons, []),
    windowReport: parseJson(row.window_report_json, {}),
    result: parseJson(row.raw_result_json, {}),
    createdAt: row.created_at
  };
}

async function getProfessionalDetails(userId, flag) {
  const sourceType = flag.sourceType;
  if (sourceType === "normal_abnormal") {
    const row = await db.get("SELECT * FROM normal_abnormal_screenings WHERE user_id = ? AND id = ?", userId, flag.screeningId);
    return publicNormalScreening(row);
  }
  if (sourceType === "sca" || sourceType === "koa") {
    const row = await db.get("SELECT * FROM sca_koa_screenings WHERE user_id = ? AND id = ? AND model_key = ?", userId, flag.screeningId, sourceType);
    const genetics = sourceType === "sca"
      ? await db.get("SELECT * FROM sca_genetic_awareness WHERE user_id = ? AND screening_id = ?", userId, flag.screeningId)
      : null;
    return {
      screening: publicScaKoaScreening(row),
      genetics: genetics ? {
        answers: parseJson(genetics.answers_json, {}),
        relatives: parseJson(genetics.relatives_json, []),
        suspected: parseJson(genetics.suspected_json, []),
        awareness: parseJson(genetics.awareness_json, {})
      } : null
    };
  }
  if (sourceType === "pd" || sourceType === "neuropathy") {
    const row = await db.get("SELECT * FROM pd_neuropathy_screenings WHERE user_id = ? AND id = ? AND model_key = ?", userId, flag.screeningId, sourceType);
    return publicPdNeuropathyScreening(row);
  }
  if (sourceType.startsWith("exercise_")) {
    const row = await db.get("SELECT * FROM exercise_screenings WHERE user_id = ? AND id = ? AND exercise_key = ?", userId, flag.screeningId, sourceType.replace("exercise_", ""));
    return publicExerciseScreening(row);
  }
  return null;
}

async function guidanceForPatient(patientId) {
  const rows = await db.all(`
    SELECT g.*, u.full_name AS doctor_name
    FROM central_profile_guidance g
    LEFT JOIN users u ON u.id = g.professional_user_id
    WHERE g.patient_user_id = ?
    ORDER BY g.created_at DESC
  `, patientId);
  await enforceSharedGuidanceReviewDates(rows);
  const all = rows.map(publicGuidance);
  return {
    risk: all.filter((item) => item.guidanceType === "risk"),
    rehab: all.filter((item) => item.guidanceType === "rehab"),
    unreadRisk: all.filter((item) => item.guidanceType === "risk" && item.isUnread).length,
    unreadRehab: all.filter((item) => item.guidanceType === "rehab" && item.isUnread).length
  };
}

function dueDateFromReviewChoice(choice, createdAt) {
  const text = String(choice || "").toLowerCase();
  let days = 0;
  if (text.includes("1 week")) days = 7;
  else if (text.includes("2 weeks")) days = 14;
  else if (text.includes("2-4 weeks")) days = 28;
  else if (text.includes("1 month")) days = 30;
  else if (text.includes("2 months")) days = 60;
  if (!days) return null;
  const start = new Date(createdAt);
  if (Number.isNaN(start.getTime())) return null;
  start.setDate(start.getDate() + days);
  start.setHours(9, 0, 0, 0);
  return start.toISOString();
}

function reviewReminderForGuidance(row) {
  const payload = parseJson(row.payload_json, {});
  const dueAt = payload.reviewDueAt || dueDateFromReviewChoice(payload.reviewWindow || payload.followUp, row.created_at);
  if (!dueAt) return null;
  const dueDate = new Date(dueAt);
  if (Number.isNaN(dueDate.getTime())) return null;
  return {
    guidanceId: row.id,
    guidanceType: row.guidance_type,
    title: row.title,
    priority: row.priority,
    patientUserId: row.patient_user_id,
    dueAt,
    isOverdue: dueDate < new Date(),
    updatedAt: row.updated_at || row.created_at
  };
}

function reminderDateKey(value) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "";
  return date.toISOString().slice(0, 10);
}

function uniqueReminderCount(reminders, predicate) {
  return new Set(
    reminders
      .filter(predicate)
      .map((reminder) => reminderDateKey(reminder.dueAt))
      .filter(Boolean)
  ).size;
}

const riskReviewChoiceLabels = {
  "1_week": "Review in 1 week",
  "2_weeks": "Review in 2 weeks",
  "1_month": "Review in 1 month",
  "2_months": "Review in 2 months"
};

const rehabReviewChoiceLabels = {
  "1_week": "Reassess after 1 week",
  "2_weeks": "Reassess after 2 weeks",
  "1_month": "Reassess after 1 month",
  "2_months": "Reassess after 2 months"
};

function reviewChoiceKey(value) {
  const text = String(value || "").toLowerCase();
  if (text.includes("1 week")) return "1_week";
  if (text.includes("2 weeks")) return "2_weeks";
  if (text.includes("1 month")) return "1_month";
  if (text.includes("2 months")) return "2_months";
  return "";
}

function reviewChoiceFromPayload(payload) {
  return payload?.reviewWindow || payload?.followUp || "";
}

function applySharedReviewPayload(payload, guidanceType, reviewDueAt, reviewChoice = "") {
  payload.reviewDueAt = reviewDueAt;
  const key = reviewChoiceKey(reviewChoice) || reviewChoiceKey(reviewChoiceFromPayload(payload));
  if (!key) return payload;
  if (guidanceType === "risk") payload.reviewWindow = riskReviewChoiceLabels[key];
  if (guidanceType === "rehab") payload.followUp = rehabReviewChoiceLabels[key];
  return payload;
}

function reviewDateOnly(value) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return String(value || "").slice(0, 10);
  return date.toISOString().slice(0, 10);
}

async function sharedGuidanceReviewConflict(patientId, reviewDueAt, excludeGuidanceId = "") {
  if (!patientId || !reviewDueAt) return null;
  const rows = await db.all(`
    SELECT id, payload_json
    FROM central_profile_guidance
    WHERE patient_user_id = ?
      AND id != ?
    ORDER BY updated_at DESC
  `, patientId, excludeGuidanceId || "");
  const requestedDate = reviewDateOnly(reviewDueAt);
  for (const row of rows) {
    const payload = parseJson(row.payload_json, {});
    if (!payload.reviewDueAt) continue;
    const existingDate = reviewDateOnly(payload.reviewDueAt);
    if (existingDate && requestedDate && existingDate !== requestedDate) {
      return payload.reviewDueAt;
    }
  }
  return null;
}

function sharedGuidanceReviewError(existingReviewDueAt) {
  return `This patient already has a doctor review date (${reviewDateOnly(existingReviewDueAt)}). Risk and rehab must use the exact same date.`;
}

function latestReminderFromList(reminders) {
  let latest = null;
  for (const reminder of reminders) {
    const updatedAt = new Date(reminder.updatedAt || 0);
    if (!latest || updatedAt > latest.updatedAt) latest = { reminder, updatedAt };
  }
  return latest?.reminder || null;
}

function normalizeGuidanceSummary(summary) {
  if (!summary?.reminders?.length) {
    summary.nextReviewDueAt = null;
    summary.nextRiskReviewDueAt = null;
    summary.nextRehabReviewDueAt = null;
    summary.overdueReviewCount = 0;
    summary.upcomingReviewCount = 0;
    return summary;
  }

  const sharedReminder = latestReminderFromList(summary.reminders) || summary.reminders[0];
  const sharedDueDate = new Date(sharedReminder.dueAt);
  const isOverdue = !Number.isNaN(sharedDueDate.getTime()) && sharedDueDate < new Date();
  summary.reminders = summary.reminders.map((reminder) => ({
    ...reminder,
    dueAt: sharedReminder.dueAt,
    isOverdue
  }));
  summary.nextReviewDueAt = sharedReminder.dueAt;
  summary.nextRiskReviewDueAt = summary.riskCount > 0 ? sharedReminder.dueAt : null;
  summary.nextRehabReviewDueAt = summary.rehabCount > 0 ? sharedReminder.dueAt : null;
  summary.overdueReviewCount = isOverdue ? 1 : 0;
  summary.upcomingReviewCount = isOverdue ? 0 : 1;
  return summary;
}

async function syncPatientGuidanceReviewDate(patientId, reviewDueAt, now = new Date().toISOString(), reviewChoice = "") {
  if (!patientId || !reviewDueAt) return;
  const rows = await db.all("SELECT id, guidance_type, payload_json FROM central_profile_guidance WHERE patient_user_id = ?", patientId);
  for (const row of rows) {
    const payload = parseJson(row.payload_json, {});
    applySharedReviewPayload(payload, row.guidance_type, reviewDueAt, reviewChoice);
    await db.run("UPDATE central_profile_guidance SET payload_json = ?, updated_at = ? WHERE id = ?", JSON.stringify(payload), now, row.id);
  }
}

function latestReviewDetailsFromRows(rows) {
  let latest = null;
  for (const row of rows) {
    const payload = parseJson(row.payload_json, {});
    const dueAt = payload.reviewDueAt || dueDateFromReviewChoice(payload.reviewWindow || payload.followUp, row.created_at);
    if (!dueAt) continue;
    const updatedAt = new Date(row.updated_at || row.created_at || 0);
    if (!latest || updatedAt > latest.updatedAt) {
      latest = { dueAt, reviewChoice: reviewChoiceFromPayload(payload), updatedAt };
    }
  }
  return latest;
}

async function enforceSharedGuidanceReviewDates(rows) {
  if (!Array.isArray(rows) || rows.length === 0) return;
  const byPatient = new Map();
  for (const row of rows) {
    const list = byPatient.get(row.patient_user_id) || [];
    list.push(row);
    byPatient.set(row.patient_user_id, list);
  }

  const now = new Date().toISOString();
  for (const patientRows of byPatient.values()) {
    const sharedReview = latestReviewDetailsFromRows(patientRows);
    if (!sharedReview?.dueAt) continue;
    for (const row of patientRows) {
      const payload = parseJson(row.payload_json, {});
      const nextPayload = applySharedReviewPayload(payload, row.guidance_type, sharedReview.dueAt, sharedReview.reviewChoice);
      if (JSON.stringify(parseJson(row.payload_json, {})) === JSON.stringify(nextPayload)) continue;
      row.payload_json = JSON.stringify(payload);
      row.updated_at = now;
      await db.run("UPDATE central_profile_guidance SET payload_json = ?, updated_at = ? WHERE id = ?", row.payload_json, now, row.id);
    }
  }
}

centralProfileRouter.get("/professional/profiles", requireAuth, requireRole("professional"), requireVerifiedProfessional, async (req, res) => {
  const dateFrom = String(req.query.dateFrom || "").trim();
  const dateTo = String(req.query.dateTo || "").trim();
  const normalFilter = String(req.query.normal || "all").trim();
  const diseaseType = String(req.query.diseaseType || "all").trim();
  const guidanceStatus = String(req.query.guidanceStatus || "all").trim();

  const rows = await db.all(`
    SELECT cpf.*, u.full_name, u.email
    FROM central_profile_flags cpf
    JOIN users u ON u.id = cpf.user_id
    WHERE u.role = 'patient'
    ORDER BY cpf.user_id, cpf.updated_at DESC
  `);

  const guidanceRows = await db.all(`
    SELECT *
    FROM central_profile_guidance
    ORDER BY created_at DESC
  `);
  await enforceSharedGuidanceReviewDates(guidanceRows);

  const guidanceByPatient = new Map();
  for (const item of guidanceRows) {
    const current = guidanceByPatient.get(item.patient_user_id) || { riskCount: 0, rehabCount: 0, lastRiskAt: null, lastRehabAt: null, reminders: [] };
    if (item.guidance_type === "risk") {
      current.riskCount += 1;
      if (!current.lastRiskAt || new Date(item.created_at) > new Date(current.lastRiskAt)) current.lastRiskAt = item.created_at;
    }
    if (item.guidance_type === "rehab") {
      current.rehabCount += 1;
      if (!current.lastRehabAt || new Date(item.created_at) > new Date(current.lastRehabAt)) current.lastRehabAt = item.created_at;
    }
    const reminder = reviewReminderForGuidance(item);
    if (reminder) current.reminders.push(reminder);
    guidanceByPatient.set(item.patient_user_id, current);
  }
  for (const current of guidanceByPatient.values()) {
    current.reminders.sort((a, b) => new Date(a.dueAt) - new Date(b.dueAt));
    normalizeGuidanceSummary(current);
  }

  const grouped = new Map();
  for (const row of rows) {
    const current = grouped.get(row.user_id) || {
      patient: { id: row.user_id, fullName: row.full_name, email: row.email },
      flags: []
    };
    current.flags.push(publicFlag(row));
    grouped.set(row.user_id, current);
  }

  let profiles = [...grouped.values()]
    .map((entry) => {
      const latestResultAt = entry.flags.reduce((latest, flag) => {
        const value = flag.snapshot?.createdAt || flag.updatedAt;
        return !latest || new Date(value) > new Date(latest) ? value : latest;
      }, null);
      const updatedAt = entry.flags.reduce((latest, flag) => (!latest || new Date(flag.updatedAt) > new Date(latest) ? flag.updatedAt : latest), null);
      const normalFlag = entry.flags.find((flag) => flag.sourceType === "normal_abnormal");
      const guidance = guidanceByPatient.get(entry.patient.id) || {
        riskCount: 0,
        rehabCount: 0,
        lastRiskAt: null,
        lastRehabAt: null,
        nextReviewDueAt: null,
        nextRiskReviewDueAt: null,
        nextRehabReviewDueAt: null,
        overdueReviewCount: 0,
        upcomingReviewCount: 0,
        reminders: []
      };
      const flaggedTypes = new Set(entry.flags.map((flag) => flag.sourceType));
      const diseaseSignals = entry.flags
        .filter((flag) => ["sca", "koa", "pd", "neuropathy"].some((type) => diseaseMatch(flag, type)))
        .map((flag) => sourceLabels[flag.sourceType]);
      const exerciseConcern = entry.flags.some((flag) => diseaseMatch(flag, "exercise"));
      if (exerciseConcern) diseaseSignals.push("Exercise quality");
      return {
        patient: entry.patient,
        completedSlots: sourceOrder.filter((sourceType) => flaggedTypes.has(sourceType)).length,
        totalSlots: sourceOrder.length,
        normalStatus: normalStatusFromFlag(normalFlag),
        diseaseSignals,
        latestResultAt,
        updatedAt,
        guidance,
        flags: entry.flags.map((flag) => ({
          sourceType: flag.sourceType,
          sourceLabel: flag.sourceLabel,
          resultLabel: flag.snapshot?.resultLabel || "Selected",
          createdAt: flag.snapshot?.createdAt || flag.updatedAt
        }))
      };
    });

  if (dateFrom) profiles = profiles.filter((profile) => new Date(profile.latestResultAt || profile.updatedAt) >= new Date(`${dateFrom}T00:00:00.000Z`));
  if (dateTo) profiles = profiles.filter((profile) => new Date(profile.latestResultAt || profile.updatedAt) <= new Date(`${dateTo}T23:59:59.999Z`));
  if (["normal", "abnormal", "inconclusive"].includes(normalFilter)) profiles = profiles.filter((profile) => profile.normalStatus === normalFilter);
  if (diseaseType !== "all") {
    profiles = profiles.filter((profile) => {
      const sourceFlag = grouped.get(profile.patient.id)?.flags.find((flag) => diseaseMatch(flag, diseaseType));
      return Boolean(sourceFlag);
    });
  }
  if (guidanceStatus === "missing-risk") profiles = profiles.filter((profile) => profile.guidance.riskCount === 0);
  if (guidanceStatus === "missing-rehab") profiles = profiles.filter((profile) => profile.guidance.rehabCount === 0);
  if (guidanceStatus === "complete") profiles = profiles.filter((profile) => profile.guidance.riskCount > 0 && profile.guidance.rehabCount > 0);

  profiles.sort((a, b) => new Date(b.updatedAt || 0) - new Date(a.updatedAt || 0));
  res.json({ profiles });
});

centralProfileRouter.get("/professional/profiles/:patientId", requireAuth, requireRole("professional"), requireVerifiedProfessional, async (req, res) => {
  const patient = await db.get("SELECT id, full_name, email, phone, date_of_birth, gender FROM users WHERE id = ? AND role = 'patient'", req.params.patientId);
  if (!patient) return res.status(404).json({ message: "Patient profile not found." });

  const rows = await db.all("SELECT * FROM central_profile_flags WHERE user_id = ?", patient.id);
  const flags = rows.map(publicFlag).sort((a, b) => sourceOrder.indexOf(a.sourceType) - sourceOrder.indexOf(b.sourceType));
  if (flags.length === 0) {
    return res.status(404).json({ message: "This patient has not selected any centralized profile results yet." });
  }

  res.json({
    patient: {
      id: patient.id,
      fullName: patient.full_name,
      email: patient.email,
      phone: patient.phone,
      dateOfBirth: patient.date_of_birth,
      gender: patient.gender
    },
    profile: centralSummary(flags),
    flags,
    details: Object.fromEntries(await Promise.all(flags.map(async (flag) => [flag.sourceType, await getProfessionalDetails(patient.id, flag)]))),
    guidance: await guidanceForPatient(patient.id),
    sourceOrder: sourceOrder.map((sourceType) => ({ sourceType, sourceLabel: sourceLabels[sourceType] }))
  });
});

centralProfileRouter.post(
  "/professional/profiles/:patientId/guidance",
  requireAuth,
  requireRole("professional"),
  requireVerifiedProfessional,
  uploadGuidanceAttachment,
  async (req, res) => {
    const patient = await db.get("SELECT id FROM users WHERE id = ? AND role = 'patient'", req.params.patientId);
    if (!patient) return res.status(404).json({ message: "Patient profile not found." });

    const flags = (await db.all("SELECT * FROM central_profile_flags WHERE user_id = ?", patient.id)).map(publicFlag);
    if (flags.length === 0) {
      return res.status(404).json({ message: "This patient has not selected any centralized profile results yet." });
    }

    const guidanceType = String(req.body.guidanceType || "").trim();
    if (!["risk", "rehab"].includes(guidanceType)) {
      return res.status(422).json({ message: "Choose risk or rehabilitation guidance." });
    }

    const title = String(req.body.title || "").trim();
    const priority = String(req.body.priority || "").trim();
    const payload = parseJson(req.body.payload, null);
    if (!title || !priority || !payload || typeof payload !== "object") {
      return res.status(422).json({ message: "Complete the required guidance fields before submitting." });
    }
    const reviewConflict = await sharedGuidanceReviewConflict(patient.id, payload.reviewDueAt);
    if (reviewConflict) {
      return res.status(422).json({ message: sharedGuidanceReviewError(reviewConflict) });
    }

    const sourceType = String(req.body.sourceType || "").trim() || null;
    const diseaseFocus = String(req.body.diseaseFocus || "").trim() || null;
    const now = new Date().toISOString();
    const id = createId(guidanceType === "risk" ? "risk" : "rehab");

    await db.run(`
      INSERT INTO central_profile_guidance (
        id, patient_user_id, professional_user_id, guidance_type, title,
        source_type, disease_focus, priority, payload_json,
        attachment_name, attachment_path, created_at, updated_at
      ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    `, id,
      patient.id,
      req.user.id,
      guidanceType,
      title,
      sourceType,
      diseaseFocus,
      priority,
      JSON.stringify(payload),
      req.file?.originalname || null,
      req.file?.path || null,
      now,
      now);
    await syncPatientGuidanceReviewDate(patient.id, payload.reviewDueAt, now, reviewChoiceFromPayload(payload));

    const saved = await db.get(`
      SELECT g.*, u.full_name AS doctor_name
      FROM central_profile_guidance g
      LEFT JOIN users u ON u.id = g.professional_user_id
      WHERE g.id = ?
    `, id);

    res.json({
      message: guidanceType === "risk" ? "Risk profile uploaded to the patient centralized profile." : "Recommendation and rehabilitation plan uploaded to the patient centralized profile.",
      guidance: publicGuidance(saved)
    });
  }
);

centralProfileRouter.patch(
  "/professional/profiles/:patientId/review-date",
  requireAuth,
  requireRole("professional"),
  requireVerifiedProfessional,
  async (req, res) => {
    const patient = await db.get("SELECT id FROM users WHERE id = ? AND role = 'patient'", req.params.patientId);
    if (!patient) return res.status(404).json({ message: "Patient profile not found." });

    const reviewDueAt = String(req.body.reviewDueAt || "").trim();
    const reviewChoice = String(req.body.reviewChoice || "").trim();
    if (!reviewDueAt || Number.isNaN(new Date(reviewDueAt).getTime())) {
      return res.status(422).json({ message: "Choose a valid shared doctor review date." });
    }

    const rows = await db.all(`
      SELECT id, guidance_type, payload_json
      FROM central_profile_guidance
      WHERE patient_user_id = ?
        AND professional_user_id = ?
    `, patient.id, req.user.id);
    if (rows.length === 0) {
      return res.status(404).json({ message: "Upload risk or rehab guidance before setting a shared review date." });
    }

    const now = new Date().toISOString();
    for (const row of rows) {
      const payload = parseJson(row.payload_json, {});
      applySharedReviewPayload(payload, row.guidance_type, reviewDueAt, reviewChoice);
      await db.run("UPDATE central_profile_guidance SET payload_json = ?, patient_viewed_at = NULL, updated_at = ? WHERE id = ?", JSON.stringify(payload), now, row.id);
    }

    res.json({ message: "Shared doctor review date updated for this patient profile.", reviewDueAt });
  }
);

centralProfileRouter.patch(
  "/professional/guidance/:id",
  requireAuth,
  requireRole("professional"),
  requireVerifiedProfessional,
  uploadGuidanceAttachment,
  async (req, res) => {
    const existing = await db.get(`
      SELECT * FROM central_profile_guidance
      WHERE id = ? AND professional_user_id = ?
    `, req.params.id, req.user.id);
    if (!existing) return res.status(404).json({ message: "Guidance item not found." });

    const title = String(req.body.title || "").trim();
    const priority = String(req.body.priority || "").trim();
    const payload = parseJson(req.body.payload, null);
    if (!title || !priority || !payload || typeof payload !== "object") {
      return res.status(422).json({ message: "Complete the required guidance fields before saving." });
    }

    const guidanceType = String(req.body.guidanceType || existing.guidance_type).trim();
    if (guidanceType !== existing.guidance_type) {
      return res.status(422).json({ message: "Guidance type cannot be changed while editing." });
    }
    const reviewConflict = await sharedGuidanceReviewConflict(existing.patient_user_id, payload.reviewDueAt, existing.id);
    if (reviewConflict) {
      return res.status(422).json({ message: sharedGuidanceReviewError(reviewConflict) });
    }

    const removeAttachment = String(req.body.removeAttachment || "").toLowerCase() === "true";
    const attachmentName = req.file?.originalname || (removeAttachment ? null : existing.attachment_name);
    const attachmentPath = req.file?.path || (removeAttachment ? null : existing.attachment_path);
    if ((req.file || removeAttachment) && existing.attachment_path) safeUnlinkGuidanceFile(existing.attachment_path);

    const now = new Date().toISOString();
    await db.run(`
      UPDATE central_profile_guidance
      SET title = ?,
          source_type = ?,
          disease_focus = ?,
          priority = ?,
          payload_json = ?,
          attachment_name = ?,
          attachment_path = ?,
          patient_viewed_at = NULL,
          updated_at = ?
      WHERE id = ? AND professional_user_id = ?
    `, title,
      String(req.body.sourceType || "").trim() || null,
      String(req.body.diseaseFocus || "").trim() || null,
      priority,
      JSON.stringify(payload),
      attachmentName,
      attachmentPath,
      now,
      existing.id,
      req.user.id);
    await syncPatientGuidanceReviewDate(existing.patient_user_id, payload.reviewDueAt, now, reviewChoiceFromPayload(payload));

    const saved = await db.get(`
      SELECT g.*, u.full_name AS doctor_name
      FROM central_profile_guidance g
      LEFT JOIN users u ON u.id = g.professional_user_id
      WHERE g.id = ?
    `, existing.id);

    res.json({ message: "Guidance item updated.", guidance: publicGuidance(saved) });
  }
);

centralProfileRouter.delete("/professional/guidance/:id", requireAuth, requireRole("professional"), requireVerifiedProfessional, async (req, res) => {
  const existing = await db.get(`
    SELECT * FROM central_profile_guidance
    WHERE id = ? AND professional_user_id = ?
  `, req.params.id, req.user.id);
  if (!existing) return res.status(404).json({ message: "Guidance item not found." });

  safeUnlinkGuidanceFile(existing.attachment_path);
  await db.run("DELETE FROM central_profile_guidance WHERE id = ? AND professional_user_id = ?", existing.id, req.user.id);
  res.json({ message: "Guidance item deleted." });
});

centralProfileRouter.get("/guidance/:id/attachment", requireAuth, async (req, res) => {
  const row = await db.get("SELECT * FROM central_profile_guidance WHERE id = ?", req.params.id);
  if (!row || !row.attachment_path) return res.status(404).json({ message: "Attachment not found." });
  const allowed = req.user.role === "patient"
    ? row.patient_user_id === req.user.id
    : req.user.role === "professional" && row.professional_user_id === req.user.id;
  if (!allowed) return res.status(403).json({ message: "You do not have access to this attachment." });
  const absolutePath = path.resolve(row.attachment_path);
  if (!absolutePath.startsWith(guidanceUploadRoot) || !fs.existsSync(absolutePath)) {
    return res.status(404).json({ message: "Attachment file not found." });
  }
  res.sendFile(absolutePath);
});

centralProfileRouter.post("/guidance/:id/read", requireAuth, requireRole("patient"), async (req, res) => {
  const row = await db.get("SELECT * FROM central_profile_guidance WHERE id = ? AND patient_user_id = ?", req.params.id, req.user.id);
  if (!row) return res.status(404).json({ message: "Guidance item not found." });
  const now = new Date().toISOString();
  await db.run("UPDATE central_profile_guidance SET patient_viewed_at = COALESCE(patient_viewed_at, ?), updated_at = ? WHERE id = ?", now, now, row.id);
  res.json({ message: "Guidance marked as viewed." });
});

centralProfileRouter.get("/", requireAuth, requireRole("patient"), async (req, res) => {
  const rows = await db.all(`
    SELECT * FROM central_profile_flags
    WHERE user_id = ?
  `, req.user.id);

  const flags = rows
    .map(publicFlag)
    .sort((a, b) => sourceOrder.indexOf(a.sourceType) - sourceOrder.indexOf(b.sourceType));

  res.json({
    profile: centralSummary(flags),
    flags,
    guidance: await guidanceForPatient(req.user.id),
    sourceOrder: sourceOrder.map((sourceType) => ({
      sourceType,
      sourceLabel: sourceLabels[sourceType]
    }))
  });
});

centralProfileRouter.post("/flags", requireAuth, requireRole("patient"), async (req, res) => {
  const sourceType = getSourceType(req.body);
  const screeningId = String(req.body?.screeningId || "").trim();
  if (!sourceType || !screeningId) {
    return res.status(422).json({ message: "Choose a valid screening result to add to the centralized profile." });
  }

  const snapshot = await getScreeningForSource(req.user.id, sourceType, screeningId);
  if (!snapshot) {
    return res.status(404).json({ message: "Screening result not found for this profile section." });
  }

  const now = new Date().toISOString();
  const existing = await db.get(`
    SELECT id, created_at FROM central_profile_flags
    WHERE user_id = ? AND source_type = ?
  `, req.user.id, sourceType);

  await db.run(`
    INSERT INTO central_profile_flags (
      id, user_id, source_type, screening_id, snapshot_json,
      flagged_at, created_at, updated_at
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(user_id, source_type) DO UPDATE SET
      screening_id = excluded.screening_id,
      snapshot_json = excluded.snapshot_json,
      flagged_at = excluded.flagged_at,
      updated_at = excluded.updated_at
  `, existing?.id || createId("cpf"),
    req.user.id,
    sourceType,
    screeningId,
    JSON.stringify(snapshot),
    now,
    existing?.created_at || now,
    now);

  const saved = await db.get(`
    SELECT * FROM central_profile_flags
    WHERE user_id = ? AND source_type = ?
  `, req.user.id, sourceType);

  res.json({
    message: `${sourceLabels[sourceType]} result selected for the centralized profile.`,
    flag: publicFlag(saved)
  });
});

centralProfileRouter.delete("/flags/:sourceType", requireAuth, requireRole("patient"), async (req, res) => {
  const sourceType = getSourceType({ sourceType: req.params.sourceType });
  if (!sourceType) return res.status(404).json({ message: "Central profile section not found." });

  await db.run(`
    DELETE FROM central_profile_flags
    WHERE user_id = ? AND source_type = ?
  `, req.user.id, sourceType);

  res.json({ message: `${sourceLabels[sourceType]} central result cleared.` });
});
