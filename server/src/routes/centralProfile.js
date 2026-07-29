import express from "express";
import { db } from "../db.js";
import { requireAuth, requireRole, requireVerifiedProfessional } from "../middleware/auth.js";
import { createId } from "../utils/security.js";

export const centralProfileRouter = express.Router();

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

function publicFlag(row) {
  return {
    id: row.id,
    sourceType: row.source_type,
    sourceLabel: sourceLabels[row.source_type] || row.source_type,
    screeningId: row.screening_id,
    snapshot: parseJson(row.snapshot_json, {}),
    flaggedAt: row.flagged_at,
    updatedAt: row.updated_at
  };
}

function orderedFlags(rows) {
  return rows
    .map(publicFlag)
    .sort((a, b) => sourceOrder.indexOf(a.sourceType) - sourceOrder.indexOf(b.sourceType));
}

centralProfileRouter.get("/patient", requireAuth, requireRole("patient"), (req, res) => {
  const rows = db.prepare(`
    SELECT *
    FROM central_profile_flags
    WHERE user_id = ?
    ORDER BY updated_at DESC
  `).all(req.user.id);

  res.json({ flags: orderedFlags(rows) });
});

centralProfileRouter.post("/patient/flags", requireAuth, requireRole("patient"), (req, res) => {
  const sourceType = String(req.body.sourceType || "").trim();
  const screeningId = String(req.body.screeningId || "").trim();
  const snapshot = req.body.snapshot && typeof req.body.snapshot === "object" ? req.body.snapshot : {};

  if (!sourceLabels[sourceType] || !screeningId) {
    return res.status(422).json({ message: "Choose a valid screening result for the centralized profile." });
  }

  const now = new Date().toISOString();
  const existing = db.prepare(`
    SELECT id
    FROM central_profile_flags
    WHERE user_id = ? AND source_type = ?
  `).get(req.user.id, sourceType);

  if (existing) {
    db.prepare(`
      UPDATE central_profile_flags
      SET screening_id = ?, snapshot_json = ?, flagged_at = ?, updated_at = ?
      WHERE id = ? AND user_id = ?
    `).run(screeningId, JSON.stringify(snapshot), now, now, existing.id, req.user.id);
  } else {
    db.prepare(`
      INSERT INTO central_profile_flags (
        id, user_id, source_type, screening_id, snapshot_json, flagged_at, created_at, updated_at
      ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    `).run(createId("cpf"), req.user.id, sourceType, screeningId, JSON.stringify(snapshot), now, now, now);
  }

  const rows = db.prepare("SELECT * FROM central_profile_flags WHERE user_id = ?").all(req.user.id);
  res.status(existing ? 200 : 201).json({ flags: orderedFlags(rows) });
});

centralProfileRouter.get("/professional/profiles", requireAuth, requireRole("professional"), requireVerifiedProfessional, (req, res) => {
  const rows = db.prepare(`
    SELECT
      u.id AS patient_id,
      u.full_name,
      u.email,
      u.created_at AS patient_created_at,
      cpf.*
    FROM users u
    JOIN central_profile_flags cpf ON cpf.user_id = u.id
    WHERE u.role = 'patient'
    ORDER BY cpf.updated_at DESC
  `).all();

  const profiles = new Map();
  for (const row of rows) {
    if (!profiles.has(row.patient_id)) {
      profiles.set(row.patient_id, {
        patient: {
          id: row.patient_id,
          fullName: row.full_name,
          email: row.email,
          createdAt: row.patient_created_at
        },
        flags: []
      });
    }

    profiles.get(row.patient_id).flags.push(publicFlag(row));
  }

  res.json({
    profiles: [...profiles.values()].map((profile) => ({
      ...profile,
      flags: orderedFlags(profile.flags.map((flag) => ({
        id: flag.id,
        source_type: flag.sourceType,
        screening_id: flag.screeningId,
        snapshot_json: JSON.stringify(flag.snapshot),
        flagged_at: flag.flaggedAt,
        updated_at: flag.updatedAt
      })))
    }))
  });
});
