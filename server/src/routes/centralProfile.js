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
