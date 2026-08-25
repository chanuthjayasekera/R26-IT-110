import React, { useCallback, useEffect, useMemo, useState } from "react";
import {
  Alert,
  Box,
  Button,
  Card,
  CardContent,
  Chip,
  CircularProgress,
  Container,
  Divider,
  Grid2,
  Checkbox,
  IconButton,
  LinearProgress,
  ListItemText,
  MenuItem,
  Stack,
  TextField,
  Typography
} from "@mui/material";
import { useSearchParams } from "react-router-dom";
import AccessibilityNewIcon from "@mui/icons-material/AccessibilityNew";
import AssessmentIcon from "@mui/icons-material/Assessment";
import AttachFileIcon from "@mui/icons-material/AttachFile";
import BarChartIcon from "@mui/icons-material/BarChart";
import BiotechIcon from "@mui/icons-material/Biotech";
import ChevronLeftIcon from "@mui/icons-material/ChevronLeft";
import ChevronRightIcon from "@mui/icons-material/ChevronRight";
import DeleteOutlineIcon from "@mui/icons-material/DeleteOutline";
import DescriptionIcon from "@mui/icons-material/Description";
import FitnessCenterIcon from "@mui/icons-material/FitnessCenter";
import HealingIcon from "@mui/icons-material/Healing";
import InfoOutlinedIcon from "@mui/icons-material/InfoOutlined";
import MonitorHeartIcon from "@mui/icons-material/MonitorHeart";
import PictureAsPdfIcon from "@mui/icons-material/PictureAsPdf";
import TableChartIcon from "@mui/icons-material/TableChart";
import TimelineIcon from "@mui/icons-material/Timeline";
import WarningAmberIcon from "@mui/icons-material/WarningAmber";
import { ResultBillboard, BiometricsTable } from "../../normal-abnormal-detection-system/pages/NormalAbnormalResults.jsx";
import { GeneticChartV2, InstabilityBodyMap, ScaKoaOverview } from "../../sca-koa-detection-system/pages/ScaKoaResults.jsx";
import { ParkinsonOverview } from "../../pd-detection-system/pages/ParkinsonResults.jsx";
import { DiseaseComparisonChart, MetricComparisonChart } from "../../pd-detection-system/components/Component3Charts.jsx";
import { ExerciseOverview, WindowList } from "../../recommendation-rehabilitation-system/pages/RehabExerciseResults.jsx";
import { api, getApiError } from "../api/http.js";
import "../../pd-detection-system/styles/parkinson-detection.css";
import "../styles/central-profile.css";

const exerciseOverviewSections = [
  { key: "exercise_gesture3", sourceType: "exercise_gesture3", label: "Left Forward Raise", title: "Seated Left-Arm Forward Raise Overview" },
  { key: "exercise_gesture5", sourceType: "exercise_gesture5", label: "Left Lateral Raise", title: "Seated Left-Arm Lateral Raise Overview" },
  { key: "exercise_gesture2", sourceType: "exercise_gesture2", label: "Right Forward Raise", title: "Seated Right-Arm Forward Raise Overview" }
];

const sourceOptions = [
  { value: "normal_abnormal", label: "Normal vs Abnormal" },
  { value: "sca", label: "SCA" },
  { value: "koa", label: "KOA" },
  { value: "pd", label: "PD" },
  { value: "neuropathy", label: "Neuropathy" },
  { value: "exercise_gesture3", label: "Left forward raise" },
  { value: "exercise_gesture5", label: "Left lateral raise" },
  { value: "exercise_gesture2", label: "Right forward raise" }
];

const riskDriverOptions = [
  "Fall or near-fall pattern",
  "Freezing, shuffling, or unstable turns",
  "Foot numbness, wounds, or unsafe footwear",
  "Medication timing, dizziness, or sleepiness",
  "Home hazards or low lighting",
  "Pain, swelling, or knee-load limitation",
  "Low adherence or fear of movement",
  "Rapid functional decline"
];

const rehabExerciseOptions = [
  "Aerobic walking or cycling",
  "Balance and gait-cue training",
  "Strength and resistance work",
  "Flexibility, posture, and trunk mobility",
  "Non-contact boxing, dance, tai chi, or yoga",
  "Task-specific arm raise correction",
  "Foot-care and protective gait routine"
];

const testOptions = [
  "Timed Up and Go / chair stand / balance testing",
  "Neurology or movement-disorder review",
  "Nerve conduction study or EMG",
  "HbA1c, vitamin B12, thyroid, and metabolic labs",
  "Foot exam, footwear review, or podiatry referral",
  "Knee X-ray, MRI, or orthopedic review",
  "Physical therapy gait assessment"
];

const triggerContextOptions = [
  "Bathroom, stairs, or narrow spaces",
  "Outdoor uneven ground",
  "Morning stiffness or first steps",
  "Medication wearing-off window",
  "Crowded public places",
  "Night walking or low lighting",
  "Exercise fatigue window",
  "Foot pain, numbness, or swelling episode"
];

const payloadLabels = {
  timeHorizon: "How soon this risk matters",
  riskDrivers: "Why your doctor is concerned",
  triggerContexts: "Situations where you should be extra careful",
  patientResources: "Extra notes or trusted links from your doctor",
  safetyPlan: "What you should do day to day",
  escalation: "When you should contact the doctor",
  reviewWindow: "Doctor follow-up plan",
  reviewDueAt: "Doctor reminder date",
  rehabGoal: "Main recovery goal",
  exercisePlan: "Exercises your doctor recommends",
  frequency: "How often to do the plan",
  testsAndScans: "Tests, scans, or referrals to discuss",
  nutritionFocus: "Food and nutrition advice",
  precautions: "Things to avoid or do carefully",
  homePlan: "Your home plan",
  followUp: "Doctor follow-up plan"
};

const payloadDescriptions = {
  timeHorizon: "This tells you whether the risk is immediate, short-term, or longer-term.",
  riskDrivers: "These are the main reasons your doctor added this risk guidance.",
  triggerContexts: "Pay extra attention in these places or moments.",
  patientResources: "These are doctor-added education notes, links, or clinic instructions.",
  safetyPlan: "Follow these practical steps at home and during daily activity.",
  escalation: "Use this to know when you should call, message, or visit the doctor.",
  reviewWindow: "This is when your doctor wants to review this guidance again.",
  reviewDueAt: "This date is also shown as a reminder on the doctor's side.",
  rehabGoal: "This is the main improvement your doctor wants you to work toward.",
  exercisePlan: "Do only the exercise types your doctor selected for you.",
  frequency: "This is the recommended schedule, adjusted to your comfort and safety.",
  testsAndScans: "These are checks or referrals your doctor wants you to discuss or complete.",
  nutritionFocus: "Use this as simple food and hydration guidance.",
  precautions: "Stop or slow down if these warnings apply.",
  homePlan: "These are the instructions to follow at home.",
  followUp: "This is when your doctor wants to reassess your rehab plan."
};

const riskPayloadOrder = ["timeHorizon", "riskDrivers", "triggerContexts", "safetyPlan", "escalation", "reviewWindow", "reviewDueAt", "patientResources"];
const rehabPayloadOrder = ["rehabGoal", "exercisePlan", "frequency", "testsAndScans", "nutritionFocus", "precautions", "homePlan", "followUp", "reviewDueAt"];

const riskReviewOptions = ["Review in 1 week", "Review in 2 weeks", "Review in 1 month", "Review in 2 months"];
const rehabReviewOptions = ["Reassess after 1 week", "Reassess after 2 weeks", "Reassess after 1 month", "Reassess after 2 months"];
const sharedReviewOptions = [
  { value: "Review in 1 week", label: "Review in 1 week" },
  { value: "Review in 2 weeks", label: "Review in 2 weeks" },
  { value: "Review in 1 month", label: "Review in 1 month" },
  { value: "Review in 2 months", label: "Review in 2 months" }
];

function normalizeMultiValue(value) {
  if (Array.isArray(value)) return value;
  if (!value) return [];
  return String(value).split(",").map((item) => item.trim()).filter(Boolean);
}

function formatGuidanceValue(value) {
  if (Array.isArray(value)) return value.join(", ");
  return value;
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function isVideoAttachment(name = "") {
  return /\.(mp4|mov|avi|mkv|webm)$/i.test(name);
}

function attachmentLabel(type, editingItem, file, removeAttachment) {
  if (file) return `Selected: ${file.name}`;
  if (removeAttachment) return "Current attachment will be removed";
  if (editingItem?.attachmentName) return type === "risk" ? "Replace patient education document" : "Replace exercise video";
  return type === "risk" ? "Attach education document" : "Upload exercise recommendation video";
}

function SelectChips({ selected }) {
  const values = normalizeMultiValue(selected);
  if (values.length === 0) return <Typography component="span" color="text.secondary">Select one or more</Typography>;
  return (
    <Box className="central-select-chip-row">
      {values.map((item) => <Chip key={item} size="small" label={item} />)}
    </Box>
  );
}

const sectionMeta = [
  { key: "normal", label: "Normal/Abnormal", icon: <MonitorHeartIcon /> },
  { key: "sca", label: "SCA Overview", icon: <AccessibilityNewIcon /> },
  { key: "koa", label: "KOA Overview", icon: <AccessibilityNewIcon /> },
  { key: "pd", label: "PD Overview", icon: <BarChartIcon /> },
  { key: "neuropathy", label: "Neuropathy Overview", icon: <BarChartIcon /> },
  ...exerciseOverviewSections.map((item) => ({ key: item.key, label: item.label, icon: <FitnessCenterIcon /> })),
  { key: "biometrics", label: "Biometrics", icon: <TableChartIcon /> },
  { key: "instability", label: "Instability", icon: <AccessibilityNewIcon /> },
  { key: "genetics", label: "Genetics of SCA", icon: <BiotechIcon /> },
  { key: "pd-neuropathy", label: "PD vs Neuropathy", icon: <BarChartIcon /> },
  { key: "exercise", label: "Exercise Quality", icon: <FitnessCenterIcon /> },
  { key: "risks", label: "Risks", icon: <WarningAmberIcon /> },
  { key: "rehab", label: "Rehab", icon: <HealingIcon /> },
  { key: "timeline", label: "Timeline", icon: <TimelineIcon /> }
];

function formatDate(value) {
  if (!value) return "-";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "-";
  return new Intl.DateTimeFormat("en", { dateStyle: "medium", timeStyle: "short" }).format(date);
}

function formatDateOnly(value) {
  if (!value) return "-";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "-";
  return new Intl.DateTimeFormat("en", { dateStyle: "medium" }).format(date);
}

function reviewDueDateFromChoice(choice) {
  const text = String(choice || "").toLowerCase();
  let days = 0;
  if (text.includes("1 week")) days = 7;
  else if (text.includes("2 weeks")) days = 14;
  else if (text.includes("1 month")) days = 30;
  else if (text.includes("2 months")) days = 60;
  if (!days) return "";
  const date = new Date();
  date.setDate(date.getDate() + days);
  date.setHours(9, 0, 0, 0);
  return date.toISOString();
}

function dateInputValue(value) {
  if (!value) return "";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return String(value).slice(0, 10);
  const year = date.getFullYear();
  const month = String(date.getMonth() + 1).padStart(2, "0");
  const day = String(date.getDate()).padStart(2, "0");
  return `${year}-${month}-${day}`;
}

function sameDateValue(left, right) {
  return Boolean(left && right) && dateInputValue(left) === dateInputValue(right);
}

function reviewChoiceKey(value) {
  const text = String(value || "").toLowerCase();
  if (text.includes("1 week")) return "1_week";
  if (text.includes("2 weeks")) return "2_weeks";
  if (text.includes("1 month")) return "1_month";
  if (text.includes("2 months")) return "2_months";
  return "";
}

function riskReviewChoiceFromAny(value) {
  const key = reviewChoiceKey(value);
  if (key === "1_week") return "Review in 1 week";
  if (key === "2_weeks") return "Review in 2 weeks";
  if (key === "1_month") return "Review in 1 month";
  if (key === "2_months") return "Review in 2 months";
  return "Review in 2 weeks";
}

function scorePercent(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return 0;
  return Math.max(0, Math.min(100, Math.round(Number(value) * 100)));
}

function normalGaugeBackground(screening) {
  const score = Math.max(0, Math.min(1, Number(screening?.meanProbAbnormal || 0)));
  const degrees = Math.round(score * 360);
  return `conic-gradient(#ef4444 0deg ${degrees}deg, #16a34a ${degrees}deg 360deg)`;
}

function sourceLabel(sourceType) {
  if (sourceType === "normal_abnormal") return "Normal vs Abnormal";
  if (sourceType === "sca") return "SCA";
  if (sourceType === "koa") return "KOA";
  if (sourceType === "pd") return "PD";
  if (sourceType === "neuropathy") return "Neuropathy";
  const exercise = exerciseOverviewSections.find((item) => item.sourceType === sourceType);
  if (exercise) return exercise.label;
  if (sourceType?.startsWith("exercise_")) return "Exercise Quality";
  return sourceType;
}

function EmptyState({ title, children }) {
  return (
    <Box className="central-empty-slot">
      <Typography variant="h6" fontWeight={900}>{title}</Typography>
      <Typography color="text.secondary">{children || "Flag a saved model result for this section and the real result output will appear here."}</Typography>
    </Box>
  );
}

function SectionShell({ icon, title, subtitle, children }) {
  return (
    <Card className="central-section-card">
      <CardContent>
        <Stack spacing={2.4}>
          <Stack direction={{ xs: "column", sm: "row" }} spacing={1.5} alignItems={{ xs: "flex-start", sm: "center" }}>
            <Box className="central-section-icon">{icon}</Box>
            <Box>
              <Typography variant="h4" fontWeight={900}>{title}</Typography>
              <Typography color="text.secondary">{subtitle}</Typography>
            </Box>
          </Stack>
          {children}
        </Stack>
      </CardContent>
    </Card>
  );
}

function RemoveFlagBar({ flag, onRemove, removingType, readOnly = false }) {
  if (!flag) return null;
  return (
    <Box className="central-remove-bar">
      <Stack direction="row" spacing={1} alignItems="center" flexWrap="wrap">
        <Chip color="primary" label={flag.sourceLabel || sourceLabel(flag.sourceType)} />
        <Typography color="text.secondary">Flagged {formatDate(flag.flaggedAt)}</Typography>
      </Stack>
      {readOnly ? (
        <Chip label="Doctor view" />
      ) : (
        <Button
          color="error"
          variant="outlined"
          startIcon={<DeleteOutlineIcon />}
          disabled={removingType === flag.sourceType}
          onClick={() => onRemove(flag.sourceType)}
        >
          {removingType === flag.sourceType ? "Removing..." : "Remove flag"}
        </Button>
      )}
    </Box>
  );
}

function RealNormalOverview({ screening }) {
  if (!screening) return <EmptyState title="No normal/abnormal result selected" />;
  const score = scorePercent(screening.meanProbAbnormal);

  return (
    <Stack spacing={3}>
      <ResultBillboard screening={screening} />
      <Grid2 container spacing={3}>
        <Grid2 size={{ xs: 12, lg: 4 }}>
          <Card className="normal-workspace-card normal-gauge-card">
            <CardContent>
              <Stack spacing={2.5} alignItems="center" textAlign="center">
                <Box className="normal-probability-gauge" sx={{ background: normalGaugeBackground(screening) }}>
                  <Box>
                    <Typography variant="h3" fontWeight={900}>{score}%</Typography>
                    <Typography>abnormal probability</Typography>
                  </Box>
                </Box>
                <Box>
                  <Typography variant="h5" fontWeight={900}>{screening.screeningSeverity || "-"}</Typography>
                  <Typography color="text.secondary">{screening.clinicalNote}</Typography>
                </Box>
                <LinearProgress className="normal-confidence-bar" variant="determinate" value={Number(screening.confidencePercent || 0)} />
                <Typography color="text.secondary">Model confidence: {Math.round(Number(screening.confidencePercent || 0))}%</Typography>
              </Stack>
            </CardContent>
          </Card>
        </Grid2>
        <Grid2 size={{ xs: 12, lg: 8 }}>
          <Grid2 container spacing={2}>
            {[
              ["Latest result", screening.finalResult || "-"],
              ["Severity", screening.screeningSeverity || "-"],
              ["Direction", screening.direction || "-"],
              ["Confidence", `${Math.round(Number(screening.confidencePercent || 0))}%`],
              ["Model suggestion", screening.modelSuggestedResult || "-"],
              ["Created", formatDate(screening.createdAt)]
            ].map(([label, value]) => (
              <Grid2 key={label} size={{ xs: 12, md: 4 }}>
                <Box className="normal-metric-tile">
                  <Typography color="text.secondary">{label}</Typography>
                  <Typography variant="h5" fontWeight={900}>{value}</Typography>
                </Box>
              </Grid2>
            ))}
          </Grid2>
        </Grid2>
      </Grid2>
    </Stack>
  );
}

function RealInstability({ sca, koa }) {
  if (!sca && !koa) return <EmptyState title="No SCA or KOA instability result selected" />;

  return (
    <Stack spacing={3}>
      {sca && (
        <Card className="sca-koa-workspace-card">
          <CardContent>
            <Stack spacing={2}>
              <Typography variant="h5" fontWeight={900}>SCA Instability Output</Typography>
              <InstabilityBodyMap map={sca.instabilityMap} />
            </Stack>
          </CardContent>
        </Card>
      )}
      {koa && (
        <Card className="sca-koa-workspace-card">
          <CardContent>
            <Stack spacing={2}>
              <Typography variant="h5" fontWeight={900}>KOA Instability Output</Typography>
              <InstabilityBodyMap map={koa.instabilityMap} />
            </Stack>
          </CardContent>
        </Card>
      )}
    </Stack>
  );
}

function RealGenetics({ sca, genetics }) {
  if (!sca) return <EmptyState title="No SCA result selected" />;
  if (!genetics?.awareness) return <Alert severity="info">No SCA genetic-awareness chart is saved for the flagged SCA result.</Alert>;

  return <GeneticChartV2 awareness={genetics.awareness} />;
}

function RealPdNeuropathyDisorder({ pd, neuropathy }) {
  if (!pd && !neuropathy) return <EmptyState title="No PD or neuropathy result selected" />;

  return (
    <Stack spacing={3}>
      {pd && <MetricComparisonChart title="PD vs Your Gait" screening={pd} modelLabel="Your uploaded gait" onDownload={false} />}
      {neuropathy && <MetricComparisonChart title="Neuropathy vs Your Gait" screening={neuropathy} modelLabel="Your uploaded gait" onDownload={false} />}
      {(pd || neuropathy) && <DiseaseComparisonChart pdScreening={pd} neuropathyScreening={neuropathy} />}
    </Stack>
  );
}

function RealExercise({ screenings }) {
  if (screenings.length === 0) return <EmptyState title="No exercise quality result selected" />;

  return (
    <Stack spacing={3}>
      {screenings.map((screening) => {
        const report = screening.windowReport || {};
        return (
          <Card key={screening.id} className="rehab-workspace-card">
            <CardContent>
              <Stack spacing={2.4}>
                <Stack direction="row" spacing={1} alignItems="center" flexWrap="wrap">
                  <Chip color={screening.isCorrect ? "success" : "error"} label={screening.exerciseLabel} />
                  <Typography fontWeight={900}>{screening.finalPrediction || "Exercise result"}</Typography>
                </Stack>
                <Grid2 container spacing={2}>
                  <Grid2 size={{ xs: 12, md: 4 }}>
                    <Box className="rehab-metric-tile">
                      <Typography color="text.secondary">Correct probability</Typography>
                      <Typography variant="h5" fontWeight={900}>{Math.round(Number(screening.meanCorrectProbability || 0) * 100)}%</Typography>
                    </Box>
                  </Grid2>
                  <Grid2 size={{ xs: 12, md: 4 }}>
                    <Box className="rehab-metric-tile">
                      <Typography color="text.secondary">Quality score</Typography>
                      <Typography variant="h5" fontWeight={900}>{screening.qualityScore ?? "-"}</Typography>
                    </Box>
                  </Grid2>
                  <Grid2 size={{ xs: 12, md: 4 }}>
                    <Box className="rehab-metric-tile">
                      <Typography color="text.secondary">Reliability</Typography>
                      <Typography variant="h5" fontWeight={900}>{screening.reliabilityLevel || "-"}</Typography>
                    </Box>
                  </Grid2>
                </Grid2>
                <Grid2 container spacing={3}>
                  <Grid2 size={{ xs: 12, lg: 6 }}>
                    <WindowList title="Incorrect Windows" windows={report.incorrect_windows || []} tone="error" />
                  </Grid2>
                  <Grid2 size={{ xs: 12, lg: 6 }}>
                    <WindowList title="Correct Windows" windows={report.correct_windows || []} tone="success" />
                  </Grid2>
                </Grid2>
              </Stack>
            </CardContent>
          </Card>
        );
      })}
    </Stack>
  );
}

function MultiSelectField({ label, value, options, onChange, required = false, error = false, helperText = "" }) {
  const selected = normalizeMultiValue(value);
  return (
    <TextField
      fullWidth
      select
      required={required}
      label={label}
      value={selected}
      error={error}
      helperText={helperText}
      SelectProps={{
        multiple: true,
        renderValue: (items) => <SelectChips selected={items} />
      }}
      onChange={(event) => {
        const next = typeof event.target.value === "string" ? event.target.value.split(",") : event.target.value;
        onChange(next);
      }}
    >
      {options.map((item) => (
        <MenuItem key={item} value={item}>
          <Checkbox checked={selected.includes(item)} />
          <ListItemText primary={item} />
        </MenuItem>
      ))}
    </TextField>
  );
}

function guidancePdfValueHtml(keyName, value) {
  if (Array.isArray(value)) {
    return `
      <div class="pdf-pill-row">
        ${value.map((item) => `<span class="pdf-pill">${escapeHtml(item)}</span>`).join("")}
      </div>
    `;
  }
  const displayValue = keyName === "reviewDueAt" ? formatDateOnly(value) : formatGuidanceValue(value);
  return `<p>${escapeHtml(displayValue || "-")}</p>`;
}

function openGuidancePdf(item) {
  const payload = item.payload || {};
  const isRisk = item.guidanceType === "risk";
  const order = isRisk ? riskPayloadOrder : rehabPayloadOrder;
  const toneClass = isRisk ? "risk" : "rehab";
  const title = item.title || (isRisk ? "Risk guidance" : "Rehabilitation plan");
  const importantKeys = ["safetyPlan", "escalation", "homePlan", "precautions", "patientResources"];
  const sections = order
    .filter((key) => payload[key])
    .map((key, index) => `
      <section class="pdf-section ${importantKeys.includes(key) ? "important" : ""}">
        <div class="pdf-section-head">
          <span>${String(index + 1).padStart(2, "0")}</span>
          <div>
            <h2>${escapeHtml(payloadLabels[key] || key)}</h2>
            ${payloadDescriptions[key] ? `<p>${escapeHtml(payloadDescriptions[key])}</p>` : ""}
          </div>
        </div>
        ${guidancePdfValueHtml(key, payload[key])}
      </section>
    `).join("");
  const attachmentHref = item.attachmentUrl ? `${api.defaults.baseURL}${item.attachmentUrl}` : "";
  const attachmentHtml = attachmentHref ? `
    <section class="pdf-section attachment">
      <h2>Attached file</h2>
      <p>${escapeHtml(item.attachmentName || "Doctor attachment")}</p>
      <p>${escapeHtml(attachmentHref)}</p>
    </section>
  ` : "";
  const reportWindow = window.open("", "_blank", "width=1100,height=800");
  if (!reportWindow) return;

  reportWindow.document.write(`
    <!doctype html>
    <html>
      <head>
        <title>${escapeHtml(title)}</title>
        <style>
          * { box-sizing: border-box; }
          body { margin: 0; padding: 30px; font-family: Arial, sans-serif; color: #0f172a; background: #f8fafc; }
          h1, h2, p { margin: 0; }
          h1 { font-size: 32px; line-height: 1.15; }
          h2 { font-size: 18px; margin-bottom: 5px; }
          p { color: #475569; line-height: 1.55; font-size: 14px; }
          .pdf-header { padding: 24px; margin-bottom: 18px; border-radius: 18px; color: #0f172a; border: 1px solid #cbd5e1; background: #ffffff; }
          .pdf-header.risk { border-color: #fed7aa; background: linear-gradient(135deg, #fff7ed, #ffffff); }
          .pdf-header.rehab { border-color: #bbf7d0; background: linear-gradient(135deg, #f0fdf4, #ffffff); }
          .pdf-badges { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 14px; }
          .pdf-badge, .pdf-pill { display: inline-flex; align-items: center; padding: 7px 11px; border-radius: 999px; font-weight: 800; font-size: 12px; background: #e2e8f0; color: #0f172a; }
          .pdf-badge.risk { background: #fed7aa; color: #9a3412; }
          .pdf-badge.rehab { background: #bbf7d0; color: #166534; }
          .pdf-meta { margin-top: 10px; }
          .pdf-section { break-inside: avoid; padding: 18px; margin-bottom: 12px; border-radius: 16px; border: 1px solid #dbeafe; background: #ffffff; }
          .pdf-section.important { background: #f0f9ff; border-color: #bae6fd; }
          .pdf-section-head { display: grid; grid-template-columns: auto 1fr; gap: 12px; align-items: flex-start; margin-bottom: 12px; }
          .pdf-section-head > span { width: 38px; height: 38px; display: grid; place-items: center; border-radius: 12px; color: #ffffff; background: #0284c7; font-weight: 900; }
          .pdf-pill-row { display: flex; flex-wrap: wrap; gap: 8px; }
          .pdf-pill { background: #eef2ff; color: #312e81; }
          .attachment p + p { margin-top: 6px; word-break: break-all; }
          @media print {
            body { background: #ffffff; padding: 18px; }
            .pdf-header, .pdf-section { page-break-inside: avoid; }
          }
        </style>
      </head>
      <body>
        <header class="pdf-header ${toneClass}">
          <div class="pdf-badges">
            <span class="pdf-badge ${toneClass}">${isRisk ? "Risk guidance" : "Rehab plan"}</span>
            ${item.priority ? `<span class="pdf-badge">${escapeHtml(item.priority)}</span>` : ""}
            ${item.diseaseFocus ? `<span class="pdf-badge">${escapeHtml(item.diseaseFocus)}</span>` : ""}
          </div>
          <h1>${escapeHtml(title)}</h1>
          <p class="pdf-meta">Uploaded by Dr. ${escapeHtml(item.doctorName || "Medical professional")} on ${escapeHtml(formatDate(item.createdAt))}</p>
          ${payload.reviewDueAt ? `<p class="pdf-meta"><strong>Next doctor review:</strong> ${escapeHtml(formatDateOnly(payload.reviewDueAt))}</p>` : ""}
        </header>
        ${sections}
        ${attachmentHtml}
        <script>
          window.addEventListener("load", () => setTimeout(() => window.print(), 300));
        </script>
      </body>
    </html>
  `);
  reportWindow.document.close();
}

function GuidanceValueBlock({ keyName, value, type, index = 0, guidanceType = "" }) {
  const label = payloadLabels[keyName] || keyName.replace(/([A-Z])/g, " $1").replace(/^./, (char) => char.toUpperCase());
  const isArray = Array.isArray(value);
  const important = ["safetyPlan", "escalation", "homePlan", "precautions", "patientResources"].includes(keyName);
  const displayValue = keyName === "reviewDueAt" ? formatDateOnly(value) : formatGuidanceValue(value);
  const isPatient = type === "patient";

  return (
    <Grid2 size={{ xs: 12, md: isArray || important ? 12 : 6 }}>
      <Box className={`central-guidance-field ${important ? "important" : ""} ${isPatient ? "patient" : ""} ${guidanceType}`}>
        <Stack direction="row" spacing={0.8} alignItems="center" className="central-guidance-field-head">
          {isPatient && <Box className="central-guidance-step-number">{String(index + 1).padStart(2, "0")}</Box>}
          {important ? <InfoOutlinedIcon fontSize="small" /> : null}
          <Typography fontWeight={900}>{label}</Typography>
        </Stack>
        {payloadDescriptions[keyName] && (
          <Typography className="central-guidance-field-help">{payloadDescriptions[keyName]}</Typography>
        )}
        {isArray ? (
          <Stack direction="row" spacing={0.8} flexWrap="wrap" useFlexGap className="central-guidance-chip-list">
            {value.map((itemValue) => <Chip key={itemValue} size="small" label={itemValue} />)}
          </Stack>
        ) : (
          <Typography className={type === "patient" ? "central-guidance-patient-text" : ""}>{displayValue}</Typography>
        )}
      </Box>
    </Grid2>
  );
}

function GuidanceAttachment({ item }) {
  if (!item.attachmentUrl) return null;
  const href = `${api.defaults.baseURL}${item.attachmentUrl}`;
  const video = isVideoAttachment(item.attachmentName || "");

  if (video) {
    return (
      <Box className="central-guidance-video">
        <Stack direction="row" spacing={1} alignItems="center">
          <AttachFileIcon fontSize="small" />
          <Typography fontWeight={900}>{item.attachmentName || "Exercise video"}</Typography>
        </Stack>
        <video controls src={href} />
      </Box>
    );
  }

  return (
    <Box className="central-guidance-document">
      <DescriptionIcon />
      <Box>
        <Typography fontWeight={900}>{item.attachmentName || "Patient education document"}</Typography>
        <Typography color="text.secondary">Your doctor attached this document so you can read the full education note, article, or clinic handout.</Typography>
      </Box>
      <Button variant="contained" component="a" href={href} target="_blank" rel="noreferrer">
        Open document
      </Button>
    </Box>
  );
}

function GuidancePatientSummary({ item, canManage }) {
  const payload = item.payload || {};
  const isRisk = item.guidanceType === "risk";
  const headline = isRisk
    ? "Your doctor wants you to stay safer during daily movement."
    : "Your doctor has set a movement and recovery plan for you.";
  const focus = isRisk
    ? normalizeMultiValue(payload.riskDrivers)[0] || item.diseaseFocus || "fall and gait safety"
    : payload.rehabGoal || item.diseaseFocus || "safe movement confidence";
  const action = isRisk
    ? payload.safetyPlan || "Follow the safety plan and be careful in the listed situations."
    : payload.homePlan || "Follow the home plan and exercise schedule selected by your doctor.";
  const reviewText = payload.reviewDueAt ? `Next doctor review: ${formatDateOnly(payload.reviewDueAt)}` : "Next doctor review will appear here.";

  return (
    <Box className={`central-guidance-summary ${isRisk ? "risk" : "rehab"} patient ${canManage ? "doctor" : ""}`}>
      <Box className="central-guidance-summary-icon">
        {isRisk ? <WarningAmberIcon /> : <HealingIcon />}
      </Box>
      <Box className="central-guidance-summary-body">
        <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
          {item.isUnread && !canManage && <Chip color="error" size="small" label="New doctor update" />}
          <Chip color={isRisk ? "warning" : "success"} size="small" label={isRisk ? "Risk guidance" : "Rehab plan"} />
          {item.priority && <Chip size="small" label={item.priority} />}
        </Stack>
        <Typography variant="h6" fontWeight={900}>{headline}</Typography>
        <Typography className="central-guidance-summary-focus">
          Main focus: <strong>{formatGuidanceValue(focus)}</strong>
        </Typography>
        <Typography className="central-guidance-summary-action">
          What to do now: {formatGuidanceValue(action)}
        </Typography>
        <Typography className="central-guidance-summary-review">{reviewText}</Typography>
      </Box>
    </Box>
  );
}

function GuidanceList({ items = [], emptyTitle, onMarkRead, onEdit, onDelete, canManage = false, deletingId = "" }) {
  if (!items.length) return <EmptyState title={emptyTitle}>Doctor-uploaded items will appear here with a new notification marker.</EmptyState>;

  return (
    <Stack spacing={2.5}>
      {items.map((item) => {
        const guidanceEntries = (item.guidanceType === "risk" ? riskPayloadOrder : rehabPayloadOrder)
          .filter((key) => item.payload?.[key])
          .map((key) => [key, item.payload[key]]);

        return (
          <Card key={item.id} className={`central-guidance-card ${item.guidanceType} ${item.isUnread ? "unread" : ""} patient-view ${canManage ? "doctor-view" : ""}`}>
            <CardContent className="central-guidance-card-content">
              <Stack spacing={2.2}>
                <Box className="central-guidance-patient-top">
                  <Stack spacing={1}>
                    <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap className="central-guidance-patient-tags">
                      {item.isUnread && <Chip color="error" size="small" label="New" />}
                      <Chip color={item.guidanceType === "risk" ? "warning" : "success"} size="small" label={item.priority || "Priority"} />
                      {item.diseaseFocus && <Chip size="small" label={item.diseaseFocus} />}
                    </Stack>
                    <Typography component="h3" className="central-guidance-patient-title">{item.title}</Typography>
                    <Typography className="central-guidance-patient-meta">
                      Uploaded by Dr. {item.doctorName || "Medical professional"} on {formatDate(item.createdAt)}
                    </Typography>
                  </Stack>
                  <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap className="central-guidance-patient-actions">
                    <Button variant="contained" startIcon={<PictureAsPdfIcon />} onClick={() => openGuidancePdf(item)}>
                      Download PDF
                    </Button>
                    {item.isUnread && onMarkRead && (
                      <Button className="central-guidance-mark-button" variant="outlined" onClick={() => onMarkRead(item.id)}>
                        Mark viewed
                      </Button>
                    )}
                    {canManage && (
                      <>
                        <Button variant="outlined" onClick={() => onEdit?.(item)}>Edit</Button>
                        <Button color="error" variant="outlined" disabled={deletingId === item.id} onClick={() => onDelete?.(item)}>
                          {deletingId === item.id ? "Deleting..." : "Delete"}
                        </Button>
                      </>
                    )}
                  </Stack>
                </Box>

                <GuidancePatientSummary item={item} canManage={canManage} />

                <Divider />
                <Grid2 container spacing={2} className="central-guidance-step-grid">
                  {guidanceEntries.map(([key, value], index) => (
                    <GuidanceValueBlock
                      key={key}
                      keyName={key}
                      value={value}
                      type="patient"
                      index={index}
                      guidanceType={item.guidanceType}
                    />
                  ))}
                </Grid2>

                <GuidanceAttachment item={item} />
              </Stack>
            </CardContent>
          </Card>
        );
      })}
    </Stack>
  );
}

function formFromGuidance(type, item) {
  const isRisk = type === "risk";
  const payload = item?.payload || {};
  const defaultRiskReview = payload.reviewWindow || "Review in 2 weeks";
  const defaultRehabReview = payload.followUp || "Reassess after 2 weeks";
  return {
    title: item?.title || (isRisk ? "Fall and gait safety risk profile" : "Rehabilitation and recommendation plan"),
    sourceType: item?.sourceType || "normal_abnormal",
    diseaseFocus: item?.diseaseFocus || (isRisk ? "Fall and injury prevention" : "Gait stability and functional recovery"),
    priority: item?.priority || "Moderate",
    timeHorizon: payload.timeHorizon || "Short term: 1-4 weeks",
    riskDrivers: normalizeMultiValue(payload.riskDrivers || riskDriverOptions.slice(0, 2)),
    triggerContexts: normalizeMultiValue(payload.triggerContexts || triggerContextOptions.slice(0, 2)),
    patientResources: payload.patientResources || "",
    safetyPlan: payload.safetyPlan || "",
    escalation: payload.escalation || "",
    reviewWindow: defaultRiskReview,
    reviewDueAt: payload.reviewDueAt || reviewDueDateFromChoice(isRisk ? defaultRiskReview : defaultRehabReview),
    rehabGoal: payload.rehabGoal || "Improve safe walking and daily movement confidence",
    exercisePlan: normalizeMultiValue(payload.exercisePlan || rehabExerciseOptions.slice(0, 2)),
    frequency: payload.frequency || "3-5 sessions per week, adjusted to tolerance",
    testsAndScans: normalizeMultiValue(payload.testsAndScans || testOptions.slice(0, 2)),
    nutritionFocus: payload.nutritionFocus || "Balanced meals, hydration, adequate protein, and condition-specific dietitian review if needed",
    precautions: payload.precautions || "",
    homePlan: payload.homePlan || "",
    followUp: defaultRehabReview
  };
}

function DoctorGuidanceForm({ type, patientId, editingItem = null, lockedReviewDueAt = "", onSaved, onCancelEdit }) {
  const isRisk = type === "risk";
  const initialForm = useMemo(() => formFromGuidance(type, editingItem), [type, editingItem]);
  const [form, setForm] = useState(initialForm);
  const [file, setFile] = useState(null);
  const [removeAttachment, setRemoveAttachment] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState("");
  const [fieldErrors, setFieldErrors] = useState({});

  useEffect(() => {
    setForm(initialForm);
    setFile(null);
    setRemoveAttachment(false);
    setError("");
    setFieldErrors({});
  }, [initialForm]);

  function update(field, value) {
    setForm((current) => ({ ...current, [field]: value }));
    setFieldErrors((current) => ({ ...current, [field]: "" }));
  }

  function updateReviewChoice(field, value) {
    const dueAt = reviewDueDateFromChoice(value);
    setForm((current) => ({ ...current, [field]: value, reviewDueAt: dueAt }));
    setFieldErrors((current) => ({ ...current, [field]: "", reviewDueAt: "" }));
  }

  function validate() {
    const next = {};
    if (!form.title.trim()) next.title = "Title is required.";
    if (!form.sourceType) next.sourceType = "Choose the profile result this guidance relates to.";
    if (!form.priority) next.priority = "Choose priority.";
    if (!form.diseaseFocus) next.diseaseFocus = "Choose clinical focus.";
    if (isRisk) {
      if (normalizeMultiValue(form.riskDrivers).length === 0) next.riskDrivers = "Choose at least one risk driver.";
      if (normalizeMultiValue(form.triggerContexts).length === 0) next.triggerContexts = "Choose at least one real-world situation.";
      const keepsExistingAttachment = Boolean(editingItem?.attachmentName && !removeAttachment);
      if (!form.patientResources.trim() && !file && !keepsExistingAttachment) next.patientResources = "Add a simple education note/link or attach a patient document.";
      if (!form.safetyPlan.trim()) next.safetyPlan = "Write a simple patient safety plan.";
      if (!form.reviewDueAt) next.reviewDueAt = "Choose the doctor reminder date.";
    } else {
      if (!form.rehabGoal.trim()) next.rehabGoal = "Rehab goal is required.";
      if (normalizeMultiValue(form.exercisePlan).length === 0) next.exercisePlan = "Choose at least one exercise type.";
      if (normalizeMultiValue(form.testsAndScans).length === 0) next.testsAndScans = "Choose at least one test, scan, or referral.";
      if (!form.homePlan.trim()) next.homePlan = "Add a simple home plan for the patient.";
      if (!form.reviewDueAt) next.reviewDueAt = "Choose the doctor reminder date.";
    }
    if (lockedReviewDueAt && form.reviewDueAt && !sameDateValue(form.reviewDueAt, lockedReviewDueAt)) {
      next.reviewDueAt = `This patient already has a doctor review date: ${formatDateOnly(lockedReviewDueAt)}. Risk and rehab must use the exact same date.`;
    }
    setFieldErrors(next);
    return Object.keys(next).length === 0;
  }

  async function submit(event) {
    event.preventDefault();
    if (!validate()) return;
    setSaving(true);
    setError("");
    try {
      const payload = isRisk ? {
        timeHorizon: form.timeHorizon,
        riskDrivers: normalizeMultiValue(form.riskDrivers),
        triggerContexts: normalizeMultiValue(form.triggerContexts),
        patientResources: form.patientResources,
        safetyPlan: form.safetyPlan,
        escalation: form.escalation,
        reviewWindow: form.reviewWindow,
        reviewDueAt: form.reviewDueAt
      } : {
        rehabGoal: form.rehabGoal,
        exercisePlan: normalizeMultiValue(form.exercisePlan),
        frequency: form.frequency,
        testsAndScans: normalizeMultiValue(form.testsAndScans),
        nutritionFocus: form.nutritionFocus,
        precautions: form.precautions,
        homePlan: form.homePlan,
        followUp: form.followUp,
        reviewDueAt: form.reviewDueAt
      };
      const body = new FormData();
      body.append("guidanceType", type);
      body.append("title", form.title);
      body.append("sourceType", form.sourceType);
      body.append("diseaseFocus", form.diseaseFocus);
      body.append("priority", form.priority);
      body.append("payload", JSON.stringify(payload));
      if (file) body.append("attachment", file);
      if (removeAttachment) body.append("removeAttachment", "true");
      const url = editingItem
        ? `/central-profile/professional/guidance/${editingItem.id}`
        : `/central-profile/professional/profiles/${patientId}/guidance`;
      const method = editingItem ? "patch" : "post";
      await api[method](url, body, {
        headers: { "Content-Type": "multipart/form-data" }
      });
      setFile(null);
      setRemoveAttachment(false);
      onSaved?.();
    } catch (err) {
      setError(getApiError(err).message || "Unable to upload guidance.");
    } finally {
      setSaving(false);
    }
  }

  return (
    <Card className="central-doctor-form">
      <CardContent>
        <Stack component="form" spacing={2} onSubmit={submit}>
          <Box className="central-doctor-form-head">
            <Stack direction={{ xs: "column", sm: "row" }} spacing={1} justifyContent="space-between" alignItems={{ xs: "flex-start", sm: "center" }}>
              <Box>
                <Typography variant="h6" fontWeight={900}>{editingItem ? "Edit" : "Upload"} {isRisk ? "Risk Profile" : "Recommendations & Rehabilitation"}</Typography>
                <Typography color="text.secondary">
                  {isRisk ? "Create patient-friendly risk guidance with clear drivers, situations, resources, and safety actions." : "Create patient-friendly exercise, test, nutrition, home-plan, and optional video guidance."}
                </Typography>
              </Box>
              {editingItem && <Button variant="text" onClick={onCancelEdit}>Cancel edit</Button>}
            </Stack>
          </Box>
          {error && <Alert severity="error">{error}</Alert>}
          {lockedReviewDueAt && (
            <Alert severity="info">
              This patient already has a doctor review date: {formatDateOnly(lockedReviewDueAt)}. Risk and rehab must use this same date.
            </Alert>
          )}
          {Object.keys(fieldErrors).length > 0 && <Alert severity="warning">Please complete the highlighted fields before uploading.</Alert>}

          <Grid2 container spacing={1.5}>
            <Grid2 size={{ xs: 12, md: 6 }}>
              <TextField fullWidth required label="Title" value={form.title} error={Boolean(fieldErrors.title)} helperText={fieldErrors.title || "Use a short title the patient can recognize."} onChange={(event) => update("title", event.target.value)} />
            </Grid2>
            <Grid2 size={{ xs: 12, md: 3 }}>
              <TextField fullWidth select required label="Profile focus" value={form.sourceType} error={Boolean(fieldErrors.sourceType)} helperText={fieldErrors.sourceType || "Which central result this guidance belongs to."} onChange={(event) => update("sourceType", event.target.value)}>
                {sourceOptions.map((item) => <MenuItem key={item.value} value={item.value}>{item.label}</MenuItem>)}
              </TextField>
            </Grid2>
            <Grid2 size={{ xs: 12, md: 3 }}>
              <TextField fullWidth select required label="Priority" value={form.priority} error={Boolean(fieldErrors.priority)} helperText={fieldErrors.priority || "How urgently the patient should act."} onChange={(event) => update("priority", event.target.value)}>
                {["Low", "Moderate", "High", "Urgent review"].map((item) => <MenuItem key={item} value={item}>{item}</MenuItem>)}
              </TextField>
            </Grid2>
            <Grid2 size={{ xs: 12, md: 6 }}>
              <TextField fullWidth select required label="Clinical focus" value={form.diseaseFocus} error={Boolean(fieldErrors.diseaseFocus)} helperText={fieldErrors.diseaseFocus || "Shown as a patient-facing category."} onChange={(event) => update("diseaseFocus", event.target.value)}>
                {(isRisk
                  ? ["Fall and injury prevention", "PD mobility risk", "Neuropathy foot safety", "KOA pain and load risk", "SCA coordination risk", "Exercise adherence risk"]
                  : ["Gait stability and functional recovery", "PD mobility conditioning", "Neuropathy protective movement", "KOA strength and load management", "SCA coordination training", "Exercise technique correction"]
                ).map((item) => <MenuItem key={item} value={item}>{item}</MenuItem>)}
              </TextField>
            </Grid2>

            {isRisk ? (
              <>
                <Grid2 size={{ xs: 12, md: 6 }}>
                  <TextField fullWidth select label="Risk horizon" value={form.timeHorizon} onChange={(event) => update("timeHorizon", event.target.value)}>
                    {["Immediate: 0-7 days", "Short term: 1-4 weeks", "Medium term: 1-3 months", "Long term: 3-12 months"].map((item) => <MenuItem key={item} value={item}>{item}</MenuItem>)}
                  </TextField>
                </Grid2>
                <Grid2 size={{ xs: 12 }}>
                  <MultiSelectField label="Risk drivers" required value={form.riskDrivers} options={riskDriverOptions} error={Boolean(fieldErrors.riskDrivers)} helperText={fieldErrors.riskDrivers || "Select the main reasons this patient needs risk guidance."} onChange={(value) => update("riskDrivers", value)} />
                </Grid2>
                <Grid2 size={{ xs: 12, md: 6 }}>
                  <MultiSelectField label="Real-world situations" required value={form.triggerContexts} options={triggerContextOptions} error={Boolean(fieldErrors.triggerContexts)} helperText={fieldErrors.triggerContexts || "Select where the patient should pay extra attention."} onChange={(value) => update("triggerContexts", value)} />
                </Grid2>
                <Grid2 size={{ xs: 12, md: 6 }}>
                  <TextField fullWidth multiline minRows={3} label="Patient education notes / links" value={form.patientResources} error={Boolean(fieldErrors.patientResources)} onChange={(event) => update("patientResources", event.target.value)} helperText={fieldErrors.patientResources || "Write simple patient words, paste trusted article links, or attach a document below."} />
                </Grid2>
                <Grid2 size={{ xs: 12, md: 6 }}>
                  <TextField fullWidth multiline minRows={3} required label="Simple safety plan" value={form.safetyPlan} error={Boolean(fieldErrors.safetyPlan)} helperText={fieldErrors.safetyPlan || "Write exactly what the patient should do day to day."} onChange={(event) => update("safetyPlan", event.target.value)} />
                </Grid2>
                <Grid2 size={{ xs: 12, md: 6 }}>
                  <TextField fullWidth multiline minRows={3} label="When to contact the doctor" value={form.escalation} onChange={(event) => update("escalation", event.target.value)} helperText="Example: fall, new numbness, severe pain, wound, dizziness, or rapid worsening." />
                </Grid2>
                <Grid2 size={{ xs: 12, md: 6 }}>
                  <TextField fullWidth select label="Next doctor review" value={form.reviewWindow} onChange={(event) => updateReviewChoice("reviewWindow", event.target.value)}>
                    {riskReviewOptions.map((item) => <MenuItem key={item} value={item}>{item}</MenuItem>)}
                  </TextField>
                </Grid2>
                <Grid2 size={{ xs: 12, md: 6 }}>
                  <TextField
                    fullWidth
                    required
                    type="date"
                    label="Doctor reminder date"
                    value={dateInputValue(form.reviewDueAt)}
                    error={Boolean(fieldErrors.reviewDueAt)}
                    helperText={fieldErrors.reviewDueAt || "Shared by risk and rehab, and shown in the doctor's review calendar."}
                    InputLabelProps={{ shrink: true }}
                    onChange={(event) => update("reviewDueAt", event.target.value)}
                  />
                </Grid2>
                <Grid2 size={{ xs: 12 }}>
                  <Box className="central-attachment-uploader">
                    <Stack direction={{ xs: "column", sm: "row" }} spacing={1} alignItems={{ xs: "flex-start", sm: "center" }}>
                      <Button component="label" variant="outlined" startIcon={<AttachFileIcon />}>
                        {attachmentLabel(type, editingItem, file, removeAttachment)}
                        <input hidden type="file" accept=".pdf,.doc,.docx,.txt,.rtf,.png,.jpg,.jpeg" onChange={(event) => {
                          setFile(event.target.files?.[0] || null);
                          setRemoveAttachment(false);
                          setFieldErrors((current) => ({ ...current, patientResources: "" }));
                        }} />
                      </Button>
                      {editingItem?.attachmentName && !file && (
                        <Button color={removeAttachment ? "error" : "inherit"} variant="text" onClick={() => setRemoveAttachment((value) => !value)}>
                          {removeAttachment ? "Document will be removed" : `Remove current document (${editingItem.attachmentName})`}
                        </Button>
                      )}
                    </Stack>
                    <Typography variant="caption" color="text.secondary">Accepted: PDF, Word, text, or image handouts for patient education.</Typography>
                  </Box>
                </Grid2>
              </>
            ) : (
              <>
                <Grid2 size={{ xs: 12, md: 6 }}>
                  <TextField fullWidth label="Rehab goal" value={form.rehabGoal} onChange={(event) => update("rehabGoal", event.target.value)} />
                </Grid2>
                <Grid2 size={{ xs: 12, md: 6 }}>
                  <MultiSelectField label="Recommended exercises" required value={form.exercisePlan} options={rehabExerciseOptions} error={Boolean(fieldErrors.exercisePlan)} helperText={fieldErrors.exercisePlan || "Select exercise categories for the patient plan."} onChange={(value) => update("exercisePlan", value)} />
                </Grid2>
                <Grid2 size={{ xs: 12, md: 6 }}>
                  <TextField fullWidth select label="Frequency" value={form.frequency} onChange={(event) => update("frequency", event.target.value)}>
                    {["Daily light practice", "3-5 sessions per week, adjusted to tolerance", "2-3 supervised sessions per week", "Pause until clinical review"].map((item) => <MenuItem key={item} value={item}>{item}</MenuItem>)}
                  </TextField>
                </Grid2>
                <Grid2 size={{ xs: 12, md: 6 }}>
                  <MultiSelectField label="Tests / scans / referrals" required value={form.testsAndScans} options={testOptions} error={Boolean(fieldErrors.testsAndScans)} helperText={fieldErrors.testsAndScans || "Select what the patient should discuss or complete next."} onChange={(value) => update("testsAndScans", value)} />
                </Grid2>
                <Grid2 size={{ xs: 12, md: 6 }}>
                  <TextField fullWidth multiline minRows={3} label="Food / nutrition recommendation" value={form.nutritionFocus} onChange={(event) => update("nutritionFocus", event.target.value)} />
                </Grid2>
                <Grid2 size={{ xs: 12, md: 6 }}>
                  <TextField fullWidth multiline minRows={3} label="Precautions" value={form.precautions} onChange={(event) => update("precautions", event.target.value)} />
                </Grid2>
                <Grid2 size={{ xs: 12, md: 6 }}>
                  <TextField fullWidth required multiline minRows={3} label="Home plan" value={form.homePlan} error={Boolean(fieldErrors.homePlan)} helperText={fieldErrors.homePlan || "Plain instructions the patient can follow at home."} onChange={(event) => update("homePlan", event.target.value)} />
                </Grid2>
                <Grid2 size={{ xs: 12, md: 6 }}>
                  <TextField fullWidth select label="Next doctor review" value={form.followUp} onChange={(event) => updateReviewChoice("followUp", event.target.value)}>
                    {rehabReviewOptions.map((item) => <MenuItem key={item} value={item}>{item}</MenuItem>)}
                  </TextField>
                </Grid2>
                <Grid2 size={{ xs: 12, md: 6 }}>
                  <TextField
                    fullWidth
                    required
                    type="date"
                    label="Doctor reminder date"
                    value={dateInputValue(form.reviewDueAt)}
                    error={Boolean(fieldErrors.reviewDueAt)}
                    helperText={fieldErrors.reviewDueAt || "Shared by risk and rehab, and shown in the doctor's review calendar."}
                    InputLabelProps={{ shrink: true }}
                    onChange={(event) => update("reviewDueAt", event.target.value)}
                  />
                </Grid2>
                <Grid2 size={{ xs: 12 }}>
                  <Box className="central-attachment-uploader">
                    <Stack direction={{ xs: "column", sm: "row" }} spacing={1} alignItems={{ xs: "flex-start", sm: "center" }}>
                    <Button component="label" variant="outlined" startIcon={<AttachFileIcon />}>
                      {attachmentLabel(type, editingItem, file, removeAttachment)}
                      <input hidden type="file" accept="video/*" onChange={(event) => {
                        setFile(event.target.files?.[0] || null);
                        setRemoveAttachment(false);
                      }} />
                    </Button>
                    {editingItem?.attachmentName && !file && (
                      <Button color={removeAttachment ? "error" : "inherit"} variant="text" onClick={() => setRemoveAttachment((value) => !value)}>
                        {removeAttachment ? "Video will be removed" : `Remove current video (${editingItem.attachmentName})`}
                      </Button>
                    )}
                    </Stack>
                    <Typography variant="caption" color="text.secondary">Optional: add a short exercise demonstration video the patient can replay at home.</Typography>
                  </Box>
                </Grid2>
              </>
            )}
          </Grid2>

          <Button type="submit" variant="contained" disabled={saving}>
            {saving ? "Saving..." : editingItem ? "Save changes" : isRisk ? "Upload risk profile" : "Upload recommendation & rehab"}
          </Button>
        </Stack>
      </CardContent>
    </Card>
  );
}

function fallbackScreeningFromFlag(flag) {
  const snapshot = flag?.snapshot || {};
  const details = snapshot.details || snapshot.screening || snapshot;
  if (!details || Object.keys(details).length === 0) return null;

  if (flag.sourceType?.startsWith("exercise_")) {
    const probabilityValue = details.meanCorrectProbability ?? snapshot.metrics?.probability;
    return {
      ...details,
      id: details.id || flag.screeningId,
      exerciseKey: details.exerciseKey || flag.sourceType.replace("exercise_", ""),
      exerciseLabel: details.exerciseLabel || snapshot.title || flag.sourceLabel,
      finalPrediction: details.finalPrediction || snapshot.resultLabel,
      isCorrect: details.isCorrect ?? details.finalLabel === 1,
      meanCorrectProbability: probabilityValue > 1 ? Number(probabilityValue) / 100 : probabilityValue,
      correctRatio: details.correctRatio ?? snapshot.metrics?.correctRatio,
      incorrectRatio: details.incorrectRatio ?? snapshot.metrics?.incorrectRatio,
      reliabilityLevel: details.reliabilityLevel || snapshot.metrics?.reliability,
      createdAt: details.createdAt || snapshot.createdAt || flag.updatedAt
    };
  }

  return {
    ...details,
    id: details.id || flag.screeningId,
    createdAt: details.createdAt || snapshot.createdAt || flag.updatedAt
  };
}

function fallbackFlagDetail(flag) {
  const screening = fallbackScreeningFromFlag(flag);
  if (flag?.sourceType === "sca" || flag?.sourceType === "koa") {
    return [flag.sourceType, { screening, genetics: screening?.genetics || null }];
  }
  return [flag?.sourceType, screening];
}

async function fetchFlagScreening(flag) {
  if (!flag?.sourceType) return ["", null];
  if (!flag.screeningId) return fallbackFlagDetail(flag);

  try {
    if (flag.sourceType === "normal_abnormal") {
      const res = await api.get(`/normal-abnormal/screenings/${flag.screeningId}`);
      return [flag.sourceType, res.data.screening || fallbackScreeningFromFlag(flag)];
    }
    if (flag.sourceType === "sca" || flag.sourceType === "koa") {
      const res = await api.get(`/sca-koa/screenings/${flag.screeningId}`);
      let genetics = null;
      if (flag.sourceType === "sca") {
        try {
          const geneticsRes = await api.get(`/sca-koa/screenings/${flag.screeningId}/genetics`);
          genetics = geneticsRes.data.genetics;
        } catch {
          genetics = null;
        }
      }
      const fallback = fallbackScreeningFromFlag(flag);
      return [flag.sourceType, { screening: res.data.screening || fallback, genetics: genetics || fallback?.genetics || null }];
    }
    if (flag.sourceType === "pd" || flag.sourceType === "neuropathy") {
      const res = await api.get(`/pd-neuropathy/screenings/${flag.screeningId}`);
      return [flag.sourceType, res.data.screening || fallbackScreeningFromFlag(flag)];
    }
    if (flag.sourceType.startsWith("exercise_")) {
      const res = await api.get(`/exercise-detection/screenings/${flag.screeningId}`);
      return [flag.sourceType, res.data.screening || fallbackScreeningFromFlag(flag)];
    }
  } catch {
    return fallbackFlagDetail(flag);
  }

  return fallbackFlagDetail(flag);
}

function firstGuidanceReviewDueAt(items = [], excludeId = "") {
  return items.find((item) => item.id !== excludeId && item.payload?.reviewDueAt)?.payload?.reviewDueAt || "";
}

function sharedReviewDueAtFromGuidance(guidance) {
  return firstGuidanceReviewDueAt(guidance.risk) || firstGuidanceReviewDueAt(guidance.rehab);
}

function sharedReviewChoiceFromGuidance(guidance) {
  const riskChoice = guidance.risk?.find((item) => item.payload?.reviewWindow)?.payload?.reviewWindow;
  const rehabChoice = guidance.rehab?.find((item) => item.payload?.followUp)?.payload?.followUp;
  return riskReviewChoiceFromAny(riskChoice || rehabChoice);
}

function SharedReviewDateCard({ patientId, guidance, onSaved }) {
  const initialReviewDueAt = sharedReviewDueAtFromGuidance(guidance);
  const initialReviewChoice = sharedReviewChoiceFromGuidance(guidance);
  const [reviewChoice, setReviewChoice] = useState(initialReviewChoice);
  const [reviewDueAt, setReviewDueAt] = useState(initialReviewDueAt);
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");

  useEffect(() => {
    setReviewChoice(initialReviewChoice);
    setReviewDueAt(initialReviewDueAt);
    setMessage("");
    setError("");
  }, [initialReviewChoice, initialReviewDueAt]);

  function updateChoice(value) {
    setReviewChoice(value);
    setReviewDueAt(reviewDueDateFromChoice(value));
    setError("");
    setMessage("");
  }

  async function submit(event) {
    event.preventDefault();
    if (!reviewDueAt) {
      setError("Choose the shared doctor review date.");
      return;
    }
    setSaving(true);
    setError("");
    setMessage("");
    try {
      await api.patch(`/central-profile/professional/profiles/${patientId}/review-date`, {
        reviewChoice,
        reviewDueAt
      });
      setMessage("Shared doctor review date updated for risk and rehab.");
      await onSaved?.();
    } catch (err) {
      setError(getApiError(err).message || "Unable to update the shared review date.");
    } finally {
      setSaving(false);
    }
  }

  const hasGuidance = Boolean(guidance.risk?.length || guidance.rehab?.length);

  return (
    <Card className="central-shared-review-card">
      <CardContent>
        <Stack component="form" spacing={1.5} onSubmit={submit}>
          <Box>
            <Typography variant="h6" fontWeight={900}>Change Shared Review Date</Typography>
            <Typography color="text.secondary">
              Use this button when the doctor wants one new reminder date for both risk and rehab.
            </Typography>
          </Box>
          {message && <Alert severity="success">{message}</Alert>}
          {error && <Alert severity="error">{error}</Alert>}
          {!hasGuidance && <Alert severity="info">Upload risk or rehab guidance first, then set the shared review date.</Alert>}
          <Grid2 container spacing={1.5} alignItems="center">
            <Grid2 size={{ xs: 12, md: 4 }}>
              <TextField fullWidth select label="Review window" value={reviewChoice} disabled={!hasGuidance || saving} onChange={(event) => updateChoice(event.target.value)}>
                {sharedReviewOptions.map((item) => <MenuItem key={item.value} value={item.value}>{item.label}</MenuItem>)}
              </TextField>
            </Grid2>
            <Grid2 size={{ xs: 12, md: 4 }}>
              <TextField
                fullWidth
                type="date"
                label="Shared doctor review date"
                value={dateInputValue(reviewDueAt)}
                disabled={!hasGuidance || saving}
                InputLabelProps={{ shrink: true }}
                onChange={(event) => {
                  setReviewDueAt(event.target.value);
                  setError("");
                  setMessage("");
                }}
              />
            </Grid2>
            <Grid2 size={{ xs: 12, md: 4 }}>
              <Button fullWidth type="submit" variant="contained" disabled={!hasGuidance || saving}>
                {saving ? "Updating..." : "Update shared date"}
              </Button>
            </Grid2>
          </Grid2>
          {initialReviewDueAt && (
            <Typography variant="body2" color="text.secondary">
              Current shared review date: {formatDateOnly(initialReviewDueAt)}
            </Typography>
          )}
        </Stack>
      </CardContent>
    </Card>
  );
}

export default function CentralizedProfile({ mode = "patient", patientId = "", onBack }) {
  const [searchParams, setSearchParams] = useSearchParams();
  const activeSection = searchParams.get("section") || "normal";
  const isProfessionalView = mode === "professional";
  const [profile, setProfile] = useState(null);
  const [patient, setPatient] = useState(null);
  const [flags, setFlags] = useState([]);
  const [details, setDetails] = useState({});
  const [guidance, setGuidance] = useState({ risk: [], rehab: [], unreadRisk: 0, unreadRehab: 0 });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [removingType, setRemovingType] = useState("");
  const [editingGuidance, setEditingGuidance] = useState(null);
  const [deletingGuidanceId, setDeletingGuidanceId] = useState("");
  const [navCollapsed, setNavCollapsed] = useState(false);

  const loadProfile = useCallback(async () => {
    setLoading(true);
    setError("");
    try {
      if (isProfessionalView) {
        if (!patientId) throw new Error("Choose a completed patient profile first.");
        const res = await api.get(`/central-profile/professional/profiles/${patientId}`);
        setPatient(res.data.patient || null);
        setProfile(res.data.profile);
        setFlags(res.data.flags || []);
        setDetails(res.data.details || {});
        setGuidance(res.data.guidance || { risk: [], rehab: [], unreadRisk: 0, unreadRehab: 0 });
        return;
      }

      const res = await api.get("/central-profile");
      const nextFlags = res.data.flags || [];
      const entryResults = await Promise.allSettled(nextFlags.map(fetchFlagScreening));
      const entries = entryResults.map((result, index) => (
        result.status === "fulfilled" ? result.value : fallbackFlagDetail(nextFlags[index])
      ));
      setPatient(null);
      setProfile(res.data.profile);
      setFlags(nextFlags);
      setDetails(Object.fromEntries(entries));
      setGuidance(res.data.guidance || { risk: [], rehab: [], unreadRisk: 0, unreadRehab: 0 });
    } catch (err) {
      const apiError = err.response ? getApiError(err) : err;
      setError(apiError.message || "Unable to load centralized profile.");
    } finally {
      setLoading(false);
    }
  }, [isProfessionalView, patientId]);

  useEffect(() => {
    loadProfile();
  }, [loadProfile]);

  const byType = useMemo(() => Object.fromEntries(flags.map((flag) => [flag.sourceType, flag])), [flags]);
  const progress = profile?.totalSlots ? Math.round((profile.completedSlots / profile.totalSlots) * 100) : 0;
  const normal = details.normal_abnormal || null;
  const sca = details.sca?.screening || null;
  const koa = details.koa?.screening || null;
  const pd = details.pd || null;
  const neuropathy = details.neuropathy || null;
  const exercises = Object.entries(details)
    .filter(([sourceType]) => sourceType.startsWith("exercise_"))
    .map(([, screening]) => screening)
    .filter(Boolean);

  async function removeFlag(sourceType) {
    if (isProfessionalView) return;
    if (!sourceType) return;
    setRemovingType(sourceType);
    setError("");
    try {
      await api.delete(`/central-profile/flags/${sourceType}`);
      await loadProfile();
    } catch (err) {
      const apiError = getApiError(err);
      setError(apiError.message || "Unable to remove this centralized profile flag.");
    } finally {
      setRemovingType("");
    }
  }

  function goSection(section) {
    const next = new URLSearchParams(searchParams);
    if (section === "normal") next.delete("section");
    else next.set("section", section);
    setSearchParams(next);
  }

  function flaggedHeader(sourceType) {
    if (isProfessionalView) return <RemoveFlagBar flag={byType[sourceType]} onRemove={() => {}} removingType="" readOnly />;
    return <RemoveFlagBar flag={byType[sourceType]} onRemove={removeFlag} removingType={removingType} />;
  }

  async function markGuidanceRead(id) {
    if (isProfessionalView) return;
    await api.post(`/central-profile/guidance/${id}/read`);
    await loadProfile();
  }

  function startEditGuidance(item) {
    setEditingGuidance(item);
  }

  async function deleteGuidance(item) {
    if (!item?.id) return;
    if (!window.confirm(`Delete "${item.title}" from this patient profile?`)) return;
    setDeletingGuidanceId(item.id);
    setError("");
    try {
      await api.delete(`/central-profile/professional/guidance/${item.id}`);
      if (editingGuidance?.id === item.id) setEditingGuidance(null);
      await loadProfile();
    } catch (err) {
      setError(getApiError(err).message || "Unable to delete guidance item.");
    } finally {
      setDeletingGuidanceId("");
    }
  }

  function renderActiveSection() {
    if (activeSection === "normal") {
      return (
        <SectionShell icon={<MonitorHeartIcon />} title="Normal/Abnormal Overview" subtitle="Real Component 1 overview from the flagged result.">
          {flaggedHeader("normal_abnormal")}
          <RealNormalOverview screening={normal} />
        </SectionShell>
      );
    }

    if (activeSection === "sca") {
      return (
        <SectionShell icon={<AccessibilityNewIcon />} title="SCA Overview" subtitle="Real SCA overview from the flagged model result.">
          {flaggedHeader("sca")}
          {sca ? <ScaKoaOverview screening={sca} /> : <EmptyState title="No SCA overview selected" />}
        </SectionShell>
      );
    }

    if (activeSection === "koa") {
      return (
        <SectionShell icon={<AccessibilityNewIcon />} title="KOA Overview" subtitle="Real KOA overview from the flagged model result.">
          {flaggedHeader("koa")}
          {koa ? <ScaKoaOverview screening={koa} /> : <EmptyState title="No KOA overview selected" />}
        </SectionShell>
      );
    }

    if (activeSection === "pd") {
      return (
        <SectionShell icon={<BarChartIcon />} title="PD Overview" subtitle="Real PD overview from the flagged model result.">
          {flaggedHeader("pd")}
          {pd ? <ParkinsonOverview screening={pd} /> : <EmptyState title="No PD overview selected" />}
        </SectionShell>
      );
    }

    if (activeSection === "neuropathy") {
      return (
        <SectionShell icon={<BarChartIcon />} title="Neuropathy Overview" subtitle="Real neuropathy overview from the flagged model result.">
          {flaggedHeader("neuropathy")}
          {neuropathy ? <ParkinsonOverview screening={neuropathy} /> : <EmptyState title="No neuropathy overview selected" />}
        </SectionShell>
      );
    }

    const exerciseOverview = exerciseOverviewSections.find((item) => item.key === activeSection);
    if (exerciseOverview) {
      const screening = details[exerciseOverview.sourceType] || null;
      return (
        <SectionShell icon={<FitnessCenterIcon />} title={exerciseOverview.title} subtitle="Real exercise-quality overview from the flagged result.">
          {flaggedHeader(exerciseOverview.sourceType)}
          {screening ? <ExerciseOverview screening={screening} /> : <EmptyState title={`No ${exerciseOverview.label.toLowerCase()} result selected`} />}
        </SectionShell>
      );
    }

    if (activeSection === "biometrics") {
      return (
        <SectionShell icon={<TableChartIcon />} title="Biometrics Table" subtitle="Real Component 1 biometrics table from the flagged result.">
          {flaggedHeader("normal_abnormal")}
          {normal ? <BiometricsTable screening={normal} /> : <EmptyState title="No Component 1 biometrics selected" />}
        </SectionShell>
      );
    }

    if (activeSection === "instability") {
      return (
        <SectionShell icon={<AccessibilityNewIcon />} title="Instability" subtitle="Real SCA/KOA instability visualization from the flagged result.">
          <Stack spacing={1.5}>
            {flaggedHeader("sca")}
            {flaggedHeader("koa")}
          </Stack>
          <RealInstability sca={sca} koa={koa} />
        </SectionShell>
      );
    }

    if (activeSection === "genetics") {
      return (
        <SectionShell icon={<BiotechIcon />} title="Genetics of SCA" subtitle="Real saved SCA genetic-awareness chart from the flagged SCA result.">
          {flaggedHeader("sca")}
          <RealGenetics sca={sca} genetics={details.sca?.genetics} />
        </SectionShell>
      );
    }

    if (activeSection === "pd-neuropathy") {
      return (
        <SectionShell icon={<BarChartIcon />} title="PD vs Neuropathy Disorder" subtitle="Real flagged PD and neuropathy outputs only.">
          <Stack spacing={1.5}>
            {flaggedHeader("pd")}
            {flaggedHeader("neuropathy")}
          </Stack>
          <RealPdNeuropathyDisorder pd={pd} neuropathy={neuropathy} />
        </SectionShell>
      );
    }

    if (activeSection === "exercise") {
      return (
        <SectionShell icon={<FitnessCenterIcon />} title="Exercise Quality" subtitle="Real exercise-quality window reports from flagged exercise results.">
          <Stack spacing={1.5}>
            {Object.keys(byType).filter((sourceType) => sourceType.startsWith("exercise_")).map((sourceType) => flaggedHeader(sourceType))}
          </Stack>
          <RealExercise screenings={exercises} />
        </SectionShell>
      );
    }

    if (activeSection === "risks") {
      const editingRisk = editingGuidance?.guidanceType === "risk" ? editingGuidance : null;
      return (
        <SectionShell icon={<WarningAmberIcon />} title="Risks" subtitle="Doctor-uploaded short-term and long-term risk guidance for this centralized profile.">
          <Stack spacing={2}>
            {isProfessionalView && (
              <>
                <SharedReviewDateCard patientId={patientId} guidance={guidance} onSaved={loadProfile} />
                <DoctorGuidanceForm
                  type="risk"
                  patientId={patientId}
                  editingItem={editingRisk}
                  lockedReviewDueAt={firstGuidanceReviewDueAt(guidance.rehab)}
                  onCancelEdit={() => setEditingGuidance(null)}
                  onSaved={async () => {
                    setEditingGuidance(null);
                    await loadProfile();
                  }}
                />
              </>
            )}
            <GuidanceList
              items={guidance.risk}
              emptyTitle="No risk profile uploaded yet"
              onMarkRead={isProfessionalView ? null : markGuidanceRead}
              canManage={isProfessionalView}
              onEdit={startEditGuidance}
              onDelete={deleteGuidance}
              deletingId={deletingGuidanceId}
            />
          </Stack>
        </SectionShell>
      );
    }

    if (activeSection === "rehab") {
      const editingRehab = editingGuidance?.guidanceType === "rehab" ? editingGuidance : null;
      return (
        <SectionShell icon={<HealingIcon />} title="Recommendations & Rehab" subtitle="Doctor-uploaded exercises, tests, food guidance, precautions, and optional exercise video.">
          <Stack spacing={2}>
            {isProfessionalView && (
              <>
                <SharedReviewDateCard patientId={patientId} guidance={guidance} onSaved={loadProfile} />
                <DoctorGuidanceForm
                  type="rehab"
                  patientId={patientId}
                  editingItem={editingRehab}
                  lockedReviewDueAt={firstGuidanceReviewDueAt(guidance.risk)}
                  onCancelEdit={() => setEditingGuidance(null)}
                  onSaved={async () => {
                    setEditingGuidance(null);
                    await loadProfile();
                  }}
                />
              </>
            )}
            <GuidanceList
              items={guidance.rehab}
              emptyTitle="No recommendation or rehab plan uploaded yet"
              onMarkRead={isProfessionalView ? null : markGuidanceRead}
              canManage={isProfessionalView}
              onEdit={startEditGuidance}
              onDelete={deleteGuidance}
              deletingId={deletingGuidanceId}
            />
          </Stack>
        </SectionShell>
      );
    }

    if (activeSection === "timeline") {
      const guidanceReminders = [...(guidance.risk || []), ...(guidance.rehab || [])]
        .filter((item) => item.payload?.reviewDueAt)
        .sort((a, b) => new Date(a.payload.reviewDueAt) - new Date(b.payload.reviewDueAt));
      return (
        <SectionShell icon={<TimelineIcon />} title="Central Timeline" subtitle="Only flagged profile results, ordered by original screening time.">
          <Stack spacing={1.5}>
            {isProfessionalView && guidanceReminders.map((item) => {
              const dueDate = new Date(item.payload.reviewDueAt);
              const isOverdue = !Number.isNaN(dueDate.getTime()) && dueDate < new Date();
              return (
                <Box key={`guidance-reminder-${item.id}`} className={isOverdue ? "central-timeline-item reminder overdue" : "central-timeline-item reminder"}>
                  <Stack spacing={0.4}>
                    <Stack direction="row" spacing={1} flexWrap="wrap">
                      <Chip size="small" color={isOverdue ? "error" : "info"} label={isOverdue ? "Review overdue" : "Review reminder"} />
                      <Chip size="small" label={item.guidanceType === "risk" ? "Risk" : "Rehab"} />
                    </Stack>
                    <Typography fontWeight={900}>{item.title}</Typography>
                    <Typography color="text.secondary">Doctor follow-up due {formatDateOnly(item.payload.reviewDueAt)}</Typography>
                  </Stack>
                  <Button variant="outlined" size="small" onClick={() => goSection(item.guidanceType === "risk" ? "risks" : "rehab")}>
                    Open upload
                  </Button>
                </Box>
              );
            })}
            {[...flags].sort((a, b) => new Date(b.snapshot?.createdAt || 0) - new Date(a.snapshot?.createdAt || 0)).map((flag) => (
              <Box key={`${flag.sourceType}-${flag.screeningId}`} className="central-timeline-item">
                <Stack spacing={0.4}>
                  <Stack direction="row" spacing={1} flexWrap="wrap">
                    <Chip size="small" color="primary" label={flag.sourceLabel} />
                    <Chip size="small" label={flag.snapshot?.resultLabel || "Selected"} />
                  </Stack>
                  <Typography fontWeight={900}>{flag.snapshot?.title || flag.sourceLabel}</Typography>
                  <Typography color="text.secondary">{formatDate(flag.snapshot?.createdAt)} | Flagged {formatDate(flag.flaggedAt)}</Typography>
                </Stack>
                {!isProfessionalView && (
                  <Button color="error" variant="outlined" size="small" startIcon={<DeleteOutlineIcon />} disabled={removingType === flag.sourceType} onClick={() => removeFlag(flag.sourceType)}>
                    Remove
                  </Button>
                )}
              </Box>
            ))}
            {flags.length === 0 && <EmptyState title="No flagged timeline yet" />}
          </Stack>
        </SectionShell>
      );
    }

    return (
      <SectionShell icon={<MonitorHeartIcon />} title="Normal/Abnormal Overview" subtitle="Real Component 1 overview from the flagged result.">
        {flaggedHeader("normal_abnormal")}
        <RealNormalOverview screening={normal} />
      </SectionShell>
    );
  }

  if (loading) {
    return (
      <Container maxWidth="lg" className="central-profile-page">
        <Card className="central-section-card">
          <CardContent>
            <Stack spacing={2} alignItems="center" py={5}>
              <CircularProgress />
              <Typography>Loading centralized profile...</Typography>
            </Stack>
          </CardContent>
        </Card>
      </Container>
    );
  }

  if (error && flags.length === 0) {
    return (
      <Container maxWidth="lg" className="central-profile-page">
        <Alert severity="error">{error}</Alert>
      </Container>
    );
  }

  return (
    <Container maxWidth="xl" className="central-profile-page">
      <Grid2 container spacing={3}>
        <Grid2 size={{ xs: 12, md: navCollapsed ? 1 : 3, lg: navCollapsed ? 0.9 : 2.5 }}>
          <Box className={navCollapsed ? "central-side-nav collapsed" : "central-side-nav"}>
            <Box className="central-nav-head">
              <AssessmentIcon />
              {!navCollapsed && <Box>
                <Typography fontWeight={900}>Centralized Profile</Typography>
                <Typography variant="caption" color="text.secondary">{profile?.completedSlots || 0}/{profile?.totalSlots || 0} selected</Typography>
              </Box>}
              <IconButton size="small" className="central-nav-collapse" onClick={() => setNavCollapsed((value) => !value)} aria-label={navCollapsed ? "Expand central profile navigation" : "Collapse central profile navigation"}>
                {navCollapsed ? <ChevronRightIcon /> : <ChevronLeftIcon />}
              </IconButton>
            </Box>
            {sectionMeta.map((item) => (
              <Button
                key={item.key}
                variant={activeSection === item.key ? "contained" : "text"}
                startIcon={item.icon}
                onClick={() => goSection(item.key)}
              >
                {!navCollapsed && (
                  <span className="central-nav-button-label">
                    {item.label}
                    {item.key === "risks" && guidance.unreadRisk > 0 && <Chip size="small" color="error" label={guidance.unreadRisk} />}
                    {item.key === "rehab" && guidance.unreadRehab > 0 && <Chip size="small" color="error" label={guidance.unreadRehab} />}
                  </span>
                )}
              </Button>
            ))}
          </Box>
        </Grid2>

        <Grid2 size={{ xs: 12, md: navCollapsed ? 11 : 9, lg: navCollapsed ? 11.1 : 9.5 }}>
          <Stack spacing={3}>
            <Box className="central-profile-hero">
              <Stack spacing={1.5}>
                <Stack direction="row" spacing={1} flexWrap="wrap">
                  {onBack && <Button variant="outlined" size="small" onClick={onBack}>Back to completed profiles</Button>}
                  <Chip icon={<AssessmentIcon />} color="primary" label={isProfessionalView ? "Doctor view: patient central profile" : "Fetched from real flagged model outputs"} />
                  {(guidance.unreadRisk + guidance.unreadRehab) > 0 && !isProfessionalView && <Chip color="error" label={`${guidance.unreadRisk + guidance.unreadRehab} new doctor upload${guidance.unreadRisk + guidance.unreadRehab === 1 ? "" : "s"}`} />}
                </Stack>
                <Typography variant="h3" fontWeight={900}>{isProfessionalView ? `${patient?.fullName || "Patient"} Centralized Profile` : "Centralized Profile"}</Typography>
                <Typography color="text.secondary">
                  {isProfessionalView
                    ? "Medical professional review mode. Inspect the same patient central profile and upload risk or rehabilitation guidance."
                    : "Select a section on the left to show only that real flagged result output inside this page."}
                </Typography>
                {isProfessionalView && patient?.email && <Typography color="text.secondary">{patient.email}{patient.phone ? ` | ${patient.phone}` : ""}</Typography>}
              </Stack>
              <Box className="central-progress-panel">
                <Typography variant="h4" fontWeight={900}>{profile?.completedSlots || 0}/{profile?.totalSlots || 0}</Typography>
                <Typography color="text.secondary">profile sections selected</Typography>
                <LinearProgress variant="determinate" value={progress} />
              </Box>
            </Box>

            {error && <Alert severity="error">{error}</Alert>}
            {flags.length === 0 && (
              <Alert severity="info">
                No results are selected yet. Open a saved model result and click Set as central result.
              </Alert>
            )}

            {renderActiveSection()}
          </Stack>
        </Grid2>
      </Grid2>
    </Container>
  );
}
