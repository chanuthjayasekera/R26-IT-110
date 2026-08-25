import React, { useEffect, useMemo, useState } from "react";
import {
  Alert,
  Box,
  Button,
  Card,
  CardContent,
  Chip,
  CircularProgress,
  Container,
  FormControl,
  Grid2,
  IconButton,
  InputLabel,
  LinearProgress,
  MenuItem,
  Select,
  Stack,
  TextField,
  Tooltip,
  Typography
} from "@mui/material";
import { Link, useLocation, useNavigate, useSearchParams } from "react-router-dom";
import AccountTreeIcon from "@mui/icons-material/AccountTree";
import AccessibilityNewIcon from "@mui/icons-material/AccessibilityNew";
import AddCircleOutlineIcon from "@mui/icons-material/AddCircleOutline";
import AssessmentIcon from "@mui/icons-material/Assessment";
import BiotechIcon from "@mui/icons-material/Biotech";
import CheckCircleIcon from "@mui/icons-material/CheckCircle";
import ChevronLeftIcon from "@mui/icons-material/ChevronLeft";
import ChevronRightIcon from "@mui/icons-material/ChevronRight";
import DeleteOutlineIcon from "@mui/icons-material/DeleteOutline";
import ErrorOutlineIcon from "@mui/icons-material/ErrorOutline";
import FamilyRestroomIcon from "@mui/icons-material/FamilyRestroom";
import PictureAsPdfIcon from "@mui/icons-material/PictureAsPdf";
import ReplayIcon from "@mui/icons-material/Replay";
import SaveIcon from "@mui/icons-material/Save";
import VideoCameraBackIcon from "@mui/icons-material/VideoCameraBack";
import WarningAmberIcon from "@mui/icons-material/WarningAmber";
import CentralProfileFlagButton from "../../common/components/CentralProfileFlagButton.jsx";
import { api, getApiError } from "../../common/api/http.js";
import "../styles/sca-koa-detection.css";

function formatDate(value) {
  if (!value) return "";
  return new Intl.DateTimeFormat("en", { dateStyle: "medium", timeStyle: "short" }).format(new Date(value));
}

function probability(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return 0;
  return Math.min(Math.max(Number(value), 0), 1);
}

function percent(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
  return `${Math.round(Number(value) * 100)}%`;
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function reportMetricValue(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
  return Number(value).toFixed(Math.abs(Number(value)) >= 100 ? 1 : 2).replace(/\.00$/, "");
}

function reportStatusClass(status) {
  if (status === "severe" || status === "red") return "red";
  if (status === "moderate" || status === "yellow") return "yellow";
  if (status === "stable" || status === "green") return "green";
  return "gray";
}

function openPdfReport(title, bodyHtml) {
  const reportWindow = window.open("", "_blank", "width=1100,height=800");
  if (!reportWindow) return;
  reportWindow.document.write(`
    <!doctype html>
    <html>
      <head>
        <title>${escapeHtml(title)}</title>
        <style>
          * { box-sizing: border-box; }
          body { margin: 0; padding: 28px; font-family: Arial, sans-serif; color: #0f172a; background: #f8fafc; }
          h1, h2, h3 { margin: 0; }
          h1 { font-size: 28px; }
          h2 { font-size: 20px; margin-bottom: 8px; }
          p { margin: 4px 0; color: #475569; line-height: 1.45; }
          .report-header { display: flex; justify-content: space-between; gap: 16px; align-items: flex-start; margin-bottom: 20px; padding-bottom: 14px; border-bottom: 2px solid #dbeafe; }
          .report-card { break-inside: avoid; padding: 16px; margin-bottom: 14px; border: 1px solid #cbd5e1; border-radius: 10px; background: #ffffff; }
          .summary-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; margin: 16px 0; }
          .summary-cell { padding: 14px; border-radius: 10px; background: #eef6ff; border: 1px solid #bfdbfe; }
          .body-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; }
          .pdf-map-layout { display: grid; grid-template-columns: 330px 1fr; gap: 18px; align-items: stretch; margin: 16px 0; }
          .pdf-body-card { position: relative; min-height: 500px; border: 1px solid #cbd5e1; border-radius: 14px; background: linear-gradient(180deg, #f8fbff, #ffffff); overflow: hidden; }
          .pdf-human { position: relative; width: 260px; height: 470px; margin: 18px auto; }
          .pdf-head, .pdf-neck, .pdf-torso, .pdf-pelvis, .pdf-arm, .pdf-leg { position: absolute; background: #e2e8f0; border: 1px solid #cbd5e1; }
          .pdf-head { left: 50%; top: 2%; width: 62px; height: 72px; transform: translateX(-50%); border-radius: 48% 48% 44% 44%; }
          .pdf-neck { left: 50%; top: 16%; width: 30px; height: 36px; transform: translateX(-50%); border-radius: 8px; }
          .pdf-torso { left: 50%; top: 22%; width: 124px; height: 156px; transform: translateX(-50%); border-radius: 38px 38px 22px 22px; }
          .pdf-pelvis { left: 50%; top: 52%; width: 114px; height: 62px; transform: translateX(-50%); border-radius: 24px 24px 36px 36px; }
          .pdf-arm { top: 24%; width: 36px; height: 180px; border-radius: 24px; }
          .pdf-arm.left { left: 18%; transform: rotate(10deg); }
          .pdf-arm.right { right: 18%; transform: rotate(-10deg); }
          .pdf-leg { top: 62%; width: 40px; height: 180px; border-radius: 24px; }
          .pdf-leg.left { left: 37%; }
          .pdf-leg.right { right: 37%; }
          .pdf-marker { position: absolute; width: 34px; height: 34px; transform: translate(-50%, -50%); display: grid; place-items: center; border-radius: 50%; border: 3px solid #ffffff; color: #ffffff; font-weight: 900; font-size: 11px; box-shadow: 0 8px 18px rgba(15, 23, 42, 0.22); }
          .pdf-marker.red { background: #dc2626; }
          .pdf-marker.yellow { background: #d97706; }
          .pdf-marker.green { background: #16a34a; }
          .pdf-legend { display: grid; gap: 8px; align-content: start; }
          .pdf-legend-row { display: grid; grid-template-columns: auto 1fr; gap: 8px; align-items: center; padding: 8px 10px; border-radius: 10px; background: #ffffff; border: 1px solid #e2e8f0; }
          .metric-row { display: flex; justify-content: space-between; gap: 12px; padding: 7px 0; border-bottom: 1px solid #e2e8f0; }
          .metric-row:last-child { border-bottom: 0; }
          .badge { display: inline-flex; align-items: center; padding: 4px 9px; border-radius: 999px; font-weight: 700; font-size: 12px; }
          .red { background: #fee2e2; color: #991b1b; }
          .yellow { background: #fef3c7; color: #92400e; }
          .green { background: #dcfce7; color: #166534; }
          .gray { background: #e2e8f0; color: #334155; }
          .family-map { display: grid; grid-template-columns: 1fr 220px 1fr; gap: 16px; align-items: start; }
          .patient { text-align: center; padding: 18px; border-radius: 14px; background: #eff6ff; border: 1px solid #bfdbfe; }
          .score { width: 96px; height: 96px; margin: 12px auto; display: grid; place-items: center; border-radius: 50%; background: #0ea5e9; color: #ffffff; font-size: 30px; font-weight: 900; }
          .lane-title { font-weight: 900; margin-bottom: 10px; }
          .family-node { padding: 12px; margin-bottom: 10px; border-radius: 10px; border: 1px solid #cbd5e1; background: #ffffff; }
          .muted { color: #64748b; }
          @media print {
            body { background: #ffffff; padding: 18px; }
            .report-card { page-break-inside: avoid; }
            .pdf-map-layout { page-break-inside: avoid; }
          }
        </style>
      </head>
      <body>
        <div class="report-header">
          <div>
            <h1>${escapeHtml(title)}</h1>
            <p>Generated from the saved gait screening profile.</p>
          </div>
          <p>${escapeHtml(new Date().toLocaleString())}</p>
        </div>
        ${bodyHtml}
        <script>
          window.addEventListener("load", () => setTimeout(() => window.print(), 300));
        </script>
      </body>
    </html>
  `);
  reportWindow.document.close();
}

function clinicalMetricReportHtml(metric) {
  if (!metric) return "";
  const label = escapeHtml(metric.label || metric.key || "Metric");
  const value = `${reportMetricValue(metric.value)}${metric.unit ? ` ${escapeHtml(metric.unit)}` : ""}${metric.inferred ? " - estimated" : ""}`;
  return `
    <div class="metric-row">
      <span>${label}: <strong>${escapeHtml(value)}</strong></span>
      <span class="badge ${reportStatusClass(metric.status)}">${escapeHtml(metric.status || "n/a")}</span>
    </div>
  `;
}

function instabilityBodyMapReportHtml(map) {
  const markers = (map.parts || []).map((part) => `
    <div
      class="pdf-marker ${reportStatusClass(part.status)}"
      style="left: ${Number(part.x || 50)}%; top: ${Number(part.y || 50)}%;"
      title="${escapeHtml(part.label)}"
    >
      ${Math.round(Number(part.score || 0) * 100)}
    </div>
  `).join("");
  const legend = (map.parts || []).map((part) => `
    <div class="pdf-legend-row">
      <span class="badge ${reportStatusClass(part.status)}">${escapeHtml(part.statusLabel || part.status)}</span>
      <span><strong>${escapeHtml(part.label)}</strong><br><span class="muted">${Math.round(Number(part.score || 0) * 100)}% regional score</span></span>
    </div>
  `).join("");
  return `
    <div class="pdf-map-layout">
      <div class="pdf-body-card">
        <div class="pdf-human">
          <div class="pdf-head"></div>
          <div class="pdf-neck"></div>
          <div class="pdf-torso"></div>
          <div class="pdf-pelvis"></div>
          <div class="pdf-arm left"></div>
          <div class="pdf-arm right"></div>
          <div class="pdf-leg left"></div>
          <div class="pdf-leg right"></div>
          ${markers}
        </div>
      </div>
      <div class="pdf-legend">${legend}</div>
    </div>
  `;
}

function downloadInstabilityPdf(map) {
  if (!map?.available || !Array.isArray(map.parts)) return;
  const summary = map.summary || {};
  const partsHtml = map.parts.map((part) => `
    <section class="report-card">
      <h2>${escapeHtml(part.label)}</h2>
      <p><span class="badge ${reportStatusClass(part.status)}">${escapeHtml(part.statusLabel || part.status)}</span></p>
      <p>${escapeHtml(part.clinicalMeaning || "")}</p>
      ${(part.clinicalMetrics || []).map(clinicalMetricReportHtml).join("")}
      <p class="muted">Raw support: sway ${reportMetricValue(part.measurements?.sway)} | speed CV ${reportMetricValue(part.measurements?.velocityVariability)} | path ${reportMetricValue(part.measurements?.pathSway)}</p>
    </section>
  `).join("");
  openPdfReport("Instability Visualization", `
    <div class="summary-grid">
      <div class="summary-cell"><h2>${escapeHtml(summary.overallStatus || "-")}</h2><p>Overall map</p></div>
      <div class="summary-cell"><h2>${summary.severeCount ?? 0}</h2><p>High instability regions</p></div>
      <div class="summary-cell"><h2>${summary.moderateCount ?? 0}</h2><p>Moderate control regions</p></div>
    </div>
    ${instabilityBodyMapReportHtml(map)}
    <div class="body-grid">${partsHtml}</div>
  `);
}

function resultTone(screening) {
  if (screening?.reliabilityLevel === "Low") return "warning";
  if (screening?.detected) return "error";
  if (screening?.tendency) return "warning";
  return "success";
}

function gaitDecisionLabel(screening) {
  const model = screening?.modelLabel || String(screening?.modelKey || "gait").toUpperCase();
  if (screening?.reliabilityLevel === "Low") return `${model} gait result needs review`;
  if (screening?.detected) return `${model} gait detected`;
  if (screening?.tendency) return `Possible ${model} gait tendency`;
  return `Non-${model} gait pattern`;
}

function clinicalInterpretation(screening) {
  const model = screening?.modelKey === "koa" ? "KOA" : "SCA";
  const version = model === "KOA" ? "KOA v14" : "SCA v28";
  if (screening?.reliabilityLevel === "Low") {
    return `The ${version} version needs a longer or clearer walking recording for a reliable interpretation.`;
  }
  if (screening?.detected) {
    return `The ${version} version found repeated ${model}-like gait evidence.`;
  }
  if (screening?.tendency) {
    return `The ${version} version found some ${model}-like gait overlap, but not enough for a confirmed detection.`;
  }
  return `The ${version} version did not find repeated ${model}-like gait evidence.`;
}

function resultVisual(screening) {
  const model = screening?.modelLabel || String(screening?.modelKey || "gait").toUpperCase();
  if (screening?.reliabilityLevel === "Low") {
    return {
      className: "sca-koa-result-billboard warning",
      icon: <WarningAmberIcon />,
      label: gaitDecisionLabel(screening),
      caption: "The model ran, but the recording quality or duration was not strong enough for a confident gait-screening decision."
    };
  }
  if (screening?.detected) {
    return {
      className: "sca-koa-result-billboard detected",
      icon: <ErrorOutlineIcon />,
      label: gaitDecisionLabel(screening),
      caption: `This upload is screened as ${model} gait detected. The gait features match the trained ${model} model.`
    };
  }
  if (screening?.tendency) {
    return {
      className: "sca-koa-result-billboard warning",
      icon: <WarningAmberIcon />,
      label: gaitDecisionLabel(screening),
      caption: `This upload shows borderline overlap with the trained ${model} gait pattern. Review the probability and reliability notes before interpretation.`
    };
  }
  return {
    className: "sca-koa-result-billboard clear",
    icon: <CheckCircleIcon />,
    label: gaitDecisionLabel(screening),
    caption: `This upload is screened as a non-${model} gait pattern. No repeated ${model}-like gait evidence was found.`
  };
}

function Metric({ label, value, helper }) {
  return (
    <Box className="sca-koa-metric-tile">
      <Typography color="text.secondary">{label}</Typography>
      <Typography variant="h5" fontWeight={900}>{value ?? "-"}</Typography>
      {helper && <Typography className="sca-koa-metric-helper">{helper}</Typography>}
    </Box>
  );
}

function metricValue(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
  return Number(value).toFixed(3).replace(/\.?0+$/, "");
}

function clinicalStatusColor(status) {
  if (status === "red") return "error";
  if (status === "yellow") return "warning";
  if (status === "green") return "success";
  return "default";
}

function formatClinicalMetric(metric) {
  if (!metric) return "";
  const value = metric.value === null || metric.value === undefined || Number.isNaN(Number(metric.value))
    ? "-"
    : Number(metric.value).toFixed(Math.abs(Number(metric.value)) >= 100 ? 1 : 2).replace(/\.00$/, "");
  const confidence = metric.inferred ? " - estimated" : "";
  return `${metric.label || metric.key}: ${value}${metric.unit ? ` ${metric.unit}` : ""}${confidence}`;
}

function ClinicalMetricList({ metrics = [], compact = false }) {
  const severityRank = { red: 0, yellow: 1, green: 2, unavailable: 3 };
  const visible = [...metrics.filter(Boolean)]
    .sort((a, b) => (severityRank[a.status] ?? 4) - (severityRank[b.status] ?? 4))
    .slice(0, compact ? 6 : 8);
  if (visible.length === 0) return null;
  return (
    <Box className={compact ? "clinical-metric-stack compact" : "clinical-metric-stack"}>
      {visible.map((metric, index) => (
        <Box key={`${metric.key}-${index}`} className={`clinical-metric-pill ${metric.status || "unavailable"}`}>
          <span>{formatClinicalMetric(metric)}</span>
          <Chip size="small" color={clinicalStatusColor(metric.status)} label={metric.status || "n/a"} />
          {!compact && metric.confidence && <small>Confidence: {metric.confidence}</small>}
          {!compact && metric.referenceRange && <small>Reference: {metric.referenceRange}</small>}
        </Box>
      ))}
    </Box>
  );
}

export function ScaKoaOverview({ screening, totalScreenings = 1 }) {
  if (!screening) return null;
  const score = probability(screening.probability);
  const degrees = Math.round(score * 360);
  const gaugeBackground = `conic-gradient(#ef4444 0deg ${degrees}deg, #16a34a ${degrees}deg 360deg)`;
  const visual = resultVisual(screening);
  const activeModel = screening.modelKey === "koa" ? "KOA" : "SCA";

  return (
    <Stack spacing={3}>
      <Box className={visual.className}>
        <Box className="sca-koa-result-symbol">{visual.icon}</Box>
        <Box>
          <Typography className="sca-koa-result-kicker">Current {screening.modelLabel} gait screening result</Typography>
          <Typography component="h2" className="sca-koa-result-label">{visual.label}</Typography>
          <Typography className="sca-koa-result-caption">{visual.caption}</Typography>
        </Box>
      </Box>

      <Grid2 container spacing={3}>
        <Grid2 size={{ xs: 12, lg: 4 }}>
          <Card className="sca-koa-workspace-card sca-koa-gauge-card">
            <CardContent>
              <Stack spacing={2.5} alignItems="center" textAlign="center">
                <Box className="sca-koa-probability-gauge" sx={{ background: gaugeBackground }}>
                  <Box>
                    <Typography variant="h3" fontWeight={900}>{percent(score)}</Typography>
                    <Typography>{screening.modelLabel} probability</Typography>
                  </Box>
                </Box>
                <Box>
                  <Typography variant="h5" fontWeight={900}>{gaitDecisionLabel(screening)}</Typography>
                  <Typography color="text.secondary">{clinicalInterpretation(screening)}</Typography>
                </Box>
                <LinearProgress className="sca-koa-confidence-bar" variant="determinate" value={Math.round(score * 100)} />
              </Stack>
            </CardContent>
          </Card>
        </Grid2>

        <Grid2 size={{ xs: 12, lg: 8 }}>
          <Grid2 container spacing={2}>
            <Grid2 size={{ xs: 12, md: 4 }}>
              <Metric label="Latest result" value={gaitDecisionLabel(screening)} helper="Saved model decision" />
            </Grid2>
            <Grid2 size={{ xs: 12, md: 4 }}>
              <Metric label="Reliability" value={screening.reliabilityLevel || "-"} helper="Model quality level" />
            </Grid2>
            <Grid2 size={{ xs: 12, md: 4 }}>
              <Metric label="Direction" value={screening.direction || "-"} helper="Estimated gait direction" />
            </Grid2>
            <Grid2 size={{ xs: 12, md: 4 }}>
              <Metric label="Max probability" value={percent(screening.maxProbability)} helper="Highest window score" />
            </Grid2>
            <Grid2 size={{ xs: 12, md: 4 }}>
              <Metric label="Positive windows" value={screening.positiveWindowCount ?? "-"} helper={`Ratio ${percent(screening.positiveWindowRatio)}`} />
            </Grid2>
            <Grid2 size={{ xs: 12, md: 4 }}>
              <Metric label={`Total ${activeModel} screenings`} value={totalScreenings} helper={`Saved ${activeModel} model runs`} />
            </Grid2>
          </Grid2>
        </Grid2>
      </Grid2>

      <Card className="sca-koa-workspace-card">
        <CardContent>
          <Stack spacing={2}>
            <Typography variant="h5" fontWeight={900}>Reliability Notes</Typography>
            {(screening.reliabilityReasons || []).length === 0 ? (
              <Alert severity="info">No additional reliability notes were returned for this screening.</Alert>
            ) : (
              screening.reliabilityReasons.map((note) => (
                <Alert key={note} severity={screening.reliabilityLevel === "High" ? "success" : "warning"}>{note}</Alert>
              ))
            )}
          </Stack>
        </CardContent>
      </Card>
    </Stack>
  );
}

export function InstabilityBodyMap({ map }) {
  if (!map?.available || !Array.isArray(map.parts)) {
    return (
      <Alert severity="info">
        Instability visualization will appear after a new SCA or KOA model run saves training-safe gait measurements.
      </Alert>
    );
  }

  return (
    <Grid2 container spacing={3}>
      <Grid2 size={{ xs: 12 }}>
        <Stack direction={{ xs: "column", md: "row" }} spacing={1.5} alignItems={{ xs: "stretch", md: "center" }} justifyContent="space-between" className="export-toolbar">
          <Box>
            <Typography fontWeight={900}>Instability visualization report</Typography>
            <Typography color="text.secondary">Download the full body-region map with clinical metric drivers.</Typography>
          </Box>
          <Button variant="outlined" startIcon={<PictureAsPdfIcon />} onClick={() => downloadInstabilityPdf(map)}>
            Download PDF
          </Button>
        </Stack>
      </Grid2>
      <Grid2 size={{ xs: 12, lg: 5 }}>
        <Box className="instability-body-card">
          <Box className="human-map">
            <Box className="human-head" />
            <Box className="human-neck" />
            <Box className="human-torso" />
            <Box className="human-pelvis" />
            <Box className="human-arm left" />
            <Box className="human-arm right" />
            <Box className="human-leg left" />
            <Box className="human-leg right" />
            {map.parts.map((part) => (
              <Tooltip
                key={part.key}
                arrow
                title={
                  <Box className="instability-tooltip">
                    <Typography fontWeight={900}>{part.label}</Typography>
                    <Typography>{part.statusLabel}</Typography>
                    {part.clinicalMeaning && <Typography>{part.clinicalMeaning}</Typography>}
                    <ClinicalMetricList metrics={part.clinicalMetrics} />
                    <Typography>Motion range: {metricValue(part.measurements?.motionRange)}</Typography>
                    <Typography>Sway: {metricValue(part.measurements?.sway)}</Typography>
                    <Typography>Speed variability ratio: {metricValue(part.measurements?.velocityVariability)}</Typography>
                    <Typography>Jerk irregularity ratio: {metricValue(part.measurements?.jerkIrregularity)}</Typography>
                    <Typography>Landmark jitter: {metricValue(part.measurements?.jitter)}</Typography>
                    {part.measurements?.footClearance !== undefined && <Typography>Foot clearance: {metricValue(part.measurements?.footClearance)}</Typography>}
                    <Typography>Left/right asymmetry: {metricValue(part.measurements?.leftRightAsymmetry)}</Typography>
                  </Box>
                }
              >
                <Box
                  className={`body-marker ${part.status}`}
                  style={{ left: `${part.x}%`, top: `${part.y}%` }}
                >
                  <span>{Math.round(Number(part.score || 0) * 100)}</span>
                </Box>
              </Tooltip>
            ))}
          </Box>
        </Box>
      </Grid2>
      <Grid2 size={{ xs: 12, lg: 7 }}>
        <Stack spacing={2}>
          <Grid2 container spacing={2}>
            <Grid2 size={{ xs: 12, md: 4 }}>
              <Metric label="Overall map" value={map.summary?.overallStatus || "-"} helper="Derived from the saved training-safe CSV" />
            </Grid2>
            <Grid2 size={{ xs: 12, md: 4 }}>
              <Metric label="High instability" value={map.summary?.severeCount ?? 0} helper="Red body regions" />
            </Grid2>
            <Grid2 size={{ xs: 12, md: 4 }}>
              <Metric label="Moderate control regions" value={map.summary?.moderateCount ?? 0} helper="Yellow body regions" />
            </Grid2>
          </Grid2>
          <Box className="instability-region-list">
            {map.parts.map((part) => (
              <Box key={part.key} className={`instability-region-row ${part.status}`}>
                <Box>
                  <Typography fontWeight={900}>{part.label}</Typography>
                  {part.clinicalMeaning && <Typography className="sca-koa-metric-helper">{part.clinicalMeaning}</Typography>}
                  <ClinicalMetricList metrics={part.clinicalMetrics} compact />
                  <Typography color="text.secondary">
                    Side-view measures drive the color. Raw support: sway {metricValue(part.measurements?.sway)} | speed CV {metricValue(part.measurements?.velocityVariability)} | path {metricValue(part.measurements?.pathSway)}
                  </Typography>
                </Box>
                <Chip label={part.statusLabel} size="small" />
              </Box>
            ))}
          </Box>
        </Stack>
      </Grid2>
    </Grid2>
  );
}

const relationOptions = [
  "Mother",
  "Father",
  "Sister",
  "Brother",
  "Daughter",
  "Son",
  "Grandmother",
  "Grandfather",
  "Aunt",
  "Uncle",
  "Cousin",
  "Other relative"
];

function blankRelative() {
  return {
    id: crypto.randomUUID?.() || `rel_${Date.now()}`,
    relation: "",
    familySide: "unknown",
    scaStatus: "",
    geneticConfirmed: false,
    ageOfOnset: "",
    gaitNotes: ""
  };
}

function blankSuspected() {
  return {
    id: crypto.randomUUID?.() || `sus_${Date.now()}`,
    relation: "",
    familySide: "unknown",
    gaitPattern: "",
    ageNoticed: "",
    notes: ""
  };
}

function relationWeight(relation = "") {
  const text = String(relation).toLowerCase();
  if (["mother", "father", "sister", "brother", "daughter", "son"].some((item) => text.includes(item))) return 32;
  if (["grandmother", "grandfather", "aunt", "uncle"].some((item) => text.includes(item))) return 22;
  if (text.includes("cousin")) return 12;
  return 8;
}

function buildAwarenessPreview(form) {
  let score = 0;
  form.relatives.forEach((relative) => {
    if (!relative.relation && !relative.scaStatus && !relative.ageOfOnset && !relative.gaitNotes && !relative.geneticConfirmed) return;
    if (relative.scaStatus === "diagnosed" || relative.scaStatus === "positive_genetic_test") score += relationWeight(relative.relation);
    if (relative.scaStatus === "suspected") score += relationWeight(relative.relation) * 0.55;
    if (relative.geneticConfirmed) score += 12;
    const onset = Number(relative.ageOfOnset);
    if (Number.isFinite(onset) && onset > 0 && onset < 45) score += 8;
  });
  form.suspected.forEach((relative) => {
    if (relative.relation || relative.gaitPattern) score += Math.max(6, relationWeight(relative.relation) * 0.35);
  });
  if (form.answers.knownFamilySca === "yes") score += 16;
  if (form.answers.familyAbnormalGait === "yes") score += 10;
  if (form.answers.geneticTesting === "positive") score += 24;
  if (form.answers.geneticTesting === "negative") score -= 10;
  const awarenessScore = Math.max(0, Math.min(100, Math.round(score)));
  return {
    awarenessScore,
    level: awarenessScore >= 70 ? "High family-history awareness" : awarenessScore >= 35 ? "Moderate family-history awareness" : "Low documented family-history awareness",
    relatives: form.relatives.filter((item) => item.relation || item.scaStatus || item.ageOfOnset || item.gaitNotes || item.geneticConfirmed),
    suspected: form.suspected.filter((item) => item.relation || item.gaitPattern)
  };
}

function familySideLabel(side) {
  if (side === "maternal") return "Maternal side";
  if (side === "paternal") return "Paternal side";
  if (side === "both") return "Both sides";
  return "Side unknown";
}

function relationInitial(relation = "") {
  const clean = String(relation || "Relative").trim();
  return clean.slice(0, 2).toUpperCase();
}

function geneticNodeReportHtml(item, type) {
  const title = escapeHtml(item.relation || "Relative");
  const side = escapeHtml(familySideLabel(item.familySide));
  if (type === "diagnosed") {
    return `
      <div class="family-node">
        <strong>${title}</strong>
        <p>${side}</p>
        <p>${escapeHtml(String(item.scaStatus || "SCA history").replaceAll("_", " "))}${item.ageOfOnset ? ` | onset ${escapeHtml(item.ageOfOnset)}` : ""}${item.geneticConfirmed ? " | genetic test confirmed" : ""}</p>
        ${item.gaitNotes ? `<p>${escapeHtml(item.gaitNotes)}</p>` : ""}
      </div>
    `;
  }
  return `
    <div class="family-node">
      <strong>${title}</strong>
      <p>${side}</p>
      <p>${escapeHtml(item.gaitPattern || "Abnormal gait behavior")}${item.ageNoticed ? ` | noticed ${escapeHtml(item.ageNoticed)}` : ""}</p>
      ${item.notes ? `<p>${escapeHtml(item.notes)}</p>` : ""}
    </div>
  `;
}

function downloadGeneticsPdf(awareness) {
  const diagnosed = awareness?.relatives || [];
  const suspected = awareness?.suspected || [];
  openPdfReport("SCA Genetic Awareness Chart", `
    <div class="family-map">
      <section class="report-card">
        <div class="lane-title">Relatives with SCA</div>
        ${diagnosed.length ? diagnosed.map((item) => geneticNodeReportHtml(item, "diagnosed")).join("") : "<p>No SCA relatives entered.</p>"}
      </section>
      <section class="patient">
        <h2>Patient</h2>
        <div class="score">${awareness?.awarenessScore || 0}</div>
        <p><strong>${escapeHtml(awareness?.level || "Family-history awareness")}</strong></p>
      </section>
      <section class="report-card">
        <div class="lane-title">Suspected gait behavior</div>
        ${suspected.length ? suspected.map((item) => geneticNodeReportHtml(item, "suspected")).join("") : "<p>No suspected gait-behavior relatives entered.</p>"}
      </section>
    </div>
  `);
}

export function GeneticChartV2({ awareness }) {
  const diagnosed = awareness?.relatives || [];
  const suspected = awareness?.suspected || [];
  return (
    <Box className="genetic-chart-shell">
      <Stack direction={{ xs: "column", md: "row" }} spacing={1.5} alignItems={{ xs: "stretch", md: "center" }} justifyContent="space-between" className="genetic-chart-actions">
        <Box>
          <Typography className="genetic-lane-title">Interactive family awareness map</Typography>
          <Typography color="text.secondary">Patient stays in the center while SCA relatives and suspected gait behavior update around them.</Typography>
        </Box>
        <Button variant="outlined" startIcon={<PictureAsPdfIcon />} onClick={() => downloadGeneticsPdf(awareness)}>
          Download PDF
        </Button>
      </Stack>
      <Box className="genetic-chart">
        <Box className="genetic-lane diagnosed">
          <Typography className="genetic-lane-title">Relatives with SCA</Typography>
          <Stack spacing={1.2} className="genetic-node-stack">
            {diagnosed.length === 0 ? (
              <Typography color="text.secondary">No SCA relatives entered yet.</Typography>
            ) : diagnosed.map((item) => (
              <Box key={item.id} className={`genetic-node diagnosed ${item.familySide || "unknown"}`}>
                <Box className="genetic-avatar">{relationInitial(item.relation)}</Box>
                <Box>
                  <Typography fontWeight={900}>{item.relation || "Relative"} - {familySideLabel(item.familySide)}</Typography>
                  <Typography color="text.secondary">
                    {item.scaStatus?.replaceAll("_", " ")}{item.ageOfOnset ? ` | onset ${item.ageOfOnset}` : ""}{item.geneticConfirmed ? " | genetic test confirmed" : ""}
                  </Typography>
                  {item.gaitNotes && <Typography>{item.gaitNotes}</Typography>}
                </Box>
              </Box>
            ))}
          </Stack>
        </Box>
        <Box className="genetic-chart-root">
          <Typography fontWeight={950}>Patient</Typography>
          <Typography color="text.secondary">{awareness?.level || "Family-history awareness"}</Typography>
          <Box className="genetic-score-ring" style={{ "--score": `${awareness?.awarenessScore || 0}%` }}>
            <span>{awareness?.awarenessScore || 0}</span>
          </Box>
          <Typography className="genetic-center-note">Family-history awareness score</Typography>
        </Box>
        <Box className="genetic-lane suspected">
          <Typography className="genetic-lane-title">Suspected gait behavior</Typography>
          <Stack spacing={1.2} className="genetic-node-stack">
            {suspected.length === 0 ? (
              <Typography color="text.secondary">No suspected gait-behavior relatives entered yet.</Typography>
            ) : suspected.map((item) => (
              <Box key={item.id} className={`genetic-node suspected ${item.familySide || "unknown"}`}>
                <Box className="genetic-avatar">{relationInitial(item.relation)}</Box>
                <Box>
                  <Typography fontWeight={900}>{item.relation || "Relative"} - {familySideLabel(item.familySide)}</Typography>
                  <Typography color="text.secondary">{item.gaitPattern || "Abnormal gait behavior"}{item.ageNoticed ? ` | noticed ${item.ageNoticed}` : ""}</Typography>
                  {item.notes && <Typography>{item.notes}</Typography>}
                </Box>
              </Box>
            ))}
          </Stack>
        </Box>
      </Box>
    </Box>
  );
}

function GeneticChart({ awareness }) {
  const diagnosed = awareness?.relatives || [];
  const suspected = awareness?.suspected || [];
  return (
    <Box className="genetic-chart">
      <Box className="genetic-chart-root">
        <Typography fontWeight={950}>Patient</Typography>
        <Typography color="text.secondary">{awareness?.level || "Family-history awareness"}</Typography>
        <Box className="genetic-score-ring" style={{ "--score": `${awareness?.awarenessScore || 0}%` }}>
          <span>{awareness?.awarenessScore || 0}</span>
        </Box>
      </Box>
      <Grid2 container spacing={2}>
        <Grid2 size={{ xs: 12, md: 6 }}>
          <Typography className="genetic-lane-title">Relatives with SCA condition</Typography>
          <Stack spacing={1.2}>
            {diagnosed.length === 0 ? (
              <Typography color="text.secondary">No SCA relatives entered yet.</Typography>
            ) : diagnosed.map((item) => (
              <Box key={item.id} className="genetic-node diagnosed">
                <Typography fontWeight={900}>{item.relation || "Relative"} · {item.familySide}</Typography>
                <Typography color="text.secondary">
                  {item.scaStatus?.replaceAll("_", " ")}{item.ageOfOnset ? ` | onset ${item.ageOfOnset}` : ""}{item.geneticConfirmed ? " | genetic test confirmed" : ""}
                </Typography>
                {item.gaitNotes && <Typography>{item.gaitNotes}</Typography>}
              </Box>
            ))}
          </Stack>
        </Grid2>
        <Grid2 size={{ xs: 12, md: 6 }}>
          <Typography className="genetic-lane-title">Suspected abnormal gait relatives</Typography>
          <Stack spacing={1.2}>
            {suspected.length === 0 ? (
              <Typography color="text.secondary">No suspected gait-behavior relatives entered yet.</Typography>
            ) : suspected.map((item) => (
              <Box key={item.id} className="genetic-node suspected">
                <Typography fontWeight={900}>{item.relation || "Relative"} · {item.familySide}</Typography>
                <Typography color="text.secondary">{item.gaitPattern || "Abnormal gait behavior"}{item.ageNoticed ? ` | noticed ${item.ageNoticed}` : ""}</Typography>
                {item.notes && <Typography>{item.notes}</Typography>}
              </Box>
            ))}
          </Stack>
        </Grid2>
      </Grid2>
    </Box>
  );
}

function GeneticAwarenessPanel({ screening, genetics, form, setForm, onSave, saving, error }) {
  const awareness = buildAwarenessPreview(form);

  function updateAnswer(name, value) {
    setForm((old) => ({ ...old, answers: { ...old.answers, [name]: value } }));
  }

  function updateRelative(index, patch) {
    setForm((old) => ({
      ...old,
      relatives: old.relatives.map((item, itemIndex) => itemIndex === index ? { ...item, ...patch } : item)
    }));
  }

  function updateSuspected(index, patch) {
    setForm((old) => ({
      ...old,
      suspected: old.suspected.map((item, itemIndex) => itemIndex === index ? { ...item, ...patch } : item)
    }));
  }

  if (screening.modelKey !== "sca" || (!screening.detected && !screening.tendency)) {
    return (
      <Alert severity="info">
        The SCA genetic awareness form appears after an SCA detection or borderline SCA tendency.
      </Alert>
    );
  }

  return (
    <Grid2 container spacing={3}>
      <Grid2 size={{ xs: 12, lg: 5 }}>
        <Stack spacing={2}>
          {error && <Alert severity="error">{error}</Alert>}
          <Card className="sca-koa-workspace-card">
            <CardContent>
              <Stack spacing={2}>
                <Stack direction="row" spacing={1.5} alignItems="center">
                  <FamilyRestroomIcon color="primary" />
                  <Box>
                    <Typography variant="h5" fontWeight={900}>SCA genetic awareness questions</Typography>
                    <Typography color="text.secondary">Family-history signals, confirmed SCA relatives, and relatives with abnormal gait behavior.</Typography>
                  </Box>
                </Stack>
                <FormControl fullWidth>
                  <InputLabel>Known SCA in family</InputLabel>
                  <Select label="Known SCA in family" value={form.answers.knownFamilySca} onChange={(event) => updateAnswer("knownFamilySca", event.target.value)}>
                    <MenuItem value="unknown">Unknown</MenuItem>
                    <MenuItem value="yes">Yes</MenuItem>
                    <MenuItem value="no">No</MenuItem>
                  </Select>
                </FormControl>
                <FormControl fullWidth>
                  <InputLabel>Family abnormal gait history</InputLabel>
                  <Select label="Family abnormal gait history" value={form.answers.familyAbnormalGait} onChange={(event) => updateAnswer("familyAbnormalGait", event.target.value)}>
                    <MenuItem value="unknown">Unknown</MenuItem>
                    <MenuItem value="yes">Yes</MenuItem>
                    <MenuItem value="no">No</MenuItem>
                  </Select>
                </FormControl>
                <FormControl fullWidth>
                  <InputLabel>Patient genetic testing</InputLabel>
                  <Select label="Patient genetic testing" value={form.answers.geneticTesting} onChange={(event) => updateAnswer("geneticTesting", event.target.value)}>
                    <MenuItem value="not_tested">Not tested</MenuItem>
                    <MenuItem value="positive">Positive / known expansion</MenuItem>
                    <MenuItem value="negative">Negative</MenuItem>
                    <MenuItem value="pending">Pending</MenuItem>
                  </Select>
                </FormControl>
                <TextField
                  label="Clinical/family notes"
                  multiline
                  minRows={3}
                  value={form.answers.notes}
                  onChange={(event) => updateAnswer("notes", event.target.value)}
                />
                <Button variant="contained" startIcon={<SaveIcon />} disabled={saving} onClick={onSave}>
                  {saving ? "Saving chart..." : "Submit genetic awareness chart"}
                </Button>
              </Stack>
            </CardContent>
          </Card>
        </Stack>
      </Grid2>
      <Grid2 size={{ xs: 12, lg: 7 }}>
        <Stack spacing={2}>
          <Card className="sca-koa-workspace-card">
            <CardContent>
              <Stack spacing={2}>
                <Stack direction={{ xs: "column", md: "row" }} justifyContent="space-between" spacing={1.5}>
                  <Box>
                    <Typography variant="h5" fontWeight={900}>Relatives with SCA</Typography>
                    <Typography color="text.secondary">Add diagnosed, genetically confirmed, or suspected SCA relatives.</Typography>
                  </Box>
                  <Button startIcon={<AddCircleOutlineIcon />} onClick={() => setForm((old) => ({ ...old, relatives: [...old.relatives, blankRelative()] }))}>
                    Add relative
                  </Button>
                </Stack>
                {form.relatives.map((relative, index) => (
                  <Box key={relative.id} className="genetic-form-row">
                    <FormControl>
                      <InputLabel>Relation</InputLabel>
                      <Select label="Relation" value={relative.relation} onChange={(event) => updateRelative(index, { relation: event.target.value })}>
                        <MenuItem value="">Select relation</MenuItem>
                        {relationOptions.map((relation) => <MenuItem key={relation} value={relation}>{relation}</MenuItem>)}
                      </Select>
                    </FormControl>
                    <FormControl>
                      <InputLabel>Family side</InputLabel>
                      <Select label="Family side" value={relative.familySide} onChange={(event) => updateRelative(index, { familySide: event.target.value })}>
                        <MenuItem value="maternal">Maternal</MenuItem>
                        <MenuItem value="paternal">Paternal</MenuItem>
                        <MenuItem value="both">Both/unknown</MenuItem>
                        <MenuItem value="unknown">Unknown</MenuItem>
                      </Select>
                    </FormControl>
                    <FormControl>
                      <InputLabel>SCA status</InputLabel>
                      <Select label="SCA status" value={relative.scaStatus} onChange={(event) => updateRelative(index, { scaStatus: event.target.value })}>
                        <MenuItem value="">Select SCA status</MenuItem>
                        <MenuItem value="diagnosed">Diagnosed SCA</MenuItem>
                        <MenuItem value="positive_genetic_test">Positive genetic test</MenuItem>
                        <MenuItem value="suspected">Suspected SCA</MenuItem>
                        <MenuItem value="unknown">Unknown</MenuItem>
                      </Select>
                    </FormControl>
                    <TextField label="Onset age" value={relative.ageOfOnset} onChange={(event) => updateRelative(index, { ageOfOnset: event.target.value })} />
                    <TextField label="Gait notes" value={relative.gaitNotes} onChange={(event) => updateRelative(index, { gaitNotes: event.target.value })} />
                    <Button color="error" onClick={() => setForm((old) => ({ ...old, relatives: old.relatives.filter((_, itemIndex) => itemIndex !== index) }))}>Remove</Button>
                  </Box>
                ))}
              </Stack>
            </CardContent>
          </Card>

          <Card className="sca-koa-workspace-card">
            <CardContent>
              <Stack spacing={2}>
                <Stack direction={{ xs: "column", md: "row" }} justifyContent="space-between" spacing={1.5}>
                  <Box>
                    <Typography variant="h5" fontWeight={900}>Suspected abnormal gait relatives</Typography>
                    <Typography color="text.secondary">Capture relatives with unsteady gait, falls, coordination loss, widened stance, or similar behavior.</Typography>
                  </Box>
                  <Button startIcon={<AddCircleOutlineIcon />} onClick={() => setForm((old) => ({ ...old, suspected: [...old.suspected, blankSuspected()] }))}>
                    Add gait history
                  </Button>
                </Stack>
                {form.suspected.map((relative, index) => (
                  <Box key={relative.id} className="genetic-form-row suspected">
                    <FormControl>
                      <InputLabel>Relation</InputLabel>
                      <Select label="Relation" value={relative.relation} onChange={(event) => updateSuspected(index, { relation: event.target.value })}>
                        <MenuItem value="">Select relation</MenuItem>
                        {relationOptions.map((relation) => <MenuItem key={relation} value={relation}>{relation}</MenuItem>)}
                      </Select>
                    </FormControl>
                    <FormControl>
                      <InputLabel>Family side</InputLabel>
                      <Select label="Family side" value={relative.familySide} onChange={(event) => updateSuspected(index, { familySide: event.target.value })}>
                        <MenuItem value="maternal">Maternal</MenuItem>
                        <MenuItem value="paternal">Paternal</MenuItem>
                        <MenuItem value="both">Both/unknown</MenuItem>
                        <MenuItem value="unknown">Unknown</MenuItem>
                      </Select>
                    </FormControl>
                    <TextField label="Observed gait behavior" value={relative.gaitPattern} onChange={(event) => updateSuspected(index, { gaitPattern: event.target.value })} />
                    <TextField label="Age noticed" value={relative.ageNoticed} onChange={(event) => updateSuspected(index, { ageNoticed: event.target.value })} />
                    <TextField label="Notes" value={relative.notes} onChange={(event) => updateSuspected(index, { notes: event.target.value })} />
                    <Button color="error" onClick={() => setForm((old) => ({ ...old, suspected: old.suspected.filter((_, itemIndex) => itemIndex !== index) }))}>Remove</Button>
                  </Box>
                ))}
              </Stack>
            </CardContent>
          </Card>
        </Stack>
      </Grid2>
      <Grid2 size={{ xs: 12 }}>
        <Card className="sca-koa-workspace-card">
          <CardContent>
            <Stack spacing={2}>
              <Typography variant="h5" fontWeight={900}>Dynamic genetic awareness chart</Typography>
              <GeneticChartV2 awareness={awareness} />
            </Stack>
          </CardContent>
        </Card>
      </Grid2>
    </Grid2>
  );
}

export default function ScaKoaResults() {
  const location = useLocation();
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const requestedId = searchParams.get("id");
  const [screening, setScreening] = useState(location.state?.screening || null);
  const [profile, setProfile] = useState(location.state?.profile || null);
  const [screenings, setScreenings] = useState([]);
  const [loading, setLoading] = useState(!location.state?.screening);
  const [error, setError] = useState("");
  const [deletingId, setDeletingId] = useState("");
  const [navCollapsed, setNavCollapsed] = useState(false);
  const [genetics, setGenetics] = useState(null);
  const [geneticsForm, setGeneticsForm] = useState({
    answers: {
      knownFamilySca: "unknown",
      familyAbnormalGait: "unknown",
      geneticTesting: "not_tested",
      notes: ""
    },
    relatives: [blankRelative()],
    suspected: [blankSuspected()]
  });
  const [geneticsSaving, setGeneticsSaving] = useState(false);
  const [geneticsError, setGeneticsError] = useState("");
  const activeSection = searchParams.get("section") || "overview";

  useEffect(() => {
    let alive = true;
    async function load() {
      setLoading(true);
      setError("");
      try {
        const profileRes = await api.get("/sca-koa/clinical-profile");
        if (!alive) return;
        setProfile(profileRes.data.profile);
        setScreenings(profileRes.data.screenings || []);
        if (requestedId) {
          const screeningRes = await api.get(`/sca-koa/screenings/${requestedId}`);
          if (!alive) return;
          setScreening(screeningRes.data.screening);
        } else if (!location.state?.screening) {
          setScreening(profileRes.data.screenings?.[0] || null);
        }
      } catch (err) {
        if (!alive) return;
        const apiError = getApiError(err);
        setError(apiError.message || "Unable to load the clinical profile.");
      } finally {
        if (alive) setLoading(false);
      }
    }
    load();
    return () => {
      alive = false;
    };
  }, [requestedId, location.state?.screening]);

  useEffect(() => {
    if (screening?.modelKey !== "sca" || (!screening.detected && !screening.tendency)) {
      setGenetics(null);
      return;
    }

    let alive = true;
    setGeneticsError("");
    api.get(`/sca-koa/screenings/${screening.id}/genetics`)
      .then((res) => {
        if (!alive) return;
        const saved = res.data.genetics;
        setGenetics(saved);
        if (saved) {
          setGeneticsForm({
            answers: {
              knownFamilySca: saved.answers?.knownFamilySca || "unknown",
              familyAbnormalGait: saved.answers?.familyAbnormalGait || "unknown",
              geneticTesting: saved.answers?.geneticTesting || "not_tested",
              notes: saved.answers?.notes || ""
            },
            relatives: saved.relatives?.length ? saved.relatives : [blankRelative()],
            suspected: saved.suspected?.length ? saved.suspected : [blankSuspected()]
          });
        }
      })
      .catch((err) => {
        if (!alive) return;
        const apiError = getApiError(err);
        setGeneticsError(apiError.message || "Unable to load SCA genetic awareness chart.");
      });
    return () => {
      alive = false;
    };
  }, [screening?.id, screening?.modelKey, screening?.detected, screening?.tendency]);

  const score = probability(screening?.probability);
  const gaugeBackground = useMemo(() => {
    const degrees = Math.round(score * 360);
    return `conic-gradient(#ef4444 0deg ${degrees}deg, #16a34a ${degrees}deg 360deg)`;
  }, [score]);

  function getSectionLink(section) {
    const params = new URLSearchParams();
    if (screening?.id) params.set("id", screening.id);
    if (section !== "overview") params.set("section", section);
    const query = params.toString();
    return `/patient/results/sca-koa${query ? `?${query}` : ""}`;
  }

  async function deleteScreening(id) {
    if (!window.confirm("Delete this saved gait-screening result?")) return;
    setDeletingId(id);
    setError("");
    try {
      const res = await api.delete(`/sca-koa/screenings/${id}`);
      const remaining = res.data.screenings || [];
      setProfile(res.data.profile);
      setScreenings(remaining);
      if (id === screening?.id) {
        const nextScreening = remaining[0] || null;
        setScreening(nextScreening);
        navigate(nextScreening ? `/patient/results/sca-koa?id=${nextScreening.id}` : "/patient/results/sca-koa", { replace: true });
      }
    } catch (err) {
      const apiError = getApiError(err);
      setError(apiError.message || "Unable to delete screening result.");
    } finally {
      setDeletingId("");
    }
  }

  async function saveGenetics() {
    if (!screening?.id) return;
    setGeneticsSaving(true);
    setGeneticsError("");
    try {
      const res = await api.put(`/sca-koa/screenings/${screening.id}/genetics`, geneticsForm);
      setGenetics(res.data.genetics);
      setGeneticsForm({
        answers: {
          knownFamilySca: res.data.genetics.answers?.knownFamilySca || "unknown",
          familyAbnormalGait: res.data.genetics.answers?.familyAbnormalGait || "unknown",
          geneticTesting: res.data.genetics.answers?.geneticTesting || "not_tested",
          notes: res.data.genetics.answers?.notes || ""
        },
        relatives: res.data.genetics.relatives?.length ? res.data.genetics.relatives : [blankRelative()],
        suspected: res.data.genetics.suspected?.length ? res.data.genetics.suspected : [blankSuspected()]
      });
    } catch (err) {
      const apiError = getApiError(err);
      setGeneticsError(apiError.message || "Unable to save SCA genetic awareness chart.");
    } finally {
      setGeneticsSaving(false);
    }
  }

  if (loading) {
    return (
      <Container maxWidth="lg" className="sca-koa-page">
        <Card className="sca-koa-workspace-card">
          <CardContent>
            <Stack spacing={2} alignItems="center" py={5}>
              <CircularProgress />
              <Typography>Loading clinical profile...</Typography>
            </Stack>
          </CardContent>
        </Card>
      </Container>
    );
  }

  if (error) {
    return (
      <Container maxWidth="lg" className="sca-koa-page">
        <Alert severity="error">{error}</Alert>
      </Container>
    );
  }

  if (!screening) {
    return (
      <Container maxWidth="lg" className="sca-koa-page">
        <Card className="sca-koa-workspace-card">
          <CardContent>
            <Stack spacing={2} alignItems="center" textAlign="center" py={5}>
              <AssessmentIcon color="primary" />
              <Typography variant="h4" fontWeight={900}>No SCA or KOA clinical profile yet</Typography>
              <Typography color="text.secondary">Run the first SCA or KOA screening and this page will show the saved outcome profile.</Typography>
              <Button component={Link} to="/patient/detection/sca-koa" variant="contained" startIcon={<ReplayIcon />}>
                Start detection
              </Button>
            </Stack>
          </CardContent>
        </Card>
      </Container>
    );
  }

  const visual = resultVisual(screening);
  const itemClass = navCollapsed ? "sca-koa-nav-label hidden" : "sca-koa-nav-label";
  const sameModelScreenings = screenings.filter((item) => item.modelKey === screening.modelKey);
  const activeModel = screening.modelKey === "koa" ? "KOA" : "SCA";

  return (
    <Container maxWidth="xl" className="sca-koa-page">
      <Grid2 container spacing={3}>
        <Grid2 size={{ xs: 12, md: navCollapsed ? 1.1 : 3, lg: navCollapsed ? 0.9 : 2.4 }}>
          <Box className={navCollapsed ? "sca-koa-side-nav collapsed" : "sca-koa-side-nav"}>
            <Button
              className="sca-koa-nav-toggle"
              variant="outlined"
              onClick={() => setNavCollapsed((old) => !old)}
              startIcon={navCollapsed ? <ChevronRightIcon /> : <ChevronLeftIcon />}
            >
              <span className={itemClass}>{navCollapsed ? "" : "Collapse"}</span>
            </Button>
            <Button component={Link} to={`/patient/detection/sca-koa?model=${screening.modelKey}`} variant="outlined" startIcon={<VideoCameraBackIcon />}>
              <span className={itemClass}>Detection</span>
            </Button>
            <Button component={Link} to={getSectionLink("overview")} variant={activeSection === "overview" ? "contained" : "text"} startIcon={<AssessmentIcon />}>
              <span className={itemClass}>Overview</span>
            </Button>
            <Button component={Link} to={getSectionLink("history")} variant={activeSection === "history" ? "contained" : "text"} startIcon={<AccessibilityNewIcon />}>
              <span className={itemClass}>History</span>
            </Button>
            <Button component={Link} to={getSectionLink("instability")} variant={activeSection === "instability" ? "contained" : "text"} startIcon={<AccessibilityNewIcon />}>
              <span className={itemClass}>Instability</span>
            </Button>
            <Button component={Link} to={getSectionLink("genetics")} variant={activeSection === "genetics" ? "contained" : "text"} startIcon={<BiotechIcon />}>
              <span className={itemClass}>SCA Genetics</span>
            </Button>
            <Button component={Link} to="/patient/central-profile" variant="text" startIcon={<AccountTreeIcon />}>
              <span className={itemClass}>Centralized Profile</span>
            </Button>
            <Button component={Link} to="/patient" variant="text">
              <span className={itemClass}>Dashboard</span>
            </Button>
          </Box>
        </Grid2>

        <Grid2 size={{ xs: 12, md: navCollapsed ? 10.9 : 9, lg: navCollapsed ? 11.1 : 9.6 }}>
          <Stack spacing={3}>
            <Box className="sca-koa-hero compact">
              <Stack spacing={1.5}>
                <Chip color={resultTone(screening)} label={`${screening.modelLabel} result`} />
                <Typography variant="h3" fontWeight={900}>{activeModel} Clinical Profile</Typography>
                <Typography color="text.secondary">Saved {activeModel} gait outcomes, reliability notes, probability scores, and longitudinal screening history.</Typography>
              </Stack>
              <Stack direction={{ xs: "column", sm: "row" }} spacing={1.2}>
                <CentralProfileFlagButton sourceType={screening.modelKey} screeningId={screening.id} />
                <Button component={Link} to={`/patient/detection/sca-koa?model=${screening.modelKey}`} variant="contained" startIcon={<ReplayIcon />}>
                  New screening
                </Button>
                <Button color="error" variant="outlined" startIcon={<DeleteOutlineIcon />} disabled={deletingId === screening.id} onClick={() => deleteScreening(screening.id)}>
                  Delete result
                </Button>
              </Stack>
            </Box>

            {activeSection === "overview" && (
              <>
                <Box className={visual.className}>
                  <Box className="sca-koa-result-symbol">{visual.icon}</Box>
                  <Box>
                    <Typography className="sca-koa-result-kicker">Current {screening.modelLabel} gait screening result</Typography>
                    <Typography component="h2" className="sca-koa-result-label">{visual.label}</Typography>
                    <Typography className="sca-koa-result-caption">{visual.caption}</Typography>
                  </Box>
                </Box>

                <Grid2 container spacing={3}>
                  <Grid2 size={{ xs: 12, lg: 4 }}>
                    <Card className="sca-koa-workspace-card sca-koa-gauge-card">
                      <CardContent>
                        <Stack spacing={2.5} alignItems="center" textAlign="center">
                          <Box className="sca-koa-probability-gauge" sx={{ background: gaugeBackground }}>
                            <Box>
                              <Typography variant="h3" fontWeight={900}>{percent(score)}</Typography>
                              <Typography>{screening.modelLabel} probability</Typography>
                            </Box>
                          </Box>
                          <Box>
                            <Typography variant="h5" fontWeight={900}>{gaitDecisionLabel(screening)}</Typography>
                            <Typography color="text.secondary">{clinicalInterpretation(screening)}</Typography>
                          </Box>
                          <LinearProgress className="sca-koa-confidence-bar" variant="determinate" value={Math.round(score * 100)} />
                        </Stack>
                      </CardContent>
                    </Card>
                  </Grid2>

                  <Grid2 size={{ xs: 12, lg: 8 }}>
                    <Grid2 container spacing={2}>
                      <Grid2 size={{ xs: 12, md: 4 }}>
                        <Metric label="Latest result" value={gaitDecisionLabel(screening)} helper="Saved model decision" />
                      </Grid2>
                      <Grid2 size={{ xs: 12, md: 4 }}>
                        <Metric label="Reliability" value={screening.reliabilityLevel || "-"} helper="Model quality level" />
                      </Grid2>
                      <Grid2 size={{ xs: 12, md: 4 }}>
                        <Metric label="Direction" value={screening.direction || "-"} helper="Estimated gait direction" />
                      </Grid2>
                      <Grid2 size={{ xs: 12, md: 4 }}>
                        <Metric label="Max probability" value={percent(screening.maxProbability)} helper="Highest window score" />
                      </Grid2>
                      <Grid2 size={{ xs: 12, md: 4 }}>
                        <Metric label="Positive windows" value={screening.positiveWindowCount ?? "-"} helper={`Ratio ${percent(screening.positiveWindowRatio)}`} />
                      </Grid2>
                      <Grid2 size={{ xs: 12, md: 4 }}>
                        <Metric label={`Total ${activeModel} screenings`} value={sameModelScreenings.length} helper={`Saved ${activeModel} model runs`} />
                      </Grid2>
                    </Grid2>
                  </Grid2>
                </Grid2>

                <Card className="sca-koa-workspace-card">
                  <CardContent>
                    <Stack spacing={2}>
                      <Typography variant="h5" fontWeight={900}>Reliability Notes</Typography>
                      {(screening.reliabilityReasons || []).map((note) => (
                        <Alert key={note} severity={screening.reliabilityLevel === "High" ? "success" : "warning"}>{note}</Alert>
                      ))}
                    </Stack>
                  </CardContent>
                </Card>

                <Card className="sca-koa-workspace-card">
                  <CardContent>
                    <Stack spacing={2}>
                      <Stack direction={{ xs: "column", md: "row" }} justifyContent="space-between" spacing={1.5}>
                        <Box>
                          <Typography variant="h5" fontWeight={900}>Instability Visualization</Typography>
                          <Typography color="text.secondary">Body-region stability map generated from the same training-safe CSV used by this model run.</Typography>
                        </Box>
                        <Button component={Link} to={getSectionLink("instability")} startIcon={<AccessibilityNewIcon />}>Open map</Button>
                      </Stack>
                      <InstabilityBodyMap map={screening.instabilityMap} />
                    </Stack>
                  </CardContent>
                </Card>

                {screening.modelKey === "sca" && (screening.detected || screening.tendency) && (
                  <Card className="sca-koa-workspace-card">
                    <CardContent>
                      <Stack spacing={2}>
                        <Stack direction={{ xs: "column", md: "row" }} justifyContent="space-between" spacing={1.5}>
                          <Box>
                            <Typography variant="h5" fontWeight={900}>SCA Genetic Awareness</Typography>
                            <Typography color="text.secondary">Editable family-history chart for diagnosed SCA relatives and suspected abnormal gait behavior.</Typography>
                          </Box>
                          <Button component={Link} to={getSectionLink("genetics")} startIcon={<BiotechIcon />}>Open chart</Button>
                        </Stack>
                        <GeneticChartV2 awareness={buildAwarenessPreview(geneticsForm)} />
                      </Stack>
                    </CardContent>
                  </Card>
                )}
              </>
            )}

            {activeSection === "instability" && (
              <Card className="sca-koa-workspace-card">
                <CardContent>
                  <Stack spacing={2}>
                    <Typography variant="h5" fontWeight={900}>Joint-Level Instability Visualization</Typography>
                    <Typography color="text.secondary">
                      Green means stable control, yellow means a moderate control region, and red means high instability for the selected body region.
                    </Typography>
                    <InstabilityBodyMap map={screening.instabilityMap} />
                  </Stack>
                </CardContent>
              </Card>
            )}

            {activeSection === "history" && (
              <Card className="sca-koa-workspace-card">
                <CardContent>
                  <Stack spacing={2}>
                    <Typography variant="h5" fontWeight={900}>{screening.modelLabel} Saved History</Typography>
                    {sameModelScreenings.map((item) => (
                      <Box key={item.id} className={item.id === screening.id ? "sca-koa-timeline-item active" : "sca-koa-timeline-item"}>
                        <Stack spacing={0.5}>
                          <Stack direction="row" spacing={1} alignItems="center" flexWrap="wrap">
                            <Chip size="small" color={resultTone(item)} label={item.detected ? "Detected" : item.tendency ? "Borderline" : "Not detected"} />
                            <Typography fontWeight={900}>{gaitDecisionLabel(item)}</Typography>
                          </Stack>
                          <Typography color="text.secondary">{formatDate(item.createdAt)} | Direction {item.direction || "-"} | Mean probability {percent(item.probability)}</Typography>
                        </Stack>
                        <Stack direction="row" spacing={1}>
                          <Button component={Link} to={`/patient/results/sca-koa?id=${item.id}&section=history`} size="small">Open</Button>
                          <IconButton size="small" disabled={deletingId === item.id} onClick={() => deleteScreening(item.id)}>
                            <DeleteOutlineIcon fontSize="small" />
                          </IconButton>
                        </Stack>
                      </Box>
                    ))}
                  </Stack>
                </CardContent>
              </Card>
            )}

            {activeSection === "genetics" && (
              <GeneticAwarenessPanel
                screening={screening}
                genetics={genetics}
                form={geneticsForm}
                setForm={setGeneticsForm}
                onSave={saveGenetics}
                saving={geneticsSaving}
                error={geneticsError}
              />
            )}
          </Stack>
        </Grid2>
      </Grid2>
    </Container>
  );
}
