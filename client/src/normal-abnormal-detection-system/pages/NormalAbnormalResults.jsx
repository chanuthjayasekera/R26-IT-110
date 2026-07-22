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
  Typography
} from "@mui/material";
import { Link, useLocation, useNavigate, useSearchParams } from "react-router-dom";
import AccessibilityNewIcon from "@mui/icons-material/AccessibilityNew";
import AccountTreeIcon from "@mui/icons-material/AccountTree";
import AssessmentIcon from "@mui/icons-material/Assessment";
import BarChartIcon from "@mui/icons-material/BarChart";
import BiotechIcon from "@mui/icons-material/Biotech";
import CheckCircleIcon from "@mui/icons-material/CheckCircle";
import ChevronLeftIcon from "@mui/icons-material/ChevronLeft";
import ChevronRightIcon from "@mui/icons-material/ChevronRight";
import DeleteOutlineIcon from "@mui/icons-material/DeleteOutline";
import DownloadDoneIcon from "@mui/icons-material/DownloadDone";
import ErrorOutlineIcon from "@mui/icons-material/ErrorOutline";
import HealingIcon from "@mui/icons-material/Healing";
import InsightsIcon from "@mui/icons-material/Insights";
import MonitorHeartIcon from "@mui/icons-material/MonitorHeart";
import PersonSearchIcon from "@mui/icons-material/PersonSearch";
import PictureAsPdfIcon from "@mui/icons-material/PictureAsPdf";
import ReplayIcon from "@mui/icons-material/Replay";
import TableChartIcon from "@mui/icons-material/TableChart";
import UploadFileIcon from "@mui/icons-material/UploadFile";
import VideoCameraBackIcon from "@mui/icons-material/VideoCameraBack";
import WarningAmberIcon from "@mui/icons-material/WarningAmber";
import CentralProfileFlagButton from "../../common/components/CentralProfileFlagButton.jsx";
import { api, getApiError } from "../../common/api/http.js";
import { useAuth } from "../../common/state/AuthContext.jsx";
import "../styles/normal-abnormal-detection.css";

const latestVideoKey = "normalAbnormalLatestVideoScreening";
const dismissedPreviewKey = "normalAbnormalManuallyDismissedPreviewIdV3";

function scopedSessionKey(key, userId) {
  return userId ? `${key}:${userId}` : key;
}

function rememberLatestVideoScreening(userId, screening) {
  if (!screening?.hasVideoPreview || !screening?.videoUrl) return;
  window.sessionStorage.setItem(scopedSessionKey(latestVideoKey, userId), JSON.stringify(screening));
}

function isDismissedPreview(userId, screening) {
  try {
    return Boolean(screening?.id) && window.sessionStorage.getItem(scopedSessionKey(dismissedPreviewKey, userId)) === screening.id;
  } catch {
    return false;
  }
}

function formatDate(value) {
  if (!value) return "";
  return new Intl.DateTimeFormat("en", {
    dateStyle: "medium",
    timeStyle: "short"
  }).format(new Date(value));
}

function percent(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
  return `${Number(value).toFixed(2)}%`;
}

function probability(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return 0;
  return Math.min(Math.max(Number(value), 0), 1);
}

function formatMetricNumber(value, unit = "") {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
  const numeric = Number(value);
  const formatted = Math.abs(numeric) >= 100 ? numeric.toFixed(1) : numeric.toFixed(2);
  return `${formatted.replace(/\.00$/, "")}${unit ? ` ${unit}` : ""}`;
}

function metricStatusColor(status) {
  if (status === "green") return "success";
  if (status === "yellow") return "warning";
  if (status === "red") return "error";
  return "default";
}

function metricStatusLabel(status) {
  if (status === "green") return "Green";
  if (status === "yellow") return "Yellow";
  if (status === "red") return "Red";
  return "Unavailable";
}

function resultTone(screening) {
  const text = String(screening?.finalResult || "").toLowerCase();
  if (text.includes("abnormal")) return "error";
  if (text.includes("normal")) return "success";
  return "warning";
}

function modelDirectionText(screening) {
  const suggested = String(screening?.modelSuggestedResult || screening?.result?.model_suggested_result || "").toLowerCase();
  const finalText = String(screening?.finalResult || "").toLowerCase();
  if (suggested.includes("abnormal") || finalText.includes("abnormal")) return "abnormal";
  if (suggested.includes("normal") || finalText.includes("normal")) return "normal";
  return "";
}

function isInconclusive(screening) {
  const text = String(screening?.finalResult || "").toLowerCase();
  return text.includes("inconclusive") || screening?.finalLabel === null || screening?.finalLabel === undefined;
}

function resultVisual(screening) {
  const direction = modelDirectionText(screening);
  const inconclusive = isInconclusive(screening);

  if (direction === "abnormal") {
    return {
      className: "normal-result-billboard abnormal",
      icon: <ErrorOutlineIcon />,
      label: "ABNORMAL GAIT",
      kicker: inconclusive ? "Model direction - inconclusive screening" : "Current screening result",
      caption: inconclusive
        ? "The model pattern leans abnormal, but recording quality or duration was not strong enough for a final reliable decision."
        : "The current screening should be reviewed with the specialist model profile."
    };
  }
  if (direction === "normal") {
    return {
      className: "normal-result-billboard normal",
      icon: <CheckCircleIcon />,
      label: "NORMAL GAIT",
      kicker: inconclusive ? "Model direction - inconclusive screening" : "Current screening result",
      caption: inconclusive
        ? "The model pattern leans normal, but recording quality or duration was not strong enough for a final reliable decision."
        : "The current screening does not strongly match abnormal gait patterns."
    };
  }
  return {
    className: "normal-result-billboard inconclusive",
    icon: <WarningAmberIcon />,
    label: "INCONCLUSIVE",
    kicker: "Current screening result",
    caption: "The current screening needs stronger recording quality or duration."
  };
}

function screeningSearchText(screening) {
  return [
    screening?.finalResult,
    screening?.modelSuggestedResult,
  ].filter(Boolean).join(" ").toLowerCase();
}

function outcomeCondition(screening) {
  const text = screeningSearchText(screening);
  if (text.includes("abnormal")) return "abnormal";
  if (text.includes("normal")) return "normal";
  return "inconclusive";
}

function outcomeDate(screening) {
  if (!screening?.createdAt) return "";
  const date = new Date(screening.createdAt);
  if (Number.isNaN(date.getTime())) return "";
  return date.toISOString().slice(0, 10);
}

function filterOutcomes(screenings, condition, date) {
  return screenings.filter((screening) => {
    const conditionOk = condition === "all" || outcomeCondition(screening) === condition;
    const dateOk = !date || outcomeDate(screening) === date;
    return conditionOk && dateOk;
  });
}

function pdfEscape(value) {
  return String(value ?? "")
    .replace(/[^\x20-\x7E]/g, " ")
    .replace(/\\/g, "\\\\")
    .replace(/\(/g, "\\(")
    .replace(/\)/g, "\\)");
}

function wrapPdfText(value, maxChars, maxLines = 2) {
  const words = String(value ?? "-").split(/\s+/).filter(Boolean);
  const lines = [];
  let line = "";

  words.forEach((word) => {
    const next = line ? `${line} ${word}` : word;
    if (next.length <= maxChars) {
      line = next;
    } else {
      if (line) lines.push(line);
      line = word.slice(0, maxChars);
    }
  });
  if (line) lines.push(line);
  return lines.slice(0, maxLines);
}

function createPdfBlob(pages) {
  const objects = [];
  objects[1] = "<< /Type /Catalog /Pages 2 0 R >>";
  objects[3] = "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>";
  objects[4] = "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>";

  const pageIds = [];
  let nextId = 5;
  pages.forEach((content) => {
    const contentId = nextId;
    const pageId = nextId + 1;
    nextId += 2;
    objects[contentId] = `<< /Length ${content.length} >>\nstream\n${content}\nendstream`;
    objects[pageId] = `<< /Type /Page /Parent 2 0 R /MediaBox [0 0 842 595] /Resources << /Font << /F1 3 0 R /F2 4 0 R >> >> /Contents ${contentId} 0 R >>`;
    pageIds.push(pageId);
  });

  objects[2] = `<< /Type /Pages /Kids [${pageIds.map((id) => `${id} 0 R`).join(" ")}] /Count ${pageIds.length} >>`;

  let pdf = "%PDF-1.4\n";
  const offsets = [0];
  for (let index = 1; index < objects.length; index += 1) {
    offsets[index] = pdf.length;
    pdf += `${index} 0 obj\n${objects[index]}\nendobj\n`;
  }

  const xrefAt = pdf.length;
  pdf += `xref\n0 ${objects.length}\n0000000000 65535 f \n`;
  for (let index = 1; index < objects.length; index += 1) {
    pdf += `${String(offsets[index]).padStart(10, "0")} 00000 n \n`;
  }
  pdf += `trailer\n<< /Size ${objects.length} /Root 1 0 R >>\nstartxref\n${xrefAt}\n%%EOF`;
  return new Blob([pdf], { type: "application/pdf" });
}

function buildBiometricsPdf(screening, biometrics, metrics) {
  const pageWidth = 842;
  const pageHeight = 595;
  const margin = 34;
  const columns = [
    { label: "Metric", x: 44, width: 120, chars: 22 },
    { label: "Value", x: 170, width: 108, chars: 19 },
    { label: "Status", x: 286, width: 66, chars: 10 },
    { label: "Clinical meaning", x: 360, width: 245, chars: 46 },
    { label: "Reference range", x: 614, width: 178, chars: 32 },
  ];
  const pages = [];
  let content = "";
  let y = margin;

  const addText = (text, x, topY, size = 9, font = "F1", color = [15, 23, 42]) => {
    content += `${(color[0] / 255).toFixed(3)} ${(color[1] / 255).toFixed(3)} ${(color[2] / 255).toFixed(3)} rg BT /${font} ${size} Tf ${x} ${pageHeight - topY} Td (${pdfEscape(text)}) Tj ET\n`;
  };
  const addRect = (x, topY, width, height, color) => {
    content += `${(color[0] / 255).toFixed(3)} ${(color[1] / 255).toFixed(3)} ${(color[2] / 255).toFixed(3)} rg ${x} ${pageHeight - topY - height} ${width} ${height} re f\n`;
  };
  const statusColor = (status) => {
    if (status === "green") return [22, 163, 74];
    if (status === "yellow") return [217, 119, 6];
    if (status === "red") return [220, 38, 38];
    return [100, 116, 139];
  };
  const addHeader = () => {
    addRect(0, 0, pageWidth, pageHeight, [255, 255, 255]);
    addRect(0, 0, pageWidth, 12, [14, 165, 233]);
    addRect(0, 12, pageWidth, 4, [34, 197, 94]);
    addText("Component 1 Biometrics Report", margin, 42, 18, "F2");
    addText(`Result: ${screening?.finalResult || "-"} | Date: ${formatDate(screening?.createdAt) || "-"}`, margin, 64, 10, "F1", [71, 85, 105]);
    addText(`Clean duration: ${formatMetricNumber(biometrics?.durationSec, "s")} | Foot events: ${biometrics?.eventCount ?? "-"}`, margin, 80, 10, "F1", [71, 85, 105]);
    addRect(margin, 102, pageWidth - margin * 2, 24, [241, 245, 249]);
    columns.forEach((column) => addText(column.label, column.x, 118, 9, "F2", [51, 65, 85]));
    y = 142;
  };
  const finishPage = () => {
    pages.push(content);
    content = "";
    y = margin;
  };

  addHeader();
  metrics.forEach((metric) => {
    if (y > 536) {
      finishPage();
      addHeader();
    }
    const rowTop = y - 14;
    addRect(margin, rowTop, pageWidth - margin * 2, 42, metric.status === "red" ? [254, 242, 242] : metric.status === "yellow" ? [255, 251, 235] : [240, 253, 244]);
    addRect(margin, rowTop, 5, 42, statusColor(metric.status));

    const value = `${formatMetricNumber(metric.value, metric.unit)}${metric.left !== null && metric.left !== undefined ? ` | L ${formatMetricNumber(metric.left, metric.unit)} / R ${formatMetricNumber(metric.right, metric.unit)}` : ""}`;
    const cells = [
      metric.label,
      value,
      metricStatusLabel(metric.status),
      metric.meaning,
      metric.referenceRange || "Varies",
    ];

    cells.forEach((cell, index) => {
      wrapPdfText(cell, columns[index].chars, index >= 3 ? 2 : 1).forEach((line, lineIndex) => {
        addText(line, columns[index].x, y + lineIndex * 11, 8.5, index === 0 || index === 2 ? "F2" : "F1", index === 2 ? statusColor(metric.status) : [15, 23, 42]);
      });
    });
    y += 46;
  });
  finishPage();
  return createPdfBlob(pages);
}

function downloadBiometricsPdf(screening, biometrics, metrics) {
  const blob = buildBiometricsPdf(screening, biometrics, metrics);
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `component-1-biometrics-${screening?.id || "report"}.pdf`;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

function Metric({ label, value, helper }) {
  return (
    <Box className="normal-metric-tile">
      <Typography color="text.secondary">{label}</Typography>
      <Typography variant="h5" fontWeight={900}>{value ?? "-"}</Typography>
      {helper && <Typography className="normal-metric-helper">{helper}</Typography>}
    </Box>
  );
}

function SectionHeader({ icon, title, subtitle, status = "Profile section" }) {
  return (
    <Stack direction={{ xs: "column", sm: "row" }} spacing={1.5} alignItems={{ xs: "flex-start", sm: "center" }} justifyContent="space-between">
      <Stack direction="row" spacing={1.5} alignItems="center">
        <Box className="normal-section-icon">{icon}</Box>
        <Box>
          <Typography variant="h5" fontWeight={900}>{title}</Typography>
          <Typography color="text.secondary">{subtitle}</Typography>
        </Box>
      </Stack>
      <Chip size="small" label={status} />
    </Stack>
  );
}

export function ResultBillboard({ screening }) {
  const visual = resultVisual(screening);

  return (
    <Box className={visual.className}>
      <Box className="normal-result-symbol">{visual.icon}</Box>
      <Box>
        <Typography className="normal-result-kicker">{visual.kicker}</Typography>
        <Typography className="normal-result-label">{visual.label}</Typography>
        <Typography>{visual.caption}</Typography>
        {isInconclusive(screening) && (
          <Chip className="normal-result-status-chip" color="warning" label="Inconclusive - repeat recommended" />
        )}
      </Box>
    </Box>
  );
}

function ProfileSideNav({ collapsed, setCollapsed, activeSection, getSectionLink, screening, userId }) {
  const itemClass = collapsed ? "normal-nav-label hidden" : "normal-nav-label";
  const detectionState = screening?.hasVideoPreview && screening?.videoUrl && !isDismissedPreview(userId, screening)
    ? { latestVideoScreening: screening, showLatestVideo: true }
    : {};
  const navItems = [
    ["overview", "Overview", <PersonSearchIcon />],
    ["biometrics", "Biometrics", <TableChartIcon />],
    ["timeline", "Timeline", <InsightsIcon />]
  ];

  return (
    <Box className={collapsed ? "normal-side-nav collapsed" : "normal-side-nav"}>
      <Button
        className="normal-nav-toggle"
        variant="outlined"
        onClick={() => setCollapsed((old) => !old)}
        startIcon={collapsed ? <ChevronRightIcon /> : <ChevronLeftIcon />}
      >
        <span className={itemClass}>{collapsed ? "" : "Collapse"}</span>
      </Button>
      <Button
        component={Link}
        to="/patient/detection/normal-abnormal"
        state={detectionState}
        variant="outlined"
        startIcon={<VideoCameraBackIcon />}
      >
        <span className={itemClass}>Detection</span>
      </Button>
      {navItems.map(([section, label, icon]) => (
        <Button
          key={section}
          component={Link}
          to={getSectionLink(section)}
          variant={activeSection === section ? "contained" : "text"}
          startIcon={icon}
        >
          <span className={itemClass}>{label}</span>
        </Button>
      ))}
      <Button component={Link} to="/patient/central-profile" variant="text" startIcon={<AccountTreeIcon />}>
        <span className={itemClass}>Centralized Profile</span>
      </Button>
      <Button component={Link} to="/patient" variant="text" startIcon={<MonitorHeartIcon />}>
        <span className={itemClass}>Dashboard</span>
      </Button>
    </Box>
  );
}

export function BiometricsTable({ screening }) {
  const biometrics = screening?.result?.biometrics;
  const metrics = biometrics?.metrics || [];
  const resultText = String(screening?.finalResult || "").toLowerCase();
  const isScreeningNormal = resultText.includes("normal") && !resultText.includes("abnormal");
  const isScreeningAbnormal = resultText.includes("abnormal");

  if (!biometrics?.available || metrics.length === 0) {
    return (
      <Stack spacing={2}>
        <Alert severity="warning">
          Biometrics are not available for this saved screening yet. Run a new Component 1 screening to save the full gait biometrics profile.
        </Alert>
        <Grid2 container spacing={2}>
          <Grid2 size={{ xs: 12, md: 4 }}>
            <Metric label="Clean duration" value={`${screening?.result?.clean_duration_sec ?? screening?.result?.cleanDurationSec ?? "-"} s`} helper="Available from extraction" />
          </Grid2>
          <Grid2 size={{ xs: 12, md: 4 }}>
            <Metric label="Total windows" value={screening?.result?.total_windows ?? "-"} helper="Temporal model windows" />
          </Grid2>
          <Grid2 size={{ xs: 12, md: 4 }}>
            <Metric label="Direction" value={screening?.direction || screening?.result?.direction || "-"} helper="Estimated gait direction" />
          </Grid2>
        </Grid2>
      </Stack>
    );
  }

  return (
    <Stack spacing={2}>
      <Grid2 container spacing={2}>
        <Grid2 size={{ xs: 12, md: 3 }}>
          <Metric label="Reference" value={biometrics.referenceFrame || "Pose biometrics"} helper="Component 1 latest result" />
        </Grid2>
        <Grid2 size={{ xs: 12, md: 3 }}>
          <Metric label="Clean duration" value={formatMetricNumber(biometrics.durationSec ?? screening?.result?.clean_duration_sec, "s")} helper="Used for this table" />
        </Grid2>
        <Grid2 size={{ xs: 12, md: 3 }}>
          <Metric label="Foot events" value={biometrics.eventCount ?? "-"} helper={`Left ${biometrics.leftEvents ?? "-"} / Right ${biometrics.rightEvents ?? "-"}`} />
        </Grid2>
        <Grid2 size={{ xs: 12, md: 3 }}>
          <Metric label="Body scale" value={formatMetricNumber(biometrics.bodyScale, "")} helper="Normalization factor" />
        </Grid2>
      </Grid2>

      {biometrics.screeningContext && (
        <Grid2 container spacing={2}>
          <Grid2 size={{ xs: 12, md: 4 }}>
            <Metric label="Preserved metrics" value={biometrics.screeningContext.greenCount ?? 0} helper="Still within model band" />
          </Grid2>
          <Grid2 size={{ xs: 12, md: 4 }}>
            <Metric label="Borderline markers" value={biometrics.screeningContext.yellowCount ?? 0} helper="Watch in current context" />
          </Grid2>
          <Grid2 size={{ xs: 12, md: 4 }}>
            <Metric
              label={isScreeningNormal ? "Monitor markers" : "Abnormal markers"}
              value={biometrics.screeningContext.redCount ?? 0}
              helper={isScreeningAbnormal ? "Supports clinical concern" : "Outside band, not model-dominant"}
            />
          </Grid2>
        </Grid2>
      )}

      <Box className="normal-biometrics-actions">
        <Box>
          <Typography variant="h6" fontWeight={900}>Clinical biometrics report</Typography>
          <Typography color="text.secondary">Current-run gait measurements aligned with the real Component 1 screening result.</Typography>
        </Box>
        <Button
          variant="contained"
          startIcon={<PictureAsPdfIcon />}
          onClick={() => downloadBiometricsPdf(screening, biometrics, metrics)}
        >
          Download PDF
        </Button>
      </Box>

      <Box className="normal-status-legend">
        <Box className="normal-status-legend-item green">
          <Chip size="small" color="success" label="Green" />
          <Typography>Within the shown guide range for this measured biomarker.</Typography>
        </Box>
        <Box className="normal-status-legend-item yellow">
          <Chip size="small" color="warning" label="Yellow" />
          <Typography>Borderline or near the edge of the guide range; monitor with the full screening result.</Typography>
        </Box>
        <Box className="normal-status-legend-item red">
          <Chip size="small" color="error" label="Red" />
          <Typography>Outside the guide range; it can support concern but does not override the full Component 1 model.</Typography>
        </Box>
      </Box>

      <Box className="normal-data-table biometrics">
        <Box className="normal-data-row biometrics head">
          <Typography>Metric</Typography>
          <Typography>Value</Typography>
          <Typography>Status</Typography>
          <Typography>Clinical meaning</Typography>
          <Typography>Reference range</Typography>
        </Box>
        {metrics.map((metric) => (
          <Box key={metric.key || metric.label} className={`normal-data-row biometrics status-${metric.status || "unavailable"}`}>
            <Box>
              <Typography fontWeight={900}>{metric.label}</Typography>
              {(metric.left !== null && metric.left !== undefined) || (metric.right !== null && metric.right !== undefined) ? (
                <Typography className="normal-metric-helper">
                  Left {formatMetricNumber(metric.left, metric.unit)} / Right {formatMetricNumber(metric.right, metric.unit)}
                </Typography>
              ) : null}
            </Box>
            <Typography fontWeight={900}>{formatMetricNumber(metric.value, metric.unit)}</Typography>
            <Chip size="small" color={metricStatusColor(metric.status)} label={metricStatusLabel(metric.status)} />
            <Box>
              <Typography color="text.secondary">{metric.meaning}</Typography>
            </Box>
            <Typography className="normal-reference-range">{metric.referenceRange || "Varies"}</Typography>
          </Box>
        ))}
      </Box>

      {(biometrics.notes || []).map((note) => (
        <Alert key={note} severity="info">{note}</Alert>
      ))}
    </Stack>
  );
}

function InstabilityMap() {
  const joints = [
    ["Head", "stable"],
    ["Left shoulder", "pending"],
    ["Right shoulder", "pending"],
    ["Trunk", "pending"],
    ["Left knee", "pending"],
    ["Right knee", "pending"],
    ["Left ankle", "pending"],
    ["Right ankle", "pending"]
  ];

  return (
    <Grid2 container spacing={2} alignItems="stretch">
      <Grid2 size={{ xs: 12, md: 5 }}>
        <Box className="normal-body-map">
          <span className="joint head stable" />
          <span className="joint shoulder-left pending" />
          <span className="joint shoulder-right pending" />
          <span className="joint trunk pending" />
          <span className="joint knee-left pending" />
          <span className="joint knee-right pending" />
          <span className="joint ankle-left pending" />
          <span className="joint ankle-right pending" />
          <span className="body-line spine" />
          <span className="body-line arms" />
          <span className="body-line leg-left" />
          <span className="body-line leg-right" />
        </Box>
      </Grid2>
      <Grid2 size={{ xs: 12, md: 7 }}>
        <Stack spacing={1.2}>
          {joints.map(([joint, status]) => (
            <Box key={joint} className="normal-instability-row">
              <Typography fontWeight={900}>{joint}</Typography>
              <Chip
                size="small"
                label={status === "stable" ? "Stable" : "Awaiting SCA/OA model"}
                color={status === "stable" ? "success" : "default"}
              />
            </Box>
          ))}
        </Stack>
      </Grid2>
    </Grid2>
  );
}

function GeneticChart() {
  return (
    <Grid2 container spacing={2}>
      {[
        ["Family SCA history", 0, "Not entered"],
        ["Relatives with abnormal gait", 0, "Not entered"],
        ["Hereditary awareness score", 0, "Awaiting SCA form"]
      ].map(([label, value, helper]) => (
        <Grid2 key={label} size={{ xs: 12, md: 4 }}>
          <Box className="normal-genetic-tile">
            <Typography fontWeight={900}>{label}</Typography>
            <Box className="normal-mini-donut" style={{ "--value": `${value}%` }}>
              <span>{value}%</span>
            </Box>
            <Typography color="text.secondary">{helper}</Typography>
          </Box>
        </Grid2>
      ))}
    </Grid2>
  );
}

function DisorderBars({ screening }) {
  const abnormalScore = Math.round(probability(screening?.meanProbAbnormal) * 100);
  const bars = [
    ["Normal/abnormal screening", abnormalScore, "Component 1 active"],
    ["Parkinson disorder", 0, "Component 3 pending"],
    ["Neuropathy disorder", 0, "Future model pending"],
    ["Spinocerebellar ataxia", 0, "Component 2 pending"],
    ["Knee osteoarthritis", 0, "Component 2 pending"]
  ];

  return (
    <Stack spacing={1.4}>
      {bars.map(([label, value, helper]) => (
        <Box key={label} className="normal-bar-row">
          <Stack direction="row" justifyContent="space-between" spacing={2}>
            <Typography fontWeight={900}>{label}</Typography>
            <Typography color="text.secondary">{value}%</Typography>
          </Stack>
          <Box className="normal-bar-track">
            <Box className="normal-bar-fill" style={{ width: `${value}%` }} />
          </Box>
          <Typography className="normal-metric-helper">{helper}</Typography>
        </Box>
      ))}
    </Stack>
  );
}

function RehabRecommendations({ screening }) {
  const isAbnormal = screening?.finalLabel === 1;
  const recommendations = isAbnormal
    ? ["Clinician review recommended before exercise assignment.", "Rehabilitation plan will appear after exercise quality analysis.", "Safety instructions will be fetched from professional-maintained rules."]
    : ["Maintain regular walking activity as clinically appropriate.", "Repeat gait screening over time for longitudinal monitoring.", "No disorder-specific rehab plan has been assigned yet."];

  return (
    <Grid2 container spacing={2}>
      {recommendations.map((item, index) => (
        <Grid2 key={item} size={{ xs: 12, md: 4 }}>
          <Box className="normal-rehab-card">
            <Chip size="small" label={`Recommendation ${index + 1}`} color={index === 0 ? "primary" : "default"} />
            <Typography fontWeight={900}>{item}</Typography>
          </Box>
        </Grid2>
      ))}
    </Grid2>
  );
}

function OutcomeFilters({ condition, setCondition, date, setDate }) {
  return (
    <Stack className="normal-history-filters" direction={{ xs: "column", sm: "row" }} spacing={1.2}>
      <FormControl size="small" className="normal-history-filter">
        <InputLabel>Condition</InputLabel>
        <Select
          label="Condition"
          value={condition}
          onChange={(event) => setCondition(event.target.value)}
        >
          <MenuItem value="all">All</MenuItem>
          <MenuItem value="normal">Normal</MenuItem>
          <MenuItem value="abnormal">Abnormal</MenuItem>
          <MenuItem value="inconclusive">Inconclusive</MenuItem>
        </Select>
      </FormControl>
      <TextField
        className="normal-history-filter"
        size="small"
        label="Date"
        type="date"
        value={date}
        onChange={(event) => setDate(event.target.value)}
        InputLabelProps={{ shrink: true }}
      />
    </Stack>
  );
}

export default function NormalAbnormalResults() {
  const location = useLocation();
  const navigate = useNavigate();
  const { user } = useAuth();
  const userId = user?.id || "";
  const [searchParams] = useSearchParams();
  const requestedId = searchParams.get("id");
  const [screening, setScreening] = useState(location.state?.screening || null);
  const [profile, setProfile] = useState(location.state?.profile || null);
  const [screenings, setScreenings] = useState([]);
  const [loading, setLoading] = useState(!location.state?.screening);
  const [error, setError] = useState("");
  const [deletingId, setDeletingId] = useState("");
  const [navCollapsed, setNavCollapsed] = useState(false);
  const [outcomeConditionFilter, setOutcomeConditionFilter] = useState("all");
  const [outcomeDateFilter, setOutcomeDateFilter] = useState("");
  const activeSection = searchParams.get("section") || "overview";

  useEffect(() => {
    let alive = true;

    async function load() {
      setLoading(true);
      setError("");
      try {
        const profileRes = await api.get("/normal-abnormal/clinical-profile");
        if (!alive) return;
        setProfile(profileRes.data.profile);
        setScreenings(profileRes.data.screenings || []);

        if (requestedId) {
          const screeningRes = await api.get(`/normal-abnormal/screenings/${requestedId}`);
          if (!alive) return;
          setScreening(screeningRes.data.screening);
        } else if (!location.state?.screening) {
          setScreening(profileRes.data.screenings?.[0] || null);
        }
      } catch (err) {
        if (!alive) return;
        const apiError = getApiError(err);
        setError(apiError.message || "Unable to load clinical profile.");
      } finally {
        if (alive) setLoading(false);
      }
    }

    load();

    return () => {
      alive = false;
    };
  }, [requestedId, location.state?.screening]);

  const score = probability(screening?.meanProbAbnormal);
  const gaugeBackground = useMemo(() => {
    const degrees = Math.round(score * 360);
    return `conic-gradient(#ef4444 0deg ${degrees}deg, #16a34a ${degrees}deg 360deg)`;
  }, [score]);

  useEffect(() => {
    rememberLatestVideoScreening(userId, screening);
  }, [screening, userId]);

  function getSectionLink(section) {
    const params = new URLSearchParams();
    if (screening?.id) params.set("id", screening.id);
    if (section !== "overview") params.set("section", section);
    const query = params.toString();
    return `/patient/results/normal-abnormal${query ? `?${query}` : ""}`;
  }

  async function deleteScreening(id) {
    if (!window.confirm("Delete this saved screening result?")) return;
    setDeletingId(id);
    setError("");

    try {
      let res;
      try {
        res = await api.delete(`/normal-abnormal/screenings/${id}`);
      } catch (deleteError) {
        if (deleteError?.response?.status !== 404 && deleteError?.response?.status !== 405) throw deleteError;
        res = await api.post(`/normal-abnormal/screenings/${id}/delete`);
      }
      const remaining = res.data.screenings || [];
      setProfile(res.data.profile);
      setScreenings(remaining);

      if (id === screening?.id) {
        const nextScreening = remaining[0] || null;
        setScreening(nextScreening);
        if (nextScreening) {
          const params = new URLSearchParams();
          params.set("id", nextScreening.id);
          if (activeSection !== "overview") params.set("section", activeSection);
          navigate(`/patient/results/normal-abnormal?${params.toString()}`, { replace: true });
        } else {
          navigate("/patient/results/normal-abnormal", { replace: true });
        }
      }
    } catch (err) {
      const apiError = getApiError(err);
      setError(apiError.message || "Unable to delete screening result.");
    } finally {
      setDeletingId("");
    }
  }

  if (loading) {
    return (
      <Container maxWidth="lg" className="normal-detection-page">
        <Card className="normal-workspace-card">
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
      <Container maxWidth="lg" className="normal-detection-page">
        <Alert severity="error">{error}</Alert>
      </Container>
    );
  }

  if (!screening) {
    return (
      <Container maxWidth="lg" className="normal-detection-page">
        <Card className="normal-empty-results">
          <CardContent>
            <Stack spacing={2} alignItems="center" textAlign="center" py={5}>
              <UploadFileIcon color="primary" />
              <Typography variant="h4" fontWeight={900}>No central clinical profile yet</Typography>
              <Typography color="text.secondary">Run the first gait screening and this page will start building the patient's longitudinal profile.</Typography>
              <Button component={Link} to="/patient/detection/normal-abnormal" state={{ startFresh: true }} variant="contained" startIcon={<ReplayIcon />}>
                Start detection
              </Button>
            </Stack>
          </CardContent>
        </Card>
      </Container>
    );
  }

  return (
    <Container maxWidth="xl" className="normal-detection-page">
      <Grid2 container spacing={3}>
        <Grid2 size={{ xs: 12, md: navCollapsed ? 1.1 : 3, lg: navCollapsed ? 0.9 : 2.4 }}>
          <ProfileSideNav
            collapsed={navCollapsed}
            setCollapsed={setNavCollapsed}
            activeSection={activeSection}
            getSectionLink={getSectionLink}
            screening={screening}
            userId={userId}
          />
        </Grid2>

        <Grid2 size={{ xs: 12, md: navCollapsed ? 10.9 : 9, lg: navCollapsed ? 11.1 : 9.6 }}>
          <Stack spacing={3}>
            <Box className="normal-results-hero compact">
              <Stack spacing={1.5}>
                <Chip icon={<AssessmentIcon />} color={resultTone(screening)} label={screening.finalResult || "Inconclusive"} />
                <Typography variant="h3" fontWeight={900}>Component 1 Clinical Profile</Typography>
                <Typography color="text.secondary">
                  Component 1 screening outcomes, biometrics, and longitudinal gait timeline.
                </Typography>
              </Stack>

              <Stack direction={{ xs: "column", sm: "row" }} spacing={1.2}>
                <CentralProfileFlagButton sourceType="normal_abnormal" screeningId={screening.id} />
                <Button component={Link} to="/patient/detection/normal-abnormal" state={{ startFresh: true }} variant="contained" startIcon={<ReplayIcon />}>
                  New screening
                </Button>
                <Button
                  className="normal-danger-button"
                  color="error"
                  variant="outlined"
                  startIcon={<DeleteOutlineIcon />}
                  disabled={deletingId === screening.id}
                  onClick={() => deleteScreening(screening.id)}
                >
                  Delete result
                </Button>
              </Stack>
            </Box>

            {activeSection === "overview" && (
              <>
                <ResultBillboard screening={screening} />
                <Grid2 container spacing={3}>
                  <Grid2 size={{ xs: 12, lg: 4 }}>
                    <Card className="normal-workspace-card normal-gauge-card">
                      <CardContent>
                        <Stack spacing={2.5} alignItems="center" textAlign="center">
                          <Box className="normal-probability-gauge" sx={{ background: gaugeBackground }}>
                            <Box>
                              <Typography variant="h3" fontWeight={900}>{Math.round(score * 100)}%</Typography>
                              <Typography>abnormal probability</Typography>
                            </Box>
                          </Box>
                          <Box>
                            <Typography variant="h5" fontWeight={900}>{screening.screeningSeverity || "-"}</Typography>
                            <Typography color="text.secondary">{screening.clinicalNote}</Typography>
                          </Box>
                          <LinearProgress className="normal-confidence-bar" variant="determinate" value={Number(screening.confidencePercent || 0)} />
                          <Typography color="text.secondary">Model confidence: {percent(screening.confidencePercent)}</Typography>
                        </Stack>
                      </CardContent>
                    </Card>
                  </Grid2>

                  <Grid2 size={{ xs: 12, lg: 8 }}>
                    <Grid2 container spacing={2}>
                      <Grid2 size={{ xs: 12, md: 4 }}>
                        <Metric label="Latest result" value={profile?.latestResult || screening.finalResult} helper="Normal, abnormal, or inconclusive" />
                      </Grid2>
                      <Grid2 size={{ xs: 12, md: 4 }}>
                        <Metric label="Severity" value={profile?.latestSeverity || screening.screeningSeverity} helper="Screening-level concern" />
                      </Grid2>
                      <Grid2 size={{ xs: 12, md: 4 }}>
                        <Metric label="Direction" value={profile?.latestDirection || screening.direction || "-"} helper="Estimated gait direction" />
                      </Grid2>
                      <Grid2 size={{ xs: 12, md: 4 }}>
                        <Metric label="Total screenings" value={profile?.totalScreenings ?? screenings.length} helper="Saved model runs" />
                      </Grid2>
                      <Grid2 size={{ xs: 12, md: 4 }}>
                        <Metric label="Normal count" value={profile?.normalScreenings ?? "-"} />
                      </Grid2>
                      <Grid2 size={{ xs: 12, md: 4 }}>
                        <Metric label="Abnormal count" value={profile?.abnormalScreenings ?? "-"} />
                      </Grid2>
                    </Grid2>
                  </Grid2>
                </Grid2>
              </>
            )}

            {activeSection === "biometrics" && (
              <Card className="normal-workspace-card">
                <CardContent>
                  <Stack spacing={2}>
                    <SectionHeader icon={<TableChartIcon />} title="Biometrics Table" subtitle="Walking speed, step length, cadence, arm swing, symmetry, and stride variability." status="Component 1 profile" />
                    <BiometricsTable screening={screening} />
                  </Stack>
                </CardContent>
              </Card>
            )}

            {activeSection === "instability" && (
              <Card className="normal-workspace-card">
                <CardContent>
                  <Stack spacing={2}>
                    <SectionHeader icon={<AccessibilityNewIcon />} title="Joint-Level Instability Visualization" subtitle="Green stable, yellow mild instability, red severe instability when SCA/OA modules are connected." status="Component 2 pending" />
                    <InstabilityMap />
                  </Stack>
                </CardContent>
              </Card>
            )}

            {activeSection === "genetics" && (
              <Card className="normal-workspace-card">
                <CardContent>
                  <Stack spacing={2}>
                    <SectionHeader icon={<BiotechIcon />} title="SCA Genetic Awareness Chart" subtitle="Family-history and hereditary gait-awareness indicators for early SCA monitoring." status="SCA genetics pending" />
                    <GeneticChart />
                  </Stack>
                </CardContent>
              </Card>
            )}

            {activeSection === "disorders" && (
              <Card className="normal-workspace-card">
                <CardContent>
                  <Stack spacing={2}>
                    <SectionHeader icon={<BarChartIcon />} title="Disorder Probability Charts" subtitle="Bar chart profile for Parkinson, neuropathy, SCA, OA, and the active screening model." status="Multi-model profile" />
                    <DisorderBars screening={screening} />
                  </Stack>
                </CardContent>
              </Card>
            )}

            {activeSection === "rehab" && (
              <Card className="normal-workspace-card">
                <CardContent>
                  <Stack spacing={2}>
                    <SectionHeader icon={<HealingIcon />} title="Rehabilitation Details and Recommendations" subtitle="Clinician-maintained plans and exercise quality outcomes will update this section." status="Exercise analysis pending" />
                    <RehabRecommendations screening={screening} />
                  </Stack>
                </CardContent>
              </Card>
            )}

            {activeSection === "timeline" && (
              <Card className="normal-workspace-card">
                <CardContent>
                  <Stack spacing={2}>
                    <SectionHeader icon={<InsightsIcon />} title="Longitudinal Model Outcome Timeline" subtitle="Every saved gait screening for this logged-in patient." status="Saved history" />
                    <OutcomeFilters
                      condition={outcomeConditionFilter}
                      setCondition={setOutcomeConditionFilter}
                      date={outcomeDateFilter}
                      setDate={setOutcomeDateFilter}
                    />
                    <Stack spacing={1.5}>
                      {filterOutcomes(screenings, outcomeConditionFilter, outcomeDateFilter).map((item) => (
                        <Box key={item.id} className={item.id === screening.id ? "normal-timeline-item active" : "normal-timeline-item"}>
                          <Stack spacing={0.5}>
                            <Stack direction="row" spacing={1} alignItems="center" flexWrap="wrap">
                              <Chip size="small" color={resultTone(item)} label={item.finalResult || "Inconclusive"} />
                              <Typography fontWeight={900}>{item.screeningSeverity || "Screening outcome"}</Typography>
                            </Stack>
                            <Typography color="text.secondary">
                              {formatDate(item.createdAt)} | Direction {item.direction || "-"} | Confidence {percent(item.confidencePercent)}
                            </Typography>
                          </Stack>
                          <Stack direction="row" spacing={1}>
                            <Button
                              className="normal-soft-button"
                              component={Link}
                              to={`/patient/results/normal-abnormal?id=${item.id}&section=timeline`}
                              size="small"
                              startIcon={<DownloadDoneIcon />}
                            >
                              Open
                            </Button>
                            <IconButton
                              className="normal-delete-icon-button"
                              size="small"
                              aria-label="Delete result"
                              title="Delete result"
                              disabled={deletingId === item.id}
                              onClick={() => deleteScreening(item.id)}
                            >
                              <DeleteOutlineIcon fontSize="small" />
                            </IconButton>
                          </Stack>
                        </Box>
                      ))}
                    </Stack>
                  </Stack>
                </CardContent>
              </Card>
            )}

            <Card className="normal-workspace-card normal-notes-card">
              <CardContent>
                <Stack spacing={2}>
                  <SectionHeader icon={<WarningAmberIcon />} title="Risk and Reliability" subtitle="Current model confidence and quality notes." status={screening.reliabilityLevel || "Pending"} />
                  {(screening.reliabilityReasons || []).map((note) => (
                    <Alert key={note} severity={screening.reliabilityLevel === "High" ? "success" : "warning"}>
                      {note}
                    </Alert>
                  ))}
                </Stack>
              </CardContent>
            </Card>
          </Stack>
        </Grid2>
      </Grid2>
    </Container>
  );
}
