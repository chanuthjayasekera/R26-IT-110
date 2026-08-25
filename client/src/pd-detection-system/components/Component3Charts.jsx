import React from "react";
import { Alert, Box, Button, Card, CardContent, Chip, Stack, Typography } from "@mui/material";
import BarChartIcon from "@mui/icons-material/BarChart";
import PictureAsPdfIcon from "@mui/icons-material/PictureAsPdf";

const MODEL_THRESHOLDS = {
  pd: 0.275,
  neuropathy: 0.48
};

const PLAUSIBLE_BIOMETRIC_RANGES = {
  walking_speed: [0.03, 1.8],
  cadence: [45, 170],
  step_length: [0.08, 1.3],
  stride_variability: [0, 45],
  arm_swing_amplitude: [0, 0.55],
  arm_swing_asymmetry: [0, 100],
  knee_rom: [5, 85],
  foot_clearance: [0, 0.18],
  trunk_sway: [0, 0.25]
};

export function percent(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
  return `${Math.round(Number(value) * 100)}%`;
}

function metricValue(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
  const numeric = Number(value);
  return numeric.toFixed(Math.abs(numeric) >= 100 ? 1 : 3).replace(/\.?0+$/, "");
}

function referenceValue(metric) {
  return metric?.referenceValue ?? metric?.normalValue;
}

function uploadedValue(metric) {
  return metric?.uploadedValue ?? metric?.detectedValue;
}

function referencePercent(metric) {
  return metric?.referencePercent ?? metric?.normalPercent ?? 100;
}

function uploadedPercent(metric) {
  return metric?.uploadedPercent ?? metric?.detectedPercent ?? 0;
}

function hasDiseaseReference(profile) {
  return (profile?.metrics || []).some((metric) => (
    metric?.diseaseRange?.source === "real-world disease gait reference range"
    && metric.referenceValue !== undefined
    && metric.uploadedValue !== undefined
  ));
}

function diseaseRangeText(metric, label = "Disease reference") {
  const range = metric?.diseaseRange;
  if (range?.low !== undefined && range?.high !== undefined) {
    return `${label} range: ${metricValue(range.low)}-${metricValue(range.high)} ${metric.unit}`;
  }
  const oldRange = metric?.normalRange;
  const [low, high] = oldRange?.green || [];
  if (low !== undefined && high !== undefined) {
    return `${label} range: ${metricValue(low)}-${metricValue(high)} ${metric.unit}`;
  }
  return `${label} range unavailable`;
}

function barHeight(percentValue) {
  const numeric = Number(percentValue || 0);
  return `${Math.max(4, Math.min(100, (numeric / 140) * 100))}%`;
}

function evidenceScore(probability, threshold) {
  const numeric = Number(probability || 0);
  if (!threshold) return 0;
  return numeric / threshold;
}

function evidencePercent(score) {
  if (!Number.isFinite(score)) return "-";
  return `${Math.round(score * 100)}% of threshold`;
}

function metricRangeFit(value, metric) {
  const numeric = Number(value);
  const low = Number(metric?.diseaseRange?.low);
  const high = Number(metric?.diseaseRange?.high);
  const center = Number(referenceValue(metric));
  if (!Number.isFinite(numeric) || !Number.isFinite(low) || !Number.isFinite(high) || !Number.isFinite(center)) return 0;
  const width = Math.max(high - low, Math.abs(center) * 0.25, 1e-6);
  const distance = numeric >= low && numeric <= high
    ? Math.abs(numeric - center) / width
    : Math.min(Math.abs(numeric - low), Math.abs(numeric - high)) / width + 1;
  return Math.max(0, 1 - Math.min(distance, 2) / 2);
}

function metricQuality(metric, value) {
  const numeric = Number(value);
  const [low, high] = PLAUSIBLE_BIOMETRIC_RANGES[metric?.key] || [];
  if (!Number.isFinite(numeric)) {
    return { usable: false, reason: `${metric?.label || "Metric"} unavailable` };
  }
  if (low === undefined || high === undefined) {
    return { usable: true, reason: "" };
  }
  if (numeric < low || numeric > high) {
    return {
      usable: false,
      reason: `${metric.label} ${metricValue(numeric)} ${metric.unit} is outside the chart reliability range`
    };
  }
  return { usable: true, reason: "" };
}

function biometricFitComparison(sharedMetrics) {
  const weights = {
    walking_speed: 1.1,
    cadence: 0.8,
    step_length: 1.1,
    stride_variability: 1.35,
    arm_swing_amplitude: 1.3,
    arm_swing_asymmetry: 1.35,
    knee_rom: 0.9,
    foot_clearance: 1.25,
    trunk_sway: 1.35
  };
  let pdTotal = 0;
  let neuropathyTotal = 0;
  let weightTotal = 0;
  const outliers = [];
  sharedMetrics.forEach(({ pdMetric, neuropathyMetric }) => {
    if (!pdMetric || !neuropathyMetric) return;
    const value = uploadedValue(neuropathyMetric) ?? uploadedValue(pdMetric);
    const quality = metricQuality(pdMetric, value);
    if (!quality.usable) {
      outliers.push(quality.reason);
      return;
    }
    const weight = weights[pdMetric.key] || 1;
    pdTotal += metricRangeFit(value, pdMetric) * weight;
    neuropathyTotal += metricRangeFit(value, neuropathyMetric) * weight;
    weightTotal += weight;
  });
  if (!weightTotal) {
    return {
      pdFit: 0,
      neuropathyFit: 0,
      usableCount: 0,
      outlierCount: outliers.length,
      text: "Bar-chart biometrics are low reliability; uploaded gait values are outside chart reliability limits"
    };
  }
  const pdFit = pdTotal / weightTotal;
  const neuropathyFit = neuropathyTotal / weightTotal;
  const margin = Math.abs(pdFit - neuropathyFit);
  if (outliers.length >= 2) {
    return {
      pdFit,
      neuropathyFit,
      usableCount: sharedMetrics.length - outliers.length,
      outlierCount: outliers.length,
      text: "Bar-chart biometrics are mixed/low reliability; model outputs should be primary"
    };
  }
  return {
    pdFit,
    neuropathyFit,
    usableCount: sharedMetrics.length - outliers.length,
    outlierCount: outliers.length,
    text: margin < 0.06
      ? "Bar-chart biometrics are currently overlapping"
      : pdFit > neuropathyFit
        ? "Bar-chart biometrics fit PD ranges more closely"
        : "Bar-chart biometrics fit neuropathy ranges more closely"
  };
}

function comparisonInterpretation(pdScreening, neuropathyScreening) {
  const pdProbability = Number(pdScreening?.probability || 0);
  const neuropathyProbability = Number(neuropathyScreening?.probability || 0);
  const pdEvidence = evidenceScore(pdProbability, MODEL_THRESHOLDS.pd);
  const neuropathyEvidence = evidenceScore(neuropathyProbability, MODEL_THRESHOLDS.neuropathy);
  const margin = Math.abs(pdEvidence - neuropathyEvidence);

  if (pdScreening?.detected && !neuropathyScreening?.detected) {
    return {
      text: "The same-video comparison leans closer to PD model evidence",
      confidence: margin >= 0.35 ? "moderate confidence" : "low confidence",
      pdEvidence,
      neuropathyEvidence
    };
  }
  if (neuropathyScreening?.detected && !pdScreening?.detected) {
    return {
      text: "The same-video comparison leans closer to neuropathy model evidence",
      confidence: margin >= 0.35 ? "moderate confidence" : "low confidence",
      pdEvidence,
      neuropathyEvidence
    };
  }
  if (margin < 0.15) {
    return {
      text: "Both model scores are currently similar",
      confidence: "low confidence",
      pdEvidence,
      neuropathyEvidence
    };
  }
  return {
    text: pdEvidence > neuropathyEvidence
      ? "The same-video comparison leans closer to PD model evidence"
      : "The same-video comparison leans closer to neuropathy model evidence",
    confidence: margin >= 0.45 ? "moderate confidence" : "low confidence",
    pdEvidence,
    neuropathyEvidence
  };
}

function metricTooltip(metric, modelLabel = "Detected") {
  const referenceLabel = metric?.diseaseRange?.label || "Disease reference";
  return `${metric.label}
${referenceLabel}: ${metricValue(referenceValue(metric))} ${metric.unit}
${modelLabel}: ${metricValue(uploadedValue(metric))} ${metric.unit}
${diseaseRangeText(metric, referenceLabel)}`;
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
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
          h1, h2, h3, p { margin: 0; }
          .header { display: flex; justify-content: space-between; gap: 18px; margin-bottom: 20px; padding-bottom: 14px; border-bottom: 2px solid #dbeafe; }
          .note { color: #475569; margin-top: 6px; line-height: 1.45; }
          .card { padding: 16px; margin-bottom: 14px; border: 1px solid #cbd5e1; border-radius: 10px; background: #ffffff; break-inside: avoid; }
          .metric { display: grid; grid-template-columns: 210px 1fr 90px 90px; gap: 12px; align-items: center; padding: 10px 0; border-bottom: 1px solid #e2e8f0; }
          .metric:last-child { border-bottom: 0; }
          .track { height: 16px; border-radius: 999px; background: #e2e8f0; overflow: hidden; }
          .bar { height: 100%; border-radius: 999px; }
          .pdf-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; }
          .pdf-metric { display: grid; gap: 10px; padding: 14px; border: 1px solid #cbd5e1; border-radius: 8px; background: #ffffff; break-inside: avoid; }
          .pdf-plot { display: grid; grid-template-columns: 44px 1fr; gap: 10px; min-height: 170px; }
          .pdf-axis { display: flex; flex-direction: column; justify-content: space-between; align-items: flex-end; padding-bottom: 34px; color: #64748b; font-size: 11px; font-weight: 700; }
          .pdf-bars { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; align-items: end; }
          .pdf-bar-group { display: grid; grid-template-rows: 1fr auto auto; gap: 5px; align-items: end; justify-items: center; text-align: center; font-size: 12px; color: #475569; }
          .pdf-bar-shell { width: 52px; height: 132px; display: flex; align-items: flex-end; overflow: hidden; border-radius: 8px 8px 4px 4px; background: #eef2f7; border: 1px solid #cbd5e1; }
          .pdf-vbar { width: 100%; min-height: 4%; border-radius: 8px 8px 4px 4px; }
          .normal { background: #0284c7; }
          .detected { background: #dc2626; }
          .neuro { background: #d97706; }
          .row-bars { display: grid; gap: 5px; }
          .badge { display: inline-flex; padding: 3px 8px; border-radius: 999px; font-size: 12px; font-weight: 700; }
          .green { background: #dcfce7; color: #166534; }
          .yellow { background: #fef3c7; color: #92400e; }
          .red { background: #fee2e2; color: #991b1b; }
          .gray { background: #e2e8f0; color: #334155; }
          @media print { body { background: #ffffff; padding: 18px; } }
        </style>
      </head>
      <body>
        <div class="header">
          <div>
            <h1>${escapeHtml(title)}</h1>
            <p class="note">Generated from saved Component 3 gait screening values. Screening support only; not a medical diagnosis.</p>
          </div>
          <p class="note">${escapeHtml(new Date().toLocaleString())}</p>
        </div>
        ${bodyHtml}
        <script>window.addEventListener("load", () => setTimeout(() => window.print(), 300));</script>
      </body>
    </html>
  `);
  reportWindow.document.close();
}

function metricPdfRows(profile, detectedClass = "detected") {
  return (profile?.metrics || []).map((metric) => `
    <div class="pdf-metric">
      <div>
        <strong>${escapeHtml(metric.label)}</strong><br>
        <p class="note">${escapeHtml(metric.diseaseRange?.source || "Real uploaded gait metrics")}</p>
      </div>
      <div class="pdf-plot">
        <div class="pdf-axis"><span>140%</span><span>70%</span><span>0</span></div>
        <div class="pdf-bars">
          <div class="pdf-bar-group">
            <div class="pdf-bar-shell"><div class="pdf-vbar normal" style="height:${barHeight(referencePercent(metric))}"></div></div>
            <strong>${escapeHtml(metricValue(referenceValue(metric)))}</strong>
            <span>Reference</span>
          </div>
          <div class="pdf-bar-group">
            <div class="pdf-bar-shell"><div class="pdf-vbar ${detectedClass}" style="height:${barHeight(uploadedPercent(metric))}"></div></div>
            <strong>${escapeHtml(metricValue(uploadedValue(metric)))}</strong>
            <span>Your value</span>
          </div>
        </div>
      </div>
    </div>
  `).join("");
}

export function latestScreening(screenings, modelKey) {
  return [...(screenings || [])]
    .filter((item) => item.modelKey === modelKey)
    .sort((a, b) => new Date(b.createdAt || 0) - new Date(a.createdAt || 0))[0] || null;
}

export function sameFileCounterpart(screening, screenings) {
  if (!screening) return null;
  const targetModel = screening.modelKey === "pd" ? "neuropathy" : "pd";
  return [...(screenings || [])]
    .filter((item) => item.modelKey === targetModel && item.fileName === screening.fileName)
    .sort((a, b) => new Date(b.createdAt || 0) - new Date(a.createdAt || 0))[0] || null;
}

export function MetricComparisonChart({ title, screening, modelLabel, onDownload = true }) {
  const profile = screening?.result?.gait_metric_profile;
  if (!profile?.available) {
    return (
      <Card className="sca-koa-workspace-card">
        <CardContent>
          <Alert severity="info">
            Run a new {modelLabel || "Component 3"} screening to generate disease-reference-vs-your-values gait metric bars.
          </Alert>
        </CardContent>
      </Card>
    );
  }
  if (!hasDiseaseReference(profile)) {
    return (
      <Card className="sca-koa-workspace-card">
        <CardContent>
          <Alert severity="info">
            This saved result was created before disease-reference biometrics were added. Run a new {modelLabel || "Component 3"} screening to compare real disease gait values with your uploaded values.
          </Alert>
        </CardContent>
      </Card>
    );
  }

  function downloadPdf() {
    openPdfReport(title, `
      <section class="card">
        <h2>${escapeHtml(screening?.fileName || "Screening")}</h2>
        <p class="note">Blue bars are disease reference gait biometrics. Red bars are your uploaded gait biometrics measured from the selected video or CSV.</p>
      </section>
      <section class="pdf-grid">${metricPdfRows(profile, "detected")}</section>
    `);
  }

  return (
    <Card className="sca-koa-workspace-card">
      <CardContent>
        <Stack spacing={2.5}>
          <Stack direction={{ xs: "column", sm: "row" }} spacing={1.5} alignItems={{ xs: "flex-start", sm: "center" }} justifyContent="space-between">
            <Stack direction="row" spacing={1.5} alignItems="center">
              <BarChartIcon color="primary" />
              <Box>
                <Typography variant="h5" fontWeight={900}>{title}</Typography>
                <Typography color="text.secondary">Blue is disease reference behavior. Red is your uploaded gait value measured from the real input.</Typography>
              </Box>
            </Stack>
            {onDownload && (
              <Button variant="outlined" startIcon={<PictureAsPdfIcon />} onClick={downloadPdf}>
                Download PDF
              </Button>
            )}
          </Stack>

          <Box className="component3-vertical-grid">
            {(profile.metrics || []).map((metric) => (
              <Box
                key={metric.key}
                className="component3-vertical-card component3-hover-tip"
                data-tooltip={metricTooltip(metric, modelLabel || "Detected")}
              >
                <Stack direction="row" spacing={1} justifyContent="space-between" alignItems="flex-start">
                  <Box>
                    <Typography fontWeight={900}>{metric.label}</Typography>
                    <Typography className="component3-chart-unit">{metric.unit}</Typography>
                  </Box>
                  <Chip size="small" variant="outlined" label="Real-world reference" />
                </Stack>
                <Box className="component3-vertical-plot">
                  <Box className="component3-vertical-axis">
                    <span>140%</span>
                    <span>70%</span>
                    <span>0</span>
                  </Box>
                  <Box className="component3-vertical-bars">
                    <Box className="component3-vertical-bar-group">
                      <Box className="component3-vertical-bar-shell">
                        <Box className="component3-vertical-bar normal" sx={{ height: barHeight(referencePercent(metric)) }} />
                      </Box>
                      <strong>{metricValue(referenceValue(metric))}</strong>
                      <span>Disease ref</span>
                    </Box>
                    <Box className="component3-vertical-bar-group">
                      <Box className="component3-vertical-bar-shell">
                        <Box className="component3-vertical-bar detected" sx={{ height: barHeight(uploadedPercent(metric)) }} />
                      </Box>
                      <strong>{metricValue(uploadedValue(metric))}</strong>
                      <span>Your value</span>
                    </Box>
                  </Box>
                </Box>
                <Typography className="component3-range-note">{metric.diseaseRange?.source || "Real-world disease gait reference range."}</Typography>
              </Box>
            ))}
          </Box>
        </Stack>
      </CardContent>
    </Card>
  );
}

export function DiseaseComparisonChart({ pdScreening, neuropathyScreening }) {
  if (!pdScreening || !neuropathyScreening) {
    const present = pdScreening || neuropathyScreening;
    const missing = present?.modelKey === "pd" ? "neuropathy" : "PD";
    const presentLabel = present?.modelKey === "pd" ? "PD" : "Neuropathy";
    const presentProbability = Number(present?.probability || 0);
    return (
      <Card className="sca-koa-workspace-card">
        <CardContent>
          <Stack spacing={2.5}>
            <Alert severity="info">
              {present
                ? `${presentLabel} probability is ready for ${present.fileName}, but there is no same-file ${missing} result yet. Upload the same video or CSV to ${missing} detection to see the PD vs neuropathy comparison.`
                : "No PD and neuropathy results for the same upload yet. Run both models on the same video or CSV to generate the overlap comparison."}
            </Alert>
            {present && (
              <Box
                className={`component3-score-card ${present.modelKey === "pd" ? "pd" : "neuro"} component3-hover-tip`}
                data-tooltip={`${presentLabel} probability: ${percent(presentProbability)}\nFile: ${present.fileName}\nWaiting for same-file ${missing} result`}
              >
                <Typography color="text.secondary">{presentLabel} model probability</Typography>
                <Box className="component3-probability-vertical">
                  <Box className="component3-vertical-bar-shell">
                    <Box
                      className={`component3-vertical-bar ${present.modelKey === "pd" ? "detected" : "neuropathy"}`}
                      sx={{ height: `${Math.max(4, presentProbability * 100)}%` }}
                    />
                  </Box>
                  <Box>
                    <Typography variant="h3" fontWeight={900}>{percent(presentProbability)}</Typography>
                    <Typography color="text.secondary">Waiting for same-file {missing} result</Typography>
                  </Box>
                </Box>
              </Box>
            )}
          </Stack>
        </CardContent>
      </Card>
    );
  }

  const pdProbability = Number(pdScreening.probability || 0);
  const neuropathyProbability = Number(neuropathyScreening.probability || 0);
  const interpretation = comparisonInterpretation(pdScreening, neuropathyScreening);
  const stronger = `${interpretation.text} (${interpretation.confidence})`;
  const pdProfile = pdScreening.result?.gait_metric_profile;
  const neuropathyProfile = neuropathyScreening.result?.gait_metric_profile;
  if (!hasDiseaseReference(pdProfile) || !hasDiseaseReference(neuropathyProfile)) {
    return (
      <Card className="sca-koa-workspace-card">
        <CardContent>
          <Stack spacing={2.5}>
            <Alert severity="info">
              This same-file comparison needs fresh disease-reference biometrics. Run both PD and neuropathy detection again on the same video or CSV to show PD ranges, neuropathy ranges, and your measured values side by side.
            </Alert>
            <Box className="component3-probability-compare">
              <Box className="component3-score-card pd">
                <Typography color="text.secondary">PD model probability</Typography>
                <Typography variant="h3" fontWeight={900}>{percent(pdProbability)}</Typography>
                <Typography color="text.secondary">Evidence {evidencePercent(interpretation.pdEvidence)}</Typography>
              </Box>
              <Box className="component3-score-card neuro">
                <Typography color="text.secondary">Neuropathy model probability</Typography>
                <Typography variant="h3" fontWeight={900}>{percent(neuropathyProbability)}</Typography>
                <Typography color="text.secondary">Evidence {evidencePercent(interpretation.neuropathyEvidence)}</Typography>
              </Box>
            </Box>
          </Stack>
        </CardContent>
      </Card>
    );
  }
  const neuropathyMetricsByKey = new Map((neuropathyProfile?.metrics || []).map((metric) => [metric.key, metric]));
  const sharedMetrics = (pdProfile?.metrics || []).map((pdMetric) => ({
    pdMetric,
    neuropathyMetric: neuropathyMetricsByKey.get(pdMetric.key)
  }));
  const biometricFit = biometricFitComparison(sharedMetrics);

  function downloadPdf() {
    const metricRows = sharedMetrics.map(({ pdMetric, neuropathyMetric }) => `
      <div class="metric">
        <div><strong>${escapeHtml(pdMetric.label)}</strong><p class="note">${escapeHtml(pdMetric.unit)}</p></div>
        <div>
          <strong>PD</strong>
          <p class="note">${escapeHtml(diseaseRangeText(pdMetric, "PD"))}</p>
          <p>Your value: ${escapeHtml(metricValue(uploadedValue(pdMetric)))} ${escapeHtml(pdMetric.unit)}</p>
        </div>
        <div>
          <strong>Neuropathy</strong>
          <p class="note">${escapeHtml(neuropathyMetric ? diseaseRangeText(neuropathyMetric, "Neuropathy") : "Neuropathy range unavailable")}</p>
          <p>Your value: ${escapeHtml(metricValue(uploadedValue(neuropathyMetric)))} ${escapeHtml(neuropathyMetric?.unit || pdMetric.unit)}</p>
        </div>
      </div>
    `).join("");
    openPdfReport("PD vs Neuropathy Same-Video Comparison", `
      <section class="card">
        <h2>${escapeHtml(stronger)}</h2>
        <p class="note">PD file: ${escapeHtml(pdScreening.fileName)} | Neuropathy file: ${escapeHtml(neuropathyScreening.fileName)}</p>
        <p class="note">Evidence is normalized against each model threshold: PD ${escapeHtml(evidencePercent(interpretation.pdEvidence))}; Neuropathy ${escapeHtml(evidencePercent(interpretation.neuropathyEvidence))}.</p>
        <p class="note">${escapeHtml(biometricFit.text)}: PD fit ${escapeHtml(percent(biometricFit.pdFit))}; Neuropathy fit ${escapeHtml(percent(biometricFit.neuropathyFit))}; ignored outliers ${escapeHtml(biometricFit.outlierCount || 0)}.</p>
      </section>
      <section class="card">
        <div class="metric"><strong>PD model probability</strong><div class="track"><div class="bar detected" style="width:${Math.round(pdProbability * 100)}%"></div></div><div></div><strong>${escapeHtml(percent(pdProbability))}</strong></div>
        <div class="metric"><strong>Neuropathy model probability</strong><div class="track"><div class="bar neuro" style="width:${Math.round(neuropathyProbability * 100)}%"></div></div><div></div><strong>${escapeHtml(percent(neuropathyProbability))}</strong></div>
      </section>
      <section class="card">${metricRows}</section>
    `);
  }

  return (
    <Card className="sca-koa-workspace-card">
      <CardContent>
        <Stack spacing={2.5}>
          <Stack direction={{ xs: "column", sm: "row" }} spacing={1.5} alignItems={{ xs: "flex-start", sm: "center" }} justifyContent="space-between">
            <Stack direction="row" spacing={1.5} alignItems="center">
              <BarChartIcon color="primary" />
              <Box>
                <Typography variant="h5" fontWeight={900}>PD vs Neuropathy</Typography>
                <Typography color="text.secondary">{stronger}. Evidence is normalized against each model threshold before comparing.</Typography>
                <Typography color="text.secondary">{biometricFit.text}: PD fit {percent(biometricFit.pdFit)}, Neuropathy fit {percent(biometricFit.neuropathyFit)}, ignored outliers {biometricFit.outlierCount || 0}.</Typography>
              </Box>
            </Stack>
            <Button variant="outlined" startIcon={<PictureAsPdfIcon />} onClick={downloadPdf}>
              Download PDF
            </Button>
          </Stack>

          <Box className="component3-probability-compare">
            <Box
              className="component3-score-card pd component3-hover-tip"
              data-tooltip={`PD probability: ${percent(pdProbability)}\nEvidence: ${evidencePercent(interpretation.pdEvidence)}\nThreshold: ${percent(MODEL_THRESHOLDS.pd)}\nFile: ${pdScreening.fileName}`}
            >
              <Typography color="text.secondary">PD model probability</Typography>
              <Box className="component3-probability-vertical">
                <Box className="component3-vertical-bar-shell">
                  <Box className="component3-vertical-bar detected" sx={{ height: `${Math.max(4, pdProbability * 100)}%` }} />
                </Box>
                <Typography variant="h3" fontWeight={900}>{percent(pdProbability)}</Typography>
                <Typography color="text.secondary">Evidence {evidencePercent(interpretation.pdEvidence)}</Typography>
              </Box>
            </Box>
            <Box
              className="component3-score-card neuro component3-hover-tip"
              data-tooltip={`Neuropathy probability: ${percent(neuropathyProbability)}\nEvidence: ${evidencePercent(interpretation.neuropathyEvidence)}\nThreshold: ${percent(MODEL_THRESHOLDS.neuropathy)}\nFile: ${neuropathyScreening.fileName}`}
            >
              <Typography color="text.secondary">Neuropathy model probability</Typography>
              <Box className="component3-probability-vertical">
                <Box className="component3-vertical-bar-shell">
                  <Box className="component3-vertical-bar neuropathy" sx={{ height: `${Math.max(4, neuropathyProbability * 100)}%` }} />
                </Box>
                <Typography variant="h3" fontWeight={900}>{percent(neuropathyProbability)}</Typography>
                <Typography color="text.secondary">Evidence {evidencePercent(interpretation.neuropathyEvidence)}</Typography>
              </Box>
            </Box>
          </Box>

          <Box className="component3-overlap-list">
            {sharedMetrics.map(({ pdMetric, neuropathyMetric }) => (
              <Box
                key={pdMetric.key}
                className="component3-overlap-row"
              >
                <Box>
                  <Typography fontWeight={900}>{pdMetric.label}</Typography>
                  <Typography color="text.secondary">{pdMetric.unit}</Typography>
                </Box>
                <Box className="component3-range-column pd">
                  <Typography fontWeight={900}>PD range</Typography>
                  <Typography color="text.secondary">{diseaseRangeText(pdMetric, "PD")}</Typography>
                  <Typography>Your value: {metricValue(uploadedValue(pdMetric))} {pdMetric.unit}</Typography>
                </Box>
                <Box className="component3-range-column neuro">
                  <Typography fontWeight={900}>Neuropathy range</Typography>
                  <Typography color="text.secondary">{neuropathyMetric ? diseaseRangeText(neuropathyMetric, "Neuropathy") : "Neuropathy range unavailable"}</Typography>
                  <Typography>Your value: {metricValue(uploadedValue(neuropathyMetric))} {neuropathyMetric?.unit || pdMetric.unit}</Typography>
                </Box>
              </Box>
            ))}
          </Box>
        </Stack>
      </CardContent>
    </Card>
  );
}
