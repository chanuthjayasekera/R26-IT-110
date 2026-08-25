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
  Grid2,
  IconButton,
  LinearProgress,
  Stack,
  Typography
} from "@mui/material";
import { Link, useLocation, useNavigate, useSearchParams } from "react-router-dom";
import AccountTreeIcon from "@mui/icons-material/AccountTree";
import AssessmentIcon from "@mui/icons-material/Assessment";
import CheckCircleIcon from "@mui/icons-material/CheckCircle";
import ChevronLeftIcon from "@mui/icons-material/ChevronLeft";
import ChevronRightIcon from "@mui/icons-material/ChevronRight";
import DeleteOutlineIcon from "@mui/icons-material/DeleteOutline";
import HistoryIcon from "@mui/icons-material/History";
import PictureAsPdfIcon from "@mui/icons-material/PictureAsPdf";
import ReplayIcon from "@mui/icons-material/Replay";
import VideoCameraBackIcon from "@mui/icons-material/VideoCameraBack";
import WarningAmberIcon from "@mui/icons-material/WarningAmber";
import CentralProfileFlagButton from "../../common/components/CentralProfileFlagButton.jsx";
import { api, getApiError } from "../../common/api/http.js";
import "../styles/rehab-exercise-detection.css";

function formatDate(value) {
  if (!value) return "";
  return new Intl.DateTimeFormat("en", { dateStyle: "medium", timeStyle: "short" }).format(new Date(value));
}

function percent(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
  return `${Math.round(Number(value) * 100)}%`;
}

function probability(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return 0;
  return Math.min(Math.max(Number(value), 0), 1);
}

function timeRange(item) {
  return `${Number(item?.start_seconds || 0).toFixed(2)}s - ${Number(item?.end_seconds || 0).toFixed(2)}s`;
}

function apiAssetUrl(path) {
  if (!path) return "";
  if (/^https?:\/\//i.test(path)) return path;
  return `${api.defaults.baseURL}${path}`;
}

function resultTone(screening) {
  if (screening?.reliabilityLevel === "Low") return "warning";
  return screening?.isCorrect ? "success" : "error";
}

function resultLabel(screening) {
  if (screening?.reliabilityLevel === "Low") return "Review needed";
  return screening?.isCorrect ? "Correct posture" : "Incorrect posture";
}

function resultVisual(screening) {
  if (screening?.reliabilityLevel === "Low") {
    return {
      className: "rehab-result-billboard warning",
      icon: <WarningAmberIcon />,
      label: "Review needed",
      caption: "The model ran, but the recording is close to the decision boundary or has limited pose evidence."
    };
  }
  if (screening?.isCorrect) {
    return {
      className: "rehab-result-billboard clear",
      icon: <CheckCircleIcon />,
      label: "Correct posture",
      caption: "The movement matches the learned correct exercise pattern across the analyzed windows."
    };
  }
  return {
    className: "rehab-result-billboard detected",
    icon: <WarningAmberIcon />,
    label: "Incorrect posture",
    caption: "The movement contains windows that do not match the learned correct exercise pattern."
  };
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function windowRowsHtml(title, windows) {
  const rows = (windows || []).length
    ? windows.map((item) => `
        <tr>
          <td>${item.window_id + 1}</td>
          <td>${escapeHtml(timeRange(item))}</td>
          <td>${percent(item.correct_probability)}</td>
        </tr>
      `).join("")
    : `<tr><td colspan="3">No ${escapeHtml(title.toLowerCase())} windows found.</td></tr>`;

  return `
    <section class="report-card">
      <h2>${escapeHtml(title)}</h2>
      <table>
        <thead><tr><th>Window</th><th>Time</th><th>Correct probability</th></tr></thead>
        <tbody>${rows}</tbody>
      </table>
    </section>
  `;
}

function openExercisePdf(screening) {
  const report = screening.windowReport || {};
  const reportWindow = window.open("", "_blank", "width=1100,height=800");
  if (!reportWindow) return;
  reportWindow.document.write(`
    <!doctype html>
    <html>
      <head>
        <title>${escapeHtml(screening.exerciseLabel)} Report</title>
        <style>
          * { box-sizing: border-box; }
          body { margin: 0; padding: 28px; font-family: Arial, sans-serif; color: #0f172a; background: #f8fafc; }
          h1, h2 { margin: 0; }
          h1 { font-size: 28px; }
          h2 { font-size: 18px; margin-bottom: 10px; }
          p { margin: 4px 0; color: #475569; line-height: 1.45; }
          table { width: 100%; border-collapse: collapse; }
          th, td { padding: 9px 10px; border-bottom: 1px solid #e2e8f0; text-align: left; font-size: 13px; }
          th { color: #334155; background: #eef6ff; }
          .report-header { display: flex; justify-content: space-between; gap: 16px; align-items: flex-start; margin-bottom: 20px; padding-bottom: 14px; border-bottom: 2px solid #bae6fd; }
          .report-card { break-inside: avoid; padding: 16px; margin-bottom: 14px; border: 1px solid #cbd5e1; border-radius: 10px; background: #ffffff; }
          .summary-grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin: 16px 0; }
          .summary-cell { padding: 14px; border-radius: 10px; background: #eef6ff; border: 1px solid #bfdbfe; }
          .badge { display: inline-flex; align-items: center; padding: 4px 9px; border-radius: 999px; font-weight: 700; font-size: 12px; }
          .good { background: #dcfce7; color: #166534; }
          .bad { background: #fee2e2; color: #991b1b; }
          .warn { background: #fef3c7; color: #92400e; }
          @media print {
            body { background: #ffffff; padding: 18px; }
            .report-card { page-break-inside: avoid; }
          }
        </style>
      </head>
      <body>
        <div class="report-header">
          <div>
            <h1>${escapeHtml(screening.exerciseLabel)} Report</h1>
            <p>${escapeHtml(screening.fileName)} | ${escapeHtml(formatDate(screening.createdAt))}</p>
          </div>
          <span class="badge ${screening.isCorrect ? "good" : "bad"}">${escapeHtml(resultLabel(screening))}</span>
        </div>
        <div class="summary-grid">
          <div class="summary-cell"><h2>${escapeHtml(screening.finalPrediction || "-")}</h2><p>Final outcome</p></div>
          <div class="summary-cell"><h2>${percent(screening.meanCorrectProbability)}</h2><p>Correct probability</p></div>
          <div class="summary-cell"><h2>${screening.numWindows ?? "-"}</h2><p>Windows analyzed</p></div>
          <div class="summary-cell"><h2>${escapeHtml(screening.reliabilityLevel || "-")}</h2><p>Reliability</p></div>
        </div>
        <section class="report-card">
          <h2>Interpretation</h2>
          <p>${report.perfect_report ? "Perfect report: every analyzed window was classified as correct." : "The report lists where incorrect posture windows occurred so users can review the relevant exercise stage."}</p>
          ${(screening.reliabilityReasons || []).map((note) => `<p class="badge warn">${escapeHtml(note)}</p>`).join("")}
        </section>
        ${windowRowsHtml("Incorrect Windows", report.incorrect_windows)}
        ${windowRowsHtml("Correct Windows", report.correct_windows)}
        <script>
          window.addEventListener("load", () => setTimeout(() => window.print(), 300));
        </script>
      </body>
    </html>
  `);
  reportWindow.document.close();
}

function Metric({ label, value, helper }) {
  return (
    <Box className="rehab-metric-tile">
      <Typography color="text.secondary">{label}</Typography>
      <Typography variant="h5" fontWeight={900}>{value ?? "-"}</Typography>
      {helper && <Typography className="rehab-metric-helper">{helper}</Typography>}
    </Box>
  );
}

export function ExerciseOverview({ screening, totalScreenings = 1 }) {
  if (!screening) return null;
  const report = screening.windowReport || {};
  const incorrectWindows = report.incorrect_windows || [];
  const correctWindows = report.correct_windows || [];
  const score = probability(screening.meanCorrectProbability);
  const degrees = Math.round(score * 360);
  const gaugeBackground = `conic-gradient(#16a34a 0deg ${degrees}deg, #ef4444 ${degrees}deg 360deg)`;
  const visual = resultVisual(screening);

  return (
    <Stack spacing={3}>
      <Box className={visual.className}>
        <Box className="rehab-result-symbol">{visual.icon}</Box>
        <Box>
          <Typography className="rehab-result-kicker">Current exercise quality result</Typography>
          <Typography component="h2" className="rehab-result-label">{visual.label}</Typography>
          <Typography className="rehab-result-caption">{visual.caption}</Typography>
        </Box>
      </Box>

      <Grid2 container spacing={3}>
        <Grid2 size={{ xs: 12, lg: 4 }}>
          <Card className="rehab-workspace-card rehab-gauge-card">
            <CardContent>
              <Stack spacing={2.5} alignItems="center" textAlign="center">
                <Box className="rehab-probability-gauge" sx={{ background: gaugeBackground }}>
                  <Box>
                    <Typography variant="h3" fontWeight={900}>{percent(score)}</Typography>
                    <Typography>Correct probability</Typography>
                  </Box>
                </Box>
                <Box>
                  <Typography variant="h5" fontWeight={900}>{screening.finalPrediction}</Typography>
                  <Typography color="text.secondary">{report.perfect_report ? "Perfect report: all analyzed windows were correct." : "Review incorrect windows to see the stages that need attention."}</Typography>
                </Box>
                <LinearProgress className="rehab-confidence-bar" variant="determinate" value={Math.round(score * 100)} />
              </Stack>
            </CardContent>
          </Card>
        </Grid2>

        <Grid2 size={{ xs: 12, lg: 8 }}>
          <Grid2 container spacing={2}>
            <Grid2 size={{ xs: 12, md: 4 }}>
              <Metric label="Latest result" value={resultLabel(screening)} helper={screening.exerciseLabel} />
            </Grid2>
            <Grid2 size={{ xs: 12, md: 4 }}>
              <Metric label="Reliability" value={screening.reliabilityLevel || "-"} helper="Model quality level" />
            </Grid2>
            <Grid2 size={{ xs: 12, md: 4 }}>
              <Metric label="Quality score" value={screening.qualityScore?.toFixed ? screening.qualityScore.toFixed(1) : screening.qualityScore} helper="0 to 100" />
            </Grid2>
            <Grid2 size={{ xs: 12, md: 4 }}>
              <Metric label="Correct windows" value={correctWindows.length} helper={`Ratio ${percent(screening.correctRatio)}`} />
            </Grid2>
            <Grid2 size={{ xs: 12, md: 4 }}>
              <Metric label="Incorrect windows" value={incorrectWindows.length} helper={`Ratio ${percent(screening.incorrectRatio)}`} />
            </Grid2>
            <Grid2 size={{ xs: 12, md: 4 }}>
              <Metric label="Total saved" value={totalScreenings} helper="Runs for this exercise" />
            </Grid2>
          </Grid2>
        </Grid2>
      </Grid2>

      <Card className="rehab-workspace-card">
        <CardContent>
          <Stack spacing={2}>
            <Typography variant="h5" fontWeight={900}>Reliability Notes</Typography>
            {(screening.reliabilityReasons || []).length === 0 ? (
              <Alert severity="info">No additional reliability notes were returned for this analysis.</Alert>
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

export function WindowList({ title, windows, tone }) {
  return (
    <Card className="rehab-workspace-card">
      <CardContent>
        <Stack spacing={2}>
          <Stack direction="row" spacing={1} alignItems="center">
            <Chip color={tone} label={title} />
            <Typography color="text.secondary">{windows?.length || 0} windows</Typography>
          </Stack>
          {(windows || []).length === 0 ? (
            <Alert severity={tone === "success" ? "success" : "info"}>No {title.toLowerCase()} found.</Alert>
          ) : (
            <Stack spacing={1}>
              {windows.slice(0, 12).map((item) => (
                <Box key={`${title}-${item.window_id}`} className="rehab-window-row">
                  <Typography fontWeight={900}>Window {item.window_id + 1}</Typography>
                  <Typography>{timeRange(item)}</Typography>
                  <Typography color="text.secondary">Correct probability {percent(item.correct_probability)}</Typography>
                </Box>
              ))}
              {windows.length > 12 && (
                <Typography color="text.secondary">Showing 12 of {windows.length} windows. Download the PDF for the full report.</Typography>
              )}
            </Stack>
          )}
        </Stack>
      </CardContent>
    </Card>
  );
}

export default function RehabExerciseResults() {
  const location = useLocation();
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const requestedId = searchParams.get("id");
  const activeSection = searchParams.get("section") || "overview";
  const [screening, setScreening] = useState(location.state?.screening || null);
  const [profile, setProfile] = useState(location.state?.profile || null);
  const [screenings, setScreenings] = useState([]);
  const [loading, setLoading] = useState(!location.state?.screening);
  const [error, setError] = useState("");
  const [deletingId, setDeletingId] = useState("");
  const [navCollapsed, setNavCollapsed] = useState(false);

  useEffect(() => {
    let alive = true;
    async function load() {
      setLoading(true);
      setError("");
      try {
        const profileRes = await api.get("/exercise-detection/clinical-profile");
        if (!alive) return;
        setProfile(profileRes.data.profile);
        setScreenings(profileRes.data.screenings || []);
        if (requestedId) {
          const screeningRes = await api.get(`/exercise-detection/screenings/${requestedId}`);
          if (!alive) return;
          setScreening(screeningRes.data.screening);
        } else if (!location.state?.screening) {
          setScreening(profileRes.data.screenings?.[0] || null);
        }
      } catch (err) {
        if (!alive) return;
        const apiError = getApiError(err);
        setError(apiError.message || "Unable to load the exercise profile.");
      } finally {
        if (alive) setLoading(false);
      }
    }
    load();
    return () => {
      alive = false;
    };
  }, [requestedId, location.state?.screening]);

  const score = probability(screening?.meanCorrectProbability);
  const gaugeBackground = useMemo(() => {
    const degrees = Math.round(score * 360);
    return `conic-gradient(#16a34a 0deg ${degrees}deg, #ef4444 ${degrees}deg 360deg)`;
  }, [score]);

  function getSectionLink(section) {
    const params = new URLSearchParams();
    if (screening?.id) params.set("id", screening.id);
    if (section !== "overview") params.set("section", section);
    const query = params.toString();
    return `/patient/results/rehab-exercise${query ? `?${query}` : ""}`;
  }

  async function deleteScreening(id) {
    if (!window.confirm("Delete this saved exercise result?")) return;
    setDeletingId(id);
    setError("");
    try {
      const res = await api.delete(`/exercise-detection/screenings/${id}`);
      const remaining = res.data.screenings || [];
      setProfile(res.data.profile);
      setScreenings(remaining);
      if (id === screening?.id) {
        const nextScreening = remaining[0] || null;
        setScreening(nextScreening);
        navigate(nextScreening ? `/patient/results/rehab-exercise?id=${nextScreening.id}` : "/patient/results/rehab-exercise", { replace: true });
      }
    } catch (err) {
      const apiError = getApiError(err);
      setError(apiError.message || "Unable to delete exercise result.");
    } finally {
      setDeletingId("");
    }
  }

  if (loading) {
    return (
      <Container maxWidth="lg" className="rehab-page">
        <Card className="rehab-workspace-card">
          <CardContent>
            <Stack spacing={2} alignItems="center" py={5}>
              <CircularProgress />
              <Typography>Loading exercise profile...</Typography>
            </Stack>
          </CardContent>
        </Card>
      </Container>
    );
  }

  if (error) {
    return (
      <Container maxWidth="lg" className="rehab-page">
        <Alert severity="error">{error}</Alert>
      </Container>
    );
  }

  if (!screening) {
    return (
      <Container maxWidth="lg" className="rehab-page">
        <Card className="rehab-workspace-card">
          <CardContent>
            <Stack spacing={2} alignItems="center" textAlign="center" py={5}>
              <AssessmentIcon color="primary" />
              <Typography variant="h4" fontWeight={900}>No exercise profile yet</Typography>
              <Typography color="text.secondary">Run the first exercise analysis and this page will show the saved outcome profile.</Typography>
              <Button component={Link} to="/patient/detection/rehab-exercise" variant="contained" startIcon={<ReplayIcon />}>
                Start exercise analysis
              </Button>
            </Stack>
          </CardContent>
        </Card>
      </Container>
    );
  }

  const visual = resultVisual(screening);
  const sameExerciseScreenings = screenings.filter((item) => item.exerciseKey === screening.exerciseKey);
  const report = screening.windowReport || {};
  const incorrectWindows = report.incorrect_windows || [];
  const correctWindows = report.correct_windows || [];
  const itemClass = navCollapsed ? "rehab-nav-label hidden" : "rehab-nav-label";

  return (
    <Container maxWidth="xl" className="rehab-page">
      <Grid2 container spacing={3}>
        <Grid2 size={{ xs: 12, md: navCollapsed ? 1.1 : 3, lg: navCollapsed ? 0.9 : 2.7 }}>
          <Box className={navCollapsed ? "rehab-side-nav collapsed" : "rehab-side-nav"}>
            <Button
              className="rehab-nav-toggle"
              variant="outlined"
              onClick={() => setNavCollapsed((old) => !old)}
              startIcon={navCollapsed ? <ChevronRightIcon /> : <ChevronLeftIcon />}
            >
              <span className={itemClass}>{navCollapsed ? "" : "Collapse"}</span>
            </Button>
            <Button component={Link} to={`/patient/detection/rehab-exercise?exercise=${screening.exerciseKey}`} variant="outlined" startIcon={<VideoCameraBackIcon />}>
              <span className={itemClass}>New analysis</span>
            </Button>
            <Button component={Link} to={getSectionLink("overview")} variant={activeSection === "overview" ? "contained" : "text"} startIcon={<AssessmentIcon />}>
              <span className={itemClass}>Overview</span>
            </Button>
            <Button component={Link} to={getSectionLink("windows")} variant={activeSection === "windows" ? "contained" : "text"} startIcon={<WarningAmberIcon />}>
              <span className={itemClass}>Window report</span>
            </Button>
            <Button component={Link} to={getSectionLink("history")} variant={activeSection === "history" ? "contained" : "text"} startIcon={<HistoryIcon />}>
              <span className={itemClass}>History</span>
            </Button>
            <Button component={Link} to={getSectionLink("video")} variant={activeSection === "video" ? "contained" : "text"} startIcon={<VideoCameraBackIcon />}>
              <span className={itemClass}>Video</span>
            </Button>
            <Button component={Link} to="/patient/central-profile" variant="text" startIcon={<AccountTreeIcon />}>
              <span className={itemClass}>Centralized Profile</span>
            </Button>
            <Button component={Link} to="/patient" variant="text">
              <span className={itemClass}>Dashboard</span>
            </Button>
          </Box>
        </Grid2>

        <Grid2 size={{ xs: 12, md: navCollapsed ? 10.9 : 9, lg: navCollapsed ? 11.1 : 9.3 }}>
          <Stack spacing={3}>
            <Box className="rehab-hero compact">
              <Stack spacing={1.5}>
                <Chip color={resultTone(screening)} label={`${screening.exerciseLabel} result`} />
                <Typography variant="h3" fontWeight={900}>Exercise Quality Profile</Typography>
                <Typography color="text.secondary">Saved model outcomes, correctness probability, timed posture windows, and report export.</Typography>
              </Stack>
              <Stack direction={{ xs: "column", sm: "row" }} spacing={1.2}>
                <CentralProfileFlagButton sourceType={`exercise_${screening.exerciseKey}`} screeningId={screening.id} />
                <Button component={Link} to={`/patient/detection/rehab-exercise?exercise=${screening.exerciseKey}`} variant="contained" startIcon={<ReplayIcon />}>
                  New analysis
                </Button>
                <Button variant="outlined" startIcon={<PictureAsPdfIcon />} onClick={() => openExercisePdf(screening)}>
                  Download PDF
                </Button>
                <Button color="error" variant="outlined" startIcon={<DeleteOutlineIcon />} disabled={deletingId === screening.id} onClick={() => deleteScreening(screening.id)}>
                  Delete result
                </Button>
              </Stack>
            </Box>

            {activeSection === "overview" && (
              <>
                <Box className={visual.className}>
                  <Box className="rehab-result-symbol">{visual.icon}</Box>
                  <Box>
                    <Typography className="rehab-result-kicker">Current exercise quality result</Typography>
                    <Typography component="h2" className="rehab-result-label">{visual.label}</Typography>
                    <Typography className="rehab-result-caption">{visual.caption}</Typography>
                  </Box>
                </Box>

                <Grid2 container spacing={3}>
                  <Grid2 size={{ xs: 12, lg: 4 }}>
                    <Card className="rehab-workspace-card rehab-gauge-card">
                      <CardContent>
                        <Stack spacing={2.5} alignItems="center" textAlign="center">
                          <Box className="rehab-probability-gauge" sx={{ background: gaugeBackground }}>
                            <Box>
                              <Typography variant="h3" fontWeight={900}>{percent(score)}</Typography>
                              <Typography>Correct probability</Typography>
                            </Box>
                          </Box>
                          <Box>
                            <Typography variant="h5" fontWeight={900}>{screening.finalPrediction}</Typography>
                            <Typography color="text.secondary">{report.perfect_report ? "Perfect report: all analyzed windows were correct." : "Review incorrect windows to see the stages that need attention."}</Typography>
                          </Box>
                          <LinearProgress className="rehab-confidence-bar" variant="determinate" value={Math.round(score * 100)} />
                        </Stack>
                      </CardContent>
                    </Card>
                  </Grid2>

                  <Grid2 size={{ xs: 12, lg: 8 }}>
                    <Grid2 container spacing={2}>
                      <Grid2 size={{ xs: 12, md: 4 }}>
                        <Metric label="Latest result" value={resultLabel(screening)} helper={screening.exerciseLabel} />
                      </Grid2>
                      <Grid2 size={{ xs: 12, md: 4 }}>
                        <Metric label="Reliability" value={screening.reliabilityLevel || "-"} helper="Model quality level" />
                      </Grid2>
                      <Grid2 size={{ xs: 12, md: 4 }}>
                        <Metric label="Quality score" value={screening.qualityScore?.toFixed ? screening.qualityScore.toFixed(1) : screening.qualityScore} helper="0 to 100" />
                      </Grid2>
                      <Grid2 size={{ xs: 12, md: 4 }}>
                        <Metric label="Correct windows" value={correctWindows.length} helper={`Ratio ${percent(screening.correctRatio)}`} />
                      </Grid2>
                      <Grid2 size={{ xs: 12, md: 4 }}>
                        <Metric label="Incorrect windows" value={incorrectWindows.length} helper={`Ratio ${percent(screening.incorrectRatio)}`} />
                      </Grid2>
                      <Grid2 size={{ xs: 12, md: 4 }}>
                        <Metric label="Total saved" value={sameExerciseScreenings.length} helper="Runs for this exercise" />
                      </Grid2>
                    </Grid2>
                  </Grid2>
                </Grid2>

                <Card className="rehab-workspace-card">
                  <CardContent>
                    <Stack spacing={2}>
                      <Typography variant="h5" fontWeight={900}>Reliability Notes</Typography>
                      {(screening.reliabilityReasons || []).length === 0 ? (
                        <Alert severity="info">No additional reliability notes were returned for this analysis.</Alert>
                      ) : (
                        screening.reliabilityReasons.map((note) => (
                          <Alert key={note} severity={screening.reliabilityLevel === "High" ? "success" : "warning"}>{note}</Alert>
                        ))
                      )}
                    </Stack>
                  </CardContent>
                </Card>
              </>
            )}

            {activeSection === "windows" && (
              <Grid2 container spacing={3}>
                <Grid2 size={{ xs: 12 }}>
                  {report.perfect_report ? (
                    <Alert severity="success">Perfect report generated: no incorrect posture windows were detected.</Alert>
                  ) : (
                    <Alert severity="warning">Incorrect windows identify the exercise stages that need posture review.</Alert>
                  )}
                </Grid2>
                <Grid2 size={{ xs: 12, lg: 6 }}>
                  <WindowList title="Incorrect Windows" windows={incorrectWindows} tone="error" />
                </Grid2>
                <Grid2 size={{ xs: 12, lg: 6 }}>
                  <WindowList title="Correct Windows" windows={correctWindows} tone="success" />
                </Grid2>
              </Grid2>
            )}

            {activeSection === "history" && (
              <Card className="rehab-workspace-card">
                <CardContent>
                  <Stack spacing={2}>
                    <Typography variant="h5" fontWeight={900}>{screening.exerciseLabel} Saved History</Typography>
                    {sameExerciseScreenings.map((item) => (
                      <Box key={item.id} className={item.id === screening.id ? "rehab-timeline-item active" : "rehab-timeline-item"}>
                        <Stack spacing={0.5}>
                          <Stack direction="row" spacing={1} alignItems="center" flexWrap="wrap">
                            <Chip size="small" color={resultTone(item)} label={resultLabel(item)} />
                            <Typography fontWeight={900}>{item.finalPrediction}</Typography>
                          </Stack>
                          <Typography color="text.secondary">{formatDate(item.createdAt)} | Correct probability {percent(item.meanCorrectProbability)} | Incorrect windows {item.windowReport?.incorrect_windows?.length ?? 0}</Typography>
                        </Stack>
                        <Stack direction="row" spacing={1}>
                          <Button component={Link} to={`/patient/results/rehab-exercise?id=${item.id}&section=history`} size="small">Open</Button>
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

            {activeSection === "video" && (
              <Grid2 container spacing={3}>
                <Grid2 size={{ xs: 12, lg: 6 }}>
                  <Card className="rehab-workspace-card">
                    <CardContent>
                      <Stack spacing={2}>
                        <Typography variant="h5" fontWeight={900}>Uploaded Video</Typography>
                        <video className="rehab-result-video" controls src={apiAssetUrl(screening.videoUrl)} />
                      </Stack>
                    </CardContent>
                  </Card>
                </Grid2>
                <Grid2 size={{ xs: 12, lg: 6 }}>
                  <Card className="rehab-workspace-card">
                    <CardContent>
                      <Stack spacing={2}>
                        <Typography variant="h5" fontWeight={900}>Annotated Video</Typography>
                        {screening.annotatedVideoUrl ? (
                          <video className="rehab-result-video" controls src={apiAssetUrl(screening.annotatedVideoUrl)} />
                        ) : (
                          <Alert severity="info">No annotated video was saved for this result.</Alert>
                        )}
                      </Stack>
                    </CardContent>
                  </Card>
                </Grid2>
              </Grid2>
            )}
          </Stack>
        </Grid2>
      </Grid2>
    </Container>
  );
}
