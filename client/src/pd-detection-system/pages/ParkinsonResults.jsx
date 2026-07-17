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
import BarChartIcon from "@mui/icons-material/BarChart";
import CheckCircleIcon from "@mui/icons-material/CheckCircle";
import ChevronLeftIcon from "@mui/icons-material/ChevronLeft";
import ChevronRightIcon from "@mui/icons-material/ChevronRight";
import CompareArrowsIcon from "@mui/icons-material/CompareArrows";
import DeleteOutlineIcon from "@mui/icons-material/DeleteOutline";
import ErrorOutlineIcon from "@mui/icons-material/ErrorOutline";
import HealingIcon from "@mui/icons-material/Healing";
import PsychologyIcon from "@mui/icons-material/Psychology";
import ReplayIcon from "@mui/icons-material/Replay";
import VideoCameraBackIcon from "@mui/icons-material/VideoCameraBack";
import WarningAmberIcon from "@mui/icons-material/WarningAmber";
import CentralProfileFlagButton from "../../common/components/CentralProfileFlagButton.jsx";
import { api, getApiError } from "../../common/api/http.js";
import {
  DiseaseComparisonChart,
  MetricComparisonChart,
  sameFileCounterpart
} from "../components/Component3Charts.jsx";
import "../../sca-koa-detection-system/styles/sca-koa-detection.css";
import "../styles/parkinson-detection.css";

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
      caption: `This upload is screened as ${model} gait detected.`
    };
  }
  if (screening?.tendency) {
    return {
      className: "sca-koa-result-billboard warning",
      icon: <WarningAmberIcon />,
      label: gaitDecisionLabel(screening),
      caption: `This upload shows borderline overlap with the trained ${model} gait pattern.`
    };
  }
  return {
    className: "sca-koa-result-billboard clear",
    icon: <CheckCircleIcon />,
    label: gaitDecisionLabel(screening),
    caption: `This upload is screened as a non-${model} gait pattern.`
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

export function ParkinsonOverview({ screening, totalScreenings = 1 }) {
  if (!screening) return null;
  const score = probability(screening.probability);
  const degrees = Math.round(score * 360);
  const gaugeBackground = `conic-gradient(#ef4444 0deg ${degrees}deg, #16a34a ${degrees}deg 360deg)`;
  const visual = resultVisual(screening);
  const activeModel = screening.modelKey === "neuropathy" ? "Neuropathy" : "PD";

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
                  <Typography color="text.secondary">{screening.clinicalNote || "Screening support only; not a medical diagnosis."}</Typography>
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

export default function ParkinsonResults() {
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
        const profileRes = await api.get("/pd-neuropathy/clinical-profile");
        if (!alive) return;
        setProfile(profileRes.data.profile);
        setScreenings(profileRes.data.screenings || []);
        if (requestedId) {
          const screeningRes = await api.get(`/pd-neuropathy/screenings/${requestedId}`);
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
    return `/patient/results/pd${query ? `?${query}` : ""}`;
  }

  async function deleteScreening(id) {
    if (!window.confirm("Delete this saved gait-screening result?")) return;
    setDeletingId(id);
    setError("");
    try {
      const res = await api.delete(`/pd-neuropathy/screenings/${id}`);
      const remaining = res.data.screenings || [];
      setProfile(res.data.profile);
      setScreenings(remaining);
      if (id === screening?.id) {
        const nextScreening = remaining[0] || null;
        setScreening(nextScreening);
        navigate(nextScreening ? `/patient/results/pd?id=${nextScreening.id}` : "/patient/results/pd", { replace: true });
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
      <Container maxWidth="lg" className="sca-koa-page component3-page">
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
      <Container maxWidth="lg" className="sca-koa-page component3-page">
        <Alert severity="error">{error}</Alert>
      </Container>
    );
  }

  if (!screening) {
    return (
      <Container maxWidth="lg" className="sca-koa-page component3-page">
        <Card className="sca-koa-workspace-card">
          <CardContent>
            <Stack spacing={2} alignItems="center" textAlign="center" py={5}>
              <AssessmentIcon color="primary" />
              <Typography variant="h4" fontWeight={900}>No component 3 clinical profile yet</Typography>
              <Typography color="text.secondary">Run the first PD or neuropathy screening and this page will show the saved outcome profile.</Typography>
              <Button component={Link} to="/patient/detection/pd" variant="contained" startIcon={<ReplayIcon />}>
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
  const activeModel = screening.modelKey === "neuropathy" ? "Neuropathy" : "PD";
  const activeModelKey = screening.modelKey === "neuropathy" ? "neuropathy" : "pd";
  const counterpart = sameFileCounterpart(screening, screenings);
  const pdComparison = screening.modelKey === "pd" ? screening : counterpart;
  const neuropathyComparison = screening.modelKey === "neuropathy" ? screening : counterpart;

  return (
    <Container maxWidth="xl" className="sca-koa-page component3-page">
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
            <Button component={Link} to={`/patient/detection/pd?model=${activeModelKey}`} variant="outlined" startIcon={<VideoCameraBackIcon />}>
              <span className={itemClass}>Screening</span>
            </Button>
            <Button component={Link} to={getSectionLink("overview")} variant={activeSection === "overview" ? "contained" : "text"} startIcon={activeModelKey === "pd" ? <PsychologyIcon /> : <HealingIcon />}>
              <span className={itemClass}>Overview</span>
            </Button>
            <Button component={Link} to={getSectionLink("history")} variant={activeSection === "history" ? "contained" : "text"} startIcon={<AssessmentIcon />}>
              <span className={itemClass}>History</span>
            </Button>
            <Button component={Link} to={getSectionLink("pd-normal")} variant={activeSection === "pd-normal" ? "contained" : "text"} startIcon={<BarChartIcon />}>
              <span className={itemClass}>PD vs Your Gait</span>
            </Button>
            <Button component={Link} to={getSectionLink("neuropathy-normal")} variant={activeSection === "neuropathy-normal" ? "contained" : "text"} startIcon={<BarChartIcon />}>
              <span className={itemClass}>Neuropathy vs Your Gait</span>
            </Button>
            <Button component={Link} to={getSectionLink("pd-neuropathy")} variant={activeSection === "pd-neuropathy" ? "contained" : "text"} startIcon={<CompareArrowsIcon />}>
              <span className={itemClass}>PD vs Neuropathy</span>
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
            <Box className="sca-koa-hero compact component3-hero">
              <Stack spacing={1.5}>
                <Chip color={resultTone(screening)} label={`${screening.modelLabel} result`} />
                <Typography variant="h3" fontWeight={900}>{activeModel} Clinical Profile</Typography>
                <Typography color="text.secondary">Saved {activeModel} gait outcomes, reliability notes, probability scores, and longitudinal screening history.</Typography>
              </Stack>
              <Stack direction={{ xs: "column", sm: "row" }} spacing={1.2}>
                <CentralProfileFlagButton sourceType={screening.modelKey} screeningId={screening.id} />
                <Button component={Link} to={`/patient/detection/pd?model=${activeModelKey}`} variant="contained" startIcon={<ReplayIcon />}>
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
                            <Typography color="text.secondary">{screening.clinicalNote || "Screening support only; not a medical diagnosis."}</Typography>
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
              </>
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
                          <Typography color="text.secondary">{formatDate(item.createdAt)} | Direction {item.direction || "-"} | Probability {percent(item.probability)}</Typography>
                        </Stack>
                        <Stack direction="row" spacing={1}>
                          <Button component={Link} to={`/patient/results/pd?id=${item.id}&section=history`} size="small">Open</Button>
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

            {activeSection === "pd-normal" && (
              <MetricComparisonChart title="PD vs Your Gait" screening={pdComparison} modelLabel="Your uploaded gait" />
            )}

            {activeSection === "neuropathy-normal" && (
              <MetricComparisonChart title="Neuropathy vs Your Gait" screening={neuropathyComparison} modelLabel="Your uploaded gait" />
            )}

            {activeSection === "pd-neuropathy" && (
              <DiseaseComparisonChart pdScreening={pdComparison} neuropathyScreening={neuropathyComparison} />
            )}
          </Stack>
        </Grid2>
      </Grid2>
    </Container>
  );
}
