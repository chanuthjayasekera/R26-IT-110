import React, { useEffect, useMemo, useState } from "react";
import {
  Alert,
  Box,
  Button,
  Card,
  CardContent,
  Chip,
  Container,
  FormControl,
  FormHelperText,
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
import { Link, useNavigate, useSearchParams } from "react-router-dom";
import AccessibilityNewIcon from "@mui/icons-material/AccessibilityNew";
import AssessmentIcon from "@mui/icons-material/Assessment";
import BiotechIcon from "@mui/icons-material/Biotech";
import ChevronLeftIcon from "@mui/icons-material/ChevronLeft";
import ChevronRightIcon from "@mui/icons-material/ChevronRight";
import CloseIcon from "@mui/icons-material/Close";
import DeleteOutlineIcon from "@mui/icons-material/DeleteOutline";
import DeleteSweepIcon from "@mui/icons-material/DeleteSweep";
import DirectionsWalkIcon from "@mui/icons-material/DirectionsWalk";
import FactCheckIcon from "@mui/icons-material/FactCheck";
import HistoryIcon from "@mui/icons-material/History";
import UploadFileIcon from "@mui/icons-material/UploadFile";
import VideoCameraBackIcon from "@mui/icons-material/VideoCameraBack";
import { api, getApiError } from "../../common/api/http.js";
import { useAuth } from "../../common/state/AuthContext.jsx";
import "../styles/sca-koa-detection.css";

const modelCopy = {
  sca: {
    label: "SCA Detection",
    shortLabel: "SCA",
    chip: "SCA screening",
    title: "Spinocerebellar Ataxia Screening",
    subtitle: "Run the SCA V28 gait-pattern model and save the result to the SCA clinical profile.",
    action: "Run SCA detection"
  },
  koa: {
    label: "KOA Detection",
    shortLabel: "KOA",
    chip: "KOA screening",
    title: "Knee Osteoarthritis Screening",
    subtitle: "Run the KOA V14 gait-pattern model and save the result to the KOA clinical profile.",
    action: "Run KOA detection"
  }
};

function formatDate(value) {
  if (!value) return "";
  return new Intl.DateTimeFormat("en", { dateStyle: "medium", timeStyle: "short" }).format(new Date(value));
}

function fileSize(file) {
  if (!file) return "";
  return `${(file.size / (1024 * 1024)).toFixed(1)} MB`;
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

function outcomeCondition(screening) {
  if (screening?.reliabilityLevel === "Low") return "low_reliability";
  if (screening?.detected) return "detected";
  if (screening?.tendency) return "borderline";
  return "not_detected";
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

function modelFromParams(value) {
  return value === "koa" ? "koa" : "sca";
}

const retainedScaKoaUploads = new Map();

function retainedUploadForUser(userId) {
  return userId ? retainedScaKoaUploads.get(userId) || null : null;
}

function rememberRetainedUpload(userId, file, inputType) {
  if (!userId) return;
  if (file) retainedScaKoaUploads.set(userId, { file, inputType });
  else retainedScaKoaUploads.delete(userId);
}

export default function ScaKoaDetection() {
  const navigate = useNavigate();
  const { user } = useAuth();
  const userId = user?.id || "";
  const [searchParams, setSearchParams] = useSearchParams();
  const activeModel = modelFromParams(searchParams.get("model"));
  const activeSection = searchParams.get("section") || activeModel;
  const [navCollapsed, setNavCollapsed] = useState(false);
  const [screenings, setScreenings] = useState([]);
  const [file, setFile] = useState(() => retainedUploadForUser(userId)?.file || null);
  const [inputType, setInputType] = useState(() => retainedUploadForUser(userId)?.inputType || "video");
  const [direction, setDirection] = useState("Auto");
  const [fps, setFps] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [deletingId, setDeletingId] = useState("");
  const [clearingAll, setClearingAll] = useState(false);
  const [previewUrl, setPreviewUrl] = useState("");
  const [outcomeConditionFilter, setOutcomeConditionFilter] = useState("all");
  const [outcomeDateFilter, setOutcomeDateFilter] = useState("");

  useEffect(() => {
    const retained = retainedUploadForUser(userId);
    setFile(retained?.file || null);
    setInputType(retained?.inputType || "video");
  }, [userId]);

  useEffect(() => {
    let alive = true;
    api.get("/sca-koa/clinical-profile")
      .then((res) => {
        if (alive) setScreenings(res.data.screenings || []);
      })
      .catch(() => {});
    return () => {
      alive = false;
    };
  }, []);

  useEffect(() => {
    if (!file || inputType !== "video") {
      setPreviewUrl("");
      return undefined;
    }
    const nextUrl = URL.createObjectURL(file);
    setPreviewUrl(nextUrl);
    return () => URL.revokeObjectURL(nextUrl);
  }, [file, inputType]);

  const modelScreenings = useMemo(
    () => screenings.filter((item) => item.modelKey === activeModel),
    [screenings, activeModel]
  );
  const filteredModelScreenings = useMemo(
    () => filterOutcomes(modelScreenings, outcomeConditionFilter, outcomeDateFilter),
    [modelScreenings, outcomeConditionFilter, outcomeDateFilter]
  );
  const copy = modelCopy[activeModel];

  function go(section, model = activeModel) {
    const params = new URLSearchParams();
    params.set("model", model);
    if (section !== model) params.set("section", section);
    setSearchParams(params);
    setError("");
  }

  function handleFile(nextFile) {
    if (!nextFile) return;
    const nextInputType = nextFile.name.toLowerCase().endsWith(".csv") ? "csv" : "video";
    setFile(nextFile);
    setInputType(nextInputType);
    setError("");
    rememberRetainedUpload(userId, nextFile, nextInputType);
  }

  function removeFile(event) {
    event?.preventDefault();
    event?.stopPropagation();
    setFile(null);
    setPreviewUrl("");
    setInputType("video");
    setError("");
    rememberRetainedUpload(userId, null);
  }

  function handleInputTypeChange(nextInputType) {
    setInputType(nextInputType);
    if (file) rememberRetainedUpload(userId, file, nextInputType);
  }

  async function submit(event) {
    event.preventDefault();
    if (!file) {
      setError(`Please upload a gait video or CSV before running ${copy.shortLabel} detection.`);
      return;
    }

    setBusy(true);
    setError("");
    const payload = new FormData();
    payload.append("gaitFile", file);
    payload.append("modelKey", activeModel);
    payload.append("inputType", inputType);
    if (direction !== "Auto") payload.append("direction", direction);
    if (fps) payload.append("fps", fps);

    try {
      const res = await api.post("/sca-koa/screenings", payload, {
        headers: { "Content-Type": "multipart/form-data" }
      });
      navigate(`/patient/results/sca-koa?id=${res.data.screening.id}`, {
        state: {
          screening: res.data.screening,
          profile: res.data.profile
        }
      });
    } catch (err) {
      const apiError = getApiError(err);
      setError(apiError.message || `${copy.shortLabel} detection failed.`);
    } finally {
      setBusy(false);
    }
  }

  async function deleteScreening(id) {
    if (!window.confirm("Delete this saved gait-screening result?")) return;
    setDeletingId(id);
    setError("");
    try {
      const res = await api.delete(`/sca-koa/screenings/${id}`);
      setScreenings(res.data.screenings || screenings.filter((item) => item.id !== id));
    } catch (err) {
      const apiError = getApiError(err);
      setError(apiError.message || "Unable to delete screening result.");
    } finally {
      setDeletingId("");
    }
  }

  async function clearModelScreenings() {
    if (modelScreenings.length === 0) return;
    if (!window.confirm(`Clear all saved ${copy.shortLabel} screening results?`)) return;
    setClearingAll(true);
    setDeletingId("all");
    setError("");
    try {
      const res = await api.delete(`/sca-koa/screenings?model=${activeModel}`);
      setScreenings(res.data.screenings || []);
    } catch (err) {
      const apiError = getApiError(err);
      setError(apiError.message || "Unable to clear screening results.");
    } finally {
      setClearingAll(false);
      setDeletingId("");
    }
  }

  const itemClass = navCollapsed ? "sca-koa-nav-label hidden" : "sca-koa-nav-label";

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
            <Button variant={activeSection === "sca" ? "contained" : "outlined"} startIcon={<DirectionsWalkIcon />} onClick={() => go("sca", "sca")}>
              <span className={itemClass}>SCA Detection</span>
            </Button>
            <Button variant={activeSection === "koa" ? "contained" : "outlined"} startIcon={<AccessibilityNewIcon />} onClick={() => go("koa", "koa")}>
              <span className={itemClass}>KOA Detection</span>
            </Button>
            <Button variant={activeSection === "instability" ? "contained" : "text"} startIcon={<AssessmentIcon />} onClick={() => go("instability")}>
              <span className={itemClass}>Instability</span>
            </Button>
            <Button variant={activeSection === "genetics" ? "contained" : "text"} startIcon={<BiotechIcon />} onClick={() => go("genetics", "sca")}>
              <span className={itemClass}>SCA Genetics</span>
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
                <Chip icon={<VideoCameraBackIcon />} label={copy.chip} className="sca-koa-hero-chip" />
                <Typography variant="h3" fontWeight={900}>{copy.title}</Typography>
                <Typography color="text.secondary">{copy.subtitle}</Typography>
              </Stack>
            </Box>

            {error && <Alert severity="error">{error}</Alert>}

            {(activeSection === "sca" || activeSection === "koa") && (
              <>
                <Grid2 container spacing={3} component="form" onSubmit={submit}>
                  <Grid2 size={{ xs: 12, lg: 8 }}>
                    <Card className="sca-koa-workspace-card">
                      <CardContent>
                        <Stack spacing={2.5}>
                          <Stack direction="row" spacing={1.5} alignItems="center">
                            <UploadFileIcon color="primary" />
                            <Box>
                              <Typography variant="h5" fontWeight={900}>Upload gait recording</Typography>
                              <Typography color="text.secondary">Use a clear side-view walking video or training-safe landmarks CSV.</Typography>
                            </Box>
                          </Stack>
                          <Box
                            className={previewUrl ? "sca-koa-upload-zone has-video" : file ? "sca-koa-upload-zone has-file" : "sca-koa-upload-zone"}
                            onDragOver={(event) => event.preventDefault()}
                            onDrop={(event) => {
                              event.preventDefault();
                              handleFile(event.dataTransfer.files?.[0]);
                            }}
                          >
                            {file && (
                              <IconButton type="button" className="sca-koa-upload-close" color="error" onClick={removeFile} aria-label="Remove selected file">
                                <CloseIcon />
                              </IconButton>
                            )}
                            {previewUrl ? (
                              <>
                                <Stack direction="row" spacing={1} alignItems="center" className="sca-koa-preview-title">
                                  <VideoCameraBackIcon color="primary" />
                                  <Box flex={1}>
                                    <Typography fontWeight={900}>Video preview</Typography>
                                    <Typography color="text.secondary">{file.name} - {fileSize(file)}</Typography>
                                  </Box>
                                  <Button component="label" variant="outlined" size="small" startIcon={<UploadFileIcon />}>
                                    Replace
                                    <input hidden type="file" accept=".mp4,.mov,.avi,.mkv,.csv,video/mp4,video/quicktime,text/csv" onChange={(event) => handleFile(event.target.files?.[0])} />
                                  </Button>
                                </Stack>
                                <video className="sca-koa-inline-video" controls src={previewUrl} />
                              </>
                            ) : (
                              <>
                                <UploadFileIcon />
                                <Typography fontWeight={900}>{file ? file.name : "Drop video here or choose a file"}</Typography>
                                <Typography color="text.secondary">{file ? `${inputType.toUpperCase()} - ${fileSize(file)}` : "MP4, MOV, AVI, MKV, or training-safe CSV"}</Typography>
                                <Button component="label" variant="contained" startIcon={<UploadFileIcon />}>
                                  Choose file
                                  <input hidden type="file" accept=".mp4,.mov,.avi,.mkv,.csv,video/mp4,video/quicktime,text/csv" onChange={(event) => handleFile(event.target.files?.[0])} />
                                </Button>
                              </>
                            )}
                          </Box>
                        </Stack>
                      </CardContent>
                    </Card>
                  </Grid2>

                  <Grid2 size={{ xs: 12, lg: 4 }}>
                    <Card className="sca-koa-workspace-card">
                      <CardContent>
                        <Stack spacing={2.5}>
                          <Stack direction="row" spacing={1.5} alignItems="center">
                            <FactCheckIcon color="primary" />
                            <Box>
                              <Typography variant="h5" fontWeight={900}>{copy.shortLabel} model settings</Typography>
                              <Typography color="text.secondary">Auto direction and default FPS match the model dashboard defaults.</Typography>
                            </Box>
                          </Stack>
                          <FormControl fullWidth>
                            <InputLabel>Input type</InputLabel>
                            <Select label="Input type" value={inputType} onChange={(event) => handleInputTypeChange(event.target.value)}>
                              <MenuItem value="video">Video</MenuItem>
                              <MenuItem value="csv">Training-safe CSV</MenuItem>
                            </Select>
                          </FormControl>
                          <FormControl fullWidth>
                            <InputLabel>Walking direction</InputLabel>
                            <Select label="Walking direction" value={direction} onChange={(event) => setDirection(event.target.value)}>
                              <MenuItem value="Auto">Auto detect</MenuItem>
                              <MenuItem value="L2R">Left to right</MenuItem>
                              <MenuItem value="R2L">Right to left</MenuItem>
                            </Select>
                            <FormHelperText>Leave auto unless the recording direction is known.</FormHelperText>
                          </FormControl>
                          <TextField label="Source FPS" type="number" value={fps} onChange={(event) => setFps(event.target.value)} helperText="Optional. Leave empty for model default." />
                          {busy && (
                            <Box>
                              <LinearProgress />
                              <Typography className="sca-koa-processing-text">Extracting landmarks, running {copy.shortLabel}, and saving the outcome...</Typography>
                            </Box>
                          )}
                          <Button type="submit" size="large" variant="contained" startIcon={<FactCheckIcon />} disabled={busy}>
                            {busy ? "Running detection..." : copy.action}
                          </Button>
                        </Stack>
                      </CardContent>
                    </Card>
                  </Grid2>
                </Grid2>

                <Card className="sca-koa-workspace-card">
                  <CardContent>
                    <Stack spacing={2}>
                      <Stack direction={{ xs: "column", md: "row" }} spacing={1.5} justifyContent="space-between">
                        <Stack direction="row" spacing={1.5} alignItems="center">
                          <HistoryIcon color="primary" />
                          <Typography variant="h5" fontWeight={900}>Latest saved {copy.shortLabel} outcomes</Typography>
                        </Stack>
                        <Stack className="sca-koa-history-toolbar" direction={{ xs: "column", lg: "row" }} spacing={1.2}>
                          <Stack className="sca-koa-history-filters" direction={{ xs: "column", sm: "row" }} spacing={1.2}>
                            <FormControl size="small" className="sca-koa-history-filter">
                              <InputLabel>Condition</InputLabel>
                              <Select
                                label="Condition"
                                value={outcomeConditionFilter}
                                onChange={(event) => setOutcomeConditionFilter(event.target.value)}
                              >
                                <MenuItem value="all">All</MenuItem>
                                <MenuItem value="detected">{copy.shortLabel} gait detected</MenuItem>
                                <MenuItem value="not_detected">Non-{copy.shortLabel} gait</MenuItem>
                                <MenuItem value="borderline">Borderline</MenuItem>
                                <MenuItem value="low_reliability">Low reliability</MenuItem>
                              </Select>
                            </FormControl>
                            <TextField
                              className="sca-koa-history-filter"
                              size="small"
                              label="Date"
                              type="date"
                              value={outcomeDateFilter}
                              onChange={(event) => setOutcomeDateFilter(event.target.value)}
                              InputLabelProps={{ shrink: true }}
                            />
                          </Stack>
                          <Button
                            className="sca-koa-danger-button sca-koa-clear-all-button"
                            color="error"
                            variant="outlined"
                            startIcon={<DeleteSweepIcon />}
                            disabled={clearingAll || modelScreenings.length === 0}
                            onClick={clearModelScreenings}
                          >
                            Clear all
                          </Button>
                        </Stack>
                      </Stack>
                      {modelScreenings.length === 0 ? (
                        <Typography color="text.secondary">No {copy.shortLabel} outcomes have been saved yet.</Typography>
                      ) : filteredModelScreenings.length === 0 ? (
                        <Typography color="text.secondary">No saved {copy.shortLabel} outcomes match these filters.</Typography>
                      ) : (
                        <Grid2 container spacing={2}>
                          {filteredModelScreenings.slice(0, 8).map((screening) => (
                            <Grid2 key={screening.id} size={{ xs: 12, md: 6, xl: 3 }}>
                              <Box className="sca-koa-history-item">
                                <Stack direction="row" alignItems="flex-start" justifyContent="space-between">
                                  <Chip size="small" color={resultTone(screening)} label={gaitDecisionLabel(screening)} />
                                  <IconButton size="small" disabled={deletingId === "all" || deletingId === screening.id} onClick={() => deleteScreening(screening.id)}>
                                    <DeleteOutlineIcon fontSize="small" />
                                  </IconButton>
                                </Stack>
                                <Typography fontWeight={900}>{gaitDecisionLabel(screening)}</Typography>
                                <Typography color="text.secondary">{formatDate(screening.createdAt)}</Typography>
                                <Typography className="sca-koa-metric-helper">Mean probability {percent(screening.probability)}</Typography>
                                <Button component={Link} to={`/patient/results/sca-koa?id=${screening.id}`} size="small" startIcon={<AssessmentIcon />}>
                                  Open profile
                                </Button>
                              </Box>
                            </Grid2>
                          ))}
                        </Grid2>
                      )}
                    </Stack>
                  </CardContent>
                </Card>
              </>
            )}

            {activeSection === "instability" && (
              <Card className="sca-koa-workspace-card">
                <CardContent>
                  <Stack spacing={2}>
                    <Typography variant="h5" fontWeight={900}>Instability Visualization</Typography>
                    <Typography color="text.secondary">This section is reserved for the joint-level green, yellow, and red instability map.</Typography>
                    <Box className="instability-strip">
                      <Box><span className="stable-dot"></span><Typography fontWeight={800}>Stable movement</Typography></Box>
                      <Box><span className="mild-dot"></span><Typography fontWeight={800}>Mild instability</Typography></Box>
                      <Box><span className="severe-dot"></span><Typography fontWeight={800}>Severe instability</Typography></Box>
                    </Box>
                  </Stack>
                </CardContent>
              </Card>
            )}

            {activeSection === "genetics" && (
              <Card className="sca-koa-workspace-card">
                <CardContent>
                  <Stack spacing={2}>
                    <Typography variant="h5" fontWeight={900}>SCA Genetic Chart</Typography>
                    <Typography color="text.secondary">The family-history and hereditary SCA chart will be connected in the next step.</Typography>
                  </Stack>
                </CardContent>
              </Card>
            )}
          </Stack>
        </Grid2>
      </Grid2>
    </Container>
  );
}
