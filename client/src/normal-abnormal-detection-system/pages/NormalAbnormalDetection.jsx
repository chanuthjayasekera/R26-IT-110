import React, { useCallback, useEffect, useMemo, useState } from "react";
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
import { Link, useLocation, useNavigate } from "react-router-dom";
import AssessmentIcon from "@mui/icons-material/Assessment";
import ChevronLeftIcon from "@mui/icons-material/ChevronLeft";
import ChevronRightIcon from "@mui/icons-material/ChevronRight";
import CloseIcon from "@mui/icons-material/Close";
import DeleteOutlineIcon from "@mui/icons-material/DeleteOutline";
import DeleteSweepIcon from "@mui/icons-material/DeleteSweep";
import DirectionsWalkIcon from "@mui/icons-material/DirectionsWalk";
import FactCheckIcon from "@mui/icons-material/FactCheck";
import HistoryIcon from "@mui/icons-material/History";
import PlayCircleIcon from "@mui/icons-material/PlayCircle";
import UploadFileIcon from "@mui/icons-material/UploadFile";
import VideoCameraBackIcon from "@mui/icons-material/VideoCameraBack";
import { api, getApiError } from "../../common/api/http.js";
import { useAuth } from "../../common/state/AuthContext.jsx";
import "../styles/normal-abnormal-detection.css";

function formatDate(value) {
  if (!value) return "";
  return new Intl.DateTimeFormat("en", {
    dateStyle: "medium",
    timeStyle: "short"
  }).format(new Date(value));
}

function resultTone(screening) {
  const text = String(screening?.finalResult || "").toLowerCase();
  if (text.includes("abnormal")) return "error";
  if (text.includes("normal")) return "success";
  return "warning";
}

function fileSize(file) {
  if (!file) return "";
  return `${(file.size / (1024 * 1024)).toFixed(1)} MB`;
}

function apiAssetUrl(path) {
  if (!path) return "";
  if (/^https?:\/\//i.test(path)) return path;
  const base = (api.defaults.baseURL || "").replace(/\/api\/?$/, "");
  return `${base}/api${path}`;
}

function outcomeCondition(screening) {
  const text = `${screening?.finalResult || ""} ${screening?.modelSuggestedResult || ""}`.toLowerCase();
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

const dismissedPreviewKey = "normalAbnormalManuallyDismissedPreviewIdV3";
const latestVideoKey = "normalAbnormalLatestVideoScreening";
const retainedUploads = new Map();

function scopedSessionKey(key, userId) {
  return userId ? `${key}:${userId}` : key;
}

function readLatestVideoScreening(userId) {
  try {
    const value = window.sessionStorage.getItem(scopedSessionKey(latestVideoKey, userId));
    return value ? JSON.parse(value) : null;
  } catch {
    return null;
  }
}

function rememberLatestVideoScreening(userId, screening) {
  if (!screening?.hasVideoPreview || !screening?.videoUrl) return;
  window.sessionStorage.setItem(scopedSessionKey(latestVideoKey, userId), JSON.stringify(screening));
}

function forgetLatestVideoScreening(userId) {
  window.sessionStorage.removeItem(scopedSessionKey(latestVideoKey, userId));
}

function retainedUploadForUser(userId) {
  return userId ? retainedUploads.get(userId) || null : null;
}

function rememberRetainedUpload(userId, file, inputType) {
  if (!userId) return;
  if (file) retainedUploads.set(userId, { file, inputType });
  else retainedUploads.delete(userId);
}

function markDetectionHistoryEntryWithVideo(screening) {
  if (!screening?.hasVideoPreview || !screening?.videoUrl) return;
  const currentState = window.history.state || {};
  const { startFresh, ...currentUserState } = currentState.usr || {};
  window.history.replaceState(
    {
      ...currentState,
      usr: {
        ...currentUserState,
        showLatestVideo: true,
        latestVideoScreening: screening
      }
    },
    "",
    window.location.href
  );
}

export default function NormalAbnormalDetection() {
  const navigate = useNavigate();
  const location = useLocation();
  const { user } = useAuth();
  const userId = user?.id || "";
  const [screenings, setScreenings] = useState([]);
  const [file, setFile] = useState(() => retainedUploadForUser(userId)?.file || null);
  const [inputType, setInputType] = useState(() => retainedUploadForUser(userId)?.inputType || "video");
  const [direction, setDirection] = useState("Auto");
  const [fps, setFps] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [deletingId, setDeletingId] = useState("");
  const [clearingAll, setClearingAll] = useState(false);
  const [dismissedPreviewId, setDismissedPreviewId] = useState(() => {
    return window.sessionStorage.getItem(scopedSessionKey(dismissedPreviewKey, userId)) || "";
  });
  const [latestSubmittedVideoScreening, setLatestSubmittedVideoScreening] = useState(null);
  const [navCollapsed, setNavCollapsed] = useState(false);
  const [outcomeConditionFilter, setOutcomeConditionFilter] = useState("all");
  const [outcomeDateFilter, setOutcomeDateFilter] = useState("");
  const [objectPreviewUrl, setObjectPreviewUrl] = useState("");

  const loadClinicalProfile = useCallback(async () => {
    const res = await api.get("/normal-abnormal/clinical-profile");
    return res.data.screenings || [];
  }, []);

  useEffect(() => {
    const retained = retainedUploadForUser(userId);
    setFile(retained?.file || null);
    setInputType(retained?.inputType || "video");
    setDismissedPreviewId(window.sessionStorage.getItem(scopedSessionKey(dismissedPreviewKey, userId)) || "");
    setLatestSubmittedVideoScreening(readLatestVideoScreening(userId));
  }, [userId]);

  useEffect(() => {
    let alive = true;

    loadClinicalProfile()
      .then((nextScreenings) => {
        if (alive) setScreenings(nextScreenings);
      })
      .catch(() => {});

    return () => {
      alive = false;
    };
  }, [loadClinicalProfile]);

  useEffect(() => {
    const refresh = () => {
      loadClinicalProfile()
        .then((nextScreenings) => setScreenings(nextScreenings))
        .catch(() => {});
    };
    window.addEventListener("focus", refresh);
    window.addEventListener("pageshow", refresh);
    return () => {
      window.removeEventListener("focus", refresh);
      window.removeEventListener("pageshow", refresh);
    };
  }, [loadClinicalProfile]);

  useEffect(() => {
    if (!file || inputType !== "video") {
      setObjectPreviewUrl("");
      return undefined;
    }

    const nextPreviewUrl = URL.createObjectURL(file);
    setObjectPreviewUrl(nextPreviewUrl);

    return () => {
      URL.revokeObjectURL(nextPreviewUrl);
    };
  }, [file, inputType]);

  useEffect(() => {
    const stateVideo = location.state?.latestVideoScreening;
    if (!stateVideo?.hasVideoPreview || !stateVideo?.videoUrl) return;
    setLatestSubmittedVideoScreening(stateVideo);
    rememberLatestVideoScreening(userId, stateVideo);
    setDismissedPreviewId("");
    window.sessionStorage.removeItem(scopedSessionKey(dismissedPreviewKey, userId));
  }, [location.state?.latestVideoScreening, userId]);

  const latestVideoScreening = latestSubmittedVideoScreening;

  useEffect(() => {
    if (!location.state?.showLatestVideo) return;
    navigate(location.pathname, { replace: true, state: {} });
  }, [location.pathname, location.state?.showLatestVideo, navigate]);

  useEffect(() => {
    if (!location.state?.startFresh) return;
    navigate(location.pathname, { replace: true, state: {} });
  }, [location.pathname, location.state?.startFresh, navigate]);

  const visibleLatestVideoScreening = null;
  const previewUrl = file ? objectPreviewUrl : "";

  const filteredScreenings = useMemo(() => {
    return filterOutcomes(screenings, outcomeConditionFilter, outcomeDateFilter);
  }, [screenings, outcomeConditionFilter, outcomeDateFilter]);

  function handleFile(nextFile) {
    if (!nextFile) return;
    const nextInputType = nextFile.name.toLowerCase().endsWith(".csv") ? "csv" : "video";
    setFile(nextFile);
    setError("");
    setDismissedPreviewId("");
    window.sessionStorage.removeItem(scopedSessionKey(dismissedPreviewKey, userId));
    setInputType(nextInputType);
    rememberRetainedUpload(userId, nextFile, nextInputType);
  }

  function handleInputTypeChange(nextInputType) {
    setInputType(nextInputType);
    if (file) rememberRetainedUpload(userId, file, nextInputType);
  }

  function removeFile() {
    rememberRetainedUpload(userId, null);
    setFile(null);
    setError("");
    setInputType("video");
    if (latestVideoScreening?.id) {
      setDismissedPreviewId(latestVideoScreening.id);
      window.sessionStorage.setItem(scopedSessionKey(dismissedPreviewKey, userId), latestVideoScreening.id);
      setLatestSubmittedVideoScreening(null);
      forgetLatestVideoScreening(userId);
    }
  }

  function dismissLatestVideo() {
    if (!latestVideoScreening?.id) return;
    setDismissedPreviewId(latestVideoScreening.id);
    window.sessionStorage.setItem(scopedSessionKey(dismissedPreviewKey, userId), latestVideoScreening.id);
    setLatestSubmittedVideoScreening(null);
    forgetLatestVideoScreening(userId);
  }

  async function submit(event) {
    event.preventDefault();
    if (!file) {
      setError("Please upload a gait video or CSV before running detection.");
      return;
    }

    setBusy(true);
    setError("");

    const payload = new FormData();
    payload.append("gaitFile", file);
    payload.append("inputType", inputType);
    if (direction !== "Auto") payload.append("direction", direction);
    if (fps) payload.append("fps", fps);

    try {
      const res = await api.post("/normal-abnormal/screenings", payload, {
        headers: { "Content-Type": "multipart/form-data" }
      });
      setDismissedPreviewId("");
      window.sessionStorage.removeItem(scopedSessionKey(dismissedPreviewKey, userId));
      if (res.data.screening?.hasVideoPreview && res.data.screening?.videoUrl) {
        setLatestSubmittedVideoScreening(res.data.screening);
        rememberLatestVideoScreening(userId, res.data.screening);
        markDetectionHistoryEntryWithVideo(res.data.screening);
      } else {
        setLatestSubmittedVideoScreening(null);
        forgetLatestVideoScreening(userId);
      }
      navigate(`/patient/results/normal-abnormal?id=${res.data.screening.id}`, {
        state: {
          screening: res.data.screening,
          profile: res.data.profile
        }
      });
    } catch (err) {
      const apiError = getApiError(err);
      setError(apiError.message || "Normal vs abnormal detection failed.");
    } finally {
      setBusy(false);
    }
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
      setScreenings(res.data.screenings || screenings.filter((item) => item.id !== id));
      if (latestSubmittedVideoScreening?.id === id) {
        setLatestSubmittedVideoScreening(null);
        forgetLatestVideoScreening(userId);
      }
    } catch (err) {
      const apiError = getApiError(err);
      setError(apiError.message || "Unable to delete screening result.");
    } finally {
      setDeletingId("");
    }
  }

  async function clearAllScreenings() {
    if (screenings.length === 0) return;
    if (!window.confirm("Clear all saved normal vs abnormal screening results?")) return;
    setClearingAll(true);
    setDeletingId("all");
    setError("");

    try {
      let res = null;
      try {
        res = await api.delete("/normal-abnormal/screenings");
      } catch (deleteError) {
        if (deleteError?.response?.status !== 404 && deleteError?.response?.status !== 405) throw deleteError;
        try {
          res = await api.post("/normal-abnormal/screenings/clear");
        } catch (clearError) {
          if (clearError?.response?.status !== 404 && clearError?.response?.status !== 405) throw clearError;
        }
      }

      if (!res) {
        for (const item of screenings) {
          try {
            res = await api.delete(`/normal-abnormal/screenings/${item.id}`);
          } catch (deleteError) {
            if (deleteError?.response?.status !== 404 && deleteError?.response?.status !== 405) throw deleteError;
            res = await api.post(`/normal-abnormal/screenings/${item.id}/delete`);
          }
        }
      }

      setScreenings(res.data.screenings || []);
      setDismissedPreviewId("");
      setLatestSubmittedVideoScreening(null);
      rememberRetainedUpload(userId, null);
      setFile(null);
      window.sessionStorage.removeItem(scopedSessionKey(dismissedPreviewKey, userId));
      forgetLatestVideoScreening(userId);
    } catch (err) {
      const apiError = getApiError(err);
      setError(apiError.message || "Unable to clear screening results.");
    } finally {
      setClearingAll(false);
      setDeletingId("");
    }
  }


  return (
    <Container maxWidth="xl" className="normal-detection-page">
      <Grid2 container spacing={3}>
        <Grid2 size={{ xs: 12, md: navCollapsed ? 1.1 : 3, lg: navCollapsed ? 0.9 : 2.4 }}>
          <Box className={navCollapsed ? "normal-side-nav collapsed" : "normal-side-nav"}>
            <Button
              className="normal-nav-toggle"
              variant="outlined"
              onClick={() => setNavCollapsed((old) => !old)}
              startIcon={navCollapsed ? <ChevronRightIcon /> : <ChevronLeftIcon />}
            >
              <span className={navCollapsed ? "normal-nav-label hidden" : "normal-nav-label"}>{navCollapsed ? "" : "Collapse"}</span>
            </Button>
            <Button fullWidth variant="contained" startIcon={<VideoCameraBackIcon />}>
              <span className={navCollapsed ? "normal-nav-label hidden" : "normal-nav-label"}>Detection</span>
            </Button>
            <Button fullWidth component={Link} to="/patient/results/normal-abnormal" variant="outlined" startIcon={<AssessmentIcon />}>
              <span className={navCollapsed ? "normal-nav-label hidden" : "normal-nav-label"}>Profile</span>
            </Button>
            <Button fullWidth component={Link} to="/patient" variant="text">
              <span className={navCollapsed ? "normal-nav-label hidden" : "normal-nav-label"}>Dashboard</span>
            </Button>
          </Box>
        </Grid2>

        <Grid2 size={{ xs: 12, md: navCollapsed ? 10.9 : 9, lg: navCollapsed ? 11.1 : 9.6 }}>
          <Stack spacing={3}>
            <Box className="normal-screening-hero compact">
              <Stack spacing={1.5}>
                <Chip icon={<DirectionsWalkIcon />} label="Component 1 model" className="normal-hero-chip" />
                <Typography variant="h3" fontWeight={900}>Normal vs Abnormal Detection</Typography>
                <Typography color="text.secondary">
                  Upload a gait video, preview it, run the model, and save the outcome to your clinical profile automatically.
                </Typography>
              </Stack>
            </Box>

            {error && <Alert severity="error">{error}</Alert>}

            <Grid2 container spacing={3} component="form" onSubmit={submit}>
              <Grid2 size={{ xs: 12, lg: 8 }}>
                <Card className="normal-workspace-card">
                  <CardContent>
                    <Stack spacing={2.5}>
                      <Stack direction="row" spacing={1.5} alignItems="center">
                        <UploadFileIcon color="primary" />
                        <Box>
                          <Typography variant="h5" fontWeight={900}>Upload gait recording</Typography>
                          <Typography color="text.secondary">Use a clear side-view walking video for best landmark extraction.</Typography>
                        </Box>
                      </Stack>

                      <Box
                        className={previewUrl ? "normal-upload-zone large has-video" : file ? "normal-upload-zone large has-file" : "normal-upload-zone large"}
                        onDragOver={(event) => event.preventDefault()}
                        onDrop={(event) => {
                          event.preventDefault();
                          handleFile(event.dataTransfer.files?.[0]);
                        }}
                      >
                        {previewUrl && (
                          <IconButton
                            className="normal-upload-close"
                            color="error"
                            onClick={file ? removeFile : dismissLatestVideo}
                            aria-label={file ? "Remove selected file" : "Hide latest video preview"}
                          >
                            <CloseIcon />
                          </IconButton>
                        )}
                        {previewUrl ? (
                          <>
                            <Stack direction="row" spacing={1} alignItems="center" className="normal-preview-title inline">
                              <PlayCircleIcon color="primary" />
                              <Box flex={1}>
                                <Typography fontWeight={900}>{file ? "Video preview" : "Latest uploaded video"}</Typography>
                                <Typography color="text.secondary">
                                  {file ? `${file.name} - ${fileSize(file)}` : `${visibleLatestVideoScreening?.fileName || "Saved gait video"} - ${formatDate(visibleLatestVideoScreening?.createdAt)}`}
                                </Typography>
                              </Box>
                              <Button component="label" variant="outlined" size="small" startIcon={<UploadFileIcon />}>
                                Replace
                                <input
                                  hidden
                                  type="file"
                                  accept=".mp4,.mov,.avi,.mkv,.csv,video/mp4,video/quicktime,text/csv"
                                  onChange={(event) => handleFile(event.target.files?.[0])}
                                />
                              </Button>
                            </Stack>
                            <video className="normal-inline-video" controls src={previewUrl} />
                          </>
                        ) : (
                          <>
                            <UploadFileIcon />
                            <Typography fontWeight={900}>
                              {file ? file.name : "Drop video here or choose a file"}
                            </Typography>
                            <Typography color="text.secondary">
                              {file ? `${inputType.toUpperCase()} - ${fileSize(file)}` : "MP4, MOV, AVI, MKV, or training-safe CSV"}
                            </Typography>
                            <Stack direction="row" spacing={1.2} alignItems="center" justifyContent="center">
                              <Button component="label" variant="contained" startIcon={<UploadFileIcon />}>
                                Choose file
                                <input
                                  hidden
                                  type="file"
                                  accept=".mp4,.mov,.avi,.mkv,.csv,video/mp4,video/quicktime,text/csv"
                                  onChange={(event) => handleFile(event.target.files?.[0])}
                                />
                              </Button>
                            </Stack>
                          </>
                        )}
                        </Box>
                    </Stack>
                  </CardContent>
                </Card>
              </Grid2>

              <Grid2 size={{ xs: 12, lg: 4 }}>
                <Card className="normal-workspace-card">
                  <CardContent>
                    <Stack spacing={2.5}>
                      <Stack direction="row" spacing={1.5} alignItems="center">
                        <FactCheckIcon color="primary" />
                        <Box>
                          <Typography variant="h5" fontWeight={900}>Model settings</Typography>
                          <Typography color="text.secondary">Auto mode is recommended for normal video uploads.</Typography>
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
                        <FormHelperText>Leave as auto unless you already know the direction.</FormHelperText>
                      </FormControl>

                      <TextField
                        label="Source FPS"
                        type="number"
                        value={fps}
                        onChange={(event) => setFps(event.target.value)}
                        helperText="Optional. Leave empty for model default."
                      />

                      {busy && (
                        <Box>
                          <LinearProgress />
                          <Typography className="normal-processing-text">Extracting landmarks, running Component 1, and saving the outcome...</Typography>
                        </Box>
                      )}

                      <Button type="submit" size="large" variant="contained" startIcon={<FactCheckIcon />} disabled={busy}>
                        {busy ? "Running detection..." : "Run normal/abnormal detection"}
                      </Button>
                    </Stack>
                  </CardContent>
                </Card>
              </Grid2>
            </Grid2>

            <Card className="normal-workspace-card">
              <CardContent>
                  <Stack spacing={2}>
                  <Stack direction={{ xs: "column", md: "row" }} spacing={1.5} alignItems={{ xs: "stretch", md: "center" }} justifyContent="space-between">
                    <Stack direction="row" spacing={1.5} alignItems="center">
                      <HistoryIcon color="primary" />
                      <Typography variant="h5" fontWeight={900}>Latest saved outcomes</Typography>
                    </Stack>
                    <Stack className="normal-history-toolbar" direction={{ xs: "column", lg: "row" }} spacing={1.2}>
                      <Stack className="normal-history-filters" direction={{ xs: "column", sm: "row" }} spacing={1.2}>
                        <FormControl size="small" className="normal-history-filter">
                          <InputLabel>Condition</InputLabel>
                          <Select
                            label="Condition"
                            value={outcomeConditionFilter}
                            onChange={(event) => setOutcomeConditionFilter(event.target.value)}
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
                          value={outcomeDateFilter}
                          onChange={(event) => setOutcomeDateFilter(event.target.value)}
                          InputLabelProps={{ shrink: true }}
                        />
                      </Stack>
                      <Button
                        className="normal-danger-button normal-clear-all-button"
                        color="error"
                        variant="outlined"
                        startIcon={<DeleteSweepIcon />}
                        disabled={clearingAll || screenings.length === 0}
                        onClick={clearAllScreenings}
                      >
                        Clear all
                      </Button>
                    </Stack>
                  </Stack>

                  {screenings.length === 0 ? (
                    <Typography color="text.secondary">No model outcomes have been saved yet.</Typography>
                  ) : filteredScreenings.length === 0 ? (
                    <Typography color="text.secondary">No saved outcomes match these filters.</Typography>
                  ) : (
                    <Grid2 container spacing={2}>
                      {filteredScreenings.slice(0, 8).map((screening) => (
                        <Grid2 key={screening.id} size={{ xs: 12, md: 6, xl: 3 }}>
                          <Box className="normal-history-item">
                            <Stack className="normal-history-card-head" direction="row" alignItems="flex-start" justifyContent="space-between">
                              <Chip size="small" color={resultTone(screening)} label={screening.finalResult || "Inconclusive"} />
                              <IconButton
                                className="normal-delete-icon-button"
                                size="small"
                                aria-label="Delete result"
                                title="Delete result"
                                disabled={deletingId === "all" || deletingId === screening.id}
                                onClick={() => deleteScreening(screening.id)}
                              >
                                <DeleteOutlineIcon fontSize="small" />
                              </IconButton>
                            </Stack>
                            <Typography fontWeight={900}>{screening.screeningSeverity || "Screening outcome"}</Typography>
                            <Typography color="text.secondary">{formatDate(screening.createdAt)}</Typography>
                            <Button
                              className="normal-soft-button"
                              component={Link}
                              to={`/patient/results/normal-abnormal?id=${screening.id}`}
                              size="small"
                              startIcon={<AssessmentIcon />}
                            >
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
          </Stack>
        </Grid2>
      </Grid2>
    </Container>
  );
}
//*normal abnoraml detection  
