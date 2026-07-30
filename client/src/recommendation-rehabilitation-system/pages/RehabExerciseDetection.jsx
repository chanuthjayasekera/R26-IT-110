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
import FitnessCenterIcon from "@mui/icons-material/FitnessCenter";
import UploadFileIcon from "@mui/icons-material/UploadFile";
import CheckCircleIcon from "@mui/icons-material/CheckCircle";
import ChevronLeftIcon from "@mui/icons-material/ChevronLeft";
import ChevronRightIcon from "@mui/icons-material/ChevronRight";
import CloseIcon from "@mui/icons-material/Close";
import DeleteOutlineIcon from "@mui/icons-material/DeleteOutline";
import DeleteSweepIcon from "@mui/icons-material/DeleteSweep";
import HistoryIcon from "@mui/icons-material/History";
import ReportIcon from "@mui/icons-material/Report";
import RepeatIcon from "@mui/icons-material/Repeat";
import TipsAndUpdatesIcon from "@mui/icons-material/TipsAndUpdates";
import VideoCameraBackIcon from "@mui/icons-material/VideoCameraBack";
import WarningAmberIcon from "@mui/icons-material/WarningAmber";
import { api, getApiError } from "../../common/api/http.js";
import { useAuth } from "../../common/state/AuthContext.jsx";
import "../styles/rehab-exercise-detection.css";

const exercises = {
  gesture3: {
    label: "Seated left-arm forward raise",
    shortLabel: "Left forward raise",
    icon: <FitnessCenterIcon />,
    action: "Analyze left forward raise"
  },
  gesture5: {
    label: "Seated left-arm lateral raise",
    shortLabel: "Left lateral raise",
    icon: <AccessibilityNewIcon />,
    action: "Analyze left lateral raise"
  },
  gesture2: {
    label: "Seated right-arm forward raise",
    shortLabel: "Right forward raise",
    icon: <FitnessCenterIcon />,
    action: "Analyze right forward raise"
  }
};

function exerciseFromParams(value) {
  return exercises[value] ? value : "gesture3";
}

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

function outcomeCondition(screening) {
  if (screening?.reliabilityLevel === "Low") return "low_reliability";
  return screening?.isCorrect ? "correct" : "incorrect";
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

function resultTone(screening) {
  if (screening?.reliabilityLevel === "Low") return "warning";
  return screening?.isCorrect ? "success" : "error";
}

function resultLabel(screening) {
  if (screening?.reliabilityLevel === "Low") return "Review needed";
  return screening?.isCorrect ? "Correct posture" : "Incorrect posture";
}

const retainedExerciseUploads = new Map();

function retainedUploadForUser(userId) {
  return userId ? retainedExerciseUploads.get(userId) || null : null;
}

function rememberUpload(userId, file) {
  if (!userId) return;
  if (file) retainedExerciseUploads.set(userId, file);
  else retainedExerciseUploads.delete(userId);
}

export default function RehabExerciseDetection() {
  const navigate = useNavigate();
  const { user } = useAuth();
  const userId = user?.id || "";
  const [searchParams, setSearchParams] = useSearchParams();
  const activeExercise = exerciseFromParams(searchParams.get("exercise"));
  const [screenings, setScreenings] = useState([]);
  const [file, setFile] = useState(() => retainedUploadForUser(userId));
  const [previewUrl, setPreviewUrl] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [deletingId, setDeletingId] = useState("");
  const [clearingAll, setClearingAll] = useState(false);
  const [conditionFilter, setConditionFilter] = useState("all");
  const [dateFilter, setDateFilter] = useState("");
  const [navCollapsed, setNavCollapsed] = useState(false);
  const cards = [
    ["Exercise upload", "Upload rehabilitation exercise videos for pose correctness detection."],
    ["Correctness check", "Prepared for trained rehab model feedback on exercise form."],
    ["Repetition quality", "Track repetition completion quality and consistency."],
    ["Recommendations", "Retrieve clinician-uploaded rehab plans and safety guidance."]
  ];

  useEffect(() => {
    setFile(retainedUploadForUser(userId));
  }, [userId]);

  useEffect(() => {
    let alive = true;
    api.get("/exercise-detection/clinical-profile")
      .then((res) => {
        if (alive) setScreenings(res.data.screenings || []);
      })
      .catch(() => {});
    return () => {
      alive = false;
    };
  }, []);

  useEffect(() => {
    if (!file) {
      setPreviewUrl("");
      return undefined;
    }
    const nextUrl = URL.createObjectURL(file);
    setPreviewUrl(nextUrl);
    return () => URL.revokeObjectURL(nextUrl);
  }, [file]);

  const exerciseScreenings = useMemo(
    () => screenings.filter((item) => item.exerciseKey === activeExercise),
    [screenings, activeExercise]
  );
  const filteredScreenings = useMemo(
    () => filterOutcomes(exerciseScreenings, conditionFilter, dateFilter),
    [exerciseScreenings, conditionFilter, dateFilter]
  );
  const copy = exercises[activeExercise];

  function chooseExercise(exerciseKey) {
    const params = new URLSearchParams();
    params.set("exercise", exerciseKey);
    setSearchParams(params);
    setError("");
  }

  function handleFile(nextFile) {
    if (!nextFile) return;
    setFile(nextFile);
    setError("");
    rememberUpload(userId, nextFile);
  }

  function removeFile(event) {
    event?.preventDefault();
    event?.stopPropagation();
    setFile(null);
    setPreviewUrl("");
    setError("");
    rememberUpload(userId, null);
  }

  async function submit(event) {
    event.preventDefault();
    if (!file) {
      setError("Please upload an exercise video before running the analysis.");
      return;
    }

    setBusy(true);
    setError("");
    const payload = new FormData();
    payload.append("exerciseFile", file);
    payload.append("exerciseKey", activeExercise);

    try {
      const res = await api.post("/exercise-detection/screenings", payload, {
        headers: { "Content-Type": "multipart/form-data" }
      });
      navigate(`/patient/results/rehab-exercise?id=${res.data.screening.id}`, {
        state: {
          screening: res.data.screening,
          profile: res.data.profile
        }
      });
    } catch (err) {
      const apiError = getApiError(err);
      setError(apiError.message || "Exercise analysis failed.");
    } finally {
      setBusy(false);
    }
  }

  async function deleteScreening(id) {
    if (!window.confirm("Delete this saved exercise result?")) return;
    setDeletingId(id);
    setError("");
    try {
      const res = await api.delete(`/exercise-detection/screenings/${id}`);
      setScreenings(res.data.screenings || screenings.filter((item) => item.id !== id));
    } catch (err) {
      const apiError = getApiError(err);
      setError(apiError.message || "Unable to delete exercise result.");
    } finally {
      setDeletingId("");
    }
  }

  async function clearExerciseScreenings() {
    if (exerciseScreenings.length === 0) return;
    if (!window.confirm(`Clear all saved ${copy.shortLabel} results?`)) return;
    setClearingAll(true);
    setDeletingId("all");
    setError("");
    try {
      const res = await api.delete(`/exercise-detection/screenings?exercise=${activeExercise}`);
      setScreenings(res.data.screenings || []);
    } catch (err) {
      const apiError = getApiError(err);
      setError(apiError.message || "Unable to clear exercise results.");
    } finally {
      setClearingAll(false);
      setDeletingId("");
    }
  }

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
            {Object.entries(exercises).map(([key, item]) => (
              <Button
                key={key}
                variant={activeExercise === key ? "contained" : "outlined"}
                startIcon={item.icon}
                onClick={() => chooseExercise(key)}
              >
                <span className={itemClass}>{item.shortLabel}</span>
              </Button>
            ))}
            <Button component={Link} to="/patient/results/rehab-exercise" variant="text" startIcon={<AssessmentIcon />}>
              <span className={itemClass}>Results</span>
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
                <Chip icon={<VideoCameraBackIcon />} label="Exercise quality analysis" className="rehab-hero-chip" />
                <Typography variant="h3" fontWeight={900}>{copy.label}</Typography>
                <Typography color="text.secondary">
                  Upload a matching seated exercise video to detect correct and incorrect movement windows, then save a report-ready outcome profile.
                </Typography>
              </Stack>
            </Box>

            {error && <Alert severity="error">{error}</Alert>}

            <Grid2 container spacing={3} component="form" onSubmit={submit}>
              <Grid2 size={{ xs: 12, lg: 8 }}>
                <Card className="rehab-workspace-card">
                  <CardContent>
                    <Stack spacing={2.5}>
                      <Stack direction="row" spacing={1.5} alignItems="center">
                        <UploadFileIcon color="primary" />
                        <Box>
                          <Typography variant="h5" fontWeight={900}>Upload exercise video</Typography>
                          <Typography color="text.secondary">Use a clear seated recording with the full upper body visible.</Typography>
                        </Box>
                      </Stack>
                      <Box
                        className={previewUrl ? "rehab-upload-zone has-video" : file ? "rehab-upload-zone has-file" : "rehab-upload-zone"}
                        onDragOver={(event) => event.preventDefault()}
                        onDrop={(event) => {
                          event.preventDefault();
                          handleFile(event.dataTransfer.files?.[0]);
                        }}
                      >
                        {file && (
                          <IconButton type="button" className="rehab-upload-close" color="error" onClick={removeFile} aria-label="Remove selected file">
                            <CloseIcon />
                          </IconButton>
                        )}
                        {previewUrl ? (
                          <>
                            <Stack direction="row" spacing={1} alignItems="center" className="rehab-preview-title">
                              <VideoCameraBackIcon color="primary" />
                              <Box flex={1}>
                                <Typography fontWeight={900}>Video preview</Typography>
                                <Typography color="text.secondary">{file.name} - {fileSize(file)}</Typography>
                              </Box>
                              <Button component="label" variant="outlined" size="small" startIcon={<UploadFileIcon />}>
                                Replace
                                <input hidden type="file" accept=".mp4,.mov,.avi,.mkv,video/mp4,video/quicktime" onChange={(event) => handleFile(event.target.files?.[0])} />
                              </Button>
                            </Stack>
                            <video className="rehab-inline-video" controls src={previewUrl} />
                          </>
                        ) : (
                          <>
                            <UploadFileIcon />
                            <Typography fontWeight={900}>{file ? file.name : "Drop video here or choose a file"}</Typography>
                            <Typography color="text.secondary">{file ? fileSize(file) : "MP4, MOV, AVI, or MKV"}</Typography>
                            <Button component="label" variant="contained" startIcon={<UploadFileIcon />}>
                              Choose file
                              <input hidden type="file" accept=".mp4,.mov,.avi,.mkv,video/mp4,video/quicktime" onChange={(event) => handleFile(event.target.files?.[0])} />
                            </Button>
                          </>
                        )}
                      </Box>
                    </Stack>
                  </CardContent>
                </Card>
              </Grid2>

              <Grid2 size={{ xs: 12, lg: 4 }}>
                <Card className="rehab-workspace-card">
                  <CardContent>
                    <Stack spacing={2.5}>
                      <Stack direction="row" spacing={1.5} alignItems="center">
                        <CheckCircleIcon color="primary" />
                        <Box>
                          <Typography variant="h5" fontWeight={900}>Model outcome</Typography>
                          <Typography color="text.secondary">Correctness, probability, and timed posture windows are saved after analysis.</Typography>
                        </Box>
                      </Stack>
                      <Box className="rehab-model-note">
                        <Typography fontWeight={900}>{copy.shortLabel}</Typography>
                        <Typography color="text.secondary">Selected exercise model</Typography>
                      </Box>
                      {busy && (
                        <Box>
                          <LinearProgress />
                          <Typography className="rehab-processing-text">Extracting pose, running the exercise model, and preparing timed windows...</Typography>
                        </Box>
                      )}
                      <Button type="submit" size="large" variant="contained" startIcon={<CheckCircleIcon />} disabled={busy}>
                        {busy ? "Analyzing..." : copy.action}
                      </Button>
                    </Stack>
                  </CardContent>
                </Card>
              </Grid2>
            </Grid2>

            <Card className="rehab-workspace-card">
              <CardContent>
                <Stack spacing={2}>
                  <Stack direction={{ xs: "column", md: "row" }} spacing={1.5} justifyContent="space-between">
                    <Stack direction="row" spacing={1.5} alignItems="center">
                      <HistoryIcon color="primary" />
                      <Typography variant="h5" fontWeight={900}>Latest saved {copy.shortLabel} outcomes</Typography>
                    </Stack>
                    <Stack className="rehab-history-toolbar" direction={{ xs: "column", lg: "row" }} spacing={1.2}>
                      <Stack className="rehab-history-filters" direction={{ xs: "column", sm: "row" }} spacing={1.2}>
                        <FormControl size="small" className="rehab-history-filter">
                          <InputLabel>Condition</InputLabel>
                          <Select label="Condition" value={conditionFilter} onChange={(event) => setConditionFilter(event.target.value)}>
                            <MenuItem value="all">All</MenuItem>
                            <MenuItem value="correct">Correct posture</MenuItem>
                            <MenuItem value="incorrect">Incorrect posture</MenuItem>
                            <MenuItem value="low_reliability">Low reliability</MenuItem>
                          </Select>
                        </FormControl>
                        <TextField
                          className="rehab-history-filter"
                          size="small"
                          label="Date"
                          type="date"
                          value={dateFilter}
                          onChange={(event) => setDateFilter(event.target.value)}
                          InputLabelProps={{ shrink: true }}
                        />
                      </Stack>
                      <Button
                        className="rehab-danger-button rehab-clear-all-button"
                        color="error"
                        variant="outlined"
                        startIcon={<DeleteSweepIcon />}
                        disabled={clearingAll || exerciseScreenings.length === 0}
                        onClick={clearExerciseScreenings}
                      >
                        Clear all
                      </Button>
                    </Stack>
                  </Stack>
                  {exerciseScreenings.length === 0 ? (
                    <Typography color="text.secondary">No {copy.shortLabel} outcomes have been saved yet.</Typography>
                  ) : filteredScreenings.length === 0 ? (
                    <Typography color="text.secondary">No saved outcomes match these filters.</Typography>
                  ) : (
                    <Grid2 container spacing={2}>
                      {filteredScreenings.slice(0, 8).map((screening) => (
                        <Grid2 key={screening.id} size={{ xs: 12, md: 6, xl: 3 }}>
                          <Box className="rehab-history-item">
                            <Stack direction="row" alignItems="flex-start" justifyContent="space-between">
                              <Chip
                                size="small"
                                color={resultTone(screening)}
                                icon={screening.isCorrect ? <CheckCircleIcon /> : <WarningAmberIcon />}
                                label={resultLabel(screening)}
                              />
                              <IconButton size="small" disabled={deletingId === "all" || deletingId === screening.id} onClick={() => deleteScreening(screening.id)}>
                                <DeleteOutlineIcon fontSize="small" />
                              </IconButton>
                            </Stack>
                            <Typography fontWeight={900}>{screening.finalPrediction || "Result unavailable"}</Typography>
                            <Typography color="text.secondary">{formatDate(screening.createdAt)}</Typography>
                            <Typography className="rehab-metric-helper">Correct probability {percent(screening.meanCorrectProbability)}</Typography>
                            <Typography className="rehab-metric-helper">Incorrect windows {screening.windowReport?.incorrect_windows?.length ?? 0}</Typography>
                            <Button component={Link} to={`/patient/results/rehab-exercise?id=${screening.id}`} size="small" startIcon={<ReportIcon />}>
                              Open report
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
