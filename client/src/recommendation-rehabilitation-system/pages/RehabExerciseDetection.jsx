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
          </Stack>
        </Grid2>
      </Grid2>
    </Container>
  );
}
