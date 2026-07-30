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

  return (
    <Container maxWidth="xl" className="rehab-page">
      <Stack spacing={4}>
        <Box className="rehab-hero">
          <Stack spacing={2}>
            <Typography variant="overline" fontWeight={900} color="primary">Component 4</Typography>
            <Typography variant="h3" fontWeight={900}>Rehab Exercise Detection</Typography>
            <Typography color="text.secondary" maxWidth="860px">
              Rehabilitation workspace for exercise video upload, pose correctness detection, repetition quality monitoring, improvement tracking, and recommendation retrieval.
            </Typography>
            <Stack direction={{ xs: "column", sm: "row" }} spacing={2}>
              <Button variant="contained" startIcon={<UploadFileIcon />}>Upload exercise video</Button>
              <Button component={Link} to="/patient/results/rehab-exercise" variant="contained">Detect</Button>
              <Button component={Link} to="/patient" variant="outlined">Back to patient dashboard</Button>
            </Stack>
          </Stack>
          <Box className="rehab-visual-card">
            <FitnessCenterIcon />
            <Typography fontWeight={900}>Rehab Support</Typography>
          </Box>
        </Box>

        <Grid2 container spacing={3}>
          {cards.map(([title, description], index) => (
            <Grid2 key={title} size={{ xs: 12, md: 6, lg: 3 }}>
              <Card className="rehab-card">
                <CardContent>
                  <Stack spacing={2}>
                    <Box className="rehab-icon">
                      {index === 0 && <UploadFileIcon />}
                      {index === 1 && <CheckCircleIcon />}
                      {index === 2 && <RepeatIcon />}
                      {index === 3 && <TipsAndUpdatesIcon />}
                    </Box>
                    <Typography variant="h6">{title}</Typography>
                    <Typography color="text.secondary">{description}</Typography>
                  </Stack>
                </CardContent>
              </Card>
            </Grid2>
          ))}
        </Grid2>

       
      </Stack>
    </Container>
  );
}
