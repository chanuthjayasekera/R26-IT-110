import React, { useCallback, useEffect, useMemo, useState } from "react";
import {
  Alert,
  Avatar,
  Box,
  Button,
  Card,
  CardContent,
  Chip,
  CircularProgress,
  Container,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  Grid2,
  MenuItem,
  Stack,
  TextField,
  Typography
} from "@mui/material";
import { useSearchParams } from "react-router-dom";
import AssignmentIcon from "@mui/icons-material/Assignment";
import EventAvailableIcon from "@mui/icons-material/EventAvailable";
import FactCheckIcon from "@mui/icons-material/FactCheck";
import HealingIcon from "@mui/icons-material/Healing";
import NotificationsActiveIcon from "@mui/icons-material/NotificationsActive";
import WarningAmberIcon from "@mui/icons-material/WarningAmber";
import CentralizedProfile from "./CentralizedProfile.jsx";
import { api, getApiError } from "../api/http.js";
import "../styles/professional-central-profiles.css";

function initials(name = "") {
  const parts = name.trim().split(/\s+/).filter(Boolean);
  if (parts.length === 0) return "PT";
  if (parts.length === 1) return parts[0].slice(0, 2).toUpperCase();
  return `${parts[0][0]}${parts[1][0]}`.toUpperCase();
}

function formatDate(value) {
  if (!value) return "-";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "-";
  return new Intl.DateTimeFormat("en", { dateStyle: "medium", timeStyle: "short" }).format(date);
}

function formatDateOnly(value) {
  if (!value) return "-";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "-";
  return new Intl.DateTimeFormat("en", { dateStyle: "medium" }).format(date);
}

function reviewDueAtFor(profile, type) {
  const guidance = profile?.guidance || {};
  const sharedReviewDueAt = profileReviewDueAt(profile);
  if (sharedReviewDueAt) return sharedReviewDueAt;
  const direct = type === "risk" ? guidance.nextRiskReviewDueAt : guidance.nextRehabReviewDueAt;
  if (direct) return direct;
  return (guidance.reminders || []).find((item) => item.guidanceType === type)?.dueAt || "";
}

function profileReviewDueAt(profile) {
  const guidance = profile?.guidance || {};
  return guidance.nextReviewDueAt
    || (guidance.reminders || [])[0]?.dueAt
    || guidance.nextRiskReviewDueAt
    || guidance.nextRehabReviewDueAt
    || "";
}

function isReviewOverdue(value) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return false;
  const today = new Date();
  today.setHours(0, 0, 0, 0);
  date.setHours(0, 0, 0, 0);
  return date <= today;
}

function daysUntilReview(value) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return null;
  const today = new Date();
  today.setHours(0, 0, 0, 0);
  date.setHours(0, 0, 0, 0);
  return Math.round((date - today) / 86400000);
}

const defaultFilters = {
  dateFrom: "",
  dateTo: "",
  normal: "all",
  diseaseType: "all",
  guidanceStatus: "all",
  reviewStatus: "all"
};

function matchesReviewStatus(profile, reviewStatus) {
  if (!reviewStatus || reviewStatus === "all") return true;
  const dueAt = profileReviewDueAt(profile);
  const days = daysUntilReview(dueAt);
  if (reviewStatus === "due") return days !== null && days <= 0;
  if (reviewStatus === "due-soon") return days !== null && days > 0 && days <= 3;
  if (reviewStatus === "upcoming") return days !== null && days > 0;
  if (reviewStatus === "no-review") return !dueAt;
  return true;
}

export default function ProfessionalCentralProfiles() {
  const [searchParams, setSearchParams] = useSearchParams();
  const selectedPatientId = searchParams.get("patientId") || "";
  const [filters, setFilters] = useState(defaultFilters);
  const [profiles, setProfiles] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [reviewNoticeOpen, setReviewNoticeOpen] = useState(false);

  const loadProfiles = useCallback(async () => {
    setLoading(true);
    setError("");
    try {
      const res = await api.get("/central-profile/professional/profiles", { params: filters });
      setProfiles(res.data.profiles || []);
    } catch (err) {
      setError(getApiError(err).message || "Unable to load centralized profiles.");
    } finally {
      setLoading(false);
    }
  }, [filters]);

  useEffect(() => {
    if (!selectedPatientId) loadProfiles();
  }, [loadProfiles, selectedPatientId]);

  const visibleProfiles = useMemo(() => profiles.filter((profile) => matchesReviewStatus(profile, filters.reviewStatus)), [profiles, filters.reviewStatus]);

  const stats = useMemo(() => {
    const missingUploads = visibleProfiles.filter((item) => item.guidance.riskCount === 0 || item.guidance.rehabCount === 0).length;
    const completeGuidance = visibleProfiles.filter((item) => item.guidance.riskCount > 0 && item.guidance.rehabCount > 0).length;
    const reviewDates = visibleProfiles.map(profileReviewDueAt).filter(Boolean);
    const dueReviews = reviewDates.filter(isReviewOverdue).length;
    const upcomingReviews = reviewDates.filter((value) => !isReviewOverdue(value)).length;
    const dueSoonReviews = reviewDates.filter((value) => {
      const days = daysUntilReview(value);
      return days !== null && days > 0 && days <= 3;
    }).length;
    return { missingUploads, completeGuidance, dueReviews, upcomingReviews, dueSoonReviews };
  }, [visibleProfiles]);

  const reviewNoticeProfiles = useMemo(() => visibleProfiles
    .map((profile) => ({ profile, dueAt: profileReviewDueAt(profile), daysUntil: daysUntilReview(profileReviewDueAt(profile)) }))
    .filter((item) => item.dueAt && item.daysUntil !== null && item.daysUntil <= 3)
    .sort((a, b) => a.daysUntil - b.daysUntil), [visibleProfiles]);

  useEffect(() => {
    if (!loading && !selectedPatientId && reviewNoticeProfiles.length > 0) {
      setReviewNoticeOpen(true);
    }
  }, [loading, reviewNoticeProfiles.length, selectedPatientId]);

  function updateFilter(field, value) {
    setFilters((current) => ({ ...current, [field]: value }));
  }

  function openProfile(patientId, section = searchParams.get("section") || "risks") {
    const next = new URLSearchParams(searchParams);
    next.set("patientId", patientId);
    if (section && section !== "normal") next.set("section", section);
    setSearchParams(next);
  }

  function backToList() {
    const next = new URLSearchParams(searchParams);
    next.delete("patientId");
    setSearchParams(next);
    loadProfiles();
  }

  if (selectedPatientId) {
    return <CentralizedProfile mode="professional" patientId={selectedPatientId} onBack={backToList} />;
  }

  return (
    <Container maxWidth="xl" className="professional-central-page">
      <Stack spacing={3}>
        <Box className="professional-central-hero">
          <Stack spacing={1}>
            <Chip icon={<FactCheckIcon />} color="primary" label="Patient centralized profiles" />
            <Typography variant="h3" fontWeight={900}>Patient Central Profiles</Typography>
            <Typography color="text.secondary">
              Patients appear here once they select at least one centralized profile result. Use filters to find the right profile for risk and rehabilitation uploads.
            </Typography>
            {(stats.dueReviews > 0 || stats.dueSoonReviews > 0) && (
              <Alert severity={stats.dueReviews > 0 ? "error" : "warning"} className="professional-review-alert">
                {stats.dueReviews > 0
                  ? `${stats.dueReviews} patient review${stats.dueReviews === 1 ? "" : "s"} due today or overdue.`
                  : `${stats.dueSoonReviews} patient review${stats.dueSoonReviews === 1 ? "" : "s"} due within 3 days.`}
              </Alert>
            )}
            <Box className="professional-review-filter">
              <Box>
                <Typography fontWeight={900}>Reviews</Typography>
                <Typography variant="body2" color="text.secondary">Filter central profiles by doctor review timing.</Typography>
              </Box>
              <TextField select size="small" label="Review status" value={filters.reviewStatus} onChange={(event) => updateFilter("reviewStatus", event.target.value)}>
                <MenuItem value="all">All reviews</MenuItem>
                <MenuItem value="due">Due / overdue</MenuItem>
                <MenuItem value="due-soon">Due within 3 days</MenuItem>
                <MenuItem value="upcoming">Upcoming</MenuItem>
                <MenuItem value="no-review">No review date</MenuItem>
              </TextField>
            </Box>
          </Stack>
          <Grid2 container spacing={1.5}>
            <Grid2 size={{ xs: 12, sm: 6, lg: 2.4 }}>
              <Box className="professional-stat-tile">
                <Typography variant="h4" fontWeight={900}>{visibleProfiles.length}</Typography>
                <Typography color="text.secondary">central profiles</Typography>
              </Box>
            </Grid2>
            <Grid2 size={{ xs: 12, sm: 6, lg: 2.4 }}>
              <Box className="professional-stat-tile warning">
                <Typography variant="h4" fontWeight={900}>{stats.missingUploads}</Typography>
                <Typography color="text.secondary">missing uploads</Typography>
              </Box>
            </Grid2>
            <Grid2 size={{ xs: 12, sm: 6, lg: 2.4 }}>
              <Box className="professional-stat-tile danger">
                <Typography variant="h4" fontWeight={900}>{stats.dueReviews}</Typography>
                <Typography color="text.secondary">due reviews</Typography>
              </Box>
            </Grid2>
            <Grid2 size={{ xs: 12, sm: 6, lg: 2.4 }}>
              <Box className="professional-stat-tile info">
                <Typography variant="h4" fontWeight={900}>{stats.upcomingReviews}</Typography>
                <Typography color="text.secondary">upcoming reviews</Typography>
              </Box>
            </Grid2>
            <Grid2 size={{ xs: 12, sm: 6, lg: 2.4 }}>
              <Box className="professional-stat-tile success">
                <Typography variant="h4" fontWeight={900}>{stats.completeGuidance}</Typography>
                <Typography color="text.secondary">fully uploaded</Typography>
              </Box>
            </Grid2>
          </Grid2>
        </Box>

        <Card className="professional-filter-card">
          <CardContent>
            <Grid2 container spacing={1.5} alignItems="center">
              <Grid2 size={{ xs: 12, md: 2 }}>
                <TextField fullWidth size="small" type="date" label="From date" value={filters.dateFrom} onChange={(event) => updateFilter("dateFrom", event.target.value)} InputLabelProps={{ shrink: true }} />
              </Grid2>
              <Grid2 size={{ xs: 12, md: 2 }}>
                <TextField fullWidth size="small" type="date" label="To date" value={filters.dateTo} onChange={(event) => updateFilter("dateTo", event.target.value)} InputLabelProps={{ shrink: true }} />
              </Grid2>
              <Grid2 size={{ xs: 12, md: 2 }}>
                <TextField fullWidth select size="small" label="Normal result" value={filters.normal} onChange={(event) => updateFilter("normal", event.target.value)}>
                  <MenuItem value="all">All</MenuItem>
                  <MenuItem value="normal">Normal</MenuItem>
                  <MenuItem value="abnormal">Abnormal</MenuItem>
                  <MenuItem value="inconclusive">Inconclusive</MenuItem>
                </TextField>
              </Grid2>
              <Grid2 size={{ xs: 12, md: 2 }}>
                <TextField fullWidth select size="small" label="Disease signal" value={filters.diseaseType} onChange={(event) => updateFilter("diseaseType", event.target.value)}>
                  <MenuItem value="all">All</MenuItem>
                  <MenuItem value="sca">SCA</MenuItem>
                  <MenuItem value="koa">KOA</MenuItem>
                  <MenuItem value="pd">PD</MenuItem>
                  <MenuItem value="neuropathy">Neuropathy</MenuItem>
                  <MenuItem value="exercise">Exercise concern</MenuItem>
                </TextField>
              </Grid2>
              <Grid2 size={{ xs: 12, md: 2 }}>
                <TextField fullWidth select size="small" label="Upload status" value={filters.guidanceStatus} onChange={(event) => updateFilter("guidanceStatus", event.target.value)}>
                  <MenuItem value="all">All</MenuItem>
                  <MenuItem value="missing-risk">Still needs risks</MenuItem>
                  <MenuItem value="missing-rehab">Still needs rehab</MenuItem>
                  <MenuItem value="complete">Risk + rehab uploaded</MenuItem>
                </TextField>
              </Grid2>
              <Grid2 size={{ xs: 12, md: 2 }}>
                <Button fullWidth variant="contained" onClick={loadProfiles}>Apply filters</Button>
              </Grid2>
            </Grid2>
          </CardContent>
        </Card>

        {error && <Alert severity="error">{error}</Alert>}
        {loading ? (
          <Stack alignItems="center" py={6}><CircularProgress /></Stack>
        ) : (
          <Grid2 container spacing={2}>
            {visibleProfiles.map((profile) => (
              <Grid2 key={profile.patient.id} size={{ xs: 12, lg: 6 }}>
                <Card className="professional-profile-card">
                  <CardContent>
                    <Stack spacing={2}>
                      <Stack direction={{ xs: "column", sm: "row" }} justifyContent="space-between" spacing={1.5}>
                        <Stack direction="row" spacing={1.5} alignItems="center">
                          <Avatar className="professional-patient-avatar">{initials(profile.patient.fullName)}</Avatar>
                          <Box>
                            <Typography variant="h6" fontWeight={900}>{profile.patient.fullName}</Typography>
                            <Typography color="text.secondary">{profile.patient.email}</Typography>
                          </Box>
                        </Stack>
                        <Chip color="success" label={`${profile.completedSlots}/${profile.totalSlots} complete`} />
                      </Stack>

                      <Stack direction="row" spacing={1} flexWrap="wrap">
                        <Chip size="small" label={`Normal/abnormal: ${profile.normalStatus}`} />
                        {(profile.diseaseSignals || []).map((item) => <Chip key={item} size="small" color="warning" label={item} />)}
                        {!profile.diseaseSignals?.length && <Chip size="small" label="No active disease signal" />}
                      </Stack>

                      <Grid2 container spacing={1.2}>
                        <Grid2 size={{ xs: 12, sm: 6, lg: 3 }}>
                          <Box className="professional-mini-tile">
                            <WarningAmberIcon />
                            <Typography fontWeight={900}>{profile.guidance.riskCount ? "Uploaded" : "Not uploaded"}</Typography>
                            <Typography color="text.secondary">Risk profile</Typography>
                            <Typography className="professional-review-date">Risk review: {formatDateOnly(reviewDueAtFor(profile, "risk"))}</Typography>
                          </Box>
                        </Grid2>
                        <Grid2 size={{ xs: 12, sm: 6, lg: 3 }}>
                          <Box className="professional-mini-tile">
                            <HealingIcon />
                            <Typography fontWeight={900}>{profile.guidance.rehabCount ? "Uploaded" : "Not uploaded"}</Typography>
                            <Typography color="text.secondary">Rehab plan</Typography>
                            <Typography className="professional-review-date">Rehab review: {formatDateOnly(reviewDueAtFor(profile, "rehab"))}</Typography>
                          </Box>
                        </Grid2>
                        <Grid2 size={{ xs: 12, sm: 6, lg: 3 }}>
                          <Box className={isReviewOverdue(profileReviewDueAt(profile)) ? "professional-mini-tile reminder overdue" : "professional-mini-tile reminder"}>
                            {isReviewOverdue(profileReviewDueAt(profile)) ? <NotificationsActiveIcon /> : <EventAvailableIcon />}
                            <Typography fontWeight={900}>{formatDateOnly(profileReviewDueAt(profile))}</Typography>
                            <Typography color="text.secondary">{isReviewOverdue(profileReviewDueAt(profile)) ? "Review overdue" : "Next review"}</Typography>
                          </Box>
                        </Grid2>
                        <Grid2 size={{ xs: 12, sm: 6, lg: 3 }}>
                          <Box className="professional-mini-tile">
                            <AssignmentIcon />
                            <Typography fontWeight={900}>{formatDate(profile.updatedAt)}</Typography>
                            <Typography color="text.secondary">Last profile update</Typography>
                          </Box>
                        </Grid2>
                      </Grid2>

                      <Stack direction={{ xs: "column", sm: "row" }} spacing={1}>
                        <Button variant="contained" onClick={() => openProfile(profile.patient.id, "risks")}>Open risk upload</Button>
                        <Button variant="outlined" onClick={() => openProfile(profile.patient.id, "rehab")}>Open recommendations</Button>
                        <Button variant="text" onClick={() => openProfile(profile.patient.id, "normal")}>View profile</Button>
                      </Stack>
                    </Stack>
                  </CardContent>
                </Card>
              </Grid2>
            ))}
            {visibleProfiles.length === 0 && (
              <Grid2 size={{ xs: 12 }}>
                <Alert severity="info">No centralized profiles match these filters yet.</Alert>
              </Grid2>
            )}
          </Grid2>
        )}
      </Stack>
      <Dialog open={reviewNoticeOpen} onClose={() => setReviewNoticeOpen(false)} fullWidth maxWidth="sm">
        <DialogTitle>Doctor Review Reminder</DialogTitle>
        <DialogContent>
          <Stack spacing={1.5} sx={{ pt: 1 }}>
            <Typography color="text.secondary">
              These patient profiles need review attention now or very soon.
            </Typography>
            {reviewNoticeProfiles.map(({ profile, dueAt, daysUntil }) => (
              <Box key={profile.patient.id} className={daysUntil <= 0 ? "professional-review-notice danger" : "professional-review-notice"}>
                <Stack direction="row" spacing={1.2} alignItems="center">
                  <Avatar className="professional-patient-avatar">{initials(profile.patient.fullName)}</Avatar>
                  <Box>
                    <Typography fontWeight={900}>{profile.patient.fullName}</Typography>
                    <Typography color="text.secondary">{formatDateOnly(dueAt)} - {daysUntil <= 0 ? "due now" : `due in ${daysUntil} day${daysUntil === 1 ? "" : "s"}`}</Typography>
                  </Box>
                </Stack>
                <Button variant="contained" size="small" onClick={() => openProfile(profile.patient.id, "risks")}>Open profile</Button>
              </Box>
            ))}
          </Stack>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setReviewNoticeOpen(false)}>Dismiss</Button>
        </DialogActions>
      </Dialog>
    </Container>
  );
}
