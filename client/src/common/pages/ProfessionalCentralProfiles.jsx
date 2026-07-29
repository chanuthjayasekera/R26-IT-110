import React, { useCallback, useEffect, useState } from "react";
import { Alert, Avatar, Box, Button, Card, CardContent, Chip, CircularProgress, Container, Stack, Typography } from "@mui/material";
import { api, getApiError } from "../api/http.js";

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

export default function ProfessionalCentralProfiles() {
  const [profiles, setProfiles] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  const loadProfiles = useCallback(async () => {
    setLoading(true);
    setError("");
    try {
      const res = await api.get("/central-profile/professional/profiles");
      setProfiles(res.data.profiles || []);
    } catch (err) {
      setError(getApiError(err).message || "Unable to load centralized profiles.");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadProfiles();
  }, [loadProfiles]);

  return (
    <Container maxWidth="lg">
      <Stack spacing={3}>
        <Box>
          <Chip color="primary" label="Centralized profiles" />
          <Typography variant="h3" fontWeight={900}>Patient Central Profiles</Typography>
          <Typography color="text.secondary">
            Review patients who have selected screening results for their centralized profile.
          </Typography>
        </Box>

        {error && <Alert severity="error">{error}</Alert>}
        {loading && <CircularProgress />}

        {!loading && profiles.length === 0 && (
          <Alert severity="info">No centralized patient profiles are available yet.</Alert>
        )}

        <Stack spacing={2}>
          {profiles.map((profile) => (
            <Card key={profile.patient.id}>
              <CardContent>
                <Stack direction={{ xs: "column", sm: "row" }} justifyContent="space-between" spacing={2}>
                  <Stack direction="row" spacing={2} alignItems="center">
                    <Avatar>{initials(profile.patient.fullName)}</Avatar>
                    <Box>
                      <Typography variant="h6" fontWeight={900}>{profile.patient.fullName}</Typography>
                      <Typography color="text.secondary">{profile.patient.email}</Typography>
                    </Box>
                  </Stack>
                  <Button variant="outlined" onClick={loadProfiles}>Refresh</Button>
                </Stack>

                <Stack direction="row" spacing={1} flexWrap="wrap" mt={2}>
                  {(profile.flags || []).map((flag) => (
                    <Chip key={flag.id} label={`${flag.sourceLabel} · ${formatDate(flag.updatedAt)}`} />
                  ))}
                </Stack>
              </CardContent>
            </Card>
          ))}
        </Stack>
      </Stack>
    </Container>
  );
}
