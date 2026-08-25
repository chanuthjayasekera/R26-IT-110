import React, { useEffect, useState } from "react";
import { Alert, Box, Card, CardContent, Chip, CircularProgress, Container, Stack, Typography } from "@mui/material";
import { api, getApiError } from "../api/http.js";

function formatDate(value) {
  if (!value) return "-";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "-";
  return new Intl.DateTimeFormat("en", { dateStyle: "medium", timeStyle: "short" }).format(date);
}

export default function CentralizedProfile() {
  const [flags, setFlags] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    let alive = true;

    async function loadProfile() {
      setLoading(true);
      setError("");
      try {
        const res = await api.get("/central-profile/patient");
        if (alive) setFlags(res.data.flags || []);
      } catch (err) {
        if (alive) setError(getApiError(err).message || "Unable to load centralized profile.");
      } finally {
        if (alive) setLoading(false);
      }
    }

    loadProfile();
    return () => {
      alive = false;
    };
  }, []);

  return (
    <Container maxWidth="lg">
      <Stack spacing={3}>
        <Box>
          <Chip color="primary" label="Centralized profile" />
          <Typography variant="h3" fontWeight={900}>My Centralized Profile</Typography>
          <Typography color="text.secondary">
            Selected gait screening results appear here as a shared timeline for patient and professional review.
          </Typography>
        </Box>

        {error && <Alert severity="error">{error}</Alert>}
        {loading && <CircularProgress />}

        {!loading && flags.length === 0 && (
          <Alert severity="info">No screening results have been added to your centralized profile yet.</Alert>
        )}

        <Stack spacing={2}>
          {flags.map((flag) => (
            <Card key={flag.id}>
              <CardContent>
                <Stack direction={{ xs: "column", sm: "row" }} justifyContent="space-between" spacing={1}>
                  <Box>
                    <Typography variant="h6" fontWeight={900}>{flag.sourceLabel}</Typography>
                    <Typography color="text.secondary">{flag.snapshot?.title || flag.snapshot?.fileName || flag.screeningId}</Typography>
                  </Box>
                  <Typography variant="body2" color="text.secondary">{formatDate(flag.updatedAt)}</Typography>
                </Stack>
              </CardContent>
            </Card>
          ))}
        </Stack>
      </Stack>
    </Container>
  );
}
