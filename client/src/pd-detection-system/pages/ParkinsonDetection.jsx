import React from "react";
import { Box, Button, Card, CardContent, Container, Grid2, Stack, Typography } from "@mui/material";
import { Link } from "react-router-dom";
import PsychologyIcon from "@mui/icons-material/Psychology";
import UploadFileIcon from "@mui/icons-material/UploadFile";
import MonitorHeartIcon from "@mui/icons-material/MonitorHeart";
import ManageSearchIcon from "@mui/icons-material/ManageSearch";
import "../styles/parkinson-detection.css";

export default function ParkinsonDetection() {
  const sections = [
    ["PD prediction", "Prepared for gait rhythm, movement amplitude, stability, and Parkinson-specific model inference."],
    ["Risk retrieval", "Connect clinician-defined risk profiles by disorder and severity."],
    ["Central profile update", "Store PD prediction, confidence, severity, and longitudinal comparison data."],
    ["Clinical visibility", "Medical professionals can later review results from patient central profiles."]
  ];

  return (
    <Container maxWidth="xl" className="pd-page">
      <Stack spacing={4}>
        <Box className="pd-hero">
          <Stack spacing={2}>
            <Typography variant="overline" fontWeight={900} color="primary">Component 3</Typography>
            <Typography variant="h3" fontWeight={900}>Parkinson Detection</Typography>
            <Typography color="text.secondary" maxWidth="860px">
              Parkinson detection workspace for prediction, severity estimation, dynamic risk interpretation, and persistent gait intelligence profile updates.
            </Typography>
            <Stack direction={{ xs: "column", sm: "row" }} spacing={2}>
              <Button component={Link} to="/patient/results/pd" variant="contained" startIcon={<UploadFileIcon />}>Detect</Button>
              <Button component={Link} to="/patient" variant="outlined">Back to patient dashboard</Button>
            </Stack>
          </Stack>
          <Box className="pd-brain-panel">
            <PsychologyIcon />
            <Typography fontWeight={900}>PD Intelligence</Typography>
          </Box>
        </Box>

        <Grid2 container spacing={3}>
          {sections.map(([title, description], index) => (
            <Grid2 key={title} size={{ xs: 12, md: 6, lg: 3 }}>
              <Card className="pd-card">
                <CardContent>
                  <Stack spacing={2}>
                    <Box className="pd-index">{index + 1}</Box>
                    <Typography variant="h6">{title}</Typography>
                    <Typography color="text.secondary">{description}</Typography>
                  </Stack>
                </CardContent>
              </Card>
            </Grid2>
          ))}
        </Grid2>

        <Card className="pd-profile-card">
          <CardContent>
            <Stack direction={{ xs: "column", md: "row" }} spacing={2} alignItems={{ xs: "flex-start", md: "center" }} justifyContent="space-between">
              <Stack spacing={1}>
                <Typography variant="h5">Centralize profile </Typography>
                <Typography color="text.secondary"></Typography>
              </Stack>
              <Stack direction="row" spacing={1}>
                <MonitorHeartIcon color="primary" />
                <ManageSearchIcon color="primary" />
              </Stack>
            </Stack>
          </CardContent>
        </Card>
      </Stack>
    </Container>
  );
}
