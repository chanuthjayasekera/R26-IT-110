import React from "react";
import { Box, Button, Card, CardContent, Container, Grid2, Stack, Typography } from "@mui/material";
import { Link } from "react-router-dom";
import UploadFileIcon from "@mui/icons-material/UploadFile";
import DirectionsWalkIcon from "@mui/icons-material/DirectionsWalk";
import AssessmentIcon from "@mui/icons-material/Assessment";
import SpeedIcon from "@mui/icons-material/Speed";
import "../styles/normal-abnormal-detection.css";

export default function NormalAbnormalDetection() {
  const steps = [
    ["01", "Upload gait video", "Add a walking video captured from a normal RGB camera."],
    ["02", "Pose extraction", "Prepared for MediaPipe landmarks and temporal windowing."],
    ["03", "Normal vs abnormal result", "Your trained screening model will classify gait status here."],
    ["04", "Severity preview", "Store confidence, abnormal ratio, and screening summary."]
  ];


  return (
    <Container maxWidth="xl" className="normal-detection-page">
      <Stack spacing={4}>
        <Box className="normal-detection-hero">
          <Stack spacing={2}>
            <Typography variant="overline" fontWeight={900} color="primary">Component 1</Typography>
            <Typography variant="h3" fontWeight={900}>Normal vs Abnormal Detection</Typography>
            <Typography color="text.secondary" maxWidth="840px">
              First-stage gait screening workspace for video upload, pose extraction, biometric preparation, confidence scoring, and normal or abnormal classification.
            </Typography>
            <Stack direction={{ xs: "column", sm: "row" }} spacing={2}>
              <Button variant="contained" startIcon={<UploadFileIcon />}>Upload gait video</Button>
              <Button component={Link} to="/patient/results/normal-abnormal" variant="contained">Detect</Button>
              <Button component={Link} to="/patient" variant="outlined">Back to patient dashboard</Button>
            </Stack>
          </Stack>
          <Box className="normal-detection-badge">
            <DirectionsWalkIcon />
            <Typography fontWeight={900}>Screening Ready</Typography>
          </Box>
        </Box>

        <Grid2 container spacing={3}>
          {steps.map(([number, title, text]) => (
            <Grid2 key={title} size={{ xs: 12, md: 6, lg: 3 }}>
              <Card className="normal-step-card">
                <CardContent>
                  <Stack spacing={2}>
                    <Box className="normal-step-number">{number}</Box>
                    <Typography variant="h6">{title}</Typography>
                    <Typography color="text.secondary">{text}</Typography>
                  </Stack>
                </CardContent>
              </Card>
            </Grid2>
          ))}
        </Grid2>

        <Grid2 container spacing={3}>
          <Grid2 size={{ xs: 12, md: 6 }}>
            <Card className="normal-info-card">
              <CardContent>
                <Stack spacing={2}>
                  <AssessmentIcon color="primary" />
                  <Typography variant="h5">Expected output</Typography>
                  <Typography color="text.secondary">Normal or abnormal label, prediction confidence, severity ratio, walking speed, step length, cadence, arm swing amplitude, symmetry index, and stride variability.</Typography>
                </Stack>
              </CardContent>
            </Card>
          </Grid2>
          <Grid2 size={{ xs: 12, md: 6 }}>
            
          </Grid2>
        </Grid2>
      </Stack>
    </Container>
  );
}
//*normal abnoraml detection
