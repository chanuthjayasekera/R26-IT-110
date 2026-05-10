import React from "react";
import { Box, Button, Card, CardContent, Container, Grid2, Stack, Typography } from "@mui/material";
import { Link, useParams } from "react-router-dom";
import UploadFileIcon from "@mui/icons-material/UploadFile";
import InsightsIcon from "@mui/icons-material/Insights";
import TimelineIcon from "@mui/icons-material/Timeline";
import HealthAndSafetyIcon from "@mui/icons-material/HealthAndSafety";
import "../styles/detection.css";

const detectionConfigs = {
  "normal-abnormal": {
    eyebrow: "Component 1",
    title: "Normal vs Abnormal Gait Detection",
    description: "Upload a gait video and prepare it for camera-only screening. Your trained model can be connected here to classify normal and abnormal gait patterns.",
    steps: ["Video upload", "Pose extraction", "Window-level screening", "Severity score preview"],
    next: "After abnormal screening, results can flow into SCA, KOA, and PD detection."
  },
  "sca-koa": {
    eyebrow: "Component 2",
    title: "SCA and KOA Detection",
    description: "This page is reserved for Spinocerebellar Ataxia and Knee Osteoarthritis detection using the trained model outputs and gait instability indicators.",
    steps: ["Load abnormal gait profile", "Run SCA classifier", "Run KOA classifier", "Show instability map"],
    next: "Model files can later connect to disorder prediction, severity, and joint-level visualization."
  },
  pd: {
    eyebrow: "Component 3",
    title: "Parkinson Detection",
    description: "This page is prepared for Parkinson detection, severity estimation, risk interpretation, and persistent profile updates.",
    steps: ["Load gait features", "Run PD model", "Retrieve risk profile", "Store longitudinal result"],
    next: "Risk results can later connect to clinician-defined rules and central patient history."
  },
  rehab: {
    eyebrow: "Component 4",
    title: "Rehab & Recommendation System",
    description: "This page is prepared for rehabilitation exercise upload, pose correctness monitoring, repetition quality, and personalized recommendations.",
    steps: ["Upload rehab exercise", "Pose correctness check", "Quality feedback", "Recommendation retrieval"],
    next: "Recommendation data can later be uploaded by verified medical professionals."
  }
};

export default function DetectionPage() {
  const { type } = useParams();
  const config = detectionConfigs[type] || detectionConfigs["normal-abnormal"];

  return (
    <Container maxWidth="xl" className="page">
      <Stack spacing={4}>
        <Box className="detection-hero">
          <Stack spacing={2}>
            <Typography variant="overline" color="primary" fontWeight={900}>{config.eyebrow}</Typography>
            <Typography variant="h3" fontWeight={900}>{config.title}</Typography>
            <Typography color="text.secondary" maxWidth="850px">{config.description}</Typography>
            <Stack direction={{ xs: "column", sm: "row" }} spacing={2}>
              <Button variant="contained" startIcon={<UploadFileIcon />}>Model upload placeholder</Button>
              <Button component={Link} to="/patient" variant="outlined">Back to patient dashboard</Button>
            </Stack>
          </Stack>
        </Box>

        <Grid2 container spacing={3}>
          {config.steps.map((step, index) => (
            <Grid2 key={step} size={{ xs: 12, md: 6, lg: 3 }}>
              <Card className="detection-step">
                <CardContent>
                  <Stack spacing={2}>
                    <span className="pipeline-pill">
                      {index === 0 && <UploadFileIcon fontSize="small" />}
                      {index === 1 && <InsightsIcon fontSize="small" />}
                      {index === 2 && <TimelineIcon fontSize="small" />}
                      {index === 3 && <HealthAndSafetyIcon fontSize="small" />}
                      Step {index + 1}
                    </span>
                    <Typography variant="h6">{step}</Typography>
                    <Typography color="text.secondary">Ready for future trained model integration.</Typography>
                  </Stack>
                </CardContent>
              </Card>
            </Grid2>
          ))}
        </Grid2>

        <Box className="model-dropzone">
          <Stack spacing={1} alignItems="center">
            <UploadFileIcon color="primary" />
            <Typography variant="h5" fontWeight={900}>Detection workspace prepared</Typography>
            <Typography color="text.secondary">{config.next}</Typography>
          </Stack>
        </Box>
      </Stack>
    </Container>
  );
}
