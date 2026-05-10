import React from "react";
import { Box, Button, Card, CardContent, Container, Grid2, Stack, Typography } from "@mui/material";
import { Link } from "react-router-dom";
import FitnessCenterIcon from "@mui/icons-material/FitnessCenter";
import UploadFileIcon from "@mui/icons-material/UploadFile";
import CheckCircleIcon from "@mui/icons-material/CheckCircle";
import RepeatIcon from "@mui/icons-material/Repeat";
import TipsAndUpdatesIcon from "@mui/icons-material/TipsAndUpdates";
import "../styles/rehab-exercise-detection.css";

export default function RehabExerciseDetection() {
  const cards = [
    ["Exercise upload", "Upload rehabilitation exercise videos for pose correctness detection."],
    ["Correctness check", "Prepared for trained rehab model feedback on exercise form."],
    ["Repetition quality", "Track repetition completion quality and consistency."],
    ["Recommendations", "Retrieve clinician-uploaded rehab plans and safety guidance."]
  ];

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
