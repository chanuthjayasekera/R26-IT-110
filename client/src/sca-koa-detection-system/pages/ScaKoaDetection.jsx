import React from "react";
import { Box, Button, Card, CardContent, Container, Grid2, Stack, Typography } from "@mui/material";
import { Link } from "react-router-dom";
import AirlineSeatLegroomNormalIcon from "@mui/icons-material/AirlineSeatLegroomNormal";
import UploadFileIcon from "@mui/icons-material/UploadFile";
import AccessibilityNewIcon from "@mui/icons-material/AccessibilityNew";
import TimelineIcon from "@mui/icons-material/Timeline";
import "../styles/sca-koa-detection.css";

export default function ScaKoaDetection() {
  const markers = [
    ["SCA indicators", "Coordination loss, irregular timing, widened stance, and trunk instability."],
    ["KOA indicators", "Reduced knee flexion, joint asymmetry, and compensatory movement patterns."],
    ["Instability map", "Green, yellow, and red body-region visualization for interpretability."],
    ["Severity output", "Mild, moderate, or severe disorder-aware status for profile storage."]
  ];

  return (
    <Container maxWidth="xl" className="sca-koa-page">
      <Stack spacing={4}>
        <Box className="sca-koa-hero">
          <Stack spacing={2}>
            <Typography variant="overline" fontWeight={900} color="primary">Component 2</Typography>
            <Typography variant="h3" fontWeight={900}>SCA vs KOA Detection</Typography>
            <Typography color="text.secondary" maxWidth="860px">
              Dedicated workspace for Spinocerebellar Ataxia and Knee Osteoarthritis model outputs, severity interpretation, and joint-level instability visualization.
            </Typography>
            <Stack direction={{ xs: "column", sm: "row" }} spacing={2}>
              <Button component={Link} to="/patient/results/sca-koa" variant="contained" startIcon={<UploadFileIcon />}>Detect</Button>
              <Button component={Link} to="/patient" variant="outlined">Back to patient dashboard</Button>
            </Stack>
          </Stack>
          <Box className="sca-koa-visual">
            <AirlineSeatLegroomNormalIcon />
            <Typography fontWeight={900}>SCA / KOA</Typography>
          </Box>
        </Box>

        <Grid2 container spacing={3}>
          {markers.map(([title, description]) => (
            <Grid2 key={title} size={{ xs: 12, md: 6 }}>
              <Card className="sca-koa-card">
                <CardContent>
                  <Stack spacing={2}>
                    <Box className="sca-koa-icon">{title.includes("KOA") ? <AccessibilityNewIcon /> : <TimelineIcon />}</Box>
                    <Typography variant="h5">{title}</Typography>
                    <Typography color="text.secondary">{description}</Typography>
                  </Stack>
                </CardContent>
              </Card>
            </Grid2>
          ))}
        </Grid2>

        <Box className="instability-strip">
          <Stack direction={{ xs: "column", md: "row" }} spacing={2} justifyContent="space-between">
            <Box><span className="stable-dot"></span><Typography fontWeight={800}>Stable movement</Typography></Box>
            <Box><span className="mild-dot"></span><Typography fontWeight={800}>Mild instability</Typography></Box>
            <Box><span className="severe-dot"></span><Typography fontWeight={800}>Severe instability</Typography></Box>
          </Stack>
        </Box>
      </Stack>
    </Container>
  );
}
