import React, { useEffect, useState } from "react";
import { Box, Card, CardContent, Container, Grid2, Stack, Typography } from "@mui/material";
import AirlineSeatLegroomNormalIcon from "@mui/icons-material/AirlineSeatLegroomNormal";
import PsychologyIcon from "@mui/icons-material/Psychology";
import DirectionsWalkIcon from "@mui/icons-material/DirectionsWalk";
import FitnessCenterIcon from "@mui/icons-material/FitnessCenter";
import UploadFileIcon from "@mui/icons-material/UploadFile";
import { api } from "../api/http.js";
import { useAuth } from "../state/AuthContext.jsx";
import FeatureCard from "../components/FeatureCard.jsx";
import gaitImage from "../../assets/gait.avif"
import "../styles/dashboard.css";

const icons = [<DirectionsWalkIcon />, <AirlineSeatLegroomNormalIcon />, <PsychologyIcon />, <FitnessCenterIcon />];

const moduleRoutes = {
  "Normal vs Abnormal Detection": "/patient/detection/normal-abnormal",
  "SCA and KOA Detection": "/patient/detection/sca-koa",
  "PD & Neuropathy Detection": "/patient/detection/pd",
  "Parkinson Detection": "/patient/detection/pd",
  "Rehab Exercise Detection": "/patient/detection/rehab-exercise",
  "Exercise Quality Analysis": "/patient/detection/rehab-exercise"
};

export default function PatientDashboard() {
  const { user } = useAuth();
  const [modules, setModules] = useState([]);

  useEffect(() => {
    api.get("/dashboard/patient").then((res) => setModules(res.data.modules));
  }, []);

  return (
    <Container maxWidth="xl" className="page">
      <Stack spacing={4}>
        <Box className="dashboard-hero">
          <Stack spacing={1}>
            <Typography variant="overline" color="primary" fontWeight={900}>Patient home</Typography>
            <Typography variant="h3">Hello {user?.fullName}, your gait analysis workspace is ready.</Typography>
            <Typography color="text.secondary">Start with normal vs abnormal screening, then continue to SCA/KOA, PD and neuropathy detection, and exercise quality analysis.</Typography>
          </Stack>
         <img src={gaitImage} alt="Patient gait rehabilitation support" />
        </Box>
        <Grid2 container spacing={3}>
          {modules.map((item, index) => (
            <Grid2 key={item.title} size={{ xs: 12, md: 6, lg: 3 }}>
              <FeatureCard icon={icons[index]} title={item.title} description={item.description} status={item.status} to={moduleRoutes[item.title]} />
            </Grid2>
          ))}
        </Grid2>
        <Card>
          <CardContent>
            <Stack direction={{ xs: "column", md: "row" }} alignItems={{ xs: "flex-start", md: "center" }} justifyContent="space-between" spacing={2}>
              <Box>
                <Typography variant="h5">Next action</Typography>
                <Typography color="text.secondary">Upload gait video integration can be connected to your trained model pipeline next.</Typography>
              </Box>
              <Box className="upload-tile"><UploadFileIcon /><Typography fontWeight={800}>Video upload placeholder</Typography></Box>
            </Stack>
          </CardContent>
        </Card>
      </Stack>
    </Container>
  );
}
