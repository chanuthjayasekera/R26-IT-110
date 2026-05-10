import React, { useEffect, useState } from "react";
import { Alert, Avatar, Box, Card, CardContent, Container, Grid2, List, ListItem, ListItemAvatar, ListItemText, Stack, Typography } from "@mui/material";
import MedicalServicesIcon from "@mui/icons-material/MedicalServices";
import RuleIcon from "@mui/icons-material/Rule";
import RecommendIcon from "@mui/icons-material/Recommend";
import GroupsIcon from "@mui/icons-material/Groups";
import AssignmentIcon from "@mui/icons-material/Assignment";
import { api, getApiError } from "../api/http.js";
import { useAuth } from "../state/AuthContext.jsx";
import FeatureCard from "../components/FeatureCard.jsx";
import "../styles/dashboard.css";

const icons = [<RuleIcon />, <RecommendIcon />, <MedicalServicesIcon />, <GroupsIcon />];

export default function ProfessionalDashboard() {
  const { user } = useAuth();
  const [modules, setModules] = useState([]);
  const [patients, setPatients] = useState([]);
  const [blocked, setBlocked] = useState(null);

  useEffect(() => {
    api.get("/dashboard/professional").then((res) => {
      setModules(res.data.modules);
      setPatients(res.data.patients || []);
      setBlocked(null);
    }).catch((error) => {
      setBlocked(getApiError(error));
    });
  }, []);

  if (blocked) {
    const severity = blocked.status === "rejected" ? "error" : "warning";
    return (
      <Container maxWidth="md" className="page">
        <Card>
          <CardContent>
            <Stack spacing={2}>
              <Alert severity={severity}>{blocked.message}</Alert>
              <Typography variant="h4">
                {blocked.status === "rejected" ? "Verification rejected" : "Admin approval pending"}
              </Typography>
              <Typography color="text.secondary">
                Your medical professional registration is protected by admin review. Once approved, your dashboard for uploading risks, recommendations, rehabilitation plans, and patient engagement will unlock.
              </Typography>
              {user?.verificationMessage && <Alert severity="info">{user.verificationMessage}</Alert>}
            </Stack>
          </CardContent>
        </Card>
      </Container>
    );
  }

  return (
    <Container maxWidth="xl" className="page">
      <Stack spacing={4}>
        <Box className="dashboard-hero professional">
          <Stack spacing={1}>
            <Typography variant="overline" color="primary" fontWeight={900}>Medical professional dashboard</Typography>
            <Typography variant="h3">Dr. {user?.fullName}, clinical management center.</Typography>
            <Typography color="text.secondary">Upload risks, upload rehabilitation recommendations, and engage with patient central profiles.</Typography>
          </Stack>
          <img src="https://images.unsplash.com/photo-1576091160399-112ba8d25d1d?auto=format&fit=crop&w=1200&q=80" alt="Medical professional reviewing gait care data" />
        </Box>
        <Grid2 container spacing={3}>
          {modules.map((item, index) => (
            <Grid2 key={item.title} size={{ xs: 12, md: 6, lg: 3 }}>
              <FeatureCard icon={icons[index]} title={item.title} description={item.description} status={item.status} />
            </Grid2>
          ))}
        </Grid2>
        <Card>
          <CardContent>
            <Stack spacing={2}>
              <Stack direction="row" spacing={1.5} alignItems="center">
                <AssignmentIcon color="primary" />
                <Typography variant="h5">Engage with patient profiles</Typography>
              </Stack>
              <List>
                {patients.length === 0 ? (
                  <Typography color="text.secondary">No patient accounts registered yet.</Typography>
                ) : patients.map((patient) => (
                  <ListItem key={patient.id} className="patient-row">
                    <ListItemAvatar><Avatar>{patient.full_name?.slice(0, 2).toUpperCase()}</Avatar></ListItemAvatar>
                    <ListItemText primary={patient.full_name} secondary={`${patient.email} • registered ${new Date(patient.created_at).toLocaleDateString()}`} />
                  </ListItem>
                ))}
              </List>
            </Stack>
          </CardContent>
        </Card>
      </Stack>
    </Container>
  );
}
