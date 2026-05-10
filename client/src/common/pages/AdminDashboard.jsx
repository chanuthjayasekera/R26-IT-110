import React, { useEffect, useState } from "react";
import { Alert, Avatar, Box, Button, Card, CardContent, Chip, Container, Dialog, DialogContent, DialogTitle, Grid2, List, ListItem, ListItemAvatar, ListItemText, Stack, Tab, Tabs, TextField, Typography } from "@mui/material";
import AdminPanelSettingsIcon from "@mui/icons-material/AdminPanelSettings";
import VerifiedUserIcon from "@mui/icons-material/VerifiedUser";
import HighlightOffIcon from "@mui/icons-material/HighlightOff";
import PendingActionsIcon from "@mui/icons-material/PendingActions";
import { api, getApiError } from "../api/http.js";
import { useAuth } from "../state/AuthContext.jsx";
import "../styles/dashboard.css";

function statusColor(status) {
  if (status === "approved") return "success";
  if (status === "rejected") return "error";
  return "warning";
}

export default function AdminDashboard() {
  const { user } = useAuth();
  const [tab, setTab] = useState("doctors");
  const [patients, setPatients] = useState([]);
  const [professionals, setProfessionals] = useState([]);
  const [toast, setToast] = useState("");
  const [message, setMessage] = useState("");
  const [selected, setSelected] = useState(null);

  async function load() {
    const res = await api.get("/dashboard/admin");
    setPatients(res.data.patients || []);
    setProfessionals(res.data.professionals || []);
  }

  useEffect(() => {
    load().catch((error) => setToast(getApiError(error).message));
  }, []);

  async function updateStatus(id, status) {
    try {
      const res = await api.patch(`/dashboard/admin/professionals/${id}/status`, { status, message });
      setToast(res.data.message);
      setMessage("");
      await load();
    } catch (error) {
      setToast(getApiError(error).message);
    }
  }

  return (
    <Container maxWidth="xl" className="page">
      <Stack spacing={4}>
        <Box className="dashboard-hero admin">
          <Stack spacing={1}>
            <Typography variant="overline" color="primary" fontWeight={900}>Admin dashboard</Typography>
            <Typography variant="h3">Hello {user?.fullName}, manage trusted access.</Typography>
            <Typography color="text.secondary">Approve medical professionals only after license proof, hospital email, and profile review.</Typography>
          </Stack>
          <img src="https://images.unsplash.com/photo-1551434678-e076c223a692?auto=format&fit=crop&w=900&q=80" alt="Admin management dashboard" />
        </Box>

        {toast && <Alert severity={toast.includes("approved") ? "success" : "info"} onClose={() => setToast("")}>{toast}</Alert>}

        <Card>
          <CardContent>
            <Stack spacing={2}>
              <Stack direction="row" alignItems="center" spacing={1.5}>
                <AdminPanelSettingsIcon color="primary" />
                <Typography variant="h5">Admin management</Typography>
              </Stack>
              <Tabs value={tab} onChange={(event, value) => setTab(value)}>
                <Tab value="doctors" label="Dr Management" />
                <Tab value="patients" label="Patient Management" />
              </Tabs>

              {tab === "doctors" && (
                <Grid2 container spacing={2}>
                  {professionals.map((doctor) => (
                    <Grid2 key={doctor.id} size={{ xs: 12, lg: 6 }}>
                      <Card variant="outlined" className="admin-card">
                        <CardContent>
                          <Stack spacing={2}>
                            <Stack direction="row" justifyContent="space-between" alignItems="center" spacing={2}>
                              <Stack direction="row" spacing={1.5} alignItems="center">
                                <Avatar>{doctor.full_name?.slice(0, 2).toUpperCase()}</Avatar>
                                <Box>
                                  <Typography fontWeight={900}>{doctor.full_name}</Typography>
                                  <Typography variant="body2" color="text.secondary">{doctor.email}</Typography>
                                </Box>
                              </Stack>
                              <Chip label={doctor.verification_status} color={statusColor(doctor.verification_status)} />
                            </Stack>
                            <Grid2 container spacing={1}>
                              <Grid2 size={{ xs: 12, md: 6 }}><Typography variant="body2"><b>License:</b> {doctor.medical_license}</Typography></Grid2>
                              <Grid2 size={{ xs: 12, md: 6 }}><Typography variant="body2"><b>Specialization:</b> {doctor.specialization}</Typography></Grid2>
                              <Grid2 size={{ xs: 12, md: 6 }}><Typography variant="body2"><b>Hospital:</b> {doctor.hospital}</Typography></Grid2>
                              <Grid2 size={{ xs: 12, md: 6 }}><Typography variant="body2"><b>Hospital Email:</b> {doctor.hospital_email}</Typography></Grid2>
                              <Grid2 size={{ xs: 12, md: 6 }}><Typography variant="body2"><b>Experience:</b> {doctor.years_experience} years</Typography></Grid2>
                              <Grid2 size={{ xs: 12, md: 6 }}><Typography variant="body2"><b>Proof:</b> {doctor.license_proof_name}</Typography></Grid2>
                            </Grid2>
                            <TextField label="Admin message" value={message} onChange={(e) => setMessage(e.target.value)} placeholder="Reason or approval note" />
                            <Stack direction={{ xs: "column", sm: "row" }} spacing={1}>
                              <Button variant="contained" color="success" startIcon={<VerifiedUserIcon />} onClick={() => updateStatus(doctor.id, "approved")}>Approve</Button>
                              <Button variant="outlined" color="warning" startIcon={<PendingActionsIcon />} onClick={() => updateStatus(doctor.id, "pending")}>Set Pending</Button>
                              <Button variant="contained" color="error" startIcon={<HighlightOffIcon />} onClick={() => updateStatus(doctor.id, "rejected")}>Reject</Button>
                              <Button variant="outlined" onClick={() => setSelected(doctor)}>View Proof</Button>
                            </Stack>
                          </Stack>
                        </CardContent>
                      </Card>
                    </Grid2>
                  ))}
                  {professionals.length === 0 && <Typography color="text.secondary">No medical professional registrations yet.</Typography>}
                </Grid2>
              )}

              {tab === "patients" && (
                <List>
                  {patients.map((patient) => (
                    <ListItem key={patient.id} className="patient-row">
                      <ListItemAvatar><Avatar>{patient.full_name?.slice(0, 2).toUpperCase()}</Avatar></ListItemAvatar>
                      <ListItemText primary={patient.full_name} secondary={`${patient.email} • ${patient.phone || "no phone"} • ${new Date(patient.created_at).toLocaleDateString()}`} />
                    </ListItem>
                  ))}
                  {patients.length === 0 && <Typography color="text.secondary">No patient registrations yet.</Typography>}
                </List>
              )}
            </Stack>
          </CardContent>
        </Card>
      </Stack>

      <Dialog open={Boolean(selected)} onClose={() => setSelected(null)} maxWidth="md" fullWidth>
        <DialogTitle>License proof: {selected?.license_proof_name}</DialogTitle>
        <DialogContent>
          {selected?.license_proof_data?.startsWith("data:image") ? (
            <img className="proof-preview" src={selected.license_proof_data} alt="License proof" />
          ) : selected?.license_proof_data ? (
            <Button href={selected.license_proof_data} download={selected.license_proof_name} variant="contained">Download proof document</Button>
          ) : (
            <Typography>No proof uploaded.</Typography>
          )}
        </DialogContent>
      </Dialog>
    </Container>
  );
}
