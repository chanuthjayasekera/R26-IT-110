import React, { useEffect, useState } from "react";
import {
  Alert,
  Avatar,
  Box,
  Button,
  Card,
  CardContent,
  Container,
  FormControl,
  FormHelperText,
  Grid2,
  InputLabel,
  MenuItem,
  Select,
  Snackbar,
  Stack,
  TextField,
  Typography
} from "@mui/material";
import CameraAltIcon from "@mui/icons-material/CameraAlt";
import SaveIcon from "@mui/icons-material/Save";
import { useAuth } from "../state/AuthContext.jsx";
import profileBg from "../../assets/profile-bg.png";
import "../styles/profile.css";

function readFileAsDataUrl(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result));
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

function initials(name = "") {
  const parts = name.trim().split(/\s+/).filter(Boolean);
  if (parts.length === 0) return "US";
  if (parts.length === 1) return parts[0].slice(0, 2).toUpperCase();
  return `${parts[0][0]}${parts[1][0]}`.toUpperCase();
}

export default function Profile() {
  const { user, updateProfile } = useAuth();
  const [form, setForm] = useState({
    fullName: "",
    phone: "",
    dateOfBirth: "",
    gender: "",
    specialization: "",
    hospital: "",
    profileImage: ""
  });
  const [errors, setErrors] = useState({});
  const [toast, setToast] = useState("");
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    if (user) {
      setForm({
        fullName: user.fullName || "",
        phone: user.phone || "",
        dateOfBirth: user.dateOfBirth || "",
        gender: user.gender || "",
        specialization: user.specialization || "",
        hospital: user.hospital || "",
        profileImage: user.profileImage || ""
      });
    }
  }, [user]);

  function update(key, value) {
    setForm((old) => ({ ...old, [key]: value }));
    setErrors((old) => ({ ...old, [key]: "" }));
  }

  async function handleImage(event) {
    const file = event.target.files?.[0];
    if (!file) return;

    if (!file.type.startsWith("image/") || file.size > 2 * 1024 * 1024) {
      setToast("Use an image under 2MB.");
      return;
    }

    update("profileImage", await readFileAsDataUrl(file));
  }

  async function submit(event) {
    event.preventDefault();
    setBusy(true);
    setErrors({});

    try {
      const res = await updateProfile(form);
      setToast(res.message);
    } catch (error) {
      setErrors(error.errors || {});
      setToast(error.message || "Profile update failed.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <Box className="profile-page-wrapper" sx={{ "--profile-bg": `url(${profileBg})` }}>
      <Container maxWidth="md" className="page profile-page">
        <Card className="profile-card">
          <CardContent>
            <Stack spacing={3} component="form" onSubmit={submit}>
              <Stack direction={{ xs: "column", sm: "row" }} spacing={3} alignItems="center">
                <Avatar src={form.profileImage} className="profile-avatar">
                  {initials(form.fullName)}
                </Avatar>

                <Box flex={1}>
                  <Typography variant="h4">Profile settings</Typography>
                  <Typography color="text.secondary">
                    Update your personal details and profile picture.
                  </Typography>

                  {user?.role === "professional" && (
                    <Alert
                      severity={
                        user.verificationStatus === "approved"
                          ? "success"
                          : user.verificationStatus === "rejected"
                            ? "error"
                            : "warning"
                      }
                      sx={{ mt: 2 }}
                    >
                      Professional status: {user.verificationStatus}.{" "}
                      {user.verificationMessage || ""}
                    </Alert>
                  )}
                </Box>

                <Button component="label" variant="outlined" startIcon={<CameraAltIcon />}>
                  Profile picture
                  <input hidden type="file" accept="image/*" onChange={handleImage} />
                </Button>
              </Stack>

              <Grid2 container spacing={2}>
                <Grid2 size={{ xs: 12, md: 6 }}>
                  <TextField
                    label="Full name"
                    value={form.fullName}
                    onChange={(e) => update("fullName", e.target.value)}
                    error={Boolean(errors.fullName)}
                    helperText={errors.fullName}
                  />
                </Grid2>

                <Grid2 size={{ xs: 12, md: 6 }}>
                  <TextField
                    label="Phone"
                    value={form.phone}
                    onChange={(e) => update("phone", e.target.value)}
                    error={Boolean(errors.phone)}
                    helperText={errors.phone}
                  />
                </Grid2>

                {user?.role === "patient" && (
                  <>
                    <Grid2 size={{ xs: 12, md: 6 }}>
                      <TextField
                        label="Date of birth"
                        type="date"
                        value={form.dateOfBirth}
                        onChange={(e) => update("dateOfBirth", e.target.value)}
                        error={Boolean(errors.dateOfBirth)}
                        helperText={errors.dateOfBirth}
                        InputLabelProps={{ shrink: true }}
                      />
                    </Grid2>

                    <Grid2 size={{ xs: 12, md: 6 }}>
                      <FormControl fullWidth error={Boolean(errors.gender)}>
                        <InputLabel>Gender</InputLabel>
                        <Select
                          label="Gender"
                          value={form.gender}
                          onChange={(e) => update("gender", e.target.value)}
                        >
                          <MenuItem value="female">Female</MenuItem>
                          <MenuItem value="male">Male</MenuItem>
                          <MenuItem value="other">Other</MenuItem>
                          <MenuItem value="prefer-not-to-say">Prefer not to say</MenuItem>
                        </Select>
                        <FormHelperText>{errors.gender}</FormHelperText>
                      </FormControl>
                    </Grid2>
                  </>
                )}

                {user?.role === "professional" && (
                  <>
                    <Grid2 size={{ xs: 12, md: 6 }}>
                      <TextField
                        label="Specialization"
                        value={form.specialization}
                        onChange={(e) => update("specialization", e.target.value)}
                        error={Boolean(errors.specialization)}
                        helperText={errors.specialization}
                      />
                    </Grid2>

                    <Grid2 size={{ xs: 12, md: 6 }}>
                      <TextField
                        label="Hospital or clinic"
                        value={form.hospital}
                        onChange={(e) => update("hospital", e.target.value)}
                        error={Boolean(errors.hospital)}
                        helperText={errors.hospital}
                      />
                    </Grid2>
                  </>
                )}
              </Grid2>

              <Button type="submit" variant="contained" size="large" startIcon={<SaveIcon />} disabled={busy}>
                {busy ? "Saving..." : "Save profile"}
              </Button>
            </Stack>
          </CardContent>
        </Card>

        <Snackbar open={Boolean(toast)} autoHideDuration={4200} onClose={() => setToast("")}>
          <Alert severity={toast.includes("success") ? "success" : "info"}>{toast}</Alert>
        </Snackbar>
      </Container>
    </Box>
  );
}