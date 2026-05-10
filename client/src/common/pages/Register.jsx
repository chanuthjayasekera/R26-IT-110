import React, { useState } from "react";
import {
  Alert,
  Box,
  Button,
  Card,
  CardContent,
  Container,
  Divider,
  FormControl,
  FormHelperText,
  Grid2,
  InputLabel,
  MenuItem,
  Select,
  Snackbar,
  Stack,
  TextField,
  ToggleButton,
  ToggleButtonGroup,
  Typography
} from "@mui/material";
import PersonAddAltIcon from "@mui/icons-material/PersonAddAlt";
import MedicalServicesIcon from "@mui/icons-material/MedicalServices";
import { useNavigate } from "react-router-dom";
import { useAuth } from "../state/AuthContext.jsx";
import patientRegisterImage from "../../assets/patient-register.png";
import professionalRegisterImage from "../../assets/professional-register.png";
import "../styles/auth.css";

const initial = {
  role: "patient",
  fullName: "",
  email: "",
  password: "",
  confirmPassword: "",
  phone: "",
  dateOfBirth: "",
  gender: "",
  medicalLicense: "",
  specialization: "",
  hospital: "",
  hospitalEmail: "",
  yearsExperience: "",
  licenseProofName: "",
  licenseProofData: ""
};

const visualContent = {
  patient: {
    icon: <PersonAddAltIcon />,
    title: "Create your patient gaitcare account.",
    body: "Start gait screening, track rehabilitation progress, and receive severity-aware follow-up support from your care team.",
    image: patientRegisterImage,
    alt: "Patient gait-care registration"
  },
  professional: {
    icon: <MedicalServicesIcon />,
    title: "Register as a verified medical professional.",
    body: "Submit your clinical details for secure access to patient gait insights, risk visibility, and rehabilitation monitoring tools.",
    image: professionalRegisterImage,
    alt: "Medical professional gait-care registration"
  }
};

function readFileAsDataUrl(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result));
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

export default function Register() {
  const { register } = useAuth();
  const navigate = useNavigate();
  const [form, setForm] = useState(initial);
  const [errors, setErrors] = useState({});
  const [toast, setToast] = useState("");
  const [busy, setBusy] = useState(false);

  const visual = visualContent[form.role];

  function update(key, value) {
    setForm((old) => ({ ...old, [key]: value }));
    setErrors((old) => ({ ...old, [key]: "" }));
  }

  async function handleProof(event) {
    const file = event.target.files?.[0];
    if (!file) return;

    if (file.size > 2.5 * 1024 * 1024) {
      setErrors((old) => ({
        ...old,
        licenseProofData: "File must be below 2.5MB."
      }));
      return;
    }

    const data = await readFileAsDataUrl(file);
    update("licenseProofName", file.name);
    update("licenseProofData", data);
  }

  async function submit(event) {
    event.preventDefault();
    setBusy(true);
    setErrors({});

    try {
      const res = await register(form);
      setToast(res.message);
      navigate(res.user.role === "professional" ? "/professional" : "/patient");
    } catch (error) {
      setErrors(error.errors || {});
      setToast(error.message || "Registration failed.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <Container maxWidth="lg" className="page">
      <Grid2 container spacing={4} alignItems="stretch">
        <Grid2 size={{ xs: 12, md: 5 }}>
          <Box className="register-visual">
            {visual.icon}
            <Typography variant="h3">{visual.title}</Typography>
            <Typography>{visual.body}</Typography>
            <img src={visual.image} alt={visual.alt} />
          </Box>
        </Grid2>

        <Grid2 size={{ xs: 12, md: 7 }}>
          <Card>
            <CardContent>
              <Stack spacing={2.4} component="form" onSubmit={submit}>
                <Typography variant="h4">Registration</Typography>

                <ToggleButtonGroup
                  exclusive
                  value={form.role}
                  onChange={(e, value) => value && update("role", value)}
                  fullWidth
                >
                  <ToggleButton value="patient">Patient</ToggleButton>
                  <ToggleButton value="professional">Medical Professional</ToggleButton>
                </ToggleButtonGroup>

                {errors.role && <Alert severity="error">{errors.role}</Alert>}

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
                      label="Email"
                      value={form.email}
                      onChange={(e) => update("email", e.target.value)}
                      error={Boolean(errors.email)}
                      helperText={errors.email}
                    />
                  </Grid2>

                  <Grid2 size={{ xs: 12, md: 6 }}>
                    <TextField
                      label="Password"
                      type="password"
                      value={form.password}
                      onChange={(e) => update("password", e.target.value)}
                      error={Boolean(errors.password)}
                      helperText={
                        errors.password ||
                        "8+ chars with uppercase, lowercase, number, symbol"
                      }
                    />
                  </Grid2>

                  <Grid2 size={{ xs: 12, md: 6 }}>
                    <TextField
                      label="Confirm password"
                      type="password"
                      value={form.confirmPassword}
                      onChange={(e) => update("confirmPassword", e.target.value)}
                      error={Boolean(errors.confirmPassword)}
                      helperText={errors.confirmPassword}
                    />
                  </Grid2>

                  <Grid2 size={{ xs: 12, md: 6 }}>
                    <TextField
                      label="Phone"
                      value={form.phone}
                      onChange={(e) => update("phone", e.target.value)}
                    />
                  </Grid2>
                </Grid2>

                <Divider />

                {form.role === "patient" ? (
                  <Grid2 container spacing={2}>
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
                          <MenuItem value="prefer-not-to-say">
                            Prefer not to say
                          </MenuItem>
                        </Select>
                        <FormHelperText>{errors.gender}</FormHelperText>
                      </FormControl>
                    </Grid2>
                  </Grid2>
                ) : (
                  <Grid2 container spacing={2}>
                    <Grid2 size={{ xs: 12, md: 6 }}>
                      <TextField
                        label="Medical license number"
                        value={form.medicalLicense}
                        onChange={(e) => update("medicalLicense", e.target.value)}
                        error={Boolean(errors.medicalLicense)}
                        helperText={errors.medicalLicense}
                      />
                    </Grid2>

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

                    <Grid2 size={{ xs: 12, md: 6 }}>
                      <TextField
                        label="Hospital email"
                        value={form.hospitalEmail}
                        onChange={(e) => update("hospitalEmail", e.target.value)}
                        error={Boolean(errors.hospitalEmail)}
                        helperText={
                          errors.hospitalEmail ||
                          "Use hospital, clinic, medical, care, university, or health domain."
                        }
                      />
                    </Grid2>

                    <Grid2 size={{ xs: 12, md: 6 }}>
                      <TextField
                        label="Years of experience"
                        type="number"
                        value={form.yearsExperience}
                        onChange={(e) => update("yearsExperience", e.target.value)}
                        error={Boolean(errors.yearsExperience)}
                        helperText={errors.yearsExperience}
                      />
                    </Grid2>

                    <Grid2 size={{ xs: 12, md: 6 }}>
                      <Button variant="outlined" component="label" fullWidth>
                        Upload license proof
                        <input
                          hidden
                          type="file"
                          accept=".pdf,.png,.jpg,.jpeg"
                          onChange={handleProof}
                        />
                      </Button>
                      <FormHelperText error={Boolean(errors.licenseProofData)}>
                        {errors.licenseProofData ||
                          form.licenseProofName ||
                          "PDF/image under 2.5MB."}
                      </FormHelperText>
                    </Grid2>

                    <Grid2 size={{ xs: 12 }}>
                      <Alert severity="warning">
                        Medical professional accounts stay pending until admin
                        approval.
                      </Alert>
                    </Grid2>
                  </Grid2>
                )}

                <Button type="submit" size="large" variant="contained" disabled={busy}>
                  {busy ? "Creating account..." : "Create account"}
                </Button>
              </Stack>
            </CardContent>
          </Card>
        </Grid2>
      </Grid2>

      <Snackbar
        open={Boolean(toast)}
        autoHideDuration={5000}
        onClose={() => setToast("")}
      >
        <Alert severity={toast.includes("successful") ? "success" : "info"}>
          {toast}
        </Alert>
      </Snackbar>
    </Container>
  );
}