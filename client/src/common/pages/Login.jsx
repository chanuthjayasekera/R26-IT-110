import React, { useState } from "react";
import { Alert, Box, Button, Card, CardContent, Container, Grid2, Link as MuiLink, Snackbar, Stack, TextField, Typography } from "@mui/material";
import LockOpenIcon from "@mui/icons-material/LockOpen";
import { Link, useNavigate } from "react-router-dom";
import { useAuth } from "../state/AuthContext.jsx";
import "../styles/auth.css";
import gaitLoginImage from "../../assets/gait-image-login.png";

function destinationFor(user) {
  if (user.role === "admin") return "/admin";
  if (user.role === "professional") return "/professional";
  return "/patient";
}

export default function Login() {
  const { login } = useAuth();
  const navigate = useNavigate();
  const [form, setForm] = useState({ email: "", password: "" });
  const [errors, setErrors] = useState({});
  const [toast, setToast] = useState("");
  const [busy, setBusy] = useState(false);

  function update(key, value) {
    setForm((old) => ({ ...old, [key]: value }));
    setErrors((old) => ({ ...old, [key]: "" }));
  }

  async function submit(event) {
    event.preventDefault();
    setBusy(true);
    setErrors({});
    try {
      const res = await login(form);
      setToast(res.message);
      navigate(destinationFor(res.user));
    } catch (error) {
      setErrors(error.errors || {});
      setToast(error.message || "Login failed.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <Container maxWidth="lg" className="page auth-page">
      <Grid2 container spacing={4} alignItems="center">
        <Grid2 size={{ xs: 12, md: 5 }}>
          <Card>
            <CardContent>
              <Stack spacing={2.5} component="form" onSubmit={submit}>
                <Box className="auth-icon"><LockOpenIcon /></Box>
                <Typography variant="h4">login</Typography>
                <Typography color="text.secondary">Continue to the right workspace</Typography>
                <TextField label="Email" value={form.email} onChange={(e) => update("email", e.target.value)} error={Boolean(errors.email)} helperText={errors.email} autoComplete="email" />
                <TextField label="Password" type="password" value={form.password} onChange={(e) => update("password", e.target.value)} error={Boolean(errors.password)} helperText={errors.password} autoComplete="current-password" />
                <Button type="submit" variant="contained" size="large" disabled={busy}>{busy ? "Checking..." : "Login"}</Button>
                <Stack direction="row" justifyContent="space-between">
                  <MuiLink component={Link} to="/forgot-password">Forgot password?</MuiLink>
                  <MuiLink component={Link} to="/register">Create account</MuiLink>
                </Stack>
              </Stack>
            </CardContent>
          </Card>
        </Grid2>
        <Grid2 size={{ xs: 12, md: 7 }}>
          <Box className="auth-side">
            <Typography variant="h3">Every step tells a clinical story.</Typography>
            <Typography color="text.secondary"> For gait screening, severity-aware follow-up, risk visibility, and rehabilitation support.</Typography>
          <img src={gaitLoginImage} alt="Gait rehabilitation walking training" />
          </Box>
        </Grid2>
      </Grid2>
      <Snackbar open={Boolean(toast)} autoHideDuration={4200} onClose={() => setToast("")}>
        <Alert severity={toast.includes("successful") ? "success" : "info"}>{toast}</Alert>
      </Snackbar>
    </Container>
  );
}
