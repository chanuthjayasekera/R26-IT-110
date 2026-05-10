import React from "react"
import { Alert, Button, Card, CardContent, Container, Link as MuiLink, Snackbar, Stack, TextField, Typography } from "@mui/material";
import PasswordIcon from "@mui/icons-material/Password";
import { Link, useSearchParams } from "react-router-dom";
import { useState } from "react";
import { api, getApiError } from "../api/http.js";
import "../styles/auth.css";

export default function ResetPassword() {
  const [search] = useSearchParams();
  const [form, setForm] = useState({ token: search.get("token") || "", password: "", confirmPassword: "" });
  const [errors, setErrors] = useState({});
  const [toast, setToast] = useState("");
  const [success, setSuccess] = useState(false);
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
      const res = await api.post("/auth/reset-password", form);
      setToast(res.data.message);
      setSuccess(true);
    } catch (error) {
      const data = getApiError(error);
      setErrors(data.errors || {});
      setToast(data.message || "Password reset failed.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <Container maxWidth="sm" className="page auth-page">
      <Card>
        <CardContent>
          <Stack spacing={2.5} component="form" onSubmit={submit}>
            <div className="auth-icon"><PasswordIcon /></div>
            <Typography variant="h4">Reset password</Typography>
            <TextField label="Reset token" value={form.token} onChange={(e) => update("token", e.target.value)} error={Boolean(errors.token)} helperText={errors.token} />
            <TextField label="New password" type="password" value={form.password} onChange={(e) => update("password", e.target.value)} error={Boolean(errors.password)} helperText={errors.password || "8+ chars with uppercase, lowercase, number, symbol"} />
            <TextField label="Confirm new password" type="password" value={form.confirmPassword} onChange={(e) => update("confirmPassword", e.target.value)} error={Boolean(errors.confirmPassword)} helperText={errors.confirmPassword} />
            <Button type="submit" variant="contained" size="large" disabled={busy || success}>{busy ? "Updating..." : "Reset password"}</Button>
            {success && <Alert severity="success"><MuiLink component={Link} to="/login">Login with your new password</MuiLink></Alert>}
          </Stack>
        </CardContent>
      </Card>
      <Snackbar open={Boolean(toast)} autoHideDuration={5000} onClose={() => setToast("")}>
        <Alert severity={success ? "success" : "info"}>{toast}</Alert>
      </Snackbar>
    </Container>
  );
}
