import React, { useState } from "react";
import {
  Alert,
  Box,
  Button,
  Card,
  CardContent,
  Container,
  Link as MuiLink,
  Snackbar,
  Stack,
  TextField,
  Typography
} from "@mui/material";
import MarkEmailReadIcon from "@mui/icons-material/MarkEmailRead";
import VerifiedUserIcon from "@mui/icons-material/VerifiedUser";
import { Link } from "react-router-dom";
import { api, getApiError } from "../api/http.js";
import forgotPasswordBg from "../../assets/forgot-password-bg.png";
import "../styles/auth.css";

export default function ForgotPassword() {
  const [step, setStep] = useState("email");
  const [form, setForm] = useState({ email: "", code: "", password: "", confirmPassword: "" });
  const [errors, setErrors] = useState({});
  const [toast, setToast] = useState("");
  const [devCode, setDevCode] = useState("");
  const [success, setSuccess] = useState(false);
  const [busy, setBusy] = useState(false);

  function update(key, value) {
    setForm((old) => ({ ...old, [key]: value }));
    setErrors((old) => ({ ...old, [key]: "" }));
  }

  async function requestCode(event) {
    event?.preventDefault();
    setBusy(true);
    setErrors({});
    setDevCode("");

    try {
      const res = await api.post("/auth/forgot-password", { email: form.email });
      setToast(res.data.message);
      setDevCode(res.data.verificationCode || "");
      setStep("verify");
    } catch (error) {
      const data = getApiError(error);
      setErrors(data.errors || {});
      setToast(data.message || "Could not send verification code.");
    } finally {
      setBusy(false);
    }
  }

  async function verifyAndUpdate(event) {
    event.preventDefault();
    setBusy(true);
    setErrors({});

    try {
      const res = await api.post("/auth/forgot-password/verify", form);
      setToast(res.data.message);
      setSuccess(true);
    } catch (error) {
      const data = getApiError(error);
      setErrors(data.errors || {});
      setToast(data.message || "Verification failed.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <Box
      className="forgot-password-wrapper"
      sx={{ "--forgot-bg": `url(${forgotPasswordBg})` }}
    >
      <Container maxWidth="sm" className="page auth-page">
        <Card className="forgot-password-card">
          <CardContent>
            {step === "email" ? (
              <Stack spacing={2.5} component="form" onSubmit={requestCode}>
                <div className="auth-icon">
                  <MarkEmailReadIcon />
                </div>

                <Typography variant="h4">Forgot password</Typography>

                <Typography color="text.secondary">
                  Enter your registered email. A verification code will be sent to that email before your password can be updated.
                </Typography>

                <TextField
                  label="Email"
                  value={form.email}
                  onChange={(e) => update("email", e.target.value)}
                  error={Boolean(errors.email)}
                  helperText={errors.email}
                />

                <Button type="submit" variant="contained" size="large" disabled={busy}>
                  {busy ? "Sending..." : "Send verification code"}
                </Button>

                <MuiLink component={Link} to="/login">
                  Back to login
                </MuiLink>
              </Stack>
            ) : (
              <Stack spacing={2.5} component="form" onSubmit={verifyAndUpdate}>
                <div className="auth-icon">
                  <VerifiedUserIcon />
                </div>

                <Typography variant="h4">Verify email</Typography>

                <Typography color="text.secondary">
                  Enter the 6-digit code sent to your email, then create your new password.
                </Typography>

                <TextField
                  label="Email"
                  value={form.email}
                  onChange={(e) => update("email", e.target.value)}
                  error={Boolean(errors.email)}
                  helperText={errors.email}
                  disabled={success}
                />

                <TextField
                  label="Verification code"
                  value={form.code}
                  onChange={(e) => update("code", e.target.value.replace(/\D/g, "").slice(0, 6))}
                  error={Boolean(errors.code)}
                  helperText={errors.code || "Check your email inbox for the code."}
                  disabled={success}
                />

                <TextField
                  label="New password"
                  type="password"
                  value={form.password}
                  onChange={(e) => update("password", e.target.value)}
                  error={Boolean(errors.password)}
                  helperText={errors.password || "8+ chars with uppercase, lowercase, number, symbol"}
                  disabled={success}
                />

                <TextField
                  label="Confirm new password"
                  type="password"
                  value={form.confirmPassword}
                  onChange={(e) => update("confirmPassword", e.target.value)}
                  error={Boolean(errors.confirmPassword)}
                  helperText={errors.confirmPassword}
                  disabled={success}
                />

                   {/* 
                   {devCode && !success && (
                   <Alert severity="info">
                   Development verification code: {devCode}
                   </Alert>
                   )}
                  */}

                <Button type="submit" variant="contained" size="large" disabled={busy || success}>
                  {busy ? "Updating..." : "Verify and update password"}
                </Button>

                {success ? (
                  <Alert severity="success">
                    <MuiLink component={Link} to="/login">Login with your new password</MuiLink>
                  </Alert>
                ) : (
                  <Button type="button" variant="text" onClick={requestCode} disabled={busy}>
                    Resend code
                  </Button>
                )}
              </Stack>
            )}
          </CardContent>
        </Card>

        <Snackbar open={Boolean(toast)} autoHideDuration={5000} onClose={() => setToast("")}>
          <Alert severity={success ? "success" : "info"}>{toast}</Alert>
        </Snackbar>
      </Container>
    </Box>
  );
}
