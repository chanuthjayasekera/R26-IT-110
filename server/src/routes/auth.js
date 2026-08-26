import express from "express";
import { db } from "../db.js";
import { config } from "../config.js";
import { createId, createResetToken, createVerificationCode, hashPassword, hashToken, publicUser, signSession, verifyPassword } from "../utils/security.js";
import { validateForgot, validateForgotVerification, validateLogin, validateProfile, validateRegister, validateReset } from "../utils/validators.js";
import { sendPasswordVerificationEmail } from "../utils/mailer.js";
import { requireAuth } from "../middleware/auth.js";

export const authRouter = express.Router();

function setSessionCookie(res, token) {
  res.cookie(config.cookieName, token, {
    httpOnly: true,
    sameSite: "lax",
    secure: config.nodeEnv === "production",
    maxAge: 7 * 24 * 60 * 60 * 1000
  });
}

function clearSessionCookie(res) {
  res.clearCookie(config.cookieName, {
    httpOnly: true,
    sameSite: "lax",
    secure: config.nodeEnv === "production"
  });
}

async function audit(userId, eventType, message) {
  await db.run("INSERT INTO audit_events (id, user_id, event_type, message, created_at) VALUES (?, ?, ?, ?, ?)", createId("evt"),
    userId,
    eventType,
    message,
    new Date().toISOString());
}

authRouter.post("/register", async (req, res) => {
  const parsed = validateRegister(req.body);
  if (!parsed.valid) return res.status(422).json({ message: "Please fix the highlighted fields.", errors: parsed.errors });

  const existing = await db.get("SELECT id FROM users WHERE email = ?", parsed.data.email);
  if (existing) return res.status(409).json({ message: "This email is already registered.", errors: { email: "Email already exists." } });

  const now = new Date().toISOString();
  const userId = createId("usr");
  const passwordHash = await hashPassword(req.body.password);
  const isProfessional = parsed.data.role === "professional";
  const verificationStatus = isProfessional ? "pending" : "approved";
  const isVerified = isProfessional ? 0 : 1;

  await db.run(`
    INSERT INTO users (
      id, role, full_name, email, password_hash, phone, date_of_birth, gender,
      medical_license, specialization, hospital, years_experience,
      is_verified, verification_status, verification_message,
      hospital_email, license_proof_name, license_proof_data,
      created_at, updated_at
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
  `, userId,
    parsed.data.role,
    parsed.data.fullName,
    parsed.data.email,
    passwordHash,
    req.body.phone || null,
    req.body.dateOfBirth || null,
    req.body.gender || null,
    req.body.medicalLicense || null,
    req.body.specialization || null,
    req.body.hospital || null,
    req.body.yearsExperience === "" || req.body.yearsExperience === undefined ? null : Number(req.body.yearsExperience),
    isVerified,
    verificationStatus,
    isProfessional ? "Submitted for admin verification." : null,
    req.body.hospitalEmail || null,
    req.body.licenseProofName || null,
    req.body.licenseProofData || null,
    now,
    now);

  const user = await db.get("SELECT * FROM users WHERE id = ?", userId);
  const token = signSession(user);
  setSessionCookie(res, token);
  await audit(userId, "register", `${user.role} account created with status ${verificationStatus}`);
  return res.status(201).json({
    message: isProfessional
      ? "Registration submitted. Admin approval is required before dashboard access."
      : "Registration successful.",
    user: publicUser(user)
  });
});

authRouter.post("/login", async (req, res) => {
  const parsed = validateLogin(req.body);
  if (!parsed.valid) return res.status(422).json({ message: "Please fix the highlighted fields.", errors: parsed.errors });

  const user = await db.get("SELECT * FROM users WHERE email = ?", parsed.data.email);
  if (!user) return res.status(401).json({ message: "Invalid email or password." });

  const ok = await verifyPassword(parsed.data.password, user.password_hash);
  if (!ok) return res.status(401).json({ message: "Invalid email or password." });

  const token = signSession(user);
  setSessionCookie(res, token);
  await audit(user.id, "login", `User logged in with status ${user.verification_status}`);
  return res.json({ message: "Login successful.", user: publicUser(user) });
});

authRouter.post("/logout", requireAuth, async (req, res) => {
  await audit(req.user.id, "logout", "User logged out");
  clearSessionCookie(res);
  return res.json({ message: "Logout successful." });
});

authRouter.get("/me", requireAuth, async (req, res) => {
  return res.json({ user: publicUser(req.user) });
});

authRouter.put("/profile", requireAuth, async (req, res) => {
  const parsed = validateProfile(req.body, req.user);
  if (!parsed.valid) return res.status(422).json({ message: "Please fix the highlighted fields.", errors: parsed.errors });

  const now = new Date().toISOString();
  await db.run(`
    UPDATE users SET
      full_name = ?,
      phone = ?,
      date_of_birth = ?,
      gender = ?,
      specialization = ?,
      hospital = ?,
      profile_image = ?,
      updated_at = ?
    WHERE id = ?
  `, String(req.body.fullName || "").trim(),
    req.body.phone || null,
    req.user.role === "patient" ? req.body.dateOfBirth || null : req.user.date_of_birth,
    req.user.role === "patient" ? req.body.gender || null : req.user.gender,
    req.user.role === "professional" ? req.body.specialization || null : req.user.specialization,
    req.user.role === "professional" ? req.body.hospital || null : req.user.hospital,
    req.body.profileImage || req.user.profile_image || null,
    now,
    req.user.id);

  const updated = await db.get("SELECT * FROM users WHERE id = ?", req.user.id);
  await audit(req.user.id, "profile_update", "User profile updated");
  return res.json({ message: "Profile updated successfully.", user: publicUser(updated) });
});

authRouter.post("/forgot-password", async (req, res) => {
  const parsed = validateForgot(req.body);
  if (!parsed.valid) return res.status(422).json({ message: "Please fix the highlighted fields.", errors: parsed.errors });

  const user = await db.get("SELECT * FROM users WHERE email = ?", parsed.data.email);
  if (!user) return res.json({ message: "If that email is registered, a verification code has been sent." });

  await db.run("UPDATE password_reset_tokens SET used_at = ? WHERE user_id = ? AND used_at IS NULL", new Date().toISOString(), user.id);

  const code = createVerificationCode();
  const expiresAt = new Date(Date.now() + 15 * 60 * 1000).toISOString();
  await db.run("INSERT INTO password_reset_tokens (id, user_id, token_hash, expires_at, created_at) VALUES (?, ?, ?, ?, ?)", createId("rst"),
    user.id,
    hashToken(code),
    expiresAt,
    new Date().toISOString());

  await sendPasswordVerificationEmail(user.email, code);
  await audit(user.id, "forgot_password", "Password verification code requested");

  return res.json({
    message: "If that email is registered, a verification code has been sent.",
    verificationCode: config.nodeEnv === "development" ? code : undefined
  });
});

authRouter.post("/forgot-password/verify", async (req, res) => {
  const parsed = validateForgotVerification(req.body);
  if (!parsed.valid) return res.status(422).json({ message: "Please fix the highlighted fields.", errors: parsed.errors });

  const user = await db.get("SELECT * FROM users WHERE email = ?", parsed.data.email);
  if (!user) return res.status(400).json({ message: "Verification failed. Please request a new code." });

  const tokenHash = hashToken(parsed.data.code);
  const record = await db.get(`
    SELECT *
    FROM password_reset_tokens
    WHERE user_id = ? AND token_hash = ?
    ORDER BY created_at DESC
    LIMIT 1
  `, user.id, tokenHash);

  if (!record || record.used_at) return res.status(400).json({ message: "Verification code is invalid or already used." });
  if (new Date(record.expires_at).getTime() < Date.now()) return res.status(400).json({ message: "Verification code expired. Please request another code." });

  const passwordHash = await hashPassword(parsed.data.password);
  await db.run("UPDATE users SET password_hash = ?, updated_at = ? WHERE id = ?", passwordHash, new Date().toISOString(), user.id);
  await db.run("UPDATE password_reset_tokens SET used_at = ? WHERE id = ?", new Date().toISOString(), record.id);
  await audit(user.id, "reset_password", "Password reset completed after email verification");

  return res.json({ message: "Email verified and password updated successfully." });
});

authRouter.post("/reset-password", async (req, res) => {
  const parsed = validateReset(req.body);
  if (!parsed.valid) return res.status(422).json({ message: "Please fix the highlighted fields.", errors: parsed.errors });

  const tokenHash = hashToken(parsed.data.token);
  const record = await db.get(`
    SELECT prt.*, u.email
    FROM password_reset_tokens prt
    JOIN users u ON u.id = prt.user_id
    WHERE prt.token_hash = ?
  `, tokenHash);

  if (!record || record.used_at) return res.status(400).json({ message: "Reset link is invalid or already used." });
  if (new Date(record.expires_at).getTime() < Date.now()) return res.status(400).json({ message: "Reset link expired. Please request another link." });

  const passwordHash = await hashPassword(parsed.data.password);
  await db.run("UPDATE users SET password_hash = ?, updated_at = ? WHERE id = ?", passwordHash, new Date().toISOString(), record.user_id);
  await db.run("UPDATE password_reset_tokens SET used_at = ? WHERE id = ?", new Date().toISOString(), record.id);
  await audit(record.user_id, "reset_password", "Password reset completed");
  return res.json({ message: "Password reset successful. You can login now." });
});
