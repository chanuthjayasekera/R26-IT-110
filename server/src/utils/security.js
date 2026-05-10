import bcrypt from "bcryptjs";
import crypto from "crypto";
import jwt from "jsonwebtoken";
import { nanoid } from "nanoid";
import { config } from "../config.js";

export function createId(prefix) {
  return `${prefix}_${nanoid(18)}`;
}

export async function hashPassword(password) {
  return bcrypt.hash(password, 12);
}

export async function verifyPassword(password, hash) {
  return bcrypt.compare(password, hash);
}

export function signSession(user) {
  return jwt.sign(
    {
      sub: user.id,
      role: user.role,
      email: user.email,
      fullName: user.full_name,
      verificationStatus: user.verification_status
    },
    config.jwtSecret,
    { expiresIn: "7d" }
  );
}

export function verifySession(token) {
  return jwt.verify(token, config.jwtSecret);
}

export function createResetToken() {
  const raw = crypto.randomBytes(32).toString("hex");
  const hash = crypto.createHash("sha256").update(raw).digest("hex");
  return { raw, hash };
}

export function createVerificationCode() {
  return String(crypto.randomInt(100000, 1000000));
}

export function hashToken(raw) {
  return crypto.createHash("sha256").update(raw).digest("hex");
}

export function publicUser(user) {
  if (!user) return null;
  return {
    id: user.id,
    role: user.role,
    fullName: user.full_name,
    email: user.email,
    phone: user.phone,
    dateOfBirth: user.date_of_birth,
    gender: user.gender,
    medicalLicense: user.medical_license,
    specialization: user.specialization,
    hospital: user.hospital,
    yearsExperience: user.years_experience,
    hospitalEmail: user.hospital_email,
    licenseProofName: user.license_proof_name,
    profileImage: user.profile_image,
    isVerified: Boolean(user.is_verified),
    verificationStatus: user.verification_status,
    verificationMessage: user.verification_message,
    createdAt: user.created_at
  };
}

export function adminUserView(user) {
  return {
    ...publicUser(user),
    licenseProofData: user.license_proof_data
  };
}
