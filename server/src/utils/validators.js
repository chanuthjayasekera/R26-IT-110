import validator from "validator";

export const roles = new Set(["patient", "professional", "admin"]);

const publicEmailDomains = new Set([
  "gmail.com",
  "yahoo.com",
  "outlook.com",
  "hotmail.com",
  "icloud.com",
  "proton.me",
  "protonmail.com",
  "aol.com",
  "live.com",
  "msn.com"
]);

export function strongPassword(password) {
  return typeof password === "string" &&
    password.length >= 8 &&
    /[a-z]/.test(password) &&
    /[A-Z]/.test(password) &&
    /\d/.test(password) &&
    /[^A-Za-z0-9]/.test(password);
}

export function isLikelyHospitalEmail(email) {
  if (!validator.isEmail(email || "")) return false;
  const domain = String(email).split("@")[1]?.toLowerCase();
  if (!domain || publicEmailDomains.has(domain)) return false;
  return domain.includes("hospital") ||
    domain.includes("clinic") ||
    domain.includes("health") ||
    domain.includes("medical") ||
    domain.includes("med") ||
    domain.includes("care") ||
    domain.includes("nhs") ||
    domain.includes("edu") ||
    domain.includes("ac.");
}

export function validateRegister(body) {
  const errors = {};
  const role = String(body.role || "").trim();
  const fullName = String(body.fullName || "").trim();
  const email = String(body.email || "").trim().toLowerCase();
  const password = String(body.password || "");
  const confirmPassword = String(body.confirmPassword || "");

  if (!["patient", "professional"].includes(role)) errors.role = "Choose Patient or Medical Professional.";
  if (fullName.length < 3) errors.fullName = "Full name must be at least 3 characters.";
  if (!validator.isEmail(email)) errors.email = "Enter a valid email address.";
  if (!strongPassword(password)) errors.password = "Password needs 8+ chars, uppercase, lowercase, number, and symbol.";
  if (password !== confirmPassword) errors.confirmPassword = "Passwords do not match.";

  if (role === "patient") {
    if (!body.dateOfBirth) errors.dateOfBirth = "Date of birth is required.";
    if (!body.gender) errors.gender = "Gender is required.";
  }

  if (role === "professional") {
    const license = String(body.medicalLicense || "").trim();
    const specialization = String(body.specialization || "").trim();
    const hospital = String(body.hospital || "").trim();
    const years = Number(body.yearsExperience);
    const hospitalEmail = String(body.hospitalEmail || "").trim().toLowerCase();
    const licenseProofData = String(body.licenseProofData || "").trim();
    const licenseProofName = String(body.licenseProofName || "").trim();

    if (license.length < 5) errors.medicalLicense = "Medical license number is required.";
    if (specialization.length < 2) errors.specialization = "Specialization is required.";
    if (hospital.length < 2) errors.hospital = "Hospital or clinic name is required.";
    if (!Number.isInteger(years) || years < 0 || years > 70) errors.yearsExperience = "Enter valid years of experience.";
    if (!isLikelyHospitalEmail(hospitalEmail)) errors.hospitalEmail = "Use a valid hospital, clinic, university, or healthcare organization email.";
    if (!licenseProofName || !licenseProofData.startsWith("data:")) errors.licenseProofData = "Upload a license proof document or image.";
  }

  return { valid: Object.keys(errors).length === 0, errors, data: { role, fullName, email } };
}

export function validateLogin(body) {
  const email = String(body.email || "").trim().toLowerCase();
  const password = String(body.password || "");
  const errors = {};
  if (!validator.isEmail(email)) errors.email = "Enter a valid email.";
  if (!password) errors.password = "Password is required.";
  return { valid: Object.keys(errors).length === 0, errors, data: { email, password } };
}

export function validateForgot(body) {
  const email = String(body.email || "").trim().toLowerCase();
  const errors = {};
  if (!validator.isEmail(email)) errors.email = "Enter a valid email.";
  return { valid: Object.keys(errors).length === 0, errors, data: { email } };
}



export function validateForgotVerification(body) {
  const email = String(body.email || "").trim().toLowerCase();
  const code = String(body.code || "").trim();
  const password = String(body.password || "");
  const confirmPassword = String(body.confirmPassword || "");
  const errors = {};
  if (!validator.isEmail(email)) errors.email = "Enter a valid email.";
  if (!/^\d{6}$/.test(code)) errors.code = "Enter the 6-digit verification code.";
  if (!strongPassword(password)) errors.password = "Password needs 8+ chars, uppercase, lowercase, number, and symbol.";
  if (password !== confirmPassword) errors.confirmPassword = "Passwords do not match.";
  return { valid: Object.keys(errors).length === 0, errors, data: { email, code, password } };
}

export function validateReset(body) {
  const token = String(body.token || "").trim();
  const password = String(body.password || "");
  const confirmPassword = String(body.confirmPassword || "");
  const errors = {};
  if (token.length < 20) errors.token = "Reset token is invalid.";
  if (!strongPassword(password)) errors.password = "Password needs 8+ chars, uppercase, lowercase, number, and symbol.";
  if (password !== confirmPassword) errors.confirmPassword = "Passwords do not match.";
  return { valid: Object.keys(errors).length === 0, errors, data: { token, password } };
}

export function validateProfile(body, user) {
  const errors = {};
  const fullName = String(body.fullName || "").trim();
  const phone = String(body.phone || "").trim();
  if (fullName.length < 3) errors.fullName = "Full name must be at least 3 characters.";
  if (phone && !/^[+0-9\s()-]{7,20}$/.test(phone)) errors.phone = "Enter a valid phone number.";

  if (user.role === "patient") {
    if (!body.dateOfBirth) errors.dateOfBirth = "Date of birth is required.";
    if (!body.gender) errors.gender = "Gender is required.";
  }

  if (user.role === "professional") {
    if (String(body.specialization || "").trim().length < 2) errors.specialization = "Specialization is required.";
    if (String(body.hospital || "").trim().length < 2) errors.hospital = "Hospital or clinic name is required.";
  }

  return { valid: Object.keys(errors).length === 0, errors };
}
