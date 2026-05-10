import { config } from "../config.js";
import { db } from "../db.js";
import { verifySession } from "../utils/security.js";

export function requireAuth(req, res, next) {
  try {
    const token = req.cookies[config.cookieName];
    if (!token) return res.status(401).json({ message: "Please login to continue." });
    const payload = verifySession(token);
    const user = db.prepare("SELECT * FROM users WHERE id = ?").get(payload.sub);
    if (!user) return res.status(401).json({ message: "Session user was not found." });
    req.user = user;
    next();
  } catch {
    return res.status(401).json({ message: "Your session expired. Please login again." });
  }
}

export function requireRole(role) {
  return (req, res, next) => {
    if (!req.user || req.user.role !== role) {
      return res.status(403).json({ message: "You do not have permission to access this area." });
    }
    next();
  };
}

export function requireVerifiedProfessional(req, res, next) {
  if (req.user.role === "professional" && req.user.verification_status !== "approved") {
    return res.status(403).json({
      message: req.user.verification_status === "rejected"
        ? "Your medical professional account was rejected. Please update your details or contact admin."
        : "Your medical professional account is pending admin approval.",
      status: req.user.verification_status,
      user: {
        id: req.user.id,
        role: req.user.role,
        fullName: req.user.full_name,
        verificationStatus: req.user.verification_status,
        verificationMessage: req.user.verification_message
      }
    });
  }
  next();
}
