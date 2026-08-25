import express from "express";
import { db } from "../db.js";
import { requireAuth, requireRole, requireVerifiedProfessional } from "../middleware/auth.js";

export const dashboardRouter = express.Router();

dashboardRouter.get("/patient", requireAuth, requireRole("patient"), (req, res) => {
  res.json({
    message: "Patient dashboard loaded.",
    modules: [
      { title: "Normal vs Abnormal Detection", status: "Ready", description: "Upload gait videos for first-stage screening, biometrics, and severity estimation." },
      { title: "SCA and KOA Detection", status: "Ready", description: "Review Spinocerebellar Ataxia, Knee Osteoarthritis, and joint instability analysis." },
      { title: "PD & Neuropathy Detection", status: "Ready", description: "Access PD and neuropathy gait screening with comparison profile views." },
      { title: "Exercise Quality Analysis", status: "Ready", description: "Upload rehabilitation exercise videos for pose correctness checks, timed posture windows, and report export." }
    ]
  });
});

dashboardRouter.get("/professional", requireAuth, requireRole("professional"), requireVerifiedProfessional, (req, res) => {
  const patients = db.prepare(`
    SELECT u.id, u.full_name, u.email, MAX(cpf.updated_at) AS created_at, COUNT(DISTINCT cpf.source_type) AS completed_slots
    FROM users u
    JOIN central_profile_flags cpf ON cpf.user_id = u.id
    WHERE u.role = 'patient'
      AND cpf.source_type IN (
        'normal_abnormal', 'sca', 'koa', 'pd', 'neuropathy',
        'exercise_gesture2', 'exercise_gesture3', 'exercise_gesture5'
      )
    GROUP BY u.id
    ORDER BY MAX(cpf.updated_at) DESC
    LIMIT 8
  `).all();
  res.json({
    message: "Medical professional dashboard loaded.",
    modules: [
      { title: "Upload Risks", status: "Ready", description: "Create dynamic risk profiles by disorder, abnormality type, and severity." },
      { title: "Upload Recommendations", status: "Ready", description: "Maintain severity-aware rehabilitation recommendations and guidance." },
      { title: "Upload Rehabilitation Plans", status: "Ready", description: "Manage condition-specific exercise plans and safety instructions." },
      { title: "Engage Patient Profiles", status: "Ready", description: "Review central profiles, monitor progress, and support follow-up." }
    ],
    patients
  });
});

dashboardRouter.get("/admin", requireAuth, requireRole("admin"), (req, res) => {
  const patients = db.prepare(`
    SELECT id, full_name, email, phone, created_at
    FROM users
    WHERE role = 'patient'
    ORDER BY created_at DESC
  `).all();

  const professionals = db.prepare(`
    SELECT id, full_name, email, phone, medical_license, specialization, hospital, hospital_email,
           years_experience, license_proof_name, license_proof_data, verification_status,
           verification_message, created_at
    FROM users
    WHERE role = 'professional'
    ORDER BY created_at DESC
  `).all();

  res.json({
    message: "Admin dashboard loaded.",
    patients,
    professionals
  });
});

dashboardRouter.patch("/admin/professionals/:id/status", requireAuth, requireRole("admin"), (req, res) => {
  const { id } = req.params;
  const status = String(req.body.status || "").trim();
  const message = String(req.body.message || "").trim();

  if (!["approved", "rejected", "pending"].includes(status)) {
    return res.status(422).json({ message: "Invalid approval status." });
  }

  const professional = db.prepare("SELECT * FROM users WHERE id = ? AND role = 'professional'").get(id);
  if (!professional) return res.status(404).json({ message: "Medical professional not found." });

  db.prepare(`
    UPDATE users
    SET verification_status = ?, is_verified = ?, verification_message = ?, updated_at = ?
    WHERE id = ?
  `).run(
    status,
    status === "approved" ? 1 : 0,
    message || (status === "approved" ? "Approved by admin." : status === "rejected" ? "Rejected by admin." : "Pending admin review."),
    new Date().toISOString(),
    id
  );

  res.json({ message: `Medical professional ${status}.` });
});
