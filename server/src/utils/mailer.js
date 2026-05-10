import nodemailer from "nodemailer";
import { config } from "../config.js";

let cachedTransporter = null;

function getTransporter() {
  if (!config.smtp.host || !config.smtp.user || !config.smtp.pass) return null;
  if (!cachedTransporter) {
    cachedTransporter = nodemailer.createTransport({
      host: config.smtp.host,
      port: config.smtp.port,
      secure: config.smtp.secure,
      auth: {
        user: config.smtp.user,
        pass: config.smtp.pass
      }
    });
  }
  return cachedTransporter;
}

export async function sendPasswordVerificationEmail(email, code) {
  const transporter = getTransporter();
  const subject = "Gait AI Care password verification code";
  const text = `Your Gait AI Care password verification code is ${code}. This code expires in 15 minutes.`;
  const html = `
    <div style="font-family:Arial,sans-serif;line-height:1.6;color:#172033">
      <h2>Gait AI Care password verification</h2>
      <p>Use this code to verify your email and reset your password:</p>
      <div style="font-size:28px;font-weight:800;letter-spacing:6px;padding:14px 18px;border-radius:12px;background:#eef5ff;display:inline-block">${code}</div>
      <p>This code expires in 15 minutes.</p>
      <p>If you did not request this, you can ignore this email.</p>
    </div>
  `;

  if (!transporter) {
    console.log(`[Gait AI Care] Password verification code for ${email}: ${code}`);
    return { delivered: false };
  }

  await transporter.sendMail({
    from: config.smtp.from,
    to: email,
    subject,
    text,
    html
  });

  return { delivered: true };
}
