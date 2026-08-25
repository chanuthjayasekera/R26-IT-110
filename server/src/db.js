/* file: server/src/db.js */

import Database from "better-sqlite3";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import { hashPassword } from "./utils/security.js";
import { createId } from "./utils/security.js";

const dirname = path.dirname(fileURLToPath(import.meta.url));
const dataDir = path.join(dirname, "..", "data");
if (!fs.existsSync(dataDir)) fs.mkdirSync(dataDir, { recursive: true });
const dbPath = path.join(dataDir, "app.db");

export const db = new Database(dbPath);
db.pragma("journal_mode = WAL");
db.pragma("foreign_keys = ON");

function columnExists(table, column) {
  const rows = db.prepare(`PRAGMA table_info(${table})`).all();
  return rows.some((row) => row.name === column);
}

function addColumn(table, column, definition) {
  if (!columnExists(table, column)) {
    db.exec(`ALTER TABLE ${table} ADD COLUMN ${column} ${definition}`);
  }
}

export async function migrate() {
  db.exec(`
    CREATE TABLE IF NOT EXISTS users (
      id TEXT PRIMARY KEY,
      role TEXT NOT NULL CHECK(role IN ('patient','professional','admin')),
      full_name TEXT NOT NULL,
      email TEXT NOT NULL UNIQUE,
      password_hash TEXT NOT NULL,
      phone TEXT,
      date_of_birth TEXT,
      gender TEXT,
      medical_license TEXT,
      specialization TEXT,
      hospital TEXT,
      years_experience INTEGER,
      family_sca_history TEXT,
      relatives_abnormal_gait TEXT,
      is_verified INTEGER NOT NULL DEFAULT 1,
      verification_status TEXT NOT NULL DEFAULT 'approved',
      verification_message TEXT,
      hospital_email TEXT,
      license_proof_name TEXT,
      license_proof_data TEXT,
      profile_image TEXT,
      created_at TEXT NOT NULL,
      updated_at TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS password_reset_tokens (
      id TEXT PRIMARY KEY,
      user_id TEXT NOT NULL,
      token_hash TEXT NOT NULL,
      expires_at TEXT NOT NULL,
      used_at TEXT,
      created_at TEXT NOT NULL,
      FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
    );

    CREATE TABLE IF NOT EXISTS audit_events (
      id TEXT PRIMARY KEY,
      user_id TEXT,
      event_type TEXT NOT NULL,
      message TEXT NOT NULL,
      created_at TEXT NOT NULL,
      FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE SET NULL
    );

    CREATE TABLE IF NOT EXISTS clinical_profiles (
      id TEXT PRIMARY KEY,
      user_id TEXT NOT NULL UNIQUE,
      height_cm REAL,
      weight_kg REAL,
      walking_aid TEXT,
      fall_history TEXT,
      primary_symptoms TEXT,
      clinical_notes TEXT,
      created_at TEXT NOT NULL,
      updated_at TEXT NOT NULL,
      FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
    );

    CREATE TABLE IF NOT EXISTS central_profile_flags (
      id TEXT PRIMARY KEY,
      user_id TEXT NOT NULL,
      source_type TEXT NOT NULL,
      screening_id TEXT NOT NULL,
      snapshot_json TEXT NOT NULL,
      flagged_at TEXT NOT NULL,
      created_at TEXT NOT NULL,
      updated_at TEXT NOT NULL,
      UNIQUE(user_id, source_type),
      FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
    );

    CREATE INDEX IF NOT EXISTS idx_central_profile_flags_user
      ON central_profile_flags(user_id, updated_at DESC);
  `);

  addColumn("users", "verification_status", "TEXT NOT NULL DEFAULT 'approved'");
  addColumn("users", "verification_message", "TEXT");
  addColumn("users", "hospital_email", "TEXT");
  addColumn("users", "license_proof_name", "TEXT");
  addColumn("users", "license_proof_data", "TEXT");
  addColumn("users", "profile_image", "TEXT");

  db.prepare("DELETE FROM users WHERE email = ?").run("admin@gaitai.local");

  const now = new Date().toISOString();
  const adminEmail = "admingaitailocal@gmail.com";
  const existingAdmin = db.prepare("SELECT id FROM users WHERE email = ?").get(adminEmail);
  if (!existingAdmin) {
    const passwordHash = await hashPassword("Admin@123!");
    db.prepare(`
      INSERT INTO users (
        id, role, full_name, email, password_hash, phone, is_verified,
        verification_status, created_at, updated_at
      ) VALUES (?, 'admin', 'System Admin', ?, ?, ?, 1, 'approved', ?, ?)
    `).run(createId("adm"), adminEmail, passwordHash, "+0000000000", now, now);
  }
}
