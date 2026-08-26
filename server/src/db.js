import { AsyncLocalStorage } from "async_hooks";
import pg from "pg";
import { config } from "./config.js";
import { createId, hashPassword } from "./utils/security.js";

const { Pool } = pg;

if (!config.databaseUrl) {
  throw new Error("DATABASE_URL is required. Add your PostgreSQL connection string to server/.env.");
}

const usesSsl = /sslmode=require/i.test(config.databaseUrl) || /\.neon\.tech/i.test(config.databaseUrl);
const pool = new Pool({
  connectionString: config.databaseUrl,
  ssl: usesSsl ? { rejectUnauthorized: false } : undefined
});

const transactionStore = new AsyncLocalStorage();

function activeClient() {
  return transactionStore.getStore() || pool;
}

function toPostgresSql(sql) {
  let parameterIndex = 0;
  return String(sql).replace(/\?/g, () => `$${++parameterIndex}`);
}

async function query(sql, params = []) {
  const values = Array.isArray(params) ? params : [params];
  return activeClient().query(toPostgresSql(sql), values);
}

export const db = {
  async exec(sql) {
    return activeClient().query(sql);
  },

  async get(sql, ...params) {
    const result = await query(sql, params);
    return result.rows[0];
  },

  async all(sql, ...params) {
    const result = await query(sql, params);
    return result.rows;
  },

  async run(sql, ...params) {
    const result = await query(sql, params);
    return { changes: result.rowCount };
  },

  async transaction(callback) {
    const client = await pool.connect();
    try {
      await client.query("BEGIN");
      const result = await transactionStore.run(client, callback);
      await client.query("COMMIT");
      return result;
    } catch (error) {
      await client.query("ROLLBACK");
      throw error;
    } finally {
      client.release();
    }
  }
};

async function addColumn(table, column, definition) {
  await db.exec(`ALTER TABLE ${table} ADD COLUMN IF NOT EXISTS ${column} ${definition}`);
}

export async function migrate() {
  await db.exec(`
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

    CREATE TABLE IF NOT EXISTS normal_abnormal_screenings (
      id TEXT PRIMARY KEY,
      user_id TEXT NOT NULL,
      clinical_profile_id TEXT,
      input_type TEXT NOT NULL,
      file_name TEXT NOT NULL,
      file_path TEXT NOT NULL,
      csv_path TEXT,
      direction TEXT,
      fps_used REAL,
      final_label INTEGER,
      final_result TEXT,
      model_suggested_label INTEGER,
      model_suggested_result TEXT,
      mean_prob_abnormal REAL,
      confidence_percent REAL,
      abnormal_ratio_threshold REAL,
      screening_severity TEXT,
      reliability_level TEXT,
      reliability_reasons TEXT,
      clinical_note TEXT,
      raw_result_json TEXT NOT NULL,
      created_at TEXT NOT NULL,
      FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE,
      FOREIGN KEY(clinical_profile_id) REFERENCES clinical_profiles(id) ON DELETE SET NULL
    );

    CREATE TABLE IF NOT EXISTS sca_koa_screenings (
      id TEXT PRIMARY KEY,
      user_id TEXT NOT NULL,
      clinical_profile_id TEXT,
      model_key TEXT NOT NULL CHECK(model_key IN ('sca','koa')),
      input_type TEXT NOT NULL,
      file_name TEXT NOT NULL,
      file_path TEXT NOT NULL,
      csv_path TEXT,
      direction TEXT,
      fps_used REAL,
      final_result TEXT,
      detected INTEGER NOT NULL DEFAULT 0,
      tendency INTEGER NOT NULL DEFAULT 0,
      probability REAL,
      max_probability REAL,
      positive_window_count REAL,
      positive_window_ratio REAL,
      pattern_strength TEXT,
      reliability_level TEXT,
      reliability_reasons TEXT,
      clinical_note TEXT,
      instability_json TEXT,
      raw_result_json TEXT NOT NULL,
      created_at TEXT NOT NULL,
      FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE,
      FOREIGN KEY(clinical_profile_id) REFERENCES clinical_profiles(id) ON DELETE SET NULL
    );

    CREATE TABLE IF NOT EXISTS sca_genetic_awareness (
      id TEXT PRIMARY KEY,
      user_id TEXT NOT NULL,
      screening_id TEXT NOT NULL UNIQUE,
      answers_json TEXT NOT NULL,
      relatives_json TEXT NOT NULL,
      suspected_json TEXT NOT NULL,
      awareness_json TEXT NOT NULL,
      created_at TEXT NOT NULL,
      updated_at TEXT NOT NULL,
      FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE,
      FOREIGN KEY(screening_id) REFERENCES sca_koa_screenings(id) ON DELETE CASCADE
    );

    CREATE TABLE IF NOT EXISTS pd_neuropathy_screenings (
      id TEXT PRIMARY KEY,
      user_id TEXT NOT NULL,
      clinical_profile_id TEXT,
      model_key TEXT NOT NULL CHECK(model_key IN ('pd','neuropathy')),
      input_type TEXT NOT NULL,
      file_name TEXT NOT NULL,
      file_path TEXT NOT NULL,
      csv_path TEXT,
      direction TEXT,
      fps_used REAL,
      final_result TEXT,
      detected INTEGER NOT NULL DEFAULT 0,
      tendency INTEGER NOT NULL DEFAULT 0,
      probability REAL,
      max_probability REAL,
      positive_window_count REAL,
      positive_window_ratio REAL,
      reliability_level TEXT,
      reliability_reasons TEXT,
      clinical_note TEXT,
      raw_result_json TEXT NOT NULL,
      created_at TEXT NOT NULL,
      FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE,
      FOREIGN KEY(clinical_profile_id) REFERENCES clinical_profiles(id) ON DELETE SET NULL
    );

    CREATE TABLE IF NOT EXISTS exercise_screenings (
      id TEXT PRIMARY KEY,
      user_id TEXT NOT NULL,
      clinical_profile_id TEXT,
      exercise_key TEXT NOT NULL CHECK(exercise_key IN ('gesture2','gesture3','gesture5')),
      exercise_label TEXT NOT NULL,
      file_name TEXT NOT NULL,
      file_path TEXT NOT NULL,
      annotated_path TEXT,
      final_prediction TEXT,
      final_label INTEGER,
      quality_score REAL,
      mean_correct_probability REAL,
      decision_threshold REAL,
      correct_ratio REAL,
      incorrect_ratio REAL,
      num_windows INTEGER,
      valid_pose_frames INTEGER,
      reliability_level TEXT,
      reliability_reasons TEXT,
      window_report_json TEXT NOT NULL,
      raw_result_json TEXT NOT NULL,
      created_at TEXT NOT NULL,
      FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE,
      FOREIGN KEY(clinical_profile_id) REFERENCES clinical_profiles(id) ON DELETE SET NULL
    );

    CREATE TABLE IF NOT EXISTS central_profile_flags (
      id TEXT PRIMARY KEY,
      user_id TEXT NOT NULL,
      source_type TEXT NOT NULL,
      screening_id TEXT NOT NULL,
      snapshot_json TEXT NOT NULL,
      risk_json TEXT,
      rehab_json TEXT,
      flagged_at TEXT NOT NULL,
      created_at TEXT NOT NULL,
      updated_at TEXT NOT NULL,
      UNIQUE(user_id, source_type),
      FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
    );

    CREATE INDEX IF NOT EXISTS idx_central_profile_flags_user
      ON central_profile_flags(user_id, updated_at DESC);

    CREATE TABLE IF NOT EXISTS central_profile_guidance (
      id TEXT PRIMARY KEY,
      patient_user_id TEXT NOT NULL,
      professional_user_id TEXT NOT NULL,
      guidance_type TEXT NOT NULL CHECK(guidance_type IN ('risk','rehab')),
      title TEXT NOT NULL,
      source_type TEXT,
      disease_focus TEXT,
      priority TEXT,
      payload_json TEXT NOT NULL,
      attachment_name TEXT,
      attachment_path TEXT,
      patient_viewed_at TEXT,
      created_at TEXT NOT NULL,
      updated_at TEXT NOT NULL,
      FOREIGN KEY(patient_user_id) REFERENCES users(id) ON DELETE CASCADE,
      FOREIGN KEY(professional_user_id) REFERENCES users(id) ON DELETE CASCADE
    );

    CREATE INDEX IF NOT EXISTS idx_central_profile_guidance_patient
      ON central_profile_guidance(patient_user_id, guidance_type, created_at DESC);
  `);

  await addColumn("users", "verification_status", "TEXT NOT NULL DEFAULT 'approved'");
  await addColumn("users", "verification_message", "TEXT");
  await addColumn("users", "hospital_email", "TEXT");
  await addColumn("users", "license_proof_name", "TEXT");
  await addColumn("users", "license_proof_data", "TEXT");
  await addColumn("users", "profile_image", "TEXT");
  await addColumn("sca_koa_screenings", "instability_json", "TEXT");
  await addColumn("central_profile_flags", "risk_json", "TEXT");
  await addColumn("central_profile_flags", "rehab_json", "TEXT");

  await db.run("DELETE FROM users WHERE email = ?", "admin@gaitai.local");

  const now = new Date().toISOString();
  const adminEmail = "admingaitailocal@gmail.com";
  const existingAdmin = await db.get("SELECT id FROM users WHERE email = ?", adminEmail);
  if (!existingAdmin) {
    const passwordHash = await hashPassword("Admin@123!");
    await db.run(`
      INSERT INTO users (
        id, role, full_name, email, password_hash, phone, is_verified,
        verification_status, created_at, updated_at
      ) VALUES (?, 'admin', 'System Admin', ?, ?, ?, 1, 'approved', ?, ?)
    `, createId("adm"), adminEmail, passwordHash, "+0000000000", now, now);
  }
}
