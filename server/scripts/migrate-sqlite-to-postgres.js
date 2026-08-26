import Database from "better-sqlite3";
import dotenv from "dotenv";
import fs from "fs";
import path from "path";
import pg from "pg";
import { fileURLToPath } from "url";

dotenv.config();

const { Pool } = pg;
const dirname = path.dirname(fileURLToPath(import.meta.url));
const serverRoot = path.resolve(dirname, "..");
const sqlitePath = process.env.SQLITE_DB_PATH || path.join(serverRoot, "data", "app.db");

if (!process.env.DATABASE_URL) {
  throw new Error("DATABASE_URL is required in server/.env before migrating data.");
}

if (!fs.existsSync(sqlitePath)) {
  throw new Error(`SQLite database was not found: ${sqlitePath}`);
}

const usesSsl = /sslmode=require/i.test(process.env.DATABASE_URL) || /\.neon\.tech/i.test(process.env.DATABASE_URL);
const pool = new Pool({
  connectionString: process.env.DATABASE_URL,
  ssl: usesSsl ? { rejectUnauthorized: false } : undefined
});

const sqlite = new Database(sqlitePath, { readonly: true });

const tables = [
  "users",
  "password_reset_tokens",
  "audit_events",
  "clinical_profiles",
  "normal_abnormal_screenings",
  "sca_koa_screenings",
  "sca_genetic_awareness",
  "pd_neuropathy_screenings",
  "exercise_screenings",
  "central_profile_flags",
  "central_profile_guidance"
];

function sqliteColumns(table) {
  return sqlite.prepare(`PRAGMA table_info(${table})`).all().map((column) => column.name);
}

function placeholders(count) {
  return Array.from({ length: count }, (_, index) => `$${index + 1}`).join(", ");
}

async function insertRows(client, table, rows) {
  if (rows.length === 0) return 0;

  const columns = sqliteColumns(table);
  const quotedColumns = columns.map((column) => `"${column}"`).join(", ");
  const sql = `
    INSERT INTO ${table} (${quotedColumns})
    VALUES (${placeholders(columns.length)})
    ON CONFLICT DO NOTHING
  `;

  for (const row of rows) {
    await client.query(sql, columns.map((column) => row[column] ?? null));
  }

  return rows.length;
}

async function main() {
  const client = await pool.connect();
  try {
    await client.query("BEGIN");
    for (const table of tables) {
      const exists = sqlite.prepare("SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?").get(table);
      if (!exists) {
        console.log(`${table}: skipped, not present in SQLite`);
        continue;
      }

      const rows = sqlite.prepare(`SELECT * FROM ${table}`).all();
      const inserted = await insertRows(client, table, rows);
      console.log(`${table}: copied ${inserted} row(s)`);
    }
    await client.query("COMMIT");
    console.log("SQLite to PostgreSQL data migration complete.");
  } catch (error) {
    await client.query("ROLLBACK");
    throw error;
  } finally {
    client.release();
    sqlite.close();
    await pool.end();
  }
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
