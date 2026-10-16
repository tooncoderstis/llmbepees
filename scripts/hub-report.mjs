#!/usr/bin/env node
// Laporan fase proyek ke hub portofolio.
// Tanpa dependensi. Jalankan: node scripts/hub-report.mjs
// Env yang dibaca: HUB_INGEST_URL, HUB_INGEST_SECRET, HUB_PROJECT_SLUG, HUB_PROJECT_NAME
// Lihat: docs/hub/project-convention.md

import { existsSync, readFileSync } from "node:fs";
import { basename, join } from "node:path";

function loadEnvFile(path) {
  if (!existsSync(path)) return {};

  const env = {};

  for (const line of readFileSync(path, "utf8").split(/\r?\n/)) {
    const match = line.match(/^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)\s*$/);

    if (!match) continue;

    let value = match[2].trim();

    if (
      (value.startsWith('"') && value.endsWith('"')) ||
      (value.startsWith("'") && value.endsWith("'"))
    ) {
      value = value.slice(1, -1);
    }

    env[match[1]] = value;
  }

  return env;
}

function slugify(value) {
  return value
    .toLowerCase()
    .normalize("NFKD")
    .replace(/[\u0300-\u036f]/g, "")
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .replace(/-{2,}/g, "-");
}

function readOptional(path) {
  return existsSync(path) ? readFileSync(path, "utf8") : undefined;
}

async function main() {
  const cwd = process.cwd();
  const fileEnv = loadEnvFile(join(cwd, ".env"));
  const env = { ...fileEnv, ...process.env };

  const baseUrl = env.HUB_INGEST_URL;
  const secret = env.HUB_INGEST_SECRET ?? "";

  if (!baseUrl) {
    console.error(
      "HUB_INGEST_URL belum diset. Tambahkan ke .env (lihat contoh di docs/hub/project-convention.md).",
    );
    process.exitCode = 1;
    return;
  }

  const packageJsonPath = join(cwd, "package.json");
  let packageName;

  try {
    packageName = JSON.parse(readFileSync(packageJsonPath, "utf8")).name;
  } catch {
    packageName = undefined;
  }

  const name = env.HUB_PROJECT_NAME || packageName || basename(cwd);
  const slug = env.HUB_PROJECT_SLUG || slugify(name);

  const statusMarkdown = readOptional(
    env.HUB_STATUS_FILE
      ? join(cwd, env.HUB_STATUS_FILE)
      : join(cwd, "STATUS.md"),
  );
  const prdMarkdown = readOptional(
    env.HUB_PRD_FILE ? join(cwd, env.HUB_PRD_FILE) : join(cwd, "PRD.md"),
  );

  if (!statusMarkdown) {
    console.error("STATUS.md tidak ditemukan di folder proyek.");
    process.exitCode = 1;
    return;
  }

  const response = await fetch(`${baseUrl.replace(/\/$/, "")}/api/hub/ingest`, {
    method: "POST",
    headers: {
      "content-type": "application/json",
      "x-hub-secret": secret,
    },
    body: JSON.stringify({
      project: { slug, name, path: cwd },
      statusMarkdown,
      prdMarkdown,
    }),
  });

  const body = await response.json().catch(() => ({}));

  if (!response.ok) {
    console.error(`Gagal lapor (${response.status}):`, body);
    process.exitCode = 1;
    return;
  }

  const started = body.started ?? [];
  const completed = body.completed ?? [];
  const parts = [];

  if (started.length > 0) {
    parts.push(
      `${started.length} mulai dikerjakan (${started
        .map((item) => item.phaseId)
        .join(", ")})`,
    );
  }

  if (completed.length > 0) {
    parts.push(
      `${completed.length} selesai (${completed
        .map((item) => item.phaseId)
        .join(", ")})`,
    );
  }

  console.log(
    `OK  ${slug}: ${body.phaseCount ?? 0} fase terbaca` +
      (parts.length > 0 ? `, ${parts.join(", ")}` : ""),
  );
}

main().catch((error) => {
  console.error("Laporan gagal:", error);
  process.exitCode = 1;
});
