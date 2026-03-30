const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://127.0.0.1:8080";

function parsePossiblyInvalidJson(text) {
  try {
    return JSON.parse(text);
  } catch {
    // Support legacy report files containing non-standard JSON tokens (NaN/Infinity).
    const normalized = text
      .replace(/\bNaN\b/g, "null")
      .replace(/\bInfinity\b/g, "null")
      .replace(/\b-Infinity\b/g, "null");
    return JSON.parse(normalized);
  }
}

async function readJson(response) {
  const contentType = response.headers.get("content-type") || "";
  if (contentType.includes("application/json")) {
    const text = await response.text();
    return parsePossiblyInvalidJson(text);
  }
  return response.text();
}

export async function getHealth() {
  const response = await fetch(`${API_BASE}/health`, { cache: "no-store" });
  if (!response.ok) {
    throw new Error("Failed to fetch health status");
  }
  return readJson(response);
}

export async function getModels() {
  const response = await fetch(`${API_BASE}/models`, { cache: "no-store" });
  if (!response.ok) {
    throw new Error("Failed to fetch models");
  }
  return readJson(response);
}

export async function getReports() {
  const response = await fetch(`${API_BASE}/reports`, { cache: "no-store" });
  if (!response.ok) {
    throw new Error("Failed to fetch reports");
  }
  return readJson(response);
}

export async function getReportFile(name) {
  const response = await fetch(`${API_BASE}/reports/${encodeURIComponent(name)}`, { cache: "no-store" });
  if (!response.ok) {
    throw new Error("Failed to fetch report file");
  }
  const contentType = response.headers.get("content-type") || "";
  const text = await response.text();
  if (contentType.includes("application/json")) {
    return {
      contentType,
      data: parsePossiblyInvalidJson(text)
    };
  }
  return {
    contentType,
    data: text
  };
}

export async function postMultipart(path, formData) {
  const response = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    body: formData
  });

  const payload = await readJson(response);
  if (!response.ok) {
    const detail = payload?.detail || JSON.stringify(payload);
    throw new Error(detail || "Request failed");
  }
  return payload;
}

export { API_BASE };
