"use client";

import { useState } from "react";
import { postMultipart } from "@/lib/api";

const initialState = {
  file: null,
  target_col: "",
  task_type: "",
  report_title: "",
  generate_llm_summary: false
};

function buildFormData(form, endpoint) {
  const fd = new FormData();
  fd.append("file", form.file);
  if (form.target_col) fd.append("target_col", form.target_col);
  if (form.task_type) fd.append("task_type", form.task_type);
  fd.append("report_title", form.report_title || (endpoint === "/eda" ? "UI EDA" : "UI Analyse"));
  fd.append("generate_llm_summary", String(form.generate_llm_summary));
  return fd;
}

export default function RunPage() {
  const [form, setForm] = useState(initialState);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");

  const setField = (key, value) => setForm((prev) => ({ ...prev, [key]: value }));

  async function handleRun(endpoint) {
    if (!form.file) {
      setError("Please choose a dataset file.");
      return;
    }
    if (!form.target_col) {
      setError("Target column is required by your backend.");
      return;
    }

    setLoading(true);
    setResult(null);
    setError("");

    try {
      const payload = await postMultipart(endpoint, buildFormData(form, endpoint));
      setResult(payload);
    } catch (err) {
      setError(err.message || "Request failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <section className="grid gap-6 lg:grid-cols-[1.2fr_1fr]">
      <div className="rounded-3xl bg-white/90 p-6 shadow-panel">
        <h1 className="text-2xl font-black text-ink">Run Pipeline</h1>
        <p className="mt-2 text-sm text-slate-600">Upload data and launch `/eda` or `/analyse`.</p>

        <div className="mt-6 space-y-4">
          <label className="block text-sm font-semibold text-slate-700">
            Dataset file
            <input
              type="file"
              className="mt-2 block w-full rounded-xl border border-slate-300 bg-white px-3 py-2"
              onChange={(e) => setField("file", e.target.files?.[0] || null)}
            />
          </label>

          <label className="block text-sm font-semibold text-slate-700">
            Target column
            <input
              value={form.target_col}
              onChange={(e) => setField("target_col", e.target.value)}
              placeholder="e.g. Survived"
              className="mt-2 block w-full rounded-xl border border-slate-300 bg-white px-3 py-2"
            />
          </label>

          <label className="block text-sm font-semibold text-slate-700">
            Task type (optional)
            <select
              value={form.task_type}
              onChange={(e) => setField("task_type", e.target.value)}
              className="mt-2 block w-full rounded-xl border border-slate-300 bg-white px-3 py-2"
            >
              <option value="">auto</option>
              <option value="classification">classification</option>
              <option value="regression">regression</option>
            </select>
          </label>

          <label className="block text-sm font-semibold text-slate-700">
            Report title
            <input
              value={form.report_title}
              onChange={(e) => setField("report_title", e.target.value)}
              placeholder="e.g. Titanic UI run"
              className="mt-2 block w-full rounded-xl border border-slate-300 bg-white px-3 py-2"
            />
          </label>

          <label className="inline-flex items-center gap-2 text-sm font-medium text-slate-700">
            <input
              type="checkbox"
              checked={form.generate_llm_summary}
              onChange={(e) => setField("generate_llm_summary", e.target.checked)}
            />
            Generate LLM summary
          </label>
        </div>

        <div className="mt-6 flex flex-wrap gap-3">
          <button
            onClick={() => handleRun("/eda")}
            disabled={loading}
            className="rounded-xl bg-ember px-4 py-2 text-sm font-bold text-white disabled:opacity-60"
          >
            {loading ? "Running..." : "Run EDA"}
          </button>
          <button
            onClick={() => handleRun("/analyse")}
            disabled={loading}
            className="rounded-xl bg-ocean px-4 py-2 text-sm font-bold text-white disabled:opacity-60"
          >
            {loading ? "Running..." : "Run Analyse"}
          </button>
        </div>

        {error ? <p className="mt-4 rounded-xl bg-rose-50 p-3 text-sm text-rose-700">{error}</p> : null}
      </div>

      <div className="rounded-3xl bg-slate-950 p-6 text-slate-100 shadow-panel">
        <h2 className="text-lg font-bold">Latest Response</h2>
        <p className="mt-2 text-xs text-slate-400">Raw API payload</p>
        <pre className="mt-4 max-h-[520px] overflow-auto rounded-xl bg-black/35 p-4 text-xs leading-relaxed">
          {result ? JSON.stringify(result, null, 2) : "No run yet."}
        </pre>
      </div>
    </section>
  );
}
