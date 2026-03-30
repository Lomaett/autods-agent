"use client";

import { useState } from "react";
import { postMultipart } from "@/lib/api";

export default function PredictPage() {
  const [file, setFile] = useState(null);
  const [modelName, setModelName] = useState("");
  const [featureCols, setFeatureCols] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);

  async function handlePredict() {
    if (!file || !modelName || !featureCols) {
      setError("File, model name, and feature columns are required.");
      return;
    }

    const fd = new FormData();
    fd.append("file", file);
    fd.append("model_name", modelName);
    fd.append("feature_cols", featureCols);

    setLoading(true);
    setError("");
    setResult(null);

    try {
      const payload = await postMultipart("/predict", fd);
      setResult(payload);
    } catch (err) {
      setError(err.message || "Prediction failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <section className="grid gap-6 lg:grid-cols-[1.1fr_1fr]">
      <div className="rounded-3xl bg-white/90 p-6 shadow-panel">
        <h1 className="text-2xl font-black text-ink">Predict</h1>
        <p className="mt-2 text-sm text-slate-600">Call `/predict` with a saved model and input file.</p>

        <div className="mt-6 space-y-4">
          <label className="block text-sm font-semibold text-slate-700">
            Input file
            <input
              type="file"
              className="mt-2 block w-full rounded-xl border border-slate-300 bg-white px-3 py-2"
              onChange={(e) => setFile(e.target.files?.[0] || null)}
            />
          </label>

          <label className="block text-sm font-semibold text-slate-700">
            Model filename
            <input
              value={modelName}
              onChange={(e) => setModelName(e.target.value)}
              placeholder="e.g. GradientBoostingClassifier.pkl"
              className="mt-2 block w-full rounded-xl border border-slate-300 bg-white px-3 py-2"
            />
          </label>

          <label className="block text-sm font-semibold text-slate-700">
            Feature columns (comma-separated)
            <textarea
              value={featureCols}
              onChange={(e) => setFeatureCols(e.target.value)}
              rows={3}
              placeholder="age,income,city_B,city_C"
              className="mt-2 block w-full rounded-xl border border-slate-300 bg-white px-3 py-2"
            />
          </label>

          <button
            onClick={handlePredict}
            disabled={loading}
            className="rounded-xl bg-mint px-4 py-2 text-sm font-bold text-white disabled:opacity-60"
          >
            {loading ? "Predicting..." : "Run Predict"}
          </button>
        </div>

        {error ? <p className="mt-4 rounded-xl bg-rose-50 p-3 text-sm text-rose-700">{error}</p> : null}
      </div>

      <div className="rounded-3xl bg-slate-950 p-6 text-slate-100 shadow-panel">
        <h2 className="text-lg font-bold">Predictions</h2>
        <pre className="mt-4 max-h-[520px] overflow-auto rounded-xl bg-black/35 p-4 text-xs leading-relaxed">
          {result ? JSON.stringify(result, null, 2) : "No predictions yet."}
        </pre>
      </div>
    </section>
  );
}
