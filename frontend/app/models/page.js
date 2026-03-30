"use client";

import { useEffect, useState } from "react";
import { getModels } from "@/lib/api";

export default function ModelsPage() {
  const [models, setModels] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    let mounted = true;
    async function run() {
      setLoading(true);
      setError("");
      try {
        const data = await getModels();
        if (mounted) setModels(data.models || []);
      } catch (err) {
        if (mounted) setError(err.message || "Failed to load models");
      } finally {
        if (mounted) setLoading(false);
      }
    }
    run();
    return () => {
      mounted = false;
    };
  }, []);

  return (
    <section className="space-y-4">
      <h1 className="text-2xl font-black text-ink">Models Registry</h1>
      <p className="text-sm text-slate-600">Artifacts currently exposed by /models.</p>

      <div className="rounded-3xl bg-white/90 p-6 shadow-panel">
        {loading ? <p className="text-sm text-slate-500">Loading models...</p> : null}
        {error ? <p className="rounded-xl bg-rose-50 p-3 text-sm text-rose-700">{error}</p> : null}
        {!loading && !error && models.length === 0 ? <p className="text-sm text-slate-600">No models yet.</p> : null}

        <ul className="grid gap-3 md:grid-cols-2">
          {models.map((name) => (
            <li key={name} className="rounded-2xl border border-slate-200 bg-slate-50 p-4">
              <p className="text-sm font-semibold text-ink">{name}</p>
            </li>
          ))}
        </ul>
      </div>
    </section>
  );
}
