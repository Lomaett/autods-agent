"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { getReports } from "@/lib/api";

export default function ReportsPage() {
  const [reports, setReports] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    let mounted = true;
    async function run() {
      setLoading(true);
      setError("");
      try {
        const data = await getReports();
        if (mounted) setReports(data.reports || []);
      } catch (err) {
        if (mounted) setError(err.message || "Failed to load reports");
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
      <h1 className="text-2xl font-black text-ink">Reports</h1>
      <p className="text-sm text-slate-600">Generated report files from /reports.</p>

      <div className="rounded-3xl bg-white/90 p-6 shadow-panel">
        {loading ? <p className="text-sm text-slate-500">Loading reports...</p> : null}
        {error ? <p className="rounded-xl bg-rose-50 p-3 text-sm text-rose-700">{error}</p> : null}
        {!loading && !error && reports.length === 0 ? <p className="text-sm text-slate-600">No reports yet.</p> : null}

        <ul className="space-y-2">
          {reports.map((name) => (
            <li key={name}>
              <Link
                href={"/reports/" + encodeURIComponent(name)}
                className="flex items-center justify-between gap-3 rounded-2xl border border-slate-200 bg-slate-50 px-4 py-3 transition hover:border-slate-300 hover:bg-white"
              >
                <p className="text-sm font-semibold text-ink">{name}</p>
                <span className="ml-auto rounded-lg bg-ocean px-3 py-1.5 text-xs font-bold text-white">Open</span>
              </Link>
            </li>
          ))}
        </ul>
      </div>
    </section>
  );
}
