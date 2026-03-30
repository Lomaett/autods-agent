"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { useParams } from "next/navigation";
import ReactMarkdown from "react-markdown";
import { getReportFile } from "@/lib/api";

function toMarkdown(name, payload) {
  if (typeof payload === "string") {
    return `# Report: ${name}\n\n${payload}`;
  }

  const brief = payload?.brief || {};
  const ml = payload?.ml_result || {};
  const report = payload?.report || {};

  const lines = [];
  lines.push(`# Report: ${name}`);
  lines.push("");

  if (Object.keys(brief).length) {
    lines.push("## Data Summary");
    if (brief.task_type) lines.push(`- Task type: **${brief.task_type}**`);
    if (brief.target_col) lines.push(`- Target column: **${brief.target_col}**`);
    if (brief.shape) lines.push(`- Input shape: ${brief.shape.rows} rows x ${brief.shape.columns} columns`);
    if (brief.clean_shape) lines.push(`- Clean shape: ${brief.clean_shape.rows} rows x ${brief.clean_shape.columns} columns`);
    if (brief.warnings?.length) {
      lines.push("");
      lines.push("### Warnings");
      brief.warnings.forEach((w) => lines.push(`- ${w}`));
    }
    lines.push("");
  }

  if (Object.keys(ml).length) {
    lines.push("## Model Summary");
    if (ml.best_model_name) lines.push(`- Best model: **${ml.best_model_name}**`);
    if (ml.task_type) lines.push(`- Task type: ${ml.task_type}`);
    const metricEntries = Object.entries(ml.metrics || {}).filter(([k]) => k !== "classification_report");
    if (metricEntries.length) {
      lines.push("");
      lines.push("### Metrics");
      metricEntries.forEach(([k, v]) => lines.push(`- ${k}: ${String(v)}`));
    }
    if (ml.metrics?.classification_report) {
      lines.push("");
      lines.push("### Classification Report");
      lines.push("```");
      lines.push(String(ml.metrics.classification_report));
      lines.push("```");
    }
    lines.push("");
  }

  if (Object.keys(report).length) {
    lines.push("## Recommendations");
    if (report.recommendations?.length) {
      report.recommendations.forEach((r) => lines.push(`- ${r}`));
    } else {
      lines.push("- No recommendations were generated.");
    }

    if (report.risks?.length) {
      lines.push("");
      lines.push("## Risks");
      report.risks.forEach((r) => lines.push(`- ${r}`));
    }

    if (report.next_steps?.length) {
      lines.push("");
      lines.push("## Next Steps");
      report.next_steps.forEach((s, idx) => lines.push(`${idx + 1}. ${s}`));
    }
  }

  if (lines.length < 3) {
    return `# Report: ${name}\n\nNo structured content found.`;
  }

  return lines.join("\n");
}

export default function ReportDetailPage() {
  const params = useParams();
  const name = useMemo(() => decodeURIComponent(params?.name || ""), [params]);
  const [markdown, setMarkdown] = useState("");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    let mounted = true;
    async function load() {
      if (!name) return;
      setLoading(true);
      setError("");

      try {
        const result = await getReportFile(name);
        const md = toMarkdown(name, result.data);
        if (mounted) setMarkdown(md);
      } catch (err) {
        if (mounted) setError(err.message || "Failed to load report");
      } finally {
        if (mounted) setLoading(false);
      }
    }
    load();
    return () => {
      mounted = false;
    };
  }, [name]);

  return (
    <section className="space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <h1 className="text-2xl font-black text-ink">Report Viewer</h1>
        <Link href="/reports" className="rounded-lg bg-slate-200 px-3 py-2 text-sm font-semibold text-slate-800">
          Back
        </Link>
      </div>

      <div className="rounded-3xl bg-white/95 p-6 shadow-panel">
        {loading ? <p className="text-sm text-slate-500">Loading report...</p> : null}
        {error ? <p className="rounded-xl bg-rose-50 p-3 text-sm text-rose-700">{error}</p> : null}

        {!loading && !error ? (
          <article className="prose prose-slate max-w-none">
            <ReactMarkdown>{markdown}</ReactMarkdown>
          </article>
        ) : null}
      </div>
    </section>
  );
}
