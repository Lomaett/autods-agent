import Link from "next/link";

const cards = [
  {
    title: "Run Pipelines",
    desc: "Upload datasets and launch EDA or full analysis jobs.",
    href: "/run",
    tone: "from-cyan-600 to-teal-600"
  },
  {
    title: "Models Registry",
    desc: "View trained model artifacts from the API.",
    href: "/models",
    tone: "from-amber-600 to-orange-600"
  },
  {
    title: "Predict",
    desc: "Score a new file with an existing trained model.",
    href: "/predict",
    tone: "from-emerald-600 to-cyan-700"
  },
  {
    title: "Reports",
    desc: "Inspect generated report files and artifact names.",
    href: "/reports",
    tone: "from-slate-700 to-cyan-900"
  }
];

export default function HomePage() {
  return (
    <section className="space-y-8">
      <div className="rounded-3xl bg-white/90 p-6 shadow-panel md:p-10">
        <p className="mb-2 inline-block rounded-full bg-fog px-3 py-1 text-xs font-bold uppercase tracking-wider text-ocean">
          Autonomous Data Science Workbench
        </p>
        <h1 className="text-3xl font-black leading-tight text-ink md:text-5xl">
          Build, train, and serve ML runs from one control room.
        </h1>
        <p className="mt-4 max-w-3xl text-slate-600">
          This UI sits on top of your FastAPI backend and lets you run EDA, launch end-to-end analyses,
          inspect reports, and run inference without touching curl.
        </p>
      </div>

      <div className="grid gap-4 md:grid-cols-2">
        {cards.map((card) => (
          <Link
            key={card.href}
            href={card.href}
            className="group rounded-2xl bg-white/85 p-5 shadow-panel transition hover:-translate-y-1"
          >
            <div className={"h-1.5 w-20 rounded-full bg-gradient-to-r " + card.tone} />
            <h2 className="mt-4 text-xl font-bold text-ink">{card.title}</h2>
            <p className="mt-2 text-sm text-slate-600">{card.desc}</p>
            <p className="mt-4 text-sm font-semibold text-ocean group-hover:underline">Open page</p>
          </Link>
        ))}
      </div>
    </section>
  );
}
