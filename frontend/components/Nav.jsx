"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const links = [
  { href: "/", label: "Dashboard" },
  { href: "/run", label: "Run" },
  { href: "/models", label: "Models" },
  { href: "/reports", label: "Reports" },
  { href: "/predict", label: "Predict" }
];

export default function Nav() {
  const pathname = usePathname();

  return (
    <header className="sticky top-0 z-20 border-b border-slate-200/70 bg-white/90 backdrop-blur">
      <nav className="mx-auto flex max-w-6xl flex-wrap items-center justify-between gap-4 px-4 py-3 md:px-6">
        <Link href="/" className="text-lg font-black tracking-tight text-ink">
          AutoDS Studio
        </Link>
        <ul className="flex flex-wrap items-center gap-2">
          {links.map((link) => {
            const active = pathname === link.href;
            return (
              <li key={link.href}>
                <Link
                  href={link.href}
                  className={`inline-block rounded-full px-4 py-2 text-sm font-semibold transition ${
                    active
                      ? "bg-ocean text-white shadow-panel"
                      : "bg-slate-100 text-slate-700 hover:bg-slate-200"
                  }`}
                >
                  {link.label}
                </Link>
              </li>
            );
          })}
        </ul>
      </nav>
    </header>
  );
}
