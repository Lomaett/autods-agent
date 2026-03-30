import "./globals.css";
import Nav from "@/components/Nav";

export const metadata = {
  title: "AutoDS Studio",
  description: "Frontend UI for autonomous data science workflows"
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>
        <div className="page-bg" />
        <Nav />
        <main className="mx-auto max-w-6xl px-4 py-8 md:px-6">{children}</main>
      </body>
    </html>
  );
}
