import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "AuthInfra Console",
  description: "Operator console for generation, datasets, and inference.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <body className="min-h-screen bg-[var(--bg)]">
        <nav className="border-b border-[var(--border)] bg-[var(--surface)] px-6 py-3">
          <div className="mx-auto flex max-w-5xl items-center gap-6">
            <a href="/" className="font-medium text-[var(--text)]">
              AuthInfra
            </a>
            <a
              href="/generate"
              className="text-sm text-[var(--muted)] hover:text-[var(--text)]"
            >
              Generate
            </a>
            <a
              href="/datasets"
              className="text-sm text-[var(--muted)] hover:text-[var(--text)]"
            >
              Datasets
            </a>
            <a
              href="/inference"
              className="text-sm text-[var(--muted)] hover:text-[var(--text)]"
            >
              Inference
            </a>
          </div>
        </nav>
        <main className="mx-auto max-w-5xl px-6 py-8">{children}</main>
      </body>
    </html>
  );
}
