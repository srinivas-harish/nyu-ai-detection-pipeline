"use client";

import { useState } from "react";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "";

export default function GeneratePage() {
  const [selectedPromptId, setSelectedPromptId] = useState("1");
  const [selectedModel, setSelectedModel] = useState("dry-run");
  const [promptVersion, setPromptVersion] = useState("v1");
  const [customPromptText, setCustomPromptText] = useState("");
  const [status, setStatus] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const promptIds = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"];
  const models = ["dry-run", "openai", "anthropic", "gemini"];

  const handleStartRun = async () => {
    setError(null);
    setStatus("Starting… (backend not wired; see TODO)");
    await new Promise((r) => setTimeout(r, 800));
    setStatus(
      "Stub: no backend. Set NEXT_PUBLIC_API_BASE and implement /api/generate."
    );
  };

  return (
    <div className="space-y-8">
      <h1 className="text-xl font-medium text-[var(--text)]">Generate</h1>
      <p className="text-sm text-[var(--muted)]">
        Select texts, choose prompt and model, start a run. Prompt editing is
        explicit and versioned. Status is observable (no synthetic progress).
      </p>

      <section className="space-y-3">
        <h2 className="text-sm font-medium text-[var(--text)]">
          Input texts
        </h2>
        <p className="text-xs text-[var(--muted)]">
          TODO: Backend not wired. Will list/upload source files and pass paths
          to generation job.
        </p>
        <div className="rounded border border-[var(--border)] bg-[var(--surface)] px-4 py-3 text-sm text-[var(--muted)]">
          No file selector yet — API not implemented
        </div>
      </section>

      <section className="space-y-3">
        <h2 className="text-sm font-medium text-[var(--text)]">Prompt</h2>
        <div className="flex flex-wrap gap-4 items-center">
          <label className="text-sm text-[var(--muted)]">
            Prompt ID
            <select
              value={selectedPromptId}
              onChange={(e) => setSelectedPromptId(e.target.value)}
              className="ml-2 rounded border border-[var(--border)] bg-[var(--surface)] px-2 py-1 text-[var(--text)]"
            >
              {promptIds.map((id) => (
                <option key={id} value={id}>
                  {id}
                </option>
              ))}
            </select>
          </label>
          <label className="text-sm text-[var(--muted)]">
            Version
            <input
              type="text"
              value={promptVersion}
              onChange={(e) => setPromptVersion(e.target.value)}
              className="ml-2 w-20 rounded border border-[var(--border)] bg-[var(--surface)] px-2 py-1 text-[var(--text)]"
            />
          </label>
        </div>
        <label className="block text-sm text-[var(--muted)]">
          Custom prompt text (overrides registry when non-empty; version should
          be set explicitly)
        </label>
        <textarea
          value={customPromptText}
          onChange={(e) => setCustomPromptText(e.target.value)}
          placeholder="Leave empty to use registry prompt"
          rows={3}
          className="w-full max-w-xl rounded border border-[var(--border)] bg-[var(--surface)] px-3 py-2 text-sm text-[var(--text)] placeholder:text-[var(--muted)]"
        />
      </section>

      <section className="space-y-3">
        <h2 className="text-sm font-medium text-[var(--text)]">Model</h2>
        <select
          value={selectedModel}
          onChange={(e) => setSelectedModel(e.target.value)}
          className="rounded border border-[var(--border)] bg-[var(--surface)] px-3 py-2 text-sm text-[var(--text)]"
        >
          {models.map((m) => (
            <option key={m} value={m}>
              {m}
            </option>
          ))}
        </select>
      </section>

      <section className="space-y-3">
        <button
          onClick={handleStartRun}
          className="rounded border border-[var(--border)] bg-[var(--surface)] px-4 py-2 text-sm text-[var(--text)] hover:bg-[var(--border)]"
        >
          Start run
        </button>
        {status && (
          <p className="text-sm text-[var(--muted)]" role="status">
            {status}
          </p>
        )}
        {error && (
          <p className="text-sm text-red-400" role="alert">
            {error}
          </p>
        )}
      </section>
    </div>
  );
}
