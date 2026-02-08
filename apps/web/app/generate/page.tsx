"use client";

import { useState } from "react";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "";

type JobStatus = {
  job_id: string;
  type: string;
  status: string;
  result?: { output_path?: string; lines_written?: number; error_count?: number };
  error?: string | null;
  created_at: string;
  updated_at: string;
};

export default function GeneratePage() {
  const [inputText, setInputText] = useState("");
  const [selectedPromptId, setSelectedPromptId] = useState("1");
  const [selectedModel, setSelectedModel] = useState("dry-run");
  const [promptVersion, setPromptVersion] = useState("v1");
  const [customPromptText, setCustomPromptText] = useState("");
  const [status, setStatus] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);
  const [jobResult, setJobResult] = useState<JobStatus["result"] | null>(null);

  const promptIds = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"];
  const models = ["dry-run", "openai", "anthropic", "gemini"];

  const handleStartRun = async () => {
    setError(null);
    setJobResult(null);
    setJobId(null);

    if (!API_BASE) {
      setStatus("Stub: set NEXT_PUBLIC_API_BASE to wire the backend.");
      return;
    }
    if (!inputText.trim()) {
      setError("Input text is required.");
      return;
    }

    setStatus("Starting…");
    try {
      const res = await fetch(`${API_BASE}/generate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          input_text: inputText.trim(),
          prompt_id: selectedPromptId,
          model: selectedModel,
          prompt_version: promptVersion || "v1",
        }),
      });
      if (!res.ok) {
        const d = await res.json().catch(() => ({}));
        throw new Error(d.detail || res.statusText);
      }
      const data = (await res.json()) as { job_id: string; status: string };
      setJobId(data.job_id);
      setStatus(`Job ${data.job_id} pending. Polling…`);

      const poll = async (): Promise<JobStatus> => {
        const r = await fetch(`${API_BASE}/jobs/${data.job_id}`);
        if (!r.ok) throw new Error(r.statusText);
        return r.json();
      };

      let job: JobStatus;
      for (;;) {
        await new Promise((r) => setTimeout(r, 1500));
        job = await poll();
        setStatus(`Job ${job.job_id}: ${job.status}`);
        if (job.status === "completed" || job.status === "failed") break;
      }

      if (job.status === "failed") {
        setError(job.error || "Job failed.");
        return;
      }
      setJobResult(job.result ?? null);
      setStatus(`Completed. Output: ${job.result?.output_path ?? "—"}`);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setStatus(null);
    }
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
          Input text
        </h2>
        <p className="text-xs text-[var(--muted)]">
          Paste source text to chunk and generate from. File upload not implemented.
        </p>
        <textarea
          value={inputText}
          onChange={(e) => setInputText(e.target.value)}
          placeholder="Paste input text for generation…"
          rows={6}
          className="w-full max-w-xl rounded border border-[var(--border)] bg-[var(--surface)] px-3 py-2 text-sm text-[var(--text)] placeholder:text-[var(--muted)]"
        />
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
          be set explicitly). Backend uses registry only for now.
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
          disabled={!inputText.trim() && !!API_BASE}
          className="rounded border border-[var(--border)] bg-[var(--surface)] px-4 py-2 text-sm text-[var(--text)] hover:bg-[var(--border)] disabled:opacity-50"
        >
          Start run
        </button>
        {jobId && (
          <p className="text-xs text-[var(--muted)]">Job ID: {jobId}</p>
        )}
        {status && (
          <p className="text-sm text-[var(--muted)]" role="status">
            {status}
          </p>
        )}
        {jobResult && (
          <p className="text-sm text-[var(--text)]">
            Lines written: {jobResult.lines_written ?? "—"}, errors: {jobResult.error_count ?? "—"}
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
