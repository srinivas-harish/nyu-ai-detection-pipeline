"use client";

import { useState } from "react";

// TODO: Wire to backend. Inference API not yet implemented; result is stubbed.
const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "";

type InferenceResult = {
  model: string;
  runtime_sec: number | null;
  probability: number | null;
  error: string | null;
  input_truncated: boolean;
};

export default function InferencePage() {
  const [text, setText] = useState("");
  const [result, setResult] = useState<InferenceResult | null>(null);
  const [running, setRunning] = useState(false);

  const runInference = async () => {
    setResult(null);
    setRunning(true);
    try {
      // TODO: POST to /api/inference with { text }. Backend calls authinfra detector-infer or equivalent.
      // Must return { model, runtime_sec, probability, error, input_truncated }. No judgments — probability only.
      await new Promise((r) => setTimeout(r, 500));
      setResult({
        model: "Hello-SimpleAI/chatgpt-detector-roberta",
        runtime_sec: null,
        probability: null,
        error: "Backend not wired. Set NEXT_PUBLIC_API_BASE and implement /api/inference.",
        input_truncated: false,
      });
    } finally {
      setRunning(false);
    }
  };

  return (
    <div className="space-y-8">
      <h1 className="text-xl font-medium text-[var(--text)]">Inference</h1>
      <p className="text-sm text-[var(--muted)]">
        Upload or paste text, run the baseline detector. Results show
        <strong className="text-[var(--text)]"> probability only</strong>, not
        judgments (e.g. no "AI detected" label). Uncertainty is visible.
      </p>

      <section className="space-y-3">
        <h2 className="text-sm font-medium text-[var(--text)]">Input text</h2>
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Paste or type text to run baseline detector on…"
          rows={8}
          className="w-full max-w-2xl rounded border border-[var(--border)] bg-[var(--surface)] px-3 py-2 text-sm text-[var(--text)] placeholder:text-[var(--muted)]"
        />
        <button
          onClick={runInference}
          disabled={running || !text.trim()}
          className="rounded border border-[var(--border)] bg-[var(--surface)] px-4 py-2 text-sm text-[var(--text)] hover:bg-[var(--border)] disabled:opacity-50"
        >
          {running ? "Running…" : "Run baseline detector"}
        </button>
      </section>

      {result && (
        <section className="space-y-3">
          <h2 className="text-sm font-medium text-[var(--text)]">Result</h2>
          <div className="rounded border border-[var(--border)] bg-[var(--surface)] p-4 space-y-2 text-sm">
            <div>
              <span className="text-[var(--muted)]">Model: </span>
              <span className="text-[var(--text)]">{result.model}</span>
            </div>
            {result.error != null && result.error !== "" ? (
              <div>
                <span className="text-[var(--muted)]">Error: </span>
                <span className="text-red-400">{result.error}</span>
              </div>
            ) : (
              <>
                <div>
                  <span className="text-[var(--muted)]">Probability (0–1): </span>
                  <span className="text-[var(--text)]">
                    {result.probability != null
                      ? result.probability.toFixed(4)
                      : "—"}
                  </span>
                </div>
                <div>
                  <span className="text-[var(--muted)]">Runtime (s): </span>
                  <span className="text-[var(--text)]">
                    {result.runtime_sec != null
                      ? result.runtime_sec.toFixed(4)
                      : "—"}
                  </span>
                </div>
                <div>
                  <span className="text-[var(--muted)]">Input truncated: </span>
                  <span className="text-[var(--text)]">
                    {result.input_truncated ? "Yes" : "No"}
                  </span>
                </div>
              </>
            )}
          </div>
          <p className="text-xs text-[var(--muted)]">
            Probability is P(AI-generated) in [0,1]. Not a binary judgment.
          </p>
        </section>
      )}

      <p className="text-xs text-[var(--muted)]">
        Sentence-level output: TODO — backend would need to split text into
        sentences and return per-sentence probabilities when implemented.
      </p>
    </div>
  );
}
