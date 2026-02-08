"use client";

import { useState, useEffect } from "react";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "";

type FilterLogEntry = { reason: string; count: number };
type Manifest = {
  schema_version?: string;
  dataset_name?: string;
  created_utc?: string;
  source_paths?: string[];
  model_ids?: string[];
  prompt_ids?: string[];
  filter_log?: FilterLogEntry[];
  train_count?: number;
  valid_count?: number;
  split_ratio?: number;
  split_seed?: number;
};

export default function DatasetsPage() {
  const [datasets, setDatasets] = useState<string[]>([]);
  const [selectedManifest, setSelectedManifest] = useState<Manifest | null>(null);
  const [loading, setLoading] = useState(true);
  const [listError, setListError] = useState<string | null>(null);

  useEffect(() => {
    if (!API_BASE) {
      setDatasets([]);
      setLoading(false);
      return;
    }
    let cancelled = false;
    (async () => {
      try {
        const res = await fetch(`${API_BASE}/datasets`);
        if (!res.ok) throw new Error(res.statusText);
        const data = (await res.json()) as { datasets: string[] };
        if (!cancelled) setDatasets(data.datasets || []);
      } catch (e) {
        if (!cancelled) {
          setListError(e instanceof Error ? e.message : String(e));
          setDatasets([]);
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => { cancelled = true; };
  }, []);

  const loadManifest = async (name: string) => {
    if (!API_BASE) {
      setSelectedManifest(null);
      return;
    }
    try {
      const res = await fetch(`${API_BASE}/datasets/${encodeURIComponent(name)}/manifest`);
      if (!res.ok) throw new Error(res.statusText);
      const data = (await res.json()) as Manifest;
      setSelectedManifest(data);
    } catch {
      setSelectedManifest(null);
    }
  };

  return (
    <div className="space-y-8">
      <h1 className="text-xl font-medium text-[var(--text)]">Datasets</h1>
      <p className="text-sm text-[var(--muted)]">
        List compiled datasets and view manifests. Filter log and counts reflect
        actual compilation; no synthetic data.
      </p>

      <section className="space-y-3">
        <h2 className="text-sm font-medium text-[var(--text)]">
          Compiled datasets
        </h2>
        {loading ? (
          <p className="text-sm text-[var(--muted)]">Loading…</p>
        ) : !API_BASE ? (
          <p className="text-sm text-[var(--muted)]">
            Set NEXT_PUBLIC_API_BASE to list datasets from the API.
          </p>
        ) : listError ? (
          <p className="text-sm text-red-400">{listError}</p>
        ) : datasets.length === 0 ? (
          <p className="text-sm text-[var(--muted)]">
            No datasets listed. Compile via API or CLI (artifacts/datasets).
          </p>
        ) : (
          <ul className="list-disc list-inside text-sm text-[var(--text)]">
            {datasets.map((name) => (
              <li key={name}>
                <button
                  onClick={() => loadManifest(name)}
                  className="hover:underline"
                >
                  {name}
                </button>
              </li>
            ))}
          </ul>
        )}
      </section>

      {selectedManifest && (
        <section className="space-y-3">
          <h2 className="text-sm font-medium text-[var(--text)]">
            Manifest
          </h2>
          <pre className="overflow-auto rounded border border-[var(--border)] bg-[var(--surface)] p-4 text-xs text-[var(--text)]">
            {JSON.stringify(selectedManifest, null, 2)}
          </pre>
          {selectedManifest.filter_log && (
            <div className="text-sm text-[var(--muted)]">
              Filter log:{" "}
              {selectedManifest.filter_log
                .map((e) => `${e.reason}: ${e.count}`)
                .join(", ")}
            </div>
          )}
        </section>
      )}

      <p className="text-xs text-[var(--muted)]">
        To compile a dataset, use the API POST /datasets/compile or the CLI
        (python -m authinfra dataset-compile). List is read from
        artifacts/datasets.
      </p>
    </div>
  );
}
