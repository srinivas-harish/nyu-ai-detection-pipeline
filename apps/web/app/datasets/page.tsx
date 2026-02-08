"use client";

import { useState, useEffect } from "react";

// TODO: Wire to backend. Datasets API not yet implemented; list/manifest are stubbed.
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

  useEffect(() => {
    // TODO: GET /api/datasets → list of dataset names or paths. Backend not wired.
    setDatasets([]);
    setLoading(false);
  }, []);

  const loadManifest = async (name: string) => {
    // TODO: GET /api/datasets/:name/manifest → manifest.json. Backend not wired.
    setSelectedManifest(null);
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
        ) : datasets.length === 0 ? (
          <p className="text-sm text-[var(--muted)]">
            No datasets listed. Backend not wired — TODO: serve list from
            artifacts/datasets or API.
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
        To test with real data, run the dataset compiler CLI and point the
        backend (when implemented) at the output folder.
      </p>
    </div>
  );
}
