export default function Home() {
  return (
    <div className="space-y-6">
      <h1 className="text-xl font-medium text-[var(--text)]">
        Operator console
      </h1>
      <p className="text-sm text-[var(--muted)] max-w-xl">
        Run generation jobs, inspect compiled datasets, and run baseline
        inference. Status reflects actual system state; no synthetic metrics.
      </p>
      <ul className="list-disc list-inside text-sm text-[var(--muted)] space-y-2">
        <li>
          <a href="/generate" className="hover:text-[var(--text)] underline">
            Generate
          </a>{" "}
          — Select texts, prompts, models; start runs. Status observable.
        </li>
        <li>
          <a href="/datasets" className="hover:text-[var(--text)] underline">
            Datasets
          </a>{" "}
          — List compiled datasets and view manifests.
        </li>
        <li>
          <a href="/inference" className="hover:text-[var(--text)] underline">
            Inference
          </a>{" "}
          — Upload text, run baseline detector. Results show probabilities, not
          judgments.
        </li>
      </ul>
    </div>
  );
}
