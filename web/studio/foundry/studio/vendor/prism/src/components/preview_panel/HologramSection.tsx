// prism-prism: Publish as a Hologram application. Ours, not Prism's (see VENDORED.md). The words come
// from the verified View through the host; the host composes the .holo through PrismPM's own archive
// code in the core, keeps it at its address on this device, and serves it under that address.
import { useCallback, useEffect, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";

type Words = Record<string, string>;
type Published = { decision: string; word?: string; kappa?: string; url?: string; byteLength?: number; ms?: number; viewBytes?: number };
type Installed = { kappa: string; url?: string; application?: string; installed: number; byteLength: number; appId?: number };
type InspectedHolo = {
  ok: boolean;
  valid: boolean;
  kappa: string;
  fingerprint: string;
  byteLength: number;
  url: string;
  application?: string | null;
  version?: string | null;
  layers?: { kind: string; content_kappa?: string; entry?: string; surface?: string | null }[];
  files?: { path: string; bytes: number }[];
  provenance?: Record<string, unknown> | null;
};

const ipc = () => (window as any).electron.ipcRenderer;
const short = (k?: string) => (k ? k.replace(/^blake3:/, "").slice(0, 12) + "…" : "");

export function HologramSection({ appId }: { appId: number }) {
  const [words, setWords] = useState<Words | null>(null);
  const [apps, setApps] = useState<Installed[]>([]);
  const [busy, setBusy] = useState(false);
  const [exporting, setExporting] = useState(false);
  const [inspectingKappa, setInspectingKappa] = useState<string | null>(null);
  const [inspected, setInspected] = useState<InspectedHolo | null>(null);
  const [result, setResult] = useState<Published | null>(null);
  const refresh = useCallback(() => {
    ipc().invoke("holo:list", appId).then(setApps).catch(() => setApps([]));
  }, [appId]);
  useEffect(() => {
    ipc().invoke("holo:view").then(setWords).catch(() => setWords(null));
    refresh();
  }, [refresh]);
  if (!words) return null;

  const publish = async () => {
    setBusy(true);
    setResult(null);
    setInspected(null);
    try {
      setResult(await ipc().invoke("holo:publish", { appId }));
      refresh();
    } catch (e: any) {
      setResult({ decision: "Refuse", word: String(e?.message || e) });
    } finally {
      setBusy(false);
    }
  };

  const inspectArchive = async (kappa?: string) => {
    setInspectingKappa(kappa || "latest");
    try {
      const info: InspectedHolo = await ipc().invoke("holo:inspect", { kappa, appId });
      setInspected(info);
    } catch (e: any) {
      setResult({ decision: "Refuse", word: String(e?.message || e) });
    } finally {
      setInspectingKappa(null);
    }
  };

  const exportProject = async () => {
    setExporting(true);
    try {
      const exported: { filename: string; json: string } = await ipc().invoke("forge:export", { appId });
      const blob = new Blob([exported.json], { type: "application/json;charset=utf-8" });
      const href = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = href;
      a.download = exported.filename || `app-${appId}.forge-export.json`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      setTimeout(() => URL.revokeObjectURL(href), 1000);
    } catch (e: any) {
      setResult({ decision: "Refuse", word: String(e?.message || e) });
    } finally {
      setExporting(false);
    }
  };

  return (
    <Card data-testid="hologram-section">
      <CardHeader className="pb-3">
        <CardTitle>{words.holoLabel}</CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        <p className="text-sm text-gray-600 dark:text-gray-400">{words.runsLabel}</p>
        <div className="flex flex-wrap items-center gap-2">
          <Button onClick={publish} disabled={busy} data-testid="hologram-publish">
            {busy ? words.publishingLabel : words.publishLabel}
          </Button>
          <Button variant="outline" onClick={exportProject} disabled={exporting} data-testid="hologram-export-project">
            {exporting ? "Exporting..." : "Export Project (.json)"}
          </Button>
          <a
            className="text-sm underline text-muted-foreground hover:text-foreground px-2"
            href="./holo.html"
            target="_blank"
            rel="noreferrer"
            data-testid="hologram-standalone-runner"
          >
            Open standalone .holo runner
          </a>
        </div>
        {result && result.decision === "Accept" && (
          <p className="text-sm" data-testid="hologram-published">
            {words.publishedLabel} · <span className="font-mono">{short(result.kappa)}</span> · {result.byteLength} bytes · {result.ms} ms ·{" "}
            <a className="underline" href={result.url} target="_blank" rel="noreferrer">
              {words.openLabel}
            </a>{" "}
            ·{" "}
            <a className="underline" href={result.url ? result.url.replace(/\/$/, "") + ".holo" : "#"} download>
              {words.downloadLabel}
            </a>
          </p>
        )}
        {result && result.decision === "Chat" && (
          <p className="text-sm text-muted-foreground" data-testid="hologram-chat">
            {result.word}
          </p>
        )}
        {result && result.decision !== "Accept" && result.decision !== "Chat" && (
          <p className="text-sm text-red-600 dark:text-red-400" data-testid="hologram-refused">
            {result.word}
          </p>
        )}
        {apps.length > 0 && (
          <ul className="space-y-1 text-sm">
            {apps.map((a) => (
              <li key={a.kappa} className="flex flex-wrap items-center gap-2">
                <span className="font-mono">{short(a.kappa)}</span>
                <span className="text-gray-500">{a.byteLength} bytes</span>
                <a className="underline" href={a.url} target="_blank" rel="noreferrer">
                  {words.openLabel}
                </a>
                <a className="underline" href={a.url ? a.url.replace(/\/$/, "") + ".holo" : "#"} download>
                  {words.downloadLabel}
                </a>
                <button
                  type="button"
                  className="underline text-xs text-muted-foreground hover:text-foreground cursor-pointer"
                  onClick={() => inspectArchive(a.kappa)}
                  disabled={inspectingKappa === a.kappa}
                  data-testid="hologram-inspect-btn"
                >
                  {inspectingKappa === a.kappa ? "Inspecting..." : "Inspect"}
                </button>
              </li>
            ))}
          </ul>
        )}
        {inspected && (
          <div className="rounded-md border border-border bg-muted/40 p-3 text-xs space-y-1.5" data-testid="hologram-inspect-summary">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <span className="font-semibold">
                {inspected.valid ? "Verified Holo v4" : "Unverified"} · <span className="font-mono">{short(inspected.kappa)}</span>
              </span>
              <a className="underline" href={inspected.url} target="_blank" rel="noreferrer">
                {words.openLabel}
              </a>
            </div>
            <div>
              Fingerprint: <span className="font-mono">{short(inspected.fingerprint)}</span> · {inspected.byteLength} bytes
            </div>
            {Array.isArray(inspected.layers) && (
              <div>Layers: {inspected.layers.map((l) => l.kind).join(", ")}</div>
            )}
            {Array.isArray(inspected.files) && (
              <div>
                View files ({inspected.files.length}): {inspected.files.map((f) => `${f.path} (${f.bytes} B)`).join(", ")}
              </div>
            )}
            {inspected.provenance && (
              <div className="font-mono truncate">
                Provenance schema: {String((inspected.provenance as { schema?: string }).schema || "prismpm/model-provenance/1")}
              </div>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
