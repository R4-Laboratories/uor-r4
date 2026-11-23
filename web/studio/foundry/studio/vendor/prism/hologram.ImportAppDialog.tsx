// prism-prism (ours, not upstream's; see VENDORED.md): imports either a Forge JSON export (`forge:import`)
// or a binary Hologram v4 `.holo` archive (`holo:import`, validated by PrismPM in core.wasm before install).
import { useRef, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { CheckCircle2, ExternalLink, Loader2, Upload } from "lucide-react";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { queryKeys } from "@/lib/queryKeys";
import { showError, showSuccess } from "@/lib/toast";

// Upstream's dialog exports this prompt and other modules may import it; it is kept so they still resolve.
export const AI_RULES_PROMPT =
  "Generate an AI_RULES.md file for this app. Describe the tech stack in 5-10 bullet points and describe clear rules about what libraries to use for what.";

type Imported = { apps: { id: number; name: string; from: string; chats: number[]; files: number }[]; objects: number; refused: number };
type ImportedHolo = {
  ok: boolean;
  valid: boolean;
  kappa: string;
  fingerprint: string;
  byteLength: number;
  url: string;
  application?: string | null;
  version?: string | null;
  files?: string[];
  ms?: number;
};

const host = () => (window as any).electron.ipcRenderer;

function encodeBytesBase64(bytes: Uint8Array): string {
  let text = "";
  for (let i = 0; i < bytes.length; i += 0x8000) {
    text += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  }
  return btoa(text);
}

export function ImportAppDialog({ isOpen, onClose }: { isOpen: boolean; onClose: () => void }) {
  const queryClient = useQueryClient();
  const input = useRef<HTMLInputElement>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [importedHolo, setImportedHolo] = useState<ImportedHolo | null>(null);

  const take = async (file: File | undefined) => {
    if (!file) return;
    setBusy(true);
    setError(null);
    setImportedHolo(null);
    try {
      const buf = new Uint8Array(await file.arrayBuffer());
      const isHolo =
        file.name.toLowerCase().endsWith(".holo") ||
        (buf.length >= 4 && buf[0] === 0x48 && buf[1] === 0x4f && buf[2] === 0x4c && buf[3] === 0x4f);
      if (isHolo) {
        const res: ImportedHolo = await host().invoke("holo:import", {
          bytesBase64: encodeBytesBase64(buf),
          filename: file.name,
        });
        setImportedHolo(res);
        showSuccess(`Verified & installed ${res.application || file.name} (.holo).`);
        return;
      }
      const text = new TextDecoder("utf-8", { fatal: true }).decode(buf);
      const result: Imported = await host().invoke("forge:import", text);
      await queryClient.invalidateQueries({ queryKey: queryKeys.apps.all });
      await queryClient.invalidateQueries({ queryKey: queryKeys.chats.all });
      const n = result.apps.length;
      const renamed = result.apps.filter((a) => a.name !== a.from).map((a) => a.name);
      showSuccess(
        `Imported ${n} app${n === 1 ? "" : "s"}.` + (renamed.length ? ` Renamed: ${renamed.join(", ")}.` : ""),
      );
      onClose();
    } catch (e: any) {
      const message = e instanceof SyntaxError ? "This file is neither a valid .holo archive nor Forge export JSON." : e?.message || String(e);
      setError(message);
      showError(message);
    } finally {
      setBusy(false);
      if (input.current) input.current.value = "";
    }
  };

  return (
    <Dialog open={isOpen} onOpenChange={(open) => !open && !busy && onClose()}>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle>Import app or .holo archive</DialogTitle>
          <DialogDescription>
            Choose a Forge export file (.json) to restore workspace projects, chats, and versions, or choose a
            self-contained Hologram v4 application (.holo) to verify and install locally in this browser.{" "}
            <a className="underline" href="./holo.html" target="_blank" rel="noreferrer">
              Open standalone .holo runner
            </a>
          </DialogDescription>
        </DialogHeader>
        <input
          ref={input}
          type="file"
          accept=".json,.holo,application/json,application/octet-stream"
          className="hidden"
          data-testid="forge-import-file"
          onChange={(e) => take(e.target.files?.[0])}
        />
        {importedHolo && (
          <div className="rounded-md border border-border bg-muted/40 p-3 text-xs space-y-1.5" data-testid="holo-import-result">
            <div className="flex items-center gap-1.5 font-semibold">
              <CheckCircle2 className="h-4 w-4 text-green-600 dark:text-green-400" />
              <span>Verified {importedHolo.application || "Hologram app"}</span>
            </div>
            <div className="font-mono truncate">kappa: {importedHolo.kappa}</div>
            <div>
              {importedHolo.byteLength} bytes · {importedHolo.files?.length || 0} view files
            </div>
            <a
              className="inline-flex items-center gap-1 underline font-medium"
              href={importedHolo.url}
              target="_blank"
              rel="noreferrer"
              data-testid="holo-import-open"
            >
              Open verified .holo app
              <ExternalLink className="h-3 w-3" />
            </a>
          </div>
        )}
        {error && <p className="text-sm text-destructive">{error}</p>}
        <DialogFooter>
          <Button variant="outline" onClick={onClose} disabled={busy}>
            {importedHolo ? "Close" : "Cancel"}
          </Button>
          <Button onClick={() => input.current?.click()} disabled={busy}>
            {busy ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <Upload className="mr-2 h-4 w-4" />}
            {busy ? "Importing..." : "Choose .json or .holo file"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
