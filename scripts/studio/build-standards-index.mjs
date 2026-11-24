#!/usr/bin/env node
// Build offline searchable UOR-R4 / Foundry / LexLean / PrismPM / Stitch standards index
// written to <dest> (default: web/studio/foundry/studio/shell/standards-index.json).

import { createHash } from 'node:crypto';
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(__dirname, '../..');
const defaultOut = resolve(repoRoot, 'web/studio/foundry/studio/shell/standards-index.json');
const MAX_TOTAL_BYTES = 110 * 1024;

function readTrimmed(relPath, maxChars = 6000) {
  const full = resolve(repoRoot, relPath);
  if (!existsSync(full)) return '';
  const text = readFileSync(full, 'utf8').replace(/\r\n/g, '\n');
  return text.length > maxChars ? text.slice(0, maxChars).trimEnd() + '\n...[truncated]' : text.trim();
}

function splitMarkdownSections(relPath, topic, defaultTags, maxSectionChars = 2200) {
  const raw = readTrimmed(relPath, 24000);
  if (!raw) return [];
  const lines = raw.split('\n');
  const sections = [];
  let title = relPath;
  let buf = [];
  const flush = () => {
    const content = buf.join('\n').trim();
    if (!content) return;
    const clipped = content.length > maxSectionChars ? content.slice(0, maxSectionChars).trimEnd() + '\n...' : content;
    const slug = `${topic}-${title}`.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '').slice(0, 64);
    const firstPara = clipped.split('\n').find((l) => l.trim() && !l.startsWith('#')) || clipped.slice(0, 180);
    sections.push({
      id: `${slug}-${sections.length}`,
      title,
      topic,
      tags: [...new Set([topic, ...defaultTags, ...title.toLowerCase().split(/\W+/).filter((w) => w.length > 2)])].slice(0, 14),
      summary: firstPara.slice(0, 220),
      content: clipped,
      source: relPath,
    });
  };
  for (const line of lines) {
    const heading = line.match(/^#{1,3}\s+(.+)$/);
    if (heading) {
      flush();
      title = heading[1].trim();
      buf = [line];
    } else {
      buf.push(line);
    }
  }
  flush();
  return sections;
}

export function buildStandardsIndex(outPath = defaultOut) {
  const sections = [
    ...splitMarkdownSections('CHARTER.md', 'charter', ['rules', 'geometry', 'native-space', 'milestones'], 2000),
    ...splitMarkdownSections('AGENTS.md', 'agents', ['onboarding', 'constraints', 'harness', 'delivery'], 1800),
  ];

  // Key Decisions D84, D85, D90 from docs/integration/DECISIONS.md
  const decRaw = readTrimmed('docs/integration/DECISIONS.md', 200000);
  if (decRaw) {
    for (const id of ['D80', 'D81', 'D82', 'D83', 'D84', 'D85', 'D90']) {
      const re = new RegExp(`^\\|\\s*${id}\\s*\\|.*$`, 'm');
      const m = decRaw.match(re);
      if (m) {
        const content = m[0].slice(0, 2200);
        sections.push({
          id: `decision-${id.toLowerCase()}`,
          title: `Decision ${id}`,
          topic: 'decisions',
          tags: ['decisions', id.toLowerCase(), id === 'D90' ? 'studio' : 'architecture', id === 'D85' ? 'native-space' : 'uor-r4'],
          summary: content.slice(0, 220),
          content,
          source: 'docs/integration/DECISIONS.md',
        });
      }
    }
  }

  // Real Foundry R4 Judge spec from web/studio/foundry/studio/shell/r4/spec.json
  const r4SpecRaw = readTrimmed('web/studio/foundry/studio/shell/r4/spec.json', 8000);
  if (r4SpecRaw) {
    sections.push({
      id: 'prism-r4-judge-spec',
      title: 'Prism R4 Judge Specification (r4/spec.json)',
      topic: 'r4-policy',
      tags: ['r4', 'prism', 'policy', 'weights', 'engine', 'boundary'],
      summary: 'Compiled Prism R4 admission rules gating in-browser integer WASM and Prism IdeaReader outputs.',
      content: r4SpecRaw.slice(0, 2400),
      source: 'web/studio/foundry/studio/shell/r4/spec.json',
    });
  }

  // Real Foundry specification & LexLean header
  const foundryLex = readTrimmed('web/studio/foundry/app/src/FoundrySpecification.lex.tex', 4500);
  if (foundryLex) {
    sections.push({
      id: 'foundry-specification-lexlean',
      title: 'FoundrySpecification LexLean 0.3.0 Kernel Module',
      topic: 'lexlean',
      tags: ['lexlean', 'foundry', 'specification', 'semantic-module', 'boundaries'],
      summary: 'Formal LexLean semantic module defining Foundry host, storage, preview, build, and chat queue boundaries.',
      content: foundryLex.slice(0, 2400),
      source: 'web/studio/foundry/app/src/FoundrySpecification.lex.tex',
    });
  }

  // Grounded technical standards for Foundry webharness subsystems
  sections.push(
    {
      id: 'lexlean-semantic-module-contract',
      title: 'LexLean 0.3.0 & Prism Spec Authoring Standard',
      topic: 'lexlean',
      tags: ['lexlean', 'prism', 'spec', 'theorems', 'verification', 'lean'],
      summary: 'How Foundry checks and proves decision (<prism-spec>) and stateful (<prism-lexlean>) modules in-browser.',
      content: [
        '# LexLean & Prism Verification in Foundry Studio',
        '- Decision apps: stored in `prism/spec.json` and rendered canonically via `lexlean.mjs` (`toLexLean`), verified in-browser against acceptance vectors and proved by Lean in `prism.js`.',
        '- Stateful apps: stored in `prism/App.lex.tex` with declarations `State`, `catalogue`, `initial`, `view`, `step`, and `theorem` obligations checked via `live/app.mjs` (`checkApp`).',
        '- Use `lexlean_check(path)` to parse, typecheck, and run semantic checks on a `.lex.tex` or `prism/spec.json` file, and `lexlean_verify(path, profile?)` to evaluate theorems / obligation checks.',
      ].join('\n'),
      source: 'web/studio/foundry/studio/shell/prism.js',
    },
    {
      id: 'prismpm-holo-v4-standard',
      title: 'PrismPM .holo v4 Capsule & Core Closure Standard',
      topic: 'prismpm',
      tags: ['prismpm', 'holo', 'closure', 'blake3', 'cha-cha20', 'core-wasm'],
      summary: 'Portable Hologram v4 archives compose view, model, source manifest, provenance, and guest.wasm with BLAKE3 addressing.',
      content: [
        '# PrismPM .holo v4 Verification Standard',
        '- Every Studio project version is addressed by BLAKE3 (`kappa`) over deterministic preimages verified by `core.wasm`.',
        '- Portable publication (`holo:publish`) requires the shell closure `manifest.json` SHA-256 to match `provenance.json` (`browser_projection_sha256`).',
        '- Secrets (such as GitHub PATs in `refs["vault:github:credentials"]`) are sealed synchronously in `core.wasm` using ChaCha20-Poly1305 (`holo_seal`/`holo_open`) under `frame.workspaceKey()`.',
      ].join('\n'),
      source: 'web/studio/foundry/studio/shell/holo.js',
    },
    {
      id: 'stitch-protocol-standard',
      title: 'Stitch Networking & 2-of-3 XOR Parity Standard',
      topic: 'stitch',
      tags: ['stitch', 'xor-2-of-3', 'webrtc', 'signal', 'afflom', 'networking', 'shards'],
      summary: 'Stitch provides `stitch/xor-2-of-3/1` bytewise shard recovery, `stitch.signal/1` WebRTC peering, and `uor-workspace-bridge/1` (`afflom/me`).',
      content: [
        '# Stitch Protocol & afflom/me Standard',
        '- `stitch/xor-2-of-3/1`: Pairs adjacent model pieces (`left`, `right`, `parity = left ^ right`). Loss or SHA-256 mismatch on either shard heals deterministically from the remaining 2 shards.',
        '- `stitch.signal/1`: Ed25519-authenticated WebRTC `RTCPeerConnection` signaling channel connecting browser VMs (`ne2k` `net0-send`/`net0-receive`) and `afflom/me` app launch (`app.json`, `embed=me`, `uor-workspace-bridge/1`).',
      ].join('\n'),
      source: 'web/studio/foundry/studio/shell/uor-v13-worker.mjs',
    },
    {
      id: 'foundry-v13-runtime-standard',
      title: 'UOR-R4 Foundry V13 Pure-Integer Serving & Native Tooling',
      topic: 'runtime',
      tags: ['uorr4v13', 'qwen3', 'tools', 'workspace', 'github', 'linux', 'v86', 'mvm'],
      summary: 'Built-in Studio runtime serves pure-integer Qwen3-0.6B (default) and Qwen3-1.7B (`UORR4V13`) with OPFS/IndexedDB pack caching and browser tools.',
      content: [
        '# UOR-R4 V13 Webharness Standard',
        '- Engine: `uor_foundry_v13_wasm` (`compute_float_ops: 0`, `dense_matmul_ops: 0` across all 49 `HoloV13Model` functions); models `uor-r4:qwen3-0.6b` (9 packs) and `uor-r4:qwen3-1.7b` (23 packs).',
        '- Editing app files in Build mode uses `<prism-write path="..." description="...">...</prism-write>` or tool `workspace_write`.',
        '- Shell commands execute inside the in-browser Linux 6.18.7 VM (`linux_exec`, `/workspace` 9p synced to Studio IndexedDB) or the RV64IMAC MVM sandbox (`mvm_exec`).',
      ].join('\n'),
      source: 'web/studio/foundry/studio/shell/r4-agent-loop.mjs',
    },
    {
      id: 'foundry-studio-architecture',
      title: 'Foundry Web Studio Architecture, Modes & Panels',
      topic: 'studio',
      tags: ['studio', 'foundry', 'modes', 'chat', 'build', 'agent', 'preview', 'terminal', 'github', 'settings'],
      summary: 'How Foundry Web Studio organizes Home Chat, Build/Agent workspace modes, Live Preview, Terminals, GitHub, and Settings.',
      content: [
        '# Foundry Web Studio Architecture & Navigation',
        '- Three Studio modes: **Home Chat** (`chat`, read-only ideation before creating an app — click "Build this in Studio" to create one), **Build** (default workspace mode for editing project files with `<prism-write>` / `workspace_write` and hot-rebuilding the esbuild preview), and **Local Agent** (autonomous multi-step tool loop).',
        '- Workspace panels: **Chat** (conversation + tool execution transcript), **Code** (workspace file tree and source editor), **Preview** (live compiled Vite React bundle served at `/p/<id>/`), **Problems** (`check-problems` TypeScript/LexLean diagnostics), **Publish** (`holo:publish` single-file HTML/.holo export & `holo:inspect`), **GitHub** (PAT vault + repo pull/push), **Terminal** (Linux 6.18.7 v86 VM `/workspace` + RV64IMAC MVM), and **Settings** (UOR-R4 V13 local WASM, Custom AI API endpoints, Session Reset via `clear-session-data`).',
      ].join('\n'),
      source: 'web/studio/foundry/studio/shell/host.js',
    },
    {
      id: 'foundry-tools-and-holo',
      title: 'Foundry 18 Native Tools & PrismPM .holo Capsule Publishing',
      topic: 'tools',
      tags: ['tools', 'workspace', 'holo', 'publish', 'inspect', 'lexlean', 'prism', 'github', 'standards'],
      summary: 'Complete reference for Studio native browser tools, <prism-write> tags, and .holo v4 portable build/inspect.',
      content: [
        '# Foundry Webharness Tools & .holo Publishing',
        '- Native browser tools callable via `<tool_call>{"name": ..., "arguments": ...}</tool_call>`: `workspace_read`, `workspace_write`, `workspace_list`, `linux_exec`, `mvm_exec`, `github_status`, `github_list_repos`, `github_clone`, `github_pull`, `github_push`, `prismpm_verify`, `holo_publish`, `holo_inspect`, `lexlean_check`, `lexlean_verify`, `standards_lookup`, `stitch_fetch`, `stitch_peer_status`.',
        '- File authoring: Build mode accepts `<prism-write path="src/App.tsx" description="...">...</prism-write>` blocks or `workspace_write({ path, content })`. Home Chat (`chat` mode) refuses file writes.',
        '- Publishing: `holo:publish` compiles React+Tailwind inside the hidden preview iframe (`compileTailwind`), bundles production JS with `esbuild`, computes BLAKE3 `kappa` in `core.wasm`, and seals a self-contained `.holo` v4 HTML archive inspectable via `holo:inspect`.',
      ].join('\n'),
      source: 'web/studio/foundry/studio/shell/r4-agent-loop.mjs',
    },
    {
      id: 'foundry-browser-apis-and-vm',
      title: 'Browser Storage (IndexedDB/OPFS), Service Worker Preview, Linux VM & Stitch',
      topic: 'browser-apis',
      tags: ['indexeddb', 'opfs', 'service-worker', 'linux', 'v86', 'mvm', 'stitch', 'cache', 'clear-session-data'],
      summary: 'How IndexedDB, OPFS, Service Worker caches, Linux 6.18.7 9p workspace, MVM, and Stitch work together in-browser.',
      content: [
        '# Browser APIs, Storage, Service Worker & VMs in Foundry Studio',
        '- Storage: IndexedDB stores apps, chats, messages, versions, refs (`app-env:*`, GitHub vault), and fallback weight shards; OPFS (`uor-r4-v13-packs`) caches downloaded `pack_00`..`pack_08` (0.6B) and `pack_00`..`pack_22` (1.7B) model shards. Clearing session data (`clear-session-data`) clears apps/chats/previews while preserving cached model weights (`"holo"` cache / OPFS).',
        '- Service Worker (`sw.js`): claims controlled pages, serves `/p/<id>/` live preview & Tailwind compilation frames from `"previews"` CacheStorage, and pre-caches the offline closure manifest (`manifest.json`).',
        '- In-browser compute: Linux 6.18.7 (`v86` x86) mounts `/workspace` over 9p synchronized with IndexedDB; MVM executes RV64IMAC ELFs with Copy-on-Write memory hashes; `uor_foundry_v13_wasm` runs pure-integer WASM inference inside a Dedicated Worker with `MessageChannel` yield and KV prefix reuse.',
      ].join('\n'),
      source: 'web/studio/foundry/studio/shell/linux-vm-host.mjs',
    },
  );

  // Enforce strict total size bound < 110 KB so shell closure stays lean.
  const index = {
    schema: 'uor-r4.standards-index/1',
    generated_at: new Date().toISOString(),
    section_count: sections.length,
    sections,
  };
  let encoded = JSON.stringify(index);
  while (Buffer.byteLength(encoded, 'utf8') > MAX_TOTAL_BYTES && sections.length > 6) {
    sections.pop();
    index.section_count = sections.length;
    encoded = JSON.stringify(index);
  }
  mkdirSync(dirname(outPath), { recursive: true });
  writeFileSync(outPath, encoded + '\n', 'utf8');
  return {
    path: outPath,
    bytes: Buffer.byteLength(encoded, 'utf8') + 1,
    sha256: createHash('sha256').update(encoded + '\n', 'utf8').digest('hex'),
    sections: sections.length,
  };
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const out = process.argv[2] ? resolve(process.argv[2]) : defaultOut;
  const info = buildStandardsIndex(out);
  console.log(JSON.stringify(info));
}
