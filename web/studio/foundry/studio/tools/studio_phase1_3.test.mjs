import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import fs from "node:fs";
import { test } from "node:test";
import vm from "node:vm";
import { commandRegistry } from "../shell/command-registry.mjs";
import {
  FOUNDRY_SYSTEM_PROMPT,
  FOUNDRY_TOOL_NAMES,
  parseToolCall,
  runUorR4AgentTurn,
} from "../shell/r4-agent-loop.mjs";

const PROVENANCE_FIELDS = [
  "source_id",
  "semantic_id",
  "compiler_semantics_id",
  "snapshot_id",
  "stdlib_semantics_id",
  "prism_stdlib_crate_sha256",
  "lexlean_commit",
  "lexlean_package_sha256",
  "lean4_prod_commit",
  "hologram_live_commit",
  "uor_hologram_commit",
  "target_profile_id",
  "lean_manifest_sha256",
  "lcnf_manifest_sha256",
  "generated_core_sha256",
  "cargo_name",
  "cargo_version",
  "cargo_crate_sha256",
  "view_model_id",
  "browser_projection_sha256",
];

function loadParsePrismTagsFromHost() {
  const hostSrc = fs.readFileSync(new URL("../shell/host.js", import.meta.url), "utf8");
  const cleanStart = hostSrc.indexOf("function cleanWorkspacePath(");
  const cleanEnd = hostSrc.indexOf("\nfunction createFoundryTools(", cleanStart);
  const parseStart = hostSrc.indexOf("function parsePrismTags(");
  const parseEnd = hostSrc.indexOf("\nasync function applyPrismTags(", parseStart);
  assert.ok(cleanStart >= 0 && cleanEnd > cleanStart, "cleanWorkspacePath slice missing");
  assert.ok(parseStart >= 0 && parseEnd > parseStart, "parsePrismTags slice missing");
  return vm.runInNewContext(
    hostSrc.slice(cleanStart, cleanEnd) + "\n" + hostSrc.slice(parseStart, parseEnd) + "\n({ cleanWorkspacePath, parsePrismTags })",
    {},
  );
}

test("r4-agent-loop advertises holo_publish/holo_inspect and repairs malformed/aliased tool JSON", () => {
  assert.ok(FOUNDRY_TOOL_NAMES.includes("holo_publish"));
  assert.ok(FOUNDRY_TOOL_NAMES.includes("holo_inspect"));
  assert.ok(FOUNDRY_SYSTEM_PROMPT.includes("holo_publish"));
  assert.ok(FOUNDRY_SYSTEM_PROMPT.includes("holo_inspect"));

  const fenced = parseToolCall(
    '<tool_call>\n```json\n{"name":"workspace_write","args":{"file_path":"src/App.tsx","code":"line1\nline2",}}\n```\n</tool_call>',
  );
  assert.deepEqual(fenced, {
    name: "workspace_write",
    arguments: {
      file_path: "src/App.tsx",
      code: "line1\nline2",
      path: "src/App.tsx",
      content: "line1\nline2",
    },
  });

  const aliasedBash = parseToolCall('<tool_call>{"tool":"bash","command":"uname -s"}</tool_call>');
  assert.deepEqual(aliasedBash, {
    name: "linux_exec",
    arguments: { command: "uname -s" },
  });

  const broken = parseToolCall("<tool_call>{broken: [</tool_call>");
  assert.equal(broken?.parseError, true);
  assert.match(broken?.error || "", /Malformed <tool_call> JSON/);
});

test("runUorR4AgentTurn retries on malformed JSON and enforces read-only non-build workspace_write", async () => {
  const retryRequests = [];
  let round = 0;
  const retryRes = await runUorR4AgentTurn({
    appId: 1,
    chatId: 1,
    modelId: "test-custom",
    messages: [],
    userPrompt: "Inspect workspace",
    chatMode: "build",
    tools: { workspace_list: async () => ({ ok: true, files: ["src/App.tsx"] }) },
    onText: () => {},
    generateFn: async (req) => {
      retryRequests.push(req);
      round += 1;
      if (round === 1) return { fullText: "<tool_call>{bad json</tool_call>" };
      return { fullText: "Recovered after JSON retry." };
    },
  });
  assert.equal(round, 2);
  assert.ok(retryRequests[1].prompt.includes("Malformed <tool_call> JSON"));
  assert.ok(retryRes.fullText.includes("Recovered after JSON retry."));

  for (const mode of ["chat", "ask", "plan"]) {
    let writeCalls = 0;
    let turn = 0;
    const requests = [];
    const roRes = await runUorR4AgentTurn({
      appId: 1,
      chatId: 2,
      modelId: "test-custom",
      messages: [],
      userPrompt: "Edit App.tsx",
      chatMode: mode,
      tools: {
        workspace_write: async () => {
          writeCalls += 1;
          return { ok: true };
        },
      },
      onText: () => {},
      generateFn: async (req) => {
        turn += 1;
        requests.push(req);
        if (turn === 1) {
          return {
            fullText: '<tool_call>{"name":"workspace_write","arguments":{"path":"src/App.tsx","content":"bad"}}</tool_call>',
          };
        }
        return { fullText: `Refused write in ${mode}.` };
      },
    });
    assert.equal(writeCalls, 0, `workspace_write must not run in ${mode} mode`);
    assert.ok(requests[1].prompt.includes('"ok":false'));
    assert.ok(roRes.fullText.includes(`Refused write in ${mode}.`));
  }
});

test("host parsePrismTags normalizes paths, deduplicates repeated writes, skips traversals, and counts unclosed tags", () => {
  const { cleanWorkspacePath, parsePrismTags } = loadParsePrismTagsFromHost();
  assert.equal(cleanWorkspacePath("/workspace/src/App.tsx"), "src/App.tsx");
  assert.equal(cleanWorkspacePath("./src/main.tsx"), "src/main.tsx");
  assert.throws(() => cleanWorkspacePath("../etc/passwd"), /Invalid workspace path/);

  const parsed = parsePrismTags(
    '<prism-write path="/workspace/src/App.tsx">first</prism-write>\n' +
      '<prism-write path="./src/App.tsx">second</prism-write>\n' +
      '<prism-write path="../escape.txt">skip</prism-write>\n' +
      '<prism-write path="src/Cut.tsx">unclosed',
  );
  assert.deepEqual(structuredClone(parsed.writes), [{ path: "src/App.tsx", description: null, content: "second" }]);
  assert.equal(parsed.unclosedWrites, 1);
});

test("commandRegistry enables Phase 2-3 channels and upstream-provenance.json matches local generated digests", () => {
  for (const ch of [
    "forge:export",
    "holo:import",
    "holo:inspect",
    "clear-session-data",
    "set-chat-favorite",
    "remove-chat-referenced-app",
  ]) {
    const entry = commandRegistry.entries.find((c) => c.channel === ch);
    assert.equal(entry?.availability, "adapter", `${ch} must be an adapter`);
    assert.equal(entry?.adapter, "studio/shell/host.js", `${ch} adapter mismatch`);
  }

  const upstream = JSON.parse(
    fs.readFileSync(new URL("../generated/upstream-provenance.json", import.meta.url), "utf8"),
  );
  assert.ok(/^[0-9a-f]{64}$/.test(upstream.attestation_id), "attestation_id must be 64-hex");
  for (const field of PROVENANCE_FIELDS) {
    assert.ok(typeof upstream[field] === "string" && upstream[field].length > 0, `missing field ${field}`);
  }
  const sha = (rel) =>
    createHash("sha256").update(fs.readFileSync(new URL(rel, import.meta.url))).digest("hex");
  assert.equal(sha("../generated/kernel.ir"), upstream.kernel_ir_sha256);
  assert.equal(sha("../generated/coverage.json"), upstream.coverage_sha256);
  assert.equal(sha("../generated/roots.json"), upstream.roots_sha256);
  assert.equal(sha("../generated/hologram_forge_core.rs"), upstream.generated_core_sha256);
});
