import assert from 'node:assert/strict';
import fs from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { test } from 'node:test';
import { fileURLToPath } from 'node:url';
import { buildStandardsIndex } from '../../../../../scripts/studio/build-standards-index.mjs';
import {
  FOUNDRY_SYSTEM_PROMPT,
  runUorR4AgentTurn,
} from '../shell/r4-agent-loop.mjs';

const __dirname = dirname(fileURLToPath(import.meta.url));
const shellDir = resolve(__dirname, '../shell');

test('buildStandardsIndex adds Studio architecture, tools/.holo, and browser-API sections within 110 KB', () => {
  const dir = fs.mkdtempSync(join(tmpdir(), 'uor-std-4a-'));
  const out = join(dir, 'standards-index.json');
  try {
    const info = buildStandardsIndex(out);
    assert.ok(info.bytes < 110 * 1024, `standards-index.json exceeded 110 KB: ${info.bytes}`);
    const parsed = JSON.parse(fs.readFileSync(out, 'utf8'));
    const ids = new Set((parsed.sections || []).map((s) => s.id));
    assert.ok(ids.has('foundry-studio-architecture'), 'missing foundry-studio-architecture');
    assert.ok(ids.has('foundry-tools-and-holo'), 'missing foundry-tools-and-holo');
    assert.ok(ids.has('foundry-browser-apis-and-vm'), 'missing foundry-browser-apis-and-vm');
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

test('runUorR4AgentTurn puts user prompt before groundingContext, preserves KV prefix across rounds, and never mutates caller messages', async () => {
  const callerMessages = [{ role: 'user', content: 'prev' }, { role: 'assistant', content: 'ok' }];
  const snapshot = structuredClone(callerMessages);
  const requests = [];
  let callIdx = 0;

  const res = await runUorR4AgentTurn({
    appId: 42,
    chatId: 'chat-4a',
    modelId: 'uor-r4:qwen3-0.6b-foundry',
    messages: callerMessages,
    userPrompt: 'Explain Studio modes',
    groundingContext: '[Foundry Context]\nWorkspace files (1): src/App.tsx',
    chatMode: 'build',
    generateFn: async (req) => {
      requests.push(req);
      callIdx += 1;
      if (callIdx === 1) {
        return {
          fullText: '<tool_call>{"name":"workspace_list","arguments":{"dir":"root"}}</tool_call>',
          weightsMatched: true,
          prefixReusedTokens: 285,
          draftProposed: 0,
          draftAccepted: 0,
        };
      }
      return {
        fullText: 'Foundry has Home Chat, Build, and Local Agent modes.',
        weightsMatched: true,
        prefixReusedTokens: 312,
        draftProposed: 4,
        draftAccepted: 2,
      };
    },
    tools: {
      workspace_list: async () => ({ ok: true, dir: 'root', files: ['src/App.tsx'] }),
    },
    onText: async () => {},
  });

  assert.deepEqual(callerMessages, snapshot, 'caller messages array must not be mutated');
  assert.equal(requests.length, 2);
  assert.equal(requests[0].maxNewTokens, 768);
  assert.equal(requests[0].speculative, undefined);
  assert.equal(requests[0].systemPrompt, FOUNDRY_SYSTEM_PROMPT);
  assert.equal(requests[1].systemPrompt, FOUNDRY_SYSTEM_PROMPT);
  assert.ok(
    requests[0].prompt.startsWith('Explain Studio modes\n\n[Foundry Context]\nWorkspace files'),
    `expected user prompt before groundingContext, got: ${requests[0].prompt}`,
  );
  // Round 1 must preserve round 0's combined user message at the start of turn extension for KV reuse.
  const r1UserEntry = requests[1].messages.find(
    (m) => m.role === 'user' && m.content.startsWith('Explain Studio modes'),
  );
  assert.equal(r1UserEntry?.content, requests[0].prompt);
  assert.equal(res.toolExecutions, 1);
  assert.equal(res.prefixReusedTokens, 285);
  assert.equal(res.draftProposed, 4);
  assert.equal(res.draftAccepted, 2);
});

test('Phase 4A worker, runtime, and host contracts keep MessageChannel yields, offline manifests, keys-only env grounding, and cold-SW waits', () => {
  const workerSrc = fs.readFileSync(join(shellDir, 'uor-v13-worker.mjs'), 'utf8');
  assert.match(workerSrc, /mapLimit\(manifest\.packs,\s*4,/);
  assert.match(workerSrc, /typeof MessageChannel === 'function'/);
  assert.match(workerSrc, /await yieldMacrotask\(\);/);
  assert.doesNotMatch(workerSrc, /await new Promise\(\(r\) => setTimeout\(r,\s*0\)\);/);
  assert.match(workerSrc, /cacheRead\(expectedContainerSha,\s*'manifest\.json'\)/);
  assert.match(workerSrc, /cacheRead\(manifest\.container_sha256,\s*'manifest\.stitch\.json'\)/);
  assert.match(workerSrc, /if \(!manifestFromCache\)\s*\{\s*await cacheWrite\(manifest\.container_sha256,\s*'manifest\.json'/);
  assert.match(workerSrc, /res\.message \|\| res\.detail \|\| res\.error/);
  assert.match(workerSrc, /speculative:\s*payload\.speculative === true/);
  assert.match(workerSrc, /action === 'prewarm_v13'/);
  assert.match(workerSrc, /action === 'configure_v13'/);

  const runtimeSrc = fs.readFileSync(join(shellDir, 'uor-r4-runtime.mjs'), 'utf8');
  assert.match(runtimeSrc, /let defaultSpeculative = false;/);
  assert.match(runtimeSrc, /cfg\.mlp_policy = mlpMode;/);
  assert.match(runtimeSrc, /hasOfflineV13Cache[\s\S]*'manifest\.json'[\s\S]*'manifest\.stitch\.json'[\s\S]*'pack_09\.pack'/);
  assert.match(runtimeSrc, /export async function prewarmIfCached\(/);
  assert.match(runtimeSrc, /export async function configure\(/);
  assert.match(runtimeSrc, /maxNewTokens = 768[\s\S]*sliceTokens = 4[\s\S]*prefillSliceTokens = 32/);

  const hostSrc = fs.readFileSync(join(shellDir, 'host.js'), 'utf8');
  assert.match(hostSrc, /import\s*\{runUorR4AgentTurn,FOUNDRY_SYSTEM_PROMPT\}\s*from\s*'\.\/r4-agent-loop\.mjs';/);
  assert.match(hostSrc, /async function buildPreview\(appId\)\s*\{\s*await coreReady;\s*await workerActive;/);
  assert.match(hostSrc, /async function compileTailwind\(page,\s*lease\)[\s\S]*await workerActive;\s*await lease\.check\(\);/);
  assert.match(hostSrc, /async function buildTurnGroundingContext\(/);
  assert.match(hostSrc, /const envKeys = envVars\.map\(\(v\) => v\.key\)\.filter\(Boolean\)/);
  assert.match(hostSrc, /uorR4\.prewarmIfCached\(\{\s*systemPrompt:\s*FOUNDRY_SYSTEM_PROMPT\s*\}\)/);
});
