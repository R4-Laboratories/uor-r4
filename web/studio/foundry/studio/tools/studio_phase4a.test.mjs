import assert from 'node:assert/strict';
import fs from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { test } from 'node:test';
import { fileURLToPath } from 'node:url';
import { buildStandardsIndex } from '../../../../../scripts/studio/build-standards-index.mjs';
import {
  FOUNDRY_BUILD_SYSTEM_PROMPT,
  FOUNDRY_CHAT_SYSTEM_PROMPT,
  FOUNDRY_SYSTEM_PROMPT,
  runUorR4AgentTurn,
} from '../shell/r4-agent-loop.mjs';

const __dirname = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(__dirname, '../../../../..');
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
    modelId: 'uor-r4:qwen3-0.6b',
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
  assert.equal(requests[0].systemPrompt, FOUNDRY_BUILD_SYSTEM_PROMPT);
  assert.equal(requests[1].systemPrompt, FOUNDRY_BUILD_SYSTEM_PROMPT);
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
  assert.match(workerSrc, /mapLimit\(manifest\.packs,\s*6,/);
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
  assert.match(runtimeSrc, /let defaultSpeculative = true;/);
  assert.match(runtimeSrc, /cfg\.mlp_policy = mlpMode;/);
  assert.match(runtimeSrc, /hasOfflineV13Cache[\s\S]*'manifest\.json'[\s\S]*'manifest\.stitch\.json'[\s\S]*lastPack/);
  assert.match(runtimeSrc, /export async function prewarmIfCached\(/);
  assert.match(runtimeSrc, /export async function configure\(/);
  assert.match(runtimeSrc, /maxNewTokens = 768[\s\S]*sliceTokens = 4[\s\S]*prefillSliceTokens = 32/);

  const hostSrc = fs.readFileSync(join(shellDir, 'host.js'), 'utf8');
  assert.match(hostSrc, /import\s*\{runUorR4AgentTurn,FOUNDRY_SYSTEM_PROMPT,FOUNDRY_CHAT_SYSTEM_PROMPT,FOUNDRY_BUILD_SYSTEM_PROMPT\}\s*from\s*'\.\/r4-agent-loop\.mjs';/);
  assert.match(hostSrc, /async function buildPreview\(appId\)\s*\{\s*await coreReady;\s*await workerActive;/);
  assert.match(hostSrc, /async function compileTailwind\(page,\s*lease\)[\s\S]*await workerActive;\s*await lease\.check\(\);/);
  assert.match(hostSrc, /async function buildTurnGroundingContext\(/);
  assert.match(hostSrc, /const envKeys = envVars\.map\(\(v\) => v\.key\)\.filter\(Boolean\)/);
  assert.match(hostSrc, /uorR4\.prewarmIfCached\(\{\s*systemPrompt:\s*FOUNDRY_CHAT_SYSTEM_PROMPT\s*\}\)/);
  assert.match(hostSrc, /FOUNDRY_DOMAIN_RE\.test\(q\)/);
  assert.match(hostSrc, /function extractFallbackCodeWrites\(/);
  assert.match(hostSrc, /function compactBuildHistory\(/);
  assert.match(workerSrc, /const packCached = new Set\(\);/);
  assert.match(workerSrc, /const orderedPacks = \[\.\.\.manifest\.packs\]/);
  assert.ok(FOUNDRY_CHAT_SYSTEM_PROMPT !== FOUNDRY_SYSTEM_PROMPT, 'Chat and tool system prompts must stay separate');
  assert.ok(FOUNDRY_BUILD_SYSTEM_PROMPT !== FOUNDRY_SYSTEM_PROMPT, 'Build app-authoring and tool prompts must stay separate');
});

test('runUorR4AgentTurn uses FOUNDRY_CHAT_SYSTEM_PROMPT in chat mode, FOUNDRY_BUILD_SYSTEM_PROMPT for Build app authoring, and FOUNDRY_SYSTEM_PROMPT for tool requests across 0.6B and 1.7B', async () => {
  for (const modelId of ['uor-r4:qwen3-0.6b', 'uor-r4:qwen3-1.7b']) {
    const seen = [];
    await runUorR4AgentTurn({
      appId: null,
      chatId: `chat-mode-${modelId}`,
      modelId,
      messages: [],
      userPrompt: 'hi how are you',
      chatMode: 'chat',
      generateFn: async (req) => {
        seen.push(req.systemPrompt);
        return { fullText: 'Doing well!', weightsMatched: true };
      },
      tools: {},
      onText: async () => {},
    });
    await runUorR4AgentTurn({
      appId: 9,
      chatId: `build-mode-${modelId}`,
      modelId,
      messages: [],
      userPrompt: 'build a todo app',
      chatMode: 'build',
      generateFn: async (req) => {
        seen.push(req.systemPrompt);
        return { fullText: '<prism-write path="src/App.tsx" description="todo">export default function App(){return <div/>;}</prism-write>', weightsMatched: true };
      },
      tools: {},
      onText: async () => {},
    });
    await runUorR4AgentTurn({
      appId: 9,
      chatId: `tool-mode-${modelId}`,
      modelId,
      messages: [],
      userPrompt: 'Call workspace_list for root.',
      chatMode: 'build',
      generateFn: async (req) => {
        seen.push(req.systemPrompt);
        return { fullText: 'Done.', weightsMatched: true };
      },
      tools: {},
      onText: async () => {},
    });
    await runUorR4AgentTurn({
      appId: 9,
      chatId: `build-mode-grounded-${modelId}`,
      modelId,
      messages: [],
      userPrompt: 'Build a UOR standards dashboard app',
      groundingContext: '[Foundry Context]\n- tool docs mention workspace_write and lexlean_check',
      chatMode: 'build',
      generateFn: async (req) => {
        seen.push(req.systemPrompt);
        return { fullText: '<prism-write path="src/App.tsx" description="uor">export default function App(){return <div/>;}</prism-write>', weightsMatched: true };
      },
      tools: {},
      onText: async () => {},
    });
    assert.equal(seen[0], FOUNDRY_CHAT_SYSTEM_PROMPT, `${modelId} chat mode must use FOUNDRY_CHAT_SYSTEM_PROMPT`);
    assert.equal(seen[1], FOUNDRY_BUILD_SYSTEM_PROMPT, `${modelId} build authoring must use FOUNDRY_BUILD_SYSTEM_PROMPT`);
    assert.equal(seen[2], FOUNDRY_SYSTEM_PROMPT, `${modelId} build tool request must use FOUNDRY_SYSTEM_PROMPT`);
    assert.equal(seen[3], FOUNDRY_BUILD_SYSTEM_PROMPT, `${modelId} groundingContext keywords must not hijack BUILD_TOOL_INTENT_RE`);
  }
});

test('transformProjectSource preserves multi-line imports/destructuring, adds missing hooks, and extractFallbackCodeWrites blocks path traversal', async () => {
  const vm = await import('node:vm');
  const hostSrc = fs.readFileSync(join(shellDir, 'host.js'), 'utf8');
  const cleanStart = hostSrc.indexOf('function cleanWorkspacePath(');
  const cleanEnd = hostSrc.indexOf('\nfunction createFoundryTools(', cleanStart);
  const fbStart = hostSrc.indexOf('function looksLikeCompleteComponent(');
  const fbEnd = hostSrc.indexOf('\nfunction compactBuildHistory(', fbStart);
  const trStart = hostSrc.indexOf('function transformProjectSource(');
  const trEnd = hostSrc.indexOf('\nfunction createProjectResolverPlugin(', trStart);
  const { extractFallbackCodeWrites, transformProjectSource } = vm.runInNewContext(
    hostSrc.slice(cleanStart, cleanEnd) +
      '\n' +
      hostSrc.slice(fbStart, fbEnd) +
      '\n' +
      hostSrc.slice(trStart, trEnd) +
      '\n({ extractFallbackCodeWrites, transformProjectSource })',
    {},
  );
  const recovered = extractFallbackCodeWrites(
    '<prism-write path="src/App.tsx" description="todo">\nexport default function App() {\n  return <><div>Don\'t fail</div></>;\n}\n```\nAll done!',
  );
  assert.equal(recovered.length, 1);
  assert.equal(recovered[0].path, 'src/App.tsx');
  assert.match(recovered[0].content, /Don't fail/);

  const traversal = extractFallbackCodeWrites(
    '<prism-write path="../evil.tsx">\nexport default function App() { return <div/>; }\n```',
  );
  assert.equal(traversal.length, 0, 'invalid/traversal path must be refused instead of falling back to src/App.tsx');

  const multiLine = 'import {\n  useState,\n  useEffect,\n} from "react";\nexport default function App() { const [x] = useState(0); useEffect(() => {}, []); return <div>{x}</div>; }';
  assert.equal((transformProjectSource('src/App.tsx', multiLine).match(/useState/g) || []).length, 2, 'multi-line import must not inject duplicate useState import');
  const destructured = 'import * as React from "react";\nconst { useState } = React;\nexport default function App() { const [x] = useState(0); return <div>{x}</div>; }';
  assert.ok(!transformProjectSource('src/App.tsx', destructured).includes('import { useState }'), 'destructured hook must not inject duplicate import');

  const workerSrc = fs.readFileSync(join(shellDir, 'uor-v13-worker.mjs'), 'utf8');
  assert.match(workerSrc, /mountedContainerSha = null;\s*mountedModelId = null;/);
  assert.match(workerSrc, /mountedContainerSha = containerSha;\s*mountedModelId = modelId;\s*try \{\s*wasm\.v13_configure/);
  assert.match(workerSrc, /createSyncAccessHandle/);
  assert.match(workerSrc, /pack_seal\.json/);
  const runtimeSrc = fs.readFileSync(join(shellDir, 'uor-r4-runtime.mjs'), 'utf8');
  assert.match(runtimeSrc, /export async function generate[\s\S]*serializeOp\(async \(\) => \{\s*await ensureModelLoadedQueued\(/);
  assert.match(runtimeSrc, /let defaultSpeculative = true;/);
  assert.match(hostSrc, /async function buildPortable[\s\S]*transformProjectSource\(args\.path,\s*src\)/);
  assert.match(hostSrc, /handlers\.set\("chat:cancel",\s*async chatId\s*=>\s*Boolean\(await chatQueue\.cancelCurrent\(chatId\)\)\)/);
  assert.match(hostSrc, /handlers\.set\("uor-r4:status",[\s\S]*prewarmMode/);
  const queueCtrlSrc = fs.readFileSync(join(shellDir, 'chat-queue-controller.mjs'), 'utf8');
  assert.match(queueCtrlSrc, /j\.pauseReason==='stop'\?\[/);
  assert.match(queueCtrlSrc, /return true;/);
});

test('parsePrismTags supports <prism-search-replace> XML/git-style hunks, blocks traversal, and gates UOR identity grounding', async () => {
  const vm = await import('node:vm');
  const hostSrc = fs.readFileSync(join(shellDir, 'host.js'), 'utf8');
  const cleanStart = hostSrc.indexOf('function cleanWorkspacePath(');
  const cleanEnd = hostSrc.indexOf('\nfunction createFoundryTools(', cleanStart);
  const parseStart = hostSrc.indexOf('function applySearchReplaceHunk(');
  const parseEnd = hostSrc.indexOf('\nasync function applyPrismTags(', parseStart);
  const { applySearchReplaceHunk, parsePrismTags } = vm.runInNewContext(
    hostSrc.slice(cleanStart, cleanEnd) +
      '\n' +
      hostSrc.slice(parseStart, parseEnd) +
      '\n({ applySearchReplaceHunk, parsePrismTags })',
    {},
  );
  const parsed = parsePrismTags(
    '<prism-search-replace path="src/App.tsx" description="filter">\n' +
      '<search>\nconst x = 1;\n</search>\n' +
      '<replace>\nconst x = 2;\n</replace>\n' +
      '</prism-search-replace>\n' +
      '<prism-search-replace path="../secret.tsx">\n<search>a</search><replace>b</replace>\n</prism-search-replace>',
  );
  assert.equal(parsed.patches.length, 1, 'traversal patch path must be dropped');
  assert.equal(parsed.patches[0].path, 'src/App.tsx');
  const original = 'import { useState } from "react";\n  const x = 1;\nexport default function App() { return <div>{x}</div>; }\n';
  const patched = applySearchReplaceHunk(original, parsed.patches[0].hunks[0].search, parsed.patches[0].hunks[0].replace);
  assert.ok(patched && patched.includes('const x = 2;') && !patched.includes('const x = 1;'));

  const gitStyle = parsePrismTags(
    '<prism-search-replace path="src/App.tsx">\n<<<<<<< SEARCH\nconst x = 1;\n=======\nconst x = 3;\n>>>>>>> REPLACE\n</prism-search-replace>',
  );
  assert.equal(gitStyle.patches.length, 1);
  assert.equal(gitStyle.patches[0].hunks[0].replace, 'const x = 3;');

  const unmatchedFence = parsePrismTags(
    '<prism-write path="src/App.tsx" description="fenced">\n```tsx\nexport default function App() { return <div/>; }\n</prism-write>\n```',
  );
  assert.equal(unmatchedFence.writes.length, 1);
  assert.equal(unmatchedFence.writes[0].content, 'export default function App() { return <div/>; }');

  assert.match(hostSrc, /are\\s\+you\\s\+a\?\\s\*transformer/);
  const screensSrc = fs.readFileSync(resolve(__dirname, '../vendor/prism/hologram.screens.mjs'), 'utf8');
  assert.equal(screensSrc.split('withdrawPendingPrompt();').length - 1, 2, 'hologram.screens.mjs must preserve 2 withdrawPendingPrompt(); occurrences across from/to');
  assert.match(screensSrc, /"uor-r4:status",\s*\{\s*prewarmMode:\s*"build"\s*\}/);
  const stdSrc = fs.readFileSync(join(repoRoot, 'scripts/studio/build-standards-index.mjs'), 'utf8');
  assert.match(stdSrc, /uor-r4-self-identity/);
});
