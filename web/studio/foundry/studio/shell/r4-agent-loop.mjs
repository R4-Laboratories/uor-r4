// Real UOR-R4 V13 Qwen3-0.6B & Qwen3-1.7B + Foundry Webharness Agent Loop.
// Separates general conversational Chat mode (`FOUNDRY_CHAT_SYSTEM_PROMPT`) from
// Build/Agent file authoring (`FOUNDRY_SYSTEM_PROMPT`) so Chat replies naturally
// and Build produces compact, complete `src/App.tsx` apps that close cleanly.

export const FOUNDRY_CHAT_SYSTEM_PROMPT =
  'You are UOR-R4 Foundry Assistant, running locally inside your user\'s browser on Hologram Foundry\'s pure-integer WebAssembly engine (zero floats, zero dense matmul, zero softmax — not a standard cloud transformer) alongside a local virtual workstation (Linux 6.18.7 VM, RV64IMAC MVM, Studio workspace, GitHub sync, PrismPM/.holo, LexLean 0.3.0, UOR Standards, Stitch).\n' +
  'Answer questions directly in clear, concise natural language without listing tools unless the user asks for an action.\n' +
  'When the user explicitly asks to inspect files, run shell/MVM commands, sync GitHub, verify PrismPM/LexLean, look up UOR standards, or fetch via Stitch, emit a single JSON call inside <tool_call>{"name": "...", "arguments": {...}}</tool_call>.\n' +
  'Available tools: workspace_read(path, start_line?, end_line?), workspace_write(path, content), workspace_list(dir), linux_exec(command, cwd?), mvm_exec(command), github_status(repo?), github_list_repos(owner?), github_clone(repo, branch?), github_pull(repo, branch?), github_push(repo, branch, message), prismpm_verify(target, action?), holo_publish(), holo_inspect(kappa?), lexlean_check(path), lexlean_verify(path, profile?), standards_lookup(query), stitch_fetch(url, method?), stitch_peer_status(peer_id?).';

export const FOUNDRY_BUILD_SYSTEM_PROMPT =
  'You are an expert React, TypeScript, and Tailwind CSS software engineer inside UOR-R4 Foundry Studio.\n' +
  'Build complete, working, interactive React applications directly in `src/App.tsx` (never placeholder stubs).\n' +
  'Always emit the file using raw XML tags:\n' +
  '<prism-write path="src/App.tsx" description="short summary">\n' +
  '// complete TypeScript React code\n' +
  '</prism-write>\n' +
  'Rules:\n' +
  '1. Import React hooks explicitly (e.g. `import { useState } from \'react\';`) and export the root component with `export default function App()`.\n' +
  '2. Do NOT put ``` markdown fences inside or around <prism-write>.\n' +
  '3. Keep the code clean and compact (~45-85 lines) so the entire file finishes cleanly before </prism-write>.\n' +
  '4. When modifying an existing app, inspect Current src/App.tsx (and any Last preview build error), preserve existing features, and emit the complete updated src/App.tsx inside <prism-write path="src/App.tsx" description="short summary">...</prism-write>.';

export const FOUNDRY_SYSTEM_PROMPT =
  'You are UOR-R4 Foundry Assistant & Software Engineer running locally in-browser on pure-integer WASM (zero floats/matmul/softmax) inside Hologram Foundry.\n' +
  'When editing or creating app files in Build mode, emit <prism-search-replace path="src/App.tsx"><search>...</search><replace>...</replace></prism-search-replace> for surgical updates or <prism-write path="src/App.tsx" description="short summary">COMPLETE FILE CONTENT</prism-write> with NO markdown ``` fences.\n' +
  'When executing workspace, VM, GitHub, PrismPM/.holo, LexLean, Standards, or Stitch actions, emit a single JSON call inside <tool_call>{"name": "...", "arguments": {...}}</tool_call>.\n' +
  'Available tools: workspace_read(path, start_line?, end_line?), workspace_write(path, content), workspace_list(dir), linux_exec(command, cwd?), mvm_exec(command), github_status(repo?), github_list_repos(owner?), github_clone(repo, branch?), github_pull(repo, branch?), github_push(repo, branch, message), prismpm_verify(target, action?), holo_publish(), holo_inspect(kappa?), lexlean_check(path), lexlean_verify(path, profile?), standards_lookup(query), stitch_fetch(url, method?), stitch_peer_status(peer_id?).';

const BUILD_TOOL_INTENT_RE =
  /\b(workspace_read|workspace_write|workspace_list|linux_exec|mvm_exec|github_status|github_list_repos|github_clone|github_pull|github_push|prismpm_verify|holo_publish|holo_inspect|lexlean_check|lexlean_verify|standards_lookup|stitch_fetch|stitch_peer_status)\b|\b(github_status|github_list_repos|github_clone|github_pull|github_push|clone|pull|push)\b.*\b(repo|github|branch)\b|\b(run|exec|execute)\s+.*(command|linux|mvm|shell|bash)\b|\b(list|read|inspect)\s+(the\s+)?(files?|workspace|dir|directory)\b|\b(publish|inspect)\s+.*\.?holo\b|\b(lexlean|prismpm|stitch_peer|standards_lookup)\b/i;

export const FOUNDRY_TOOL_NAMES = Object.freeze([
  'workspace_read',
  'workspace_write',
  'workspace_list',
  'linux_exec',
  'mvm_exec',
  'github_status',
  'github_list_repos',
  'github_clone',
  'github_pull',
  'github_push',
  'prismpm_verify',
  'holo_publish',
  'holo_inspect',
  'lexlean_check',
  'lexlean_verify',
  'standards_lookup',
  'stitch_fetch',
  'stitch_peer_status',
]);

const MAX_TOOL_ROUNDS = 6;

function repairToolJson(text) {
  let src = String(text || '').trim();
  const fence = src.match(/^```(?:json)?\s*([\s\S]*?)\s*```$/i);
  if (fence) src = fence[1].trim();
  const start = src.indexOf('{');
  if (start < 0) return null;
  src = src.slice(start);
  // Replace raw control newlines inside string literals and drop trailing commas before closers.
  let inStr = false;
  let esc = false;
  let balanced = '';
  const stack = [];
  for (let i = 0; i < src.length; i += 1) {
    const ch = src[i];
    if (inStr) {
      if (esc) {
        esc = false;
        balanced += ch;
      } else if (ch === '\\') {
        esc = true;
        balanced += ch;
      } else if (ch === '"') {
        inStr = false;
        balanced += ch;
      } else if (ch === '\n') {
        balanced += '\\n';
      } else if (ch === '\r') {
        balanced += '\\r';
      } else {
        balanced += ch;
      }
      continue;
    }
    balanced += ch;
    if (ch === '"') {
      inStr = true;
    } else if (ch === '{') {
      stack.push('}');
    } else if (ch === '[') {
      stack.push(']');
    } else if (ch === '}' || ch === ']') {
      if (stack.length && stack[stack.length - 1] === ch) {
        stack.pop();
        if (stack.length === 0) break;
      }
    }
  }
  const closed = (inStr ? balanced + '"' : balanced)
    .replace(/,\s*([}\]])/g, '$1')
    .replace(/[,:\s]+$/, '') + stack.reverse().join('');
  try {
    return JSON.parse(closed);
  } catch {
    return null;
  }
}

export function parseToolCall(text) {
  const src = String(text || '');
  const match = src.match(/<tool_call>\s*([\s\S]*?)\s*(?:<\/tool_call>|$)/);
  if (!match) return null;
  const raw = match[1].trim();
  if (!raw) {
    return { parseError: true, error: 'Empty <tool_call> block. Provide JSON {"name": "...", "arguments": {...}}.' };
  }
  const fence = raw.match(/^```(?:json)?\s*([\s\S]*?)\s*```$/i);
  const candidate = fence ? fence[1].trim() : raw;
  let parsed = null;
  try {
    parsed = JSON.parse(candidate);
  } catch {
    parsed = repairToolJson(candidate);
  }
  if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
    return {
      parseError: true,
      error: 'Malformed <tool_call> JSON. Emit valid JSON inside <tool_call>{"name":"...","arguments":{...}}</tool_call>.',
    };
  }
  const rawName = String(parsed.name || parsed.tool || '').trim();
  if (!rawName) {
    return {
      parseError: true,
      error: 'Missing tool "name" inside <tool_call>.',
    };
  }
  const rawArgs =
    parsed.arguments && typeof parsed.arguments === 'object' && !Array.isArray(parsed.arguments)
      ? { ...parsed.arguments }
      : parsed.args && typeof parsed.args === 'object' && !Array.isArray(parsed.args)
        ? { ...parsed.args }
        : {};
  if (rawName === 'bash' || rawName === 'shell' || rawName === 'exec' || rawName === 'terminal') {
    return {
      name: 'linux_exec',
      arguments: {
        ...rawArgs,
        command: String(rawArgs.command || rawArgs.cmd || parsed.command || ''),
      },
    };
  }
  if (rawName === 'mvm') {
    return {
      name: 'mvm_exec',
      arguments: {
        ...rawArgs,
        command: String(rawArgs.command || rawArgs.cmd || parsed.command || ''),
      },
    };
  }
  if (rawName === 'workspace_read' || rawName === 'workspace_write') {
    if (rawArgs.path === undefined) {
      rawArgs.path = rawArgs.file_path ?? rawArgs.filename ?? rawArgs.file ?? parsed.path;
    }
    if (rawName === 'workspace_write' && rawArgs.content === undefined) {
      rawArgs.content = rawArgs.text ?? rawArgs.code ?? rawArgs.body ?? '';
    }
  } else if (rawName === 'workspace_list' && rawArgs.dir === undefined) {
    rawArgs.dir = rawArgs.path ?? rawArgs.directory ?? 'root';
  }
  return { name: rawName, arguments: rawArgs };
}

export async function dispatchFoundryTool(call, tools) {
  const { name, arguments: args } = call;
  if (!FOUNDRY_TOOL_NAMES.includes(name)) {
    return { ok: false, error: `Unknown tool: ${name}` };
  }
  const fn = tools?.[name];
  if (typeof fn !== 'function') {
    return { ok: false, error: `Tool ${name} is not wired in this context` };
  }
  try {
    const result = await fn(args || {});
    return result && typeof result === 'object' ? result : { ok: true, result };
  } catch (err) {
    return { ok: false, error: String(err?.message || err) };
  }
}

export async function runUorR4AgentTurn({
  appId = null,
  chatId,
  modelId,
  messages,
  userPrompt,
  groundingContext = '',
  chatMode = 'build',
  maxNewTokens = 768,
  maxToolRounds = MAX_TOOL_ROUNDS,
  speculative,
  signal,
  uorR4Runtime,
  generateFn,
  mvmHost,
  tools = {},
  onText,
}) {
  const prompt = String(userPrompt || '').trim();
  const grounding = String(groundingContext || '').trim();
  const round0Prompt = grounding ? `${prompt}\n\n${grounding}` : prompt;

  // Explicit user slash-command runs RV64IMAC MVM or Linux VM directly.
  const directMatch = prompt.match(/^\/(mvm|linux|bash|sh)\s+([\s\S]+)$/i);
  if (directMatch) {
    const kind = directMatch[1].toLowerCase();
    const cmd = directMatch[2].trim();
    if (kind !== 'mvm' && typeof tools.linux_exec === 'function') {
      const res = await tools.linux_exec({ command: cmd });
      const body = [
        `**Linux 6.18.7 Shell** (\`${cmd}\`) · exit \`${res.exit_code ?? 0}\` · cwd \`${res.cwd || '/workspace'}\``,
        res.stdout ? `\`\`\`\n${String(res.stdout).trimEnd()}\n\`\`\`` : '',
        res.stderr ? `**stderr:**\n\`\`\`\n${String(res.stderr).trimEnd()}\n\`\`\`` : '',
      ]
        .filter(Boolean)
        .join('\n\n');
      await onText(body);
      return {
        fullText: body,
        toolExecutions: 1,
        updatedFiles: Boolean(res.updatedFiles),
        weightsMatched: true,
      };
    }
    const res = await mvmHost.execCommand(appId, cmd, { echoToSession: true });
    const body = [
      `**MVM RV64IMAC Shell** (\`${cmd}\`) · exit \`${res.exit_code}\` · retired \`${res.instructions_retired}\` · elf \`${res.compiled_elf_machine}\` · cwd \`${res.cwd}\``,
      res.stdout ? `\`\`\`\n${res.stdout.trimEnd()}\n\`\`\`` : '',
      res.stderr ? `**stderr:**\n\`\`\`\n${res.stderr.trimEnd()}\n\`\`\`` : '',
      `CoW SHA-256: \`${res.cow_state_sha256}\``,
    ]
      .filter(Boolean)
      .join('\n\n');
    await onText(body);
    return {
      fullText: body,
      toolExecutions: 1,
      updatedFiles: false,
      weightsMatched: true,
    };
  }

  let transcript = '';
  let turnMessages = Array.isArray(messages) ? [...messages] : [];
  let weightsMatched = true;
  let toolExecutions = 0;
  let updatedFiles = false;
  let prefixReusedTokens = 0;
  let draftProposed = 0;
  let draftAccepted = 0;
  const isV13 = /^((uor-r4:)?qwen3-)/i.test(String(modelId || ''));
  const hasCustomGenerate = typeof generateFn === 'function';
  const useFoundryPrompt = isV13 || hasCustomGenerate;
  const wantsToolPrompt = chatMode === 'local-agent' || (chatMode === 'build' && BUILD_TOOL_INTENT_RE.test(prompt));
  const activeSystemPrompt = wantsToolPrompt
    ? FOUNDRY_SYSTEM_PROMPT
    : chatMode === 'build'
      ? FOUNDRY_BUILD_SYSTEM_PROMPT
      : FOUNDRY_CHAT_SYSTEM_PROMPT;

  const maxRounds =
    Number.isInteger(maxToolRounds) && maxToolRounds >= 1 && maxToolRounds <= 12
      ? maxToolRounds
      : MAX_TOOL_ROUNDS;

  for (let round = 0; round <= maxRounds; round += 1) {
    const roundPrefix = transcript;
    const genReq = {
      modelId,
      chatId: `${chatId}-r${round}`,
      systemPrompt: useFoundryPrompt ? activeSystemPrompt : '',
      prompt: round === 0 ? round0Prompt : turnMessages[turnMessages.length - 1].content,
      messages: round === 0 ? turnMessages : turnMessages.slice(0, -1),
      maxNewTokens: useFoundryPrompt ? maxNewTokens : 48,
      ...(useFoundryPrompt && typeof speculative === 'boolean' ? { speculative } : {}),
      stopSequences: useFoundryPrompt ? ['</tool_call>'] : undefined,
      signal,
      onToken: ({ fullText }) => {
        void onText(roundPrefix + fullText);
      },
    };
    const gen = hasCustomGenerate
      ? await generateFn(genReq)
      : await uorR4Runtime.generate(genReq);
    if (!gen.weightsMatched) {
      weightsMatched = false;
    }
    if (round === 0) {
      prefixReusedTokens = gen.prefixReusedTokens ?? 0;
    }
    draftProposed += gen.draftProposed ?? 0;
    draftAccepted += gen.draftAccepted ?? 0;
    const cleanOut = String(gen.fullText || '').trim();
    transcript = roundPrefix + cleanOut;
    await onText(transcript);

    const call = parseToolCall(cleanOut);
    if (!call || round === maxRounds) {
      break;
    }
    if (call.parseError) {
      const refusal = { ok: false, error: call.error };
      const responseBlock = `\n<tool_response>${JSON.stringify(refusal)}</tool_response>\n`;
      transcript += responseBlock;
      await onText(transcript);
      const basePrior =
        round === 0
          ? [...turnMessages, { role: 'user', content: round0Prompt }]
          : turnMessages;
      turnMessages = [
        ...basePrior,
        { role: 'assistant', content: cleanOut },
        { role: 'user', content: `<tool_response>${JSON.stringify(refusal)}</tool_response>` },
      ];
      continue;
    }
    // Enforce read-only guard whenever chatMode is not a file-mutating Build mode.
    if (chatMode !== 'build' && chatMode !== 'local-agent' && call.name === 'workspace_write') {
      const refusal = {
        ok: false,
        error:
          chatMode === 'chat'
            ? 'Home Chat is read-only. Click "Build this in Studio" to write files.'
            : 'Read-only mode; switch to Build mode to write workspace files.',
      };
      const responseBlock = `\n<tool_response>${JSON.stringify(refusal)}</tool_response>\n`;
      transcript += responseBlock;
      await onText(transcript);
      const basePrior =
        round === 0
          ? [...turnMessages, { role: 'user', content: round0Prompt }]
          : turnMessages;
      turnMessages = [
        ...basePrior,
        { role: 'assistant', content: cleanOut },
        { role: 'user', content: `<tool_response>${JSON.stringify(refusal)}</tool_response>` },
      ];
      continue;
    }

    toolExecutions += 1;
    const result = await dispatchFoundryTool(call, tools);
    if (result?.updatedFiles) {
      updatedFiles = true;
    }
    const compact = JSON.stringify(result);
    const bounded = compact.length > 6000 ? compact.slice(0, 6000) + '..."}' : compact;
    const responseBlock = `\n<tool_response>${bounded}</tool_response>\n`;
    transcript += responseBlock;
    await onText(transcript);

    const basePrior =
      round === 0
        ? [...turnMessages, { role: 'user', content: round0Prompt }]
        : turnMessages;
    turnMessages = [
      ...basePrior,
      { role: 'assistant', content: cleanOut },
      { role: 'user', content: `<tool_response>${bounded}</tool_response>` },
    ];
  }

  return {
    fullText: transcript,
    toolExecutions,
    updatedFiles,
    weightsMatched,
    prefixReusedTokens,
    draftProposed,
    draftAccepted,
  };
}
