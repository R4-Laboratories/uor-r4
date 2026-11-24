// FoundrySpecification.chatProviderRoutingBoundary: no implicit engine substitution.
import { LargeLanguageModelSchema } from './chat-contracts.mjs';

export const UOR_R4_PROVIDER = 'uor-r4';
export const CUSTOM_API_PROVIDER = 'custom-api';
export const UOR_R4_DEFAULT_MODEL = 'uor-r4:qwen3-0.6b';
export const LOCAL_MODEL = 'foundry:reader';
export const ALLOWED_UOR_R4_MODELS = new Set([
  'qwen3-0.6b',
  'uor-r4:qwen3-0.6b',
  'qwen3-1.7b',
  'uor-r4:qwen3-1.7b',
  'qwen3-0.6b-foundry',
  'uor-r4:qwen3-0.6b-foundry',
]);
export function normalizeUorR4ModelName(raw) {
  const clean = String(raw || '').trim();
  const short = clean.startsWith('uor-r4:') ? clean.slice('uor-r4:'.length) : clean;
  if (short === 'qwen3-0.6b' || short === 'qwen3-0.6b-foundry') {
    return 'uor-r4:qwen3-0.6b';
  }
  if (short === 'qwen3-1.7b') {
    return 'uor-r4:qwen3-1.7b';
  }
  return null;
}
const FORMER = ['webgpu:Qwen', 'webgpu:Bonsai', 'webgpu:BitNet'];

const fail = (code) =>
  Object.assign(
    new Error(
      code === 'unsupportedChatProvider'
        ? 'The chat provider is unsupported'
        : 'The chat model selection is invalid',
    ),
    { code },
  );

export function chatModelRoute(chatSelection, defaultSelection) {
  const chosen = chatSelection ?? defaultSelection;
  try {
    LargeLanguageModelSchema.parse(chosen);
    for (const value of [chosen?.provider, chosen?.name]) {
      if (
        typeof value !== 'string' ||
        !value.length ||
        value.length > 256 ||
        value.trim() !== value ||
        /[\x00-\x1f\x7f]/.test(value)
      ) {
        throw Error();
      }
    }
  } catch {
    throw fail('invalidChatModelSelection');
  }
  const { provider, name } = chosen;
  if (!['uor-r4', 'custom-api', 'local', 'openrouter', 'featherless'].includes(provider)) {
    throw fail('unsupportedChatProvider');
  }
  if (provider === 'uor-r4') {
    const normalized = normalizeUorR4ModelName(name);
    if (!normalized) {
      throw fail('invalidChatModelSelection');
    }
    return Object.freeze({ provider, model: normalized, paid: false });
  }
  if (provider === 'custom-api') {
    return Object.freeze({ provider, model: name, paid: true });
  }
  if (provider === 'local' && name !== LOCAL_MODEL && !FORMER.includes(name)) {
    throw fail('invalidChatModelSelection');
  }
  return Object.freeze({
    provider,
    model: provider === 'local' ? LOCAL_MODEL : `${provider}/${name}`,
    paid: provider !== 'local',
  });
}
