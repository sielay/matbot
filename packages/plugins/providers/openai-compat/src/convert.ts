import type { Message, Tool, JSONSchema } from '@matatbread/matbot-plugin-api';

// ── Internal OpenAI API types ─────────────────────────────────────────────────

type OAIRole    = 'system' | 'user' | 'assistant' | 'tool';

export interface OAIMessage {
  role:         OAIRole;
  content?:     string | OAIContentPart[] | null;
  tool_calls?:  OAIToolCall[];
  tool_call_id?: string;
  name?:        string;
}

type CacheControl = { type: 'ephemeral' };

type OAIContentPart =
  | { type: 'text';      text: string; cache_control?: CacheControl }
  | { type: 'image_url'; image_url: { url: string } };

interface OAIToolCall {
  id:       string;
  type:     'function';
  function: { name: string; arguments: string };
}

export interface OAIToolDef {
  type:     'function';
  function: { name: string; description: string; parameters: JSONSchema };
}

// ── Message conversion ────────────────────────────────────────────────────────

export function toOAIMessages(messages: Message[], opts?: { promptCaching?: boolean }): OAIMessage[] {
  const result: OAIMessage[] = [];

  for (const msg of messages) {
    if (msg.role === 'system') {
      const text = msg.content
        .filter(c => c.type === 'text')
        .map(c => (c as { type: 'text'; text: string }).text)
        .join('\n\n');
      result.push({ role: 'system', content: text });
      continue;
    }

    if (msg.role === 'tool') {
      for (const c of msg.content) {
        if (c.type === 'tool-result') {
          result.push({
            role:         'tool',
            tool_call_id: c.id,
            content:      JSON.stringify(c.result),
          });
        }
      }
      continue;
    }

    if (msg.role !== 'user' && msg.role !== 'assistant') continue;

    const toolCalls = msg.content.filter(c => c.type === 'tool-call');
    const parts     = msg.content.filter(c => c.type !== 'tool-call');

    const contentParts: OAIContentPart[] = parts.flatMap((c): OAIContentPart[] => {
      switch (c.type) {
        case 'text':      return [{ type: 'text', text: c.text }];
        case 'image':     return [{ type: 'image_url', image_url: { url: `data:${c.mimeType};base64,${c.data}` } }];
        case 'image-url': return [{ type: 'image_url', image_url: { url: c.url, ...(c.detail !== undefined ? { detail: c.detail } : {}) } }];
        case 'file-ref':  return [{ type: 'text', text: `[Attached file: ${c.name}]` }];
        case 'document':  return [{ type: 'text', text: `[Document: ${c.name ?? c.mimeType}]` }];
        case 'audio':     return [{ type: 'text', text: `[Audio: ${c.mimeType}]` }];
        case 'thinking':
        case 'redacted-thinking':
        case 'reasoning':
        case 'tool-result':    // only in role === 'tool' messages, handled above
        case 'refusal':
        case 'form':
        case 'form-response':
        case 'marker':         // opaque UI annotation; transparent to the model
        case 'unknown-content':
          return [];
      }
    });

    let content: string | OAIContentPart[] | null = null;
    const first = contentParts[0];
    if (contentParts.length === 1 && first !== undefined && first.type === 'text') {
      content = first.text;  // plain string for text-only messages
    } else if (contentParts.length > 0) {
      content = contentParts;
    }

    const oaiMsg: OAIMessage = { role: msg.role, content };

    if (toolCalls.length > 0) {
      oaiMsg.tool_calls = toolCalls.map(c => {
        if (c.type !== 'tool-call') return null!;
        return {
          id:       c.id,
          type:     'function' as const,
          function: { name: c.name, arguments: JSON.stringify(c.input) },
        };
      }).filter(Boolean);
    }

    // Provider-specific reasoning/thinking blocks are intentionally stripped
    // above. If that leaves a message with no OpenAI-compatible payload, drop
    // it rather than sending an assistant/user message with null content and no
    // tool calls, which some OpenAI-compatible providers reject.
    if (content === null && (oaiMsg.tool_calls?.length ?? 0) === 0) continue;

    result.push(oaiMsg);
  }

  // Prompt caching (opt-in): OpenAI/DeepSeek cache prefixes automatically, but OpenRouter requires
  // explicit `cache_control` breakpoints to cache Anthropic/Gemini models — without them every call
  // re-bills the full prompt. Only emit when the provider config opts in (`parameters.promptCaching`),
  // since strict OpenAI-compatible endpoints can reject the unknown field. Mark the system prompt and
  // the second-to-last user turn (mirrors the Anthropic adapter): the breakpoint caches the prefix
  // before it, so the growing transcript reads from cache instead of re-billing each turn.
  if (opts?.promptCaching) {
    for (const m of result) {
      if (m.role === 'system') { markCacheBreakpoint(m); break; }
    }
    const userIdx = result.reduce<number[]>((acc, m, i) => {
      if (m.role === 'user') acc.push(i);
      return acc;
    }, []);
    if (userIdx.length >= 2) markCacheBreakpoint(result[userIdx[userIdx.length - 2]!]!);
  }

  return result;
}

// Attach an ephemeral cache breakpoint to a message's last text part, converting plain-string content
// to the array form `cache_control` requires. No-op for messages with no textual content (e.g. an
// assistant turn that is only tool_calls) — there is nothing to anchor the breakpoint to.
function markCacheBreakpoint(msg: OAIMessage): void {
  if (typeof msg.content === 'string') {
    if (msg.content.length === 0) return;
    msg.content = [{ type: 'text', text: msg.content, cache_control: { type: 'ephemeral' } }];
    return;
  }
  if (!Array.isArray(msg.content)) return;
  for (let i = msg.content.length - 1; i >= 0; i--) {
    const part = msg.content[i]!;
    if (part.type === 'text') { part.cache_control = { type: 'ephemeral' }; return; }
  }
}

export function toOAITools(tools: readonly Tool[]): OAIToolDef[] {
  return tools.map(t => ({
    type:     'function' as const,
    function: { name: t.name, description: t.description, parameters: t.inputSchema },
  }));
}
