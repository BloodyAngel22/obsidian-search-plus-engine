/**
 * Structured filters and capability negotiation through every transport: stdio
 * JSON lines, the MCP search/status tools, and the shared status payload. All of them
 * use the same parser and search entry point, so the same request must give the same
 * notes everywhere.
 */
import type { Server } from '@modelcontextprotocol/sdk/server/index.js';
import assert from 'node:assert/strict';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { afterAll, beforeAll, describe, it, vi } from 'vitest';

const vaultDir = mkdtempSync(path.join(tmpdir(), 'ohs-structured-transports-'));
process.env.OBSIDIAN_VAULT_PATH = vaultDir;

/** Code-unit order: paths are compared the way SQLite's BINARY collation does. */
function byCodeUnit(a: string, b: string): number {
  if (a < b) return -1;
  return a > b ? 1 : 0;
}

vi.resetModules();

const embedder = await import('../src/embedder.js');
vi.spyOn(embedder, 'embed').mockResolvedValue([new Float32Array([0.1, 0.2, 0.3, 0.4])]);
vi.spyOn(embedder, 'getContextLength').mockResolvedValue(512);

const { closeDb, openDb, initVecTable, upsertNote } = await import('../src/db.js');
const { searchWithMeta, search, bumpIndexVersion } = await import('../src/searcher.js');
const { handleStdioLine } = await import('../src/stdio-server.js');
const { createMcpRuntime, createMcpServer } = await import('../src/mcp-runtime.js');
const { buildStatusPayload } = await import('../src/status-payload.js');

const embedding = new Float32Array([0.1, 0.2, 0.3, 0.4]);
const filters = {
  conditions: [
    { field: 'projects', operator: 'IN', values: ['alpha'] },
    { field: 'type', operator: 'EQ', value: 'learn' },
  ],
};

beforeAll(() => {
  openDb();
  initVecTable(4);
  bumpIndexVersion();
  const notes: Array<[string, Record<string, unknown>]> = [
    ['t/one.md', { projects: ['alpha', 'beta'], type: 'learn' }],
    ['t/two.md', { projects: ['alpha'], type: 'project' }],
    ['t/three.md', { projects: 'alpha', type: 'learn' }],
    ['t/four.md', { type: 'learn' }],
  ];
  for (const [notePath, properties] of notes) {
    upsertNote({
      path: notePath,
      title: path.basename(notePath, '.md'),
      tags: [],
      content: `Redis stream notes ${notePath}`,
      frontmatter: properties,
      properties,
      mtime: 1,
      hash: notePath,
      chunks: [{ text: `Redis stream notes ${notePath}`, embedding }],
    });
  }
});

afterAll(() => {
  vi.restoreAllMocks();
  closeDb();
  rmSync(vaultDir, { recursive: true, force: true });
});

async function stdio(request: unknown): Promise<Record<string, unknown>> {
  let out = '';
  await handleStdioLine(JSON.stringify(request), searchWithMeta, (line) => {
    out = line;
  });
  return JSON.parse(out) as Record<string, unknown>;
}

function getHandler(server: Server, method: string) {
  const handlers = (server as unknown as { _requestHandlers: Map<string, unknown> })
    ._requestHandlers;
  return handlers.get(method) as (req: unknown, extra: unknown) => Promise<unknown>;
}

async function mcpCall(
  name: string,
  args: Record<string, unknown>,
): Promise<{ content: Array<{ text: string }>; isError?: boolean }> {
  const runtime = await createMcpRuntime();
  const server = createMcpServer(runtime);
  return (await getHandler(server, 'tools/call')(
    { method: 'tools/call', params: { name, arguments: args } },
    {},
  )) as { content: Array<{ text: string }>; isError?: boolean };
}

const expected = ['t/one.md', 't/three.md'];

describe('stdio transport', () => {
  it('reports capabilities through a dedicated action', async () => {
    const response = await stdio({ id: 'c1', action: 'capabilities' });
    assert.equal(response.id, 'c1');
    const caps = response.capabilities as { apiVersion: number; capabilities: string[] };
    assert.equal(caps.apiVersion, 2);
    assert.ok(caps.capabilities.includes('structuredFiltersV1'));
    assert.ok(caps.capabilities.includes('metadataOnlyPagination'));
  });

  it('runs a metadata-only search with meta and exact totals', async () => {
    const response = await stdio({ id: 's1', query: '', options: { filters, limit: 1 } });
    const results = response.results as Array<{ path: string }>;
    const meta = response.meta as { total: number; hasMore: boolean; nextOffset: number };
    assert.equal(results.length, 1);
    assert.equal(meta.total, 2);
    assert.equal(meta.hasMore, true);
    const next = await stdio({
      id: 's2',
      query: '',
      options: { filters, limit: 1, offset: meta.nextOffset },
    });
    const all = [...results, ...(next.results as Array<{ path: string }>)].map((r) => r.path);
    assert.deepEqual(all.sort(byCodeUnit), expected);
  });

  it('applies filters to ranked text searches', async () => {
    const response = await stdio({
      id: 's3',
      query: 'redis',
      options: { filters, mode: 'fulltext' },
    });
    const results = (response.results as Array<{ path: string }>)
      .map((r) => r.path)
      .sort(byCodeUnit);
    assert.deepEqual(results, expected);
    assert.equal((response.meta as { total: unknown }).total, null);
  });

  it('accepts filters as JSON text', async () => {
    const response = await stdio({
      id: 's4',
      query: '',
      options: { filters: JSON.stringify(filters), limit: 0 },
    });
    assert.deepEqual(
      (response.results as Array<{ path: string }>).map((r) => r.path).sort(byCodeUnit),
      expected,
    );
  });

  it('returns a readable error for invalid filters and never results', async () => {
    const response = await stdio({
      id: 's5',
      query: 'redis',
      options: { filters: { conditions: [{ field: 'x', operator: 'LIKE', value: 'y' }] } },
    });
    assert.equal(response.id, 's5');
    assert.match(String(response.error), /Invalid filters/);
    assert.equal(response.results, undefined);
  });

  it('rejects offset for ranked searches', async () => {
    const response = await stdio({ id: 's6', query: 'redis', options: { offset: 3 } });
    assert.match(String(response.error), /metadata-only/);
  });

  it('keeps the legacy response shape for plain-array search functions', async () => {
    let out = '';
    await handleStdioLine(JSON.stringify({ id: 'l1', query: 'redis' }), search, (line) => {
      out = line;
    });
    const response = JSON.parse(out) as Record<string, unknown>;
    assert.ok(Array.isArray(response.results));
    assert.equal(response.meta, undefined);
  });
});

describe('MCP transport', () => {
  it('returns the same notes as stdio for the same filters', async () => {
    const result = await mcpCall('search', { query: '', filters, limit: 0 });
    assert.notEqual(result.isError, true);
    const body = JSON.parse(result.content[0]!.text) as {
      results: Array<{ path: string }>;
      meta: { total: number; kind: string };
    };
    assert.deepEqual(body.results.map((r) => r.path).sort(byCodeUnit), expected);
    assert.equal(body.meta.total, 2);
    assert.equal(body.meta.kind, 'metadata');
  });

  it('reports invalid filters as a tool error', async () => {
    const result = await mcpCall('search', {
      query: 'redis',
      filters: { conditions: [{ field: '', operator: 'EXISTS' }] },
    });
    assert.equal(result.isError, true);
    assert.match(result.content[0]!.text, /Invalid filters/);
  });

  it('includes capabilities in the status tool', async () => {
    const result = await mcpCall('status', {});
    const text = result.content[0]!.text;
    const body = JSON.parse(text.slice(0, text.lastIndexOf('}') + 1)) as {
      api_version: number;
      capabilities: { capabilities: string[] };
    };
    assert.equal(body.api_version, 2);
    assert.ok(body.capabilities.capabilities.includes('structuredFiltersV1'));
  });

  it('documents filters and offset in the search tool schema', async () => {
    const runtime = await createMcpRuntime();
    const list = (await getHandler(createMcpServer(runtime), 'tools/list')(
      { method: 'tools/list', params: {} },
      {},
    )) as { tools: Array<{ name: string; inputSchema: { properties: Record<string, unknown> } }> };
    const searchTool = list.tools.find((tool) => tool.name.endsWith('search'))!;
    assert.ok(searchTool.inputSchema.properties.filters);
    assert.ok(searchTool.inputSchema.properties.offset);
  });
});

describe('status payload', () => {
  it('carries the API version and capabilities', () => {
    const payload = buildStatusPayload({ contextLength: 512, version: 'test' });
    assert.equal(payload.api_version, 2);
    assert.ok(
      (payload.capabilities as { capabilities: string[] }).capabilities.includes(
        'structuredFiltersV1',
      ),
    );
  });
});
