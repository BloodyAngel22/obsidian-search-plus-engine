/**
 * Structured metadata filters end to end through search(): typed matching, operator
 * semantics, every retrieval arm, filter-before-limit, reranking, metadata-only
 * pagination and index updates.
 */
import assert from 'node:assert/strict';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { afterAll, beforeAll, beforeEach, describe, it, vi } from 'vitest';

const vaultDir = mkdtempSync(path.join(tmpdir(), 'ohs-structured-filters-'));
process.env.OBSIDIAN_VAULT_PATH = vaultDir;

/** Code-unit order: paths are compared the way SQLite's BINARY collation does. */
function byCodeUnit(a: string, b: string): number {
  if (a < b) return -1;
  return a > b ? 1 : 0;
}

vi.resetModules();

const NEAR = new Float32Array([1, 0, 0, 0]);
const FAR = new Float32Array([0, 0, 0, 1]);

const embedder = await import('../src/embedder.js');
const embedSpy = vi
  .spyOn(embedder, 'embed')
  .mockImplementation((texts: string[]) => Promise.resolve(texts.map(() => NEAR)));

const { closeDb, openDb, initVecTable, upsertNote, deleteNote, upsertLinks, getDb } =
  await import('../src/db.js');
const { search, searchWithMeta, bumpIndexVersion } = await import('../src/searcher.js');
const { parseStructuredFilters, StructuredFiltersError } =
  await import('../src/structured-filters.js');
const { reranker } = await import('../src/reranker.js');

type Props = Record<string, unknown>;

function addNote(
  notePath: string,
  properties: Props,
  options: { content?: string; title?: string; embedding?: Float32Array; tags?: string[] } = {},
): void {
  const content = options.content ?? `Body of ${notePath}`;
  upsertNote({
    path: notePath,
    title: options.title ?? path.basename(notePath, '.md'),
    tags: options.tags ?? [],
    content,
    frontmatter: properties,
    properties,
    mtime: Date.now(),
    hash: `h-${notePath}-${JSON.stringify(properties)}`,
    chunks: [{ text: content, embedding: options.embedding ?? FAR }],
  });
}

function f(conditions: unknown[]) {
  return parseStructuredFilters({ conditions })!;
}

async function metaPaths(conditions: unknown[], extra: Record<string, unknown> = {}) {
  const results = await search('', { filters: f(conditions), limit: 0, ...extra });
  return results.map((r) => r.path).sort(byCodeUnit);
}

beforeAll(() => {
  openDb();
  initVecTable(4);
  bumpIndexVersion();
  addNote('typed/number.md', { value: 1, projects: ['mobivion', 'klikmedia', 'mobivion'] });
  addNote('typed/string.md', { value: '1', projects: 'mobivion' });
  addNote('typed/bool.md', { value: true, enabled: true });
  addNote('typed/null.md', { value: null });
  addNote('typed/empty-text.md', { value: '' });
  addNote('typed/empty-list.md', { value: [] });
  addNote('typed/missing.md', { other: 'x' });
  addNote('typed/unicode.md', { Тип: 'Заметка', 'key: with "quotes"': "it's: ok" });
  addNote('typed/case.md', { Status: 'Done' });
  addNote('typed/lower.md', { status: 'done' });
});

afterAll(() => {
  embedSpy.mockRestore();
  closeDb();
  rmSync(vaultDir, { recursive: true, force: true });
});

describe('typed operator semantics (metadata-only)', () => {
  it('EQ distinguishes number, string, boolean, null and empty string', async () => {
    assert.deepEqual(await metaPaths([{ field: 'value', operator: 'EQ', value: 1 }]), [
      'typed/number.md',
    ]);
    assert.deepEqual(await metaPaths([{ field: 'value', operator: 'EQ', value: '1' }]), [
      'typed/string.md',
    ]);
    assert.deepEqual(await metaPaths([{ field: 'value', operator: 'EQ', value: true }]), [
      'typed/bool.md',
    ]);
    assert.deepEqual(await metaPaths([{ field: 'value', operator: 'EQ', value: null }]), [
      'typed/null.md',
    ]);
    assert.deepEqual(await metaPaths([{ field: 'value', operator: 'EQ', value: '' }]), [
      'typed/empty-text.md',
    ]);
  });

  it('IN matches array elements exactly and scalars equal to a value (OR inside)', async () => {
    assert.deepEqual(
      await metaPaths([{ field: 'projects', operator: 'IN', values: ['klikmedia'] }]),
      ['typed/number.md'],
    );
    assert.deepEqual(
      await metaPaths([{ field: 'projects', operator: 'IN', values: ['mobivion', 'nope'] }]),
      ['typed/number.md', 'typed/string.md'],
    );
    // Substrings never match: this is element equality, not text search.
    assert.deepEqual(
      await metaPaths([{ field: 'projects', operator: 'IN', values: ['klik'] }]),
      [],
    );
  });

  it('EXISTS counts null, empty string and empty array as present; MISSING only absent', async () => {
    assert.deepEqual(await metaPaths([{ field: 'value', operator: 'EXISTS' }]), [
      'typed/bool.md',
      'typed/empty-list.md',
      'typed/empty-text.md',
      'typed/null.md',
      'typed/number.md',
      'typed/string.md',
    ]);
    const missing = await metaPaths([{ field: 'value', operator: 'MISSING' }]);
    assert.ok(missing.includes('typed/missing.md'));
    assert.ok(!missing.includes('typed/null.md'));
    assert.ok(!missing.includes('typed/empty-list.md'));
  });

  it('NOT_IN excludes notes containing any value and keeps notes without the key', async () => {
    const paths = await metaPaths([
      { field: 'projects', operator: 'NOT_IN', values: ['mobivion'] },
    ]);
    assert.ok(!paths.includes('typed/number.md'));
    assert.ok(!paths.includes('typed/string.md'));
    assert.ok(paths.includes('typed/missing.md'));
    const present = await metaPaths([
      { field: 'value', operator: 'EXISTS' },
      { field: 'value', operator: 'NOT_IN', values: [1, '1'] },
    ]);
    assert.deepEqual(present, [
      'typed/bool.md',
      'typed/empty-list.md',
      'typed/empty-text.md',
      'typed/null.md',
    ]);
  });

  it('ANDs conditions across fields', async () => {
    assert.deepEqual(
      await metaPaths([
        { field: 'value', operator: 'EQ', value: true },
        { field: 'enabled', operator: 'EQ', value: true },
      ]),
      ['typed/bool.md'],
    );
    assert.deepEqual(
      await metaPaths([
        { field: 'value', operator: 'EQ', value: 1 },
        { field: 'enabled', operator: 'EXISTS' },
      ]),
      [],
    );
  });

  it('matches Unicode, quotes, spaces and colons in keys and values', async () => {
    assert.deepEqual(await metaPaths([{ field: 'Тип', operator: 'EQ', value: 'Заметка' }]), [
      'typed/unicode.md',
    ]);
    assert.deepEqual(
      await metaPaths([{ field: 'key: with "quotes"', operator: 'EQ', value: "it's: ok" }]),
      ['typed/unicode.md'],
    );
  });

  it('matches field names and values case-sensitively, without collisions', async () => {
    assert.deepEqual(await metaPaths([{ field: 'Status', operator: 'EXISTS' }]), ['typed/case.md']);
    assert.deepEqual(await metaPaths([{ field: 'status', operator: 'EQ', value: 'done' }]), [
      'typed/lower.md',
    ]);
    assert.deepEqual(await metaPaths([{ field: 'status', operator: 'EQ', value: 'Done' }]), []);
  });

  it('treats SQL-looking input as plain data', async () => {
    const hostile = "x' OR 1=1 --";
    assert.deepEqual(await metaPaths([{ field: hostile, operator: 'EXISTS' }]), []);
    assert.deepEqual(await metaPaths([{ field: 'value', operator: 'IN', values: [hostile] }]), []);
    assert.ok((getDb().prepare('SELECT COUNT(*) AS c FROM notes').get() as { c: number }).c >= 10);
  });
});

describe('legacy filters combined with structured filters', () => {
  it('ANDs the legacy frontmatter text filter with structured conditions', async () => {
    assert.deepEqual(
      await metaPaths([{ field: 'status', operator: 'EXISTS' }], { frontmatter: 'status:done' }),
      ['typed/lower.md'],
    );
  });

  it('returns nothing for conflicting conditions instead of dropping one', async () => {
    assert.deepEqual(
      await metaPaths([{ field: 'status', operator: 'EQ', value: 'open' }], {
        frontmatter: 'status:done',
      }),
      [],
    );
  });

  it('ANDs scope with structured conditions', async () => {
    addNote('other/number.md', { value: 1 });
    assert.deepEqual(
      await metaPaths([{ field: 'value', operator: 'EQ', value: 1 }], { scope: 'typed' }),
      ['typed/number.md'],
    );
    deleteNote('other/number.md');
  });
});

describe('metadata-only pagination', () => {
  beforeAll(() => {
    for (let i = 0; i < 25; i++) {
      addNote(
        `page/note-${String(i).padStart(2, '0')}.md`,
        { kind: 'paged' },
        {
          title: i % 2 === 0 ? 'Same title' : `Title ${String(i).padStart(2, '0')}`,
        },
      );
    }
  });

  it('returns stable, non-overlapping pages with an exact document count', async () => {
    const filters = f([{ field: 'kind', operator: 'EQ', value: 'paged' }]);
    const seen: string[] = [];
    let offset = 0;
    for (;;) {
      const { results, meta } = await searchWithMeta('', { filters, limit: 10, offset });
      assert.equal(meta.kind, 'metadata');
      assert.equal(meta.total, 25);
      assert.equal(meta.totalExact, true);
      assert.equal(meta.offset, offset);
      seen.push(...results.map((r) => r.path));
      if (!meta.hasMore) {
        assert.equal(meta.nextOffset, null);
        break;
      }
      offset = meta.nextOffset!;
    }
    assert.equal(seen.length, 25);
    assert.equal(new Set(seen).size, 25);
    const again = await searchWithMeta('', { filters, limit: 25 });
    assert.deepEqual(
      again.results.map((r) => r.path),
      seen,
    );
  });

  it('reports ranks continuing across pages and an empty page past the end', async () => {
    const filters = f([{ field: 'kind', operator: 'EQ', value: 'paged' }]);
    const page = await searchWithMeta('', { filters, limit: 10, offset: 10 });
    assert.equal(page.results[0]!.rank, 11);
    const end = await searchWithMeta('', { filters, limit: 10, offset: 40 });
    assert.equal(end.results.length, 0);
    assert.equal(end.meta.hasMore, false);
  });

  it('does not embed, rank or rerank', async () => {
    const before = embedSpy.mock.calls.length;
    const rerankSpy = vi.spyOn(reranker, 'scoreAll');
    await searchWithMeta('', {
      filters: f([{ field: 'kind', operator: 'EQ', value: 'paged' }]),
      limit: 5,
      rerank: true,
    });
    assert.equal(embedSpy.mock.calls.length, before);
    assert.equal(rerankSpy.mock.calls.length, 0);
    rerankSpy.mockRestore();
  });

  it('rejects offset for ranked searches', async () => {
    await assert.rejects(
      search('anything', { offset: 5 }),
      (err: unknown) => err instanceof StructuredFiltersError && /metadata-only/.test(err.message),
    );
  });

  it('reports total: null for ranked searches', async () => {
    const { meta } = await searchWithMeta('Body', {
      mode: 'fulltext',
      filters: f([{ field: 'kind', operator: 'EQ', value: 'paged' }]),
    });
    assert.equal(meta.kind, 'ranked');
    assert.equal(meta.total, null);
    assert.equal(meta.totalExact, false);
  });
});

describe('ranked retrieval applies structured filters before the limit', () => {
  // 30 strong notes rank first in every arm; the single needle only matches the filter.
  beforeAll(() => {
    for (let i = 0; i < 30; i++) {
      addNote(
        `rank/strong-${i}.md`,
        { group: 'strong' },
        {
          title: `Zebra quantum ${i}`,
          content: 'zebra quantum zebra quantum zebra quantum zebra quantum',
          embedding: NEAR,
        },
      );
    }
    addNote(
      'rank/needle.md',
      { group: 'needle', tags: ['x'] },
      { title: 'Needle zebra note', content: 'a single zebra mention', embedding: FAR },
    );
  });

  const needleOnly = () => f([{ field: 'group', operator: 'EQ', value: 'needle' }]);

  for (const mode of ['fulltext', 'semantic', 'hybrid', 'title'] as const) {
    it(`${mode}: limit 1 still finds the only matching note`, async () => {
      bumpIndexVersion();
      const results = await search('zebra', { mode, limit: 1, filters: needleOnly() });
      assert.deepEqual(
        results.map((r) => r.path),
        ['rank/needle.md'],
      );
    });
  }

  it('without the filter the needle is not in the top results (fixture sanity)', async () => {
    const results = await search('zebra', { mode: 'semantic', limit: 5 });
    assert.ok(!results.some((r) => r.path === 'rank/needle.md'));
  });

  it('reranking cannot reintroduce excluded notes', async () => {
    const rerankSpy = vi
      .spyOn(reranker, 'scoreAll')
      .mockImplementation((_q, candidates) => Promise.resolve(candidates.map(() => 10)));
    const results = await search('zebra', {
      mode: 'hybrid',
      limit: 10,
      rerank: true,
      filters: f([{ field: 'group', operator: 'NOT_IN', values: ['strong'] }]),
    });
    assert.ok(results.length > 0);
    assert.ok(results.every((r) => !r.path.startsWith('rank/strong-')));
    for (const call of rerankSpy.mock.calls) {
      assert.ok(call[1].every((c) => !String(c.title).startsWith('Zebra quantum')));
    }
    rerankSpy.mockRestore();
  });

  it('applies filters in multi-query fan-out', async () => {
    const results = await search('zebra', {
      queries: ['zebra', 'quantum'],
      mode: 'fulltext',
      limit: 3,
      filters: needleOnly(),
    });
    assert.deepEqual(
      results.map((r) => r.path),
      ['rank/needle.md'],
    );
  });

  it('applies filters to related (graph) results', async () => {
    upsertLinks('rank/strong-0.md', ['rank/needle.md', 'rank/strong-1.md']);
    bumpIndexVersion();
    const results = await search('rank/strong-0.md', {
      related: true,
      filters: needleOnly(),
    });
    assert.deepEqual(
      results.map((r) => r.path),
      ['rank/needle.md'],
    );
  });

  it('applies filters to similar-note lookups', async () => {
    const results = await search('', {
      notePath: 'rank/strong-5.md',
      limit: 1,
      filters: needleOnly(),
    });
    assert.deepEqual(
      results.map((r) => r.path),
      ['rank/needle.md'],
    );
  });
});

describe('index updates', () => {
  beforeEach(() => bumpIndexVersion());

  it('replaces a note contribution when its properties change', async () => {
    addNote('upd/a.md', { stage: 'draft' });
    assert.deepEqual(await metaPaths([{ field: 'stage', operator: 'EQ', value: 'draft' }]), [
      'upd/a.md',
    ]);
    addNote('upd/a.md', { stage: 'final' });
    assert.deepEqual(await metaPaths([{ field: 'stage', operator: 'EQ', value: 'draft' }]), []);
    assert.deepEqual(await metaPaths([{ field: 'stage', operator: 'EQ', value: 'final' }]), [
      'upd/a.md',
    ]);
  });

  it('removes deleted notes and follows renames', async () => {
    addNote('upd/old-name.md', { stage: 'renamed' });
    deleteNote('upd/old-name.md');
    addNote('upd/new-name.md', { stage: 'renamed' });
    assert.deepEqual(await metaPaths([{ field: 'stage', operator: 'EQ', value: 'renamed' }]), [
      'upd/new-name.md',
    ]);
    const orphanRows = getDb()
      .prepare(
        'SELECT COUNT(*) AS c FROM note_property_values WHERE note_id NOT IN (SELECT id FROM notes)',
      )
      .get() as { c: number };
    assert.equal(orphanRows.c, 0);
  });

  it('removes a property that was deleted from the note', async () => {
    addNote('upd/drop.md', { temp: 'yes' });
    addNote('upd/drop.md', {});
    assert.deepEqual(await metaPaths([{ field: 'temp', operator: 'EXISTS' }]), []);
  });
});
