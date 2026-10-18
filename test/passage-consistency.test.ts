/**
 * One primary passage per ranked result: the snippet, its heading path, the primary
 * preview anchor and the reranker input must all describe the same chunk, and a short
 * relevant snippet must never be swapped for the start of the note.
 *
 * Embeddings are synthetic (fixed vectors), so these tests pin passage SELECTION and
 * snippet/anchor consistency — not the real embedding model's ranking quality.
 */
import assert from 'node:assert/strict';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { afterAll, beforeAll, describe, it, vi } from 'vitest';

const vaultDir = mkdtempSync(path.join(tmpdir(), 'ohs-passage-'));
process.env.OBSIDIAN_VAULT_PATH = vaultDir;

vi.resetModules();

const NEAR = new Float32Array([1, 0, 0, 0]);
const MID = new Float32Array([0.6, 0.8, 0, 0]);
const FAR = new Float32Array([0, 0, 0, 1]);

const embedder = await import('../src/embedder.js');
vi.spyOn(embedder, 'embed').mockImplementation((texts: string[]) =>
  Promise.resolve(texts.map(() => NEAR)),
);

const { closeDb, openDb, initVecTable, upsertNote } = await import('../src/db.js');
const { search, bumpIndexVersion, locateMarkedMatch, focusSnippetOnHit } =
  await import('../src/searcher.js');
const { reranker } = await import('../src/reranker.js');

const intro = `${'Redis streams store an append-only log of events for consumers. '.repeat(12)}`;
const redisBody = [
  '# Redis Stream',
  '',
  intro.trim(),
  '',
  '## Configuration',
  '',
  'DefaultStreamMaxLen caps how many entries a stream keeps.',
  '',
  '### Analytics export',
  '',
  'Events are mirrored into ClickHouse tables using the MergeTree engine.',
].join('\n');

interface ChunkSpec {
  text: string;
  headingPath: string | null;
  embedding: Float32Array;
}

function addNote(notePath: string, body: string, chunks: ChunkSpec[], title?: string): void {
  upsertNote({
    path: notePath,
    title: title ?? path.basename(notePath, '.md'),
    tags: [],
    content: body,
    frontmatter: {},
    properties: {},
    mtime: 1,
    hash: notePath,
    chunks: chunks.map((chunk) => {
      const charStart = body.indexOf(chunk.text);
      assert.ok(charStart >= 0, `chunk text must be part of ${notePath}`);
      return {
        text: chunk.text,
        headingPath: chunk.headingPath,
        embedding: chunk.embedding,
        charStart,
        charEnd: charStart + chunk.text.length,
      };
    }),
  });
}

const tracingBody = [
  '# Трассировка',
  '',
  `${'Общий контекст про сервисы и логи. '.repeat(12)}`,
  '',
  '## Заголовки',
  '',
  'Идентификатор trace_id передаётся в заголовке X-Trace между сервисами.',
].join('\n');
const feedBody = [
  '# Feed filters',
  '',
  `${'General background about the feed. '.repeat(12)}`,
  '',
  '## Constants',
  '',
  '```go',
  'const FEED_DOMAIN_ADULT_EXCLUDED = 3',
  '```',
].join('\n');

const configChunk = '## Configuration\n\nDefaultStreamMaxLen caps how many entries a stream keeps.';
const analyticsChunk =
  '### Analytics export\n\nEvents are mirrored into ClickHouse tables using the MergeTree engine.';

beforeAll(() => {
  openDb();
  initVecTable(4);
  bumpIndexVersion();
  addNote('redis-stream.md', redisBody, [
    { text: `# Redis Stream\n\n${intro.trim()}`, headingPath: '# Redis Stream', embedding: FAR },
    { text: configChunk, headingPath: '# Redis Stream > ## Configuration', embedding: NEAR },
    {
      text: analyticsChunk,
      headingPath: '# Redis Stream > ## Configuration > ### Analytics export',
      embedding: MID,
    },
  ]);
  const clickhouseBody = '# ClickHouse\n\nColumnar database for analytics workloads.';
  addNote('clickhouse.md', clickhouseBody, [
    { text: clickhouseBody, headingPath: '# ClickHouse', embedding: MID },
  ]);
  const repeatBody = [
    '## Notes',
    '',
    'Retry the job later.',
    'Some filler words in between.',
    'Retry the job later with ReplacingMergeTree dedup.',
  ].join('\n');
  addNote('repeat.md', repeatBody, [{ text: repeatBody, headingPath: '## Notes', embedding: FAR }]);
  // One-chunk notes: the semantic passage is the whole note, the BM25 hit lies inside it.
  addNote('tracing.md', tracingBody, [{ text: tracingBody, headingPath: null, embedding: MID }]);
  addNote('feed.md', feedBody, [{ text: feedBody, headingPath: null, embedding: MID }]);
});

afterAll(() => {
  vi.restoreAllMocks();
  closeDb();
  rmSync(vaultDir, { recursive: true, force: true });
});

function find<T extends { path: string }>(results: T[], notePath: string): T {
  const result = results.find((r) => r.path === notePath);
  assert.ok(result, `${notePath} should be in the results`);
  return result;
}

describe('BM25 passage', () => {
  it('uses the chunk that holds the exact hit, deep in the note', async () => {
    const results = await search('MergeTree', {
      mode: 'fulltext',
      anchors: true,
      snippetLength: 400,
    });
    const r = find(results, 'redis-stream.md');
    assert.ok(r.snippet.includes('MergeTree'), r.snippet);
    assert.ok(r.snippet.startsWith('# Redis Stream > ## Configuration > ### Analytics export\n'));
    assert.equal(r.snippetSource, 'bm25');
    const anchor = r.previewAnchors![r.primaryAnchorIndex!]!;
    assert.equal(anchor.kind, 'bm25');
    assert.equal(anchor.headingPath, '# Redis Stream > ## Configuration > ### Analytics export');
    assert.equal(redisBody.slice(anchor.matchStart ?? -1, anchor.matchEnd ?? -1), 'MergeTree');
    assert.equal(anchor.charStart, redisBody.indexOf(analyticsChunk));
  });

  it('locates the marked occurrence, not the first repeat of the same line', async () => {
    const results = await search('ReplacingMergeTree', { mode: 'fulltext', anchors: true });
    const r = find(results, 'repeat.md');
    const anchor = r.previewAnchors![0]!;
    const body = [
      '## Notes',
      '',
      'Retry the job later.',
      'Some filler words in between.',
      'Retry the job later with ReplacingMergeTree dedup.',
    ].join('\n');
    assert.equal(anchor.matchStart, body.indexOf('ReplacingMergeTree'));
  });
});

describe('semantic passage', () => {
  it('keeps a short relevant snippet instead of the note opening', async () => {
    const results = await search('how long can a stream grow', {
      mode: 'semantic',
      anchors: true,
      snippetLength: 400,
    });
    const r = find(results, 'redis-stream.md');
    assert.ok(r.snippet.includes('DefaultStreamMaxLen'), r.snippet);
    assert.ok(!r.snippet.includes('append-only log'), 'intro text must not replace the passage');
    assert.equal(r.snippetSource, 'semantic');
    assert.equal(r.previewAnchors![0]!.kind, 'semantic');
  });
});

describe('hybrid passage selection', () => {
  it('shows and anchors the exact identifier hit when the semantic passage lacks it', async () => {
    const results = await search('MergeTree', {
      mode: 'hybrid',
      anchors: true,
      snippetLength: 400,
    });
    const r = find(results, 'redis-stream.md');
    assert.equal(r.snippetSource, 'bm25');
    assert.ok(r.snippet.includes('MergeTree'), r.snippet);
    const primary = r.previewAnchors![r.primaryAnchorIndex!]!;
    assert.equal(primary.kind, 'bm25');
    // Displayed heading = primary anchor heading.
    assert.ok(r.snippet.startsWith(`${primary.headingPath}\n`));
    // The semantic anchor is kept as a secondary one.
    assert.ok(r.previewAnchors!.some((a) => a.kind === 'semantic'));
  });

  it('keeps the semantic passage for natural-language questions', async () => {
    const results = await search('why does the stream keep entries', {
      mode: 'hybrid',
      anchors: true,
      snippetLength: 400,
    });
    const r = find(results, 'redis-stream.md');
    assert.equal(r.snippetSource, 'semantic');
    assert.ok(r.snippet.includes('DefaultStreamMaxLen'), r.snippet);
    assert.equal(r.previewAnchors![r.primaryAnchorIndex!]!.kind, 'semantic');
  });

  it('never reorders documents: results stay sorted by the hybrid score', async () => {
    const results = await search('MergeTree', { mode: 'hybrid', anchors: true });
    const scores = results.map((r) => r.scores.hybrid!);
    assert.deepEqual(
      scores,
      [...scores].sort((a, b) => b - a),
    );
  });

  it('gives the reranker the displayed passage', async () => {
    const spy = vi
      .spyOn(reranker, 'scoreAll')
      .mockImplementation((_q, candidates) => Promise.resolve(candidates.map(() => 0)));
    await search('MergeTree', { mode: 'hybrid', rerank: true, limit: 5 });
    const candidates = spy.mock.calls[0]![1];
    const redis = candidates.find((c) => c.title === 'redis-stream');
    assert.equal(redis?.chunkText, analyticsChunk);
    spy.mockRestore();
  });
});

describe('technical identifiers in one-chunk notes', () => {
  for (const [query, notePath, heading] of [
    ['trace_id', 'tracing.md', '# Трассировка > ## Заголовки'],
    ['FEED_DOMAIN_ADULT_EXCLUDED', 'feed.md', '# Feed filters > ## Constants'],
  ] as const) {
    it(`${query}: hybrid shows and anchors the exact hit, not the note opening`, async () => {
      const results = await search(query, { mode: 'hybrid', anchors: true, snippetLength: 400 });
      const r = find(results, notePath);
      assert.equal(r.snippetSource, 'bm25');
      assert.ok(r.snippet.includes(query), r.snippet);
      const anchor = r.previewAnchors![r.primaryAnchorIndex!]!;
      assert.equal(anchor.kind, 'bm25');
      // The heading chain at the hit, even though the chunk itself has none.
      assert.equal(anchor.headingPath, heading);
      assert.ok(anchor.matchStart! > 0);
    });
  }
});

describe('results without a passage', () => {
  it('title-only matches use the note opening, marked as intro, with no anchor', async () => {
    const results = await search('clickhouse', { mode: 'title', anchors: true });
    const r = find(results, 'clickhouse.md');
    assert.equal(r.snippetSource, 'intro');
    assert.equal(r.previewAnchors, undefined);
  });
});

describe('locateMarkedMatch', () => {
  const S = '\u0001';
  const E = '\u0002';

  it('maps the marked term to body offsets', () => {
    const body = 'alpha beta gamma delta';
    assert.deepEqual(locateMarkedMatch(body, `...beta ${S}gamma${E} delta`, S, E), {
      start: 11,
      end: 16,
    });
  });

  it('refuses a window that occurs twice', () => {
    const body = 'one two one two';
    assert.equal(locateMarkedMatch(body, `${S}one${E} two`, S, E), null);
  });

  it('returns null without a marked hit', () => {
    assert.equal(locateMarkedMatch('text', 'text', S, E), null);
  });
});

describe('focusSnippetOnHit', () => {
  const S = '\u0001';
  const E = '\u0002';

  it('keeps a window whose hit already fits', () => {
    assert.equal(focusSnippetOnHit(`short ${S}hit${E} text`, 100, S, E), 'short hit text');
  });

  it('starts the window at the hit line when the hit would be cut off', () => {
    const marked = `${'intro words '.repeat(40)}\nThe ${S}MergeTree${E} engine.`;
    const out = focusSnippetOnHit(marked, 120, S, E)!;
    assert.ok(out.startsWith('...The MergeTree engine.'), out);
    assert.ok(out.indexOf('MergeTree') < 120);
  });

  it('returns null without a marked hit', () => {
    assert.equal(focusSnippetOnHit('plain', 50, S, E), null);
  });
});
