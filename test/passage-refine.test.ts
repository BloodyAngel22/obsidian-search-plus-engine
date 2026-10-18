/**
 * Stage two of passage selection: inside the primary chunk, the block that answers the
 * query becomes the snippet, heading and anchor. Embeddings are synthetic: the mock
 * makes the "merge" paragraph closest to the ClickHouse question, which is what the
 * real local model measures (see passage-blocks.test.ts), so these tests pin the
 * wiring and the snippet/anchor invariant, not the model's quality.
 */
import assert from 'node:assert/strict';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { afterAll, beforeAll, describe, it, vi } from 'vitest';

const vaultDir = mkdtempSync(path.join(tmpdir(), 'ohs-refine-'));
process.env.OBSIDIAN_VAULT_PATH = vaultDir;

vi.resetModules();

const QUERY = new Float32Array([1, 0, 0, 0]);
const ANSWER = new Float32Array([0.95, 0.31, 0, 0]);
const RELATED = new Float32Array([0.6, 0.8, 0, 0]);
const FAR = new Float32Array([0, 0, 0, 1]);

const embedder = await import('../src/embedder.js');
const embedSpy = vi.spyOn(embedder, 'embed').mockImplementation((texts: string[], type) =>
  Promise.resolve(
    texts.map((text) => {
      if (type === 'query') return QUERY;
      return /объединяются|trace_id —/.test(text) ? ANSWER : RELATED;
    }),
  ),
);

const { closeDb, openDb, initVecTable, upsertNote } = await import('../src/db.js');
const { search, bumpIndexVersion, hasTitleEvidence } = await import('../src/searcher.js');
type SearchResult = Awaited<ReturnType<typeof search>>[number];
type MatchAnchor = NonNullable<SearchResult['previewAnchors']>[number];

const clickhouseBody = [
  '# RTB Adult Domain Filter',
  '',
  'Общее описание фильтра доменов. '.repeat(10).trim(),
  '',
  '### Что я узнал нового для себя',
  '',
  'Clickhouse хранит данные не в одной непрерывной таблице, а в отдельных частях.',
  '',
  'Когда в clickhouse поступают новые записи, clickhouse создает части. Позже части объединяются (merge).',
  '',
  '`ReplacingMergeTree` - схлопывает дубликаты при слиянии частей. Дубликаты определяются по ORDER BY.',
  '',
  'Курсор запоминает до какого момента были обработаны данные, чтобы не проходится по всем данным.',
].join('\n');
const learnedChunk = clickhouseBody.slice(clickhouseBody.indexOf('### Что я узнал'));

const observationBody = [
  '# RTB Video Observation',
  '',
  '### Важные моменты',
  '',
  '**Дополнительные особенности:**',
  '',
  '- `sid` — определяет сервер `rtb-exchange`, на котором нужно искать ADM.',
  '',
  '- `trace_id` — идентификатор для поиска сохранённого ADM.',
  '',
  '- `inner_imp_id` — позволяет связать наблюдение с конкретным impression.',
].join('\n');
const observationChunk = observationBody.slice(observationBody.indexOf('### Важные'));

interface ChunkSpec {
  text: string;
  headingPath: string | null;
  embedding: Float32Array;
  charEnd?: number;
}

function addNote(notePath: string, body: string, chunks: ChunkSpec[], title?: string): void {
  upsertNote({
    path: notePath,
    title: title ?? path.basename(notePath, '.md'),
    tags: [],
    content: body,
    frontmatter: { status: 'active' },
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
        charEnd: chunk.charEnd ?? charStart + chunk.text.length,
      };
    }),
  });
}

beforeAll(() => {
  openDb();
  initVecTable(4);
  bumpIndexVersion();
  addNote('rtb-adult.md', clickhouseBody, [
    {
      text: clickhouseBody.slice(0, clickhouseBody.indexOf('\n\n###')),
      headingPath: '# RTB Adult Domain Filter',
      embedding: FAR,
    },
    {
      text: learnedChunk,
      headingPath: '# RTB Adult Domain Filter > ### Что я узнал нового для себя',
      embedding: QUERY,
    },
  ]);
  addNote('observation.md', observationBody, [
    {
      text: observationChunk,
      headingPath: '# RTB Video Observation > ### Важные моменты',
      embedding: RELATED,
    },
  ]);
  const staleBody = 'Short note about clickhouse parts.';
  // The stored chunk claims to reach far past the note: offsets are stale.
  addNote('stale.md', staleBody, [
    { text: staleBody, headingPath: null, embedding: RELATED, charEnd: staleBody.length + 500 },
  ]);
});

afterAll(() => {
  vi.restoreAllMocks();
  closeDb();
  rmSync(vaultDir, { recursive: true, force: true });
});

function find(results: SearchResult[], notePath: string): SearchResult {
  const result = results.find((r) => r.path === notePath);
  assert.ok(result, `${notePath} should be in the results`);
  return result;
}

/** `text.slice(start, end)` for offsets that must be present. */
function cut(text: string, start: number | undefined, end: number | undefined): string {
  assert.ok(start !== undefined && end !== undefined, 'offsets must be set');
  return text.slice(start, end);
}

function primary(r: SearchResult): MatchAnchor {
  const anchor = r.previewAnchors?.[r.primaryAnchorIndex ?? 0];
  assert.ok(anchor, 'result must have a primary anchor');
  return anchor;
}

describe('semantic paragraph selection (ClickHouse regression)', () => {
  it('shows and anchors the paragraph that explains the merge, not the first one', async () => {
    const results = await search('Как ClickHouse объединяет части данных?', {
      mode: 'semantic',
      anchors: true,
      snippetLength: 400,
    });
    const r = find(results, 'rtb-adult.md');
    const anchor = primary(r);
    const answerStart = clickhouseBody.indexOf('Когда в clickhouse');
    assert.equal(anchor.passageStart, answerStart);
    assert.ok(anchor.matchText.startsWith('Когда в clickhouse поступают'));
    assert.equal(anchor.headingPath, '# RTB Adult Domain Filter > ### Что я узнал нового для себя');
    // Snippet = heading line + the selected paragraph; same range as the anchor.
    assert.equal(
      r.snippet,
      `${anchor.headingPath}\n${cut(clickhouseBody, anchor.passageStart, anchor.passageEnd)}`,
    );
    assert.equal(r.snippetSource, 'semantic');
    // Semantic mode never claims a literal match.
    assert.equal(anchor.matchStart, undefined);
  });

  it('only embeds blocks of the final results, and reuses them on the next search', async () => {
    embedSpy.mockClear();
    bumpIndexVersion();
    await search('Как ClickHouse объединяет части данных?', { mode: 'semantic', anchors: true });
    const documentCalls = embedSpy.mock.calls.filter(([, type]) => type === 'document');
    assert.equal(documentCalls.length, 0, 'block embeddings come from the cache');
  });
});

describe('exact identifiers inside a semantic passage', () => {
  it('hybrid: anchors trace_id exactly, in its own list item', async () => {
    const results = await search('trace_id', { mode: 'hybrid', anchors: true, snippetLength: 400 });
    const r = find(results, 'observation.md');
    const anchor = primary(r);
    assert.equal(cut(observationBody, anchor.matchStart, anchor.matchEnd), 'trace_id');
    assert.ok(anchor.passageStart! <= anchor.matchStart! && anchor.matchEnd! <= anchor.passageEnd!);
    assert.ok(r.snippet.split('\n')[1]!.startsWith('- `trace_id`'), r.snippet);
  });

  it('hybrid: MergeTree is not reported as an exact hit inside ReplacingMergeTree', async () => {
    const results = await search('MergeTree', { mode: 'hybrid', anchors: true });
    const r = find(results, 'rtb-adult.md');
    assert.equal(primary(r).matchStart, undefined);
  });
});

describe('full-text passage', () => {
  it('a multi-word query picks the block that covers the most terms', async () => {
    const results = await search('clickhouse части объединяются', {
      mode: 'fulltext',
      anchors: true,
      snippetLength: 400,
    });
    const r = find(results, 'rtb-adult.md');
    const anchor = primary(r);
    assert.equal(anchor.passageStart, clickhouseBody.indexOf('Когда в clickhouse'));
    const passage = cut(clickhouseBody, anchor.passageStart, anchor.passageEnd);
    assert.ok(passage.includes(cut(clickhouseBody, anchor.matchStart, anchor.matchEnd)));
    // matchText must come from the passage so preview and navigation agree.
    assert.ok(anchor.matchText.startsWith('Когда в clickhouse'));
  });
});

describe('stale offsets', () => {
  it('keeps the chunk passage when stored offsets reach past the note', async () => {
    const results = await search('clickhouse parts', { mode: 'semantic', anchors: true });
    const r = find(results, 'stale.md');
    assert.equal(primary(r).passageStart, undefined);
  });
});

describe('hybrid title evidence', () => {
  it('ignores a title that shares only a stray trigram with the query', () => {
    assert.equal(hasTitleEvidence('MergeTree', { title: 'Redis Stream', aliases: null }), false);
    assert.equal(
      hasTitleEvidence('Как ClickHouse объединяет части данных?', {
        title: 'RTB Video как живет и работает',
        aliases: null,
      }),
      false,
    );
  });

  it('keeps typos, titles named inside the query, and aliases', () => {
    assert.equal(hasTitleEvidence('zettlekasten', { title: 'Zettelkasten', aliases: null }), true);
    assert.equal(
      hasTitleEvidence('how does spaced repetition work', {
        title: 'Spaced repetition',
        aliases: null,
      }),
      true,
    );
    assert.equal(
      hasTitleEvidence('redis', { title: 'Something else', aliases: '["Redis notes"]' }),
      true,
    );
  });
});
