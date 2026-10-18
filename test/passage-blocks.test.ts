import assert from 'node:assert/strict';
import { describe, it } from 'vitest';
import {
  blockEmbeddingText,
  candidateBlocks,
  chooseBlock,
  findExactOccurrence,
  findTermOccurrence,
  lexicalBlockScores,
  passageRange,
  queryTerms,
  splitMarkdownBlocks,
} from '../src/passage-blocks.js';

const CLICKHOUSE = [
  '### Что я узнал нового для себя',
  '',
  'Clickhouse хранит данные не в одной непрерывной таблице, а в отдельных частях.',
  '',
  'Когда в clickhouse поступают новые записи, clickhouse создает части. Позже части объединяются (merge).',
  '',
  '`ReplacingMergeTree` - схлопывает дубликаты при слиянии частей. Дубликаты определяются по ORDER BY.',
  '',
  'Курсор запоминает до какого момента были обработаны данные, чтобы не проходится по всем данным каждый раз.',
].join('\n');

function blocksOf(content: string) {
  return splitMarkdownBlocks(content, 0, content.length);
}

describe('splitMarkdownBlocks', () => {
  it('splits paragraphs on blank lines and keeps exact source offsets', () => {
    const blocks = blocksOf(CLICKHOUSE);
    assert.deepEqual(
      blocks.map((b) => b.kind),
      ['heading', 'paragraph', 'paragraph', 'paragraph', 'paragraph'],
    );
    for (const block of blocks) assert.equal(CLICKHOUSE.slice(block.start, block.end), block.text);
    assert.ok(blocks[2]!.text.startsWith('Когда в clickhouse'));
  });

  it('widens a chunk that starts mid-word to the start of its line', () => {
    const start = CLICKHOUSE.indexOf('ранит данные'); // inside "хранит"
    const blocks = splitMarkdownBlocks(CLICKHOUSE, start, CLICKHOUSE.length);
    assert.ok(blocks[0]!.text.startsWith('Clickhouse хранит'));
  });

  it('never splits a fenced code block, even when the chunk starts inside it', () => {
    const content = [
      'Intro paragraph.',
      '',
      '```lua',
      'local a = 1',
      '',
      'return a',
      '```',
      '',
      'After the code.',
    ].join('\n');
    const inside = content.indexOf('return a');
    const blocks = splitMarkdownBlocks(content, inside, content.length);
    assert.equal(blocks[0]!.kind, 'code');
    assert.ok(blocks[0]!.text.startsWith('```lua'));
    assert.ok(blocks[0]!.text.endsWith('```'));
    assert.equal(blocks[1]!.text, 'After the code.');
  });

  it('treats a heading inside a code fence as code', () => {
    const content = '```bash\n# not a heading\necho hi\n```\n\nText.';
    assert.deepEqual(
      blocksOf(content).map((b) => b.kind),
      ['code', 'paragraph'],
    );
  });

  it('keeps a list item with its nested items and continuation lines together', () => {
    const content = [
      '- first item',
      '  continues here',
      '  - nested one',
      '    - nested deeper',
      '- second item',
      '',
      '1. numbered',
      '',
      '    indented note after a blank line',
    ].join('\n');
    const blocks = blocksOf(content);
    assert.equal(blocks.length, 3);
    assert.ok(blocks[0]!.text.includes('nested deeper'));
    assert.equal(blocks[1]!.text, '- second item');
    assert.ok(blocks[2]!.text.includes('indented note after a blank line'));
  });

  it('keeps a Markdown table and a quote as one block each', () => {
    const content = ['| a | b |', '|---|---|', '| 1 | 2 |', '', '> quoted', '> more'].join('\n');
    const blocks = blocksOf(content);
    assert.deepEqual(
      blocks.map((b) => b.kind),
      ['table', 'quote'],
    );
    assert.ok(blocks[0]!.text.includes('| 1 | 2 |'));
  });

  it('preserves Unicode text and Markdown formatting verbatim', () => {
    const content = 'Привет, **мир** 🌍 — `код`.\n\nВторой абзац.';
    const blocks = blocksOf(content);
    assert.equal(blocks[0]!.text, 'Привет, **мир** 🌍 — `код`.');
  });
});

describe('chooseBlock', () => {
  const blocks = blocksOf(CLICKHOUSE);

  it('picks the semantically best block, not the first one (ClickHouse regression)', () => {
    // Cosine values measured with the default local model (multilingual-e5-small).
    const semanticScores = [null, 0.8694, 0.893, 0.8286, 0.8383];
    const index = chooseBlock({
      blocks,
      query: 'Как ClickHouse объединяет части данных?',
      semanticScores,
    });
    assert.ok(blocks[index]!.text.startsWith('Когда в clickhouse поступают'));
  });

  it('falls back to lexical coverage without embeddings', () => {
    const index = chooseBlock({ blocks, query: 'зачем курсор обработки данных' });
    assert.ok(blocks[index]!.text.startsWith('Курсор запоминает'));
  });

  it('lets a verified exact hit pick its own block', () => {
    const hitOffset = CLICKHOUSE.indexOf('ReplacingMergeTree');
    const index = chooseBlock({ blocks, query: 'anything', hitOffset });
    assert.ok(blocks[index]!.text.includes('ReplacingMergeTree'));
  });

  it('prefers blocks holding a literal identifier whole', () => {
    const index = chooseBlock({
      blocks,
      query: 'ReplacingMergeTree',
      semanticScores: [null, 0.99, 0.98, 0.5, 0.97],
    });
    assert.ok(blocks[index]!.text.includes('ReplacingMergeTree'));
  });

  it('does not treat MergeTree as present inside ReplacingMergeTree', () => {
    assert.equal(candidateBlocks(blocks, 'MergeTree').length, 4); // no narrowing
    const scores = lexicalBlockScores(blocks, 'MergeTree');
    assert.ok(scores.every((s) => s === 0));
  });

  it('breaks ties toward the earlier of repeated paragraphs', () => {
    const repeated = 'Same text here about caches.\n\nSame text here about caches.';
    const index = chooseBlock({ blocks: blocksOf(repeated), query: 'caches' });
    assert.equal(index, 0);
  });

  it('skips headings and bare markup', () => {
    const content = '## Title\n\n---\n\nReal content paragraph.';
    const index = chooseBlock({ blocks: blocksOf(content), query: 'zzz' });
    assert.equal(blocksOf(content)[index]!.text, 'Real content paragraph.');
  });
});

describe('passageRange', () => {
  it('adds the following block after a lead-in ending with a colon', () => {
    const content =
      '**Дополнительные особенности:**\n\n- `trace_id` — идентификатор ADM.\n\nДругое.';
    const blocks = blocksOf(content);
    const range = passageRange(blocks, 0);
    assert.equal(content.slice(range.start, range.end).split('\n\n').length, 2);
  });

  it('starts at the chosen block and keeps a self-contained paragraph alone', () => {
    const blocks = blocksOf(CLICKHOUSE);
    const range = passageRange(blocks, 2);
    assert.equal(range.start, blocks[2]!.start);
    assert.equal(range.end, blocks[2]!.end);
  });
});

describe('exact identifier offsets', () => {
  const text = 'Use ReplacingMergeTree here. Plain MergeTree there. trace id and trace_id.';

  it('matches MergeTree only as a whole token, keeping the original spelling', () => {
    const hit = findExactOccurrence(text, 0, text.length, 'mergetree');
    assert.ok(hit);
    assert.equal(text.slice(hit.start, hit.end), 'MergeTree');
    assert.equal(hit.start, text.indexOf('Plain MergeTree') + 6);
  });

  it('matches trace_id literally, not "trace id"', () => {
    const hit = findExactOccurrence(text, 0, text.length, 'trace_id');
    assert.ok(hit);
    assert.equal(text.slice(hit.start, hit.end), 'trace_id');
  });

  it('returns null when only a longer identifier contains the query', () => {
    const only = 'Only ReplacingMergeTree.';
    assert.equal(findExactOccurrence(only, 0, only.length, 'MergeTree'), null);
  });

  it('marks only a word FTS would match: the query term as a prefix', () => {
    const hit = findTermOccurrence(CLICKHOUSE, 0, CLICKHOUSE.length, 'объединя');
    assert.ok(hit);
    assert.equal(CLICKHOUSE.slice(hit.start, hit.end), 'объединяются');
    assert.equal(findTermOccurrence(CLICKHOUSE, 0, CLICKHOUSE.length, 'объединяет'), null);
  });
});

describe('query terms and embedding text', () => {
  it('drops stop words and keeps identifiers whole', () => {
    assert.deepEqual(queryTerms('Как ClickHouse объединяет части данных?'), [
      'clickhouse',
      'объединяет',
      'части',
      'данных',
    ]);
    assert.deepEqual(queryTerms('trace_id'), ['trace_id']);
  });

  it('strips list markers and inline markup for embedding', () => {
    assert.equal(blockEmbeddingText('- `trace_id` — **id**'), 'trace_id — id');
  });
});

describe('performance', () => {
  it('splits and scores a large note quickly', () => {
    const paragraph = 'clickhouse redis stream merge parts index query table column cursor. ';
    const content = Array.from({ length: 4000 }, (_, i) =>
      i % 50 === 0 ? `## Section ${i}` : paragraph.repeat(4),
    ).join('\n\n');
    const started = performance.now();
    const blocks = splitMarkdownBlocks(content, content.length - 5000, content.length);
    chooseBlock({ blocks, query: 'merge parts cursor' });
    assert.ok(blocks.length > 5);
    assert.ok(performance.now() - started < 200, 'block selection must stay cheap');
  });
});
