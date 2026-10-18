/**
 * Paragraph selection with the REAL local embedding model (multilingual-e5-small):
 * the ClickHouse question must open the paragraph that explains merging, not the first
 * paragraph of its section. Runs in the integration suite (model download needed).
 */
import assert from 'node:assert/strict';
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { afterAll, beforeAll, describe, it } from 'vitest';

const vaultDir = mkdtempSync(path.join(tmpdir(), 'ohs-passage-int-'));
process.env.OBSIDIAN_VAULT_PATH = vaultDir;

const note = [
  '---',
  'tags: [clickhouse]',
  '---',
  '# RTB Adult Domain Filter',
  '',
  '### Что я узнал нового для себя',
  '',
  'Clickhouse хранит данные не в одной непрерывной таблице, а в отдельных частях.',
  '',
  'Когда в clickhouse поступают новые записи, clickhouse создает части. Позже части объединяются (merge).',
  '',
  '`ReplacingMergeTree` - схлопывает дубликаты при слиянии частей. Дубликаты определяются по ORDER BY.',
  '',
  'Курсор запоминает до какого момента были обработаны данные, чтобы не проходится по всем данным каждый раз.',
  '',
  '`location.ancestorOrigins` - отдает всю цепочку origin для текущего iframe.',
].join('\n');
writeFileSync(path.join(vaultDir, 'RTB Adult Domain Filter.md'), note);
writeFileSync(
  path.join(vaultDir, 'Redis Stream.md'),
  '# Redis Stream\n\nОчередь событий в Redis. `DefaultStreamMaxLen = 200_000` ограничивает длину.',
);

const { openDb, initVecTable, closeDb } = await import('../src/db.js');
const { getEmbeddingDim, getContextLength } = await import('../src/embedder.js');
const { indexVaultSync } = await import('../src/indexer.js');
const { search } = await import('../src/searcher.js');

beforeAll(async () => {
  openDb();
  const [, dim] = await Promise.all([getContextLength(), getEmbeddingDim()]);
  initVecTable(dim);
  await indexVaultSync();
}, 120_000);

afterAll(() => {
  closeDb();
  rmSync(vaultDir, { recursive: true, force: true });
});

describe('paragraph selection with the local model', () => {
  for (const mode of ['semantic', 'hybrid'] as const) {
    it(`${mode}: the ClickHouse question opens the merge paragraph`, async () => {
      const results = await search('Как ClickHouse объединяет части данных?', {
        mode,
        anchors: true,
        snippetLength: 400,
      });
      const top = results[0];
      assert.ok(top);
      assert.equal(top.path, 'RTB Adult Domain Filter.md');
      const anchor = top.previewAnchors?.[top.primaryAnchorIndex ?? 0];
      assert.ok(anchor);
      assert.ok(anchor.matchText.startsWith('Когда в clickhouse поступают'), anchor.matchText);
      assert.ok(top.snippet.includes('объединяются (merge)'), top.snippet);
    });
  }

  it('a paraphrased question finds its paragraph without word overlap', async () => {
    const results = await search('как убираются дубликаты', { mode: 'semantic', anchors: true });
    const anchor = results[0]?.previewAnchors?.[0];
    assert.ok(anchor);
    assert.ok(anchor.matchText.startsWith('ReplacingMergeTree'), anchor.matchText);
  });
});
