/**
 * Migration of an existing (pre-structured-filter) index to the typed property index:
 * rebuilt from stored frontmatter, transactional, retried after interruption, never
 * touching notes, chunks or embeddings, and self-healing after an older engine writes.
 */
import assert from 'node:assert/strict';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { afterAll, beforeEach, describe, it, vi } from 'vitest';
import { stringify as yamlStringify } from 'yaml';

const vaultDir = mkdtempSync(path.join(tmpdir(), 'ohs-property-migration-'));
process.env.OBSIDIAN_VAULT_PATH = vaultDir;

/** Code-unit order: paths are compared the way SQLite's BINARY collation does. */
function byCodeUnit(a: string, b: string): number {
  if (a < b) return -1;
  return a > b ? 1 : 0;
}

vi.resetModules();

const {
  closeDb,
  openDb,
  initVecTable,
  upsertNote,
  getDb,
  getPropertyIndexError,
  wipeDatabaseFiles,
} = await import('../src/db.js');
const { search, bumpIndexVersion } = await import('../src/searcher.js');
const { parseStructuredFilters, StructuredFiltersError } =
  await import('../src/structured-filters.js');
const { getEngineCapabilities } = await import('../src/capabilities.js');

const embedding = new Float32Array([0.1, 0.2, 0.3, 0.4]);

/** An index as the previous engine version left it: no typed tables, no marker. */
function seedLegacyIndex(): void {
  wipeDatabaseFiles();
  openDb();
  initVecTable(4);
  upsertNote({
    path: 'legacy/a.md',
    title: 'Legacy A',
    tags: [],
    content: 'Legacy note alpha about migration.',
    // gray-matter would hand the engine a Date for an unquoted date literal.
    frontmatter: { type: 'learn', priority: 3, done: false, created: new Date('2024-01-01') },
    mtime: 1,
    hash: 'ha',
    chunks: [{ text: 'Legacy note alpha', embedding }],
  });
  upsertNote({
    path: 'legacy/b.md',
    title: 'Legacy B',
    tags: [],
    content: 'Legacy note beta.',
    frontmatter: { type: 'project', priority: '3', list: [], empty: null },
    mtime: 1,
    hash: 'hb',
    chunks: [{ text: 'Legacy note beta', embedding }],
  });
  const db = getDb();
  db.exec(`
    DROP TABLE note_property_keys;
    DROP TABLE note_property_values;
    DROP TABLE note_property_sync;
    DELETE FROM settings WHERE key = 'property_lookup_version';
  `);
  closeDb();
}

function snapshot(): { notes: unknown[]; chunks: unknown[]; vectors: number } {
  const db = getDb();
  return {
    notes: db.prepare('SELECT id, path, hash, frontmatter, content FROM notes ORDER BY id').all(),
    chunks: db.prepare('SELECT id, note_id, text, embedding_status FROM chunks ORDER BY id').all(),
    vectors: (db.prepare('SELECT COUNT(*) AS c FROM vec_chunks').get() as { c: number }).c,
  };
}

async function paths(conditions: unknown[]): Promise<string[]> {
  bumpIndexVersion();
  const results = await search('', {
    filters: parseStructuredFilters({ conditions })!,
    limit: 0,
  });
  return results.map((r) => r.path).sort(byCodeUnit);
}

afterAll(() => {
  closeDb();
  rmSync(vaultDir, { recursive: true, force: true });
});

describe('typed property index migration', () => {
  beforeEach(() => {
    seedLegacyIndex();
  });

  it('builds typed rows from stored frontmatter without touching notes, chunks or vectors', async () => {
    openDb();
    const before = snapshot();
    closeDb();
    openDb();
    assert.deepEqual(snapshot(), before);
    assert.equal(getPropertyIndexError(), null);
    const version = getDb()
      .prepare("SELECT value FROM settings WHERE key = 'property_lookup_version'")
      .get() as { value: string };
    assert.equal(version.value, '1');

    assert.deepEqual(await paths([{ field: 'priority', operator: 'EQ', value: 3 }]), [
      'legacy/a.md',
    ]);
    assert.deepEqual(await paths([{ field: 'priority', operator: 'EQ', value: '3' }]), [
      'legacy/b.md',
    ]);
    assert.deepEqual(await paths([{ field: 'done', operator: 'EQ', value: false }]), [
      'legacy/a.md',
    ]);
    assert.deepEqual(await paths([{ field: 'created', operator: 'EQ', value: '2024-01-01' }]), [
      'legacy/a.md',
    ]);
    assert.deepEqual(await paths([{ field: 'list', operator: 'EXISTS' }]), ['legacy/b.md']);
    assert.deepEqual(await paths([{ field: 'empty', operator: 'EQ', value: null }]), [
      'legacy/b.md',
    ]);
  });

  it('rolls back an interrupted migration and redoes it on the next start', async () => {
    // A crash mid-migration leaves an uncommitted transaction, which SQLite discards.
    // Model the worst committed state: tables exist with partial rows but no marker.
    openDb();
    const db = getDb();
    db.exec(`
      DROP TABLE note_property_keys; DROP TABLE note_property_values; DROP TABLE note_property_sync;
      DELETE FROM settings WHERE key = 'property_lookup_version';
      CREATE TABLE note_property_keys (note_id INTEGER, field TEXT, shape TEXT, PRIMARY KEY (note_id, field));
      CREATE TABLE note_property_values (note_id INTEGER, field TEXT, value_type TEXT, value_text TEXT,
        PRIMARY KEY (note_id, field, value_type, value_text));
      CREATE TABLE note_property_sync (note_id INTEGER PRIMARY KEY, hash TEXT);
      INSERT INTO note_property_values VALUES (1, 'type', 'string', 'stale-partial-row');
    `);
    closeDb();

    openDb();
    assert.equal(getPropertyIndexError(), null);
    assert.deepEqual(
      await paths([{ field: 'type', operator: 'EQ', value: 'stale-partial-row' }]),
      [],
    );
    assert.deepEqual(await paths([{ field: 'type', operator: 'EQ', value: 'learn' }]), [
      'legacy/a.md',
    ]);
  });

  it('keeps the database usable and refuses structured filters when migration fails', async () => {
    openDb();
    const db = getDb();
    db.exec(`
      DROP TABLE note_property_keys; DROP TABLE note_property_values; DROP TABLE note_property_sync;
      DELETE FROM settings WHERE key = 'property_lookup_version';
      CREATE VIEW note_property_values AS SELECT 1 AS note_id, 'x' AS field, 'string' AS value_type, 'x' AS value_text;
    `);
    closeDb();

    openDb();
    const error = getPropertyIndexError();
    assert.ok(error && /migration failed/.test(error), String(error));
    assert.ok(!getEngineCapabilities().capabilities.includes('structuredFiltersV1'));
    assert.ok(getEngineCapabilities().propertyIndexError);
    // No marker: the next start retries.
    assert.equal(
      getDb().prepare("SELECT value FROM settings WHERE key = 'property_lookup_version'").get(),
      undefined,
    );
    // Plain search still works.
    bumpIndexVersion();
    const plain = await search('migration', { mode: 'fulltext' });
    assert.ok(plain.some((r) => r.path === 'legacy/a.md'));
    // Structured filters fail loudly instead of returning unfiltered results.
    await assert.rejects(
      search('', {
        filters: parseStructuredFilters({
          conditions: [{ field: 'type', operator: 'EQ', value: 'learn' }],
        })!,
      }),
      (err: unknown) => err instanceof StructuredFiltersError,
    );

    // Once the obstacle is gone, the next start completes the migration.
    getDb().exec('DROP VIEW note_property_values');
    closeDb();
    openDb();
    assert.equal(getPropertyIndexError(), null);
    assert.deepEqual(await paths([{ field: 'type', operator: 'EQ', value: 'learn' }]), [
      'legacy/a.md',
    ]);
  });

  it('rebuilds rows for notes changed by an older engine sharing the database', async () => {
    openDb();
    assert.deepEqual(await paths([{ field: 'type', operator: 'EQ', value: 'learn' }]), [
      'legacy/a.md',
    ]);
    // An older engine updates the note: new frontmatter + hash, no typed rows.
    getDb()
      .prepare('UPDATE notes SET frontmatter = ?, hash = ? WHERE path = ?')
      .run(yamlStringify({ type: 'reference' }), 'ha-2', 'legacy/a.md');
    getDb()
      .prepare(
        "UPDATE settings SET value = CAST(CAST(value AS INTEGER) + 1 AS TEXT) WHERE key = 'db_version'",
      )
      .run();
    assert.deepEqual(await paths([{ field: 'type', operator: 'EQ', value: 'learn' }]), []);
    assert.deepEqual(await paths([{ field: 'type', operator: 'EQ', value: 'reference' }]), [
      'legacy/a.md',
    ]);
  });

  it('is idempotent across restarts', () => {
    openDb();
    closeDb();
    openDb();
    const rows = getDb().prepare('SELECT COUNT(*) AS c FROM note_property_values').get() as {
      c: number;
    };
    assert.ok(rows.c > 0);
    closeDb();
    openDb();
    const again = getDb().prepare('SELECT COUNT(*) AS c FROM note_property_values').get() as {
      c: number;
    };
    assert.equal(again.c, rows.c);
  });
});
