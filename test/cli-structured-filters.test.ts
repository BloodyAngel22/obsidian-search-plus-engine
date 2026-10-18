/**
 * CLI exposure of structured filters (--filters-json, --offset, --with-meta) and the
 * `capabilities` command. Uses the compiled dist/src/cli.js like the other CLI suites;
 * run `npm run build` first if dist/ is stale.
 */
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { existsSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { afterAll, beforeAll, describe, it, vi } from 'vitest';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.join(__dirname, '..');
const CLI = path.join(ROOT, 'dist/src/cli.js');

const vaultDir = mkdtempSync(path.join(tmpdir(), 'ohs-cli-structured-'));
process.env.OBSIDIAN_VAULT_PATH = vaultDir;
vi.resetModules();

const { closeDb, openDb, initVecTable, upsertNote } = await import('../src/db.js');

function runCli(args: string[]): { stdout: string; stderr: string; status: number } {
  const result = spawnSync(process.execPath, [CLI, ...args], {
    env: {
      ...process.env,
      OBSIDIAN_VAULT_PATH: vaultDir,
      // Keep the CLI offline: metadata-only search must not need embeddings.
      OPENAI_BASE_URL: 'http://127.0.0.1:9/v1',
      OPENAI_API_KEY: 'test',
    },
    encoding: 'utf-8',
    cwd: ROOT,
  });
  return { stdout: result.stdout ?? '', stderr: result.stderr ?? '', status: result.status ?? 1 };
}

beforeAll(() => {
  openDb();
  initVecTable(4);
  for (const [name, properties] of [
    ['a.md', { type: 'learn', projects: ['alpha'] }],
    ['b.md', { type: 'learn' }],
    ['c.md', { type: 'project' }],
  ] as const) {
    writeFileSync(path.join(vaultDir, name), `---\ntype: ${properties.type}\n---\nbody`);
    upsertNote({
      path: name,
      title: name.replace('.md', ''),
      tags: [],
      content: 'body',
      frontmatter: { ...properties },
      properties: { ...properties },
      mtime: 1,
      hash: name,
      chunks: [{ text: 'body', embedding: new Float32Array([0.1, 0.2, 0.3, 0.4]) }],
    });
  }
  closeDb();
});

afterAll(() => {
  rmSync(vaultDir, { recursive: true, force: true });
});

describe('CLI structured filters', { skip: !existsSync(CLI) }, () => {
  it('lists the new options in --help', () => {
    const { stdout } = runCli(['search', '--help']);
    for (const flag of ['--filters-json', '--offset', '--with-meta']) {
      assert.ok(stdout.includes(flag), `${flag} missing from help`);
    }
  });

  it('prints capabilities as JSON', () => {
    const { stdout, status } = runCli(['capabilities']);
    assert.equal(status, 0);
    const caps = JSON.parse(stdout) as { apiVersion: number; capabilities: string[] };
    assert.equal(caps.apiVersion, 2);
    assert.ok(caps.capabilities.includes('structuredFiltersV1'));
  });

  it('runs a metadata-only search with exact totals and paging', () => {
    const filters = JSON.stringify({
      conditions: [{ field: 'type', operator: 'EQ', value: 'learn' }],
    });
    const { stdout, status, stderr } = runCli([
      '--filters-json',
      filters,
      '--limit',
      '1',
      '--offset',
      '1',
      '--json',
      '--with-meta',
    ]);
    assert.equal(status, 0, stderr);
    const body = JSON.parse(stdout) as {
      results: Array<{ path: string }>;
      meta: { total: number; offset: number };
    };
    assert.equal(body.meta.total, 2);
    assert.equal(body.meta.offset, 1);
    assert.deepEqual(
      body.results.map((r) => r.path),
      ['b.md'],
    );
  });

  it('fails with a readable message for invalid filters', () => {
    const { stderr, status } = runCli(['--filters-json', '{"conditions":[{"field":"x"}]}']);
    assert.notEqual(status, 0);
    assert.match(stderr, /Invalid filters/);
  });
});
