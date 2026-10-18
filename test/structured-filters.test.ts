import assert from 'node:assert/strict';
import { describe, it } from 'vitest';
import {
  MAX_FILTER_CONDITIONS,
  MAX_FILTER_VALUES,
  parseStructuredFilters,
  StructuredFiltersError,
  structuredFiltersKey,
} from '../src/structured-filters.js';

const valid = {
  version: 1,
  operator: 'AND',
  conditions: [
    { field: 'projects', operator: 'IN', values: ['mobivion', 'klikmedia'] },
    { field: 'type', operator: 'EQ', value: 'learn' },
    { field: 'priority', operator: 'NOT_IN', values: [1, 2] },
    { field: 'enabled', operator: 'EXISTS' },
    { field: 'archived', operator: 'MISSING' },
    { field: 'flag', operator: 'EQ', value: null },
  ],
};

function rejects(value: unknown, pattern: RegExp): void {
  assert.throws(
    () => parseStructuredFilters(value),
    (err: unknown) => err instanceof StructuredFiltersError && pattern.test(err.message),
  );
}

describe('parseStructuredFilters', () => {
  it('accepts the documented contract', () => {
    assert.deepEqual(parseStructuredFilters(valid), valid);
  });

  it('accepts JSON text', () => {
    assert.deepEqual(parseStructuredFilters(JSON.stringify(valid)), valid);
  });

  it('treats null and undefined as no filters', () => {
    assert.equal(parseStructuredFilters(undefined), undefined);
    assert.equal(parseStructuredFilters(null), undefined);
  });

  it('accepts an empty condition list', () => {
    assert.deepEqual(parseStructuredFilters({ conditions: [] }), { conditions: [] });
  });

  it('rejects unknown operators and top-level OR', () => {
    rejects({ conditions: [{ field: 'a', operator: 'LIKE', values: ['x'] }] }, /operator/);
    rejects({ operator: 'OR', conditions: [] }, /operator/);
  });

  it('rejects missing or misplaced values', () => {
    rejects({ conditions: [{ field: 'a', operator: 'EQ' }] }, /value/);
    rejects({ conditions: [{ field: 'a', operator: 'IN', values: [] }] }, /at least one/);
    rejects({ conditions: [{ field: 'a', operator: 'EQ', values: ['x'] }] }, /Invalid filters/);
    rejects({ conditions: [{ field: 'a', operator: 'EXISTS', value: 'x' }] }, /Invalid filters/);
  });

  it('rejects unsupported value types', () => {
    rejects({ conditions: [{ field: 'a', operator: 'EQ', value: { nested: 1 } }] }, /value/);
    rejects({ conditions: [{ field: 'a', operator: 'IN', values: [['x']] }] }, /values/);
  });

  it('rejects bad field names', () => {
    rejects({ conditions: [{ field: '', operator: 'EXISTS' }] }, /field/);
    rejects({ conditions: [{ field: '   ', operator: 'EXISTS' }] }, /blank/);
    rejects({ conditions: [{ field: 'a\u0000b', operator: 'EXISTS' }] }, /control/);
    rejects({ conditions: [{ field: 'x'.repeat(257), operator: 'EXISTS' }] }, /field/);
  });

  it('enforces condition, value and size limits', () => {
    const tooManyConditions = Array.from({ length: MAX_FILTER_CONDITIONS + 1 }, (_, i) => ({
      field: `f${i}`,
      operator: 'EXISTS',
    }));
    rejects({ conditions: tooManyConditions }, /at most/);
    const tooManyValues = Array.from({ length: MAX_FILTER_VALUES + 1 }, (_, i) => `v${i}`);
    rejects({ conditions: [{ field: 'a', operator: 'IN', values: tooManyValues }] }, /at most/);
    rejects('x'.repeat(70 * 1024), /exceeds/);
    rejects(
      { conditions: [{ field: 'a', operator: 'EQ', value: 'x'.repeat(70 * 1024) }] },
      /exceeds/,
    );
  });

  it('rejects unknown keys and malformed JSON', () => {
    rejects({ conditions: [], extra: true }, /Invalid filters/);
    rejects('{not json', /JSON object/);
    rejects([], /Invalid filters/);
  });
});

describe('structuredFiltersKey', () => {
  it('is empty without conditions and stable otherwise', () => {
    assert.equal(structuredFiltersKey(undefined), '');
    assert.equal(structuredFiltersKey({ conditions: [] }), '');
    const filters = parseStructuredFilters(valid)!;
    assert.equal(
      structuredFiltersKey(filters),
      structuredFiltersKey(parseStructuredFilters(valid)),
    );
  });
});
