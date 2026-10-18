import assert from 'node:assert/strict';
import { describe, it } from 'vitest';
import {
  canonicalScalar,
  extractProperties,
  parseFrontmatterYaml,
} from '../src/property-values.js';

describe('canonicalScalar', () => {
  it('keeps number, string, boolean and null distinct', () => {
    assert.deepEqual(canonicalScalar(1), { type: 'number', text: '1' });
    assert.deepEqual(canonicalScalar('1'), { type: 'string', text: '1' });
    assert.deepEqual(canonicalScalar(true), { type: 'boolean', text: 'true' });
    assert.deepEqual(canonicalScalar('true'), { type: 'string', text: 'true' });
    assert.deepEqual(canonicalScalar(null), { type: 'null', text: '' });
    assert.deepEqual(canonicalScalar(''), { type: 'string', text: '' });
  });

  it('spells equal numbers the same way', () => {
    assert.deepEqual(canonicalScalar(1.0), canonicalScalar(1));
    assert.deepEqual(canonicalScalar(-0), canonicalScalar(0));
  });

  it('normalizes strings to NFC', () => {
    assert.equal(canonicalScalar('Café').text, 'Café');
  });
});

describe('parseFrontmatterYaml + extractProperties', () => {
  const raw = [
    'number_value: 1',
    'string_value: "1"',
    'enabled: true',
    'empty_text: ""',
    'empty_list: []',
    'null_value:',
    'date: 2024-01-01',
    'projects:',
    '  - mobivion',
    '  - klikmedia',
    '  - mobivion',
    'nested:',
    '  key: value',
    'objects:',
    '  - name: a',
    '  - plain',
    '"key: with colon": "quoted \\"value\\""',
    'Ключ: значение',
  ].join('\n');

  const { keys, values } = extractProperties(parseFrontmatterYaml(raw));
  const shape = (field: string) => keys.find((k) => k.field === field)?.shape;
  const valuesOf = (field: string) =>
    values.filter((v) => v.field === field).map((v) => `${v.type}:${v.text}`);

  it('records presence and shape for every top-level key', () => {
    assert.equal(shape('number_value'), 'number');
    assert.equal(shape('string_value'), 'string');
    assert.equal(shape('enabled'), 'boolean');
    assert.equal(shape('empty_text'), 'string');
    assert.equal(shape('empty_list'), 'array');
    assert.equal(shape('null_value'), 'null');
    assert.equal(shape('nested'), 'object');
    assert.equal(shape('objects'), 'array');
    assert.equal(shape('missing'), undefined);
  });

  it('keeps typed values', () => {
    assert.deepEqual(valuesOf('number_value'), ['number:1']);
    assert.deepEqual(valuesOf('string_value'), ['string:1']);
    assert.deepEqual(valuesOf('enabled'), ['boolean:true']);
    assert.deepEqual(valuesOf('empty_text'), ['string:']);
    assert.deepEqual(valuesOf('empty_list'), []);
    assert.deepEqual(valuesOf('null_value'), ['null:']);
  });

  it('keeps unquoted dates as the text Obsidian shows', () => {
    assert.deepEqual(valuesOf('date'), ['string:2024-01-01']);
  });

  it('stores each primitive array element once', () => {
    assert.deepEqual(valuesOf('projects'), ['string:mobivion', 'string:klikmedia']);
  });

  it('skips nested object content but keeps primitive siblings', () => {
    assert.deepEqual(valuesOf('nested'), []);
    assert.deepEqual(valuesOf('objects'), ['string:plain']);
  });

  it('preserves unusual and Unicode key spelling', () => {
    assert.deepEqual(valuesOf('key: with colon'), ['string:quoted "value"']);
    assert.deepEqual(valuesOf('Ключ'), ['string:значение']);
  });

  it('treats an empty document as no properties and a non-mapping as null', () => {
    assert.deepEqual(parseFrontmatterYaml(''), {});
    assert.equal(parseFrontmatterYaml('- a\n- b'), null);
    assert.deepEqual(extractProperties(null), { keys: [], values: [] });
  });

  it('accepts duplicate keys instead of throwing (last one wins)', () => {
    const data = parseFrontmatterYaml('a: 1\na: 2');
    assert.deepEqual(data, { a: 2 });
  });
});

describe('extractProperties — legacy stored frontmatter', () => {
  it('maps gray-matter midnight dates back to YYYY-MM-DD', () => {
    const { values } = extractProperties({ created: '2024-01-01T00:00:00.000Z' }, true);
    assert.deepEqual(values, [{ field: 'created', type: 'string', text: '2024-01-01' }]);
  });

  it('formats Date objects from gray-matter', () => {
    const { values } = extractProperties({ created: new Date('2024-03-05T00:00:00Z') });
    assert.deepEqual(values, [{ field: 'created', type: 'string', text: '2024-03-05' }]);
  });

  it('does not rewrite ISO strings when the data is not legacy', () => {
    const { values } = extractProperties({ created: '2024-01-01T00:00:00.000Z' }, false);
    assert.equal(values[0]!.text, '2024-01-01T00:00:00.000Z');
  });
});
