import { parse as yamlParse } from 'yaml';

/**
 * Typed view of YAML frontmatter, used by the structured metadata filters.
 *
 * The legacy `note_frontmatter_fields` table flattens every value to lowercased text,
 * so `1` and `"1"` collide, and `null`, `""` and `[]` disappear. This module keeps
 * enough type information for exact matching:
 *
 *   - one KEY row per top-level property (records presence and its shape), and
 *   - one VALUE row per primitive value (a scalar, or each primitive array element).
 *
 * Only top-level keys are typed. Nested objects and arrays of objects are recorded as
 * present (shape `object` / `array`) but their content is not filterable in this phase.
 */

export type PropertyValueType = 'string' | 'number' | 'boolean' | 'null';
export type PropertyShape = PropertyValueType | 'array' | 'object';
export type PropertyScalar = string | number | boolean | null;

export interface PropertyKeyRow {
  field: string;
  shape: PropertyShape;
}

export interface PropertyValueRow {
  field: string;
  type: PropertyValueType;
  text: string;
}

export interface ExtractedProperties {
  keys: PropertyKeyRow[];
  values: PropertyValueRow[];
}

/** Property names and string values are compared in NFC so a decomposed spelling
 *  (common on macOS) matches the precomposed one Obsidian shows. */
export function normalizePropertyText(text: string): string {
  return text.normalize('NFC');
}

/**
 * Canonical (type, text) pair for one scalar. The SAME function serves the write side
 * and the query side, so matching never depends on how a value was spelled in YAML:
 * `1.0` and `1` are both the number 1, while `"1"` stays the string "1".
 */
export function canonicalScalar(value: PropertyScalar): { type: PropertyValueType; text: string } {
  if (value === null) return { type: 'null', text: '' };
  if (typeof value === 'boolean') return { type: 'boolean', text: value ? 'true' : 'false' };
  if (typeof value === 'number')
    return { type: 'number', text: Object.is(value, -0) ? '0' : String(value) };
  return { type: 'string', text: normalizePropertyText(value) };
}

/**
 * gray-matter (js-yaml) turns unquoted dates into Date objects, and the stored
 * `notes.frontmatter` column re-serialises them as ISO timestamps. Obsidian shows such
 * a property as the original text, so a date-only literal is mapped back to
 * `YYYY-MM-DD`. Other timestamps keep their ISO form (documented limitation for
 * notes indexed before the typed index existed).
 */
function dateToText(date: Date): string | null {
  if (Number.isNaN(date.getTime())) return null;
  const iso = date.toISOString();
  return iso.endsWith('T00:00:00.000Z') ? iso.slice(0, 10) : iso;
}

const LEGACY_MIDNIGHT_ISO = /^(\d{4}-\d{2}-\d{2})T00:00:00\.000Z$/;

// eslint-disable-next-line sonarjs/function-return-type -- a YAML scalar is inherently a union type
function toScalar(value: unknown, legacyDates: boolean): PropertyScalar | undefined {
  if (value === null || value === undefined) return null;
  if (typeof value === 'string') {
    if (legacyDates) {
      const m = LEGACY_MIDNIGHT_ISO.exec(value);
      if (m) return m[1]!;
    }
    return value;
  }
  if (typeof value === 'boolean') return value;
  if (typeof value === 'number') return Number.isFinite(value) ? value : String(value);
  if (typeof value === 'bigint') return value.toString();
  if (value instanceof Date) return dateToText(value);
  return undefined;
}

/**
 * Extract typed key and value rows from a parsed frontmatter object.
 *
 * @param legacyDates true when `data` came from the stored, re-serialised frontmatter
 *   column (see dateToText) rather than from the raw YAML text.
 */
export function extractProperties(
  data: Record<string, unknown> | null | undefined,
  legacyDates = false,
): ExtractedProperties {
  const keys: PropertyKeyRow[] = [];
  const values: PropertyValueRow[] = [];
  if (!data || typeof data !== 'object' || Array.isArray(data)) return { keys, values };

  const seenKeys = new Set<string>();
  for (const [rawKey, raw] of Object.entries(data)) {
    if (rawKey.trim() === '') continue;
    const field = normalizePropertyText(rawKey);
    // Two raw keys that normalise to one NFC spelling: first wins, never both.
    if (seenKeys.has(field)) continue;
    seenKeys.add(field);

    if (Array.isArray(raw)) {
      keys.push({ field, shape: 'array' });
      const seenValues = new Set<string>();
      for (const item of raw) {
        const scalar = toScalar(item, legacyDates);
        if (scalar === undefined) continue; // nested object inside an array
        const { type, text } = canonicalScalar(scalar);
        const dedupe = `${type}\u0000${text}`;
        if (seenValues.has(dedupe)) continue;
        seenValues.add(dedupe);
        values.push({ field, type, text });
      }
      continue;
    }

    const scalar = toScalar(raw, legacyDates);
    if (scalar === undefined) {
      keys.push({ field, shape: 'object' });
      continue;
    }
    const { type, text } = canonicalScalar(scalar);
    keys.push({ field, shape: type });
    values.push({ field, type, text });
  }
  return { keys, values };
}

/**
 * Parse raw frontmatter YAML (the text between the `---` fences) with the YAML 1.2
 * core schema. Unlike gray-matter's js-yaml default, it keeps dates as the strings
 * Obsidian displays. Returns null when the text is not a YAML mapping.
 */
export function parseFrontmatterYaml(raw: string): Record<string, unknown> | null {
  if (!raw.trim()) return {};
  const parsed: unknown = yamlParse(raw, { schema: 'core', uniqueKeys: false });
  if (parsed === null || parsed === undefined) return {};
  if (typeof parsed !== 'object' || Array.isArray(parsed)) return null;
  return parsed as Record<string, unknown>;
}
