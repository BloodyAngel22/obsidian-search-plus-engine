import { z } from 'zod';
import { formatValidationError } from './boundary-validation.js';

/**
 * Structured metadata filters, contract version 1.
 *
 *   { "version": 1, "operator": "AND", "conditions": [
 *       { "field": "projects", "operator": "IN", "values": ["alpha", "beta"] },
 *       { "field": "type", "operator": "EQ", "value": "learn" },
 *       { "field": "archived", "operator": "MISSING" } ] }
 *
 * Semantics (see README "Structured metadata filters"):
 *   - Conditions are ANDed. Values inside one IN / NOT_IN are ORed.
 *   - `field` is the exact top-level YAML key (case-sensitive, NFC-normalized).
 *   - Values are typed: the number 1, the string "1", true and null never match each other.
 *   - EQ / IN match a scalar property equal to a value, or an array property that
 *     contains it as an element.
 *   - NOT_IN matches notes whose property contains none of the values — including notes
 *     without the property. Combine with EXISTS to require presence.
 *   - EXISTS matches any explicitly present key, even when it is null, "" or [].
 *     MISSING matches only notes without the key.
 */

export const STRUCTURED_FILTERS_VERSION = 1;
export const MAX_FILTER_CONDITIONS = 32;
export const MAX_FILTER_VALUES = 256;
export const MAX_FILTER_FIELD_LENGTH = 256;
export const MAX_FILTER_STRING_LENGTH = 2048;
export const MAX_FILTERS_JSON_BYTES = 64 * 1024;

// eslint-disable-next-line no-control-regex -- control characters are exactly what is rejected
const CONTROL_CHARS = /[\u0000-\u001f\u007f]/;

const FieldSchema = z
  .string()
  .min(1, 'must not be empty')
  .max(MAX_FILTER_FIELD_LENGTH, `must be at most ${MAX_FILTER_FIELD_LENGTH} characters`)
  .refine((field) => field.trim() !== '', 'must not be blank')
  .refine((field) => !CONTROL_CHARS.test(field), 'must not contain control characters');

const ScalarSchema = z.union([
  z.string().max(MAX_FILTER_STRING_LENGTH),
  z.number().finite(),
  z.boolean(),
  z.null(),
]);

const ValueListSchema = z
  .array(ScalarSchema)
  .min(1, 'must contain at least one value')
  .max(MAX_FILTER_VALUES, `must contain at most ${MAX_FILTER_VALUES} values`);

const EqConditionSchema = z
  .object({ field: FieldSchema, operator: z.literal('EQ'), value: ScalarSchema })
  .strict();
const InConditionSchema = z
  .object({ field: FieldSchema, operator: z.literal('IN'), values: ValueListSchema })
  .strict();
const NotInConditionSchema = z
  .object({ field: FieldSchema, operator: z.literal('NOT_IN'), values: ValueListSchema })
  .strict();
const ExistsConditionSchema = z
  .object({ field: FieldSchema, operator: z.literal('EXISTS') })
  .strict();
const MissingConditionSchema = z
  .object({ field: FieldSchema, operator: z.literal('MISSING') })
  .strict();

const ConditionSchema = z.discriminatedUnion('operator', [
  EqConditionSchema,
  InConditionSchema,
  NotInConditionSchema,
  ExistsConditionSchema,
  MissingConditionSchema,
]);

export const StructuredFiltersSchema = z
  .object({
    version: z.literal(STRUCTURED_FILTERS_VERSION).optional(),
    operator: z.literal('AND').optional(),
    conditions: z
      .array(ConditionSchema)
      .max(MAX_FILTER_CONDITIONS, `must contain at most ${MAX_FILTER_CONDITIONS} conditions`),
  })
  .strict();

export type FilterScalar = z.infer<typeof ScalarSchema>;
export type FilterCondition = z.infer<typeof ConditionSchema>;
export type StructuredFilters = z.infer<typeof StructuredFiltersSchema>;

export class StructuredFiltersError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'StructuredFiltersError';
  }
}

/**
 * Validate a transport value into StructuredFilters. Accepts the object itself or its
 * JSON text (CLI flags and some MCP clients send strings). `undefined`/`null` mean
 * "no structured filters". Throws StructuredFiltersError with a readable message.
 */
export function parseStructuredFilters(value: unknown): StructuredFilters | undefined {
  if (value === undefined || value === null) return undefined;
  let raw: unknown = value;
  if (typeof value === 'string') {
    if (Buffer.byteLength(value, 'utf8') > MAX_FILTERS_JSON_BYTES) {
      throw new StructuredFiltersError(
        `Invalid filters: request exceeds ${MAX_FILTERS_JSON_BYTES} bytes`,
      );
    }
    try {
      raw = JSON.parse(value) as unknown;
    } catch {
      throw new StructuredFiltersError('Invalid filters: expected a JSON object');
    }
  } else {
    let size: number;
    try {
      size = Buffer.byteLength(JSON.stringify(value) ?? '', 'utf8');
    } catch {
      throw new StructuredFiltersError('Invalid filters: value is not serialisable');
    }
    if (size > MAX_FILTERS_JSON_BYTES) {
      throw new StructuredFiltersError(
        `Invalid filters: request exceeds ${MAX_FILTERS_JSON_BYTES} bytes`,
      );
    }
  }
  const parsed = StructuredFiltersSchema.safeParse(raw);
  if (!parsed.success) {
    throw new StructuredFiltersError(formatValidationError('filters', parsed.error));
  }
  return parsed.data;
}

export function hasStructuredFilters(filters: StructuredFilters | undefined): boolean {
  return filters !== undefined && filters.conditions.length > 0;
}

/** Stable text for cache keys: same conditions in the same order give the same key. */
export function structuredFiltersKey(filters: StructuredFilters | undefined): string {
  return hasStructuredFilters(filters) ? JSON.stringify(filters!.conditions) : '';
}
