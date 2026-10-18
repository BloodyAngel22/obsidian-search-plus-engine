import { getPropertyIndexError } from './db.js';
import {
  MAX_FILTER_CONDITIONS,
  MAX_FILTER_VALUES,
  STRUCTURED_FILTERS_VERSION,
} from './structured-filters.js';

/**
 * Engine API version and feature flags, reported through the stdio `capabilities`
 * action, the `status` payload (stdio, CLI and MCP) and `ohs capabilities`.
 *
 * Clients MUST check for a capability before sending the matching request fields:
 * engines older than API version 2 silently drop unknown search options, so an
 * unchecked `filters` request would come back unfiltered.
 */
export const ENGINE_API_VERSION = 2;

export const CAPABILITY_STRUCTURED_FILTERS_V1 = 'structuredFiltersV1';
export const CAPABILITY_METADATA_ONLY_PAGINATION = 'metadataOnlyPagination';
export const CAPABILITY_SEARCH_META = 'searchMeta';

export interface EngineCapabilities {
  apiVersion: number;
  capabilities: string[];
  structuredFilters: {
    version: number;
    operators: string[];
    maxConditions: number;
    maxValuesPerCondition: number;
  };
  /** Present when the typed metadata index could not be built; structured filters are
   *  then unavailable instead of silently wrong. */
  propertyIndexError?: string;
}

export function getEngineCapabilities(): EngineCapabilities {
  const propertyIndexError = getPropertyIndexError();
  const capabilities = [CAPABILITY_SEARCH_META];
  if (propertyIndexError === null) {
    capabilities.push(CAPABILITY_STRUCTURED_FILTERS_V1, CAPABILITY_METADATA_ONLY_PAGINATION);
  }
  return {
    apiVersion: ENGINE_API_VERSION,
    capabilities,
    structuredFilters: {
      version: STRUCTURED_FILTERS_VERSION,
      operators: ['EQ', 'IN', 'NOT_IN', 'EXISTS', 'MISSING'],
      maxConditions: MAX_FILTER_CONDITIONS,
      maxValuesPerCondition: MAX_FILTER_VALUES,
    },
    ...(propertyIndexError !== null && { propertyIndexError }),
  };
}
