import path from 'node:path';
import { ENGINE_API_VERSION } from './capabilities.js';
import { buildMatchText, buildMatchTextFromMarkedSnippet, stripSnippetMarkers } from './chunker.js';
import { config } from './config.js';
import {
  countNotesByPredicate,
  ensurePropertyIndexFresh,
  filterNotePathsByFrontmatter,
  filterNotePathsByPredicate,
  filterNotePathsByTag,
  getBacklinksForPaths,
  getChunkEmbeddingsByPath,
  getDb,
  getDbVersion,
  getLinksForPaths,
  getMarkdownLinksForPaths,
  getMatchingNotesByFrontmatter,
  getNoteByPath,
  getOutgoingLinksForPaths,
  getPropertyIndexError,
  getUrlsForPaths,
  hasVecTable,
  queryNotesByPredicate,
  resolveNotePath,
} from './db.js';
import { embed } from './embedder.js';
import { buildFilterPredicate, hasFilterValue, type FilterPredicate } from './filter-predicate.js';
import {
  extractMarkdownReferenceOccurrences,
  resolveMarkdownNoteLinks,
} from './markdown-references.js';
import {
  blockEmbeddingText,
  candidateBlocks,
  chooseBlock,
  findExactOccurrence,
  findTermOccurrence,
  isLiteralQuery,
  passageRange,
  queryTerms,
  splitMarkdownBlocks,
  type PassageBlock,
} from './passage-blocks.js';
import {
  choosePrimaryPassage,
  type NotePassages,
  type Passage,
  type PassageKind,
} from './passage-selection.js';
import { reranker, type RerankCandidate } from './reranker.js';
import { measureSearchStage, measureSearchStageSync } from './search-profile.js';
import {
  hasStructuredFilters,
  StructuredFiltersError,
  structuredFiltersKey,
  type StructuredFilters,
} from './structured-filters.js';

export interface NoteReadResult {
  path: string;
  found: true;
  title: string;
  aliases: string[];
  tags: string[];
  content: string;
  links: string[];
  backlinks: string[];
  markdownLinks: string[];
  markdownBacklinks: string[];
  urls: string[];
}

export interface NoteReadMiss {
  path: string;
  found: false;
  suggestions: string[];
}

export type ReadResult = NoteReadResult | NoteReadMiss;

export class AmbiguousNotePathError extends Error {
  constructor(
    public readonly input: string,
    public readonly candidates: string[],
  ) {
    super(`Ambiguous note path "${input}"`);
    this.name = 'AmbiguousNotePathError';
  }
}

export function isAmbiguousNotePathError(err: unknown): err is AmbiguousNotePathError {
  return err instanceof AmbiguousNotePathError;
}

export interface MatchAnchor {
  kind: 'bm25' | 'semantic';
  headingPath: string | null;
  matchText: string; // first ~80 stripped chars of chunk body for DOM block lookup
  charStart: number | null; // null if chunk indexed before this migration
  charEnd: number | null;
  /**
   * Body-relative offsets of a verified exact hit: the FTS hit for BM25 anchors, or the
   * whole identifier/phrase inside a semantic passage (hybrid and full-text only).
   */
  matchStart?: number;
  matchEnd?: number;
  /**
   * Body-relative bounds of the selected passage (one Markdown block, sometimes with
   * the next block as context). The snippet starts here and `matchText` is its first
   * line (semantic) or the hit line (BM25). Ranked text searches only.
   */
  passageStart?: number;
  passageEnd?: number;
}

export interface SearchResult {
  path: string;
  title: string;
  tags: string[];
  aliases: string[];
  score: number;
  rank?: number; // 1-based position in the result set (populated by search())
  depth?: number; // only present in related mode (negative = backlink direction)
  snippet: string;
  matchedBy: string[];
  links: string[];
  backlinks: string[];
  markdownLinks: string[];
  markdownBacklinks: string[];
  urls: string[];
  scores: {
    semantic: number | null;
    bm25: number | null;
    fuzzy_title: number | null;
    hybrid: number | null;
  };
  previewAnchors?: MatchAnchor[];
  primaryAnchorIndex?: number;
  /**
   * Where `snippet` came from: the selected primary passage (`bm25` / `semantic`), or
   * the start of the note (`intro`) when the result has no retrieval passage (e.g. a
   * title-only match). Ranked searches only.
   */
  snippetSource?: 'bm25' | 'semantic' | 'intro';
}

export interface SearchOptions {
  mode?: 'hybrid' | 'semantic' | 'fulltext' | 'title';
  scope?: string | string[];
  limit?: number;
  threshold?: number;
  tag?: string | string[];
  frontmatter?: string | string[];
  related?: boolean;
  depth?: number;
  direction?: 'outgoing' | 'backlinks' | 'both';
  linkType?: 'wiki' | 'markdown' | 'all';
  snippetLength?: number;
  /** Explicit note path for similarity/related lookup — overrides the input heuristic */
  notePath?: string;
  rerank?: boolean;
  /**
   * Multi-query fan-out: run parallel searches for each query and merge via RRF.
   * Use when you have 2–4 reformulations of the same question.
   * Reranking (if enabled) is applied once after all results are merged.
   * When length ≤ 1, falls back to single-query behaviour.
   */
  queries?: string[];
  /** When true, populates previewAnchors on each result with chunk location data. */
  anchors?: boolean;
  /**
   * Structured metadata conditions (validated by parseStructuredFilters). ANDed with
   * tag/scope/frontmatter and applied inside every retrieval arm, before any limit.
   */
  filters?: StructuredFilters;
  /** Page offset. Only supported for metadata-only searches (empty query + filters). */
  offset?: number;
}

/**
 * Describes the result page. `total` is an exact document count and is only reported
 * when the engine actually computed it (metadata-only searches). Ranked searches report
 * `total: null` — the number of notes that would match a ranked query is not known.
 */
export interface SearchMeta {
  apiVersion: number;
  kind: 'ranked' | 'metadata' | 'related' | 'similar';
  returned: number;
  offset: number;
  /** Page size used; null means "no limit". */
  limit: number | null;
  total: number | null;
  totalExact: boolean;
  hasMore: boolean | null;
  nextOffset: number | null;
}

export interface SearchResponse {
  results: SearchResult[];
  meta: SearchMeta;
}

interface RawResult {
  path: string;
  title: string;
  tags: string;
  parsedTags?: string[];
  aliases?: string | null;
  snippet: string;
  score: number;
  chunkText?: string; // best chunk text for reranker input
  scores: {
    semantic?: number;
    bm25?: number;
    fuzzy_title?: number;
    hybrid?: number; // RRF score, set when mode='hybrid'
  };
  semanticAnchor?: MatchAnchor;
  bm25Anchor?: MatchAnchor;
  /** Per-branch passages; the one chosen by choosePrimaryPassage drives snippet + anchor. */
  passages?: NotePassages;
  primaryPassage?: PassageKind | null;
}

export function matchesScopeFilter(notePath: string, scope: string | string[]): boolean {
  const scopes = (Array.isArray(scope) ? scope : [scope]).map((s) => s.normalize('NFD'));
  const includes = scopes.filter((s) => !s.startsWith('-'));
  const excludes = scopes.filter((s) => s.startsWith('-')).map((s) => s.slice(1));
  // A scope prefix matches a path when the path is exactly the prefix or the
  // prefix is a parent directory (followed by '/'). This prevents 'notes' from
  // matching 'notes-old/foo.md'. Multiple excludes = AND logic (note must NOT
  // match ANY of the excludes).
  const scopeMatches = (path: string, prefix: string): boolean =>
    path === prefix || path.startsWith(prefix.endsWith('/') ? prefix : prefix + '/');
  if (excludes.some((ex) => scopeMatches(notePath, ex))) return false;
  // Multiple includes = OR logic (note must match ANY of the includes)
  if (includes.length === 0) return true;
  return includes.some((inc) => scopeMatches(notePath, inc));
}

function applyThreshold(results: RawResult[], threshold: number): RawResult[] {
  return results.filter((r) => r.score >= threshold);
}

function applyTagFilter(results: RawResult[], tag: string | string[]): RawResult[] {
  const allowedPaths = filterNotePathsByTag(
    results.map((result) => result.path),
    tag,
  );
  return results.filter((result) => allowedPaths.has(result.path));
}

function applyFrontmatterFilter(results: RawResult[], frontmatter: string | string[]): RawResult[] {
  const allowedPaths = filterNotePathsByFrontmatter(
    results.map((result) => result.path),
    frontmatter,
  );
  return results.filter((result) => allowedPaths.has(result.path));
}

function toSearchResult(r: RawResult): SearchResult {
  const tags = getParsedTags(r);
  const aliases = parseAliases(r.aliases);
  const matchedBy: string[] = [];
  if (r.scores.semantic != null) matchedBy.push('semantic');
  if (r.scores.bm25 != null) matchedBy.push('bm25');
  if (r.scores.fuzzy_title != null) matchedBy.push('title');
  const { chunkText: _chunkText, passages: _passages, primaryPassage: _primary, ...rest } = r;
  return {
    ...rest,
    tags,
    aliases,
    matchedBy,
    links: [],
    backlinks: [],
    markdownLinks: [],
    markdownBacklinks: [],
    urls: [],
    scores: {
      semantic: r.scores.semantic ?? null,
      bm25: r.scores.bm25 ?? null,
      fuzzy_title: r.scores.fuzzy_title ?? null,
      hybrid: r.scores.hybrid ?? null,
    },
  };
}

function parseAliases(raw: string | null | undefined): string[] {
  if (!raw) return [];
  try {
    return JSON.parse(raw) as string[];
  } catch {
    return [];
  }
}

function getParsedTags(result: RawResult): string[] {
  if (result.parsedTags) return result.parsedTags;
  try {
    result.parsedTags = JSON.parse(result.tags || '[]') as string[];
  } catch {
    result.parsedTags = [];
  }
  return result.parsedTags;
}

function sanitizeFtsQuery(query: string): string {
  return query
    .replace(/["*^()]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

function toFtsQuery(query: string, operator: 'AND' | 'OR' = 'AND'): string {
  const clean = sanitizeFtsQuery(query);
  const words = clean.split(/\s+/).filter(Boolean);
  if (words.length === 0) return '""';
  return words.map((w) => `"${w}"*`).join(` ${operator} `);
}

type FtsRow = {
  path: string;
  title: string;
  tags: string;
  aliases: string | null;
  snippet: string;
  rank: number;
};

// Sentinel markers bracketing the actual FTS match inside snippet() output. A snippet
// window spans several sentences, so the true match isn't necessarily on its first
// line — these let buildMatchTextFromMarkedSnippet find the right one. Stripped from
// every snippet before it's exposed on RawResult; control chars won't appear in notes.
const SNIPPET_MARK_START = '';
const SNIPPET_MARK_END = '';

/**
 * The `AND <predicate>` fragment and the params to splice, or empty strings/arrays when
 * no filter is active.
 *
 * An inactive filter MUST contribute a byte-identical SQL string. Not for statement
 * reuse — better-sqlite3 has no SQL-text statement cache, every db.prepare() compiles
 * afresh — but because identical text is what keeps the unfiltered path's query plan and
 * its output stable, which is the property the frozen output capture pins. An
 * unconditional `AND 1` would hand SQLite a different statement to plan on every
 * unfiltered query in the codebase for no gain.
 */
function filterFragment(predicate: FilterPredicate | undefined): {
  clause: string;
  params: Array<string | number>;
} {
  if (predicate === undefined || predicate.isEmpty) return { clause: '', params: [] };
  return { clause: ` AND ${predicate.sql}`, params: predicate.params };
}

export function searchBm25(
  query: string,
  limit: number,
  snippetLength = 300,
  buildAnchors = false,
  predicate?: FilterPredicate,
): RawResult[] {
  const db = getDb();
  const numTokens = Math.max(10, Math.ceil(snippetLength / 4));
  const filter = filterFragment(predicate);
  // Widened from the former fixed 5-tuple: the filter params are spliced in before
  // `limit`, so the bind list is variable-length.
  const stmt = db.prepare<Array<string | number>, FtsRow>(
    `
      SELECT n.path, n.title, n.tags, n.aliases,
             snippet(notes_fts_bm25, 2, ?, ?, '...', ?) AS snippet,
             bm25(notes_fts_bm25, 10.0, 5.0, 1.0) AS rank
      FROM notes_fts_bm25
      JOIN notes n ON n.id = notes_fts_bm25.rowid
      WHERE notes_fts_bm25 MATCH ?${filter.clause}
      ORDER BY rank
      LIMIT ?
    `,
  );

  try {
    // Use OR so that documents are ranked by how many query terms they match (BM25
    // naturally scores full-match documents higher).  AND was filtering out
    // relevant documents whenever a single query word (e.g. "between", "organize")
    // was absent from the document—even if that document was the best conceptual
    // match for every other term in the query.
    const rows = stmt.all(
      SNIPPET_MARK_START,
      SNIPPET_MARK_END,
      numTokens,
      toFtsQuery(query, 'OR'),
      ...filter.params,
      limit,
    );

    const results: RawResult[] = rows.map((row) => ({
      path: row.path,
      title: row.title ?? '',
      tags: row.tags ?? '[]',
      aliases: row.aliases,
      snippet: stripSnippetMarkers(row.snippet ?? '', SNIPPET_MARK_START, SNIPPET_MARK_END),
      score: Math.max(0, Math.abs(row.rank) / (1 + Math.abs(row.rank))),
      scores: {
        bm25: Math.max(0, Math.abs(row.rank) / (1 + Math.abs(row.rank))),
      },
    }));
    // Enrich BM25 snippets with heading breadcrumb and build the BM25 passage. Its
    // chunk is the one holding the located FTS hit, so the snippet, heading path and
    // anchor all describe the same place. Skipped when neither snippet nor anchors are
    // needed (snippetLength=0 and buildAnchors=false).
    if (snippetLength > 0 || buildAnchors) {
      const markedSnippets = new Map(rows.map((row) => [row.path, row.snippet ?? '']));
      const chunkDataMap = getChunkDataForBm25Results(results, markedSnippets);
      for (let i = 0; i < results.length; i++) {
        const result = results[i]!;
        const markedSnippet = rows[i]!.snippet ?? '';
        const data = chunkDataMap.get(result.path);
        if (snippetLength > 0) {
          // Keep the hit inside the final snippetLength cap: snippet() counts tokens and
          // may start its window at the top of the note, far before the hit.
          const prefix = data?.headingPath ? `${data.headingPath}\n` : '';
          const window = focusSnippetOnHit(markedSnippet, snippetLength - prefix.length);
          if (window !== null) result.snippet = window;
          result.snippet = prefix + result.snippet;
        }
        let anchor: MatchAnchor | undefined;
        if (data) {
          // Use the still-marked raw snippet (positioned near the match) for matchText,
          // not the full chunk — the sentinel markers pin down which line is the real hit.
          const rawSnippet = markedSnippet.replace(/\.\.\./g, '');
          anchor = {
            kind: 'bm25',
            headingPath: data.headingPath,
            matchText: buildMatchTextFromMarkedSnippet(
              rawSnippet,
              SNIPPET_MARK_START,
              SNIPPET_MARK_END,
            ),
            charStart: data.charStart,
            charEnd: data.charEnd,
            ...(data.matchStart !== undefined && {
              matchStart: data.matchStart,
              matchEnd: data.matchEnd,
            }),
          };
          if (buildAnchors) result.bm25Anchor = anchor;
        }
        result.passages = {
          bm25: {
            kind: 'bm25',
            snippet: result.snippet,
            ...(data && data.chunkText !== '' && { chunkText: data.chunkText }),
            // Positions are kept even without --anchors: passage selection compares them.
            ...(anchor && { anchor }),
            // Markers appear in the content column only when the body itself matched.
            bodyMatch: markedSnippet.includes(SNIPPET_MARK_START),
          },
        };
      }
    }
    return results;
  } catch {
    return [];
  }
}

function buildTrigramOrQuery(text: string): string {
  if (text.length < 3) return `"${text.replace(/"/g, '')}"`;
  const trigrams: string[] = [];
  for (let i = 0; i <= text.length - 3; i++) {
    trigrams.push(`"${text.slice(i, i + 3).replace(/"/g, '')}"`);
  }
  return trigrams.join(' OR ');
}

/**
 * Normalized BM25 score below which a hit only matched terms present in most notes:
 * FTS5 floors such a term's IDF at 1e-6, so its score is around 1e-6 while any
 * discriminative term scores orders of magnitude higher.
 */
const MIN_BM25_EVIDENCE = 1e-4;

/** Share of trigrams one side must share with the other to count as title evidence. */
const MIN_TITLE_TRIGRAM_SHARE = 0.34;

/**
 * True when a partial title match is real evidence: enough of the query's trigrams
 * appear in the title or alias (a typo or partial word), or enough of the title's
 * trigrams appear in the query (the title is named inside a longer query).
 */
export function hasTitleEvidence(
  query: string,
  result: Pick<RawResult, 'title' | 'aliases'>,
): boolean {
  const names = [result.title ?? '', ...parseAliases(result.aliases).filter((a) => a.length >= 3)];
  return names.some(
    (name) =>
      name.length > 0 &&
      (calculateTrigramOverlap(query, name) >= MIN_TITLE_TRIGRAM_SHARE ||
        calculateTrigramOverlap(name, query) >= MIN_TITLE_TRIGRAM_SHARE),
  );
}

function getTrigrams(text: string): Set<string> {
  const trigrams = new Set<string>();
  if (text.length < 3) {
    trigrams.add(text.toLowerCase());
    return trigrams;
  }
  for (let i = 0; i <= text.length - 3; i++) {
    trigrams.add(text.slice(i, i + 3).toLowerCase());
  }
  return trigrams;
}

function calculateTrigramOverlap(query: string, title: string): number {
  const queryTrigrams = getTrigrams(query);
  if (queryTrigrams.size === 0) return 0;

  const titleTrigrams = getTrigrams(title);
  let matchCount = 0;
  for (const t of queryTrigrams) {
    if (titleTrigrams.has(t)) matchCount++;
  }

  return matchCount / queryTrigrams.size;
}

/**
 * Exact alias match using JS-level Unicode case-folding (NFD + toLowerCase).
 * Handles short aliases (< 3 chars) that the trigram FTS index can't tokenize,
 * and Cyrillic / non-ASCII aliases that SQLite's lower() doesn't fold correctly.
 */
function searchByAliasExact(
  query: string,
  limit: number,
  predicate?: FilterPredicate,
): RawResult[] {
  const db = getDb();
  const queryNfd = query.normalize('NFD').toLowerCase();
  const filter = filterFragment(predicate);

  const rows = db
    .prepare(
      `
      SELECT DISTINCT n.path, n.title, n.tags, n.aliases
      FROM note_aliases a
      JOIN notes n ON n.id = a.note_id
      WHERE a.alias_norm = ?${filter.clause}
      LIMIT ?
    `,
    )
    .all(queryNfd, ...filter.params, limit) as Array<{
    path: string;
    title: string;
    tags: string;
    aliases: string | null;
  }>;

  return rows.map((row) => ({
    path: row.path,
    title: row.title ?? '',
    tags: row.tags ?? '[]',
    aliases: row.aliases,
    snippet: '',
    score: 1.0,
    scores: { fuzzy_title: 1.0 },
  }));
}

/**
 * `predicate` is optional because readNotes() calls this for path-miss suggestions,
 * where the caller's filters must NOT apply.
 */
export function searchFuzzyTitle(
  query: string,
  limit: number,
  predicate?: FilterPredicate,
): RawResult[] {
  // Exact alias match first — handles short aliases (< 3 chars) and Cyrillic that
  // the trigram FTS can't tokenize, ensuring they always surface in title/hybrid mode.
  // It MUST get the same predicate: alias hits enter RRF at weight 2.0, and once the
  // post-filter is gone there is nothing downstream left to drop unfiltered ones.
  const aliasExact = searchByAliasExact(query, limit, predicate);
  const aliasExactPaths = new Set(aliasExact.map((r) => r.path));

  const db = getDb();
  const ftsQuery = buildTrigramOrQuery(query);
  const filter = filterFragment(predicate);

  try {
    const ftsRows = db
      .prepare(
        `
      SELECT n.path, n.title, n.tags, n.aliases,
             bm25(notes_fts_fuzzy) AS rank
      FROM notes_fts_fuzzy
      JOIN notes n ON n.id = notes_fts_fuzzy.rowid
      WHERE notes_fts_fuzzy MATCH ?${filter.clause}
      ORDER BY rank
      LIMIT ?
    `,
      )
      .all(ftsQuery, ...filter.params, limit) as Array<{
      path: string;
      title: string;
      tags: string;
      aliases: string | null;
      rank: number;
    }>;

    const trigramResults = ftsRows
      .map((row) => {
        const titleOverlap = calculateTrigramOverlap(query, row.title ?? '');
        // Only consider aliases with ≥3 chars — shorter strings produce no trigrams
        // in the FTS index and would always give overlap=0 anyway.
        const aliasOverlap = parseAliases(row.aliases)
          .filter((a) => a.length >= 3)
          .reduce((max, a) => Math.max(max, calculateTrigramOverlap(query, a)), 0);
        const overlap = Math.max(titleOverlap, aliasOverlap);
        const baseScore = Math.max(0, Math.abs(row.rank) / (1 + Math.abs(row.rank)));
        const adjustedScore = baseScore * overlap * overlap;

        return {
          path: row.path,
          title: row.title ?? '',
          tags: row.tags ?? '[]',
          aliases: row.aliases,
          snippet: '',
          score: adjustedScore,
          scores: {
            fuzzy_title: adjustedScore,
          },
        };
      })
      .filter((r) => r.score > 0 && !aliasExactPaths.has(r.path));

    // Alias exact matches at the front, trigram results appended (deduped)
    return [...aliasExact, ...trigramResults].slice(0, limit);
  } catch {
    return [...aliasExact];
  }
}

/**
 * Embed a search query. Returns null if embedding failed (already retried internally).
 */
async function embedQuery(text: string): Promise<Float32Array | null> {
  return measureSearchStage('embedQuery', async () => {
    try {
      const [emb] = await embed([text], 'query');
      if (emb) {
        // Passage refinement compares blocks with the same query vector after
        // retrieval; retrieval itself never reads this memo.
        if (text.length <= 2000) queryEmbeddingMemo.set(embeddingMemoKey(text), emb);
        return emb;
      }
      // null = embedding failed (already retried internally)
      return null;
    } catch {
      return null;
    }
  });
}

/**
 * Format a chunk snippet with optional heading breadcrumb.
 * Strips the leading heading line from chunk text to avoid repeating what's in headingPath.
 */
function formatChunkSnippet(
  headingPath: string | null,
  chunkText: string,
  isNonFirst: boolean,
): string {
  const continuationPrefix = isNonFirst ? '...' : '';
  if (!headingPath) return continuationPrefix + chunkText;
  // Strip the leading heading line (e.g. "## Section\n") from chunk text — it's already
  // shown in the breadcrumb above, so displaying it again is redundant.
  const body = chunkText.replace(/^#{1,6}\s+[^\n]*\n?/, '');
  return `${headingPath}\n${continuationPrefix}${body}`;
}

/**
 * Look up the heading_path for the section containing the given BM25 snippet text.
 *
 * Strategy 1 — chunk lookup: works when the note was split into section chunks
 * (contextLength < note size). Finds the chunk whose text contains the snippet key.
 *
 * Strategy 2 — content scan fallback: used when the entire note is stored as one
 * chunk (large-context models like OpenAI text-embedding-3-small fit whole notes).
 * Finds the snippet position in notes.content, then walks backwards to build the
 * heading chain from surrounding headings.
 */
function getHeadingPathForSnippet(notePath: string, snippetText: string): string | null {
  const db = getDb();
  const clean = snippetText
    .replace(/^\.\.\./, '')
    .replace(/\.\.\.$/, '')
    .trim();
  if (clean.length < 15) return null;
  const key = clean.slice(0, 60);

  // Strategy 1: chunk-based lookup
  try {
    const row = db
      .prepare(
        `SELECT c.heading_path FROM chunks c
         JOIN notes n ON n.id = c.note_id
         WHERE n.path = ? AND instr(c.text, ?) > 0
           AND c.heading_path IS NOT NULL
         ORDER BY c.chunk_index LIMIT 1`,
      )
      .get(notePath, key) as { heading_path: string } | undefined;
    if (row?.heading_path) return row.heading_path;
  } catch {
    // heading_path column may not exist yet (pre-reindex) — fall through to strategy 2
  }

  // Strategy 2: scan note content up to the snippet position
  const note = db.prepare('SELECT content FROM notes WHERE path = ?').get(notePath) as
    { content: string } | undefined;
  if (!note?.content) return null;

  const pos = note.content.indexOf(key);
  if (pos === -1) return null;

  const before = note.content.slice(0, pos);
  const headingSlots: (string | null)[] = [null, null, null, null, null, null];
  for (const line of before.split('\n')) {
    const heading = parseMarkdownHeadingLine(line);
    if (heading) {
      headingSlots[heading.level - 1] = `${heading.marker} ${heading.text}`;
      for (let i = heading.level; i < 6; i++) headingSlots[i] = null;
    }
  }
  const chain = headingSlots.filter((s): s is string => s !== null);
  return chain.length > 0 ? chain.join(' > ') : null;
}

type Bm25ChunkData = {
  headingPath: string | null;
  chunkText: string;
  charStart: number | null;
  charEnd: number | null;
  /** Body-relative offsets of the located FTS hit. */
  matchStart?: number;
  matchEnd?: number;
};

/**
 * Body-relative position of the FTS hit in a marked snippet() window. snippet() copies
 * a contiguous run of the indexed content (external content table = notes.content), so
 * the window text, minus the "..." ellipses and markers, occurs verbatim in the body.
 * Returns null when there is no marked hit or the window is not found exactly once.
 */
export function locateMarkedMatch(
  content: string,
  markedSnippet: string,
  markStart = SNIPPET_MARK_START,
  markEnd = SNIPPET_MARK_END,
): { start: number; end: number } | null {
  let window = markedSnippet;
  if (window.startsWith('...')) window = window.slice(3);
  if (window.endsWith('...')) window = window.slice(0, -3);
  const open = window.indexOf(markStart);
  if (open === -1) return null;
  const close = window.indexOf(markEnd, open);
  if (close === -1) return null;
  const before = stripSnippetMarkers(window.slice(0, open), markStart, markEnd);
  const term = stripSnippetMarkers(window.slice(open, close), markStart, markEnd);
  const plain = stripSnippetMarkers(window, markStart, markEnd);
  if (!plain.trim() || !term) return null;
  const pos = content.indexOf(plain);
  if (pos === -1 || content.indexOf(plain, pos + 1) !== -1) return null;
  const start = pos + before.length;
  return { start, end: start + term.length };
}

/**
 * The stripped snippet() window, shortened from the front when the marked hit would
 * otherwise fall outside `budget` characters. Returns null when there is no marked hit
 * (the window is used as is).
 */
export function focusSnippetOnHit(
  markedSnippet: string,
  budget: number,
  markStart = SNIPPET_MARK_START,
  markEnd = SNIPPET_MARK_END,
): string | null {
  const open = markedSnippet.indexOf(markStart);
  if (open === -1 || budget <= 0) return null;
  const close = markedSnippet.indexOf(markEnd, open);
  const plain = stripSnippetMarkers(markedSnippet, markStart, markEnd);
  const hitStart = stripSnippetMarkers(markedSnippet.slice(0, open), markStart, markEnd).length;
  const hitLength =
    close === -1
      ? 0
      : stripSnippetMarkers(markedSnippet.slice(open, close), markStart, markEnd).length;
  if (hitStart + hitLength <= Math.floor(budget * 0.8)) return plain;
  // Start a little before the hit, preferably at the start of its line.
  let start = Math.max(0, hitStart - Math.floor(budget * 0.3));
  const lineStart = plain.lastIndexOf('\n', hitStart - 1) + 1;
  if (lineStart > start) start = lineStart;
  else {
    const space = plain.indexOf(' ', start);
    if (space !== -1 && space < hitStart) start = space + 1;
  }
  return `...${plain.slice(start)}`;
}

/** Heading chain (raw heading lines joined with " > ") in effect at `offset`. */
function headingPathAtOffset(content: string, offset: number): string | null {
  const headingSlots: (string | null)[] = [null, null, null, null, null, null];
  let insideFence = false;
  for (const line of content.slice(0, offset).split('\n')) {
    if (/^(`{3,}|~{3,})/.test(line)) {
      insideFence = !insideFence;
      continue;
    }
    if (insideFence) continue;
    const heading = parseMarkdownHeadingLine(line);
    if (!heading) continue;
    headingSlots[heading.level - 1] = `${heading.marker} ${heading.text}`;
    for (let i = heading.level; i < 6; i++) headingSlots[i] = null;
  }
  const chain = headingSlots.filter((heading): heading is string => heading !== null);
  return chain.length > 0 ? chain.join(' > ') : null;
}

type ChunkRow = {
  path: string;
  heading_path: string | null;
  text: string;
  chunk_index: number;
  char_start: number | null;
  char_end: number | null;
};

/** The chunk whose bounds contain the hit; with overlapping chunks, the one starting last. */
function chunkContaining(rows: ChunkRow[], start: number, end: number): ChunkRow | undefined {
  let best: ChunkRow | undefined;
  for (const row of rows) {
    if (row.char_start === null || row.char_end === null) continue;
    if (row.char_start > start || row.char_end < end) continue;
    if (!best || row.char_start > best.char_start!) best = row;
  }
  return best;
}

function getChunkDataForBm25Results(
  results: RawResult[],
  markedSnippets: Map<string, string> = new Map(),
): Map<string, Bm25ChunkData> {
  if (results.length === 0) return new Map();

  const db = getDb();
  const located = locateBm25Matches(db, results, markedSnippets);
  const snippetKeys = getSnippetKeysForHeadingLookup(results.filter((r) => !located.has(r.path)));
  if (located.size === 0 && snippetKeys.size === 0) return new Map();

  const uniquePaths = Array.from(new Set([...located.keys(), ...snippetKeys.keys()]));
  const placeholders = uniquePaths.map(() => '?').join(', ');

  try {
    const rows = db
      .prepare(
        `SELECT n.path, c.heading_path, c.text, c.chunk_index, c.char_start, c.char_end
         FROM notes n
         JOIN chunks c ON c.note_id = n.id
         WHERE n.path IN (${placeholders})
         ORDER BY n.path, c.chunk_index`,
      )
      .all(...uniquePaths) as ChunkRow[];

    const chunkData = matchChunkDataFromRows(rows, snippetKeys);
    for (const [notePath, hit] of located) {
      const noteRows = rows.filter((row) => row.path === notePath);
      const chunk = chunkContaining(noteRows, hit.start, hit.end);
      chunkData.set(
        notePath,
        chunk
          ? {
              // The heading chain at the hit itself: a chunk that merged several small
              // sections carries no (or only its first) heading path.
              headingPath:
                headingPathAtOffset(hit.content, hit.start) ?? chunk.heading_path ?? null,
              chunkText: chunk.text,
              charStart: chunk.char_start,
              charEnd: chunk.char_end,
              matchStart: hit.start,
              matchEnd: hit.end,
            }
          : {
              // Chunks without offsets (older index): the hit position still gives the
              // heading chain and an exact location.
              headingPath: headingPathAtOffset(hit.content, hit.start),
              chunkText: hit.lineText,
              charStart: null,
              charEnd: null,
              matchStart: hit.start,
              matchEnd: hit.end,
            },
      );
    }
    fillMissingChunkData(db, chunkData, snippetKeys);
    return chunkData;
  } catch {
    const fallback = new Map<string, Bm25ChunkData>();
    for (const result of results) {
      const key = snippetKeys.get(result.path) ?? '';
      const headingPath = getHeadingPathForSnippet(result.path, result.snippet);
      if (headingPath) {
        fallback.set(result.path, { headingPath, chunkText: key, charStart: null, charEnd: null });
      }
    }
    return fallback;
  }
}

function locateBm25Matches(
  db: ReturnType<typeof getDb>,
  results: RawResult[],
  markedSnippets: Map<string, string>,
): Map<string, { start: number; end: number; content: string; lineText: string }> {
  const located = new Map<
    string,
    { start: number; end: number; content: string; lineText: string }
  >();
  const paths = results
    .map((r) => r.path)
    .filter((p) => markedSnippets.get(p)?.includes(SNIPPET_MARK_START));
  if (paths.length === 0) return located;
  const placeholders = paths.map(() => '?').join(', ');
  try {
    const notes = db
      .prepare(`SELECT path, content FROM notes WHERE path IN (${placeholders})`)
      .all(...paths) as Array<{ path: string; content: string | null }>;
    for (const note of notes) {
      if (!note.content) continue;
      const hit = locateMarkedMatch(note.content, markedSnippets.get(note.path) ?? '');
      if (!hit) continue;
      const lineStart = note.content.lastIndexOf('\n', hit.start - 1) + 1;
      const lineEndRaw = note.content.indexOf('\n', hit.end);
      const lineText = note.content.slice(
        lineStart,
        lineEndRaw === -1 ? note.content.length : lineEndRaw,
      );
      located.set(note.path, { ...hit, content: note.content, lineText });
    }
  } catch {
    // Missing columns on very old databases: fall back to snippet-key matching.
  }
  return located;
}

function getSnippetKeysForHeadingLookup(results: RawResult[]): Map<string, string> {
  const snippetKeys = new Map<string, string>();
  for (const result of results) {
    const clean = result.snippet
      .replace(/^\.\.\./, '')
      .replace(/\.\.\.$/, '')
      .trim();
    if (clean.length < 15) continue;
    snippetKeys.set(result.path, clean.slice(0, 60));
  }
  return snippetKeys;
}

function matchChunkDataFromRows(
  rows: Array<{
    path: string;
    heading_path: string | null;
    text: string;
    char_start: number | null;
    char_end: number | null;
  }>,
  snippetKeys: Map<string, string>,
): Map<string, Bm25ChunkData> {
  const result = new Map<string, Bm25ChunkData>();
  for (const row of rows) {
    const key = snippetKeys.get(row.path);
    if (!key || !row.text.includes(key)) continue;

    const existing = result.get(row.path);
    // Don't replace a heading-path match with a null-heading one (prefer heading context)
    if (existing?.headingPath) continue;

    result.set(row.path, {
      headingPath: row.heading_path ?? null,
      chunkText: row.text,
      charStart: row.char_start ?? null,
      charEnd: row.char_end ?? null,
    });
  }
  return result;
}

function fillMissingChunkData(
  db: ReturnType<typeof getDb>,
  chunkData: Map<string, Bm25ChunkData>,
  snippetKeys: Map<string, string>,
): void {
  const stmt = db.prepare('SELECT content FROM notes WHERE path = ?');
  for (const [notePath, key] of snippetKeys) {
    if (chunkData.has(notePath)) continue;
    const note = stmt.get(notePath) as { content: string } | undefined;
    if (!note?.content) continue;
    // Set data even when there's no heading before the match (headingPath may be null)
    if (!note.content.includes(key)) continue;
    const headingPath = getHeadingPathFromContent(note.content, key);
    chunkData.set(notePath, { headingPath, chunkText: key, charStart: null, charEnd: null });
  }
}

function getHeadingPathFromContent(content: string, key: string): string | null {
  const pos = content.indexOf(key);
  if (pos === -1) return null;

  const before = content.slice(0, pos);
  const headingSlots: (string | null)[] = [null, null, null, null, null, null];
  for (const line of before.split('\n')) {
    const heading = parseMarkdownHeadingLine(line);
    if (!heading) continue;
    headingSlots[heading.level - 1] = `${heading.marker} ${heading.text}`;
    for (let i = heading.level; i < 6; i++) headingSlots[i] = null;
  }
  const chain = headingSlots.filter((heading): heading is string => heading !== null);
  return chain.length > 0 ? chain.join(' > ') : null;
}

function parseMarkdownHeadingLine(
  line: string,
): { level: number; marker: string; text: string } | null {
  const trimmed = line.trim();
  let level = 0;
  while (level < 6 && trimmed[level] === '#') level++;
  if (level === 0 || !isMarkdownWhitespace(trimmed[level] ?? '')) return null;

  let textStart = level + 1;
  while (textStart < trimmed.length && isMarkdownWhitespace(trimmed[textStart]!)) textStart++;
  const text = trimmed.slice(textStart);
  if (!text) return null;
  return { level, marker: '#'.repeat(level), text };
}

function isMarkdownWhitespace(char: string): boolean {
  return char === ' ' || char === '\t' || char === '\r' || char === '\n' || char === '\f';
}

/**
 * Hard ceiling sqlite-vec enforces on a KNN `k`. Exceeding it throws
 * "k value in knn query too large", which the catch below used to swallow into an
 * empty result — so any search with limit > 819 (k = limit * 5) silently returned
 * nothing, on every vault regardless of size. Clamp instead: k = 4096 already
 * over-fetches far past any sane result count, and the outer LIMIT does the trimming.
 */
const VEC_MAX_K = 4096;

/**
 * @param excludeLinkedFrom when set, omits that note and everything it links to —
 *   the "already known" set for a similarity lookup.
 */
function searchVector(
  queryEmbedding: Float32Array,
  limit: number,
  predicate?: FilterPredicate,
  excludeLinkedFrom?: string,
): RawResult[] {
  if (!hasVecTable()) return [];

  // ── The vector arm's restriction form is the OPPOSITE of the FTS arms', and the
  // wrong one fails SILENTLY. sqlite-vec pre-filters a KNN ONLY when the restriction
  // is on its primary key as `vc.chunk_id IN (subquery)`. The correlated
  // `EXISTS (… WHERE nt.note_id = n.id …)` form — exactly what searchBm25 and
  // searchFuzzyTitle correctly use — compiles to a POST-filter here: k is taken first
  // and the restriction then throws the survivors away, so a narrow filter returns 0
  // rows instead of `limit`. Measured. Do NOT "harmonize" these two forms.
  // sqlite-vec applies a KNN restriction BEFORE taking k only when it is expressed
  // positively on the primary key, as `vc.chunk_id IN (subquery)`. `NOT IN` is NOT
  // recognized by that optimization: k is taken first and the exclusion then discards
  // the survivors, so a source note whose links fill the top-k returns nothing.
  // Measured. Everything therefore collapses into ONE positive `IN`, with the
  // exclusion carried inside it as a `NOT IN` over note ids — cheap there, because by
  // that point we are already inside a plain SQL subquery rather than the KNN itself.
  //
  // Do not "harmonize" this with the correlated EXISTS the FTS arms use: that form
  // compiles to a post-filter here and silently returns zero rows.
  const conditions: string[] = [];
  const extraParams: Array<string | number> = [];

  if (predicate !== undefined && !predicate.isEmpty) {
    // The predicate's clauses reference the alias `n`; inside this subquery the notes
    // table is aliased `n2` so it cannot shadow the outer join above.
    conditions.push(predicate.sql.replace(/\bn\./g, 'n2.'));
    extraParams.push(...predicate.params);
  }

  if (excludeLinkedFrom !== undefined) {
    // Driven from `links` rather than a correlated EXISTS: the correlated form forces
    // a full chunk scan and doubled the arm's cost (7.33 ms vs 4.94 ms on a 157-link
    // hub note). A bound parameter list is impossible — hub notes can have thousands
    // of links and such a list cannot be batched inside a single k-limited KNN.
    conditions.push(
      `c2.note_id NOT IN (
             SELECT n4.id FROM notes n4 WHERE n4.path = ?
             UNION
             SELECT n5.id FROM notes n5 JOIN links l ON l.to_path = n5.path WHERE l.from_path = ?)`,
    );
    extraParams.push(excludeLinkedFrom, excludeLinkedFrom);
  }

  const restrictions =
    conditions.length === 0
      ? []
      : [
          `vc.chunk_id IN (SELECT c2.id FROM chunks c2 JOIN notes n2 ON n2.id = c2.note_id WHERE ${conditions.join(' AND ')})`,
        ];

  const restrictionSql = restrictions.map((r) => `\n          AND ${r}`).join('');

  return measureSearchStageSync('vectorSearch', () => {
    const db = getDb();

    try {
      const rows = db
        .prepare(
          `
      WITH ranked AS (
        SELECT vc.chunk_id, vc.distance,
               c.note_id, c.chunk_index, c.text AS chunk_text, c.heading_path,
               c.char_start, c.char_end,
               n.path, n.title, n.tags, n.aliases,
               ROW_NUMBER() OVER (PARTITION BY c.note_id ORDER BY vc.distance, c.chunk_index) AS row_num
        FROM vec_chunks AS vc
        JOIN chunks c ON c.id = vc.chunk_id
        JOIN notes n ON n.id = c.note_id
        WHERE vc.embedding MATCH ?
          AND k = ?${restrictionSql}
      )
      SELECT chunk_id, distance, note_id, chunk_index, chunk_text, heading_path,
             char_start, char_end, path, title, tags, aliases
      FROM ranked
      WHERE row_num = 1
      ORDER BY distance, chunk_index
      LIMIT ?
    `,
        )
        // `k` is validated by sqlite-vec BEFORE the pre-filter runs, so the VEC_MAX_K
        // clamp stays regardless of how narrow the restriction is.
        .all(queryEmbedding, Math.min(limit * 5, VEC_MAX_K), ...extraParams, limit) as Array<{
        chunk_id: number;
        distance: number;
        note_id: number;
        chunk_index: number;
        chunk_text: string;
        heading_path: string | null;
        char_start: number | null;
        char_end: number | null;
        path: string;
        title: string;
        tags: string;
        aliases: string | null;
      }>;

      return rows.map((row): RawResult => {
        // cosine similarity from L2 for unit vectors: cos = 1 - L2² / 2
        const similarity = Math.max(0, 1 - (row.distance * row.distance) / 2);
        const snippet = formatChunkSnippet(row.heading_path, row.chunk_text, row.chunk_index > 0);
        const semanticAnchor: MatchAnchor = {
          kind: 'semantic' as const,
          headingPath: row.heading_path ?? null,
          matchText: buildMatchText(row.chunk_text),
          charStart: row.char_start ?? null,
          charEnd: row.char_end ?? null,
        };
        return {
          path: row.path,
          title: row.title ?? '',
          tags: row.tags ?? '[]',
          aliases: row.aliases,
          snippet,
          chunkText: row.chunk_text,
          score: similarity,
          scores: { semantic: similarity },
          semanticAnchor,
          passages: {
            semantic: {
              kind: 'semantic',
              snippet,
              chunkText: row.chunk_text,
              anchor: semanticAnchor,
            },
          },
        };
      });
    } catch (error) {
      // Never swallow silently: an empty vector result is indistinguishable from
      // "no semantic matches", which hid the k-ceiling bug above for a long time.
      process.stderr.write(
        `Vector search failed: ${error instanceof Error ? error.message : String(error)}\n`,
      );
      return [];
    }
  });
}

function rrfFusion(lists: RawResult[][], k = 60, weights?: number[]): RawResult[] {
  const scores = new Map<string, { rrfScore: number; result: RawResult }>();

  for (const [listIndex, list] of lists.entries()) {
    const w = weights?.[listIndex] ?? 1;
    list.forEach((result, rank) => {
      const rrfScore = w / (k + rank + 1);
      const existing = scores.get(result.path);
      if (existing) {
        existing.rrfScore += rrfScore;
        // Merge score details, prefer semantic snippet
        if (result.scores.semantic !== undefined) {
          existing.result.snippet = result.snippet;
          existing.result.scores.semantic = result.scores.semantic;
          if (result.semanticAnchor) existing.result.semanticAnchor = result.semanticAnchor;
        }
        if (result.scores.bm25 !== undefined) {
          existing.result.scores.bm25 = result.scores.bm25;
          if (!existing.result.scores.semantic) existing.result.snippet = result.snippet;
          if (result.bm25Anchor) existing.result.bm25Anchor = result.bm25Anchor;
        }
        if (result.scores.fuzzy_title !== undefined) {
          existing.result.scores.fuzzy_title = result.scores.fuzzy_title;
        }
        // Keep every branch's passage (the first list to bring one wins, i.e. the
        // primary query in a multi-query fan-out); passage selection picks later.
        if (result.passages) {
          existing.result.passages = { ...result.passages, ...existing.result.passages };
        }
      } else {
        scores.set(result.path, {
          rrfScore,
          result: {
            ...result,
            scores: { ...result.scores },
            ...(result.passages && { passages: { ...result.passages } }),
          },
        });
      }
    });
  }

  const activeLists = lists
    .map((l, i) => ({ list: l, weight: weights?.[i] ?? 1 }))
    .filter(({ list }) => list.length > 0);
  const maxPossibleScore =
    activeLists.length > 0 ? activeLists.reduce((sum, { weight }) => sum + weight, 0) / (k + 1) : 1;

  return Array.from(scores.values())
    .sort((a, b) => b.rrfScore - a.rrfScore)
    .map(({ rrfScore, result }) => ({
      ...result,
      score: Math.min(1, rrfScore / maxPossibleScore),
    }));
}

// ─── Graph traversal ─────────────────────────────────────

/** Get the first maxChars characters of a note's content as a fallback snippet. */
function getSnippetFallbacks(notePaths: string[], maxChars: number): Map<string, string> {
  if (notePaths.length === 0 || maxChars <= 0) return new Map();

  const db = getDb();
  const uniquePaths = Array.from(new Set(notePaths));
  const placeholders = uniquePaths.map(() => '?').join(', ');
  const rows = db
    .prepare(`SELECT path, title, content FROM notes WHERE path IN (${placeholders})`)
    .all(...uniquePaths) as Array<{ path: string; title: string | null; content: string | null }>;

  return new Map(
    rows.map((row) => {
      const contentSnippet = row.content ? row.content.slice(0, maxChars).trim() : '';
      const titleSnippet = row.title ? row.title.slice(0, maxChars).trim() : '';
      return [row.path, contentSnippet || titleSnippet];
    }),
  );
}

/**
 * Extract the sentence/line context around a wikilink [[target]] inside note content.
 */
function getLinkContext(noteContent: string, linkedNotePath: string, windowSize = 300): string {
  const basename = path.basename(linkedNotePath, '.md');
  // Match [[basename]], [[basename|alias]], [[basename#heading]], [[folder/basename]], etc.
  const escaped = basename.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const regex = new RegExp(`\\[\\[(?:[^\\]|#]*\\/)?${escaped}(?:[|#][^\\]]*)?\\]\\]`, 'i');

  const match = regex.exec(noteContent);
  if (!match) return '';

  let start = Math.max(0, match.index - windowSize);
  let end = Math.min(noteContent.length, match.index + match[0].length + windowSize);

  // Snap to word boundaries so we don't cut mid-word
  if (start > 0) {
    while (start < match.index && !/\s/.test(noteContent[start]!)) start++;
  }
  if (end < noteContent.length) {
    while (end > match.index + match[0].length && !/\s/.test(noteContent[end - 1]!)) end--;
  }

  const prefix = start > 0 ? '...' : '';
  const suffix = end < noteContent.length ? '...' : '';
  // Preserve newlines so CLI can strip line-based markdown (headings, blockquotes)
  return prefix + noteContent.slice(start, end).trim() + suffix;
}

function getContextAroundMatch(
  noteContent: string,
  matchIndex: number,
  matchLength: number,
  windowSize: number,
): string {
  const contextBudget = Math.max(0, windowSize - matchLength);
  const beforeBudget = Math.floor(contextBudget / 2);
  const afterBudget = contextBudget - beforeBudget;
  let start = Math.max(0, matchIndex - beforeBudget);
  let end = Math.min(noteContent.length, matchIndex + matchLength + afterBudget);

  if (start > 0) {
    while (start < matchIndex && !/\s/.test(noteContent[start]!)) start++;
  }
  if (end < noteContent.length) {
    while (end > matchIndex + matchLength && !/\s/.test(noteContent[end - 1]!)) end--;
  }

  const prefix = start > 0 ? '...' : '';
  const suffix = end < noteContent.length ? '...' : '';
  return prefix + noteContent.slice(start, end).trim() + suffix;
}

function getMarkdownLinkContext(
  noteContent: string,
  sourcePath: string,
  linkedNotePath: string,
  windowSize = 300,
): string {
  const normalizedTarget = linkedNotePath.normalize('NFD');
  const occurrences = extractMarkdownReferenceOccurrences(noteContent).localDestinations;
  for (const occurrence of occurrences) {
    const resolved = resolveMarkdownNoteLinks(
      sourcePath,
      [occurrence.destination],
      new Set([normalizedTarget]),
    );
    if (resolved.includes(normalizedTarget)) {
      return getContextAroundMatch(
        noteContent,
        occurrence.startOffset,
        occurrence.endOffset - occurrence.startOffset,
        windowSize,
      );
    }
  }

  return '';
}

function getCachedNoteContent(
  db: ReturnType<typeof getDb>,
  cache: Map<string, string>,
  notePath: string,
): string {
  const cached = cache.get(notePath);
  if (cached !== undefined) return cached;
  const note = db.prepare('SELECT content FROM notes WHERE path = ?').get(notePath) as
    { content: string } | undefined;
  const content = note?.content ?? '';
  cache.set(notePath, content);
  return content;
}

// eslint-disable-next-line sonarjs/cognitive-complexity -- primary search aggregation pipeline, intentionally dense
function searchRelated(
  notePath: string,
  maxDepth: number,
  direction: 'outgoing' | 'backlinks' | 'both' = 'both',
  snippetLength = 300,
  linkType: 'wiki' | 'markdown' | 'all' = 'wiki',
): SearchResult[] {
  const db = getDb();
  const sourcePath = notePath.normalize('NFD');
  const results: SearchResult[] = [];
  const resultIndexes = new Map<string, number>();
  const contentCache = new Map<string, string>();

  const makeResult = (notePth: string, depth: number, snippet: string): SearchResult | null => {
    const note = db
      .prepare('SELECT path, title, tags, aliases FROM notes WHERE path = ?')
      .get(notePth) as
      { path: string; title: string; tags: string; aliases: string | null } | undefined;
    if (!note) return null;
    let tags: string[];
    try {
      tags = JSON.parse(note.tags || '[]') as string[];
    } catch {
      tags = [];
    }
    return {
      path: note.path,
      title: note.title ?? '',
      tags,
      aliases: parseAliases(note.aliases),
      score: 1 / (1 + Math.abs(depth)),
      depth,
      snippet,
      matchedBy: depth === 0 ? ['source'] : depth > 0 ? ['link'] : ['backlink'],
      links: [],
      backlinks: [],
      markdownLinks: [],
      markdownBacklinks: [],
      urls: [],
      scores: { semantic: null, bm25: null, fuzzy_title: null, hybrid: null },
    };
  };

  // Source note at depth 0
  const source = makeResult(sourcePath, 0, '');
  if (!source) return [];
  results.push(source);
  resultIndexes.set(sourcePath, 0);

  const addOrMergeResult = (
    notePth: string,
    depth: number,
    snippet: string,
    matchedBy: 'link' | 'backlink' | 'markdown_link' | 'markdown_backlink',
  ): boolean => {
    const existingIndex = resultIndexes.get(notePth);
    if (existingIndex !== undefined) {
      const existing = results[existingIndex]!;
      const existingAbs = Math.abs(existing.depth ?? 0);
      const nextAbs = Math.abs(depth);
      if (nextAbs < existingAbs) {
        const replacement = makeResult(notePth, depth, snippet);
        if (!replacement) return false;
        replacement.matchedBy = [matchedBy];
        results[existingIndex] = replacement;
        return false;
      }
      if (nextAbs === existingAbs && !existing.matchedBy.includes(matchedBy)) {
        existing.matchedBy.push(matchedBy);
      }
      return false;
    }

    const r = makeResult(notePth, depth, snippet);
    if (!r) return false;
    r.matchedBy = [matchedBy];
    resultIndexes.set(notePth, results.length);
    results.push(r);
    return true;
  };

  const getOutgoingEdges = (
    frontierPaths: string[],
  ): Map<string, Array<{ path: string; matchedBy: 'link' | 'markdown_link' }>> => {
    const edges = new Map<string, Array<{ path: string; matchedBy: 'link' | 'markdown_link' }>>();
    for (const frontierPath of frontierPaths) edges.set(frontierPath, []);
    if (linkType === 'wiki' || linkType === 'all') {
      const wikiLinks = getOutgoingLinksForPaths(frontierPaths);
      for (const frontierPath of frontierPaths) {
        for (const target of wikiLinks.get(frontierPath) ?? []) {
          edges.get(frontierPath)!.push({ path: target, matchedBy: 'link' });
        }
      }
    }
    if (linkType === 'markdown' || linkType === 'all') {
      const markdownLinks = getMarkdownLinksForPaths(frontierPaths).links;
      for (const frontierPath of frontierPaths) {
        for (const target of markdownLinks.get(frontierPath) ?? []) {
          edges.get(frontierPath)!.push({ path: target, matchedBy: 'markdown_link' });
        }
      }
    }
    return edges;
  };

  const getBacklinkEdges = (
    frontierPaths: string[],
  ): Map<string, Array<{ path: string; matchedBy: 'backlink' | 'markdown_backlink' }>> => {
    const edges = new Map<
      string,
      Array<{ path: string; matchedBy: 'backlink' | 'markdown_backlink' }>
    >();
    for (const frontierPath of frontierPaths) edges.set(frontierPath, []);
    if (linkType === 'wiki' || linkType === 'all') {
      const wikiBacklinks = getBacklinksForPaths(frontierPaths);
      for (const frontierPath of frontierPaths) {
        for (const source of wikiBacklinks.get(frontierPath) ?? []) {
          edges.get(frontierPath)!.push({ path: source, matchedBy: 'backlink' });
        }
      }
    }
    if (linkType === 'markdown' || linkType === 'all') {
      const markdownBacklinks = getMarkdownLinksForPaths(frontierPaths).backlinks;
      for (const frontierPath of frontierPaths) {
        for (const source of markdownBacklinks.get(frontierPath) ?? []) {
          edges.get(frontierPath)!.push({ path: source, matchedBy: 'markdown_backlink' });
        }
      }
    }
    return edges;
  };

  // Forward BFS — follow outgoing links (+depth)
  if (direction === 'outgoing' || direction === 'both') {
    const visitedFwd = new Set<string>([sourcePath]);
    let frontier = [sourcePath];
    for (let d = 1; d <= maxDepth; d++) {
      const next: string[] = [];
      const linksByParent = getOutgoingEdges(frontier);
      for (const parentPath of frontier) {
        for (const edge of linksByParent.get(parentPath) ?? []) {
          const parentContent = getCachedNoteContent(db, contentCache, parentPath);
          const snippet =
            edge.matchedBy === 'markdown_link'
              ? getMarkdownLinkContext(parentContent, parentPath, edge.path, snippetLength)
              : getLinkContext(parentContent, edge.path, snippetLength);
          addOrMergeResult(edge.path, d, snippet, edge.matchedBy);
          if (visitedFwd.has(edge.path)) continue;
          visitedFwd.add(edge.path);
          if (resultIndexes.has(edge.path)) next.push(edge.path);
        }
      }
      frontier = next;
      if (frontier.length === 0) break;
    }
  }

  // Backward BFS — follow backlinks (-depth)
  if (direction === 'backlinks' || direction === 'both') {
    const visitedBwd = new Set<string>([sourcePath]);
    let frontier = [sourcePath];
    for (let d = 1; d <= maxDepth; d++) {
      const next: string[] = [];
      const backlinksByParent = getBacklinkEdges(frontier);
      for (const parentPath of frontier) {
        for (const edge of backlinksByParent.get(parentPath) ?? []) {
          const fromContent = getCachedNoteContent(db, contentCache, edge.path);
          const snippet =
            edge.matchedBy === 'markdown_backlink'
              ? getMarkdownLinkContext(fromContent, edge.path, parentPath, snippetLength)
              : getLinkContext(fromContent, parentPath, snippetLength);
          addOrMergeResult(edge.path, -d, snippet, edge.matchedBy);
          if (visitedBwd.has(edge.path)) continue;
          visitedBwd.add(edge.path);
          if (resultIndexes.has(edge.path)) next.push(edge.path);
        }
      }
      frontier = next;
      if (frontier.length === 0) break;
    }
  }

  // Sort: -N ... -1, 0, +1 ... +N
  results.sort((a, b) => (a.depth ?? 0) - (b.depth ?? 0));

  // Populate links/backlinks for all results
  const paths = results.map((r) => r.path);
  const { links, backlinks } = getLinksForPaths(paths);
  const markdown = getMarkdownLinksForPaths(paths);
  const urls = getUrlsForPaths(paths);
  for (const r of results) {
    r.links = links.get(r.path) ?? [];
    r.backlinks = backlinks.get(r.path) ?? [];
    r.markdownLinks = markdown.links.get(r.path) ?? [];
    r.markdownBacklinks = markdown.backlinks.get(r.path) ?? [];
    r.urls = urls.get(r.path) ?? [];
  }

  return results;
}

// ─── LRU cache ───────────────────────────────────────────
// Avoids redundant searches when the same query is repeated (common in MCP usage).
// Path-based lookups are cached by path:mtime so stale entries auto-invalidate.

class LRUCache<V> {
  private map = new Map<string, V>();
  constructor(private maxSize: number) {}

  get(key: string): V | undefined {
    const val = this.map.get(key);
    if (val !== undefined) {
      this.map.delete(key);
      this.map.set(key, val);
    }
    return val;
  }

  set(key: string, val: V): void {
    if (this.map.has(key)) this.map.delete(key);
    else if (this.map.size >= this.maxSize) this.map.delete(this.map.keys().next().value!);
    this.map.set(key, val);
  }
}

const searchCache = new LRUCache<SearchResponse>(100);

// In-process counter for test isolation: incremented by bumpIndexVersion() so that
// separate test suites sharing the same Node.js process don't get cross-suite cache hits.
let localVersion = 0;

/**
 * Increment the local (in-process) version so that subsequent searches bypass the
 * cache. Used by tests to avoid cross-suite cache pollution. In production code,
 * DB-level versioning (getDbVersion) handles cross-process invalidation automatically.
 */
export function bumpIndexVersion(): void {
  localVersion++;
}

/** Ceiling used when the caller asks for "no limit" (limit === 0). Mirrors the
 *  filter-only branch's FETCH_ALL so both paths cap the vault the same way. */
const UNLIMITED_RESULTS = 10000;

function cacheKey(input: string, options: SearchOptions): string {
  const scopeStr = Array.isArray(options.scope) ? options.scope.join(',') : (options.scope ?? '');
  const tagStr = Array.isArray(options.tag) ? options.tag.join(',') : (options.tag ?? '');
  const fmStr = Array.isArray(options.frontmatter)
    ? options.frontmatter.join(',')
    : (options.frontmatter ?? '');
  // Include reranker model so that changing RERANKER_MODEL invalidates the cache
  const rerankStr = options.rerank ? config.rerankerModel : '';
  const queriesStr = options.queries && options.queries.length > 1 ? options.queries.join('|') : '';
  const anchorsStr = options.anchors ? 'a' : '';
  const filtersStr = structuredFiltersKey(options.filters);
  const offsetStr = options.offset ? String(options.offset) : '';
  // Two-component version:
  //   getDbVersion() — shared via SQLite settings; any process that modifies the DB
  //                    bumps it, invalidating caches in all other processes.
  //   localVersion   — in-process counter; bumpIndexVersion() for test-suite isolation.
  return `v${getDbVersion()}_${localVersion}\0${input}\0${options.mode ?? ''}\0${scopeStr}\0${options.limit ?? ''}\0${options.threshold ?? ''}\0${tagStr}\0${fmStr}\0${options.snippetLength ?? ''}\0${options.notePath ?? ''}\0${rerankStr}\0${queriesStr}\0${anchorsStr}\0${filtersStr}\0${offsetStr}`;
}

function rankedMeta(
  kind: SearchMeta['kind'],
  results: SearchResult[],
  limit: number | null,
): SearchMeta {
  return {
    apiVersion: ENGINE_API_VERSION,
    kind,
    returned: results.length,
    offset: 0,
    limit,
    total: null,
    totalExact: false,
    hasMore: null,
    nextOffset: null,
  };
}

function toMetadataResult(m: {
  path: string;
  title: string;
  tags: string;
  aliases: string | null;
}): RawResult {
  return {
    path: m.path,
    title: m.title ?? '',
    tags: m.tags ?? '[]',
    aliases: m.aliases,
    snippet: '',
    score: 1.0,
    scores: { hybrid: 1.0 },
  };
}

/**
 * Metadata-only search: no text query, so no embedding, no ranking and no reranking.
 * One SQL page over `notes` in a stable title/path order, plus an exact document count.
 */
function searchMetadataOnly(options: SearchOptions, limit: number): SearchResponse {
  const predicate = buildFilterPredicate(options);
  const offset = options.offset ?? 0;
  const pageLimit = limit === 0 ? UNLIMITED_RESULTS : limit;
  const rows = queryNotesByPredicate(predicate, pageLimit, offset);
  const total = countNotesByPredicate(predicate);
  const paths = rows.map((row) => row.path);
  const { links, backlinks } = getLinksForPaths(paths);
  const results = rows.map((row, i) => {
    const sr = toSearchResult(toMetadataResult(row));
    return {
      ...sr,
      rank: offset + i + 1,
      links: links.get(sr.path) ?? [],
      backlinks: backlinks.get(sr.path) ?? [],
    };
  });
  const end = offset + results.length;
  return {
    results,
    meta: {
      apiVersion: ENGINE_API_VERSION,
      kind: 'metadata',
      returned: results.length,
      offset,
      limit: limit === 0 ? null : limit,
      total,
      totalExact: true,
      hasMore: end < total,
      nextOffset: end < total ? end : null,
    },
  };
}

function assertStructuredFiltersUsable(options: SearchOptions): void {
  if (!hasStructuredFilters(options.filters)) return;
  const error = getPropertyIndexError();
  if (error !== null) throw new StructuredFiltersError(error);
  ensurePropertyIndexFresh();
}

/** Ranked search returning only the result list (the long-standing API). */
export async function search(input: string, options: SearchOptions = {}): Promise<SearchResult[]> {
  return (await searchWithMeta(input, options)).results;
}

// eslint-disable-next-line sonarjs/cognitive-complexity -- primary search entry-point; complexity is inherent in the multi-mode, multi-filter pipeline
export async function searchWithMeta(
  input: string,
  options: SearchOptions = {},
): Promise<SearchResponse> {
  const mode = options.mode ?? 'hybrid';
  const limit = options.limit ?? 10;
  const threshold = options.threshold ?? 0.0;
  const noLimit = limit === 0;
  const snippetLength = options.snippetLength ?? 300;
  assertStructuredFiltersUsable(options);

  // Path-based lookup when --path is given explicitly, OR when related mode is
  // requested and the input looks like a note path (allows the shorthand
  // `search('note.md', { related: true })` used in tests and the MCP server).
  // We intentionally do NOT apply the heuristic for plain text queries: previously
  // `input.includes('/')` would silently redirect to searchSimilar and ignore --mode,
  // causing `ohs "some/path.md"` and `ohss "some/path.md"` to return identical results.
  const isPathLookup =
    options.notePath !== undefined ||
    (options.related === true && (input.includes('/') || input.endsWith('.md')));
  let resolvedPath = options.notePath ?? input;
  if (isPathLookup) {
    const resolution = resolveNotePath(resolvedPath);
    if (resolution?.type === 'ambiguous') {
      throw new AmbiguousNotePathError(resolvedPath, resolution.candidates);
    }
    if (resolution?.type === 'resolved') {
      resolvedPath = resolution.path;
    }
  }

  // Related mode: graph traversal, skip the normal search pipeline
  if (isPathLookup && options.related) {
    const maxDepth = options.depth ?? 1;
    const direction = options.direction ?? 'both';
    const linkType = options.linkType ?? 'wiki';
    const scopeStr = Array.isArray(options.scope) ? options.scope.join(',') : (options.scope ?? '');
    const tagStr = Array.isArray(options.tag) ? options.tag.join(',') : (options.tag ?? '');
    const fmStr = Array.isArray(options.frontmatter)
      ? options.frontmatter.join(',')
      : (options.frontmatter ?? '');
    const key = `related\0v${getDbVersion()}_${localVersion}\0${resolvedPath}\0${maxDepth}\0${direction}\0${linkType}\0${snippetLength}\0${scopeStr}\0${tagStr}\0${fmStr}\0${structuredFiltersKey(options.filters)}`;
    const cached = searchCache.get(key);
    if (cached) return cached;
    let related = searchRelated(resolvedPath, maxDepth, direction, snippetLength, linkType);
    // Apply scope, tag, and frontmatter filters (related bypasses the normal pipeline)
    if (hasFilterValue(options.scope)) {
      related = related.filter((r) => matchesScopeFilter(r.path, options.scope!));
    }
    if (hasFilterValue(options.tag)) {
      const allowedPaths = filterNotePathsByTag(
        related.map((result) => result.path),
        options.tag!,
      );
      related = related.filter((result) => allowedPaths.has(result.path));
    }
    if (hasFilterValue(options.frontmatter)) {
      const allowedPaths = filterNotePathsByFrontmatter(
        related.map((result) => result.path),
        options.frontmatter!,
      );
      related = related.filter((result) => allowedPaths.has(result.path));
    }
    if (hasStructuredFilters(options.filters)) {
      const allowedPaths = filterNotePathsByPredicate(
        related.map((result) => result.path),
        buildFilterPredicate({ filters: options.filters }),
      );
      related = related.filter((result) => allowedPaths.has(result.path));
    }
    // The note's opening text only stands in for a missing link context; a short
    // context around the link is the relevant part and is kept as is.
    const fallbackSnippets = getSnippetFallbacks(
      related.filter((r) => !r.snippet).map((r) => r.path),
      snippetLength,
    );
    for (const r of related) {
      if (!r.snippet) r.snippet = fallbackSnippets.get(r.path) ?? '';
      if (r.snippet.length > snippetLength) r.snippet = r.snippet.slice(0, snippetLength);
    }
    related.forEach((r, i) => {
      r.rank = i + 1;
    });
    const relatedResponse = { results: related, meta: rankedMeta('related', related, null) };
    searchCache.set(key, relatedResponse);
    return relatedResponse;
  }

  // Filter-only mode: no text query, but filters present (frontmatter, tag, scope)
  // Return all matching notes sorted by title
  const hasFrontmatterFilter = hasFilterValue(options.frontmatter);
  const hasTagFilter = hasFilterValue(options.tag);
  const hasScopeFilter = hasFilterValue(options.scope);
  const hasStructured = hasStructuredFilters(options.filters);
  const isMetadataOnly =
    !input &&
    !isPathLookup &&
    (hasFrontmatterFilter || hasTagFilter || hasScopeFilter || hasStructured);

  if ((options.offset ?? 0) > 0 && !isMetadataOnly) {
    throw new StructuredFiltersError(
      'Invalid search options: offset is only supported for metadata-only searches (empty query with filters)',
    );
  }

  if (isMetadataOnly && (hasStructured || options.offset !== undefined)) {
    const metaKey = cacheKey('', options);
    const cachedMeta = searchCache.get(metaKey);
    if (cachedMeta) return cachedMeta;
    const response = measureSearchStageSync('metadataOnly', () =>
      searchMetadataOnly(options, limit),
    );
    searchCache.set(metaKey, response);
    return response;
  }

  if (isMetadataOnly) {
    const fmKey = cacheKey('', options);
    const cachedFm = searchCache.get(fmKey);
    if (cachedFm) return cachedFm;

    // Always fetch a large batch so filters can see the full vault, then slice to limit.
    const FETCH_ALL = 10000;
    let fmResults: RawResult[];

    if (hasFrontmatterFilter) {
      const matches = getMatchingNotesByFrontmatter(options.frontmatter!, FETCH_ALL);
      fmResults = matches.map((m) => ({
        path: m.path,
        title: m.title ?? '',
        tags: m.tags ?? '[]',
        aliases: m.aliases,
        snippet: '',
        score: 1.0,
        scores: { hybrid: 1.0 },
      }));
    } else {
      const db = getDb();
      const rows = db
        .prepare('SELECT path, title, tags, aliases FROM notes ORDER BY title ASC LIMIT ?')
        .all(FETCH_ALL) as Array<{
        path: string;
        title: string;
        tags: string;
        aliases: string | null;
      }>;
      fmResults = rows.map((m) => ({
        path: m.path,
        title: m.title ?? '',
        tags: m.tags ?? '[]',
        aliases: m.aliases,
        snippet: '',
        score: 1.0,
        scores: { hybrid: 1.0 },
      }));
    }

    // Apply scope filter
    if (hasScopeFilter) {
      fmResults = fmResults.filter((r) => matchesScopeFilter(r.path, options.scope!));
    }

    // Apply tag filter
    if (hasTagFilter) {
      fmResults = applyTagFilter(fmResults, options.tag!);
    }

    // Apply frontmatter filter
    if (hasFrontmatterFilter) {
      fmResults = applyFrontmatterFilter(fmResults, options.frontmatter!);
    }

    // Apply limit (0 = no limit)
    const totalMatches = fmResults.length;
    if (limit > 0) {
      fmResults = fmResults.slice(0, limit);
    }

    const finalResults = fmResults.map((r, i) => {
      const sr = toSearchResult(r);
      sr.rank = i + 1;
      return sr;
    });

    // Exact unless the FETCH_ALL ceiling truncated the candidate list.
    const totalExact = totalMatches < FETCH_ALL;
    const fmResponse: SearchResponse = {
      results: finalResults,
      meta: {
        apiVersion: ENGINE_API_VERSION,
        kind: 'metadata',
        returned: finalResults.length,
        offset: 0,
        limit: limit > 0 ? limit : null,
        total: totalExact ? totalMatches : null,
        totalExact,
        hasMore: totalExact ? finalResults.length < totalMatches : null,
        nextOffset: totalExact && finalResults.length < totalMatches ? finalResults.length : null,
      },
    };
    searchCache.set(fmKey, fmResponse);

    return fmResponse;
  }

  // For path lookups, include mtime in cache key so stale entries auto-invalidate
  let key = cacheKey(resolvedPath, options);
  if (isPathLookup) {
    const note = getNoteByPath(resolvedPath.normalize('NFD'));
    if (note) key += `\0${note.mtime}`;
  }
  const cached = searchCache.get(key);
  if (cached) return cached;

  let results: RawResult[];

  // limit === 0 means "no limit" — the semantics the filter-only branch above
  // already implements and the CLI/MCP flags document. Retrieval still needs a
  // finite pool, so it borrows the same ceiling the filter-only branch uses; the
  // final slice below is what is actually skipped.
  const retrievalLimit = noLimit ? UNLIMITED_RESULTS : limit;

  // Built ONCE and threaded into every retrieval arm, so the filter narrows the pool
  // the query itself scans instead of trimming an already-truncated result list. A
  // narrow filter used to return nothing at a small limit for exactly that reason.
  const predicate = buildFilterPredicate(options);

  if (isPathLookup) {
    results = await searchSimilar(resolvedPath, retrievalLimit, predicate);
  } else if (options.queries && options.queries.length > 1) {
    // Multi-query fan-out: run each query in parallel, merge via RRF, then rerank once.
    // Each sub-search uses a larger candidate pool so RRF has enough signal to rank correctly.
    const candidateLimit = Math.max(retrievalLimit * 2, 20);
    const perQueryResults = await Promise.all(
      options.queries.map((q) =>
        searchByQuery(
          q,
          mode,
          candidateLimit,
          snippetLength,
          false,
          options.anchors ?? false,
          predicate,
        ),
      ),
    );
    results = measureSearchStageSync('rrfFusion', () => rrfFusion(perQueryResults, 60));
    // Populate scores.hybrid on merged results (mirrors single-query hybrid path)
    for (const r of results) {
      r.scores.hybrid = r.score;
    }
    applyPassageSelection(results, options.queries[0]!);
    // Rerank after full merge — only in hybrid mode
    if (options.rerank) {
      if (mode !== 'hybrid') {
        process.stderr.write('Reranking is only supported in hybrid mode. Ignoring --rerank.\n');
      } else {
        results = await measureSearchStage('rerank', () =>
          applyRerank(results, options.queries![0]!, candidateLimit),
        );
      }
    }
  } else {
    results = await searchByQuery(
      input,
      mode,
      retrievalLimit,
      snippetLength,
      options.rerank ?? false,
      options.anchors ?? false,
      predicate,
    );
  }

  // Scope, tag and frontmatter are pushed into every retrieval arm as a SQL
  // predicate, so there is nothing left for them to trim here. applyThreshold
  // STAYS: it is a score cutoff, cannot be expressed as a predicate over `notes`,
  // and deleting it silently turns --threshold into a no-op. It must keep running
  // BEFORE the final slice, or a below-threshold result would consume a slot.
  let ranked = applyThreshold(results, threshold);
  if (!noLimit) ranked = ranked.slice(0, limit);
  // Stage two: the documents are final; only the passage each one shows is narrowed.
  const passageQuery = options.queries && options.queries.length > 1 ? options.queries[0]! : input;
  if (!isPathLookup && passageQuery.trim()) {
    await measureSearchStage('passageRefine', () =>
      refinePassages(ranked, passageQuery, mode, snippetLength),
    );
  }

  const { filteredResults, final } = measureSearchStageSync('filterAndFormat', () => {
    const filteredResults = ranked;

    const paths = filteredResults.map((r) => r.path);
    const { links, backlinks } = getLinksForPaths(paths);
    const markdown = getMarkdownLinksForPaths(paths);
    const urls = getUrlsForPaths(paths);
    // The start of the note is only a stand-in for results WITHOUT a retrieval passage
    // (title-only matches). A short passage snippet is still the relevant one and is
    // never replaced just because the note's opening text is longer.
    const fallbackSnippets = getSnippetFallbacks(
      paths.filter((_path, index) => !filteredResults[index]?.snippet),
      snippetLength,
    );

    const final = filteredResults.map((r, i) => {
      const sr = toSearchResult(r);
      if (r.primaryPassage) {
        sr.snippetSource = r.primaryPassage;
      } else if (!sr.snippet) {
        sr.snippet = fallbackSnippets.get(sr.path) ?? '';
        if (sr.snippet) sr.snippetSource = 'intro';
      }
      if (sr.snippet.length > snippetLength) sr.snippet = sr.snippet.slice(0, snippetLength);
      return {
        ...sr,
        rank: i + 1,
        links: links.get(sr.path) ?? [],
        backlinks: backlinks.get(sr.path) ?? [],
        markdownLinks: markdown.links.get(sr.path) ?? [],
        markdownBacklinks: markdown.backlinks.get(sr.path) ?? [],
        urls: urls.get(sr.path) ?? [],
      };
    });

    return { filteredResults, final };
  });

  // Populate previewAnchors when requested (not for related mode — handled above)
  if (options.anchors) {
    for (let i = 0; i < final.length; i++) {
      const raw = filteredResults[i];
      if (!raw) continue;
      // The primary passage's anchor comes first; the other branch's anchor stays as a
      // secondary one unless it points at the same block.
      const [first, second] =
        raw.primaryPassage === 'bm25'
          ? [raw.bm25Anchor, raw.semanticAnchor]
          : [raw.semanticAnchor, raw.bm25Anchor];
      const anchors: MatchAnchor[] = [];
      if (first) anchors.push(first);
      if (second && (!first || second.matchText !== first.matchText)) anchors.push(second);
      if (anchors.length > 0) {
        final[i]!.previewAnchors = anchors;
        final[i]!.primaryAnchorIndex = 0; // anchors[0] belongs to the displayed passage
      }
    }
  }

  const response: SearchResponse = {
    results: final,
    meta: rankedMeta(isPathLookup ? 'similar' : 'ranked', final, noLimit ? null : limit),
  };
  searchCache.set(key, response);
  return response;
}

/** Results whose passage is narrowed to a block; the rest keep their chunk passage. */
const PASSAGE_REFINE_LIMIT = 20;
/** Results whose blocks are compared by embedding; lower ones use lexical coverage. */
const PASSAGE_EMBED_LIMIT = 8;
/** Block embeddings computed per search at most (~5 ms each locally); cached ones are free. */
const MAX_NEW_BLOCK_EMBEDDINGS = 24;
/** Block embeddings do not depend on the query, so typing reuses them. */
const blockEmbeddingCache = new LRUCache<Float32Array>(4096);
const queryEmbeddingMemo = new LRUCache<Float32Array>(64);

/** Cache key for an embedding: the embedding configuration, the index version and the text. */
function embeddingMemoKey(text: string, withVersion = true): string {
  const model = `${config.apiBaseUrl}|${config.apiModel}|${config.localModel}`;
  return withVersion
    ? `${model}\0v${getDbVersion()}_${localVersion}\0${text}`
    : `${model}\0${text}`;
}

interface RefineJob {
  result: RawResult;
  kind: PassageKind;
  content: string;
  blocks: PassageBlock[];
  candidates: number[];
  hitOffset?: number;
}

function cosine(a: Float32Array, b: Float32Array): number | null {
  // A different dimension means another model wrote it: no score, never a prefix one.
  if (a.length !== b.length) return null;
  let dot = 0;
  let na = 0;
  let nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i]! * b[i]!;
    na += a[i]! * a[i]!;
    nb += b[i]! * b[i]!;
  }
  return na > 0 && nb > 0 ? dot / Math.sqrt(na * nb) : 0;
}

/**
 * Passage selection, stage two: inside the primary passage's chunk, pick the Markdown
 * block that answers the query and make it the snippet, heading and anchor. Document
 * order is never touched. A verified exact hit (BM25) picks its own block; otherwise
 * semantic passages are compared with block embeddings from the same local model,
 * computed only for the final top results and cached across searches. When no block
 * embedding is available, lexical coverage decides.
 */
async function refinePassages(
  results: RawResult[],
  query: string,
  mode: string,
  snippetLength: number,
): Promise<void> {
  const targets = results.slice(0, PASSAGE_REFINE_LIMIT).filter((r) => {
    const anchor = r.primaryPassage ? r.passages?.[r.primaryPassage]?.anchor : undefined;
    return anchor && (anchor.charStart !== null || anchor.matchStart !== undefined);
  });
  if (targets.length === 0) return;
  const contents = getNoteContents(targets.map((r) => r.path));

  const jobs: RefineJob[] = [];
  for (const result of targets) {
    const kind = result.primaryPassage!;
    const anchor = result.passages![kind]!.anchor!;
    const content = contents.get(result.path);
    if (!content) continue;
    let from = anchor.charStart;
    let to = anchor.charEnd;
    if (from === null || to === null) {
      // Older index without chunk offsets: the paragraph around the located hit.
      const hit = anchor.matchStart!;
      const before = content.lastIndexOf('\n\n', hit);
      const after = content.indexOf('\n\n', anchor.matchEnd ?? hit);
      from = before === -1 ? 0 : before + 2;
      to = after === -1 ? content.length : after;
    }
    if (to > content.length) continue; // stale offsets: keep the chunk passage
    const blocks = splitMarkdownBlocks(content, from, to);
    if (blocks.length === 0) continue;
    // The FTS hit decides alone for a one-term query or when its block holds the whole
    // query; for a longer full-text query it is just the first matched word, so the
    // block with the best term coverage is chosen instead.
    const hitBlock =
      kind === 'bm25' && anchor.matchStart !== undefined
        ? blocks.find((b) => anchor.matchStart! >= b.start && anchor.matchStart! < b.end)
        : undefined;
    const hitOffset =
      hitBlock &&
      (queryTerms(query).length <= 1 ||
        findExactOccurrence(content, hitBlock.start, hitBlock.end, query) !== null)
        ? anchor.matchStart
        : undefined;
    jobs.push({
      result,
      kind,
      content,
      blocks,
      candidates: candidateBlocks(blocks, query),
      hitOffset,
    });
  }

  const scores = await semanticBlockScores(
    jobs
      .filter((job) => job.kind === 'semantic' && job.candidates.length > 1)
      .slice(0, PASSAGE_EMBED_LIMIT),
    query,
  );
  for (const job of jobs) {
    const index = chooseBlock({
      blocks: job.blocks,
      query,
      hitOffset: job.hitOffset,
      semanticScores: scores.get(job),
    });
    if (index === -1) continue;
    applyRefinedPassage(job, passageRange(job.blocks, index), query, mode, snippetLength);
  }
}

async function semanticBlockScores(
  jobs: RefineJob[],
  query: string,
): Promise<Map<RefineJob, Array<number | null>>> {
  const scores = new Map<RefineJob, Array<number | null>>();
  if (jobs.length === 0) return scores;
  const texts = new Map<string, Float32Array | null>();
  for (const job of jobs) {
    for (const i of job.candidates) {
      const text = blockEmbeddingText(job.blocks[i]!.text);
      if (text && !texts.has(text))
        texts.set(text, blockEmbeddingCache.get(embeddingMemoKey(text, false)) ?? null);
    }
  }
  const missing = [...texts.entries()]
    .filter(([, vector]) => vector === null)
    .map(([text]) => text)
    .slice(0, MAX_NEW_BLOCK_EMBEDDINGS)
    // A local batch is padded to its longest text: similar lengths together are faster.
    .sort((a, b) => a.length - b.length);
  const [queryVector, vectors] = await Promise.all([
    queryEmbeddingMemo.get(embeddingMemoKey(query)) ?? embedQuery(query),
    missing.length > 0 ? embed(missing, 'document').catch(() => []) : Promise.resolve([]),
  ]);
  if (!queryVector) return scores;
  missing.forEach((text, i) => {
    const vector = vectors[i];
    if (!vector) return;
    texts.set(text, vector);
    blockEmbeddingCache.set(embeddingMemoKey(text, false), vector);
  });
  for (const job of jobs) {
    const row: Array<number | null> = job.blocks.map(() => null);
    for (const i of job.candidates) {
      const vector = texts.get(blockEmbeddingText(job.blocks[i]!.text));
      if (vector) row[i] = cosine(queryVector, vector);
    }
    // Blocks over the budget have no score; then lexical coverage decides for all.
    if (job.candidates.every((i) => row[i] !== null)) scores.set(job, row);
  }
  return scores;
}

/** Display text that keeps a far-in exact hit visible: cut at a line or sentence start. */
function passageDisplayText(
  content: string,
  range: { start: number; end: number },
  hit: { start: number; end: number } | null,
): string {
  const text = content.slice(range.start, range.end);
  if (!hit || hit.start - range.start <= 160) return text;
  const offset = hit.start - range.start;
  const floor = Math.max(0, offset - 120);
  const window = text.slice(floor, offset);
  const cut = Math.max(
    window.lastIndexOf('\n'),
    window.lastIndexOf('. '),
    window.lastIndexOf('; '),
  );
  const start = cut === -1 ? floor + Math.max(0, window.indexOf(' ') + 1) : floor + cut + 1;
  return `...${text.slice(start).trimStart()}`;
}

function applyRefinedPassage(
  job: RefineJob,
  range: { start: number; end: number },
  query: string,
  mode: string,
  snippetLength: number,
): void {
  const { result, kind, content } = job;
  const passage = result.passages![kind]!;
  const anchor = passage.anchor!;
  const text = content.slice(range.start, range.end);
  // The exact evidence to highlight: the FTS hit, or the whole identifier/phrase
  // inside a semantic passage — never in semantic mode, which shows meaning only.
  let hit: { start: number; end: number } | null = null;
  const ftsHitInside =
    kind === 'bm25' &&
    anchor.matchStart !== undefined &&
    anchor.matchEnd !== undefined &&
    anchor.matchStart >= range.start &&
    anchor.matchEnd <= range.end;
  if (ftsHitInside) {
    hit = { start: anchor.matchStart!, end: anchor.matchEnd! };
  } else if (mode !== 'semantic' && isLiteralQuery(query)) {
    hit = findExactOccurrence(content, range.start, range.end, query);
  }
  if (!hit && kind === 'bm25') hit = findTermOccurrence(content, range.start, range.end, query);
  const headingPath = headingPathAtOffset(content, range.start) ?? anchor.headingPath;
  const refined = {
    ...anchor,
    headingPath,
    // matchText must lie in the passage: the FTS hit line while the hit is inside it,
    // otherwise the passage's first line.
    matchText: ftsHitInside ? anchor.matchText : buildMatchText(text),
    charStart: anchor.charStart === null ? null : Math.min(anchor.charStart, range.start),
    charEnd: anchor.charEnd === null ? null : Math.max(anchor.charEnd, range.end),
    matchStart: hit?.start,
    matchEnd: hit?.end,
    passageStart: range.start,
    passageEnd: range.end,
  };
  passage.anchor = refined;
  if (kind === 'semantic' && result.semanticAnchor) result.semanticAnchor = refined;
  if (kind === 'bm25' && result.bm25Anchor) result.bm25Anchor = refined;
  if (snippetLength > 0) {
    const display = passageDisplayText(content, range, hit);
    passage.snippet = headingPath ? `${headingPath}\n${display}` : display;
    result.snippet = passage.snippet;
  }
}

function getNoteContents(paths: string[]): Map<string, string> {
  const contents = new Map<string, string>();
  if (paths.length === 0) return contents;
  try {
    const rows = getDb()
      .prepare(`SELECT path, content FROM notes WHERE path IN (${paths.map(() => '?').join(', ')})`)
      .all(...paths) as Array<{ path: string; content: string | null }>;
    for (const row of rows) if (row.content) contents.set(row.path, row.content);
  } catch {
    // Very old databases without stored content keep the chunk-level passage.
  }
  return contents;
}

/**
 * Make the passage chosen by choosePrimaryPassage drive the result: its snippet is
 * displayed and its chunk is what the reranker scores. Document order is untouched.
 */
function applyPassageSelection(results: RawResult[], query: string): void {
  for (const result of results) {
    const kind = choosePrimaryPassage(result.passages, query);
    result.primaryPassage = kind;
    if (!kind) continue;
    const passage: Passage = result.passages![kind]!;
    result.snippet = passage.snippet;
    if (passage.chunkText) result.chunkText = passage.chunkText;
  }
}

/**
 * For candidates that don't yet have chunkText (BM25/fuzzy results),
 * fetch the first chunk text from the DB. Mutates candidates in place.
 * better-sqlite3 is synchronous — no async needed.
 */
function fetchMissingChunkTexts(candidates: RawResult[]): void {
  const missingPaths = Array.from(
    new Set(
      candidates.filter((candidate) => !candidate.chunkText).map((candidate) => candidate.path),
    ),
  );
  if (missingPaths.length === 0) return;

  const db = getDb();
  const placeholders = missingPaths.map(() => '?').join(', ');
  const rows = db
    .prepare(
      `SELECT path, text
       FROM (
         SELECT n.path, c.text,
                ROW_NUMBER() OVER (PARTITION BY n.path ORDER BY c.chunk_index) AS row_num
         FROM chunks c
         JOIN notes n ON n.id = c.note_id
         WHERE n.path IN (${placeholders})
       )
       WHERE row_num = 1`,
    )
    .all(...missingPaths) as Array<{ path: string; text: string }>;

  const chunkTexts = new Map(rows.map((row) => [row.path, row.text]));
  for (const candidate of candidates) {
    if (candidate.chunkText) continue;
    const chunkText = chunkTexts.get(candidate.path);
    if (chunkText) candidate.chunkText = chunkText;
  }
}

/**
 * Apply cross-encoder re-ranking to a candidate list.
 * `query` is used as the reranker prompt (use primary query for multi-query).
 * Mutates nothing — returns a new sorted array.
 */
async function applyRerank(
  results: RawResult[],
  query: string,
  candidateLimit: number,
): Promise<RawResult[]> {
  if (results.length <= 1) return results;
  try {
    const candidates = results.slice(0, candidateLimit);
    fetchMissingChunkTexts(candidates); // sync — better-sqlite3 has no async API
    const rerankScores = await reranker.scoreAll(
      query,
      candidates.map((c): RerankCandidate => ({
        title: c.title,
        chunkText: c.chunkText,
        snippet: c.snippet,
      })),
    );
    // Position-aware blending: mix normalized hybrid score with sigmoid(logit).
    // Pre-rerank hybrid position determines how much we trust retrieval vs reranker:
    //   ranks  0-9  → 75% hybrid + 25% reranker  (high retrieval confidence)
    //   ranks 10-19 → 60% hybrid + 40% reranker
    //   ranks 20+   → 40% hybrid + 60% reranker  (low retrieval confidence)
    // Hybrid scores are min-max normalized within the candidate batch so both
    // signals live in [0, 1] before blending.
    const withLogits = candidates.map((c, i) => ({
      c,
      logit: rerankScores[i] ?? 0,
      origRank: i,
    }));

    const hybridScores = withLogits.map(({ c }) => c.scores.hybrid ?? c.score);
    const minH = Math.min(...hybridScores);
    const maxH = Math.max(...hybridScores);
    const rangeH = maxH - minH || 1; // guard against single-candidate edge case

    const blended = withLogits.map(({ c, logit, origRank }) => {
      const normHybrid = ((c.scores.hybrid ?? c.score) - minH) / rangeH;
      const sigmaScore = 1 / (1 + Math.exp(-logit));
      const w = origRank < 10 ? 0.75 : origRank < 20 ? 0.6 : 0.4;
      return { c, score: w * normHybrid + (1 - w) * sigmaScore };
    });

    blended.sort((a, b) => b.score - a.score);
    return blended.map(({ c, score }) => ({ ...c, score }));
  } catch (err) {
    process.stderr.write(
      `Reranking failed: ${err instanceof Error ? err.message : String(err)}. Returning original order.\n`,
    );
    return results;
  }
}

async function searchByQuery(
  query: string,
  mode: string,
  limit: number,
  snippetLength: number,
  rerank = false,
  buildAnchors = false,
  predicate?: FilterPredicate,
): Promise<RawResult[]> {
  if (rerank && mode !== 'hybrid') {
    process.stderr.write('Reranking is only supported in hybrid mode. Ignoring --rerank.\n');
    rerank = false;
  }

  if (mode === 'fulltext') {
    const bm25 = measureSearchStageSync('bm25', () =>
      searchBm25(query, limit, snippetLength, buildAnchors, predicate),
    );
    applyPassageSelection(bm25, query);
    return bm25;
  }

  if (mode === 'title') {
    // Fetch a larger candidate pool from FTS (BM25 order ≠ adjustedScore order),
    // then re-sort by adjustedScore and slice. Without this, notes ranked low by
    // BM25 but with high trigram overlap (e.g. exact title match) would be silently
    // dropped before scoring. Only done for title mode — hybrid uses rrfFusion which
    // depends on the original BM25 ordering from searchFuzzyTitle.
    const candidates = measureSearchStageSync('fuzzyTitle', () =>
      searchFuzzyTitle(query, Math.max(limit * 5, 50), predicate),
    );
    return candidates.sort((a, b) => b.score - a.score).slice(0, limit);
  }

  if (mode === 'semantic') {
    const f32 = await embedQuery(query);
    // If embedding permanently failed after retries, return empty rather than
    // polluting results with uniform zero-vector scores.
    if (!f32) return [];
    const semantic = searchVector(f32, limit, predicate);
    applyPassageSelection(semantic, query);
    return semantic;
  }

  // hybrid: RRF fusion of all three
  // embedQuery() retries on transient failures; on permanent failure it returns null
  // and vectorResults becomes [], so RRF degrades to BM25 + fuzzy (still useful).
  //
  // Use a minimum candidate pool for sub-queries so RRF has enough signal to rank
  // correctly even when limit=1. Without this, each sub-query returns only its own
  // #1 result and the BM25 winner (weight=2.0) always beats the true hybrid winner.
  // The caller (search()) already slices to `limit` after filtering.
  const candidateLimit = Math.max(limit, 20);
  const f32 = await embedQuery(query);
  const [bm25Results, fuzzyResults, vectorResults] = await Promise.all([
    Promise.resolve(
      measureSearchStageSync('bm25', () =>
        searchBm25(query, candidateLimit, snippetLength, buildAnchors, predicate),
      ),
    ),
    Promise.resolve(
      measureSearchStageSync('fuzzyTitle', () =>
        searchFuzzyTitle(query, candidateLimit, predicate),
      ),
    ),
    f32 ? searchVector(f32, candidateLimit, predicate) : Promise.resolve([]),
  ]);

  // Exact alias matches (fuzzy_title=1.0) are canonical identity signals — treated like BM25.
  // Partial fuzzy matches (trigram overlap < 1.0) remain at low weight to avoid false positives.
  const exactAliasResults = fuzzyResults.filter((r) => r.scores.fuzzy_title === 1.0);
  // RRF credits a list position, not a score, so weak evidence still moves documents.
  // Two kinds carry no real evidence and are kept out of the fusion: a title that
  // shares only a stray trigram with the query ("Stream" for "MergeTree"), and a BM25
  // hit that matched only words found in most notes (FTS5 floors their IDF, so the
  // score is ~0 — e.g. "как" in a question).
  const partialFuzzyResults = fuzzyResults.filter(
    (r) => r.scores.fuzzy_title !== 1.0 && hasTitleEvidence(query, r),
  );
  const bm25Evidence = bm25Results.filter((r) => (r.scores.bm25 ?? 0) >= MIN_BM25_EVIDENCE);
  let results = measureSearchStageSync('rrfFusion', () =>
    rrfFusion(
      [vectorResults, bm25Evidence, exactAliasResults, partialFuzzyResults],
      60,
      [1.5, 1.5, 2.0, 0.25],
    ),
  );

  // Populate scores.hybrid for hybrid mode — always, regardless of rerank flag
  for (const r of results) {
    r.scores.hybrid = r.score;
  }
  // Passage selection never reorders documents; it runs before reranking so the
  // reranker scores the same passage the result will display.
  applyPassageSelection(results, query);

  if (rerank) {
    results = await measureSearchStage('rerank', () => applyRerank(results, query, candidateLimit));
  }

  return results;
}

/**
 * Source vectors for a similarity lookup.
 *
 * Prefers the chunk embeddings stored at index time — re-embedding would truncate
 * long notes (the local model caps at 512 tokens). Falls back to re-embedding the
 * whole note only when it was indexed without embeddings, e.g. the API was down.
 *
 * The "already known" set (the note itself plus everything it links to) is NOT
 * resolved here: it is pushed into the KNN as a SQL restriction, so it narrows the
 * pool the query scans instead of eating into an already-truncated top-N.
 */
async function getSimilaritySource(normalizedPath: string): Promise<Float32Array[] | null> {
  const note = getNoteByPath(normalizedPath);
  if (!note) return null;

  const stored = getChunkEmbeddingsByPath(normalizedPath);
  if (stored.length > 0) return stored;

  const f32 = await embedQuery(`${note.title}\n\n${note.content}`);
  if (!f32) return null;
  return [f32];
}

/**
 * KNN similarity to a note, with the tag/scope/frontmatter filter and the
 * self+outgoing-link exclusion both applied INSIDE the KNN.
 *
 * There is no `limit + 1` over-fetch: it existed only to absorb self-exclusion after
 * the fact, and the exclusion is now part of the query.
 */
async function searchSimilar(
  notePath: string,
  limit: number,
  predicate?: FilterPredicate,
): Promise<RawResult[]> {
  // macOS stores filenames as NFD; normalize to match DB paths
  const normalizedPath = notePath.normalize('NFD');
  const embeddings = await getSimilaritySource(normalizedPath);
  if (!embeddings) return [];

  // Run vector search per source chunk, deduplicate by path keeping the max score
  const allResults = embeddings.flatMap((f32) =>
    searchVector(f32, limit, predicate, normalizedPath),
  );

  const byPath = new Map<string, RawResult>();
  for (const r of allResults) {
    const existing = byPath.get(r.path);
    if (!existing || r.score > existing.score) byPath.set(r.path, r);
  }

  return [...byPath.values()].sort((a, b) => b.score - a.score).slice(0, limit);
}

/**
 * Fetch one or more notes by vault-relative path.
 * Returns enriched note data (title, aliases, tags, content, links, backlinks).
 * On path miss: returns found:false with top-3 fuzzy title suggestions.
 * Results are returned in the same order as the input paths array.
 */
export function readNotes(
  paths: string[],
  options: { snippetLength?: number; related?: boolean } = {},
): ReadResult[] {
  const { snippetLength, related = true } = options;
  const results: ReadResult[] = [];

  for (const inputPath of paths) {
    const normalizedPath = inputPath.normalize('NFD');
    const note = getNoteByPath(normalizedPath);

    if (!note) {
      const basename = path.basename(inputPath, '.md');
      const suggestions = searchFuzzyTitle(basename, 3).map((r) => r.path);
      results.push({ path: inputPath, found: false, suggestions });
      continue;
    }

    let tags: string[];
    try {
      tags = JSON.parse(note.tags || '[]') as string[];
    } catch {
      tags = [];
    }

    const aliases = parseAliases(note.aliases);
    let content = note.content ?? '';
    if (snippetLength !== undefined && content.length > snippetLength) {
      content = content.slice(0, snippetLength);
    }

    let links: string[] = [];
    let backlinks: string[] = [];
    let markdownLinks: string[] = [];
    let markdownBacklinks: string[] = [];
    let urls: string[] = [];

    if (related) {
      const linkMap = getLinksForPaths([normalizedPath]);
      links = linkMap.links.get(normalizedPath) ?? [];
      backlinks = linkMap.backlinks.get(normalizedPath) ?? [];
      const markdownMap = getMarkdownLinksForPaths([normalizedPath]);
      markdownLinks = markdownMap.links.get(normalizedPath) ?? [];
      markdownBacklinks = markdownMap.backlinks.get(normalizedPath) ?? [];
    }
    urls = getUrlsForPaths([normalizedPath]).get(normalizedPath) ?? [];

    results.push({
      path: note.path,
      found: true,
      title: note.title ?? '',
      aliases,
      tags,
      content,
      links,
      backlinks,
      markdownLinks,
      markdownBacklinks,
      urls,
    });
  }

  return results;
}
