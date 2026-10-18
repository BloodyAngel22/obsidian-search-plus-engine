/**
 * Passage selection: which part of a note a ranked result shows and opens.
 *
 * Document ranking (RRF, reranking) decides WHERE a note appears in the list. This
 * module only decides WHICH passage of that note is displayed: the snippet, its heading
 * path, the primary preview anchor and the navigation target all come from the one
 * passage chosen here, so they can never point at different chunks.
 *
 * Policy (hybrid, when both retrieval branches found a passage in the same note):
 * - The BM25 passage wins only when it holds VERIFIED exact evidence for the query —
 *   the complete query as a whole token sequence, found in the matched note text —
 *   AND the query is a technical identifier (MergeTree, trace_id, FEED_DOMAIN_X) or a
 *   multi-word phrase, AND either the semantic passage does not contain that phrase,
 *   or the BM25 hit lies inside the semantic chunk itself (then both describe the same
 *   passage and the BM25 window is simply the more precise view of it — common when a
 *   short note is a single chunk and its snippet would otherwise start at the top).
 * - Otherwise the semantic passage stays primary: a single ordinary word, partial
 *   token overlap or a natural-language question never overrides it.
 * Single-branch results use the branch they have; results without any passage
 * (title or metadata matches) get no passage at all.
 */

export type PassageKind = 'bm25' | 'semantic';

export interface PassageAnchor {
  kind: PassageKind;
  headingPath: string | null;
  matchText: string;
  charStart: number | null;
  charEnd: number | null;
  matchStart?: number;
  matchEnd?: number;
  passageStart?: number;
  passageEnd?: number;
}

export interface Passage {
  kind: PassageKind;
  /** Display snippet built from this passage only (heading path line + text). */
  snippet: string;
  /** Text of the chunk the passage belongs to; reranker input when primary. */
  chunkText?: string;
  anchor?: PassageAnchor;
  /** BM25 only: the FTS hit was located in the note body (not only in the title). */
  bodyMatch?: boolean;
}

export interface NotePassages {
  semantic?: Passage;
  bm25?: Passage;
}

/** Lowercased tokens joined by single spaces; `_` stays part of a token. */
function tokenText(text: string): string {
  return text
    .toLowerCase()
    .replace(/[^\p{L}\p{N}_]+/gu, ' ')
    .trim();
}

/** True when `text` contains the whole query as a complete token sequence. */
export function containsExactQuery(text: string | undefined, query: string): boolean {
  if (!text) return false;
  const needle = tokenText(query);
  if (!needle) return false;
  return ` ${tokenText(text)} `.includes(` ${needle} `);
}

/**
 * A single token that only makes sense literally: snake_case, camelCase/PascalCase
 * with an inner capital, letters mixed with digits, or an all-caps acronym.
 */
export function isIdentifierLike(query: string): boolean {
  const words = query.trim().split(/\s+/).filter(Boolean);
  if (words.length !== 1) return false;
  const word = words[0]!;
  return (
    /\w_\w/.test(word) ||
    /\p{Ll}\p{Lu}/u.test(word) ||
    /\p{L}\p{N}|\p{N}\p{L}/u.test(word) ||
    /^\p{Lu}{2,}$/u.test(word) ||
    /\w[.:]\w/.test(word)
  );
}

function wordCount(query: string): number {
  return tokenText(query).split(' ').filter(Boolean).length;
}

/** Exact evidence in a BM25 passage: a located body hit whose passage holds the query. */
export function hasExactEvidence(passage: Passage | undefined, query: string): boolean {
  if (!passage || passage.kind !== 'bm25' || !passage.bodyMatch) return false;
  return containsExactQuery(passage.snippet, query) || containsExactQuery(passage.chunkText, query);
}

export function choosePrimaryPassage(
  passages: NotePassages | undefined,
  query: string,
): PassageKind | null {
  const semantic = passages?.semantic;
  const bm25 = passages?.bm25;
  if (!semantic && !bm25) return null;
  if (!semantic) return 'bm25';
  if (!bm25) return 'semantic';
  const literalQuery = isIdentifierLike(query) || wordCount(query) >= 2;
  if (
    literalQuery &&
    hasExactEvidence(bm25, query) &&
    (hitInsideChunk(bm25.anchor, semantic.anchor) ||
      !containsExactQuery(semantic.chunkText ?? semantic.snippet, query))
  ) {
    return 'bm25';
  }
  return 'semantic';
}

/** True when the BM25 hit (or its chunk) lies within the semantic passage's chunk. */
function hitInsideChunk(
  bm25: PassageAnchor | undefined,
  semantic: PassageAnchor | undefined,
): boolean {
  if (!bm25 || !semantic) return false;
  const { charStart, charEnd } = semantic;
  if (charStart === null || charEnd === null) return false;
  const start = bm25.matchStart ?? bm25.charStart;
  const end = bm25.matchEnd ?? bm25.charEnd;
  if (start == null || end == null) return false;
  return start >= charStart && end <= charEnd;
}
