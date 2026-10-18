/**
 * Markdown blocks inside a retrieved chunk, and the choice of the block a result shows.
 *
 * A chunk is a whole section or a sliding window, so it often holds several paragraphs
 * and may even start in the middle of a word. Document ranking never looks at this
 * module; it only narrows an already chosen passage to the block that answers the
 * query. Every block keeps its offsets in the note body (the text after frontmatter,
 * the same coordinates as chunk `char_start`/`char_end`), so the snippet, heading and
 * navigation target can all be cut from one range.
 */

import { containsExactQuery, isIdentifierLike } from './passage-selection.js';

export type BlockKind = 'paragraph' | 'list' | 'code' | 'table' | 'quote' | 'heading';

export interface PassageBlock {
  kind: BlockKind;
  /** Body offsets: `content.slice(start, end)` is the block text. */
  start: number;
  end: number;
  text: string;
}

/** Bounds of a body range that start and end on whole lines, outside half code fences. */
function widenToLines(content: string, from: number, to: number): [number, number] {
  let start = Math.max(0, Math.min(from, content.length));
  let end = Math.max(start, Math.min(to, content.length));
  start = content.lastIndexOf('\n', start - 1) + 1;
  if (end > 0 && end < content.length && content[end - 1] !== '\n') {
    const lineEnd = content.indexOf('\n', end);
    end = lineEnd === -1 ? content.length : lineEnd;
  }
  // A chunk that starts inside a fenced code block: begin at the fence instead.
  const fenceStart = openFenceBefore(content, start);
  if (fenceStart !== null) start = fenceStart;
  return [start, end];
}

const FENCE = /^\s{0,3}(`{3,}|~{3,})/;

/** Start of the fence line when `offset` lies inside a fenced code block, else null. */
function openFenceBefore(content: string, offset: number): number | null {
  if (offset === 0 || (!content.includes('```') && !content.includes('~~~'))) return null;
  let open: number | null = null;
  let pos = 0;
  while (pos < offset) {
    const lineEnd = content.indexOf('\n', pos);
    const stop = lineEnd === -1 ? content.length : lineEnd;
    if (FENCE.test(content.slice(pos, stop))) open = open === null ? pos : null;
    if (lineEnd === -1) break;
    pos = lineEnd + 1;
  }
  return open;
}

const LIST_ITEM = /^(\s*)([-*+]|\d{1,9}[.)])\s+/;

function indentWidth(line: string): number {
  let width = 0;
  for (const ch of line) {
    if (ch === ' ') width++;
    else if (ch === '\t') width += 4;
    else break;
  }
  return width;
}

/** End offset of a fenced code block opened at `next`'s previous line (closing fence line end). */
function fenceEnd(content: string, next: number, marker: string): number {
  const closing = marker[0]!.repeat(marker.length);
  let scan = next;
  while (scan < content.length) {
    const scanEnd = content.indexOf('\n', scan);
    const scanStop = scanEnd === -1 ? content.length : scanEnd;
    const candidate = content.slice(scan, scanStop).trim();
    if (candidate.startsWith(closing) && /^(`+|~+)$/.test(candidate)) return scanStop;
    if (scanEnd === -1) break;
    scan = scanEnd + 1;
  }
  return content.length;
}

function lineKind(line: string): BlockKind {
  if (/^\s*\|/.test(line)) return 'table';
  if (/^\s*>/.test(line)) return 'quote';
  if (LIST_ITEM.test(line)) return 'list';
  return 'paragraph';
}

/** True when a line of `kind` continues the open block of `active` kind. */
function continuesBlock(active: BlockKind, kind: BlockKind, indent: number): boolean {
  if (kind === 'list') return active === 'list' && indent >= 2;
  if (kind === active) return true;
  // Lazy continuation lines of list items and quotes.
  return kind === 'paragraph' && (active === 'list' || active === 'quote');
}

interface OpenBlock {
  kind: BlockKind;
  start: number;
  end: number;
}

/**
 * Split `content[from, to)` into Markdown blocks. The range is first widened to whole
 * lines (and to the opening fence of a code block it starts in), so a chunk that begins
 * mid-word still yields complete blocks. Fenced code is never split, even when it runs
 * past `to`. Blank lines separate blocks; a top-level list item with its nested items
 * and continuation lines is one block; a table and a quote are one block each.
 */
export function splitMarkdownBlocks(content: string, from: number, to: number): PassageBlock[] {
  const [start, end] = widenToLines(content, from, to);
  const blocks: PassageBlock[] = [];
  let current: OpenBlock | null = null;
  // Blank lines since the last block closed: an indented line after them may reopen a
  // list item (nested items separated by empty lines).
  let afterBlank = false;

  const push = (block: OpenBlock) => {
    blocks.push({ ...block, text: content.slice(block.start, block.end) });
  };
  const flush = () => {
    if (current) push(current);
    current = null;
    afterBlank = false;
  };

  let pos = start;
  while (pos < end) {
    const lineEnd = content.indexOf('\n', pos);
    const stop = lineEnd === -1 ? content.length : lineEnd;
    const next = lineEnd === -1 ? content.length : lineEnd + 1;
    const line = content.slice(pos, stop);

    const fence = FENCE.exec(line);
    if (fence) {
      flush();
      const close = fenceEnd(content, next, fence[1]!);
      push({ kind: 'code', start: pos, end: close });
      pos = close + 1;
      continue;
    }
    if (line.trim() === '') {
      if (current) flush();
      afterBlank = true;
    } else if (/^\s{0,3}#{1,6}\s/.test(line)) {
      flush();
      push({ kind: 'heading', start: pos, end: stop });
    } else {
      const kind = lineKind(line);
      const indent = indentWidth(line);
      const last = blocks[blocks.length - 1];
      if (current && continuesBlock(current.kind, kind, indent)) {
        current.end = stop;
      } else if (
        !current &&
        afterBlank &&
        indent >= 2 &&
        kind !== 'table' &&
        kind !== 'quote' &&
        last?.kind === 'list'
      ) {
        blocks.pop();
        current = { kind: 'list', start: last.start, end: stop };
      } else {
        flush();
        current = { kind, start: pos, end: stop };
      }
    }
    pos = next;
  }
  flush();
  return blocks;
}

// ─── Block scoring ─────────────────────────────────────────────────────────────

/** Frequent function words that carry no topic (Russian and English). */
const STOPWORDS = new Set([
  'и',
  'в',
  'во',
  'на',
  'с',
  'со',
  'к',
  'ко',
  'по',
  'о',
  'об',
  'от',
  'до',
  'из',
  'за',
  'для',
  'не',
  'ни',
  'но',
  'а',
  'же',
  'ли',
  'бы',
  'что',
  'чтобы',
  'как',
  'так',
  'это',
  'этот',
  'эта',
  'то',
  'тот',
  'та',
  'те',
  'где',
  'когда',
  'почему',
  'зачем',
  'какой',
  'какая',
  'какие',
  'кто',
  'при',
  'про',
  'или',
  'его',
  'её',
  'их',
  'мы',
  'вы',
  'он',
  'она',
  'они',
  'я',
  'ты',
  'есть',
  'the',
  'a',
  'an',
  'of',
  'to',
  'in',
  'on',
  'for',
  'and',
  'or',
  'is',
  'are',
  'was',
  'were',
  'be',
  'how',
  'what',
  'why',
  'when',
  'where',
  'which',
  'who',
  'does',
  'do',
  'did',
  'with',
  'by',
  'at',
  'from',
  'it',
  'this',
  'that',
  'as',
  'can',
  'i',
]);

function wordTokens(text: string): string[] {
  return text.toLowerCase().match(/[\p{L}\p{N}_]+/gu) ?? [];
}

/** Lowercased query terms without stop words; identifiers are kept whole. */
export function queryTerms(query: string): string[] {
  const terms = query
    .split(/\s+/)
    .flatMap((word) => (isIdentifierLike(word) ? [word.toLowerCase()] : wordTokens(word)))
    .filter((term) => term.length >= 2 && !STOPWORDS.has(term));
  return [...new Set(terms)];
}

/**
 * A light prefix stem for inflected words ("объединяет" → "объедин"), so a term matches
 * other forms of the same word. Identifiers (snake_case, CamelCase, digits) only match
 * whole, so MergeTree never matches inside ReplacingMergeTree.
 */
function termMatcher(term: string): (token: string) => boolean {
  if (/[_\d]/.test(term) || term.length <= 4) return (token) => token === term;
  const stem = term.slice(0, Math.max(4, Math.ceil(term.length * 0.7)));
  return (token) => token.startsWith(stem);
}

/**
 * Weighted share of query terms found in each block, 0..1. A term is weighted by how
 * rare it is among the blocks, so a word every block shares decides nothing.
 */
export function lexicalBlockScores(blocks: readonly PassageBlock[], query: string): number[] {
  const terms = queryTerms(query);
  if (terms.length === 0 || blocks.length === 0) return blocks.map(() => 0);
  const matchers = terms.map((term) => {
    // Identifier terms are compared on the original spelling, split like FTS does.
    return termMatcher(term);
  });
  const blockTokens = blocks.map((block) => new Set(wordTokens(block.text)));
  const hits = blockTokens.map((tokens) =>
    matchers.map((match, i) => {
      const term = terms[i]!;
      if (term.includes('_'))
        return containsExactQuery([...tokens].join(' '), term) || tokens.has(term);
      for (const token of tokens) if (match(token)) return true;
      return false;
    }),
  );
  const weights = terms.map((_term, i) => {
    const df = hits.filter((row) => row[i]).length;
    return df === 0 ? 0 : Math.log(1 + blocks.length / df);
  });
  const total = weights.reduce((sum, w) => sum + w, 0);
  if (total === 0) return blocks.map(() => 0);
  return hits.map((row) => row.reduce((sum, hit, i) => sum + (hit ? weights[i]! : 0), 0) / total);
}

/** True for blocks worth showing on their own (not headings, rules or bare markup). */
export function isCandidateBlock(block: PassageBlock): boolean {
  if (block.kind === 'heading') return false;
  const letters = block.text.match(/[\p{L}\p{N}]/gu)?.length ?? 0;
  return letters >= 3;
}

export interface BlockChoiceInput {
  blocks: readonly PassageBlock[];
  query: string;
  /** Body offset of a verified exact hit (BM25); its block wins outright. */
  hitOffset?: number;
  /** Cosine similarity of each block to the query, when block embeddings were made. */
  semanticScores?: ReadonlyArray<number | null>;
}

/** How much lexical coverage can shift a semantic decision (cosine units). */
const LEXICAL_WEIGHT = 0.02;

/**
 * Index of the block that best answers the query, or -1 when no block qualifies.
 *
 * 1. A verified exact hit picks its own block.
 * 2. A literal query (identifier or phrase) prefers blocks that contain it whole.
 * 3. Among the remaining candidates, block embeddings decide (lexical coverage only
 *    breaks near-ties); without embeddings, lexical coverage decides. Word overlap
 *    alone cannot rank paraphrased questions reliably, so it is only the fallback.
 * Ties go to the earlier block.
 */
export function chooseBlock(input: BlockChoiceInput): number {
  const { blocks, query, hitOffset, semanticScores } = input;
  if (hitOffset !== undefined) {
    const index = blocks.findIndex((b) => hitOffset >= b.start && hitOffset < b.end);
    if (index !== -1) return index;
  }
  const candidates = candidateBlocks(blocks, query);
  if (candidates.length === 0) return -1;
  if (candidates.length === 1) return candidates[0]!;
  const lexical = lexicalBlockScores(blocks, query);
  const useSemantic = semanticScores && candidates.some((i) => semanticScores[i] != null);
  let best = candidates[0]!;
  let bestScore = -Infinity;
  for (const i of candidates) {
    const score = useSemantic
      ? (semanticScores[i] ?? -1) + LEXICAL_WEIGHT * lexical[i]!
      : lexical[i]!;
    if (score > bestScore + 1e-9) {
      best = i;
      bestScore = score;
    }
  }
  return best;
}

/**
 * Indexes of the blocks that compete for a query: showable blocks, narrowed to those
 * holding the whole query when it is literal (an identifier or a phrase) and any does.
 */
export function candidateBlocks(blocks: readonly PassageBlock[], query: string): number[] {
  const candidates = blocks.map((_b, i) => i).filter((i) => isCandidateBlock(blocks[i]!));
  if (!isLiteralQuery(query)) return candidates;
  const exact = candidates.filter((i) => containsExactQuery(blocks[i]!.text, query));
  return exact.length > 0 ? exact : candidates;
}

export function isLiteralQuery(query: string): boolean {
  return isIdentifierLike(query) || wordTokens(query).length >= 2;
}

/** Below this length a lone block lacks context; the next block is added. */
const MIN_PASSAGE_CHARS = 60;

/**
 * The passage for a chosen block: the block itself, plus the following block when the
 * chosen one cannot stand alone — a short lead-in that ends with ":" (it introduces
 * what follows) or a fragment under MIN_PASSAGE_CHARS. The passage always STARTS at
 * the chosen block, so the snippet and the navigation target begin at the answer.
 */
export function passageRange(
  blocks: readonly PassageBlock[],
  index: number,
): { start: number; end: number } {
  const block = blocks[index]!;
  let end = block.end;
  const plain = block.text.replace(/[*_`]+/g, '').trim();
  const needsContext =
    block.kind !== 'code' && (plain.endsWith(':') || plain.length < MIN_PASSAGE_CHARS);
  const next = blocks[index + 1];
  if (needsContext && next && next.kind !== 'heading') end = next.end;
  return { start: block.start, end };
}

/** Text used to embed a block: Markdown noise trimmed, length capped. */
export function blockEmbeddingText(text: string): string {
  return (
    text
      .split('\n')
      .map((line) => {
        const marker = LIST_ITEM.exec(line);
        return marker ? line.slice(marker[0].length) : line;
      })
      .join('\n')
      // Underscores stay: they belong to identifiers such as trace_id.
      .replace(/[`*]{1,3}/g, '')
      .replace(/\s+/g, ' ')
      .trim()
      .slice(0, 600)
  );
}

function escapeRegExp(text: string): string {
  return text.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

/**
 * First occurrence of the whole query in `content[start, end)`, in source offsets, so
 * the original spelling is kept. An identifier must appear literally (case-insensitive)
 * and as a whole token: `MergeTree` does not match inside `ReplacingMergeTree`, and
 * `trace_id` does not match `trace id`. A phrase may be separated by any spacing or
 * punctuation. Returns null when the query does not occur.
 */
export function findExactOccurrence(
  content: string,
  start: number,
  end: number,
  query: string,
): { start: number; end: number } | null {
  const trimmed = query.trim();
  if (!trimmed) return null;
  let pattern: string;
  if (isIdentifierLike(trimmed)) {
    pattern = escapeRegExp(trimmed);
  } else {
    const tokens = trimmed.match(/[\p{L}\p{N}_]+/gu);
    if (!tokens) return null;
    pattern = tokens.map(escapeRegExp).join('[^\\p{L}\\p{N}_]+');
  }
  const re = new RegExp(`(?<![\\p{L}\\p{N}_])${pattern}(?![\\p{L}\\p{N}_])`, 'iu');
  const match = re.exec(content.slice(start, end));
  if (!match) return null;
  return { start: start + match.index, end: start + match.index + match[0].length };
}

/**
 * First word in `content[start, end)` that FTS itself would match for a query term
 * (the term as a prefix; identifiers only whole), in source offsets. Used to mark
 * literal evidence when a full-text passage moved away from the first FTS hit.
 */
export function findTermOccurrence(
  content: string,
  start: number,
  end: number,
  query: string,
): { start: number; end: number } | null {
  const matchers = queryTerms(query).map(
    (term) => (token: string) => (/[_\d]/.test(term) ? token === term : token.startsWith(term)),
  );
  if (matchers.length === 0) return null;
  const re = /[\p{L}\p{N}_]+/gu;
  const text = content.slice(start, end);
  for (let m = re.exec(text); m; m = re.exec(text)) {
    const token = m[0].toLowerCase();
    if (matchers.some((match) => match(token))) {
      return { start: start + m.index, end: start + m.index + m[0].length };
    }
  }
  return null;
}
