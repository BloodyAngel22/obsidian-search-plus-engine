import assert from 'node:assert/strict';
import { describe, it } from 'vitest';
import {
  choosePrimaryPassage,
  containsExactQuery,
  isIdentifierLike,
  type Passage,
} from '../src/passage-selection.js';

const semantic = (text: string): Passage => ({ kind: 'semantic', snippet: text, chunkText: text });
const bm25 = (text: string, bodyMatch = true): Passage => ({
  kind: 'bm25',
  snippet: text,
  chunkText: text,
  bodyMatch,
});

describe('isIdentifierLike', () => {
  it('recognises technical identifiers', () => {
    for (const q of [
      'MergeTree',
      'ReplacingMergeTree',
      'trace_id',
      'DefaultStreamMaxLen',
      'FEED_DOMAIN_ADULT_EXCLUDED',
      'CollectVideoDomainObservationJob',
      'HTTP',
      'utf8',
      'os.path',
    ]) {
      assert.equal(isIdentifierLike(q), true, q);
    }
  });

  it('does not treat ordinary words or sentences as identifiers', () => {
    for (const q of ['redis', 'Redis', 'stream', 'Why was HTTP Referer removed?', 'merge tree']) {
      assert.equal(isIdentifierLike(q), false, q);
    }
  });
});

describe('containsExactQuery', () => {
  it('needs the whole token, not a part of a longer one', () => {
    assert.equal(containsExactQuery('uses ReplacingMergeTree here', 'MergeTree'), false);
    assert.equal(containsExactQuery('uses MergeTree here', 'mergetree'), true);
    assert.equal(containsExactQuery('field `trace_id` is set', 'trace_id'), true);
    assert.equal(containsExactQuery('trace id', 'trace_id'), false);
  });

  it('matches multi-word phrases across punctuation and case', () => {
    assert.equal(containsExactQuery('The **HTTP** Referer header', 'HTTP Referer'), true);
    assert.equal(containsExactQuery('HTTP headers; the Referer', 'HTTP Referer'), false);
  });
});

describe('choosePrimaryPassage', () => {
  it('uses the only passage a result has, or none', () => {
    assert.equal(choosePrimaryPassage({ semantic: semantic('x') }, 'q'), 'semantic');
    assert.equal(choosePrimaryPassage({ bm25: bm25('x') }, 'q'), 'bm25');
    assert.equal(choosePrimaryPassage({}, 'q'), null);
    assert.equal(choosePrimaryPassage(undefined, 'q'), null);
  });

  it('prefers verified exact identifier evidence over a semantic passage without it', () => {
    const passages = {
      semantic: semantic('Stream length is capped by DefaultStreamMaxLen.'),
      bm25: bm25('Events go to tables using the MergeTree engine.'),
    };
    assert.equal(choosePrimaryPassage(passages, 'MergeTree'), 'bm25');
  });

  it('keeps semantic when the semantic passage has the identifier too', () => {
    const passages = {
      semantic: semantic('MergeTree merges parts in the background.'),
      bm25: bm25('Another MergeTree mention.'),
    };
    assert.equal(choosePrimaryPassage(passages, 'MergeTree'), 'semantic');
  });

  it('does not count partial token overlap or title-only hits as exact evidence', () => {
    assert.equal(
      choosePrimaryPassage(
        { semantic: semantic('About engines.'), bm25: bm25('Uses ReplacingMergeTree.') },
        'MergeTree',
      ),
      'semantic',
    );
    assert.equal(
      choosePrimaryPassage(
        { semantic: semantic('About engines.'), bm25: bm25('MergeTree', false) },
        'MergeTree',
      ),
      'semantic',
    );
  });

  it('does not blindly prefer BM25 for a single ordinary word', () => {
    const passages = {
      semantic: semantic('Consumers read events in order.'),
      bm25: bm25('The redis client library.'),
    };
    assert.equal(choosePrimaryPassage(passages, 'redis'), 'semantic');
  });

  it('keeps the semantic passage for a natural-language question with scattered words', () => {
    const passages = {
      semantic: semantic('We dropped the header to stop leaking internal URLs.'),
      bm25: bm25('HTTP clients send a Referer by default.'),
    };
    assert.equal(choosePrimaryPassage(passages, 'Why was HTTP Referer removed?'), 'semantic');
  });

  it('accepts an exact multi-word phrase', () => {
    const passages = {
      semantic: semantic('Privacy notes.'),
      bm25: bm25('We removed the HTTP Referer header in v2.'),
    };
    assert.equal(choosePrimaryPassage(passages, 'HTTP Referer'), 'bm25');
  });

  it('shows the BM25 window when the hit lies inside the semantic chunk (one-chunk note)', () => {
    const whole = 'Intro text.\n\nEvents use the MergeTree engine.';
    const passages = {
      semantic: {
        ...semantic(whole),
        anchor: {
          kind: 'semantic' as const,
          headingPath: null,
          matchText: 'Intro text.',
          charStart: 0,
          charEnd: whole.length,
        },
      },
      bm25: {
        ...bm25('...Events use the MergeTree engine.'),
        chunkText: whole,
        anchor: {
          kind: 'bm25' as const,
          headingPath: null,
          matchText: 'Events use the MergeTree engine.',
          charStart: 0,
          charEnd: whole.length,
          matchStart: whole.indexOf('MergeTree'),
          matchEnd: whole.indexOf('MergeTree') + 9,
        },
      },
    };
    assert.equal(choosePrimaryPassage(passages, 'MergeTree'), 'bm25');
    // An ordinary word keeps the semantic passage even in the same chunk.
    assert.equal(choosePrimaryPassage(passages, 'events'), 'semantic');
  });
});
