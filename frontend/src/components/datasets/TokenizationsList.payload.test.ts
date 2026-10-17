/**
 * The tokenize payload must actually carry the controls the form renders.
 *
 * WHY THIS FILE EXISTS. `text_column`, `chat_format` and `pack_sequences` were
 * added to the backend schema in the SAE data-path arc and sent by NOTHING —
 * `handleCreate` hardcoded its payload, so an operator clicking Tokenize got
 * the old defaults no matter what. The column override is the specific control
 * that would have prevented Bloomberg being tokenized on `Headline` (17 tokens)
 * while `Article` (464) sat unused, and it was unreachable from the UI.
 *
 * That is round 4's H2 defect repeating: a control made VISIBLE whose value
 * never reaches the request — which is worse than its honest absence, because
 * the UI now claims a capability it does not deliver.
 *
 * So these assert the PAYLOAD, from a pure builder, rather than that an input
 * rendered. A rendered input proves nothing.
 */

import { describe, it, expect } from 'vitest';
import {
  buildTokenizationPayload,
  type TokenizationFormValues,
} from './TokenizationsList';

const defaults: TokenizationFormValues = {
  maxLength: 512,
  textColumn: '',
  chatFormat: 'auto',
  packSequences: false,
  enableCleaning: false,
  filterEnabled: false,
  filterMode: 'conservative',
  junkRatioThreshold: 0.7,
  removeAllPunctuation: false,
  customFilterChars: '',
  shuffle: true,
  shuffleSeed: '',
};

const withValues = (overrides: Partial<TokenizationFormValues> = {}) =>
  buildTokenizationPayload({ ...defaults, ...overrides });

describe('buildTokenizationPayload', () => {
  describe('text_column — the Bloomberg control', () => {
    it('sends the column the operator chose', () => {
      expect(withValues({ textColumn: 'Article' })).toMatchObject({
        text_column: 'Article',
      });
    });

    it('OMITS the key when blank, so blank still means auto-detect', () => {
      const payload = withValues({ textColumn: '' });
      expect('text_column' in payload).toBe(false);
    });

    it('treats whitespace as blank rather than sending " " as a column name', () => {
      const payload = withValues({ textColumn: '   ' });
      expect('text_column' in payload).toBe(false);
    });

    it('trims a pasted column name', () => {
      expect(withValues({ textColumn: ' Article ' })).toMatchObject({
        text_column: 'Article',
      });
    });
  });

  describe('chat_format', () => {
    it.each(['auto', 'chat_template', 'plain', 'none'] as const)(
      'transmits %s',
      (format) => {
        expect(withValues({ chatFormat: format })).toMatchObject({
          chat_format: format,
        });
      }
    );

    it('is always present, because the backend column is nullable and an absent value means "unrecorded"', () => {
      expect(withValues()).toHaveProperty('chat_format');
    });
  });

  describe('pack_sequences', () => {
    it('transmits true', () => {
      expect(withValues({ packSequences: true })).toMatchObject({
        pack_sequences: true,
      });
    });

    it('transmits false explicitly rather than omitting it', () => {
      const payload = withValues({ packSequences: false });
      expect(payload.pack_sequences).toBe(false);
      expect('pack_sequences' in payload).toBe(true);
    });
  });

  describe('shuffle — on unless the operator opts out', () => {
    it('sends true by default, because extraction reads a PREFIX of the blocks', () => {
      // Without a shuffle, every extraction sees the same opening slice of the
      // corpus — 6.5% of Bloomberg, which is dated financial news.
      expect(withValues()).toMatchObject({ shuffle: true });
    });

    it('transmits false explicitly rather than omitting it', () => {
      // Omitting falls back to the backend default of true, so a deliberate
      // opt-out would silently shuffle anyway and the row would say so.
      const payload = withValues({ shuffle: false });
      expect(payload.shuffle).toBe(false);
      expect('shuffle' in payload).toBe(true);
    });

    it('OMITS shuffle_seed when blank, so blank means "derive one and record it"', () => {
      expect('shuffle_seed' in withValues({ shuffleSeed: '' })).toBe(false);
    });

    it('treats whitespace as blank rather than sending NaN', () => {
      expect('shuffle_seed' in withValues({ shuffleSeed: '   ' })).toBe(false);
    });

    it('sends a pinned seed', () => {
      expect(withValues({ shuffleSeed: '4242' })).toMatchObject({ shuffle_seed: 4242 });
    });

    it('sends 0, which is a legitimate seed and not an empty field', () => {
      // `|| undefined` here would discard it and the run would derive a
      // different seed than the operator asked for.
      const payload = withValues({ shuffleSeed: '0' });
      expect(payload.shuffle_seed).toBe(0);
      expect('shuffle_seed' in payload).toBe(true);
    });

    it('sends a number, not the input element’s string', () => {
      expect(typeof withValues({ shuffleSeed: '7' }).shuffle_seed).toBe('number');
    });
  });

  describe('enable_cleaning — off unless the operator asks', () => {
    it('sends false by default, because cleaning rewrites the corpus', () => {
      expect(withValues()).toMatchObject({ enable_cleaning: false });
    });

    it('sends true when the operator turns it on', () => {
      expect(withValues({ enableCleaning: true })).toMatchObject({ enable_cleaning: true });
    });
  });

  describe('the fields that already worked must not regress', () => {
    it('carries max_length and the filter block', () => {
      expect(
        withValues({
          maxLength: 2048,
          filterEnabled: true,
          filterMode: 'strict',
          junkRatioThreshold: 0.4,
          removeAllPunctuation: true,
          customFilterChars: '~@#',
        })
      ).toMatchObject({
        max_length: 2048,
        tokenization_filter_enabled: true,
        tokenization_filter_mode: 'strict',
        tokenization_junk_ratio_threshold: 0.4,
        remove_all_punctuation: true,
        custom_filter_chars: '~@#',
      });
    });
  });

  describe('the Bloomberg configuration end to end', () => {
    it('produces exactly the request that fixes the corpus', () => {
      expect(
        withValues({
          maxLength: 2048,
          textColumn: 'Article',
          chatFormat: 'none',
          packSequences: true,
        })
      ).toMatchObject({
        max_length: 2048,
        text_column: 'Article',
        chat_format: 'none',
        pack_sequences: true,
      });
    });

    it('NEGATIVE CONTROL — the defaults are not that request', () => {
      const payload = withValues();
      expect('text_column' in payload).toBe(false);
      expect(payload.pack_sequences).toBe(false);
      expect(payload.max_length).toBe(512);
    });
  });
});
