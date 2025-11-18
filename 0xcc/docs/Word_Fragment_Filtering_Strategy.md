# Word Fragment Filtering Strategy

## Overview
This document describes the approach for filtering BPE (Byte Pair Encoding) word fragments from token analysis in miStudio's feature discovery system.

## Problem Statement
Modern tokenizers (GPT-2, LLaMA, etc.) use BPE tokenization, which creates subword units. When analyzing feature activations, these fragments add noise:
- **Verb suffixes**: "ing", "ed", "ated"
- **Noun suffixes**: "tion", "ness", "ment"
- **Adjective suffixes**: "ous", "ive", "ful", "less"
- **Partial words**: "ber", "ial", "ian", "ine"
- **Geographic fragments**: "stan", "burg", "istan"

These fragments rarely carry standalone semantic meaning and obscure the actual content words.

## Solution Approach

### 1. Comprehensive Fragment Database
Created `WORD_FRAGMENTS` set with **379 fragments** across categories:

**Morphological Fragments:**
- Verb suffixes: ing, ed, en, es, ated, ating, ized, izing, ified, ifying
- Noun suffixes: tion, sion, ation, ment, ness, ity, ance, ence, er, or, ist, ian
- Adjective suffixes: able, ible, ful, less, ous, ious, ive, ative, al, ial, ical, ish
- Adverb suffixes: ly, ally, ily, ely
- Verb-forming: ize, ise, ify, fy, ate
- Negation prefixes: un, im, il, ir, dis, non, mis
- Position prefixes: pre, post, fore, ante, retro, inter, intra, trans
- Quantity prefixes: uni, mono, bi, tri, multi, poly, semi
- Quality prefixes: anti, contra, counter, auto, syn, meta, para, hyper, hypo

**BPE-Common Fragments:**
- Consonant clusters: ch, sh, th, wh, ph, gh, ck, ng, qu, st, sp, str, spr, thr
- Common endings: le, el, al, er, ar, or, se, ce, ge, te, de, ne
- Middle fragments: ough, augh, eigh, ight, ould, cial, tial
- Partial words: ber, ter, ver, fer, ual, ial, ian, ine, ana, ary, ery, ory

**Geographic Fragments:**
- Place suffixes: burg, burgh, berg, ville, town, port, ford, field, land, stan, istan, shire
- Region fragments: ulf, asi, afr, eur, amer, austr, arab, turk, chin, palest, isra

**Syllable Fragments:**
- Common 2-letter combinations that aren't standalone words: ab, ad, ag, ap, av, eb, ec, eg, ep, ev, ib, id, ig, ip, iv, ob, od, og, ot, ov, ub, ud, ug, ut, uv

### 2. Conservative Exclusions
**Avoided false positives by excluding valid standalone words:**
- Common prepositions: in, on, to, by, of, at, up
- Pronouns: he, we, me, us, it
- Verbs: is, be, do, go
- Determiners: the, a, an
- Conjunctions: or, if, so, no, as

### 3. Additional Filter Logic
Beyond the explicit fragment set, the filter also catches:
- **Short fragments without vowels** (≤3 chars): "th", "str", "ps", "by"
- But **allows short words with vowels**: "he", "in", "on", "are"

## Implementation

### Backend: `backend/src/utils/token_filters.py`

```python
WORD_FRAGMENTS: Set[str] = {
    # 379 fragments organized by category
    # (see code for full list)
}

def is_junk_token(
    token: str,
    filter_fragments: bool = True,
    # ... other filter flags
) -> bool:
    """Filter tokens based on categories."""
    cleaned = token.replace('▁', '').strip().lower()

    if filter_fragments:
        # Check explicit fragment set
        if cleaned in WORD_FRAGMENTS:
            return True
        # Check short fragments without vowels
        if len(cleaned) <= 3 and not re.search(r'[aeiou]', cleaned):
            return True

    return False
```

### Sort Order
Results are sorted by:
1. **COUNT descending** (most frequent first)
2. **TOKEN ascending** (alphabetical within same count)

```python
sorted_tokens = sorted(
    filtered_tokens.items(),
    key=lambda x: (-x[1], x[0])  # (-count, token)
)
```

## Effectiveness

### Test Results (Feature feat_train_aa_00964)

**Without fragment filter:**
- Unique tokens: 235
- Total occurrences: 373
- Top tokens include fragments: "ated", "ie", "ing", "al", "ber", "ed"

**With fragment filter:**
- Unique tokens: 201 (34 fewer)
- Total occurrences: 314 (59 fewer)
- **34 fragments removed**: ated, ie, ing, al, ber, ed, ial, ian, ies, ine, or, ulf, ana, Palest, Arab, stan, town, etc.
- Top tokens are meaningful: "example", "police", "program", "City", "Foreign", "Grey", "High"

**Improvement:**
- **14.5% reduction** in unique tokens
- **15.8% reduction** in total occurrences
- **Cleaner semantic signal** - remaining tokens are content words

## Usage

### API Parameters
All 6 filter categories are independently toggleable:

```bash
GET /api/v1/features/{feature_id}/token-analysis?
  filter_fragments=true      # NEW: Filter word fragments (default: true)
  filter_stop_words=false    # Filter common stop words (default: false)
  filter_special=true        # Filter <s>, </s>, etc.
  filter_single_char=true    # Filter single characters
  filter_punctuation=true    # Filter pure punctuation
  filter_numbers=true        # Filter pure numbers
```

### Frontend UI
Filter Options panel with 6 checkboxes:
- ☑️ Special tokens (`<s>`, `</s>`)
- ☑️ Single characters
- ☑️ Punctuation
- ☑️ Numbers
- ☑️ **Word fragments** (tion, ing, ed)
- ☐ Stop words (the, and, is)

## Future Enhancements

### Potential Additions
1. **Language-specific fragments**: Add fragments for other languages (Spanish, French, etc.)
2. **Domain-specific fragments**: Medical, legal, technical terminology fragments
3. **User-customizable fragments**: Allow users to add/remove fragments
4. **Statistical learning**: Automatically identify fragments from corpus

### Balancing Precision vs. Recall
Current approach favors **precision** (few false positives):
- Conservative with short tokens (excluded valid 2-letter words)
- Explicit fragment set rather than aggressive pattern matching
- Independent toggles allow user control

Could increase **recall** (catch more fragments):
- Add more aggressive pattern matching
- Include borderline cases (currently excluded to avoid false positives)
- Machine learning approach to identify corpus-specific fragments

## References
- **GPT-2 Tokenizer**: https://github.com/openai/gpt-2/blob/master/src/encoder.py
- **SentencePiece**: https://github.com/google/sentencepiece
- **BPE Paper**: "Neural Machine Translation of Rare Words with Subword Units" (Sennrich et al., 2016)
- **Token Classification**: https://arxiv.org/abs/1907.11692

## Changelog
- **2025-11-17**: Expanded WORD_FRAGMENTS from ~50 to 379 fragments
- **2025-11-17**: Added comprehensive category organization
- **2025-11-17**: Implemented conservative exclusion of valid standalone words
- **2025-11-16**: Initial implementation with basic fragment filtering
