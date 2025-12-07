"""
Token filtering utilities for feature analysis.

This module provides utilities for filtering and cleaning tokens extracted
from feature activation examples. It removes junk tokens (special markers,
punctuation, single characters, etc.) to focus on meaningful content.

Supports toggleable filter categories for granular control.
"""

import re
from typing import List, Dict, Tuple, Set
from collections import Counter


# Common English stop words (helping words)
STOP_WORDS: Set[str] = {
    'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
    'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
    'to', 'was', 'will', 'with', 'this', 'but', 'they', 'have', 'had',
    'what', 'when', 'where', 'who', 'which', 'why', 'how', 'not', 'or',
    'if', 'can', 'do', 'so', 'up', 'out', 'than', 'then', 'now', 'only',
    'just', 'any', 'all', 'some', 'no', 'also', 'there', 'their', 'them',
    'these', 'those', 'been', 'being', 'were', 'am', 'could',
    'would', 'should', 'may', 'might', 'must', 'shall', 'did', 'does',
    'my', 'your', 'his', 'her', 'our', 'me', 'you', 'him', 'us',
    'i', 'we', 'she', 'one', 'two', 'three'
}

# Common BPE word fragments (subword units that are rarely standalone)
# Based on analysis of GPT-2, LLaMA, and other common BPE tokenizers
WORD_FRAGMENTS: Set[str] = {
    # === VERB SUFFIXES ===
    'ing', 'ed', 'en', 'es', 's', 'd', 'n', 't', 'm', 'll', 've',
    'ated', 'ating', 'ized', 'izing', 'ised', 'ising',
    'ified', 'ifying', 'ened', 'ening',

    # === NOUN SUFFIXES ===
    'tion', 'sion', 'ation', 'ition', 'ution', 'ction', 'ption',
    'ment', 'ness', 'ity', 'ty', 'dom', 'hood', 'ship',
    'ance', 'ence', 'ancy', 'ency', 'ency',
    'er', 'or', 'ar', 'ist', 'ian', 'ant', 'ent', 'ee', 'eer',
    'ism', 'ster', 'ling', 'let',

    # === ADJECTIVE SUFFIXES ===
    'able', 'ible', 'ful', 'less', 'ous', 'ious', 'eous', 'uous',
    'ive', 'ative', 'itive', 'al', 'ial', 'ical', 'ual',
    'ic', 'tic', 'atic', 'etic', 'itic',
    'ish', 'like', 'y', 'ry', 'some',
    'er', 'est', 'ier', 'iest',

    # === ADVERB SUFFIXES ===
    'ly', 'ally', 'ily', 'ely', 'ully',
    'ward', 'wards', 'wise',

    # === VERB-FORMING SUFFIXES ===
    'ize', 'ise', 'ify', 'fy', 'ate', 'en',

    # === ADJECTIVE-FORMING SUFFIXES ===
    'ory', 'ary', 'ery', 'ory', 'ure',

    # === NEGATION PREFIXES ===
    'un', 'im', 'il', 'ir', 'dis', 'non', 'mis',

    # === POSITION/TIME PREFIXES ===
    'pre', 'post', 'fore', 'ante', 'retro',
    'inter', 'intra', 'trans', 'circum',

    # === QUANTITY PREFIXES ===
    'uni', 'mono', 'bi', 'di', 'tri', 'quad', 'multi', 'poly',
    'semi', 'hemi', 'demi',

    # === QUALITY/RELATION PREFIXES ===
    'anti', 'contra', 'counter', 'auto', 'syn',
    'meta', 'para', 'hyper', 'hypo',

    # === COMMON BPE FRAGMENTS (2-3 chars) ===
    # Consonant clusters
    'ch', 'sh', 'th', 'wh', 'ph', 'gh', 'ck', 'ng', 'qu',
    'st', 'sp', 'sc', 'sk', 'sm', 'sn', 'sw', 'sl', 'str', 'spr', 'scr',
    'tr', 'pr', 'br', 'cr', 'dr', 'fr', 'gr', 'thr', 'shr',
    'bl', 'cl', 'fl', 'gl', 'pl', 'sl',
    'tw', 'dw', 'sw',

    # Common endings
    'le', 'el', 'al', 'ol', 'ul', 'il',
    're', 'er', 'ar', 'or', 'ir', 'ur',
    'se', 'ce', 'ge', 'ze',
    'te', 'de', 'ne', 'me', 'ke', 'pe', 've',

    # === COMMON BPE MIDDLE FRAGMENTS ===
    'ough', 'augh', 'eigh', 'ight', 'aight',
    'ould', 'tion', 'sion', 'cial', 'tial',

    # === COMMON WORD BEGINNINGS (BPE fragments) ===
    'con', 'com', 'def', 'ref', 'inf', 'diff', 'eff',

    # === COMMON WORD ENDINGS (BPE fragments) ===
    'ies', 'ied', 'ier', 'iest',
    'ness', 'less', 'ful', 'ment',
    'tion', 'sion', 'ation', 'ition',
    'ous', 'ious', 'eous', 'uous',
    'ive', 'ative', 'itive',
    'ize', 'ise', 'ify', 'fy',
    'able', 'ible',
    'ance', 'ence', 'ancy', 'ency',
    'ity', 'ty', 'ety', 'iety',

    # === COMMON PARTIAL WORDS (seen in BPE tokenization) ===
    'ber', 'ter', 'ver', 'fer', 'ger', 'ker', 'mer', 'ner', 'per', 'ser', 'wer',
    'der', 'ler', 'rer', 'cer',
    'ual', 'ial', 'eal', 'eal', 'ial',
    'ual', 'uel', 'iel',
    'ian', 'ean', 'ean',
    'ane', 'ene', 'ine', 'one', 'une',
    'ana', 'ina', 'ona', 'una',
    'ary', 'ery', 'ory', 'ury',
    'ica', 'ica', 'ica',
    'ise', 'ize', 'yse', 'yze',
    'ula', 'ule', 'ulo',
    'ata', 'ate', 'ati', 'ato',
    'ite', 'iti', 'ito',
    'ota', 'ote', 'oti', 'oto',
    'uta', 'ute', 'uti', 'uto',

    # === COMMON GEOGRAPHIC/NAME FRAGMENTS ===
    'burg', 'burgh', 'berg', 'borg',
    'ville', 'town', 'port', 'ford', 'field',
    'land', 'stan', 'istan',
    'shire', 'chester', 'cester',

    # === COMMON SYLLABLE FRAGMENTS ===
    # (Excluding valid 2-letter words: is, it, in, if, to, be, he, we, me, so, do, go, no, or, as, at, by, of, on, up, us)
    'ab', 'ad', 'ag', 'ap', 'av',
    'eb', 'ec', 'eg', 'ep', 'ev',
    'ib', 'id', 'ig', 'ip', 'iv',
    'ob', 'od', 'og', 'ot', 'ov',
    'ub', 'ud', 'ug', 'ut', 'uv',

    # === ADDITIONAL COMMON BPE FRAGMENTS ===
    'ia', 'ie', 'io', 'iu',
    'ea', 'ee', 'eo', 'eu',
    'oa', 'oe', 'oi', 'oo', 'ou',
    'ua', 'ue', 'ui', 'uo',
    'ya', 'ye', 'yo', 'yu',
    'wa', 'we', 'wi', 'wo', 'wu',

    # === COMMON 2-LETTER ENDINGS ===
    'bs', 'cs', 'ds', 'fs', 'gs', 'hs', 'js', 'ks', 'ls', 'ms',
    'ns', 'ps', 'rs', 'ss', 'ts', 'vs', 'ws', 'xs', 'ys', 'zs',

    # === PARTIAL COUNTRY/REGION NAMES ===
    'ulf', 'ulf', 'asi', 'afr', 'eur', 'amer', 'austr',
    'arab', 'persian', 'turk', 'chin', 'jap', 'kore',
    'brit', 'franc', 'germ', 'ital', 'span', 'russ',
    'afgh', 'pak', 'ind', 'bang', 'thai', 'viet',
    'palest', 'isra', 'saud', 'iran', 'iraq', 'syri',

    # === COMMON TITLE/DESCRIPTOR FRAGMENTS ===
    'mini', 'maxi', 'mega', 'micro', 'macro',
    'neo', 'paleo', 'proto', 'pseudo',
    'semi', 'quasi', 'ultra',
}


def is_junk_token(
    token: str,
    filter_special: bool = True,
    filter_single_char: bool = True,
    filter_punctuation: bool = True,
    filter_numbers: bool = True,
    filter_fragments: bool = True,
    filter_stop_words: bool = False
) -> bool:
    """
    Determine if a token should be filtered as junk based on specified filter flags.

    Args:
        token: The token string to check
        filter_special: Filter special tokens (<s>, </s>, <pad>, <unk>, BOM)
        filter_single_char: Filter single character tokens
        filter_punctuation: Filter pure punctuation tokens
        filter_numbers: Filter pure numeric tokens
        filter_fragments: Filter word fragments (BPE subwords, short without vowels)
        filter_stop_words: Filter common stop words (a, the, and, is, etc.)

    Returns:
        True if token should be filtered, False otherwise

    Examples:
        >>> is_junk_token('<s>')
        True
        >>> is_junk_token('a', filter_stop_words=True)
        True
        >>> is_junk_token('The', filter_stop_words=True)
        True
        >>> is_junk_token('tion', filter_fragments=True)
        True
        >>> is_junk_token('news')
        False
    """
    # Clean token for comparison (lowercase, no space markers)
    cleaned = token.replace('▁', '').strip().lower()

    # Special tokens
    if filter_special:
        # Common special tokens across tokenizers
        special_tokens = {
            # LLaMA/SentencePiece
            '<s>', '</s>', '<pad>', '<unk>',
            # GPT-2/OpenAI
            '<|endoftext|>', '<|pad|>',
            # LLaMA 3 / newer models
            '<|begin_of_text|>', '<|end_of_text|>',
            '<|eot_id|>', '<|start_header_id|>', '<|end_header_id|>',
            '<|pad_id|>', '<|finetune_right_pad_id|>',
            # Gemma
            '<bos>', '<eos>', '<pad>',
            # Unicode BOM
            '\ufeff',
        }
        if token in special_tokens:
            return True
        # Tokens that look like special tokens (angle bracket pattern)
        if token.startswith('<|') and token.endswith('|>'):
            return True
        if token.startswith('<') and token.endswith('>') and len(token) > 2:
            # Likely a special token like <mask>, <sep>, etc.
            inner = token[1:-1]
            if inner.isalpha() or inner.replace('_', '').isalpha():
                return True
        # Only whitespace marker
        if token == '▁':
            return True

    # Single characters
    if filter_single_char and len(token) <= 1:
        return True

    # Just punctuation
    if filter_punctuation and re.match(r'^[^a-zA-Z0-9]+$', token):
        return True

    # Just numbers
    if filter_numbers and re.match(r'^[0-9]+$', token):
        return True

    # Word fragments
    if filter_fragments:
        # Check if token is a known BPE fragment
        if cleaned in WORD_FRAGMENTS:
            return True

        # Short fragments without vowels (likely BPE pieces)
        # But allow short words with vowels (like "he", "in", "on", "are")
        if len(cleaned) <= 3 and not re.search(r'[aeiou]', cleaned):
            return True

    # Stop words (helping words)
    if filter_stop_words and cleaned in STOP_WORDS:
        return True

    return False


def clean_token_display(token: str) -> str:
    """
    Clean up token for display by removing tokenizer artifacts.

    Removes leading space marker (▁) used by SentencePiece and similar
    tokenizers. If the token becomes empty after cleaning, returns the
    original token.

    Args:
        token: The token string to clean

    Returns:
        Cleaned token string suitable for display

    Examples:
        >>> clean_token_display('▁Hello')
        'Hello'
        >>> clean_token_display('▁')
        '▁'
        >>> clean_token_display('world')
        'world'
    """
    # Remove leading space marker
    cleaned = token.replace('▁', '').strip()

    # If empty after cleaning, return original
    if not cleaned:
        return token

    return cleaned


def analyze_feature_tokens(
    tokens_list: List[List[str]],
    apply_filters: bool = True,
    filter_special: bool = True,
    filter_single_char: bool = True,
    filter_punctuation: bool = True,
    filter_numbers: bool = True,
    filter_fragments: bool = True,
    filter_stop_words: bool = False
) -> Dict:
    """
    Analyze tokens from feature activation examples.

    Counts token occurrences, optionally filters junk tokens with granular
    control, and returns analysis results including summary statistics and
    sorted token list.

    Args:
        tokens_list: List of token arrays from feature activations
        apply_filters: Master switch for all filtering (default: True)
        filter_special: Filter special tokens (<s>, </s>, etc.)
        filter_single_char: Filter single character tokens
        filter_punctuation: Filter pure punctuation
        filter_numbers: Filter pure numeric tokens
        filter_fragments: Filter word fragments (BPE subwords)
        filter_stop_words: Filter common stop words (the, and, is, etc.)

    Returns:
        Dictionary containing:
        - summary: Statistics about the analysis (total_examples, token counts,
                   diversity_percent, filter counts per category)
        - tokens: List of dicts with rank, token, count, percentage

    Example:
        >>> tokens = [['<s>', 'The', 'dog'], ['The', 'cat', '<s>']]
        >>> result = analyze_feature_tokens(tokens)
        >>> result['summary']['original_token_count']
        4
        >>> result['summary']['filtered_token_count']
        2
    """
    # Count all tokens
    token_counter = Counter()
    for tokens in tokens_list:
        if tokens:
            for token in tokens:
                token_counter[token] += 1

    original_count = len(token_counter)
    total_occurrences = sum(token_counter.values())

    # Apply filters if requested
    if apply_filters:
        # Track filter statistics
        filter_stats = {
            'special': 0,
            'single_char': 0,
            'punctuation': 0,
            'numbers': 0,
            'fragments': 0,
            'stop_words': 0
        }

        filtered_tokens = {}
        for token, count in token_counter.items():
            # Check which filters apply to this token
            is_filtered = False

            # Track which filter caught this token (for stats)
            if filter_special and token in ['<s>', '</s>', '<pad>', '<unk>', '\ufeff', '▁']:
                filter_stats['special'] += 1
                is_filtered = True
            elif filter_single_char and len(token) <= 1:
                filter_stats['single_char'] += 1
                is_filtered = True
            elif filter_punctuation and re.match(r'^[^a-zA-Z0-9]+$', token):
                filter_stats['punctuation'] += 1
                is_filtered = True
            elif filter_numbers and re.match(r'^[0-9]+$', token):
                filter_stats['numbers'] += 1
                is_filtered = True
            else:
                cleaned = token.replace('▁', '').strip().lower()

                if filter_fragments:
                    if cleaned in WORD_FRAGMENTS or (len(cleaned) <= 3 and not re.search(r'[aeiou]', cleaned)):
                        filter_stats['fragments'] += 1
                        is_filtered = True

                if not is_filtered and filter_stop_words and cleaned in STOP_WORDS:
                    filter_stats['stop_words'] += 1
                    is_filtered = True

            if not is_filtered:
                filtered_tokens[token] = count

    else:
        filtered_tokens = dict(token_counter)
        filter_stats = {}

    filtered_count = len(filtered_tokens)
    junk_removed = original_count - filtered_count
    filtered_occurrences = sum(filtered_tokens.values())

    # Sort by count descending, then by token value
    sorted_tokens = sorted(
        filtered_tokens.items(),
        key=lambda x: (-x[1], x[0])
    )

    # Build result list with cleaned tokens
    tokens_result = []
    for rank, (token, count) in enumerate(sorted_tokens, 1):
        percentage = (count / filtered_occurrences * 100) if filtered_occurrences > 0 else 0.0
        tokens_result.append({
            'rank': rank,
            'token': clean_token_display(token),
            'count': count,
            'percentage': round(percentage, 2)
        })

    # Calculate diversity percentage
    diversity_percent = (
        (filtered_count / filtered_occurrences * 100)
        if filtered_occurrences > 0
        else 0.0
    )

    summary = {
        'total_examples': len(tokens_list),
        'original_token_count': original_count,
        'filtered_token_count': filtered_count,
        'junk_removed': junk_removed,
        'total_token_occurrences': total_occurrences,
        'filtered_token_occurrences': filtered_occurrences,
        'diversity_percent': round(diversity_percent, 2)
    }

    # Add filter statistics if filters were applied
    if apply_filters:
        summary['filter_stats'] = filter_stats

    return {
        'summary': summary,
        'tokens': tokens_result
    }


def filter_token_stats(
    token_stats: Dict[str, Dict],
    filter_special: bool = True,
    filter_single_char: bool = True,
    filter_punctuation: bool = True,
    filter_numbers: bool = True,
    filter_fragments: bool = True,
    filter_stop_words: bool = False
) -> Dict[str, Dict]:
    """
    Filter token statistics dictionary based on specified filter flags.

    This function is designed for use with token_stats from labeling service,
    which has format: {'token': {'count': N, 'total_activation': X}, ...}

    Args:
        token_stats: Dictionary mapping token to stats dict with 'count' and 'total_activation'
        filter_special: Filter special tokens (<s>, </s>, etc.)
        filter_single_char: Filter single character tokens
        filter_punctuation: Filter pure punctuation
        filter_numbers: Filter pure numeric tokens
        filter_fragments: Filter word fragments (BPE subwords)
        filter_stop_words: Filter common stop words (the, and, is, etc.)

    Returns:
        Filtered dictionary with same structure as input

    Example:
        >>> stats = {'<s>': {'count': 10, 'total_activation': 5.0}, 'news': {'count': 5, 'total_activation': 3.0}}
        >>> filtered = filter_token_stats(stats, filter_special=True)
        >>> 'news' in filtered
        True
        >>> '<s>' in filtered
        False
    """
    filtered = {}

    for token, stats in token_stats.items():
        # Check if token should be filtered
        if not is_junk_token(
            token,
            filter_special=filter_special,
            filter_single_char=filter_single_char,
            filter_punctuation=filter_punctuation,
            filter_numbers=filter_numbers,
            filter_fragments=filter_fragments,
            filter_stop_words=filter_stop_words
        ):
            filtered[token] = stats

    return filtered
