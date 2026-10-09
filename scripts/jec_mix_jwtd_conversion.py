"""Published per-row JWTD v2 conversion math, unchanged from the pinned producer.

Code: MIT. Derived JWTD data: CC BY-SA 3.0. This module neither releases private
quarantine forms nor claims to reproduce their full joint graph. It accepts an
explicit publisher row and preserves its literal source/target and metadata.
Original producer SHA-256: 4b8ab14709d08602c7e90f5ead19a98ff183e47a51fc64c673effe033be49504
"""
import hashlib
import json
import unicodedata

def normalized(text):
    return ''.join(c for c in unicodedata.normalize('NFKC', text) if not c.isspace())

def fingerprint(text):
    return hashlib.sha256(text.encode('utf-8')).digest()

def positions(text, span):
    if not span:
        return range(len(text) + 1)
    found = []
    start = 0
    while True:
        start = text.find(span, start)
        if start < 0:
            return found
        found.append(start)
        start += 1

def minimal_interval(source, target):
    left = 0
    while left < min(len(source), len(target)) and source[left] == target[left]:
        left += 1
    right = 0
    while (right < len(source) - left and right < len(target) - left
           and source[-1 - right] == target[-1 - right]):
        right += 1
    source_end = len(source) - right
    target_end = len(target) - right
    anchored = 'none'
    if source_end == left or target_end == left:
        if right:
            source_end += 1
            target_end += 1
            anchored = 'right'
        elif left:
            left -= 1
            anchored = 'left'
        else:
            raise ValueError('cannot_anchor_empty_interval')
    return left, source_end, target[left:target_end], anchored

def convert(row, line, raw_hash, split, exclude_not_typo=False):
    """Return either an exact supplied-span record or one first-exclusion reason."""
    fields = ('page', 'title', 'pre_rev', 'post_rev', 'pre_text', 'post_text')
    if not isinstance(row, dict) or not all(isinstance(row.get(k), str) for k in fields):
        return None, 'missing_or_nonstring_required_field'
    source, target = row['pre_text'], row['post_text']
    if not all(row[k].isdigit() for k in ('page', 'pre_rev', 'post_rev')):
        return None, 'invalid_publisher_identifier'
    if not 4 <= len(source) <= 256:
        return None, 'source_length_outside_4_256'
    if source == target or normalized(source) == normalized(target):
        return None, 'unchanged_raw_or_normalized_pair'
    if any(char in source or char in target for char in ('\r', '\n')):
        return None, 'multiline_publisher_sentence'
    if '<|' in source or '<|' in target:
        return None, 'reserved_qwen_chat_marker_in_sentence'
    if any(mark in source for mark in ('<error>', '</error>')):
        return None, 'reserved_prompt_marker_in_source'
    diffs = row.get('diffs')
    if not isinstance(diffs, list) or len(diffs) != 1:
        return None, 'not_exactly_one_publisher_diff'
    diff = diffs[0]
    if not isinstance(diff, dict) or not all(isinstance(diff.get(k), str)
                                             for k in ('pre_str', 'post_str', 'category')):
        return None, 'invalid_publisher_diff_schema'
    if exclude_not_typo and diff['category'] == 'not-typo':
        return None, 'publisher_marks_diff_not_typo'
    old, new = diff['pre_str'], diff['post_str']
    if old == new:
        return None, 'unchanged_publisher_diff'
    matches = [i for i in positions(source, old)
               if source[:i] + new + source[i + len(old):] == target]
    if not matches:
        return None, 'publisher_diff_does_not_reconstruct_post_text'
    if len(matches) != 1:
        return None, 'ambiguous_publisher_diff_placement'
    try:
        start, end, revision, anchored = minimal_interval(source, target)
    except ValueError as exc:
        return None, str(exc)
    span = source[start:end]
    if not span or not revision or span == revision:
        return None, 'empty_or_unchanged_supplied_span'
    if len(span) > 40:
        return None, 'supplied_span_longer_than_40'
    if len(revision) > max(44, 4 * len(span)):
        return None, 'replacement_exceeds_existing_bound'
    if revision != revision.strip():
        return None, 'replacement_incompatible_with_decoder_strip'
    if source[:start] + revision + source[end:] != target:
        raise AssertionError('derived interval failed full-sentence reconstruction')
    # Preserve every publisher field without duplicating the two complete texts.
    metadata = {k: v for k, v in row.items() if k not in ('pre_text', 'post_text')}
    try:
        json.dumps(metadata, ensure_ascii=False, allow_nan=False)
    except (ValueError, TypeError):
        return None, 'nonfinite_or_nonserializable_publisher_provenance'
    ns, nt = normalized(source), normalized(target)
    pair_hash = hashlib.sha256(json.dumps([ns, nt], ensure_ascii=False,
                                          separators=(',', ':')).encode()).hexdigest()
    result = {
        'id': f'JWTD-v2-{split}-{line:09d}',
        'text': source,
        'error_type': 'external_typo',
        'gold_span_start': start,
        'gold_span_end': end,
        'gold_span': span,
        'gold_revision': revision,
        'corrected_text': target,
        'source': 'jwtd_v2_wikipedia_revision',
        'source_split': split,
        'source_line': line,
        'source_line_sha256': raw_hash,
        'article_group': 'jwtd-v2-page-' + row['page'],
        'normalized_source_sha256': fingerprint(ns).hex(),
        'normalized_corrected_sha256': fingerprint(nt).hex(),
        'normalized_pair_sha256': pair_hash,
        'publisher_diff_start': matches[0],
        'publisher_diff_end': matches[0] + len(old),
        'empty_interval_anchor': anchored,
        'publisher_metadata': metadata,
    }
    return result, None
