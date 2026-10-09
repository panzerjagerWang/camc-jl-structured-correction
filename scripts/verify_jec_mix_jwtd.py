"""Verify the public JEC-Mix 1.0 JWTD-only artifact; Python 3.11+, no ML dependencies.

Optional --publisher-train checks every released record against its original
line in the official training member. It never opens publisher test/gold data.
"""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import re

from jec_mix_jwtd_conversion import convert, normalized, fingerprint


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify(artifact, manifest, publisher_train=None):
    spec = json.loads(Path(manifest).read_text(encoding='utf-8'))
    require(sha(artifact) == spec['artifact']['gzip_sha256'], 'Compressed SHA mismatch')
    digest = hashlib.sha256()
    selected, identities, ids, articles = {}, set(), set(), set()
    categories = Counter()
    byte_count = 0
    expected_keys = set(spec['schema']['top_level_fields'])
    with gzip.open(artifact, 'rb') as stream:
        for raw in stream:
            digest.update(raw)
            byte_count += len(raw)
            row = json.loads(raw)
            require(set(row) == expected_keys, 'Unexpected row fields')
            require(row['source_split'] == 'train', 'Non-training row')
            require(row['source'] == 'jwtd_v2_wikipedia_revision', 'Unexpected source')
            require(row['error_type'] == 'external_typo', 'Category must remain external_typo')
            require(type(row['source_line']) is int and row['source_line'] > 0, 'Invalid source line')
            require(row['source_line'] not in selected and row['id'] not in ids, 'Duplicate row identity')
            source, target = row['text'], row['corrected_text']
            start, end = row['gold_span_start'], row['gold_span_end']
            require(type(start) is int and type(end) is int and 0 <= start < end <= len(source), 'Invalid interval')
            require(source[start:end] == row['gold_span'], 'Source span mismatch')
            revision = row['gold_revision']
            require(revision and revision == revision.strip(), 'Empty or whitespace-edged revision')
            require(1 <= end-start <= 40 and len(revision) <= max(44, 4*(end-start)), 'Decoder bounds mismatch')
            require(source[:start] + revision + source[end:] == target, 'Literal reconstruction mismatch')
            for key in ['source_line_sha256','normalized_source_sha256','normalized_corrected_sha256','normalized_pair_sha256','joint_component_id']:
                require(isinstance(row[key], str) and re.fullmatch('[0-9a-f]{64}', row[key]), 'Invalid SHA field')
            ns, nt = normalized(source), normalized(target)
            require(fingerprint(ns).hex() == row['normalized_source_sha256'], 'Normalized source SHA mismatch')
            require(fingerprint(nt).hex() == row['normalized_corrected_sha256'], 'Normalized target SHA mismatch')
            pair = hashlib.sha256(json.dumps([ns,nt], ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()
            require(pair == row['normalized_pair_sha256'] and pair not in identities, 'Pair SHA/uniqueness mismatch')
            meta = row['publisher_metadata']
            require(set(meta) == {'page','title','pre_rev','post_rev','diffs','lstm_average_likelihood'}, 'Publisher schema mismatch')
            require(all(isinstance(meta[k], str) and meta[k].isdigit() for k in ['page','pre_rev','post_rev']), 'Invalid revision provenance')
            require(row['article_group'] == 'jwtd-v2-page-'+meta['page'], 'Article identity mismatch')
            native = dict(meta, pre_text=source, post_text=target)
            replay, reason = convert(native, row['source_line'], row['source_line_sha256'], 'train', exclude_not_typo=True)
            require(reason is None and replay == {k:v for k,v in row.items() if k != 'joint_component_id'}, 'Conversion/metadata mismatch')
            categories[meta['diffs'][0]['category']] += 1
            selected[row['source_line']] = row
            identities.add(pair); ids.add(row['id']); articles.add(meta['page'])
    require(digest.hexdigest() == spec['artifact']['jsonl_sha256'], 'Uncompressed SHA mismatch')
    require(byte_count == spec['artifact']['jsonl_bytes'], 'Uncompressed byte-count mismatch')
    require(len(selected) == spec['records'] == 50000, 'Record-count mismatch')
    require(len(articles) == spec['article_groups'] == 40439, 'Article-count mismatch')
    require(dict(sorted(categories.items())) == spec['native_categories'], 'Native category-count mismatch')
    replayed = 0
    if publisher_train:
        require(sha(publisher_train) == spec['source']['training_member_sha256'], 'Official train SHA mismatch')
        with Path(publisher_train).open('rb') as stream:
            for line, raw in enumerate(stream, 1):
                if line not in selected:
                    continue
                observed = selected[line]
                require(hashlib.sha256(raw).hexdigest() == observed['source_line_sha256'], 'Original source-line SHA mismatch')
                expected, reason = convert(json.loads(raw), line, observed['source_line_sha256'], 'train', exclude_not_typo=True)
                require(reason is None and expected == {k:v for k,v in observed.items() if k != 'joint_component_id'}, 'Original publisher-row replay mismatch')
                replayed += 1
        require(replayed == 50000, 'Incomplete original publisher replay')
    return {'status':'passed','records':len(selected),'article_groups':len(articles),
            'gzip_sha256':sha(artifact),'jsonl_sha256':digest.hexdigest(),
            'source_rows_replayed':replayed,'other_publisher_splits_opened':False,
            'model_loaded':False,'metrics_computed':False,
            'joint_component_membership_independently_rebuilt':False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('artifact')
    parser.add_argument('--manifest', default=str(Path(__file__).resolve().parents[1] / 'data/jec-mix-v1/manifest.json'))
    parser.add_argument('--publisher-train')
    args = parser.parse_args()
    print(json.dumps(verify(args.artifact, args.manifest, args.publisher_train), indent=2))


if __name__ == '__main__':
    main()
