"""Verify the public synthesis against preserved aggregates, figure bytes and PDF.

This verifies recorded calculations and artifact completeness. It is not a new
semantic review, new model inference, or an estimate of direct human work hours.
"""
from pathlib import Path
from collections import Counter
from datetime import date
import argparse, csv, hashlib, json, math, re
from pypdf import PdfReader


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize(value):
    value = re.sub(r'\[([^\]]+)\]\(https?://[^\s)]+\)', r'\1', str(value))
    return re.sub(r'\s+', '', value.replace('**', '').replace('\u2212', '-').replace('\u2011', '-').replace('\u2060', ''))


def verify(root):
    data = root / 'public-data'
    read = lambda name: json.loads((data / name).read_text(encoding='utf-8'))
    d = read('report_data.json')
    evidence = read('synthesis_evidence.json')
    sections = d['report']['sections']
    queries = d['queries']
    assert d['buildStatus'] == 'complete'
    assert len(sections) == 51 == len({s['id'] for s in sections})
    for s in sections:
        assert s['queryId'] in queries
        for c in s.get('columns', []):
            assert all(c['field'] in row for row in queries[s['queryId']]['rows']), (s['id'], c)

    manifest = json.loads((root / 'figures/figure_manifest.json').read_text(encoding='utf-8'))
    assert manifest['canonical_data_sha256'] == sha(data / 'report_data.json')
    assert len(manifest['figures']) == sum('figure' in s for s in sections) == 18
    for f in manifest['figures']:
        assert all(q in queries for q in f['queries'])
        for suffix, digest in f['files'].items():
            assert sha(root / 'figures' / (f['figure'] + suffix)) == digest

    historical = {r['variant']: r for r in evidence['historical_metrics']
                  if r['regime'] == 'extended_8192' and r['scope'] == 'private'}
    original_rows = queries['synthesis-original-models']['rows']
    assert len(original_rows) == len(historical) == 13
    for r in original_rows:
        src = historical[r['variant']]
        assert (r['semantic'], r['instruction'], r['joint'], r['n']) == (
            src['semantic_correct'], src['instruction_passes'], src['passes'], src['total_questions'])
        assert r['semanticRate'] == r['semantic'] / r['n']
        assert r['jointRate'] == r['joint'] / r['n']

    development = {r['model']: r for r in csv.DictReader((data / 'development_metrics.csv').read_text(encoding='utf-8-sig').splitlines())}
    recovery = read('recovery_selection.json')
    all_rows = queries['synthesis-all-development']['rows']
    assert len(all_rows) == 13 == len({r['model'] for r in all_rows})
    for r in all_rows:
        if r['model'].startswith('R6'):
            src = recovery['tables'][r['model']]
            expected = [src[k] for k in ['document', 'valid_explanations', 'gsm8k', 'kobest_boolq', 'kobest_corrected']]
        else:
            src = development[r['model']]
            expected = [int(src[k]) for k in ['document_correct', 'valid_explanations', 'gsm8k_correct', 'kobest_official', 'kobest_corrected']]
        assert [r[k] for k in ['document', 'valid', 'gsm8k', 'kobest', 'corrected']] == expected
        for k, denom in [('document', 96), ('valid', 96), ('gsm8k', 40), ('kobest', 40)]:
            assert r[k + 'Rate'] == r[k] / denom
    assert next(r for r in all_rows if r['model'] == 'C5-152')['corrected'] == 37
    assert recovery['selected'] is None and recovery['eligible'] == [] and not recovery['final_allowed']

    confirmed = {(r['cohort'], r['model']): r for r in evidence['metrics']}
    for r in queries['synthesis-confirmation']['rows']:
        for model in ['S0', 'ARGOS']:
            src = confirmed[(r['cohort_id'], model)]
            assert r[model] == src['semantic_correct'] and r['n'] == src['n']
        assert r['delta_pp'] == round(100 * (r['ARGOS'] - r['S0']) / r['n'], 2)
    assert sum(r['emitted_atomic_claims'] for r in evidence['claim_metrics']) == 409
    for criterion in ['overall', 'naturalness']:
        matrix = [r for r in queries['synthesis-pair-matrix']['rows'] if r['criterion'] == criterion]
        assert len(matrix) == 9 and sum(r['count'] for r in matrix) == 72
        diagonal = sum(r['count'] for r in matrix if r['codex'] == r['project_pro'])
        marginal_a = Counter(); marginal_b = Counter()
        for r in matrix:
            marginal_a[r['codex']] += r['count']; marginal_b[r['project_pro']] += r['count']
        chance = sum(marginal_a[k] * marginal_b[k] for k in ['A', 'B', 'tie']) / 72**2
        agreement = read('pair_aggregate.json')['agreement'][criterion]
        assert diagonal == agreement['agree']
        assert math.isclose((diagonal / 72 - chance) / (1 - chance), agreement['cohen_kappa'], abs_tol=1e-12)

    activity = read('synthesis_activity.json')
    assert activity['raw_user_role_records'] - sum(activity['excluded_records'].values()) == 177
    assert sum(r['messages'] for r in activity['daily']) == activity['remaining_user_interaction_records'] == 177
    assert activity['direct_human_work_hours_measured'] is False
    for scope in ['all_observed_dates', 'late_phase_from_oct23']:
        for row in activity[scope]:
            spans = row['anonymous_session_spans']
            assert len(spans) == row['session_count']
            assert sum(s['interactions'] == 1 for s in spans) == row['single_message_sessions']
            assert all(s['span_seconds'] >= 0 and (s['interactions'] != 1 or s['span_seconds'] == 0) for s in spans)
            assert math.isclose(sum(s['span_seconds'] for s in spans) / 3600, row['linked_message_window_hours'], abs_tol=1e-10)
            assert sum(s['interactions'] for s in spans) == (177 if scope == 'all_observed_dates' else 158)
    assert (date.fromisoformat(activity['research_close']) - date.fromisoformat(activity['documented_design_start'])).days == 79
    assert activity['pc_uptime_hours_user_reported'] == 96 and activity['project_expenditure_USD_user_reported'] == 40
    assert read('synthesis_peer_review.json')['completed_rounds'] == 2

    reader = PdfReader(root / 'ARGOS_Improvement_Report_KO.pdf')
    lines = []
    for page_number, page in enumerate(reader.pages, 1):
        for line_number, line in enumerate((page.extract_text() or '').splitlines()):
            decoration = line_number < 8 and line.strip() in ['CoMPaSS / ARGOS', '제작과 개선 연구 최종보고서', '2026. 10. 27.', str(page_number)]
            if not decoration:
                lines.append(line)
    pdf_text = normalize('\n'.join(lines))
    assert '\ufffd' not in pdf_text
    assert all(normalize(s['title']) in pdf_text for s in sections)
    markdown = (root / 'REPORT_KO.md').read_text(encoding='utf-8')
    assert all(s['body'] in markdown for s in sections)
    table_values = 0
    for s in sections:
        for row in queries[s['queryId']]['rows']:
            for col in s.get('columns', []):
                value = row[col['field']]
                if value is not None:
                    assert normalize(value) in pdf_text, (s['id'], col['field'], value)
                table_values += 1
    if (data / 'synthesis_design_preservation.json').exists():
        baseline = read('synthesis_design_preservation.json')
        assert sha(data / 'report_data.json') == baseline['canonical_data_sha256']
        for name, digest in baseline['figures'].items():
            assert sha(root / 'figures' / name) == digest

    return dict(status='verified', sections=51, figures=18, original_model_conditions=13,
                development_model_conditions=13, pair_records=72, activity_records=177,
                direct_human_work_hours_measured=False, pdf_pages=len(reader.pages),
                table_cells_inspected=table_values, pdf_sha256=sha(root / 'ARGOS_Improvement_Report_KO.pdf'),
                canonical_data_sha256=sha(data / 'report_data.json'),
                scope='Aggregate, cross-artifact and layout-data integrity; no independent semantic adjudication or new inference.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    print(json.dumps(verify(args.root), ensure_ascii=False, indent=2))
