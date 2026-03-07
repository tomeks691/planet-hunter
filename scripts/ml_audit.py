#!/usr/bin/env python3
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

DB_PATH = Path('/app/data/ml_training.db')
OUT_JSON = Path('/app/data/ml/reports/data_audit_v1.json')
OUT_MD = Path('/app/data/ml/reports/data_audit_v1.md')

COLUMNS = [
    'period','depth','snr','duration','secondary_depth','odd_even_sigma',
    'sinusoid_better','sectors_ratio','tmag','teff','star_radius',
    'planet_radius','equilibrium_temp'
]


def q(cur, sql, params=()):
    return cur.execute(sql, params).fetchone()[0]


def main():
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    cur = conn.cursor()

    total = q(cur, 'SELECT COUNT(*) FROM training_data')
    class_dist = cur.execute('SELECT label, COUNT(*) FROM training_data GROUP BY label ORDER BY COUNT(*) DESC').fetchall()

    nulls = {}
    stats = {}
    for c in COLUMNS:
        nulls[c] = int(q(cur, f"SELECT SUM(CASE WHEN {c} IS NULL THEN 1 ELSE 0 END) FROM training_data") or 0)
        row = cur.execute(f"SELECT MIN({c}), MAX({c}), AVG({c}) FROM training_data WHERE {c} IS NOT NULL").fetchone()
        stats[c] = {'min': row[0], 'max': row[1], 'avg': row[2]}

    # Outlier candidate counts by hard rules (first pass)
    outlier_rules = {
        'depth_gt_1': "depth > 1",
        'snr_gt_300': "snr > 300",
        'odd_even_sigma_gt_20': "odd_even_sigma > 20",
        'duration_gt_0_6_period': "duration > 0.6 * period",
        'secondary_gt_primary': "secondary_depth > depth",
    }
    outliers = {k: int(q(cur, f"SELECT COUNT(*) FROM training_data WHERE {cond}")) for k, cond in outlier_rules.items()}

    kp_outliers = {k: int(q(cur, f"SELECT COUNT(*) FROM training_data WHERE label='KNOWN_PLANET' AND {cond}")) for k, cond in outlier_rules.items()}

    payload = {
        'rows_total': total,
        'class_distribution': {k: v for k, v in class_dist},
        'null_counts': nulls,
        'feature_stats': stats,
        'outliers_all': outliers,
        'outliers_known_planet': kp_outliers,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding='utf-8')

    lines = []
    lines.append('# Data Audit v1')
    lines.append('')
    lines.append(f'- Total rows: **{total}**')
    lines.append('- Class distribution:')
    for k, v in class_dist:
        lines.append(f'  - {k}: {v}')
    lines.append('')
    lines.append('## Null counts')
    for c in COLUMNS:
        lines.append(f'- {c}: {nulls[c]}')
    lines.append('')
    lines.append('## Outlier candidates (all rows)')
    for k, v in outliers.items():
        lines.append(f'- {k}: {v}')
    lines.append('')
    lines.append('## Outlier candidates (KNOWN_PLANET only)')
    for k, v in kp_outliers.items():
        lines.append(f'- {k}: {v}')

    OUT_MD.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    conn.close()

    print('Audit written:')
    print(OUT_JSON)
    print(OUT_MD)


if __name__ == '__main__':
    main()
