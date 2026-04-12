import json
import pandas as pd
import os
from math import sqrt

# Paths
SUB = 'data/processed/subdataset_for_conversion.csv'
META = 'data/processed/subdataset_for_conversion_meta.json'
STRATUM = 'data/processed/subdataset_for_conversion_stratum.json' 
OUT_CSV = 'report/figures/cr_summary.csv'

def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (None, None)
    p = k / n
    z2 = z * z
    denom = 1 + z2 / n
    centre = (p + z2 / (2 * n)) / denom
    half = (z * ((p * (1 - p) / n) + (z2 / (4 * n * n)))**0.5) / denom
    return max(0.0, centre - half), min(1.0, centre + half)

def main():
    # load meta
    if not os.path.exists(META):
        print('Meta file not found:', META)
        return
    meta = json.load(open(META))

    # population CR from meta (if available)
    pop_conv = meta.get('population_converters')
    pop_total = meta.get('population_users')
    pop_cr = None
    if pop_conv is not None and pop_total:
        pop_cr = pop_conv / pop_total

    # read subdataset and compute user-level converted flag
    if not os.path.exists(SUB):
        print('Subdataset not found:', SUB)
        return
    df = pd.read_csv(SUB, usecols=['visitorid','event'], low_memory=False)
    df['visitorid'] = df['visitorid'].astype(str)
    users = df.groupby('visitorid')['event'].apply(lambda s: (s=='transaction').any()).reset_index(name='converted')
    n_users = len(users)
    n_conv = int(users['converted'].sum())
    unweighted_cr = n_conv / n_users if n_users else None
    ci_low, ci_up = wilson_ci(n_conv, n_users)

    # weighted CR if stratum map exists
    weighted_cr = None
    if os.path.exists(STRATUM):
        s = json.load(open(STRATUM))
        # support two formats: {'stratum_map': {id: 'conv'/'nonconv'}} or direct mapping
        if 'stratum_map' in s:
            stratum_map = s['stratum_map']
        else:
            stratum_map = s
        # ensure keys are strings
        stratum_map = {str(k): v for k, v in stratum_map.items()}
        users['stratum'] = users['visitorid'].map(stratum_map).fillna('nonconv')
        weights = meta.get('weights', {})
        w_conv = weights.get('conv')
        w_nonconv = weights.get('nonconv')
        if w_conv and w_nonconv:
            users['weight'] = users['stratum'].map({'conv': w_conv, 'nonconv': w_nonconv})
            users['w_conv'] = users['converted'] * users['weight']
            weighted_cr = users['w_conv'].sum() / users['weight'].sum()

    # prepare summary and save
    summary = {
        'subsample_users': n_users,
        'subsample_converters': n_conv,
        'unweighted_cr': unweighted_cr,
        'unweighted_cr_ci_low': ci_low,
        'unweighted_cr_ci_up': ci_up,
        'population_cr_from_meta': pop_cr,
        'weighted_cr_from_sample': weighted_cr,
        'meta_weights': meta.get('weights')
    }

    os.makedirs(os.path.dirname(OUT_CSV) or '.', exist_ok=True)
    pd.Series(summary).to_csv(OUT_CSV)

    # print friendly
    print('Subsample users:', n_users)
    print('Subsample converters:', n_conv)
    print('Unweighted CR (sample):', unweighted_cr)
    print('95% Wilson CI for sample CR:', (ci_low, ci_up))
    if pop_cr is not None:
        print('Population CR (from meta):', pop_cr)
    if weighted_cr is not None:
        print('Weighted CR (HT-style using strata weights):', weighted_cr)
    else:
        print('No stratum map found; weighted CR not computed. To enable, add', STRATUM)
    print('Summary written to', OUT_CSV)

if __name__ == '__main__':
    main()
