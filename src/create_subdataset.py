import pandas as pd
from collections import defaultdict
import random
import json
import os

# Config
FULL = 'data/processed/events_cleaned.csv'
OUT = 'data/processed/subdataset_for_conversion.csv'
META_OUT = 'data/processed/subdataset_for_conversion_meta.json' 
CHUNK = 200_000
N_USERS = 20000 
MIN_CONVERTERS = 2000
RANDOM_SEED = 42

random.seed(RANDOM_SEED)

def pass1_user_stats():
    stats = {}  # visitorid -> {'events': int, 'has_tx': bool}
    cols = ['visitorid', 'event']
    for chunk in pd.read_csv(FULL, usecols=cols, chunksize=CHUNK, low_memory=False):
        chunk = chunk.dropna(subset=['visitorid'])
        # visitorid may be numeric; convert to str for consistent keys
        chunk['visitorid'] = chunk['visitorid'].astype(str)
        for vid, g in chunk.groupby('visitorid'):
            s = stats.get(vid)
            if s is None:
                stats[vid] = {'events': len(g), 'has_tx': ('transaction' in g['event'].values)}
            else:
                s['events'] += len(g)
                if not s['has_tx'] and ('transaction' in g['event'].values):
                    s['has_tx'] = True
    return stats

def sample_users(stats, n_users=N_USERS, min_converters=MIN_CONVERTERS):
    conv = [u for u,v in stats.items() if v['has_tx']]
    nonconv = [u for u,v in stats.items() if not v['has_tx']]
    pop_conv = len(conv)
    pop_nonconv = len(nonconv)
    pop_total = pop_conv + pop_nonconv
    if pop_total == 0:
        raise SystemExit('No users found in dataset (visitorid empty).')

    # default proportionate sample (unbiased) for converters
    prop_conv = int(round(n_users * pop_conv / pop_total))
    # ensure minimum converters if possible
    sample_conv = min(pop_conv, max(prop_conv, min_converters))
    sample_conv = min(sample_conv, pop_conv)
    sample_nonconv = max(1, n_users - sample_conv)
    sample_nonconv = min(sample_nonconv, pop_nonconv)
    # if shortage, adjust
    if sample_conv + sample_nonconv < n_users:
        # fill remaining from nonconv if available, else conv
        remaining = n_users - (sample_conv + sample_nonconv)
        add_nonconv = min(remaining, pop_nonconv - sample_nonconv)
        sample_nonconv += add_nonconv
        remaining -= add_nonconv
        sample_conv += min(remaining, pop_conv - sample_conv)

    sampled_conv = random.sample(conv, sample_conv) if sample_conv < len(conv) else conv
    sampled_nonconv = random.sample(nonconv, sample_nonconv) if sample_nonconv < len(nonconv) else nonconv
    sampled_users = set(sampled_conv + sampled_nonconv)

    # sampling fractions per stratum for weighting
    frac_conv = (len(sampled_conv) / pop_conv) if pop_conv>0 else None
    frac_nonconv = (len(sampled_nonconv) / pop_nonconv) if pop_nonconv>0 else None
    weights = {'conv': (1.0/frac_conv) if frac_conv else None, 'nonconv': (1.0/frac_nonconv) if frac_nonconv else None}
    meta = {
        'n_users_requested': n_users,
        'population_users': pop_total,
        'population_converters': pop_conv,
        'population_nonconverters': pop_nonconv,
        'sampled_users_total': len(sampled_users),
        'sampled_conv': len(sampled_conv),
        'sampled_nonconv': len(sampled_nonconv),
        'sampling_frac_conv': frac_conv,
        'sampling_frac_nonconv': frac_nonconv,
        'weights': weights,
        'random_seed': RANDOM_SEED
    }
    return sampled_users, meta

def pass2_extract(sampled_users):
    os.makedirs(os.path.dirname(OUT) or '.', exist_ok=True)
    reader = pd.read_csv(FULL, chunksize=CHUNK, low_memory=False)
    header_written = False
    written_rows = 0
    for chunk in reader:
        chunk['visitorid'] = chunk['visitorid'].astype(str)
        filt = chunk[chunk['visitorid'].isin(sampled_users)]
        if len(filt)==0:
            continue
        if not header_written:
            filt.to_csv(OUT, index=False, header=True, mode='w')
            header_written = True
        else:
            filt.to_csv(OUT, index=False, header=False, mode='a')
        written_rows += len(filt)
    return written_rows

def main():
    print('Pass 1: computing user stats (this may take a while)...')
    stats = pass1_user_stats()
    print('Sampling users...')
    sampled_users, meta = sample_users(stats)
    print('Pass 2: extracting rows for sampled users...')
    written = pass2_extract(sampled_users)
    meta['written_rows'] = written
    # save meta
    with open(META_OUT, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)
    print('Done. Wrote', written, 'rows to', OUT)
    print('Sample metadata saved to', META_OUT)

if __name__ == '__main__':
    main()