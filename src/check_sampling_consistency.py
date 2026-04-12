#!/usr/bin/env python3
"""Check sampling consistency between subdataset and full dataset.
"""
from __future__ import annotations
import json
import pandas as pd
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
META = ROOT / 'data' / 'processed' / 'subdataset_for_conversion_meta.json'
SUB = ROOT / 'data' / 'processed' / 'subdataset_for_conversion.csv'
FULL = ROOT / 'data' / 'processed' / 'events_cleaned.csv'

def print_header(t):
    print('\n' + '='*8 + ' ' + t + ' ' + '='*8)

def load_meta():
    if not META.exists():
        print('Meta file not found at', META)
        return None
    with open(META, 'r', encoding='utf-8') as f:
        m = json.load(f)
    print('Loaded meta keys:', ', '.join(m.keys()))
    print('population_users =', m.get('population_users'))
    print('population_converters =', m.get('population_converters'))
    print('sampled_users =', m.get('sampled_users'))
    print('sampled_converters =', m.get('sampled_converters'))
    print('weights (if present) =', m.get('weights'))
    return m

def count_sample_converters():
    if not SUB.exists():
        print('Subdataset file not found at', SUB)
        return None
    print('Reading subdataset (visitorid,event) grouping by user...')
    usecols = ['visitorid','event']
    df = pd.read_csv(SUB, usecols=usecols, low_memory=False)
    df['visitorid'] = df['visitorid'].astype(str)
    users = df.groupby('visitorid')['event'].apply(lambda s: (s.astype(str).str.strip().str.lower()=='transaction').any())
    n_users = users.shape[0]
    n_conv = int(users.sum())
    print('sample: users =', n_users, ', converters =', n_conv)
    return {'sample_users': n_users, 'sample_converters': n_conv}

def count_population_converters(chunk_size=200_000):
    if not FULL.exists():
        print('Full data file not found at', FULL)
        return None
    print(f'Scanning full file in chunks (chunksize={chunk_size}) to compute user-level converters...')
    converters = set()
    users_seen = set()
    cols = ['visitorid','event']
    for i,chunk in enumerate(pd.read_csv(FULL, usecols=cols, chunksize=chunk_size, low_memory=False)):
        # normalize
        chunk['visitorid'] = chunk['visitorid'].astype(str)
        ev = chunk['event'].astype(str).str.strip().str.lower()
        # mark converters by 'transaction' label
        mask_tr = ev == 'transaction'
        converters.update(chunk.loc[mask_tr, 'visitorid'].unique().tolist())
        users_seen.update(chunk['visitorid'].unique().tolist())
        if (i+1) % 10 == 0:
            print(f'  processed {((i+1)*chunk_size):,} rows, converters so far = {len(converters):,}, users seen = {len(users_seen):,}')
    print('Done scanning full file.')
    print('population users seen =', len(users_seen))
    print('population converters (by event==transaction) =', len(converters))
    return {'population_users_seen': len(users_seen), 'population_converters_by_event': len(converters)}

def list_event_labels_sample(n=100000):
    if not FULL.exists():
        return None
    print(f'Listing distinct event labels (first {n} rows sample)')
    s = pd.read_csv(FULL, usecols=['event'], nrows=n, low_memory=False)
    vals = s['event'].dropna().astype(str).str.strip().unique().tolist()
    print('sample event labels (unique):', vals[:50])
    return vals

def count_by_transactionid(chunk_size=200_000):
    if not FULL.exists():
        return None
    print('Counting converters by transactionid != 0 (chunked)')
    conv_set = set()
    cols = ['visitorid','transactionid']
    for chunk in pd.read_csv(FULL, usecols=cols, chunksize=chunk_size, low_memory=False):
        chunk['visitorid'] = chunk['visitorid'].astype(str)
        # consider non-null and not 0
        mask = chunk['transactionid'].notnull() & (chunk['transactionid'] != 0)
        conv_set.update(chunk.loc[mask, 'visitorid'].unique().tolist())
    print('converters by transactionid != 0 =', len(conv_set))
    return len(conv_set)

def check_visitorid_formats(n=50):
    if not FULL.exists():
        return None
    print('Checking visitorid formats (first rows)')
    df = pd.read_csv(FULL, usecols=['visitorid'], nrows=n, low_memory=False)
    print(df['visitorid'].head(20).to_list())
    types = df['visitorid'].apply(lambda x: type(x)).value_counts().to_dict()
    print('visitorid sample python types:', types)
    return True

def main():
    print_header('CHECK SAMPLING CONSISTENCY')
    meta = load_meta()
    print_header('SAMPLE COUNTS')
    sample_info = count_sample_converters()
    print_header('POPULATION COUNTS (SCAN)')
    pop_info = count_population_converters()
    print_header('EVENT LABELS (SAMPLE)')
    labels = list_event_labels_sample()
    print_header('TRANSACTIONID-BASED CONVERTERS')
    conv_by_tid = count_by_transactionid()
    print_header('VISITORID FORMATS')
    check_visitorid_formats()

    print('\nSummary suggestions:')
    if meta:
        pop_meta = meta.get('population_converters')
        if pop_meta is not None and pop_info:
            if pop_meta != pop_info.get('population_converters_by_event'):
                print('- NOTE: population_converters in meta (%s) DOES NOT match counted converters from full file (%s).' % (pop_meta, pop_info.get('population_converters_by_event')))
            else:
                print('- population_converters in meta matches scan result.')
    # compare sample converters
    if sample_info and pop_info:
        print('- sample converters:', sample_info.get('sample_converters'))
        print('- pop converters (by event):', pop_info.get('population_converters_by_event'))
        if sample_info.get('sample_converters') > pop_info.get('population_converters_by_event'):
            print('\nALERT: number of converters in sample is GREATER than population converters counted in full file.')
            print('Possible causes: event label mismatch, different definitions (transactionid vs event), or sample was built from a different source. Check event label variants and transactionid counts above.')

if __name__ == '__main__':
    main()
