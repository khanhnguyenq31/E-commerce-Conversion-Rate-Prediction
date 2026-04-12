import pandas as pd, json
df = pd.read_csv('data/processed/subdataset_for_conversion.csv', usecols=['visitorid','event'], low_memory=False)
users = df.groupby('visitorid')['event'].apply(lambda s: (s=='transaction').any()).reset_index(name='is_conv')
stratum_map = {str(uid): ('conv' if is_conv else 'nonconv') for uid, is_conv in zip(users['visitorid'], users['is_conv'])}
json.dump(stratum_map, open('data/processed/subdataset_for_conversion_stratum.json','w'), indent=2)
print('Saved stratum_map for', len(stratum_map), 'users')