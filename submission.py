"""Fill a competition template in its original row and column order."""
import csv
from pathlib import Path
import numpy as np
SUBMISSION_COLUMNS = ['row_id', 'recording_id', 'sample_index'] + [f'angle_{i:02d}' for i in range(20)]


def output_csv(model, signals, template, output):
    with template.open(newline='',encoding='utf-8-sig') as source:
        reader=csv.DictReader(source)
        if reader.fieldnames!=SUBMISSION_COLUMNS:
            raise ValueError('Expected the official template column order.')
        cache={}
        seen={}
        with output.open('w',newline='',encoding='utf-8') as target:
            writer=csv.DictWriter(target,fieldnames=reader.fieldnames)
            writer.writeheader()
            for row in reader:
                rid=row['recording_id']
                if Path(rid).name!=rid:
                    raise ValueError('Invalid recording filename.')
                if rid not in cache:
                    cache[rid]=model.predict(np.load(signals/(rid+'.npy'),allow_pickle=False))
                    seen[rid]=set()
                idx=int(row['sample_index'])
                if not 1000<=idx<len(cache[rid]) or idx in seen[rid]:
                    raise ValueError('Duplicate or out-of-range sample.')
                seen[rid].add(idx)
                row.update({f'angle_{j:02d}':format(float(v),'.9g') for j,v in enumerate(cache[rid][idx])})
                writer.writerow(row)
            if not cache or any(len(seen[rid])!=len(p)-1000 for rid,p in cache.items()):
                raise ValueError('Incomplete template.')
