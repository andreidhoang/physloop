import json,sys,statistics as st,glob,os
for fn in sorted(glob.glob(os.path.join(sys.argv[1],'od_*.json'))):
    o=json.load(open(fn))['offers']
    if not o: print(fn,'none'); continue
    pg=[x['dph_total']/x['num_gpus'] for x in o]
    pb=[x['dph_base']/x['num_gpus'] for x in o]
    bids=[x['min_bid']/x['num_gpus'] for x in o if x.get('min_bid') is not None]
    ver=sum(1 for x in o if x.get('verification')=='verified' or x.get('verified'))
    gpus=sum(x['num_gpus'] for x in o)
    fr=set(x.get('gpu_frac') for x in o)
    ram=set(x['gpu_ram'] for x in o)
    print(f"{os.path.basename(fn):28s} n_offers={len(o):3d} n_gpus={gpus:4d} verified={ver:3d} min={min(pg):.3f} med={st.median(pg):.3f} max={max(pg):.3f} | base min={min(pb):.3f} med={st.median(pb):.3f} | minbid min={min(bids):.3f} med={st.median(bids):.3f} gpuram={sorted(ram)[:3]}")
