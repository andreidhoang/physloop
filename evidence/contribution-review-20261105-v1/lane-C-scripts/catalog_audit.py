import csv, sys, collections, os
d = sys.argv[1]
def rows(f):
    with open(os.path.join(d, f), newline='') as h:
        return list(csv.DictReader(h))
s01, s04 = rows('sim_scenes.csv'), rows('sim_scenes_2604.csv')
suites, cur = rows('sim_suites.csv'), rows('sim_suites_curated.csv')
print('scenes 26.01', len(s01), 'unique scene_id', len({r['scene_id'] for r in s01}))
print('scenes 26.04 (minus 26.01)', len(s04), 'unique', len({r['scene_id'] for r in s04}))
print('overlap scene_id 01/04', len({r['scene_id'] for r in s01} & {r['scene_id'] for r in s04}))
c = collections.Counter(r['test_suite_id'] for r in cur)
print('curated suites', dict(c))
val = [r['scene_id'] for r in cur if r['test_suite_id']=='nurec_curated_val']
tr = [r['scene_id'] for r in cur if r['test_suite_id']=='nurec_curated_train']
print('val unique', len(set(val)), 'train unique', len(set(tr)), 'val&train', len(set(val)&set(tr)))
src = {r['scene_id']: '26.01' for r in s01}
for r in s04: src.setdefault(r['scene_id'], '26.04')
print('val by release', collections.Counter(src.get(s,'missing') for s in val))
print('train by release', collections.Counter(src.get(s,'missing') for s in tr))
print('nre versions', collections.Counter(r['nre_version_string'] for r in s01+s04))
print('hf revisions', collections.Counter(r['hf_revision'] for r in s01+s04))
print('columns', list(s01[0].keys()))
print('public_2601 count', sum(r['test_suite_id']=='public_2601' for r in suites))
print('val in public_2601', len(set(val) & {r['scene_id'] for r in suites if r['test_suite_id']=='public_2601'}))
# path pattern check
print('path prefixes', collections.Counter('/'.join(r['path'].split('/')[:2]) for r in s01+s04))
print('scene_id == clipgt-<path uuid>', all(r['scene_id']=='clipgt-'+r['path'].split('/')[2] for r in s01+s04))
