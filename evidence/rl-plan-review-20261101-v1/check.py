from pathlib import Path
import datetime,difflib,hashlib,json,re,subprocess,sys
r=Path(__file__).resolve().parents[2]
review=r/'docs/research/2026-11-01-rl-direction-review'
paths=['AGENTS.md','CLAUDE.md','DOC_AUTHORITY.md','README.md','docs/00_START_HERE.md','docs/01_SYSTEM_DESIGN.md','docs/02_PHYSLOOP_PLAN.md','docs/03_MASTERY_PLAN.md','docs/04_CORE_PLAN_TECHNICAL.md','docs/05_ROLE_MAPPING.md','docs/RESEARCH_OPERATING_MODEL.md','docs/S2_ENGINEERING_SPEC.md','docs/S2_IMPLEMENTATION_PLAN.md','docs/S2_HANDOFF.md','docs/S2_RENTAL_APPROVAL.md','docs/research/2026-11-01-rl-research-path-proposal.md','docs/DECISIONS.md','journal/2026-11-01.md','docs/archive/02_PHYSLOOP_PLAN_v4.1_2026-11-01.md']
paths+=sorted(str(p.relative_to(r)) for p in review.glob('*.md'))
checks={};missing=[];count=0
for name in paths:
 p=r/name
 if name in ['docs/DECISIONS.md','docs/archive/02_PHYSLOOP_PLAN_v4.1_2026-11-01.md']:continue
 for target in re.findall(r'\]\(([^)]+)\)',p.read_text()):
  target=target.split('#')[0]
  if not target or re.match(r'^[a-zA-Z][a-zA-Z0-9+.-]*:',target):continue
  q=Path(target) if target.startswith('/') else p.parent/target
  count+=1
  if not q.exists():missing.append({'source':name,'target':target})
checks['local_markdown_destinations']={'pass':not missing,'count':count,'missing':missing,'limit':'Checks destination existence, not remote URLs or heading anchors.'}
archive=hashlib.sha256((r/paths[-1]).read_bytes()).hexdigest() if False else hashlib.sha256((r/'docs/archive/02_PHYSLOOP_PLAN_v4.1_2026-11-01.md').read_bytes()).hexdigest()
checks['pre_amendment_archive']={'pass':archive=='d30751e4467803c8fa0db0c55f830c8f03ed0cf1c9cd2ef5661f4f5d219204d4','sha256':archive}
def normalize_contract(s):
 s=s[s.index('You work with Dang Huy'):]
 s=re.sub(r'^(Codex has no `/reread`|`/reread` performs).*$','<tool command>',s,flags=re.M)
 s=re.sub(r'- \*\*Starred cores.*?(?=## Executing a task)','<tool enforcement>\n\n',s,flags=re.S)
 s=re.sub(r'^\| `(CLAUDE|AGENTS)\.md` \|.*$','<mirror reference>',s,flags=re.M)
 return s
ac=normalize_contract((r/'AGENTS.md').read_text());cc=normalize_contract((r/'CLAUDE.md').read_text())
checks['agent_contract_substance']={'pass':ac==cc,'normalization':'Only header/mirror intro, reread command, guarded-tool implementation and mirror-file row are ignored.'}
plan=(r/'docs/02_PHYSLOOP_PLAN.md').read_text()
checks['stable_plan_sections']={'pass':all(re.search(r'^## '+str(n)+r'\.',plan,re.M) for n in [2,3,6])}
checks['adoption_record']={'pass':'D-021' in (r/'docs/DECISIONS.md').read_text() and 'Authoritative, adopted 2026-11-01 under D-021' in plan}
av=len(re.findall(r'^### \d+\.',(review/'av-literature.md').read_text(),re.M));foundations=len(re.findall(r'^### F\d+\.',(review/'rl-foundations.md').read_text(),re.M))
checks['paper_set_arithmetic']={'pass':av==13 and foundations==17 and av+foundations+3+1==34,'av_ledger_entries':av,'foundation_ledger_entries':foundations,'adjacent_studies':3,'dagger_direct_source':1,'total':av+foundations+3+1,'limit':'Selected sources, not independent replications or citation-count ranking.'}
checks['active_s2_rollout_arithmetic']={'pass':2*441*3==2646 and '2,646' in (r/'docs/S2_ENGINEERING_SPEC.md').read_text(),'full_rollouts_before_diagnostics_retries':2*441*3}
checks['cost_table_arithmetic']={'pass':[68*1.8,68*2.49,170*1.8,170*2.49]==[122.4,169.32000000000002,306.0,423.3],'limit':'Historical rounded throughput/price scenario; not a current quote or measurement.'}
diff=subprocess.run(['git','diff','--check'],cwd=r,text=True,capture_output=True)
checks['git_diff_whitespace']={'pass':diff.returncode==0,'returncode':diff.returncode,'stdout':diff.stdout,'stderr':diff.stderr}
checks['temporary_draft_removed']={'pass':not (r/'docs/02_PHYSLOOP_PLAN_v5_DRAFT.md').exists()}
result={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'kind':'source-review documentation integrity; not GPU/model execution','git_base':subprocess.check_output(['git','rev-parse','HEAD'],cwd=r,text=True).strip(),'working_tree':'contains pre-existing uncommitted work; file hashes identify the reviewed document snapshot','checks':checks,'files':{n:hashlib.sha256((r/n).read_bytes()).hexdigest() for n in paths},'checker_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'all_pass':all(c['pass'] for c in checks.values())}
print(json.dumps(result,indent=2,ensure_ascii=False))
sys.exit(0 if result['all_pass'] else 1)
