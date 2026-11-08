import shutil, subprocess, sys
SUP, SPW = "gate/superiority.py", "gate/superiority_power.py"
M = [
 ("S1 agg drops guard_strata", SUP, '            tuple(d.get("guard_strata", ())),\n', '', 0),
 ("S2 dup strata accepted", SUP, 'if not out or len(set(out)) != len(out) or any(', 'if not out or any(', 0),
 ("S7 empty strata accepted", SUP, 'if not out or len(set(out)) != len(out) or any(', 'if len(set(out)) != len(out) or any(', 0),
 ("S3 guard subset", SUP, 'if set(g_strata) != set(expected_guard_strata):', 'if not set(g_strata) <= set(expected_guard_strata):', 0),
 ("S4 guard superset", SUP, 'if set(g_strata) != set(expected_guard_strata):', 'if not set(g_strata) >= set(expected_guard_strata):', 0),
 ("S5 guard_strata ignored", SUP, 'required if guard_strata is None else _strata_tuple(guard_strata, "guard strata")', 'required', 0),
 ("S6 records primary strata", SUP, '"guard_strata": sorted(expected_guard_strata),', '"guard_strata": sorted(required),', 0),
 ("N1 paired-observation check off", SUP, 'if refined and _total(g_counts) != _total(primary_counts):', 'if False:', 0),
 ("N2 guard worse flag wrong", SUP, '"worse_bound_unadjusted": g_lower is not None and g_lower > 0,', '"worse_bound_unadjusted": g_lower is not None and g_lower > 1,', 0),
 ("N3 any_guard flag constant", SUP, '        "any_guard_stratum_worse_bound_unadjusted": any(', '        "any_guard_stratum_worse_bound_unadjusted": False and any(', 0),
 ("N4 agg drops guard margin", SUP, '                    guards[g].get("margin"),\n', '', 0),
 ("N6 agg drops primary metric", SUP, '            json.dumps(d.get("metric"), sort_keys=True),\n', '', 0),
 ("P1 guard at primary margin", SPW, 'guards={"same_endpoint_noninferiority": stratified[m]},', 'guards={"same_endpoint_noninferiority": stratified[margin]},', 0),
 ("P2 drops BELOW_MIN", SPW, 'improved = c[sup.SUPPORTED] + c[sup.IMPROVED_BELOW_MIN]', 'improved = c[sup.SUPPORTED]', 0),
 ("P3 guard_go at primary margin", SPW, 'guard_go[label][s] += stratified[m].strata[s] == "GO"', 'guard_go[label][s] += stratified[margin].strata[s] == "GO"', 0),
 ("P4 guard_go counts joint", SPW, 'guard_go[label][s] += stratified[m].strata[s] == "GO"', 'guard_go[label][s] += stratified[m].overall == "GO"', 0),
 ("N5 joint counts one stratum", SPW, 'guard_joint[label] += stratified[m].overall == "GO"', 'guard_joint[label] += stratified[m].strata[required[0]] == "GO"', 0),
 ("P5 rate off by one", SPW, '"improvement_rate": improved / trials,', '"improvement_rate": improved / (trials + 1),', 0),
 ("P6 guard rate off by one", SPW, 'm: {s: g / trials for s, g in per.items()}', 'm: {s: g / (trials + 1) for s, g in per.items()}', 0),
 ("P7 effects dropped (design)", SPW, '            effects=effects,\n', '            effects=None,\n', 1),
 ("P8 pooled first stratum", SPW, 'pooled_rows = [{**row, "stratum": pooled_label} for row in rows]', 'pooled_rows = [{**row, "stratum": pooled_label} for row in rows if row["stratum"] == required[0]]', 0),
 ("P9 label check removed", SPW, 'if pooled_label in required and len(required) > 1:', 'if False:', 0),
 ("P10 paired seeds (design)", SPW, 'int(np.random.SeedSequence([seed, trial]).generate_state(1)[0]),', 'int(np.random.SeedSequence([seed, trial // 2]).generate_state(1)[0]),', 1),
 ("P11 label uniqueness off", SPW, 'if len(set(margins)) != len(margins) or len(set(labels)) != len(labels):', 'if len(set(margins)) != len(margins):', 0),
 ("G1 gating subset check off", SUP, 'if not set(gating) <= set(expected_guard_strata):', 'if False:', 0),
 ("G2 guards_go ignores gating", SUP, 'guards_go = all(g["gating_go"] for g in guard_report.values())', 'guards_go = all(g["state"] == "GO" for g in guard_report.values())', 0),
 ("G3 gating ignored", SUP, '        if gating_strata is None\n', '        if True\n', 0),
 ("G4 agg drops gating", SUP, '            tuple(d.get("gating_strata", ())),\n', '', 0),
 ("G5 runner gating check off", SPW, 'if gating_strata is not None and not set(gating_strata) <= set(required):', 'if False:', 0),
 ("G6 runner gating uses all strata", SPW, '                    gating_strata=tuple(gating_strata),', '                    gating_strata=required,', 0),
]
survivors = []
for name, path, old, new, occ in M:
    src = open(path).read()
    parts = src.split(old)
    if len(parts) - 1 <= occ:
        print(f"NO MATCH  {name}"); survivors.append(name + " (no match)"); continue
    mutated = old.join(parts[: occ + 1]) + new + old.join(parts[occ + 1 :])
    shutil.copy(path, path + ".bak")
    try:
        open(path, "w").write(mutated)
        r = subprocess.run([".venv/bin/python", "-m", "pytest", "-q", "-x", "tests/test_superiority.py"], capture_output=True, text=True)
        status = "KILLED " if r.returncode else "SURVIVED"
        if not r.returncode: survivors.append(name)
        print(f"{status} {name}")
    finally:
        shutil.move(path + ".bak", path)
print("survivors:", survivors)
