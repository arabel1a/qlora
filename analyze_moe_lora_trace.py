#!/usr/bin/env python3
"""Analyze the b16 MoE-LoRA vs base traces to explain why LoRA (bgmv) dominates
the MoE (GroupedMatmul) time. Reads rank0 ASCEND_PROFILER_OUTPUT csvs."""
import csv, glob, sys
from collections import defaultdict

def find(root, name):
    g = glob.glob(f"traces/{root}/trace/*rank0*/ASCEND_PROFILER_OUTPUT/{name}")
    assert g, f"not found: {root}/{name}"
    return g[0]

def op_stat(root):
    rows = []
    with open(find(root, "op_statistic.csv")) as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows

def bucket(op):
    n = op.lower()
    if n.startswith("bgmv") or n.startswith("sgmv"):
        return "LoRA(bgmv/sgmv)"
    if "groupedmatmul" in n:
        return "MoE-FFN(GroupedMatmul)"
    return None

for root in ("base/prof_base_v018_b16", "lora/prof_lora_v018_b16"):
    print(f"\n{'='*70}\n{root}\n{'='*70}")
    rows = op_stat(root)
    tot = sum(float(r["Total Time(us)"]) for r in rows)
    agg = defaultdict(lambda: [0.0, 0])
    for r in rows:
        b = bucket(r["OP Type"])
        if b:
            agg[b][0] += float(r["Total Time(us)"]); agg[b][1] += int(r["Count"])
    print(f"total kernel time on rank0: {tot/1e3:.1f} ms")
    for b, (t, c) in sorted(agg.items(), key=lambda x: -x[1][0]):
        print(f"  {b:28s} {t/1e3:9.1f} ms  ({100*t/tot:5.1f}% of all)  count={c}")
    # top bgmv variants
    print("  -- bgmv/sgmv variants --")
    for r in sorted(rows, key=lambda r: -float(r["Total Time(us)"])):
        if bucket(r["OP Type"]) == "LoRA(bgmv/sgmv)":
            print(f"    {r['OP Type']:32s} {r['Core Type']:16s} cnt={r['Count']:>6} "
                  f"tot={float(r['Total Time(us)'])/1e3:8.1f}ms avg={float(r['Avg Time(us)']):8.1f}us "
                  f"max={float(r['Max Time(us)']):8.1f}us")

# --- kernel_details: input shapes of the expensive lora bgmv ---
print(f"\n{'='*70}\nlora bgmv kernel input-shape distribution (rank0 kernel_details)\n{'='*70}")
kd = find("lora/prof_lora_v018_b16", "kernel_details.csv")
by_name_shape = defaultdict(lambda: [0.0, 0])
with open(kd) as f:
    for r in csv.DictReader(f):
        nm = r["Name"]
        if not (nm.startswith("bgmv") or nm.startswith("sgmv")):
            continue
        shp = r["Input Shapes"].replace('"', '')
        # first input shape = the activation token dim
        first = shp.split(";")[0] if shp else "?"
        dur = float(r["Duration(us)"]) if r["Duration(us)"] else 0.0
        by_name_shape[(nm.split("bfloat16")[0]+ "…", first)][0] += dur
        by_name_shape[(nm.split("bfloat16")[0]+ "…", first)][1] += 1
for (nm, shp), (t, c) in sorted(by_name_shape.items(), key=lambda x: -x[1][0])[:20]:
    print(f"  {nm:26s} in0={shp:20s} cnt={c:>6} tot={t/1e3:8.1f}ms avg={t/max(c,1):7.1f}us")
