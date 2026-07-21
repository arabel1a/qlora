#!/usr/bin/env python3
"""Dig into WHY bgmv is slow in absolute terms: pull per-kernel hardware counters
(core type, block dim, memory-vs-compute ratios) for bgmv_shrink/expand vs the
expert GroupedMatmul from the lora b16 kernel_details.csv."""
import csv, glob
from collections import defaultdict

kd = glob.glob("traces/lora/prof_lora_v018_b16/trace/*rank0*/ASCEND_PROFILER_OUTPUT/kernel_details.csv")[0]

def fnum(s):
    try: return float(s)
    except: return None

groups = defaultdict(list)
with open(kd) as f:
    for r in csv.DictReader(f):
        nm = r["Name"]
        if nm.startswith("bgmv_shrink"): key = "bgmv_shrink"
        elif nm.startswith("bgmv_expand"): key = "bgmv_expand"
        elif nm == "GroupedMatmul" or r["Type"] == "GroupedMatmul": key = "GroupedMatmul(MoE-FFN)"
        elif nm == "MatMulV2": key = "MatMulV2"
        else: continue
        groups[key].append(r)

def col_mean(rows, c):
    vals = [fnum(r.get(c, "")) for r in rows]
    vals = [v for v in vals if v is not None]
    return sum(vals)/len(vals) if vals else None

def col_max(rows, c):
    vals = [fnum(r.get(c, "")) for r in rows]
    vals = [v for v in vals if v is not None]
    return max(vals) if vals else None

print(f"{'kernel':22s} {'core':16s} {'cnt':>5} {'dur_avg':>8} {'dur_max':>9} {'blkdim':>7} "
      f"{'mte2%':>7} {'vec%':>6} {'mac%':>6} {'cube_util%':>10}")
for k, rows in sorted(groups.items(), key=lambda x: -sum(fnum(r["Duration(us)"]) or 0 for r in x[1])):
    core = rows[0]["Accelerator Core"]
    dur_avg = col_mean(rows, "Duration(us)"); dur_max = col_max(rows, "Duration(us)")
    blk = col_mean(rows, "Block Dim")
    mte2 = col_mean(rows, "aiv_mte2_ratio"); vec = col_mean(rows, "aiv_vec_ratio")
    mac = col_mean(rows, "aic_mac_ratio"); cube = col_mean(rows, "cube_utilization(%)")
    def p(x, s=100): return f"{x*s:6.1f}" if x is not None else "   -- "
    print(f"{k:22s} {core:16s} {len(rows):>5} {dur_avg:8.1f} {dur_max:9.1f} "
          f"{(blk or 0):7.1f} {p(mte2)} {p(vec)} {p(mac)} {p(cube,1):>10}")

# block-dim = how many AI cores the kernel spreads across (occupancy proxy)
print("\nblock-dim distribution (occupancy) for the expensive bgmv_shrink:")
bd = defaultdict(int)
for r in groups["bgmv_shrink"]:
    bd[r["Block Dim"]] += 1
for k, v in sorted(bd.items(), key=lambda x: -x[1])[:8]:
    print(f"  block_dim={k:>4}  count={v}")
