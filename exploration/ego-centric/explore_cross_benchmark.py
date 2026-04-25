"""
Cross-benchmark analysis: video overlap across benchmarks, combined statistics,
and temporal coverage analysis.
"""

import json
from pathlib import Path
from collections import Counter, defaultdict

DATA_DIR = Path(__file__).parent / "ego4d_data" / "v2"
ANN_DIR = DATA_DIR / "annotations"


def load_video_uids(pattern):
    uids = set()
    for fp in ANN_DIR.glob(pattern):
        with open(fp) as f:
            data = json.load(f)
        if "videos" in data:
            for v in data["videos"]:
                uids.add(v["video_uid"])
        elif "clips" in data:
            for c in data["clips"]:
                uids.add(c["video_uid"])
    return uids


def main():
    print(f"{'='*60}")
    print(f"CROSS-BENCHMARK ANALYSIS")
    print(f"{'='*60}")

    # Load metadata
    with open(DATA_DIR / "ego4d.json") as f:
        meta = json.load(f)
    all_uids = {v["video_uid"] for v in meta["videos"]}
    video_lookup = {v["video_uid"]: v for v in meta["videos"]}

    # Load benchmark UIDs
    benchmarks = {
        "Narrations": load_video_uids("narration.json"),
        "NLQ": load_video_uids("nlq_*.json"),
        "Moments": load_video_uids("moments_*.json"),
        "FHO": load_video_uids("fho_main.json"),
        "AV": load_video_uids("av_*.json"),
    }

    print(f"\nTotal videos in metadata: {len(all_uids)}")
    print(f"\n--- Videos Per Benchmark ---")
    for name, uids in benchmarks.items():
        pct = 100 * len(uids) / len(all_uids) if all_uids else 0
        bar = "█" * max(1, int(pct / 2))
        print(f"  {name:12s} {len(uids):3d} ({pct:.0f}%) {bar}")

    # Overlap matrix
    bench_names = list(benchmarks.keys())
    print(f"\n--- Benchmark Overlap Matrix ---")
    header = f"{'':12s}" + "".join(f"{n:>10s}" for n in bench_names)
    print(header)
    for i, name_i in enumerate(bench_names):
        row = f"{name_i:12s}"
        for j, name_j in enumerate(bench_names):
            overlap = len(benchmarks[name_i] & benchmarks[name_j])
            row += f"{overlap:10d}"
        print(row)

    # Videos in multiple benchmarks
    video_bench_count = Counter()
    video_benches = defaultdict(set)
    for name, uids in benchmarks.items():
        for uid in uids:
            video_bench_count[uid] += 1
            video_benches[uid].add(name)

    print(f"\n--- Videos by Number of Benchmarks ---")
    count_dist = Counter(video_bench_count.values())
    for num_benchmarks in sorted(count_dist.keys()):
        count = count_dist[num_benchmarks]
        bar = "█" * max(1, count)
        print(f"  {num_benchmarks} benchmark(s): {count:3d} videos {bar}")

    # Scenario breakdown per benchmark
    print(f"\n--- Scenario Distribution Per Benchmark ---")
    for name, uids in benchmarks.items():
        scenarios = Counter()
        for uid in uids:
            if uid in video_lookup:
                for s in video_lookup[uid]["scenarios"]:
                    scenarios[s] += 1
        top_scenarios = scenarios.most_common(5)
        top_str = ", ".join(f"{s}({c})" for s, c in top_scenarios)
        print(f"  {name:12s}: {top_str}")

    # Duration breakdown per benchmark
    print(f"\n--- Duration Stats Per Benchmark ---")
    for name, uids in benchmarks.items():
        durations = [video_lookup[uid]["duration_sec"] for uid in uids if uid in video_lookup]
        if durations:
            total_h = sum(durations) / 3600
            mean_m = (sum(durations) / len(durations)) / 60
            print(f"  {name:12s}: {total_h:.1f}h total, {mean_m:.1f}min avg, {len(durations)} videos")

    # University distribution per benchmark
    print(f"\n--- University Source Per Benchmark ---")
    for name, uids in benchmarks.items():
        unis = Counter()
        for uid in uids:
            if uid in video_lookup:
                unis[video_lookup[uid]["video_source"]] += 1
        top_unis = unis.most_common(5)
        top_str = ", ".join(f"{u}({c})" for u, c in top_unis)
        print(f"  {name:12s}: {top_str}")


if __name__ == "__main__":
    main()
