import json
from json.tool import main
import math
import argparse
from collections import defaultdict
import os
import sys

import matplotlib.pyplot as plt
import config


DIFFICULTY_ORDER = ["simple", "moderate", "challenging"]
METRICS = [
    #("jaccard_index", "Jaccard Index"),
    ("jaccard_index_new", "Jaccard Index"),
    ("exact_match", "Exact Match"),
    ("total_time", "Total time (s)"),
    ("spark_time", "Spark time (s)"),
    ("translation_time", "Translation\ntime (s)"),
]


def is_finite_number(x) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x)


def as_list(x):
    """Ensure x is a list; if missing/None -> empty list."""
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return [x]


def count_errors(entry: dict) -> int:
    """
    Count errors for one aggregated entry (one query_id+llm).
    Uses execution_status == "ERROR" or non-null spark_error items.
    """
    statuses = as_list(entry.get("execution_status"))
    spark_errs = as_list(entry.get("spark_error"))

    err_status_cnt = sum(1 for s in statuses if str(s).upper() == "ERROR")
    err_spark_cnt = sum(1 for e in spark_errs if e not in (None, "", "null"))

    # Avoid double-counting the same iteration if both signals exist:
    # We'll approximate by taking max of both counts (common in your data).
    return max(err_status_cnt, err_spark_cnt)


def extract_metric_values(entry: dict, metric_name: str):
    """
    Return per-iteration values for a metric as a list of floats if possible.
    Keep only finite numeric values.
    """
    vals = as_list(entry.get(metric_name))
    out = []

    for v in vals:
        # already numeric
        if isinstance(v, (int, float)) and math.isfinite(v):
            out.append(float(v))
            continue

        # numeric string
        if isinstance(v, str):
            s = v.strip()
            if not s:
                continue
            try:
                fv = float(s)
                if math.isfinite(fv):
                    out.append(fv)
            except ValueError:
                pass
            continue

        # sometimes your aggregation might store single values as e.g. numpy scalars etc.
        try:
            fv = float(v)
            if math.isfinite(fv):
                out.append(fv)
        except Exception:
            pass

    return out



def compute_global_ylims(grouped):
    """
    Compute y-limits per metric across all difficulties for comparability.
    For jaccard metrics: fixed 0..1
    For time metrics: 0..global_max (with small headroom)
    """
    ylims = {}

    # fixed for jaccard
    # ylims["jaccard_index"] = (-0.1, 1.1)
    ylims["jaccard_index_new"] = (-0.1, 1.1)
    ylims["exact_match"] = (-0.1, 1.1)

    # times: compute global max
    for metric_name in ["total_time", "spark_time", "translation_time"]:
        all_vals = []
        for diff_dict in grouped.values():
            for qid_dict in diff_dict.values():
                all_vals.extend(qid_dict["metrics"].get(metric_name, []))

        if not all_vals:
            ylims[metric_name] = (0.0, 1.0)
            continue

        vmax = max(all_vals)
        # add a bit of headroom
        ylims[metric_name] = (0.0, vmax * 1.05)

    return ylims


def add_error_counts_below_ticks(ax, x_positions, error_counts):
    """
    Draw small red numbers below the x tick labels.
    Uses axis coordinates so it stays below regardless of y-limits.
    """
    for x, e in zip(x_positions, error_counts):
        if e > 0:
            ax.text(
                x+0.25, -0.18, str(e),
                transform=ax.get_xaxis_transform(),
                ha="center", va="top",
                fontsize=12,
                color="red",
                clip_on=False,
            )


def plot_difficulty(grouped, difficulty: str, ylims, out_folder: str = config.PLOTS_FOLDER):
    """
    One figure per difficulty, with 5 subplots (Option A).
    x-axis: query_id
    Each subplot: boxplot per query_id for that metric.
    """
    if difficulty not in grouped or not grouped[difficulty]:
        return
    
    out_path = os.path.join(out_folder, f"benchmark_{difficulty}.png")
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    if os.path.exists(out_path) and not config.RECOMPUTE_PLOTS:
        print(f"Plot already exists, skipping: {out_path}")
        return

    # Sort query_ids for consistent ordering
    print("UNSORTED query_ids:", list(grouped[difficulty].keys())[:10])
    query_ids = sorted(grouped[difficulty].keys(), key=int)
    print("SORTED query_ids:", query_ids[:10])
    # print(f"Plotting difficulty='{difficulty}' with {len(query_ids)} query_ids... {query_ids[:10]}...")
    x_positions = list(range(1, len(query_ids) + 1))

    # error counts per query_id
    error_counts = [grouped[difficulty][qid]["errors"] for qid in query_ids]

    fig, axes = plt.subplots(
        nrows=len(METRICS),
        ncols=1,
        figsize=(max(10, len(query_ids) * 0.35), 12),
        sharex=True
    )
    title = f"Benchmark Results — Difficulty: {difficulty}\n"
    title += f"#Iterations per query: {config.NUM_ITERATIONS}"
    title += f"  |  Red number next to query-id = #errors"

    for ax, (metric_name, metric_label) in zip(axes, METRICS):
        data_for_boxes = []
        for qid in query_ids:
            vals = grouped[difficulty][qid]["metrics"].get(metric_name, [])
            # Matplotlib boxplot requires at least one value.
            # If empty, use [nan] so the position still exists without crashing.
            if vals:
                data_for_boxes.append(vals)
            else:
                data_for_boxes.append([float("nan")])

        ax.boxplot(data_for_boxes, positions=x_positions, widths=0.6, showfliers=True)
        ax.set_ylabel(metric_label)
        ax.set_ylim(*ylims[metric_name])
        ax.grid(True, axis="y", linestyle=":", linewidth=0.7)

    # Add error counts below x tick labels on last axis
    add_error_counts_below_ticks(axes[-1], x_positions, error_counts)

    # x-axis formatting on bottom plot
    axes[-1].set_xticks(x_positions)
    axes[-1].set_xticklabels([str(qid) for qid in query_ids], rotation=90)
    axes[-1].set_xlabel("query_id")
    # increase font size for all axis labels and axis tick labels
    for ax in axes:
        for label in (ax.get_xticklabels() + ax.get_yticklabels()):
            label.set_fontsize(18)

    # increase font for title and all axis titles
    fig.suptitle(title, fontsize=18)
    for ax in axes:
        # make it bold
        ax.set_ylabel(ax.get_ylabel(), fontsize=18, fontweight="bold")
        ax.set_xlabel(ax.get_xlabel(), fontsize=18, fontweight="bold")

    plt.tight_layout()
    # plt.show()
    fig.savefig(out_path)
    print(f"Saved plot to: {out_path}")

def iter_aggregated_entries(path: str):
    """
    Stream (key, entry_dict) pairs from a huge top-level JSON object:
      { "google_557": {...}, "google_944": {...}, ... }

    Requires: pip install ijson
    Falls back to json.load if ijson isn't available.
    """
    try:
        import ijson
    except ImportError:
        # Fallback (old behavior, may be slow / memory heavy)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for k, v in data.items():
            yield k, v
        return

    # Streaming parse: iterate over top-level key/value pairs
    with open(path, "rb") as f:
        for k, v in ijson.kvitems(f, ""):
            yield k, v

def load_and_group(path: str):
    """
    Returns:
      grouped[difficulty][query_id]["metrics"][metric_name] -> list[float]
      grouped[difficulty][query_id]["errors"] -> int

    Streaming JSON parsing for huge files.
    """
    grouped = defaultdict(lambda: defaultdict(lambda: {"metrics": defaultdict(list), "errors": 0}))

    for _, entry in iter_aggregated_entries(path):
        difficulty = entry.get("difficulty", "unknown")
        query_id = entry.get("query_id", None)
        if query_id is None:
            continue

        qbucket = grouped[difficulty][int(query_id)]

        # accumulate metrics
        for metric_name, _ in METRICS:
            vals = extract_metric_values(entry, metric_name)
            if vals:
                qbucket["metrics"][metric_name].extend(vals)

        # accumulate errors
        qbucket["errors"] += count_errors(entry)

    return grouped

def debug_one(grouped, difficulty, qid):
    if difficulty not in grouped or qid not in grouped[difficulty]:
        print(f"Not found: {difficulty=} {qid=}")
        return
    bucket = grouped[difficulty][qid]
    print(f"\n=== DEBUG {difficulty} qid={qid} ===")
    print("errors:", bucket["errors"])
    for metric, _ in METRICS:
        vals = bucket["metrics"].get(metric, [])
        print(f"{metric}: n={len(vals)} sample={vals[:5]}")
    print("========================\n")

def plotting(json_path: str):
    out_path = json_path.rsplit(".", 1)[0] + "_grouped.json"

    # only compute if not already done
    if not os.path.exists(out_path) or config.RECOMPUTE_PLOTTING_DATA:
        print("Loading and grouping data...")
        grouped = load_and_group(json_path)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(grouped, f, indent=2)
    else:
        with open(out_path, "r", encoding="utf-8") as f:
            grouped = json.load(f)

    # print("GROUPED: ", grouped)
    # print("\n=== DEBUG: grouped structure ===")
    # for diff in grouped:
    #     qids = sorted(grouped[diff].keys())
    #     print(f"difficulty={diff}  #query_ids={len(qids)}  sample_qids={qids[:10]}")
    # print("================================\n")

    # # Example debug output for one difficulty/query_id
    # debug_one(grouped, "simple", 778)

    ylims = compute_global_ylims(grouped)

    # Plot in a consistent order; include any unknown difficulties at the end
    difficulties_present = list(grouped.keys())
    ordered = [d for d in DIFFICULTY_ORDER if d in grouped] + [
        d for d in difficulties_present if d not in DIFFICULTY_ORDER
    ]

    for diff in ordered:
        plot_difficulty(grouped, diff, ylims, config.PLOTS_FOLDER)

    # save grouped data as JSON for further analysis if needed
    if not os.path.exists(out_path) or config.RECOMPUTE_PLOTTING_DATA:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(grouped, f, indent=2)

    # for report: get query_id, nl_query,golden_query and first sparksql_query entry 
    # sorted based on difficulty and query_id
    report_path = json_path.rsplit(".", 1)[0] + "_report.json"
    if not os.path.exists(report_path) or config.RECOMPUTE_PLOTTING_DATA:
        report_entries = []
        for _, entry in iter_aggregated_entries(json_path):
            report_entry = {
                "query_id": entry.get("query_id"),
                "difficulty": entry.get("difficulty"),
                "nl_query": entry.get("nl_query"),
                "golden_query": entry.get("golden_query"),
                "sparksql_query": None,
                "jaccard_index_new_avg": str(entry.get("jaccard_index_new_avg")),
                "exact_match_avg": str(entry.get("exact_match_avg")),
            }
            sparksql_queries = as_list(entry.get("sparksql_query"))
            if sparksql_queries:
                report_entry["sparksql_query"] = sparksql_queries[0]
            report_entries.append(report_entry)

        # sort by difficulty and query_id
        def sort_key(e):
            diff = e.get("difficulty", "unknown")
            diff_index = DIFFICULTY_ORDER.index(diff) if diff in DIFFICULTY_ORDER else len(DIFFICULTY_ORDER)
            return (diff_index, e.get("query_id", 0))
    
        report_entries.sort(key=sort_key)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report_entries, f, indent=2)
        print(f"Saved report entries to: {report_path}")

    # for report: get minimum, maximum, average total, spark and translation times per difficulty
    time_report_path = json_path.rsplit(".", 1)[0] + "_time_report.json"
    if not os.path.exists(time_report_path) or config.RECOMPUTE_PLOTTING_DATA:
        time_report = {}
        for difficulty in grouped:
            time_report[difficulty] = {}
            for metric_name in ["total_time", "spark_time", "translation_time"]:
                all_vals = []
                for qid in grouped[difficulty]:
                    all_vals.extend(grouped[difficulty][qid]["metrics"].get(metric_name, []))
                if all_vals:
                    time_report[difficulty][metric_name] = {
                        "min": min(all_vals),
                        "max": max(all_vals),
                        "avg": sum(all_vals) / len(all_vals),
                    }
                else:
                    time_report[difficulty][metric_name] = {
                        "min": None,
                        "max": None,
                        "avg": None,
                    }
        with open(time_report_path, "w", encoding="utf-8") as f:
            json.dump(time_report, f, indent=2)
        print(f"Saved time report to: {time_report_path}")  


if __name__ == "__main__":
    if len(sys.argv) > 1:
        plotting(sys.argv[1])
    else:
        print("Usage: python plotting.py <path_to_aggregated_json>")