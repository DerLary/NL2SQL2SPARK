import os
import config
import glob
import json
from evaluation import postprocess_with_new_jaccard_index, recompute_ground_truth

# after "benchmark_queries" has run we have folders like 
    # benchmark_results_20260101_google_ce2b3070
    # which have subfolders for each query_id like /125/
    # and then contain the individual json result files for each iteration like "20260101_124413_ID_125_ITER_1_5fdcc5f5.json"

# the following functions aggregate these results (all orchestrated in "aggregate_without_new_run")
    # 1. sort_experiments_in_folders: move all json files with the same query_id into a subfolder (for the pipepline this is already done)
    # 2. for each query_id folder, postprocess the json files with the new jaccard index logic (postprocess_with_new_jaccard_index)
    # 3. for early configuration the ground_truth was missing -> compute it now and add it to the json files (recompute_ground_truth)
    # 3. aggregate_results: for each query_id folder, aggregate the results of the different iterations into a single (per QUERY_ID) file "AGG_ID_<query_id>_<parent_folder>.json"
    # 4. aggregate_queryAggregates: for all query_id folders, aggregate the per query_id aggregated files into a single file "AGGREGATED_<parent_folder>.json"
    # -> now we have a single file per benchmark run (e.g., "AGGREGATED_benchmark_results_20260101_google_ce2b3070.json")

# 5. (optional) aggregate_aggregates: aggregate all benchmark runs into a single file "ALL_AGGR_benchmark_results.json"

def sort_experiments_in_folders(base_folder="."):
    # for a given subfolder like "benchmark_results_20260101_google_ce2b3070"
    # move all json file with a common prefix into the same subfolder
    # for exmaple: move benchmark_results_20260101_google_ce2b3070/20260101_124413_ID_1169_ITER_1_5fdcc5f5.json and benchmark_results_20260101_google_ce2b3070/20260101_124413_ID_1169_ITER_2_5fdcc5f5.json
    # in the subfolder benchmark_results_20260101_google_ce2b3070/1169/
    json_files = glob.glob(os.path.join(base_folder, "*_ID_*_ITER_*.json"))
    for json_file in json_files:
        filename = os.path.basename(json_file)
        parts = filename.split("_")
        query_id = parts[3]
        target_folder = os.path.join(base_folder, query_id)
        os.makedirs(target_folder, exist_ok=True)
        target_file = os.path.join(target_folder, filename)
        os.rename(json_file, target_file)

# aggregate the results for a given query_id folder
def aggregate_results(result_files, base_folder="."):
    parent_folder = os.path.basename(os.path.dirname(base_folder))
    query_id = os.path.basename(base_folder)
    output_file = os.path.join(base_folder, f"AGG_ID_{query_id}_{parent_folder}.json")
    
    # only compute if file does not already exist
    if os.path.exists(output_file) and not config.FORCE_REAGG:
        # print(f"[Internal Log] Aggregated results file already exists: {output_file}. Skipping aggregation.")
        return

    # aggregate multiple runs with identical llm, query_id together and calculate average metrics
    aggregated = {}
    for file in result_files:
        try:
            with open(os.path.join(base_folder, file), 'r') as f:
                data = json.load(f)
                key = (data.get("llm"), data.get("query_id"))
                # make a proper string 
                key = str(key[0]) + "_" + str(key[1])
                if key not in aggregated:
                    aggregated[key] = {
                        "llm": data.get("llm"),
                        "query_id": data.get("query_id"),
                        "nl_query": data.get("nl_query"),
                        "golden_query": data.get("golden_query"),
                        "sparksql_query": [],
                        "difficulty": data.get("difficulty"),

                        "execution_status": [],
                        "query_result": [],
                        "ground_truth": [],
                        "used_hil_query": [],
                        "spark_error": [],
                        "total_time": [],
                        "spark_time": [],
                        "translation_time": [],
                        "jaccard_index": [],
                        "jaccard_index_new": [],
                        "exact_match": []
                    }
                aggregated[key]["sparksql_query"].append(data.get("sparksql_query", ""))
                aggregated[key]["query_result"].append(data.get("query_result", []))
                # there are experiments without ground_truth because I only added it later
                aggregated[key]["spark_error"].append(data.get("spark_error", ""))

                aggregated[key]["total_time"].append(data.get("total_time", 0))
                aggregated[key]["spark_time"].append(data.get("spark_time", 0))
                aggregated[key]["translation_time"].append(data.get("translation_time", 0))
                if not "ground_truth" in data and data.get("execution_status", "") != "ERROR":
                    print(f"[Internal Log] No ground_truth in query_id: {data.get('query_id')} \n file: {file}")
                    print("QUERY ID: ", data.get("query_id"))
                else:   
                    aggregated[key]["ground_truth"].append(data.get("ground_truth", []))

                if "used_hil_query" in data:
                    aggregated[key]["used_hil_query"].append(data.get("used_hil_query", []))

                aggregated[key]["execution_status"].append(data.get("execution_status", ""))
                # only if execution_status = VALID
                if "jaccard_index" in data:
                    aggregated[key]["jaccard_index"].append(data["jaccard_index"])
                    aggregated[key]["jaccard_index_new"].append(data["jaccard_index_new"])
                if "exact_match" in data:
                    aggregated[key]["exact_match"].append(data["exact_match"]) 
        except Exception as e:
            print(f"[Internal Log] Error processing file {file}: {e}")
            raise e

    # Now calculate averages
    for key in aggregated:
        for metric in ["total_time", "spark_time", "translation_time"]:
            values = aggregated[key][metric]
            aggregated[key][metric+"_avg"] = sum(values) / len(values) if values else 0
        
        for metric in ["jaccard_index", "jaccard_index_new", "exact_match"]:
            values = aggregated[key][metric]
            aggregated[key][metric+"_avg"] = sum(values) / len(values) if values else None
    
    with open(output_file, 'w') as f:
        # append current time to filename
        json.dump(aggregated, f, indent=4)

    # print(f"[Internal Log] Aggregated results query_id {query_id}saved to {output_file}")

def merge_json_files(base_folder, file_pattern="AGG_ID_*.json"):
    merged_results = {}

    aggregated_files = glob.glob(os.path.join(base_folder, "*", file_pattern))

    def as_list(x):
        """Wrap scalars into a list; pass lists through unchanged."""
        return x if isinstance(x, list) else [x]

    # produce a final json by merging the separate json files together
    for agg_file in aggregated_files:
        with open(agg_file, "r") as f:
            data = json.load(f)

            for key, value in data.items():
                # should be the case when this model-query_id combination was only run in one benchmark run
                if key not in merged_results:
                    merged_results[key] = value
                    merged_results[key]["source_file"] = [agg_file]
                else:
                    for sub_key, sub_value in value.items():
                        try: 
                            if sub_key not in merged_results[key]:
                                merged_results[key][sub_key] = sub_value
                            else:
                                merged_results[key]["duplicate_source"] = [agg_file]
                                # if not a list -> make it a list 
                                if isinstance(merged_results[key][sub_key], list):
                                    merged_results[key][sub_key].extend(sub_value)
                                else:
                                    old_value = merged_results[key][sub_key]
                                    merged_results[key][sub_key] = as_list(old_value) + as_list(sub_value)
                                    
                        except Exception as e:
                            print(f"[Internal Log] Error merging key {key} sub_key {sub_key} from file {agg_file}: {e}")
                            raise e
    return merged_results

# Aggregate all "AGG_ID_<query_id>_<parent_folder>.json" files (per query_id files) in one file per benchmark run
# for base folders like "benchmark_results_20251231_google_d598f842" collect all aggregated files "benchmark_results_20251231_google_d598f842/1532/AGG_ID_1532_benchmark_results_20251231_google_d598f842.json"
# and produce a final file "AGGREGATED_benchmark_results_20251231_google_d598f842.json" in the base folder
# that contains all aggregated results (by merging the per query_id aggregated results)
def aggregate_queryAggregates(base_folder="."):
    output_file = os.path.join(base_folder, f"AGGREGATED_{os.path.basename(base_folder)}.json")
    if os.path.exists(output_file) and not config.FORCE_REAGG:
        print(f"[Internal Log] Aggregated results file {output_file} already exists, skipping aggregation. Use FORCE_REAGG = True to override.")
        return

    merged_results = merge_json_files(base_folder, file_pattern="AGG_ID_*.json")
    # save merged results
    with open(output_file, "w") as f:
        json.dump(merged_results, f, indent=4)
    print(f"[Internal Log] Aggregated results over all per-query_id folders saved to \n {output_file}")

# merge the aggregated files per benchmark run (AGGREGATED_*.json) into a single file (ALL_AGGR_*.json)
# merge all files like "COPY_benchmark_results_20260101_google_ce2b3070/AGGREGATED_COPY_benchmark_results_20260101_google_ce2b3070.json"
# in folders starting with "benchmark_results_" into a single file "AGGREGATED_ALL_benchmark_results.json" in the current folder# 
def aggregate_aggregates(base_folder="."):

    output_file = os.path.join(base_folder, f"ALL_AGGR_{os.path.basename(base_folder)}.json")
    if os.path.exists(output_file) and not config.FORCE_REAGG:
        print(f"[Internal Log] ALL Aggregated results file {output_file} already exists, skipping aggregation. Use FORCE_REAGG = True to override.")
        return

    merged_results = merge_json_files(base_folder, file_pattern="AGGREGATED_*.json")

    aggregated_files = glob.glob(os.path.join(base_folder, "*", "AGG_ID_*.json"))
    # produce a final json by merging the separate json files together
    for agg_file in aggregated_files:
        with open(agg_file, "r") as f:
            data = json.load(f)
            for key, value in data.items():
                if key not in merged_results:
                    merged_results[key] = value
                else:
                    for sub_key, sub_value in value.items():
                        if sub_key not in merged_results[key]:
                            merged_results[key][sub_key] = sub_value
                        else:
                            merged_results[key][sub_key].extend(sub_value)
    # save merged results
    with open(output_file, "w") as f:
        json.dump(merged_results, f, indent=4)
    print(f"[Internal Log] ALL aggregated results saved to {output_file}")


# aggregate already existing results without running new benchmarks
def aggregate_without_new_run(base_folder="."):
    folders_to_analyze = []
    # if no base_folder given -> aggregate all results per folder and then all together
    if base_folder == ".":
        benchmark_folders = [f.path for f in os.scandir(config.RAW_RESULTS_FOLDER) if f.is_dir() and f.name.startswith("benchmark_results_")]
        folders_to_analyze.extend(benchmark_folders)
    else:
        folders_to_analyze.append(base_folder)

    for folder in folders_to_analyze:
        base_folder = folder
        # sort experiments with the same query_id into the same folder
        sort_experiments_in_folders(base_folder=folder)

        # for all query_id subfolders, postprocess the json files with the new jaccard index logic
        query_id_folders = [f.path for f in os.scandir(folder) if f.is_dir()]
        for qfolder in query_id_folders:
            json_files = glob.glob(os.path.join(qfolder, "*_ID_*_ITER_*.json"))
            for json_file in json_files:
                recompute_ground_truth(json_file)
                postprocess_with_new_jaccard_index(json_file)
            # aggregate results on the query_id folder
            filenames = glob.glob(os.path.join(qfolder, "*_ID_*_ITER_*.json"))
            aggregate_results(filenames, base_folder=qfolder)

        aggregate_queryAggregates(base_folder=folder)
    if config.MERGE_ALL_RESULTS:
        aggregate_aggregates(base_folder=config.RAW_RESULTS_FOLDER)
    return

