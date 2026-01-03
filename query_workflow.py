import argparse
from datetime import datetime
import json
import uuid
import dotenv
import glob
import random
import config
from aggregation import (
    aggregate_results,
    aggregate_without_new_run
)

from spark_nl import (
    get_spark_session,
    get_spark_sql,
    get_spark_agent,
    run_nl_query,
    process_result,
    print_results,
    pretty_print_cot,
    run_sparksql_query,
    save_results
)
from benchmark_ds import (
    load_tables,
    load_query_info
)
from llm import get_llm
from evaluation import (
    translate_sqlite_to_spark,
    jaccard_index,
    result_to_obj,
    evaluate_spark_sql,
    postprocess_with_new_jaccard_index
)
import os
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_DEBUG"] = "true"
from langchain_core.globals import set_debug, set_verbose

# set_debug(True)     # prints internal LangChain debug info
# set_verbose(True)   # more agent/tool logging

import sys
import logging
from plotting import plotting

class TeeStdout:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


def benchmark_query(query_id, provider, hil_query=None, iteration=1, base_folder="."):

    dotenv.load_dotenv()

    spark_session = get_spark_session()
    additional_data = {}

    database_name, nl_query, golden_query, difficulty = load_query_info(query_id)
    additional_data["nl_query"] = nl_query

    # replace nl_query with hil_query if provided
    if hil_query is not None:
        additional_data = {"used_hil_query": hil_query}
        nl_query = hil_query
    golden_query_spark = translate_sqlite_to_spark(golden_query)
    additional_data["golden_query"] = golden_query_spark

    print(f"--- Benchmarking Query ID {query_id} on Database '{database_name}' ---")

    load_tables(spark_session, database_name)
    spark_sql = get_spark_sql(spark_session)
    llm = get_llm(provider=provider)
    agent, tools = get_spark_agent(spark_sql, llm=llm)
    # print("\n\n\n\n")
    # print("TOOLS: ", tools)
    # print("\n\n\n\n")
    run_nl_query(agent, nl_query, llm=llm, query_id=query_id, tools=tools, spark_session=spark_session, iteration=iteration, difficulty=difficulty)
    json_result = process_result()
    print_results(json_result)
    # pretty_print_cot(json_result)
    
    print(f"NL Query: \033[92m{nl_query}\033[0m")
    print(f"Golden Query (Spark SQL): \033[93m{golden_query_spark}\033[0m")
    # only available if execution was valid -> execution_status

    if json_result["execution_status"] == "VALID":
        ground_truth_df = run_sparksql_query(spark_session, golden_query_spark)
        print("Ground Truth:")
        ground_truth_df.show()
        # serialize DataFrame to list of dicts
        ground_truth_obj = ground_truth_df.collect()
        additional_data["ground_truth"] = [row.asDict() for row in ground_truth_obj]

        # Execution Accuracy
        inferred_result = json_result["query_result"]
        print("Inferred Result:")
        print(inferred_result)
        ea = jaccard_index(ground_truth_df, inferred_result)
        additional_data["jaccard_index"] = ea
        print(f"Jaccard Index: {ea}")
        
        # Structural Accuracy        
        spark_sql_query = json_result.get("sparksql_query")
        if spark_sql_query:
            em_score = evaluate_spark_sql(golden_query_spark, spark_sql_query, spark_session)
            additional_data["exact_match"] = em_score
            print(f"Spider Exact Match Score: {em_score}")


    filename = save_results(json_result, output_file=None, query_id=query_id, iteration=iteration, additional_data=additional_data, base_folder=base_folder)

    return filename

# run 20 different query ids each 10 times and aggregate the results together for future analysis
# the results should be aggregated per query id and should include the number of iteration (so 1 -5), the difficulty based on the query id (easy, medium, hard) and at the end the average execution accuracy and structural accuracy and time should be stored for each query
# the number of iterations should be configurable via command line argument and the query ids should be configurable via the configuration file
def benchmark_queries(provider, iterations=config.NUM_ITERATIONS):
    query_ids = select_random_queries()

    # generate a string consisting of date + random suffix to create a folder name
    current_date = datetime.now().strftime("%Y%m%d")
    random_suffix = str(uuid.uuid4())[:8]
    output_folder = f"benchmark_results_{current_date}_{provider}_{random_suffix}"
    os.makedirs(output_folder, exist_ok=True)

    # do not overwrite, just append with current time as header:
    if os.path.exists(os.path.join(output_folder, "spark_execution.log")):
        print(f"Log file already exists in {output_folder}, appending new log entries.")
        with open(os.path.join(output_folder, "spark_execution.log"), "a") as log_file:
            log_file.write(f"############################################################################\n")
            log_file.write(f"\n\n--- New Benchmark Run at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---\n\n")
            log_file.write(f"############################################################################\n")

    log_file = open(os.path.join(output_folder, "spark_execution.log"), "w")

    sys.stdout = TeeStdout(sys.stdout, log_file)
    sys.stderr = TeeStdout(sys.stderr, log_file)

    # create file and print the selected query ids
    with open(os.path.join(output_folder, "selected_query_ids.json"), "w") as f:
        file_dict = {
            "provider": provider,
            "output_folder": output_folder,
            "iterations": iterations,
            "query_ids": {}
        }
        
        for qid in query_ids:
            database_name, nl_query, golden_query, difficulty = load_query_info(qid)
            file_dict["query_ids"][qid] = {
                "database_name": database_name,
                "difficulty": difficulty,
                "nl_query": nl_query,
                "golden_query": golden_query
            }
        json.dump(file_dict, f, indent=4)

    filenames = []
    for query_id in query_ids:
        # create a new folder per query id inside the output folder
        query_folder = os.path.join(output_folder, str(query_id))
        os.makedirs(query_folder, exist_ok=True)
        for i in range(iterations):
            print(f"--- Benchmarking Query ID {query_id}, Iteration {i+1}/{iterations} ---")
            filenames.append(benchmark_query(query_id, provider, iteration=i+1, base_folder=query_folder))

    # filenames = ["/home/lars/Privat/RWTH/Auslandssemester/#KURSE/Safe Distributed Systems/Exercises/Practical_Exercise/NL2SQL2SPARK/20251230_185115_ID_0_ITER_1_63f0f2f2.json", \
                #  "/home/lars/Privat/RWTH/Auslandssemester/#KURSE/Safe Distributed Systems/Exercises/Practical_Exercise/NL2SQL2SPARK/20251230_185116_ID_0_ITER_2_3f2e9424.json", \
                #  "/home/lars/Privat/RWTH/Auslandssemester/#KURSE/Safe Distributed Systems/Exercises/Practical_Exercise/NL2SQL2SPARK/20251230_185118_ID_2_ITER_1_251ca350.json", \
                #  "/home/lars/Privat/RWTH/Auslandssemester/#KURSE/Safe Distributed Systems/Exercises/Practical_Exercise/NL2SQL2SPARK/20251230_185120_ID_2_ITER_2_f7505643.json"]


def select_random_queries():
    # use database_name, nl_query, golden_query, difficulty = load_query_info(query_id) to load random queries from the benchmark file
    # select NUM_SIMPLE + NUM_MODERATE + NUM_CHALLENGING random query ids from the benchmark file
    query_data_file = os.path.join("db", "dev.json")
    with open(query_data_file, 'r') as f:
        all_queries = json.load(f)
    simple_queries = [q for q in all_queries if q.get("difficulty") == "simple"]
    medium_queries = [q for q in all_queries if q.get("difficulty") == "moderate"]
    hard_queries = [q for q in all_queries if q.get("difficulty") == "challenging"]
    selected_query_ids = []
    simple_queries = random.sample([q["question_id"] for q in simple_queries], config.NUM_SIMPLE)
    moderate_queries = random.sample([q["question_id"] for q in medium_queries], config.NUM_MODERATE)
    challenging_queries = random.sample([q["question_id"] for q in hard_queries], config.NUM_CHALLENGING)
    selected_query_ids += simple_queries + moderate_queries + challenging_queries
    print("Selected Query IDs: ", selected_query_ids)

    return selected_query_ids

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark a specific query ID.")
    parser.add_argument("--id", type=int, default=1, help="Query ID to benchmark (default: 1)")
    parser.add_argument("--provider", type=str, default="google", help="LLM provider (default: google)")
    parser.add_argument("--hil-query", type=str, default=None, help="Optional natural language query to use instead of loading from benchmark.")
    parser.add_argument("--base-folder", type=str, default=None, help="If set, aggregate results from the specified folder without running new benchmarks.")
    parser.add_argument("--only-aggregate", action="store_true", help="If set, only aggregate results without running new benchmarks.")
    parser.add_argument("--run-pipeline", action="store_true", help="If set, run the automated pipeline on predefined query IDs.")
    parser.add_argument("--plotting-json", type=str, default=None, help="If set, generate plots from the specified aggregated results JSON file.")
    args = parser.parse_args()
    
    if args.only_aggregate:
        base_folder = args.base_folder if args.base_folder else "."
        aggregate_without_new_run(base_folder=base_folder)
        exit(0)
    elif args.run_pipeline:
        benchmark_queries(provider=args.provider, iterations=config.NUM_ITERATIONS)
        exit(0)
    elif args.plotting_json is not None:

        aggregated_results_file = args.plotting_json
        plotting(aggregated_results_file)
        exit(0)
    else:
        base_folder = "/home/lars/Privat/RWTH/Auslandssemester/#KURSE/Safe Distributed Systems/Exercises/Practical_Exercise/NL2SQL2SPARK/SINGLE_RUNS"
        benchmark_query(args.id, args.provider, hil_query=args.hil_query, base_folder=base_folder)