from unittest import result
import dotenv
from llm import get_llm
from spark_nl import get_spark_session, get_spark_sql, get_spark_sql, run_sparksql_query
from benchmark_ds import load_tables, load_query_info
from evaluation import translate_sqlite_to_spark, jaccard_index, evaluate_spark_sql
import os
import config
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_DEBUG"] = "true"
import json
from explainer_metrics import analyze_execution_plans

# adapted from Marc Sanchez-Artigas
# https://github.com/Artigas81/NL2SparkSQL/blob/main/sqlexplain.py
def display_execution_plan(spark, sql_query, query_name=""):
    output = []  # collect all strings here

    def write(line=""):
        output.append(line)

    write(f"\nExecution Plan for: {query_name if query_name else sql_query[:50]}...")
    write("=" * 80)

    # Get the DataFrame without executing it
    try:
        df = spark.sql(sql_query)
    except Exception as e:
        write(f"Could not create DataFrame for the query: {e}")
        return "\n".join(output)

    write("\n1. Logical plan:")
    write("-" * 40)
    try:
        logical_plan = df._jdf.queryExecution().logical().toString()
        write(logical_plan)
    except Exception as e:
        write(f"Could not display logical plan: {e}")

    write("\n2. Optimized logical plan (After Catalyst AFAIK):")
    write("-" * 40)
    try:
        optimized_plan = df._jdf.queryExecution().optimizedPlan().toString()
        write(optimized_plan)
    except Exception as e:
        write(f"Could not display optimized plan: {e}")

    write("\n3. Physical plan:")
    write("-" * 40)
    try:
        physical_plan = df._jdf.queryExecution().executedPlan().toString()
        # Format for better readability
        lines = physical_plan.split('\n')
        for line in lines:
            if line.strip():
                indent = len(line) - len(line.lstrip())
                indent_str = "  " * (indent // 2)
                write(f"{indent_str}{line.lstrip()}")
    except Exception as e:
        write(f"Could not display physical plan: {e}")

    # 4. Show Simple EXPLAIN output (traditional SQL style)
    write("\n4. Explain output:")
    write("-" * 40)
    try:
        explain_df = spark.sql(f"EXPLAIN {sql_query}")
        explain_result = explain_df.collect()[0][0]

        sections = explain_result.split("== ")
        for section in sections:
            if section.strip():
                lines = section.strip().split('\n')
                header = lines[0] if lines else ""
                if header:
                    write(f"\n{header}:")
                    write("-" * (len(header) + 1))
                    for line in lines[1:]:
                        if line.strip():
                            write(f"  {line.strip()}")
    except Exception as e:
        write(f"Could not display explain output: {e}")

    write("=" * 80)

    return "\n".join(output)

def display_exectution_plan_to_file(spark, sql_query, query_name="", file_output=None):
    # when file already exists, load and return it, otherwise create it
    if file_output is not None and os.path.exists(file_output) and not config.RECOMPUTE_EXPLAIN_DATA:
        with open(file_output, 'r') as f:
            plan = f.read()
        print("Loaded existing execution plan from file:", file_output)
        return plan
    else:
        with open(file_output, 'w') as f:
            plan = display_execution_plan(spark, sql_query, query_name)
            f.write(plan)
        print("Wrote execution plan to file:", file_output)
        return plan

def extract_results_from_json_dict(json_result_file):
    dotenv.load_dotenv()

    # read json result file
    with open(json_result_file, 'r') as f:
        json_result = json.load(f)

    # when "nl_query" is present -> this is the first (initial) query
    # when "used_hil_query" is present -> this is the modified query after HIL
    is_initial = "nl_query" in json_result

    result_dict = {
        "query_id": json_result.get("query_id"),
        "difficulty": json_result.get("difficulty"),
        "is_initial": is_initial,
        "nl_query": json_result.get("used_hil_query") or json_result.get("nl_query"),
        "golden_query": json_result.get("golden_query"),
        "sparksql_query": json_result.get("sparksql_query"),
        "jaccard_index": json_result.get("jaccard_index_new"),
        "exact_match": json_result.get("exact_match"),
        "ground_truth": json_result.get("ground_truth"),
        "query_result": json_result.get("query_result"),
        "execution_plans": {},
        "result_file_path": json_result_file
    }

    return result_dict

def execute_query(json_result_file):
    dotenv.load_dotenv()

    result_dict = extract_results_from_json_dict(json_result_file)
    query_id = result_dict["query_id"]

    # check if execution plans are already present
    base_dir = os.path.dirname(json_result_file)
    # use the same prefix as the original file RAW_RESULTS/benchmark_results_20251231_google_d598f842/30/20251231_120509_ID_30_ITER_10_51a5255e.json
    prefix = os.path.basename(json_result_file).split(".json")[0]
    prefix_parts = prefix.split("_")
    iteration_index = prefix_parts.index("ITER") + 1
    used_prefix = "_".join(prefix_parts[:iteration_index + 1])
    base_dir = config.BASE_FOLDER_EXPLAIN
    query_id_folder = os.path.join(base_dir, str(query_id))

    output_json_file = os.path.join(query_id_folder, f"{used_prefix}_with_explain_{query_id}.json")
    
    if os.path.exists(output_json_file) and not config.RECOMPUTE_EXPLAIN_DATA:
        print(f"Execution plans already exist for query ID {query_id} at {output_json_file}. Skipping execution.")
        return

    os.makedirs(query_id_folder, exist_ok=True)

    golden_query_spark = result_dict["golden_query"]
    model_query = result_dict["sparksql_query"]

    spark_session = get_spark_session()
    database_name, _, _, _ = load_query_info(query_id)
    
    load_tables(spark_session, database_name)

    plan_gold = display_exectution_plan_to_file(spark_session, golden_query_spark, query_name=f"Golden Query ID {query_id}", file_output=os.path.join(query_id_folder, f"{used_prefix}_golden_query_plan_{query_id}.txt"))
    plan_model = display_exectution_plan_to_file(spark_session, model_query, query_name=f"Model Query ID {query_id}", file_output=os.path.join(query_id_folder, f"{used_prefix}_model_query_plan_{query_id}.txt"))
    
    result_dict["execution_plans"]["golden_query"] = plan_gold
    result_dict["execution_plans"]["model_query"] = plan_model

    plan_metrics = analyze_execution_plans(result_dict)
    result_dict["explainer_metrics"] = plan_metrics

    with open(output_json_file, 'w') as f:
        json.dump(result_dict, f, indent=4)
    print(f"Wrote final results with explainer metrics to: {output_json_file}")

def has_broadcast_joins(explainer_json_file):
    # open json file
    if explainer_json_file is None or not os.path.exists(explainer_json_file):
        print("Explainer JSON file not provided or does not exist.")
    else:
        with open(explainer_json_file, 'r') as f:
            results = json.load(f)

        has_golden = results.get("explainer_metrics", {}).get("golden_query", {}).get("has_broadcast_join", False)
        has_model = results.get("explainer_metrics", {}).get("model_query", {}).get("has_broadcast_join", False)
        # parse json "false/true" to boolean
        if has_golden or has_model:
            print(f"Query ID {results.get('query_id')} has broadcast join(s): Golden={has_golden}, Model={has_model}")
            return explainer_json_file

if __name__ == "__main__":
    # simple (134):
    # json_result_file = "/home/lars/Privat/RWTH/Auslandssemester/#KURSE/Safe Distributed Systems/Exercises/Practical_Exercise/NL2SQL2SPARK/SINGLE_RUNS_J/134/20251231_124816_ID_134_ITER_1_4cc109de.json"
    # moderate (892):
    # json_result_file = "/home/lars/Privat/RWTH/Auslandssemester/#KURSE/Safe Distributed Systems/Exercises/Practical_Exercise/NL2SQL2SPARK/RAW_RESULTS/benchmark_results_20251231_google_0d954987/892/20251231_133804_ID_892_ITER_1_9b4356c7.json"
    # challenging (219):
    json_result_file = "/home/lars/Privat/RWTH/Auslandssemester/#KURSE/Safe Distributed Systems/Exercises/Practical_Exercise/NL2SQL2SPARK/SINGLE_RUNS/219/20260110_175926_ID_219_ITER_1_9f901d09.json"

    execute_query(json_result_file)


