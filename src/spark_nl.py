import os
import types
import time
import json
import datetime
import uuid

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from pyspark.sql import SparkSession
from langchain_core.callbacks import BaseCallbackHandler
from langchain_cloudflare import ChatCloudflareWorkersAI

import pandas as pd
import config
from evaluation import result_to_obj
from llm import get_cloudflare_neuron_pricing
from spark_toolkit.toolkit import SparkSQLToolkit
from spark_toolkit.base import create_spark_sql_agent
from spark_toolkit.spark_sql import SparkSQL

import json
import re

def parse_react_action(text: str):
    """
    Parse:
    Action: tool_name
    Action Input: {...}
    """
    action_match = re.search(r"Action:\s*(\w+)", text)
    input_match = re.search(r"Action Input:\s*(\{.*\})", text, re.DOTALL)

    if not action_match or not input_match:
        return None, None

    tool = action_match.group(1)
    params = json.loads(input_match.group(1))
    return tool, params

class AgentEarlyExit(BaseException):
    def __init__(self, answer):
        self.answer = answer


class AgentLoopException(BaseException):
    pass


class AgentMonitoringCallback(BaseCallbackHandler):
    def __init__(self):
        self.count = 0
        self.chain_of_thought = []
        self.input_tokens = 0
        self.output_tokens = 0
        self.last_prompt = None
        self.last_answer = None
        self.schema_sql_db_count = 0

    def on_llm_start(self, serialized, prompts, **kwargs):
        self.count += 1
        if prompts:
            self.last_prompt = prompts[0]

    def on_llm_end(self, response, **kwargs):

        if hasattr(response, "generations"):
            for g in response.generations:
                for gen in g:
                    if hasattr(gen, "text"):
                        self.last_answer = gen.text
                    
                    gen_dict = gen.__dict__
                    if "message" in gen_dict:
                        message_dict = gen_dict["message"].__dict__
                        self.input_tokens += message_dict["usage_metadata"]["input_tokens"]
                        self.output_tokens += message_dict["usage_metadata"]["output_tokens"]

    def on_agent_action(self, action, **kwargs):
        log_message = action.log
        self.chain_of_thought.append(log_message)
        print(f"\n[Real-time CoT] {log_message}")

        if action.tool == "schema_sql_db":
            self.schema_sql_db_count += 1
            if self.schema_sql_db_count > config.SCHEMA_LOOP_COUNT:
                raise AgentLoopException("Loop detected: schema_sql_db called too many times")

    def on_tool_end(self, output, **kwargs):
        message = f"Observation: {output}"
        self.chain_of_thought.append(message)
        print(f"\n[Real-time CoT] {message}")
        
    def on_agent_finish(self, finish, **kwargs):
        log_message = finish.log
        self.chain_of_thought.append(log_message)
        print(f"\n[Real-time CoT] {log_message}")

    def on_tool_start(self, serialized, input_str, **kwargs):
        tool_name = serialized.get("name") if serialized else "unknown"
        message = f"Action: {tool_name}\nAction Input: {input_str}"
        self.chain_of_thought.append(message)
        print(f"\n[Real-time CoT] {message}")
        
        if tool_name == "schema_sql_db":
            self.schema_sql_db_count += 1
            if self.schema_sql_db_count > config.SCHEMA_LOOP_COUNT:
                raise AgentLoopException("Loop detected: schema_sql_db called too many times")


def parsing_error_handler(error: Exception):
    str_error = str(error)
    # Check if this is the specific parsing error we want to catch
    if "Could not parse LLM output:" in str_error:
        print(f"[Internal Log] Parsing error detected. Asking LLM to retry...")
        return f"An output parsing error occurred. Please ensure you are using the correct format. Error: {str_error}"

    return f"Agent Error: {str_error}"


#TODO
def get_spark_session():
    """
    Creates a Spark session with SQLite access.
    """
    jar_path = "/usr/share/java/xerial-sqlite-jdbc-3.44.1.0.jar"
    spark = SparkSession.builder \
        .master("local[*]") \
        .appName("NL2SQL2Spark") \
        .config("spark.jars", jar_path) \
        .getOrCreate()
    return spark


def get_schema_manually(self, table_names):
    all_schemas = []

    if not table_names:
        table_names = [t.name for t in self._spark.catalog.listTables()]

    for table in table_names:
        try:
            df = self._spark.table(table)
            columns = ", ".join(
                [f"{f.name} {f.dataType.simpleString()}" for f in df.schema]
            )
            all_schemas.append(f"CREATE TABLE {table} ({columns});")
        except Exception:
            pass

    return "\n\n".join(all_schemas)


def get_spark_sql(_spark_session):

    spark_sql = SparkSQL(schema=None)
    spark_sql.get_table_info = types.MethodType(get_schema_manually, spark_sql)
    return spark_sql
# def get_spark_sql(spark_session):
#     spark_sql = SparkSQL(spark_session=spark_session, schema=None)
#     spark_sql.get_table_info = types.MethodType(get_schema_manually, spark_sql)
#     return spark_sql


#TODO
def run_sparksql_query(spark_session, query):
    """
    Runs a Spark SQL query on a given Spark session.

    Args:
        spark_session: Spark session to run the query on.
        query: A string with the Spark SQL query to run.
    """
    # df = spark_session.sql(query)
    # pd_df = df.toPandas()
    # return pd_df
    return spark_session.sql(query)


def get_spark_agent(spark_sql, llm):

    original_run = spark_sql.run

    def timed_run(self, command, fetch="all"):

        start_t = time.time()
        
        # Log to chain of thought if callback is attached
        if hasattr(self, 'cb') and self.cb:
            self.cb.chain_of_thought.append(f"Spark Query Executed: {command}")

        try:
            result = original_run(command, fetch)
            error = None
        except Exception as e:
            result = None
            error = str(e)

        end_t = time.time()
        duration = end_t - start_t

        config.metrics["query"] = command
        config.metrics["spark_time"] = duration
        config.metrics["result"] = result if error is None else None
        config.metrics["spark_error"] = error

        print(f"\n[Create_Agent_Internal_Log] Spark Query Executed in {duration:.4f}s")
        print("Query:", command)
        print("Result/Error:", result if error is None else error)

        # FORCE EARLY EXIT IMMEDIATELY AFTER FIRST SPARK QUERY
        if error:
            raise AgentEarlyExit(f"[SPARK ERROR]\n{error}")
        else:
            raise AgentEarlyExit(f"[SPARK RESULT]\n{result}")

    spark_sql.run = types.MethodType(timed_run, spark_sql)
    toolkit = SparkSQLToolkit(db=spark_sql, llm=llm)
    agent, tools = create_spark_sql_agent(
        llm=llm,
        toolkit=toolkit, verbose=True,
        handle_parsing_errors=parsing_error_handler
    )

    return agent, tools

def schema_context(spark):
    out = []
    for t in spark.catalog.listTables():
        name = t.name
        cols = spark.table(name).schema
        col_str = ", ".join([f"{c.name}:{c.dataType.simpleString()}" for c in cols])
        out.append(f"- {name}({col_str})")
    print("\n\n\n")
    # print("[Internal Log] Available tables/views:")
    # for line in out:
    #     print("  ", line)
    # print("\n\n\n")
    return "Available tables/views:\n" + "\n".join(out)

def run_nl_query(agent, nl_query, llm=None, query_id=None, tools=None, spark_session=None, iteration=1, difficulty=None):

    print("--- Starting Agent ---")
    total_start = time.time()
    
    cb = AgentMonitoringCallback()
    
    # Attach callback to the db object so timed_run can access it
    # agent.tools is a list of tools. We need to find the one with the db.
    # Usually SparkSQLToolkit adds tools that share the same db instance.
    if hasattr(agent, 'tools'):
        for tool in agent.tools:
            if hasattr(tool, 'db'):
                tool.db.cb = cb
                break

    try:
        nl_query = nl_query + "\n\n" + config.DEFAULT_PROMPT_SUFIX
        from langchain_core.messages import HumanMessage
        response = agent.invoke({"messages": [HumanMessage(content=nl_query)]}, config={"callbacks": [cb]})
        final_answer = response["messages"][-1].content

        # for cloudflare: explicitly state which tables are available
        # nl_query = (
        #     schema_context(spark_session)
        #     + "\n\nQuestion:\n"
        #     + nl_query
        #     + "\n\nIMPORTANT: Use ONLY the tables listed above. Do not invent table/column names."
        #     + "\n\n"
        #     + config.DEFAULT_PROMPT_SUFIX
        # )

        print("\n=== RAW AGENT RESPONSE ===")
        print(response)
        print("=== RAW AGENT MESSAGES ===")
        for m in response.get("messages", []):
            print(type(m), getattr(m, "content", None))
        print("==========================\n")
        # response = agent.invoke({"messages": [HumanMessage(content=nl_query)]})
        # msg = response["messages"][-1]
        # text = msg.content

        # # parse response to extract tool calls for cloudflare
        # tool_name, tool_args = parse_react_action(text)
        # print(f"[ReAct Adapter] Parsed tool: {tool_name}, args: {tool_args}")
        # # print("agent.tools:", [t.name for t in agent.tools])

        # if tool_name:
        #     print(f"[ReAct Adapter] Executing tool: {tool_name}")
        #     tool_map = {t.name: t for t in tools}
        #     if tool_name not in tool_map:
        #         raise ValueError(f"Tool '{tool_name}' not found. Available: {list(tool_map)}")

        #     result = tool_map[tool_name].invoke(tool_args)

        #     # store metrics like before
        #     config.metrics["query"] = tool_args.get("query")
        #     config.metrics["result"] = result
        #     config.metrics["spark_time"] = 0.0

        #     final_answer = result
        # else:
        #     final_answer = text

        print("Final answer:", final_answer )

    except AgentEarlyExit as e:
        print("--- Exit Triggered (Parsing Bypass) ---")
        print(e)
        final_answer = e.answer

    except AgentLoopException as e:
        print("--- Loop Detected ---")
        print(e)
        final_answer = str(e)
        config.metrics["query"] = None

    except Exception as e:
        print("--- Agent Error Occurred ---")
        print(str(e))
        final_answer = str(e)

    total_end = time.time()

    llm_name = ""
    if llm and isinstance(llm, ChatOpenAI) or isinstance(llm, ChatCloudflareWorkersAI):
        llm_name = "cloudflare"
    else:
        if isinstance(llm, ChatGoogleGenerativeAI):
             llm_name = "google"
        else:
            llm_name = "unknown"     

    config.metrics["llm"] = llm_name  
    config.metrics["answer"] = final_answer
    total_time = total_end - total_start
    config.metrics["total_time"] = total_time
    spark_time = config.metrics.get("spark_time", total_time)
    config.metrics["translation_time"] = total_time - spark_time
    config.metrics["llm_requests"] = cb.count
    config.metrics["chain_of_thought"] = cb.chain_of_thought
    config.metrics["input_tokens"] = cb.input_tokens
    config.metrics["output_tokens"] = cb.output_tokens
    config.metrics["prompt"] = cb.last_prompt
    config.metrics["final_answer"] = cb.last_answer
    config.metrics["query_id"] = query_id
    config.metrics["iteration"] = iteration
    config.metrics["difficulty"] = difficulty

    # Calculate Cloudflare Neurons if applicable
    if llm and isinstance(llm, ChatCloudflareWorkersAI):
        model_name = llm.model
        pricing = get_cloudflare_neuron_pricing(model_name)
        if pricing:
            input_neurons = (cb.input_tokens / 1_000_000) * pricing["input_neurons_per_m"]
            output_neurons = (cb.output_tokens / 1_000_000) * pricing["output_neurons_per_m"]
            total_neurons = input_neurons + output_neurons
            config.metrics["cloudflare_neurons"] = total_neurons
            print(f"[Cloudflare Cost] Estimated Neurons: {total_neurons:.2f}")


def process_result():
    result = config.metrics.get("result", None)
    result = result_to_obj(result)
    error = config.metrics.get("spark_error", None)
    
    json_result = {
        "llm": config.metrics.get("llm", None),
        "sparksql_query": config.metrics.get("query", None),
        "query_id": config.metrics.get("query_id", None),
        "iteration": config.metrics.get("iteration", None),
        "difficulty": config.metrics.get("difficulty", None),
        "execution_status": "ERROR" if error else ("VALID" if config.metrics.get("query", None) else "NOT_EXECUTED"),
        "query_result": result,
        "spark_error": error,
        "total_time": config.metrics.get("total_time", -1),
        "spark_time": config.metrics.get("spark_time", -1),
        "translation_time": config.metrics.get("translation_time", -1),
        "llm_requests": config.metrics.get("llm_requests", 0),
        "chain_of_thought": config.metrics.get("chain_of_thought", []),
        "input_tokens": config.metrics.get("input_tokens", 0),
        "output_tokens": config.metrics.get("output_tokens", 0),
        "cloudflare_neurons": config.metrics.get("cloudflare_neurons", None),
        "prompt": config.metrics.get("prompt", None),
        "final_answer": config.metrics.get("final_answer", None)
    }
    
    return json_result


def print_results(json_result, print_result=False):
    print("\n" + "="*40)
    print(" PERFORMANCE METRICS")
    print("="*40)
    
    total_time = json_result.get("total_time")
    spark_time = json_result.get("spark_time")
    translation_time = json_result.get("translation_time")
    
    status = json_result.get('execution_status')
    color_start = ""
    color_end = "\033[0m"

    if status == "VALID":
        color_start = "\033[92m"  # Green
    elif status == "ERROR":
        color_start = "\033[91m"  # Red
    elif status == "NOT_EXECUTED":
        color_start = "\033[93m"  # Yellow

    print(f"Execution Status: {color_start}{status}{color_end}")
    print(f"1. Total End-to-End Time    : {total_time:.4f} sec" if total_time is not None and total_time != -1 else "1. Total End-to-End Time    : N/A")
    print(f"2. Spark Execution Time     : {spark_time:.4f} sec" if spark_time is not None and spark_time != -1 else "2. Spark Execution Time     : N/A")
    print(f"3. Input Translation (LLM)  : {translation_time:.4f} sec" if translation_time is not None and translation_time != -1 else "3. Input Translation (LLM) Time  : N/A")
    print(f"4. LLM Requests             : {json_result.get('llm_requests')}")
    print(f"5. Input Tokens             : {json_result.get('input_tokens')}")
    print(f"6. Output Tokens            : {json_result.get('output_tokens')}")
    
    neurons = json_result.get('cloudflare_neurons')
    if neurons is not None:
        print(f"7. Cloudflare Neurons       : {neurons:.2f}")
    
    print(f"Spark Query: {color_start}{json_result.get('sparksql_query')}{color_end}")
    
    error = json_result.get("spark_error")
    print(f"Spark Error (first 50 chars): {error[:50] if error else 'None'}")
    print("="*40)
    
    if json_result.get('execution_status') == "VALID" and print_result:
        print(f"Query Result: {json_result.get('query_result')}")


def pretty_print_cot(json_result):
    print("\n" + "="*40)
    print(" CHAIN OF THOUGHT")
    print("="*40)
    
    cot = json_result.get("chain_of_thought", [])
    if not cot:
        print("No Chain of Thought available.")
        return

    for step in cot:
        print(step)
        print("-" * 20)
    print("="*40)


def save_results(results, output_file=None, query_id=None, iteration=1, additional_data=None, base_folder="."):
    if output_file is None:
        current_date = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        random_suffix = str(uuid.uuid4())[:8]
        output_file = f"{current_date}_ID_{query_id}_ITER_{iteration}_{random_suffix}.json"
    print(f"[Internal Log] Saving results to {output_file}")
        
    if additional_data:
        results.update(additional_data)

    with open(os.path.join(base_folder, output_file), 'w') as f:
        json.dump(results, f, indent=4)

    return os.path.join(base_folder, output_file)