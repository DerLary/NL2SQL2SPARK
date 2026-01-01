DB_PATH = "db"
BENCHMARK_FILE = "dev.json"
DEFAULT_TEMPERATURE = 0.0
# DEFAULT_PROMPT_SUFIX = "Always use the Spark SQL tools to execute the query and return the result."
# DEFAULT_PROMPT_SUFIX = "You must respond in exactly ONE of the following formats: 1) To call a tool: Action: <tool_name> \n Action Input: <valid JSON object> 2) If you are done: Final: <your final answer> Do not output anything else. No markdown. No extra text."
DEFAULT_PROMPT_SUFIX = ""
SCHEMA_LOOP_COUNT = 3

from enum import Enum

class Provider(Enum):
    GOOGLE = "google"
    CLOUDFLARE = "cloudflare"


metrics = {
    "total_time": -1,
    "spark_exec_time": -1,
    "translation_time": -1,
    "sparksql_query": None,
    "answer": None
}

DEFAULT_MODELS = {
    Provider.GOOGLE: "gemini-2.5-flash",
    Provider.CLOUDFLARE: '@cf/meta/llama-4-scout-17b-16e-instruct'
}

# AUTOMATED_PIPELINE_IDS_SIMPLE = [0, 2, 8, 17, 18, 19, 22, 53]
# AUTOMATED_PIPELINE_IDS_MODERATE = [1, 4, 12, 23, 25, 27, 32, 33]
# AUTOMATED_PIPELINE_IDS_CHALLENGING = [28, 36, 62, 83, 87, 94, 115, 116]

NUM_SIMPLE = 10
NUM_MODERATE = 10
NUM_CHALLENGING = 10
NUM_ITERATIONS = 10

RAW_RESULTS_FOLDER = "/home/lars/Privat/RWTH/Auslandssemester/#KURSE/Safe Distributed Systems/Exercises/Practical_Exercise/NL2SQL2SPARK/RAW_RESULTS"

FORCE_REAGG = True
MERGE_ALL_RESULTS = True
