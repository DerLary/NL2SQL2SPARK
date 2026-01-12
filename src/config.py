DB_PATH = "db"
BENCHMARK_FILE = "dev.json"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_PROMPT_SUFIX = ""
SCHEMA_LOOP_COUNT = 5

NUM_SIMPLE = 10
NUM_MODERATE = 10
NUM_CHALLENGING = 10
NUM_ITERATIONS = 10

RAW_RESULTS_FOLDER = "/home/lars/Privat/RWTH/Auslandssemester/#KURSE/Safe Distributed Systems/Exercises/Practical_Exercise/NL2SQL2SPARK/RAW_RESULTS"
BASE_FOLDER_SINGLE_RUNS = "/home/lars/Privat/RWTH/Auslandssemester/#KURSE/Safe Distributed Systems/Exercises/Practical_Exercise/NL2SQL2SPARK/SINGLE_RUNS"

PLOTS_FOLDER = "/home/lars/Privat/RWTH/Auslandssemester/#KURSE/Safe Distributed Systems/Exercises/Practical_Exercise/NL2SQL2SPARK/PLOTS"

FORCE_REAGG = False
MERGE_ALL_RESULTS = True

RECOMPUTE_PLOTTING_DATA = False
RECOMPUTE_PLOTS = False

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