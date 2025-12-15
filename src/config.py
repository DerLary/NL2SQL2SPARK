DB_PATH = "db"
BENCHMARK_FILE = "dev.json"
DEFAULT_TEMPERATURE = 0.0
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