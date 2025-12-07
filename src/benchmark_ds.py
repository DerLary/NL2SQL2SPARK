from config import (
    DB_PATH,
    BENCHMARK_FILE
)
import os
import json

import sqlite3

#TODO
def load_tables(spark_session, db_name):
    """
    Loads all tables from a SQLite database into a Spark session.

    Args:
        spark_session: Spark session to use for loading tables.
        db_name: Name of the SQLite database file to load tables from.
    """
    pass


def load_query_info(query_id: int):

    query_data_file = os.path.join(DB_PATH, BENCHMARK_FILE)
    with open(query_data_file, 'r') as f:
        all_queries = json.load(f)

    query_info = None
    for query_entry in all_queries:
        if query_entry['question_id'] == query_id:
            query_info = query_entry
            break

    if query_info is None:
        raise ValueError(f"Query ID {query_id} not found")

    database_name = query_info['db_id']
    question = " ".join([
        query_info["question"],
        query_info["evidence"]
    ])
    golden_query = query_info["SQL"]

    return database_name, question, golden_query
