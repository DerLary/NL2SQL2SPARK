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
    db_dir = os.path.join(DB_PATH, db_name)
    sqlite_path = os.path.join(db_dir, f"{db_name}.sqlite")

    if not os.path.exists(sqlite_path):
        raise FileNotFoundError(f"SQLite database not found: {sqlite_path}")

    # connect to SQLite
    conn = sqlite3.connect(sqlite_path)
    cursor = conn.cursor()

    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table';"
    )
    tables = [row[0] for row in cursor.fetchall()]

    def quote_sqlite_ident(name: str) -> str:
        # SQLite uses double quotes for identifiers; escape embedded quotes
        # query_id 134:
            #   problem: a table name is "order" which is a reserved keyword in SQL
        # query_id 168:
            #   problem: pyspark.errors.exceptions.captured.AnalysisException: [DATATYPE_MISMATCH.UNEXPECTED_INPUT_TYPE] Cannot resolve "sum((gender = F))" due to data type mismatch: The first parameter requires the "NUMERIC" or "ANSI INTERVAL" type, however "(gender = F)" has the type "BOOLEAN"
        return '"' + name.replace('"', '""') + '"'
    
    # query_id 1272 / 523:
        # problem: java.sql.SQLException: Error parsing date
        # Unparseable date: "1993-02-10" does not match (\p{Nd}++)\Q-\E(\p{Nd}++)\Q-\E(\p{Nd}++)\Q \E(\p{Nd}++)\Q:\E(\p{Nd}++)\Q:\E(\p{Nd}++)\Q.\E(\p{Nd}++) 
    def is_date_like(sqlite_decl_type: str) -> bool:
        if not sqlite_decl_type:
            return False
        t = sqlite_decl_type.strip().upper()
        # Be conservative + schema-agnostic: only act on explicit date-ish declarations
        return ("DATE" in t) or ("DATETIME" in t) or ("TIMESTAMP" in t)


    # load tables into Spark
    for table_name in tables:
        # get table schema to identify date-like columns
        cursor.execute(f"PRAGMA table_info({quote_sqlite_ident(table_name)});")

        cols = cursor.fetchall()
        if not cols:
            continue

        select_exprs = []
        for _, col_name, col_type, *_ in cols:
            qcol = quote_sqlite_ident(col_name)
            if is_date_like(col_type):
                # Prevent SQLite JDBC driver from calling getDate()/getTimestamp()
                select_exprs.append(f"CAST({qcol} AS TEXT) AS {qcol}")
            else:
                select_exprs.append(f"{qcol}")

        qtable = quote_sqlite_ident(table_name)
        select_sql = f"SELECT {', '.join(select_exprs)} FROM {qtable}"
        # Spark JDBC expects dbtable either as a table name or a parenthesized subquery + alias
        dbtable = f"({select_sql}) AS {qtable}"
            
        # dbtable = quote_sqlite_ident(table_name)

        df = spark_session.read \
            .format("jdbc") \
            .option("url", f"jdbc:sqlite:{sqlite_path}") \
            .option("dbtable", dbtable) \
            .option("driver", "org.sqlite.JDBC") \
            .load()

        # register as Spark SQL table
        df.createOrReplaceTempView(table_name)

    conn.close()


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
    difficulty = query_info.get("difficulty")

    return database_name, question, golden_query, difficulty
