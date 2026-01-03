import ast
import json
import re
import sqlglot
import pandas as pd
import pyspark.sql
from sqlglot import exp
from pyspark.sql import functions as F
import os 

#TODO
def translate_sqlite_to_spark(sqlite_query):
    """
    Transpiles a SQLite query to Spark SQL.

    Args:
        sqlite_query: A String with the SQLite query to transpile.
    """
    if not sqlite_query or not isinstance(sqlite_query, str):
        return sqlite_query

    q = sqlite_query.strip().rstrip(";")

    try:
        spark_sql = sqlglot.transpile(q, read="sqlite", write="spark")[0]
        # return sqlglot.transpile(q, read="sqlite", write="spark")[0]
        # query_id 168:
            #   problem: pyspark.errors.exceptions.captured.AnalysisException: [DATATYPE_MISMATCH.UNEXPECTED_INPUT_TYPE] Cannot resolve "sum((gender = F))" due to data type mismatch: The first parameter requires the "NUMERIC" or "ANSI INTERVAL" type, however "(gender = F)" has the type "BOOLEAN"
        
    except Exception:
        # return original if sqlglot can't parse/transpile it
        return q
    
    # parse the Spark SQL into an AST for transformations
    try:
        tree = sqlglot.parse_one(spark_sql, read="spark")
    except Exception:
        return spark_sql

    # 1) Fix SUM(<boolean predicate>) which works in SQLite (0/1) but not in Spark.
    # replace: SUM(predicate) -> SUM(CASE WHEN predicate THEN 1 ELSE 0 END)
    def rewrite_sum_of_predicate(node: exp.Expression) -> exp.Expression:
        if isinstance(node, exp.Sum):
            arg = node.this
            # sqlglot: comparisons/boolean logic are Expressions like EQ, GT, And, Or, Like, In, etc.
            predicate_types = (
                exp.EQ, exp.NEQ, exp.GT, exp.GTE, exp.LT, exp.LTE,
                exp.And, exp.Or, exp.Not,
                exp.Like, exp.ILike,
                exp.In, exp.Between,
                exp.Is, exp.Paren,
            )
            # manual translation of predicate -> CASE WHEN predicate THEN 1 ELSE 0 END
            if isinstance(arg, predicate_types):
                case_expr = exp.Case(
                    ifs=[exp.When(this=arg, true=exp.Literal.number(1))], default=exp.Literal.number(0),
                )
                return exp.Sum(this=case_expr)
        return node

    tree = tree.transform(rewrite_sum_of_predicate)

    return tree.sql(dialect="spark")

def result_to_obj(s):
    if s and isinstance(s, str):
        try:
            parsed = json.loads(s)
        except Exception:
            try:
                parsed = ast.literal_eval(s)
            except Exception:
                parsed = [{"value": s}]
        result = parsed
    else:
        result = s

    return result


#TODO
def jaccard_index(df1, df2):

    """
    Calculates the Jaccard index between two dataframes.

    Args:
        df1: The first dataframe.
        df2: The second dataframe.
    """
    def to_row_set(x):
        # 1. Spark DataFrames
        if pyspark and isinstance(x, pyspark.sql.dataframe.DataFrame):
            cols = sorted(x.columns)

            # problem query_id 1169: column name is a SQL expression:
                # pyspark.errors.exceptions.captured.AnalysisException: [UNRESOLVED_COLUMN.WITH_SUGGESTION] A column, variable, or function parameter with name 
                    #   `(CAST(sum(CASE WHEN ((UA <= 8`.`0) AND (SEX = M)) THEN 1 ELSE 0 END) AS FLOAT) / sum(CASE WHEN ((UA <= 6`.`5) AND (SEX = F)) THEN 1 ELSE 0 END))` 
                    #   cannot be resolved. Did you mean one of the following? ...;
            def safe_col(c):
            # quote if it contains chars Spark treats specially
                if any(ch in c for ch in " .()/-+*'\"`"):
                    return F.col(f"`{c}`").alias(c)
                return F.col(c)

            return {tuple(row) for row in x.select([safe_col(c) for c in cols]).collect()}
            # return {tuple(row) for row in x.select(*cols).collect()}

        # 2. Pandas DataFrames (Your Test Case)
        if isinstance(x, pd.DataFrame):
            # sort to ensure consistent tuple structure
            cols = sorted(x.columns)
            return {tuple(row) for row in x[cols].itertuples(index=False, name=None)}

        # 3. List of Dicts / Results (Agent Output)
        if isinstance(x, list):
            out = set()
            for item in x:
                if isinstance(item, dict):
                    # sort by key to match the sorted columns above
                    out.add(tuple(item[k] for k in sorted(item.keys())))
                elif isinstance(item, (list, tuple)):
                    out.add(tuple(item))
                else:
                    out.add((item,))
            return out

        # 4. Fallback for scalars (strings/ints)
        return { (str(x),) }

    s1 = to_row_set(df1)
    s2 = to_row_set(df2)

    union = s1 | s2
    if not union:
        return 1.0

    return len(s1 & s2) / len(union)

import pandas as pd

# problem with above: numbers are returned as e.g., "3.0" (string) and not 3 (number) 
# alternative: treat values as equal when their string representation matches
def jaccard_index_new(df1, df2):
    """
    Calculates the Jaccard index between two dataframes.
    - Converts ALL values to strings (including numbers and None) -> because the agent output may have everything as strings (at least for google)
    - Converts Spark DataFrames to list-of-dicts using column names.

    Args:
        df1: The first dataframe.
        df2: The second dataframe.
    """

    def to_str(v):
        if v is None:
            return "NULL"
        return str(v)

    def rowdict_to_tuple(d: dict):
        # sort keys so ordering doesn't matter
        return tuple((k, to_str(d[k])) for k in sorted(d.keys()))

    def to_row_set(x):
        # 1) Spark DataFrame -> list-of-dicts -> canonical tuples
        try:
            import pyspark
            from pyspark.sql.dataframe import DataFrame as SparkDF
        except Exception:
            pyspark = None
            SparkDF = None

        if SparkDF is not None and isinstance(x, SparkDF):
            cols = x.columns
            # collect Rows, convert to dicts keyed by column names
            rows = x.collect()
            dict_rows = []
            for r in rows:
                rd = r.asDict(recursive=True)
                # ensure only the dataframe columns and in consistent form
                dict_rows.append({c: to_str(rd.get(c)) for c in cols})
            return {rowdict_to_tuple(d) for d in dict_rows}

        # 2) Pandas DataFrame -> list-of-dicts -> canonical tuples
        if isinstance(x, pd.DataFrame):
            dict_rows = x.to_dict(orient="records")
            # make all values strings
            dict_rows = [{k: to_str(v) for k, v in d.items()} for d in dict_rows]
            return {rowdict_to_tuple(d) for d in dict_rows}

        # 3) List handling
        if isinstance(x, list):
            out = set()
            for item in x:
                # list of dicts (already keyed)
                if isinstance(item, dict):
                    out.add(rowdict_to_tuple({k: to_str(v) for k, v in item.items()}))

                # list of lists/tuples (NOT keyed) -> treat as positional columns: c0,c1,...
                elif isinstance(item, (list, tuple)):
                    d = {f"c{i}": to_str(v) for i, v in enumerate(item)}
                    out.add(rowdict_to_tuple(d))

                else:
                    # scalar element in list
                    d = {"c0": to_str(item)}
                    out.add(rowdict_to_tuple(d))
            return out

        # 4) Scalar fallback
        return {rowdict_to_tuple({"c0": to_str(x)})}

    s1 = to_row_set(df1)
    s2 = to_row_set(df2)

    union = s1 | s2
    if not union:
        return 1.0
    return len(s1 & s2) / len(union)

# process the NL2SQL json file again to compute the ground_truth if it is missing (for earlier configuration runs)
def recompute_ground_truth(json_file):
    from spark_nl import compute_only_golden_query_result 
    
    with open(json_file, "r") as f:
        data = json.load(f)
    if data.get("execution_status", "") == "ERROR":
        # print("Skipping file with ERROR status: ", json_file)
        return
    ground_truth = data.get("ground_truth", None)
    
    if ground_truth is None:
        golden_query = data.get("golden_query", None)
        if golden_query is None:
            print("No golden_query found to compute ground_truth in the json file.", json_file)
            raise ValueError("No golden_query found to compute ground_truth")
        # compute the ground_truth
        ground_truth = compute_only_golden_query_result(json_file)
        data["ground_truth"] = ground_truth
        with open(json_file, "w") as f:
            json.dump(data, f, indent=4)


# process the NL2SQL json file again to compute the jaccard index again based on the new logic
def postprocess_with_new_jaccard_index(json_file):
    # input: json object with 
    # "query_result": [
    #     [
    #         "Sebastian",
    #         "Vettel",
    #         "397.0"
    #     ]
    # ],
    # and "ground_truth": [
    #     {
    #         "forename": "Sebastian",
    #         "surname": "Vettel",
    #         "points": 397.0
    #     }
    # ],"
    # print("HANDLING FILE: ", json_file, "\n ")

    with open(json_file, "r") as f:
        data = json.load(f)
    if data.get("execution_status", "") == "ERROR":
        # print("Skipping file with ERROR status: ", json_file)
        return
    ground_truth = data.get("ground_truth", None)
    query_result = data.get("query_result", None)
    if query_result is None:
        print("No query_result found in the json file.", json_file)
        return
    if ground_truth is None:
        print("No ground_truth found in the json file.", json_file)
        return

    # convert ground_truth to dataframe
    # ground_truth is list of dicts with 1 key; convert to a 1-col DF named c0
    ground_truth_df = pd.DataFrame(ground_truth)
    if not ground_truth_df.empty:
        ground_truth_df = ground_truth_df.iloc[:, :].copy()
        ground_truth_df.columns = [f"c{i}" for i in range(ground_truth_df.shape[1])]

    # convert query_result to dataframe
    # build with its own number of columns
    ncols = len(query_result[0]) if query_result and len(query_result) > 0 else 0
    query_result_df = pd.DataFrame(query_result, columns=[f"c{i}" for i in range(ncols)])

    # compute jaccard index
    jaccard = jaccard_index_new(ground_truth_df, query_result_df)
    # print(f"Jaccard Index: {jaccard}")
    # print("GROUND TRUTH DF:")
    # print(ground_truth_df)
    # print("QUERY RESULT DF:")
    # print(query_result_df)

    # save the jaccard index back to the json file
    data["jaccard_index_new"] = jaccard

    with open(json_file, "w") as f:
        json.dump(data, f, indent=4)

# -----------------------------------------------------------------------------
# Spider Evaluation Logic (adapted from https://github.com/taoyds/spider)
# -----------------------------------------------------------------------------

CLAUSE_KEYWORDS = ('select', 'from', 'where', 'group', 'order', 'limit', 'intersect', 'union', 'except')
JOIN_KEYWORDS = ('join', 'on', 'as', 'inner', 'cross', 'left', 'right', 'full', 'semi', 'anti', 'outer')

WHERE_OPS = ('not', 'between', '=', '>', '<', '>=', '<=', '!=', 'in', 'like', 'is', 'exists')
UNIT_OPS = ('none', '-', '+', "*", '/')
AGG_OPS = ('none', 'max', 'min', 'count', 'sum', 'avg')
TABLE_TYPE = {
    'sql': "sql",
    'table_unit': "table_unit",
}

COND_OPS = ('and', 'or')
SQL_OPS = ('intersect', 'union', 'except')
ORDER_OPS = ('desc', 'asc')

DISABLE_VALUE = True
DISABLE_DISTINCT = True

class Schema:

    def __init__(self, schema):
        self._schema = schema
        self._idMap = self._map(self._schema)

    @property
    def schema(self):
        return self._schema

    @property
    def idMap(self):
        return self._idMap

    def _map(self, schema):
        idMap = {'*': "__all__"}
        id = 1
        for key, vals in schema.items():
            for val in vals:
                idMap[key.lower() + "." + val.lower()] = "__" + key.lower() + "." + val.lower() + "__"
                id += 1

        for key in schema:
            idMap[key.lower()] = "__" + key.lower() + "__"
            id += 1

        return idMap


def get_schema(spark, db_name="default"):

    schema = {}
    if db_name == "default":
        db_prefix = ""
    else:
        db_prefix = f"{db_name}."
    try:
        tables = spark.catalog.listTables(db_name)
        for t in tables:
            table_name = t.name
            columns = spark.catalog.listColumns(f"{db_prefix}{table_name}")
            schema[table_name.lower()] = [c.name.lower() for c in columns]
    except Exception as e:
        print(f"Error getting schema: {e}")
    return schema


def tokenize(string):
    string = str(string)
    string = string.replace("\'", "\"") 
    quote_idxs = [idx for idx, char in enumerate(string) if char == '"']
    assert len(quote_idxs) % 2 == 0, "Unexpected quote"

    vals = {}
    for i in range(len(quote_idxs)-1, -1, -2):
        qidx1 = quote_idxs[i-1]
        qidx2 = quote_idxs[i]
        val = string[qidx1: qidx2+1]
        key = "__val_{}_{}__".format(qidx1, qidx2)
        string = string[:qidx1] + key + string[qidx2+1:]
        vals[key] = val

    toks = [t for t in re.split(r"(\s+|[(),;=<>!*/+-]|`[^`]+`)", string) if t and not t.strip() == '']
    
    cleaned_toks = []
    for t in toks:
        t = t.strip()
        if not t:
            continue

        if t.startswith('`') and t.endswith('`'):
            t = t[1:-1]
        cleaned_toks.append(t.lower())
    
    toks = cleaned_toks

    for i in range(len(toks)):
        if toks[i] in vals:
            toks[i] = vals[toks[i]]

    eq_idxs = [idx for idx, tok in enumerate(toks) if tok == "="]
    eq_idxs.reverse()
    prefix = ('!', '>', '<')
    for eq_idx in eq_idxs:
        if eq_idx > 0:
            pre_tok = toks[eq_idx-1]
            if pre_tok in prefix:
                toks = toks[:eq_idx-1] + [pre_tok + "="] + toks[eq_idx+1: ]

    return toks


def scan_alias(toks):
    as_idxs = [idx for idx, tok in enumerate(toks) if tok == 'as']
    alias = {}
    for idx in as_idxs:
        if idx + 1 < len(toks) and idx - 1 >= 0:
            alias[toks[idx+1]] = toks[idx-1]
    return alias


def get_tables_with_alias(schema, toks):
    tables = scan_alias(toks)
    for idx, tok in enumerate(toks):
        if tok in schema:
            if idx + 1 < len(toks):
                next_tok = toks[idx+1]
                if (next_tok not in CLAUSE_KEYWORDS and 
                    next_tok not in JOIN_KEYWORDS and 
                    next_tok not in [',', ')', ';', '(', '.'] and
                    next_tok not in schema):
                    tables[next_tok] = tok

    for key in schema:
        assert key not in tables, "Alias {} has the same name in table".format(key)
        tables[key] = key
    return tables


def parse_col(toks, start_idx, tables_with_alias, schema, default_tables=None):

    tok = toks[start_idx]
    
    if tok == 'case':
        idx = start_idx
        idx += 1
        depth = 1
        while idx < len(toks) and depth > 0:
            if toks[idx] == 'case':
                depth += 1
            elif toks[idx] == 'end':
                depth -= 1
            idx += 1
        return idx, " ".join(toks[start_idx:idx])

    if tok == "*":
        return start_idx + 1, schema.idMap[tok]

    if '.' in tok:
        parts = tok.split('.')
        if len(parts) == 2 and parts[1] == '':
             # Case: "T1." followed by "Col" (likely due to backticks splitting)
             alias = parts[0]
             col = toks[start_idx+1]
             key = tables_with_alias[alias] + "." + col
             return start_idx+2, schema.idMap[key]

        alias, col = parts
        key = tables_with_alias[alias] + "." + col
        return start_idx+1, schema.idMap[key]

    assert default_tables is not None and len(default_tables) > 0, "Default tables should not be None or empty"

    for alias in default_tables:
        table = tables_with_alias[alias]
        if tok in schema.schema[table]:
            key = table + "." + tok
            return start_idx+1, schema.idMap[key]

    return start_idx+1, tok


def parse_col_unit(toks, start_idx, tables_with_alias, schema, default_tables=None):

    idx = start_idx
    len_ = len(toks)
    isBlock = False
    isDistinct = False
    if toks[idx] == '(':
        isBlock = True
        idx += 1

    if toks[idx] in AGG_OPS:
        agg_id = AGG_OPS.index(toks[idx])
        idx += 1
        assert idx < len_ and toks[idx] == '('
        idx += 1
        if toks[idx] == "distinct":
            idx += 1
            isDistinct = True
        idx, col_id = parse_col(toks, idx, tables_with_alias, schema, default_tables)
        assert idx < len_ and toks[idx] == ')'
        idx += 1
        return idx, (agg_id, col_id, isDistinct)

    if toks[idx] == "distinct":
        idx += 1
        isDistinct = True
    agg_id = AGG_OPS.index("none")
    idx, col_id = parse_col(toks, idx, tables_with_alias, schema, default_tables)

    if isBlock:
        assert toks[idx] == ')'
        idx += 1  # skip ')'

    return idx, (agg_id, col_id, isDistinct)


def parse_val_unit(toks, start_idx, tables_with_alias, schema, default_tables=None):
    idx = start_idx
    len_ = len(toks)
    isBlock = False
    if toks[idx] == '(':
        isBlock = True
        idx += 1

    col_unit1 = None
    col_unit2 = None
    unit_op = UNIT_OPS.index('none')

    idx, col_unit1 = parse_col_unit(toks, idx, tables_with_alias, schema, default_tables)
    if idx < len_ and toks[idx] in UNIT_OPS:
        unit_op = UNIT_OPS.index(toks[idx])
        idx += 1
        idx, col_unit2 = parse_col_unit(toks, idx, tables_with_alias, schema, default_tables)

    if isBlock:
        assert toks[idx] == ')'
        idx += 1  # skip ')'

    return idx, (unit_op, col_unit1, col_unit2)


def parse_table_unit(toks, start_idx, tables_with_alias, schema):

    idx = start_idx
    len_ = len(toks)
    key = tables_with_alias[toks[idx]]

    if idx + 1 < len_ and toks[idx+1] == "as":
        idx += 3
    elif idx + 1 < len_ and toks[idx+1] in tables_with_alias and tables_with_alias[toks[idx+1]] == key:
        idx += 2
    else:
        idx += 1

    return idx, schema.idMap[key], key


def parse_value(toks, start_idx, tables_with_alias, schema, default_tables=None):
    idx = start_idx
    len_ = len(toks)

    isBlock = False
    if toks[idx] == '(':
        isBlock = True
        idx += 1

    if toks[idx] == 'select':
        idx, val = parse_sql(toks, idx, tables_with_alias, schema)
    elif "\"" in toks[idx]:
        val = toks[idx]
        idx += 1
    elif toks[idx] == 'null':
        val = None
        idx += 1
    else:
        try:
            val = float(toks[idx])
            idx += 1
        except:
            end_idx = idx
            while end_idx < len_ and toks[end_idx] != ',' and toks[end_idx] != ')'\
                and toks[end_idx] != 'and' and toks[end_idx] not in CLAUSE_KEYWORDS and toks[end_idx] not in JOIN_KEYWORDS:
                    end_idx += 1

            idx, val = parse_col_unit(toks[start_idx: end_idx], 0, tables_with_alias, schema, default_tables)
            idx = end_idx

    if isBlock:
        assert toks[idx] == ')'
        idx += 1

    return idx, val


def parse_condition(toks, start_idx, tables_with_alias, schema, default_tables=None):
    idx = start_idx
    len_ = len(toks)
    conds = []

    while idx < len_:
        leading_not = False
        if toks[idx] == 'not':
            leading_not = True
            idx += 1

        val_unit = None
        val1_is_col = True
        try:
            idx, val_unit = parse_val_unit(toks, idx, tables_with_alias, schema, default_tables)
        except AssertionError as e:
            if "Error col" in str(e):
                idx, val_unit = parse_value(toks, idx, tables_with_alias, schema, default_tables)
                val1_is_col = False
            else:
                raise e
        not_op = False
        if toks[idx] == 'not':
            not_op = True
            idx += 1
        
        if leading_not:
            not_op = not not_op

        assert idx < len_ and toks[idx] in WHERE_OPS, "Error condition: idx: {}, tok: {}".format(idx, toks[idx])
        op_id = WHERE_OPS.index(toks[idx])
        idx += 1
        
        if toks[idx-1] == 'is' and idx < len_ and toks[idx] == 'not':
            not_op = not not_op
            idx += 1

        val1 = val2 = None
        if op_id == WHERE_OPS.index('between'):
            idx, val1 = parse_value(toks, idx, tables_with_alias, schema, default_tables)
            assert toks[idx] == 'and'
            idx += 1
            idx, val2 = parse_value(toks, idx, tables_with_alias, schema, default_tables)
        else:
            idx, val1 = parse_value(toks, idx, tables_with_alias, schema, default_tables)
            val2 = None

            if not val1_is_col and isinstance(val1, tuple):
                val_rhs = val_unit
                val_unit = (0, val1, None)
                val1 = val_rhs

                op_str = WHERE_OPS[op_id]
                if op_str == '>': new_op = '<'
                elif op_str == '<': new_op = '>'
                elif op_str == '>=': new_op = '<='
                elif op_str == '<=': new_op = '>='
                else: new_op = op_str
                op_id = WHERE_OPS.index(new_op)

        conds.append((not_op, op_id, val_unit, val1, val2))

        if idx < len_ and (toks[idx] in CLAUSE_KEYWORDS or toks[idx] in (")", ";") or toks[idx] in JOIN_KEYWORDS):
            break

        if idx < len_ and toks[idx] in COND_OPS:
            conds.append(toks[idx])
            idx += 1

    return idx, conds


def parse_select(toks, start_idx, tables_with_alias, schema, default_tables=None):
    idx = start_idx
    len_ = len(toks)

    assert toks[idx] == 'select', "'select' not found"
    idx += 1
    isDistinct = False
    if idx < len_ and toks[idx] == 'distinct':
        idx += 1
        isDistinct = True
    val_units = []

    while idx < len_ and toks[idx] not in CLAUSE_KEYWORDS:
        agg_id = AGG_OPS.index("none")
        if toks[idx] in AGG_OPS:
            agg_id = AGG_OPS.index(toks[idx])
            idx += 1
        idx, val_unit = parse_val_unit(toks, idx, tables_with_alias, schema, default_tables)
        
        if idx < len_ and toks[idx] == 'as':
            idx += 2

        val_units.append((agg_id, val_unit))
        if idx < len_ and toks[idx] == ',':
            idx += 1

    return idx, (isDistinct, val_units)


def parse_from(toks, start_idx, tables_with_alias, schema):
    """
    Assume in the from clause, all table units are combined with join
    """
    assert 'from' in toks[start_idx:], "'from' not found"

    len_ = len(toks)
    idx = toks.index('from', start_idx) + 1
    default_tables = []
    table_units = []
    conds = []

    while idx < len_:
        isBlock = False
        if toks[idx] == '(':
            isBlock = True
            idx += 1

        if toks[idx] == 'select':
            idx, sql = parse_sql(toks, idx, tables_with_alias, schema)
            table_units.append((TABLE_TYPE['sql'], sql))
        else:

            while idx < len_ and toks[idx] in ('inner', 'cross', 'left', 'right', 'full', 'semi', 'anti', 'outer'):
                idx += 1

            if idx < len_ and toks[idx] == 'join':
                idx += 1
            idx, table_unit, table_name = parse_table_unit(toks, idx, tables_with_alias, schema)
            table_units.append((TABLE_TYPE['table_unit'],table_unit))
            default_tables.append(table_name)
        if idx < len_ and toks[idx] == "on":
            idx += 1
            idx, this_conds = parse_condition(toks, idx, tables_with_alias, schema, default_tables)
            if len(conds) > 0:
                conds.append('and')
            conds.extend(this_conds)

        if isBlock:
            assert toks[idx] == ')'
            idx += 1
        if idx < len_ and (toks[idx] in CLAUSE_KEYWORDS or toks[idx] in (")", ";")):
            break

    return idx, table_units, conds, default_tables


def parse_where(toks, start_idx, tables_with_alias, schema, default_tables):
    idx = start_idx
    len_ = len(toks)

    if idx >= len_ or toks[idx] != 'where':
        return idx, []

    idx += 1
    idx, conds = parse_condition(toks, idx, tables_with_alias, schema, default_tables)
    return idx, conds


def parse_group_by(toks, start_idx, tables_with_alias, schema, default_tables):
    idx = start_idx
    len_ = len(toks)
    col_units = []

    if idx >= len_ or toks[idx] != 'group':
        return idx, col_units

    idx += 1
    assert toks[idx] == 'by'
    idx += 1

    while idx < len_ and not (toks[idx] in CLAUSE_KEYWORDS or toks[idx] in (")", ";")):
        idx, col_unit = parse_col_unit(toks, idx, tables_with_alias, schema, default_tables)
        col_units.append(col_unit)
        if idx < len_ and toks[idx] == ',':
            idx += 1  # skip ','
        else:
            break

    return idx, col_units


def parse_order_by(toks, start_idx, tables_with_alias, schema, default_tables):
    idx = start_idx
    len_ = len(toks)
    val_units = []
    order_type = 'asc'

    if idx >= len_ or toks[idx] != 'order':
        return idx, val_units

    idx += 1
    assert toks[idx] == 'by'
    idx += 1

    while idx < len_ and not (toks[idx] in CLAUSE_KEYWORDS or toks[idx] in (")", ";")):
        idx, val_unit = parse_val_unit(toks, idx, tables_with_alias, schema, default_tables)
        val_units.append(val_unit)
        if idx < len_ and toks[idx] in ORDER_OPS:
            order_type = toks[idx]
            idx += 1
        if idx < len_ and toks[idx] == ',':
            idx += 1  # skip ','
        else:
            break

    return idx, (order_type, val_units)


def parse_having(toks, start_idx, tables_with_alias, schema, default_tables):
    idx = start_idx
    len_ = len(toks)

    if idx >= len_ or toks[idx] != 'having':
        return idx, []

    idx += 1
    idx, conds = parse_condition(toks, idx, tables_with_alias, schema, default_tables)
    return idx, conds


def parse_limit(toks, start_idx):
    idx = start_idx
    len_ = len(toks)

    if idx < len_ and toks[idx] == 'limit':
        idx += 2
        return idx, int(toks[idx-1])

    return idx, None


def skip_semicolon(toks, start_idx):
    idx = start_idx
    while idx < len(toks) and toks[idx] == ";":
        idx += 1
    return idx


def parse_sql(toks, start_idx, tables_with_alias, schema):
    isBlock = False
    len_ = len(toks)
    idx = start_idx

    sql = {}
    if toks[idx] == '(':
        isBlock = True
        idx += 1

    from_end_idx, table_units, conds, default_tables = parse_from(toks, start_idx, tables_with_alias, schema)
    sql['from'] = {'table_units': table_units, 'conds': conds}
    _, select_col_units = parse_select(toks, idx, tables_with_alias, schema, default_tables)
    idx = from_end_idx
    sql['select'] = select_col_units
    idx, where_conds = parse_where(toks, idx, tables_with_alias, schema, default_tables)
    sql['where'] = where_conds
    idx, group_col_units = parse_group_by(toks, idx, tables_with_alias, schema, default_tables)
    sql['groupBy'] = group_col_units
    idx, having_conds = parse_having(toks, idx, tables_with_alias, schema, default_tables)
    sql['having'] = having_conds
    idx, order_col_units = parse_order_by(toks, idx, tables_with_alias, schema, default_tables)
    sql['orderBy'] = order_col_units
    idx, limit_val = parse_limit(toks, idx)
    sql['limit'] = limit_val

    idx = skip_semicolon(toks, idx)
    if isBlock:
        assert toks[idx] == ')'
        idx += 1  # skip ')'
    idx = skip_semicolon(toks, idx)

    for op in SQL_OPS:
        sql[op] = None
    if idx < len_ and toks[idx] in SQL_OPS:
        sql_op = toks[idx]
        idx += 1
        idx, IUE_sql = parse_sql(toks, idx, tables_with_alias, schema)
        sql[sql_op] = IUE_sql
    return idx, sql


def get_sql(schema, query):
    toks = tokenize(query)
    tables_with_alias = get_tables_with_alias(schema.schema, toks)
    _, sql = parse_sql(toks, 0, tables_with_alias, schema)

    return sql

# -----------------------------------------------------------------------------
# Evaluator Class
# -----------------------------------------------------------------------------

def get_scores(count, pred_total, label_total):
    if pred_total != label_total:
        return 0,0,0
    elif count == pred_total:
        return 1,1,1
    return 0,0,0

def eval_sel(pred, label):
    pred_sel = pred['select'][1]
    label_sel = label['select'][1]
    label_wo_agg = [unit[1] for unit in label_sel]
    pred_total = len(pred_sel)
    label_total = len(label_sel)
    cnt = 0
    cnt_wo_agg = 0

    for unit in pred_sel:
        if unit in label_sel:
            cnt += 1
            label_sel.remove(unit)
        if unit[1] in label_wo_agg:
            cnt_wo_agg += 1
            label_wo_agg.remove(unit[1])

    return label_total, pred_total, cnt, cnt_wo_agg

def eval_where(pred, label):
    pred_conds = [unit for unit in pred['where'][::2]]
    label_conds = [unit for unit in label['where'][::2]]
    label_wo_agg = [unit[2] for unit in label_conds]
    pred_total = len(pred_conds)
    label_total = len(label_conds)
    cnt = 0
    cnt_wo_agg = 0

    for unit in pred_conds:
        if unit in label_conds:
            cnt += 1
            label_conds.remove(unit)
        if unit[2] in label_wo_agg:
            cnt_wo_agg += 1
            label_wo_agg.remove(unit[2])

    return label_total, pred_total, cnt, cnt_wo_agg

def eval_group(pred, label):
    pred_cols = [unit[1] for unit in pred['groupBy']]
    label_cols = [unit[1] for unit in label['groupBy']]
    pred_total = len(pred_cols)
    label_total = len(label_cols)
    cnt = 0
    pred_cols = [pred.split(".")[1] if "." in pred else pred for pred in pred_cols]
    label_cols = [label.split(".")[1] if "." in label else label for label in label_cols]
    for col in pred_cols:
        if col in label_cols:
            cnt += 1
            label_cols.remove(col)
    return label_total, pred_total, cnt

def eval_having(pred, label):
    pred_total = label_total = cnt = 0
    if len(pred['groupBy']) > 0:
        pred_total = 1
    if len(label['groupBy']) > 0:
        label_total = 1

    pred_cols = [unit[1] for unit in pred['groupBy']]
    label_cols = [unit[1] for unit in label['groupBy']]
    if pred_total == label_total == 1 \
            and pred_cols == label_cols \
            and pred['having'] == label['having']:
        cnt = 1

    return label_total, pred_total, cnt

def eval_order(pred, label):
    pred_total = label_total = cnt = 0
    if len(pred['orderBy']) > 0:
        pred_total = 1
    if len(label['orderBy']) > 0:
        label_total = 1
    if len(label['orderBy']) > 0 and pred['orderBy'] == label['orderBy'] and \
            ((pred['limit'] is None and label['limit'] is None) or (pred['limit'] is not None and label['limit'] is not None)):
        cnt = 1
    return label_total, pred_total, cnt

def eval_and_or(pred, label):
    pred_ao = pred['where'][1::2]
    label_ao = label['where'][1::2]
    pred_ao = set(pred_ao)
    label_ao = set(label_ao)

    if pred_ao == label_ao:
        return 1,1,1
    return len(pred_ao),len(label_ao),0

def get_nestedSQL(sql):
    nested = []
    for cond_unit in sql['from']['conds'][::2] + sql['where'][::2] + sql['having'][::2]:
        if type(cond_unit[3]) is dict:
            nested.append(cond_unit[3])
        if type(cond_unit[4]) is dict:
            nested.append(cond_unit[4])
    if sql['intersect'] is not None:
        nested.append(sql['intersect'])
    if sql['except'] is not None:
        nested.append(sql['except'])
    if sql['union'] is not None:
        nested.append(sql['union'])
    return nested

def eval_nested(pred, label):
    label_total = 0
    pred_total = 0
    cnt = 0
    if pred is not None:
        pred_total += 1
    if label is not None:
        label_total += 1
    if pred is not None and label is not None:
        cnt += Evaluator().eval_exact_match(pred, label)
    return label_total, pred_total, cnt

def eval_IUEN(pred, label):
    lt1, pt1, cnt1 = eval_nested(pred['intersect'], label['intersect'])
    lt2, pt2, cnt2 = eval_nested(pred['except'], label['except'])
    lt3, pt3, cnt3 = eval_nested(pred['union'], label['union'])
    label_total = lt1 + lt2 + lt3
    pred_total = pt1 + pt2 + pt3
    cnt = cnt1 + cnt2 + cnt3
    return label_total, pred_total, cnt

def get_keywords(sql):
    res = set()
    if len(sql['where']) > 0:
        res.add('where')
    if len(sql['groupBy']) > 0:
        res.add('group')
    if len(sql['having']) > 0:
        res.add('having')
    if len(sql['orderBy']) > 0:
        res.add(sql['orderBy'][0])
        res.add('order')
    if sql['limit'] is not None:
        res.add('limit')
    if sql['except'] is not None:
        res.add('except')
    if sql['union'] is not None:
        res.add('union')
    if sql['intersect'] is not None:
        res.add('intersect')

    # or keyword
    ao = sql['from']['conds'][1::2] + sql['where'][1::2] + sql['having'][1::2]
    if len([token for token in ao if token == 'or']) > 0:
        res.add('or')

    cond_units = sql['from']['conds'][::2] + sql['where'][::2] + sql['having'][::2]
    # not keyword
    if len([cond_unit for cond_unit in cond_units if cond_unit[0]]) > 0:
        res.add('not')

    # in keyword
    if len([cond_unit for cond_unit in cond_units if cond_unit[1] == WHERE_OPS.index('in')]) > 0:
        res.add('in')

    # like keyword
    if len([cond_unit for cond_unit in cond_units if cond_unit[1] == WHERE_OPS.index('like')]) > 0:
        res.add('like')

    return res

def eval_keywords(pred, label):
    pred_keywords = get_keywords(pred)
    label_keywords = get_keywords(label)
    pred_total = len(pred_keywords)
    label_total = len(label_keywords)
    cnt = 0

    for k in pred_keywords:
        if k in label_keywords:
            cnt += 1
    return label_total, pred_total, cnt


class Evaluator:
    """A simple evaluator"""
    def __init__(self):
        self.partial_scores = None

    def eval_exact_match(self, pred, label):
        partial_scores = self.eval_partial_match(pred, label)
        self.partial_scores = partial_scores

        for _, score in partial_scores.items():
            if score['f1'] != 1:
                return 0
        if len(label['from']['table_units']) > 0:
            label_tables = sorted(label['from']['table_units'])
            pred_tables = sorted(pred['from']['table_units'])
            return 1 if label_tables == pred_tables else 0
        return 1

    def eval_partial_match(self, pred, label):
        res = {}

        label_total, pred_total, cnt, cnt_wo_agg = eval_sel(pred, label)
        acc, rec, f1 = get_scores(cnt, pred_total, label_total)
        res['select'] = {'acc': acc, 'rec': rec, 'f1': f1,'label_total':label_total,'pred_total':pred_total}
        acc, rec, f1 = get_scores(cnt_wo_agg, pred_total, label_total)
        res['select(no AGG)'] = {'acc': acc, 'rec': rec, 'f1': f1,'label_total':label_total,'pred_total':pred_total}

        label_total, pred_total, cnt, cnt_wo_agg = eval_where(pred, label)
        acc, rec, f1 = get_scores(cnt, pred_total, label_total)
        res['where'] = {'acc': acc, 'rec': rec, 'f1': f1,'label_total':label_total,'pred_total':pred_total}
        acc, rec, f1 = get_scores(cnt_wo_agg, pred_total, label_total)
        res['where(no OP)'] = {'acc': acc, 'rec': rec, 'f1': f1,'label_total':label_total,'pred_total':pred_total}

        label_total, pred_total, cnt = eval_group(pred, label)
        acc, rec, f1 = get_scores(cnt, pred_total, label_total)
        res['group(no Having)'] = {'acc': acc, 'rec': rec, 'f1': f1,'label_total':label_total,'pred_total':pred_total}

        label_total, pred_total, cnt = eval_having(pred, label)
        acc, rec, f1 = get_scores(cnt, pred_total, label_total)
        res['group'] = {'acc': acc, 'rec': rec, 'f1': f1,'label_total':label_total,'pred_total':pred_total}

        label_total, pred_total, cnt = eval_order(pred, label)
        acc, rec, f1 = get_scores(cnt, pred_total, label_total)
        res['order'] = {'acc': acc, 'rec': rec, 'f1': f1,'label_total':label_total,'pred_total':pred_total}

        label_total, pred_total, cnt = eval_and_or(pred, label)
        acc, rec, f1 = get_scores(cnt, pred_total, label_total)
        res['and/or'] = {'acc': acc, 'rec': rec, 'f1': f1,'label_total':label_total,'pred_total':pred_total}

        label_total, pred_total, cnt = eval_IUEN(pred, label)
        acc, rec, f1 = get_scores(cnt, pred_total, label_total)
        res['IUEN'] = {'acc': acc, 'rec': rec, 'f1': f1,'label_total':label_total,'pred_total':pred_total}

        label_total, pred_total, cnt = eval_keywords(pred, label)
        acc, rec, f1 = get_scores(cnt, pred_total, label_total)
        res['keywords'] = {'acc': acc, 'rec': rec, 'f1': f1,'label_total':label_total,'pred_total':pred_total}

        return res


def evaluate_spark_sql(gold_sql, pred_sql, spark, db_name="default"):
    """
    Evaluates a predicted Spark SQL query against a gold standard query.
    Returns the exact match score (0 or 1).
    """
    try:
        schema_dict = get_schema(spark, db_name)
        schema = Schema(schema_dict)
        g_sql = get_sql(schema, gold_sql)
        p_sql = get_sql(schema, pred_sql)
        
        evaluator = Evaluator()
        exact_score = evaluator.eval_exact_match(p_sql, g_sql)
        return exact_score
    except Exception as e:
        print(f"Evaluation Error: {e}")
        return 0

if __name__ == "__main__":
    # test the jaccard_index./sr     function
    # Source - https://stackoverflow.com/a
# Posted by user2285236, modified by community. See post 'Timeline' for change history
# Retrieved 2025-12-30, License - CC BY-SA 4.0

    # from sklearn.metrics.pairwise import pairwise_distances
    # res = 1 - pairwise_distances(df.T.to_numpy(), metric='jaccard')

    # df1 = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
    # df2 = pd.DataFrame({'a': [2, 3], 'b': [4, 5]})  
    # sim = jaccard_index(df1, df2)
    # print(f"Jaccard Similarity: {sim}")
    # print("SHould be: 0.3333")

    recompute_ground_truth("/home/lars/Privat/RWTH/Auslandssemester/#KURSE/Safe Distributed Systems/Exercises/Practical_Exercise/NL2SQL2SPARK/RAW_RESULTS/benchmark_results_20251231_google_d598f842/1374/20251231_115747_ID_1374_ITER_5_c14e62f2.json")