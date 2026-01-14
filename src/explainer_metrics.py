import re

# keyword for shuffle = Exchange: https://jaceklaskowski.gitbooks.io/mastering-spark-sql/content/spark-sql-SparkPlan-ShuffleExchangeExec.html
def count_shuffles(plan_text: str):
    """
    Counts shuffle exchanges in a Spark physical plan string.
    Heuristic: Exchange with hashpartitioning/rangepartitioning indicates shuffle.
    """
    if not plan_text:
        return 0

    shuffle_exchange_patterns = [
        r"Exchange\s+hashpartitioning\(",
        r"Exchange\s+rangepartitioning\(",
        r"Exchange\s+SinglePartition",
        r"ShuffleExchange",  
    ]

    counter = 0
    count = {pat: 0 for pat in shuffle_exchange_patterns}
    # store count for each pattern
    for pat in shuffle_exchange_patterns:
        count[pat] += len(re.findall(pat, plan_text))
        counter += count[pat]
    count["TOTAL"] = counter
    return count

# https://jaceklaskowski.gitbooks.io/mastering-spark-sql/content/spark-sql-SparkPlan-BroadcastHashJoinExec.html
def has_broadcast_join(plan_text: str) -> bool:
    if not plan_text:
        return False

    broadcast_markers = [
        "BroadcastHashJoin",
        "BroadcastNestedLoopJoin",
        "BroadcastExchange",
        "broadcast",
    ]

    lower = plan_text.lower()
    return any(m.lower() in lower for m in broadcast_markers)

def extract_pushed_filters(plan_text: str):
    """
    Returns a list of dicts like:
      [{"scan_line": "...", "filters": ["*IsNotNull(Score)", "*GreaterThan(Score,5)"]}, ...]
    """
    if not plan_text:
        return []

    results = []
    # find scan lines that contain PushedFilters: [...]
    for line in plan_text.splitlines():
        if "Scan" in line and "PushedFilters:" in line:
            m = re.search(r"PushedFilters:\s*\[(.*?)\]", line)
            if m:
                raw = m.group(1).strip()
                filters = [f.strip() for f in raw.split(",")] if raw else []
                results.append({"scan_line": line.strip(), "filters": filters})
    return results

def count_filter_before_and_after(analyzed, optimized):
    filters_before = analyzed.count("Filter")
    filters_after = optimized.count("Filter")

    return filters_before, filters_after

def has_predicate_pushdown(plan_text: str) -> bool:
    pushed = extract_pushed_filters(plan_text)
    # Consider pushdown true if any scan has any pushed filter besides empty list
    return any(item["filters"] and item["filters"] != [""] for item in pushed)

def extract_all_plans(plan_text: str) -> dict:
    """
    Extracts the '3. Physical plan:' block from the stored output.
    Falls back to the 'Physical Plan ==' section if needed.
    """
    if not plan_text:
        return  {}
    
    sections = {}
    pattern = r"(\d)\.\s*([^:]+):\s*\n-+\n(.*?)(?=\n\s*\d\.\s*|$)"

    for match in re.finditer(pattern, plan_text, flags=re.DOTALL):
        section_num = match.group(1)
        # section_title = match.group(2).strip() # e.g., "Optimized logical plan"
        content = match.group(3).strip()
        sections[section_num] = content

    plans = {}

    plans["logical"] = sections.get("1", "")
    plans["optimized"] = sections.get("2", "")
    plans["physical"] = sections.get("3", "")

    return plans

def normalize_plan(plan_block: str) -> str:
    """
    Normalizes a plan so irrelevant IDs don't affect equality.
    """
    s = plan_block

    # Remove expression IDs like Age#63 -> Age
    s = re.sub(r"#\d+", "", s)

    # Remove plan ids like [plan_id=148]
    s = re.sub(r"\[plan_id=\d+\]", "", s)

    # Remove long numeric suffixes like sum#94L -> sum
    s = re.sub(r"\b([A-Za-z_]+)\d*L\b", r"\1", s)

    # Optional: remove partition count numbers (keep operator names)
    s = re.sub(r"hashpartitioning\(([^,]+),\s*\d+\)", r"hashpartitioning(\1, N)", s)

    # Compress whitespace
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n+", "\n", s)

    return s.strip()

def plans_equivalent(plan_text_a: str, plan_text_b: str) -> bool:
    plans_a = extract_all_plans(plan_text_a)
    a_block = plans_a.get("physical", "")
    plans_b = extract_all_plans(plan_text_b)
    b_block = plans_b.get("physical", "")
    return normalize_plan(a_block) == normalize_plan(b_block)

def analyze_execution_plans(result_dict: dict) -> dict:
    plans = result_dict.get("execution_plans", {})
    out = {}

    for name, plan_text in plans.items():
        plan_dict = extract_all_plans(plan_text)
        phys = plan_dict.get("physical", "")
        out[name] = {
            "shuffle_count": count_shuffles(phys),
            "has_broadcast_join": has_broadcast_join(phys),
            "has_predicate_pushdown": has_predicate_pushdown(phys),
            "pushed_filters": extract_pushed_filters(plan_text),
        }
        logical = plan_dict.get("logical", "")
        optimized = plan_dict.get("optimized", "")
        filters_before, filters_after = count_filter_before_and_after(logical, optimized)
        out[name]["filter_count_before_optimization"] = filters_before
        out[name]["filter_count_after_optimization"] = filters_after
        
    # equivalence (golden vs model)
    g = plans.get("golden_query", "")
    m = plans.get("model_query", "")
    out["plan_equivalence"] = {
        "golden_vs_model_physical_equivalent": plans_equivalent(g, m),
    }
    return out