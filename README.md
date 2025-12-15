# NL2SQL2SPARK

## Description
This repository provides tools for automating the execution of natural language queries on Apache Spark, along with metrics to measure the accuracy of these translations.

It leverages Large Language Models to translate natural language into Spark SQL. The generated SQL is then executed against a (currently local) PySpark session, and the results are evaluated for accuracy against ground truth data.

## Goal
The goal of this project is to provide a SDK and example workflow for building and benchmarking NL-to-SQL agent accuracy and performance on top of Spark.

## Instructions

### Requirements
- **Python**: 3.12 or greater.
- **Java**: Java JDK/OpenJDK 21 or greater.

### 1. Installing and Setting Up Spark requirements
PySpark requires a Java Development Kit (JDK) to run.

**On Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install openjdk-21-jdk
java -version
```

**On macOS (using Homebrew):**
```bash
brew install openjdk@21
```

**On Windows:**
(Easiest option) Download and install Temurin JDK 21 from [Adoptium](https://adoptium.net/) or use `winget`:
```powershell
winget install Microsoft.OpenJDK.21
```

Ensure `JAVA_HOME` is set if Spark has trouble finding Java.

### 2. Installing the Virtual Environment
It is recommended to use a Python virtual environment to manage dependencies.

```bash
# Create a virtual environment
python3.12 -m venv sparkai-env

# Activate the environment
source sparkai-env/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

You should activate the virtual environment every time you want to run the code.

### 3. Setting up LLM Provider
This project supports multiple LLM providers. You can choose between Google (Google AI Studio) and Cloudflare (Workers AI).

#### 3.1 Google (Default)
The default provider uses Google's Gemini models (default: `gemini-2.5-flash`). You need an API key from Google AI Studio.

1.  Go to [Google AI Studio](https://aistudio.google.com/).
2.  Create an API key.
3.  (Optional) You can set up billing in Google Cloud. The free tier (up to 10 requests per minute and 250 requests/day for `gemini-2.5-flash`) might be sufficient for basic testing.

Add your key to the `.env` file:
```bash
GOOGLE_API_KEY=your_api_key_here
```

#### 3.2 Cloudflare
You can also use Cloudflare Workers AI. This requires an Account ID and an API Token.

1.  Go to the [Cloudflare Dashboard](https://dash.cloudflare.com/).
2.  Navigate to **Account Home**.
3.  Get your **Account ID** from the ⋮ options.
4.  Navigate to **Workers AI**.
5.  Create an **API Token** with the "Workers AI" template.
6.  Copy the **API Token**.

Add your credentials to the `.env` file:
```bash
CLOUDFLARE_ACCOUNT_ID=your_account_id
CLOUDFLARE_API_TOKEN=your_api_token
```

**Note on Cloudflare Pricing (Neurons):**
Cloudflare uses a unit called **Neurons** for billing.
- **Free Tier:** All users get **10,000 Neurons per day** for free.
- **Usage:** Different models consume different amounts of Neurons per input/output token. High-end models consume more Neurons, you can check model pricing [here](https://developers.cloudflare.com/workers-ai/platform/pricing/).

### 4. Downloading the Database
The project requires a sample database (the [Bird benchmark](https://bird-bench.github.io/) development set).

1.  Download the database zip file from [here](https://drive.google.com/file/d/1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J/view?usp=sharing).
2. Unzip the file and go to dev/dev_20240607/
4. Copy the `.json` files `db` directory in the root of this repository.
5. Unzip `dev_databases.zip`.
6. Copy the directories under `dev_databases` into the `db` directory in the root of this repository.


Structure should look like:
```
/path/to/repo/
  ├── db/
  │   ├── california_schools/
  │   ├── card_games/
  │   ├── ...
  │   ├── dev_tables.json
  │   ├── dev_tied_append.json
  │   ├── dev.json
  ├── src/
  ├── ...
```

### 5. Running an Example
To run the benchmark workflow which processes a natural language query, converts it to Spark SQL, and evaluates it:

```bash
# Run with default provider (Google)
python3 query_workflow.py --id 2

# Run with Cloudflare provider
python3 query_workflow.py --id 2 --provider cloudflare
```

This script will:
1.  Load a sample query (the query with ID 2, in this case).
2.  Use the LLM to generate a Spark SQL query.
3.  Execute the generated query.
4.  Compare the result with the ground truth (Jaccard Index).
5.  Compare the generated SQL structure with the gold standard (Spider Exact Match).

### 6. Interpreting the Output

The `query_workflow.py` script outputs detailed performance metrics and accuracy scores. Here is how to interpret them:

#### Performance Metrics
- **Execution Status**:
    - `VALID`: The query was successfully generated and executed against the Spark session.
    - `ERROR`: The generated query failed during execution (e.g., syntax error, schema mismatch).
    - `NOT_EXECUTED`: The query was not executed due to an undetermined error.
- **Total End-to-End Time**: Total time taken for the entire process.
- **Spark Execution Time**: Time taken for Spark to execute the generated SQL.
- **Input Translation (LLM)**: Time taken by the LLM to generate the SQL (including LLM connection latency and all the necessary requests).
- **LLM Requests**: Number of calls made to the LLM (useful for rate limit monitoring).
- **Input/Output Tokens**: The number of tokens sent to and received from the LLM. This is critical for estimating billing costs.
- **Input/Output Neurons (Cloudflare Only)**: The number of Neurons consumed by the request. Cloudflare provides a daily free tier of 10k Neurons.

#### Accuracy Metrics
- **Jaccard Index (Execution Accuracy)**:
    - Measures the overlap between the result set of the generated query and the ground truth query.
    - Floating point value in the range [0.0, 1.0] where `0.0` (no match) to `1.0` (perfect match).
- **Spider Exact Match Score (Structural Accuracy)**:
    - Measures whether the structure of the generated SQL matches the gold standard SQL.
    - Binary value: `1` (match) or `0` (no match).

### 7. Running with Custom Queries (HITL UI)

You can use the `hitl-ui.py` script to run the workflow with your own natural language queries via the command line, while still benchmarking against a specific ground truth (Query ID) for context.

```bash
python3 hitl-ui.py --id <QUERY_ID> --nl-query "<YOUR_QUERY>"
```

**Example:**
```bash
python3 hitl-ui.py --id 2 --nl-query "Please list the zip code of all the charter schools in Fresno County Office of Education."
```

This is useful for testing how different phrasings affect the generated SQL.
