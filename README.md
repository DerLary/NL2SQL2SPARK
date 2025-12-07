# NL2SQL2SPARK

## Description
This repository provides tools for automating the execution of natural language queries on Apache Spark, along with metrics to measure the accuracy of these translations.

It leverages Large Language Models (specifically Google's Gemini) to translate natural language into Spark SQL. The generated SQL is then executed against a (currently local) PySpark session, and the results are evaluated for accuracy against ground truth data.

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

### 3. Setting up Google AI Studio API Key
This project uses Google's Gemini models (default: `gemini-2.5-flash`). You need an API key from Google AI Studio.

1.  Go to [Google AI Studio](https://aistudio.google.com/).
2.  Create an API key.
3.  (Optional) You can set up billing in Google Cloud to get $300 in free credits if you need higher rate limits. The free tier (up to 10 requests per minute and 250 requests/day for `gemini-2.5-flash`, as for 7-12-2025) might be sufficient just for basic testing.

Create a `.env` file in the root of the repository and add your key:

```bash
echo "GOOGLE_API_KEY=your_api_key_here" > .env
```

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
python3 query_workflow.py --id 2
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

#### Accuracy Metrics
- **Jaccard Index (Execution Accuracy)**:
    - Measures the overlap between the result set of the generated query and the ground truth query.
    - Floating point value in the range [0.0, 1.0] where `0.0` (no match) to `1.0` (perfect match).
- **Spider Exact Match Score (Structural Accuracy)**:
    - Measures whether the structure of the generated SQL matches the gold standard SQL.
    - Binary value: `1` (match) or `0` (no match).
