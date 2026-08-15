## Environment Setup

### Step 1. Create a virtual environment

Run the command for your operating system in a new terminal:

```bash
# macOS or Linux
python3 -m venv .venv

# Windows
py -m venv .venv
```

If the command above is not available, replace `python3` or `py` with `python`.

### Step 2. Activate the virtual environment

Run the command for your operating system:

```bash
# macOS or Linux
source .venv/bin/activate
```

After activation, `(.venv)` should appear at the beginning of the terminal prompt.

### Step 3. Install the required packages

Install the python packages listed in `requirements.txt`. These packages are required to run the project:

```bash
python -m pip install -r requirements.txt
```

## How to Run

After setting up the virtual environment and installing the required packages, the project can be run step by step to:

- prepare the corpus, 
- generate the chunk collections, 
- prepare BM25 and dense retrieval, and 
- evaluate the retrieval and reranking methods. 

> **Note:** Please run the following scripts in order.

### Step 1. Prepare the corpus

```bash
python i_prepare_corpus.py
```

This script selects the 10 scientific papers used in the project and saves the resulting corpus to the `custom_corpus` folder. 

> **Note:** The saved corpus can then be loaded directly with `load_from_disk("custom_corpus")` in the later steps, so the original dataset does not need to be processed again.


### Step 2. Generate the chunk collections

First, run the chunking utility file:

```bash
python ii_chunk_util.py
```

This file contains the functions used for the four chunking methods.

Then run:

```bash
python ii_chunking_collections.py
```

This step loads the saved `custom_corpus`, applies the four chunking methods, and saves all generated chunks to `chunk_collections.pkl`. 

> **Note:** This file is used directly in the later indexing steps, so the articles do not need to be chunked again.


### Step 3. Build the retrieval indexes

First, build the BM25 indexes:

```bash
python iii_bm25_indexing.py
```

This step loads `chunk_collections.pkl`, builds a separate BM25 index for each chunking method, and saves all BM25 indexes to `bm25_indexes.pkl`.

Then build the dense retrieval indexes:

```bash
python iii_dense_indexing.py
```

This step also loads `chunk_collections.pkl`, converts the chunks into embeddings using the four selected embedding models, and saves the resulting dense indexes to `dense_indexes.pkl`.

> **Note:** The saved `bm25_indexes.pkl` and `dense_indexes.pkl` files are loaded directly during retrieval, so the indexes do not need to be rebuilt each time.


### Step 4. Run the initial retrieval

BM25 retrieval is implemented in:

```bash
python iv_bm25_retrieval.py
```

This step loads `bm25_indexes.pkl` and retrieves the top-ranked chunks from the BM25 index for a given query.

Dense retrieval is implemented in:

```bash
python iv_dense_retrieval.py
```

This step loads `dense_indexes.pkl`, encodes the query using the selected embedding model same as the indexes, and retrieves the chunks with the highest similariy scores. 

> **Note:** These two files provide the retrieval functions used in the evaluation step. 


### Step 5. Run the initial retrieval evaluation

```bash
python v_evaluation.py
```

This step evaluates BM25 and dense retrieval across the different chunking methods and embedding models using Recall@5, Recall@10, and Recall@50. The evaluation results are saved to `v_evaluation_results.txt`.


### Step 6. Run the reranking evaluation

The cross-encoder and hybrid ranking methods are implemented in:

```bash
python vi_cross_encoder_reranking.py
python vi_hybrid_ranking.py
```

Then the final reranking evaluation is run with:

```bash
python vii_evaluation2.py
```

This step evaluates BM25 + cross-encoder, dense retrieval + cross-encoder, and hybrid ranking using Recall@5, Recall@10, and Recall@50. The results are saved to `vii_evaluation2_results.txt`.

> **Note:** `vi_cross_encoder_reranking.py` and `vi_hybrid_ranking.py` provide the reranking functions used by `vii_evaluation2.py`. 


### Step 7. Generate responses with the LLM

```bash
python vi_llm.py
```

This final step uses the retrieved chunks as context for the LLM and generates the final response to the query.

## For demo, run below steps. This uses retrieval configuration that had the highest score in our evaluation.
### The first-time loading does take 3-4 minutes,
### then it starts an interactive session for user to ask questions over sleep disorder research.
```
export OPENAI_API_KEY=<Add your open api key>

python3 demo.py
```
