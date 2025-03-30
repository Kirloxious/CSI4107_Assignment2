# Name & Tasks

Name: Alexandre Ringuette   
Student Number: 300251252      
Tasks: Reranking process, MAP Score and Report

Name: Alexis Laplante   
Student Number: 300220658   
Tasks: Embedding Computation, Similarity Calculations and Report

Name: Louka Papineau    
Student Number: 300236645   
Tasks: Data Loading, Preprocessing and Report



# Functionality

| File Name                 | Functionality                                                                                                   |
|---------------------------|-----------------------------------------------------------------------------------------------------------------|
| reranker.py               | Main script that loads initial retrieval results, computes query and document embeddings, and reranks documents using cosine similarity. |
| saved/results.tsv         | Contains the initial retrieval results from the first-stage ranking.                                            |
| scifact/corpus.jsonl      | Contains the corpus documents in JSONL format.                                                                  |
| scifact/queries.jsonl     | Contains the queries in JSONL format.                                                                           |
| scifact/qrels/test.tsv    | Contains the test relevance judgments (qrels) for evaluation.                                                   |
| scifact/qrels/train.tsv   | Contains the training relevance judgments (qrels) for evaluation.                                                 |
| scifact/stopwords.txt     | Contains stopwords for preprocessing (if applicable in extended pipelines).                                     |
| reranked_results.tsv      | Contains the final reranked document results produced by the neural reranking process.                            |

The program begins by loading the initial retrieval results along with the corpus and queries. It then computes embeddings for the queries and documents using a pre-trained transformer model, calculates cosine similarity between embeddings, and generates a combined score. Finally, it reranks the documents based on this combined score and outputs the results to `reranked_results.tsv`.



# Instructions

- **Prerequisites:** Python 3 and the required libraries: `torch`, `numpy`, and `transformers`.
- **Dependencies Installation:**  
  ```bash
  pip install torch numpy transformers
  ```
- **Once installed:** You can run the following command in the home directory of the assignment: `python reranker.py`
- The program will then run and output the result to `reranked_results.tsv`.



# Explanation of Algorithms, Data Structures, and Optimizations

### Data loading and Preprocessing
- File parsing:
    - The `read_jsonl` function reads JSONL files such as the `corpus.jsonl` and `queries.jsonl` and converts each line into a dictionary keyed by document or query ID.
    - The `load_results` function reads the initial retrieved results from Assigment 1's `results.tsv` file parsing each line into tuples containing the query and document identifiers, rank, score and run identifiers.
- Data Structure:
    - Dictionaries are used to store corpus and query data.
    - A dictionary is maintained to cache computed embeddings to avoid redundant calculations.

### Embedding Computation and Similarity Calculation
- Embedding Extraction:
    - The `get_embedding` function uses a pre-trained transformer model from the `transformers` package to generate vector embeddings for text. Mean pooling is applied to the token embeddings.
- Cosine Similarity:
    - The `cosine_similarity` function converts embeddings to PyTorch tensors and calculates cosine similarity. This similarity measure is used to assess how closely a document matches a query.

### Reranking Process
- Combined Scoring:
    - In the `rerank_documents` function, for each query, the document's initial score is combined with the computed cosin similarity (using an equal weighting of 0.5 for each).
    - Documents are then sorted in descending order based on this combined score.
- Output Generation:
    - The final ranked results are written to `reranked_results.tsv` in a standard TREC format.

### Optimization
- Embedding Caching:
    - The embedding dictionary makes sure that each query and document embedding is computed only once, which reduces redundant computation and improves efficiency.
- Efficient Sorting:
    - Sorting is preformed for each query's document list based on the combined score, ensuring that the highest scoring documents are returned first.

### Query Test Results

Here are the results of the top 10 answers of queries 1 and 3


1	Q0	13231899	1	0.43747454	4843292\
1	Q0	40212412	2	0.35902283	6657981\
1	Q0	3770726	3	0.33882580	3770727\
1	Q0	6863070	4	0.31040906	6863071\
1	Q0	6550579	5	0.30677886	6550580\
1	Q0	7581911	6	0.29729487	7581912\
1	Q0	18953920	7	0.29013319	2176705\
1	Q0	38805486	8	0.28468330	5251055\
1	Q0	1944452	9	0.27584964	1944453\
1	Q0	3566945	10	0.24389256	3566946

3	Q0	4632921	1	0.64650798	4632924\
3	Q0	23389795	2	0.62611788	6612582\
3	Q0	4414547	3	0.62213230	4414550\
3	Q0	13519661	4	0.59236674	5131056\
3	Q0	2739854	5	0.59012562	2739857\
3	Q0	2107238	6	0.57862449	2107241\
3	Q0	32181055	7	0.57547262	7015234\
3	Q0	14717500	8	0.57222185	6328895\
3	Q0	4378885	9	0.57095116	4378888\
3	Q0	15570962	10	0.55586183	7182357

We can see from these query results that Query 1 shows a clear standout top result with a broader score range, potentially indicating a strong candidate among a more diverse set of document, as Query 3 features a generally higher and more compressed score range, which could suggest that many documents are very similar in terms of semantic relevance.



# Mean Average Precision

Using this trec eval command: 
./trec_eval -m map -m P.10 ../../Assignment2/scifact/qrels/test.tsv ../../Assignment2/reranked_results.tsv

We find these results with the following models: 

Bert Model: "dslim/bert-base-NER"   
| Measure | Score| 
|---|---|
|Map| 0.5241|
|P@10|0.0783|


Mpnet Model: "sentence-transformers/all-mpnet-base-v2"  
| Measure | Score| 
| --- | --- |
|Map| 0.6447|
|P@10|0.0897|
