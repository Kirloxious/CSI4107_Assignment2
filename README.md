

# Results

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

