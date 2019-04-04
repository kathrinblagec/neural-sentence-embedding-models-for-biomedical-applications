# Neural sentence embedding models for biomedical applications
Contains datasets and code for the publication 'Neural sentence embedding models for semantic similarity estimation in the biomedical domain'; K. Blagec, H. Xu, A. Agibetov, M. Samwald

## Jupyter Notebooks
Neural-sentence-embedding-models.ipynb
String-based-similarity-measures.ipynb
Results-overview.ipynb
Datasets.ipynb

## Files in folder data

### Benchmark dataset (Sentences and annotation scores)
biosses_sentence_pairs_test_derived_from_github.txt      
biosses_sentence_pairs_test_derived_from_github_preprocessed.txt         
annotation_scores_from_github.txt      

### Experimental contradiction subset  
biosses_antonym_subset.txt   
biosses_negation_subset.txt 
    
## Availability of used datasets / models / parser

### Corpus

PubmedCentral Open Access dataset: http://ftp.ncbi.nlm.nih.gov/pub/pmc

### Embedding model implementations

Skip-thoughts: https://github.com/tensorflow/models/tree/master/research/skip_thoughts   
Sent2vec including pre-trained models: https://github.com/epfml/sent2vec      
fastText: https://github.com/facebookresearch/fastText
Paragraph Vector: 

### Parser

Stanford Core NLP: https://stanfordnlp.github.io/CoreNLP/



