import os

PROJECT_DIR = os.path.dirname((os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, 'data')

# Scores assigned by the human experts for each of the 100 sentence pairs
ANNOTATION_SCORES = os.path.join(DATA_DIR, 'annotation_scores_from_github.txt')

# Results overview
RESULTS_OVERVIEW = os.path.join(DATA_DIR, 'results_overview.csv')

SENTENCE_COSINE_SIMILARITIES = os.path.join(DATA_DIR, 'sentences-and-calculated-similarities.csv')

# Scores predicted with BERT classifier
PREDICTED_SCORES= os.path.join(DATA_DIR, 'bert_base_classifier_predictions.txt')

# Original BIOSSES sentence pairs (no pre-processing)
BIOSSES_SENTENCE_PAIRS = os.path.join(DATA_DIR, 'biosses_sentence_pairs_test_derived_from_github.txt')

# Pre-processed sentences (lower-case, words and punctuation separated by whitespaces), one sentence per line.
BIOSSES_SENTENCE_PAIRS_PREPROCESSED = os.path.join(DATA_DIR, 'biosses_sentence_pairs_test_derived_from_github_preprocessed.txt')

# Stopwords
RANKS_STOPWORDS = os.path.join(DATA_DIR, 'ranks_stopwords.txt')
STANFORD_STOPWORDS = os.path.join(DATA_DIR, 'stanford_core_stopwords.txt')

# Contradiction datasets and vectors

BIOSSES_NEGATION_SUBSET = os.path.join(DATA_DIR, 'biosses_negation_subset.txt')
BIOSSES_ANTONYM_SUBSET = os.path.join(DATA_DIR, 'biosses_antonym_subset.txt')

SENT2VEC_NEGATION_SUBSET_VECTORS = os.path.join(DATA_DIR, 'sent2vec_negation_subset.txt')
SENT2VEC_ANTONYM_SUBSET_VECTORS = os.path.join(DATA_DIR, 'sent2vec_antonym_subset.txt')

PVDM_NEGATION_SUBSET_VECTORS = os.path.join(DATA_DIR, 'biosses_negation_subset.txtskipgram.vec')
PVDM_ANTONYM_SUBSET_VECTORS = os.path.join(DATA_DIR, 'biosses_antonym_subset.txtantonym_skipgram.vec')

PVDBOW_NEGATION_SUBSET_VECTORS = os.path.join(DATA_DIR, 'biosses_negation_subset.txtnegation.vec')
PVDBOW_ANTONYM_SUBSET_VECTORS = os.path.join(DATA_DIR, 'biosses_antonym_subset.txtantonym.vec')

SKIPTHOUGHTS_NEGATION_SUBSET_VECTORS = os.path.join(DATA_DIR, 'BIOSSES-skipthought-negation-subset.txt')
SKIPTHOUGHTS_ANTONYM_SUBSET_VECTORS = os.path.join(DATA_DIR, 'BIOSSES-skipthought-antonym-subset.txt')

subset_file_list = [SENT2VEC_NEGATION_SUBSET_VECTORS,
                    SENT2VEC_ANTONYM_SUBSET_VECTORS,
                    SKIPTHOUGHTS_NEGATION_SUBSET_VECTORS,
                   SKIPTHOUGHTS_ANTONYM_SUBSET_VECTORS,
                   PVDM_NEGATION_SUBSET_VECTORS,
                   PVDM_ANTONYM_SUBSET_VECTORS,
                   PVDBOW_NEGATION_SUBSET_VECTORS,
                   PVDBOW_ANTONYM_SUBSET_VECTORS]

# Cosine similarity files

COSINE_SIMILARITY_FILES = {
    'COSINES_SKIPTHOUGHT_400_VOCAB_EXT' : 'cosines-skipthought-400-vocab-ext.csv',
    'COSINES_SKIPTHOUGHT_1000_VOCAB_EXT' : 'cosines-skipthought-1000-vocab-ext.csv',
    'COSINES_SKIPTHOUGHT_400' : 'cosines-skipthought-400.csv',
    'COSINES_SKIPTHOUGHT_1000' : 'cosines-skipthought-1000.csv',
    'COSINES_SENT2VEC_200_100d' : 'cosines-sent2vec-200-1000d.csv',
    'COSINES_FASTTEXT_SKIP_GRAM_MAX_POOLING' : 'cosines-fasttext-skip-gram-max-pooling.csv',
    'COSINES_FASTTEXT_SKIP_GRAM_MIN_POOLING' : 'cosines-fasttext-skip-gram-min-pooling.csv',
    'COSINES_FASTTEXT_SKIP_GRAM_AVG_POOLING' : 'cosines-fasttext-skip-gram-avg-pooling.csv',
    'COSINES_FASTTEXT_SKIP_GRAM_SUM_POOLING' : 'cosines-fasttext-skip-gram-sum-pooling.csv',
    'COSINES_FASTTEXT_CBOW_MAX_POOLING' : 'cosines-fasttext-cbow-max-pooling.csv',
    'COSINES_FASTTEXT_CBOW_MIN_POOLING' : 'cosines-fasttext-cbow-min-pooling.csv',
    'COSINES_FASTTEXT_CBOW_AVG_POOLING' : 'cosines-fasttext-cbow-avg-pooling.csv',
    'COSINES_FASTTEXT_CBOW_SUM_POOLING' : 'cosines-fasttext-cbow-sum-pooling.csv',
    'COSINES_PARAGRAPH_VECTOR_DM' : 'cosines-pv-dm.csv',
    'COSINES_PARAGRAPH_VECTOR_DBOW' : 'cosines-pv-dbow.csv'}

# Skip-though model files

# Model checkpoint directory
SKIPTHOUGHT_DIR = '/media/matthias/Big-disk/trained-models/skipthought-model-pmcoa'

# Training files for PMCOA corpus produced by pre-processing script (corpus filtered for line length < 400, hyphens separated)
SKIPTHOUGHT_PREPROCESSING_DIR = '/media/matthias/Big-disk/trained-models/skipthought-model-pmcoa/PMC-OA-filtered-400-hyphens-separated-skip-thoughts-preprocessed'

# Vocab file for PMCOA corpus produced by pre-processing script (corpus filtered for line length < 400, hyphens separated)
SKIPTHOUGHT_VOCAB = 'vocab.txt'

# Expanded vocab file and embedding file for skipthoughts model trained on PMCOA corpus based on google news word2vec model
SKIPTHOUGHT_VOCAB_EXT = 'exp-vocab-google-news/vocab.txt'
SKIPTHOUGHT_EMBEDDINGS = 'exp-vocab-google-news/embeddings.npy'

# Google news word2vec model used for vocab expansion:
GOOGLE_NEWS_MODEL = '/media/matthias/Big-disk/trained-models/word2vec-google-news/GoogleNews-vectors-negative300.bin'

# Script for retrieving the vectors for the BIOSSES sentences
SKIPTHOUGHT_ENCODING = '/home/matthias/Documents/Intelligence/SentEval/examples/skipthoughts_encoding.py'

# PMCOA corpus (filtered for line length < 400, hyphens separated), vocab extension based on Google news word2vec model:
SKIPTHOUGHT_EMBEDDINGS = '/home/matthias/Documents/Intelligence/SDL1-Embeddings/data/BIOSSES-skipthought-PMCOA-400-encodings-exp-vocab-google-news.txt'

# Pre-trained skip-thought vectors:
SKIPTHOUGHT_PRETRAINED = '/media/matthias/Big-disk/trained-models/skipthought-vectors/dictionary.txt'