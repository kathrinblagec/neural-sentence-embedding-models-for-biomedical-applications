import matplotlib.pyplot as plt
import numpy as np
from numpy import dot, random
from numpy.linalg import norm
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats.stats import pearsonr, spearmanr
import pandas as pd
import seaborn as sns
import os
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import ngram


def calculate_cos_sim(x,y):
    cos_sim = (dot(x, y) / (norm(x)*norm(y)))
    return cos_sim

def read_vectors(filename, model=None):
    """Function for reading vectors from .txt file in to a numpy array."""
    if model == None:
        with open(filename, "r", encoding="utf-8") as f:
            vectors = np.loadtxt(f)
    # first column needs to be removed from PV vector output files 
    if model == 'PV':
        vectors = []
        for line in open(filename):
            vectors.append(line.split(" ")[1:])
        vectors = vectors[1:]
        vectors = np.array(vectors, dtype=float)
    if model == 'Fasttext':
        vector = []
        words = []
        for line in open(filename, 'r', encoding="utf8"):
            vector.append(line.split()[1:])
            words.append(line.split(None, 1)[0])
            vectors = np.array(vector).astype(np.float)
    return vectors

def calculate_vector_similarity(vectors):
    """Function for calculating the cosine similarity between vectors of each sentence pair and storing them in the list cos_sim. 
    Starting with index 0, two censecutive sentences form a sentence pair, e.g., 
    vectors[0] + vectors[1], vectors[2] + vectors[3], etc. 
    Returns a numpy array with dimension (number of vectors/2,)"
    """
    cos_sim = []
    i= 0
    while (i <= vectors.shape[0]-1):
        cos_sim.append(calculate_cos_sim(vectors[i], vectors[i+1]))                      
        #print(cosine_similarity([vectors[i]], [vectors[i+1]], dense_output=True)[0])
        i += 2
    cos_sim = np.array(cos_sim, dtype=float)
    return cos_sim

def calculate_ppmc(x, y):
    """Returns the PPMC and p-value."""
    
    ppmc, ppmc_p = pearsonr(x, y)
    return (ppmc, ppmc_p)

def calculate_spearman(x, y):
    """Returns Spearman correlation and p-value."""
    
    spearman, spearman_p = spearmanr(x, y)
    return (spearman, spearman_p)

def plot_correlation(annotation_scores, similarities, alpha=0.5):
    plt.subplots(figsize=(4,4))
    c = sns.regplot(x=annotation_scores, y=similarities, scatter_kws={"s": 14})
    c.set_xlabel("Annotation scores.",fontsize=10)
    c.set_ylabel("Calculated similarity",fontsize=10)
    c.axes.set_title("Correlation between given similarities and scores assigned by human annotators.",fontsize=11)
    c.tick_params(labelsize=10)
    return plt.show()

def evaluate_similarities(similarities, annotation_scores, results_overview, name = None):
    """Calculate PPMC and Spearman correlation; draw plot"""  
    plot_correlation(annotation_scores, similarities)
    ppmc = calculate_ppmc(similarities, annotation_scores)
    spearman_correlation = calculate_spearman(similarities, annotation_scores)
    
    print("PPMC: " + str(ppmc[0]))
    print("p-value: " + str(ppmc[1]))
    print("Spearman correlation: " + str(spearman_correlation[0]))
    print("p-value: " + str(spearman_correlation[1]))
    
    if name:
        results_overview[name] = round(ppmc[0], 3)
        
def generate_dictionary(files):
    """Generate dictionaries with cosine similarities and means / std"""
    results = {}
    means_and_stds = {}
    for i in files:
        cos, mean_cos, std_cos = calculate_cosines_for_subsets(i)
        results[str(i)]= cos
        means_and_stds[str(i)] = [mean_cos, std_cos]
    return results, means_and_stds

def calculate_cosines_for_subsets(file):
    """Calculate cosine similarities and means for antonym and negation subsets"""
    if os.path.splitext(file)[1] == '.vec':
        vec = PV_read_vectors(file)
    else:
        vec = read_vectors(file)
    cos = calculate_vector_similarity(vec)
    mean_cos = np.mean(cos)
    std_cos = np.std(cos)
    return cos, mean_cos, std_cos

def retrieve_cosines_for_similar_subset(similarities):
    similar_subset_dict = {}
    cos = []
    for j in similarities.keys():
        cos = []
        for i in similar_sentences_idx:
            #print(similarities.get(j)[i])
            cos.append(similarities.get(j)[i])
            mean = np.mean(cos)
            std = np.std(cos) 
            for j in similarities.keys():
                similar_subset_dict[str(j)] = [cos, mean, std]
    return similar_subset_dict

def pool_vectors(word_vectors, pooling_method, sent):
    pooled_vectors =[]
    i = 0
    k = 0 
    if pooling_method == "max":
        while i < len(sent):
            pooled_vectors.append(np.amax(word_vectors[k:k+len(sent[i])], axis = 0))
            k+=len(sent[i])
            i+=1
    if pooling_method == "min":
        while i < len(sent):
            pooled_vectors.append(np.amin(word_vectors[k:k+len(sent[i])], axis = 0))
            k+=len(sent[i])
            i+=1
    if pooling_method == "sum":
        while i < len(sent):
            pooled_vectors.append(np.sum(word_vectors[k:k+len(sent[i])], axis = 0))
            k+=len(sent[i])
            i+=1
    if pooling_method == "avg":
        while i < len(sent):
            pooled_vectors.append(np.average(word_vectors[k:k+len(sent[i])], axis = 0))
            k+=len(sent[i])
            i+=1
    return pooled_vectors

def read_scores(filename):
    with open(filename, "r", encoding="utf-8") as f:
        scores = np.loadtxt(f)
    return scores

def generate_dictionary(files):
    """Generate dictionaries with cosine similarities and means / std"""
    results = {}
    means_and_stds = {}
    for i in files:
        cos, mean_cos, std_cos = calculate_cosines_for_subsets(i)
        results[str(i)]= cos
        means_and_stds[str(i)] = [mean_cos, std_cos]
    return results, means_and_stds

def calculate_cosines_for_subsets(file):
    """Calculate cosine similarities and means for antonym and negation subsets"""
    if os.path.splitext(file)[1] == '.vec':
        vec = read_vectors(file, 'PV')
    else:
        vec = read_vectors(file)
    cos = calculate_vector_similarity(vec)
    mean_cos = np.mean(cos)
    std_cos = np.std(cos)
    return cos, mean_cos, std_cos


