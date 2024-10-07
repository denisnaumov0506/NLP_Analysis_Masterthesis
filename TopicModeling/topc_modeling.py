from cuml.cluster import HDBSCAN, all_points_membership_vectors
import torch
import cudf
import numpy as np
import pandas as pd
from cuml.manifold import UMAP
import cupy as cp
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, ENGLISH_STOP_WORDS
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import gensim
from gensim.models.coherencemodel import CoherenceModel
from octis.evaluation_metrics.diversity_metrics import TopicDiversity

def calculate_topic_diversity(cluster_term_dataframes, top_n):
    all_top_words = []
    
    # Collect the top-N words from each cluster
    for cluster_id, cluster_df in cluster_term_dataframes.items():
        top_words = cluster_df['Term'].values[:top_n]  # Take top-N words for this cluster
        all_top_words.extend(top_words)
    
    # Convert the list to a set to count unique words
    unique_top_words = set(all_top_words)
    
    # Total number of top words across all clusters
    total_top_words = top_n * len(cluster_term_dataframes)
    
    # Topic diversity score
    topic_diversity = len(unique_top_words) / total_top_words
    return topic_diversity

def calculate_gensim_coherence(cluster_term_dataframes, top_n, texts, dictionary):
    coherence_scores = []
    
    # Loop through each cluster
    for cluster_id, cluster_df in cluster_term_dataframes.items():
        # Get the top-N terms for the current cluster
        top_terms = cluster_df['Term'].values[:top_n]

        # Convert terms to their IDs using the Gensim dictionary
        top_term_ids = [dictionary.token2id[term] for term in top_terms if term in dictionary.token2id]

        # Ensure there are enough terms to calculate coherence
        if len(top_term_ids) < 2:
            continue

        # Create a coherence model for this cluster
        coherence_model = CoherenceModel(topics=[top_term_ids], 
                                         texts=texts, 
                                         dictionary=dictionary, 
                                         coherence='c_v')
        
        # Calculate coherence score for this cluster
        coherence_score = coherence_model.get_coherence()
        coherence_scores.append(coherence_score)
    
    # Average coherence score across all clusters
    avg_coherence = np.mean(coherence_scores)
    return avg_coherence

topic_coherence_scores = []
topic_diversity_scores = []

# Check if CUDA is available
if torch.cuda.is_available():
    print("Yes, it is available!")
    print("Device name: ", torch.cuda.get_device_name(0))
    print("Device count: ", torch.cuda.device_count())

# Function to read and parse the embeddings
def read_embeddings(path):
    embeddings = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            # Parse the line as a list of floats
            embedding = np.fromstring(line.strip('[]'), sep=',')
            embeddings.append(embedding)
    return embeddings

for i in range(7):
    label = i
    app = "whatsapp"

    # Path to the embeddings file and original DataFrame
    path = f"{app}_embeddings/{app}_{label}_results_splitted_corrected_sbert.txt"
    df_path = pd.read_csv(f"{app}_embeddings/{app}_classifier.csv")

    # Filter the DataFrame to get the data where 'label' == 0
    df_data = df_path[(df_path['label'] == label) & (df_path['probs'] > 0.70)].reset_index(drop=True)

    # Read the embeddings
    embeddings = read_embeddings(path)

    # Ensure the number of embeddings matches the rows in df_data
    assert len(embeddings) == len(df_data), "Mismatch between embeddings and df_data length!"

    # Determine the maximum length of sentence embeddings
    max_length = max(len(sent_embedding) for sent_embedding in embeddings)

    # Pad all embeddings to the same length
    padded_sentence_embeddings = np.array([np.pad(sent_embedding, (0, max_length - len(sent_embedding))) for sent_embedding in embeddings])

    # Convert embeddings to torch tensor with float precision
    padded_sentence_embeddings = torch.tensor(padded_sentence_embeddings, dtype=torch.float32)

    # Convert data to GPU array using CuPy
    embeddings_gpu = cp.asarray(padded_sentence_embeddings)

    # Initialize UMAP model
    umap_model = UMAP(n_neighbors=10, n_components=2, min_dist=0.1, random_state=42)

    # Fit and transform the data using UMAP
    embedding_umap = umap_model.fit_transform(embeddings_gpu)

    # Convert the result back to CPU array
    embedding_umap_cpu = cp.asnumpy(embedding_umap)

    # Convert to cuDF DataFrame for HDBSCAN clustering
    embeddings_cudf = cudf.DataFrame(embedding_umap_cpu)

    # Cluster data using HDBSCAN
    clusterer = HDBSCAN(allow_single_cluster=True, cluster_selection_method='eom', prediction_data=True, alpha=1.0, metric='euclidean', cluster_selection_epsilon=0.2, min_samples=15, min_cluster_size=15, max_cluster_size=30)
    cluster_labels = clusterer.fit_predict(embeddings_cudf)

    # Obtain the probabilities of belonging to clusters
    cluster_probabilities = clusterer.probabilities_

    # Convert the cluster labels and probabilities to NumPy arrays
    cluster_labels_np = cluster_labels.to_numpy()
    cluster_probabilities_np = cluster_probabilities.to_numpy()

    # Add cluster labels and probabilities to df_data
    df_data['cluster'] = cluster_labels_np
    df_data['cluster_probabilities'] = cluster_probabilities_np

    # Create c-TF-IDF for each cluster
    top_n = 5  # Number of top terms to display per cluster

    path = f"./results_all/{app}/class_{label}"
    os.makedirs(path, exist_ok=True)

    # Save df_data
    df_data.to_csv(f"{path}/{app}_all_{label}.csv", index=False)

    # Dictionary to store DataFrames for each cluster
    cluster_term_dataframes = {}

    # Loop through clusters to compute top terms and scores
    for cluster_id in np.unique(cluster_labels_np):
        cluster_data = df_data[df_data['cluster'] == cluster_id]

        # Convert frozenset to list for stop words
        stop_words_list = list(ENGLISH_STOP_WORDS)

        # Apply CountVectorizer on the 'text' column of each cluster, with stop word removal
        vectorizer = CountVectorizer(stop_words=stop_words_list)
        X = vectorizer.fit_transform(cluster_data['text'].values)
        
        # Apply TF-IDF transformer
        transformer = TfidfTransformer()
        X_tfidf = transformer.fit_transform(X)

        # Get feature names (terms)
        terms = vectorizer.get_feature_names_out()

        # Aggregate TF-IDF scores for each term in this cluster
        tfidf_scores = np.sum(X_tfidf.toarray(), axis=0)

        top_n = -1

        # Get the top terms for this cluster
        top_indices = np.argsort(tfidf_scores)[::-1][:-1]
        top_terms = [terms[i] for i in top_indices]
        top_scores = [tfidf_scores[i] for i in top_indices]

        # Create a DataFrame for the current cluster
        cluster_df = pd.DataFrame({
            'Term': top_terms,
            'TF-IDF Score': top_scores
        })

        # Store the DataFrame in the dictionary with the cluster_id as the key
        cluster_term_dataframes[cluster_id] = cluster_df

    # At this point, `cluster_term_dataframes` holds a DataFrame for each cluster, indexed by cluster ID
    # print(cluster_term_dataframes[0])

    for cluster_id in np.unique(cluster_labels_np):
        path = f"results/{app}/class_{label}"
        os.makedirs(path, exist_ok=True)
        cluster_term_dataframes[cluster_id].to_csv(f"{path}/{cluster_id}.csv")

    # Calculate the sizes of each cluster
    cluster_sizes = df_data['cluster'].value_counts().reset_index()

    # Rename columns for clarity
    cluster_sizes.columns = ['Cluster', 'Size']

    # Sort the DataFrame by size in descending order
    cluster_sizes = cluster_sizes.sort_values(by='Size', ascending=False).reset_index(drop=True)

    print(cluster_sizes)

    # Create a scatter plot using Matplotlib
    plt.figure(figsize=(10, 8))

    # Plot the UMAP-reduced data points, color-coded by cluster label
    scatter = plt.scatter(embedding_umap_cpu[:, 0], embedding_umap_cpu[:, 1], c=cluster_labels_np, cmap='Spectral', s=50, alpha=0.8)

    # Add a legend for the clusters
    legend1 = plt.legend(*scatter.legend_elements(), title="Clusters")
    plt.gca().add_artist(legend1)

    # Add title and labels
    plt.title('UMAP projection of the embeddings')
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')

    # Save the figure
    plt.savefig(f"{app}_embeddings/{label}_{app}.png")  # Save as PNG file

    # Assuming texts contains your processed text data
    texts = [doc.split() for doc in df_data['text'].values]

    # Create a Gensim dictionary from the text data
    dictionary = gensim.corpora.Dictionary(texts)

    # Calculate topic coherence using Gensim
    global_coherence_score = calculate_gensim_coherence(cluster_term_dataframes, 5, texts, dictionary)

    # Print the coherence score and the time taken
    print(f"Global Topic Coherence (Gensim): {global_coherence_score}")

    topic_coherence_scores.append((global_coherence_score, label))

    # Prepare model output in the format required by Octis
    model_output = {"topics": []}  # Initialize as a dictionary with "topics" key

    # Loop through each cluster to extract top-N words
    for cluster_id, cluster_df in cluster_term_dataframes.items():
        top_words = cluster_df['Term'].values[:top_n]  # Get the top-N words for this cluster
        model_output["topics"].append(top_words.tolist())  # Append the top words for this topic

    # Initialize the TopicDiversity metric
    metric = TopicDiversity(topk=10)

    # Calculate the Topic Diversity score
    topic_diversity_score = metric.score(model_output)

    print(f"Topic Diversity: {topic_diversity_score}")

    # Store the result
    topic_diversity_scores.append((topic_diversity_score, label))

path = f"results/metrics"
os.makedirs(path, exist_ok=True)

with open(f"{path}/{app}_topic_coherence.txt", "w", encoding="utf-8") as f:
    for coh, lbl in topic_coherence_scores:
        f.write(f"Average Topic_Coherence ({lbl}): {coh}\n")

path = f"results/metrics"
os.makedirs(path, exist_ok=True)

with open(f"{path}/{app}_topic_diversity.txt", "w", encoding="utf-8") as f:
    for div, lbl in topic_diversity_scores:
        f.write(f"Topic_Diversity ({lbl}): {div}\n")

