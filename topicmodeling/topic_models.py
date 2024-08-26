import numpy as np
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer


def setup_bertopic(sentences: list[str], emb_model_name: str = "all-MiniLM-L6-v2",
                   clustering: str = "hdbscan", calc_probs: bool = True,
                   seed: int = 42, verbose: bool = False) \
        -> tuple[BERTopic, np.ndarray, SentenceTransformer]:
    """Instantiate BERTopic while allowing for some flexibility.

    Parameters
    ----------
    :param sentences: Corpus of documents to be fed to the embedding model
    :type sentences: List[str]
    :param emb_model_name: Embedding model name from SentenceTransformers
    :type emb_model_name: str
    :param clustering: Clustering algorithm used by BERTopic, must be either 'hdbscan' or 'kmeans'
    :type clustering: str
    :param calc_probs: Whether to calculate probabilities for each topic. Notice these are not available for 'kmeans'
    :type calc_probs: bool
    :param seed: Random seed for reproducibility
    :type seed: int
    :param verbose: Whether to print progress bars for embedding and fitting
    :type verbose: bool
    :return: Tuple containing the BERTopic model, embeddings, and the SentenceTransformer model
    :rtype: Tuple[BERTopic, np.ndarray, SentenceTransformer]
    """
    assert clustering in ["hdbscan", "kmeans"], "Clustering must be either 'hdbscan' or 'kmeans'."
    # Pre-computing embeddings can speed up experimentation and hyperparameter tuning
    embedding_model = SentenceTransformer(emb_model_name, tokenizer_kwargs={"clean_up_tokenization_spaces": True})
    embeddings = embedding_model.encode(sentences, show_progress_bar=verbose)

    # A fixed random state for the dimensionality reduction is advised for stability and reproducibility
    umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=seed)
    if clustering == "hdbscan":
        clustering_model = HDBSCAN(min_cluster_size=150, metric='euclidean', cluster_selection_method='eom',
                                   prediction_data=True)
    else:
        clustering_model = KMeans(n_clusters=5, random_state=seed)
    vectorizer_model = CountVectorizer(stop_words="english", min_df=2, ngram_range=(1, 2))
    representation_model = KeyBERTInspired()  # advised for better results, see BERTopic best-practices
    bert = BERTopic(
        # Pipeline models
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=clustering_model,
        vectorizer_model=vectorizer_model,
        representation_model=representation_model,
        top_n_words=10,
        verbose=verbose,
        calculate_probabilities=calc_probs
    )
    return bert, embeddings, embedding_model
