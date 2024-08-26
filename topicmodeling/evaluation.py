from bertopic import BERTopic
from typing import Optional, List
from numpy.typing import NDArray
from collections import defaultdict
import numpy as np
from hdbscan import HDBSCAN
import gensim.corpora as corpora
from gensim.models.coherencemodel import CoherenceModel
from topicmodeling.utils import simple_preprocessing


def evaluate_bertopic(model: BERTopic, docs: list[str], predictions: List[int], probabilities: Optional[NDArray] = None,
                      coherence_metrics: tuple[str] = ("c_v", 'u_mass')) -> dict[str, dict]:
    """Computes coherence metric for the topic model following BERTopic's own implementation, see
    https://github.com/MaartenGr/BERTopic/issues/90#issuecomment-820270553. Notice that perplexity can only be computed
    with a density-based clustering method (i.e. HDBSCAN) but cannot be computed with k-means.

    Parameters
    ----------
    :param model: Fitted BERTopic model
    :type model: BERTopic
    :param docs: Corpus of documents
    :type docs: List[str]
    :param predictions: Predicted topics from the model
    :type predictions: list
    :param probabilities: Topic probabilities for each document, or None (in the case of k-means)
    :type probabilities: Optional[NDArray]
    :param coherence_metrics: Coherence metrics to be used, see gensim documentation for options
    :type coherence_metrics: List[str]
    """
    metrics = {
        'coherence': defaultdict(None),
        'perplexity': None
    }
    # Compute coherence
    cleaned_docs = simple_preprocessing(np.array(docs))  # "\n", "\t" => " " and keeps only alphanumeric
    vectorizer = model.vectorizer_model
    tokenizer = vectorizer.build_tokenizer()
    tokens = [tokenizer(doc) for doc in cleaned_docs]
    dictionary = corpora.Dictionary(tokens)
    corpus = [dictionary.doc2bow(token) for token in tokens]
    topic_words = [[words for words, _ in model.get_topic(topic)]
                   for topic in range(len(set(predictions)) - 1)]
    for coherence_metric in coherence_metrics:
        coherence_model = CoherenceModel(topics=topic_words,
                                         texts=tokens,
                                         corpus=corpus,
                                         dictionary=dictionary,
                                         coherence=coherence_metric)
        metrics['coherence'][coherence_metric] = coherence_model.get_coherence()
    # Compute perplexity (only if clustering method is HDBSCAN)
    if isinstance(model.hdbscan_model, HDBSCAN):
        metrics['perplexity'] = np.exp(-1 * np.mean(np.log(np.sum(probabilities, axis=1))))
    return metrics
