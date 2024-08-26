import logging
from data_loading import load_data
from topic_models import setup_bertopic
from evaluation import evaluate_bertopic
from config import BBC_PATH, CATEGORIES, CLUSTERING, EMBEDDING_MODEL_NAME, VERBOSE
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    # Currently, only used to suppress a warning, but could be used for API keys if .gitignore-d
    load_dotenv()

    # Load BBC Data
    articles, labels = load_data(path=BBC_PATH, categories=CATEGORIES, shuffle=True)

    # Instantiate BERTopic and pre-compute embeddings
    topic_model, article_embeddings, emb_model = setup_bertopic(
        sentences=articles, emb_model_name=EMBEDDING_MODEL_NAME, verbose=VERBOSE, clustering=CLUSTERING
    )

    # Fit using pre-computed embeddings
    topics, probs = topic_model.fit_transform(documents=articles, embeddings=article_embeddings)

    # Evaluate coherence and perplexity
    eval_metrics = evaluate_bertopic(model=topic_model, predictions=topics, docs=articles, probabilities=probs)
    logger.info(f"Perplexity: {eval_metrics['perplexity']}")
    for key, value in eval_metrics["coherence"].items():
        logger.info(f"{key.capitalize()} coherence: {value}")
