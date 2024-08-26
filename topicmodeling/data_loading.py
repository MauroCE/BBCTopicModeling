import os
import glob
import numpy as np
from typing import Optional


def load_data(path: str, categories: list[str], shuffle: bool = True,
              rng: Optional[np.random.Generator] = None) -> tuple[list[str], list[str]]:
    """Combines BBC articles into a single list of strings and a list of their
    corresponding categories.

    Parameters
    ----------
    :param path: Path to BBC dataset
    :type path: str
    :param categories: List of BBC categories, corresponding to sub-folders
    :type categories: list
    :param shuffle: Whether to shuffle the data
    :type shuffle: bool
    :param rng: Random number generator for reproducibility
    :type rng: np.random.Generator
    :return articles, labels: Articles and their corresponding labels
    :rtype: Tuple[List[str], List[str]]
    """
    articles_list, labels_list = [], []
    for category in categories:
        files = glob.glob(os.path.join(path, category, "*.txt"))  # only text files
        for file in files:
            with open(file, "rt") as f:
                article = f.read().strip()
                articles_list.append(article)
                labels_list.append(category)
    n_articles = len(articles_list)
    if shuffle:
        gen = np.random.default_rng(seed=np.random.randint(0, 1000)) if rng is None else rng
        indices = gen.choice(n_articles, size=n_articles, replace=False)
        articles_list = np.array(articles_list)[indices].tolist()
        labels_list = np.array(labels_list)[indices].tolist()
    return articles_list, labels_list
