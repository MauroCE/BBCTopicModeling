import re
from typing import List
from numpy.typing import NDArray


def simple_preprocessing(documents: NDArray) -> List[str]:
    """Replaces newlines and tabs with white space, and keeps only alphanumeric characters.

    Parameters
    ----------
    :param documents: Array of documents to preprocess
    :type documents: np.ndarray
    :return cleaned_documents: Documents of alphanumeric characters, without new lines or tabs
    :rtype cleaned_documents: list

    Notes
    -----
    This function is identical to `BERTopic._preprocess_text` and I have only replicated it here to avoid accessing
    a protected method.
    """
    # Replace "\n" and "\t" with " "
    cleaned_documents = [doc.replace("\n", " ").replace("\t", " ") for doc in documents]
    # Only alphanumeric characters
    cleaned_documents = [re.sub(r"[^A-Za-z\d ]+", "", doc) for doc in cleaned_documents]
    # Empty documents are mapped to "emptydoc"
    cleaned_documents = [doc if doc != "" else "emptydoc" for doc in cleaned_documents]
    return cleaned_documents
