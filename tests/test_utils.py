from topicmodeling.utils import simple_preprocessing
import numpy as np
import re


def test_simple_preprocessing():
    docs = np.array(["document with newline \n", "document with tab \t", "document with non-alphanumeric #%£"])
    cleaned_docs = simple_preprocessing(documents=docs)
    assert isinstance(cleaned_docs, list)
    assert len(docs) == len(cleaned_docs)
    assert all(substring not in cleaned_docs for substring in ["\n", "\t"])
    assert all([re.match(r"^[\w-]+$", doc) is None for doc in cleaned_docs])
