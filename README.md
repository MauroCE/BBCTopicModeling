# BBC Topic Modeling
Unsupervised BBC News Topic Modeling using BERTopic. Although BBCNews is typically a benchmark dataset for supervised document classification, in this repo we will approach it as an unsupervised learning problem.

### How to use this repo
To use this repo:
1. Clone the repo `git clone https://github.com/MauroCE/BBCTopicModeling.git`
2. Create a virtual environment and install dependencies `python3 -m venv bbctopic`, `source bbctopic/bin/activate`, `pip install -r requirements.txt`.
3. Download [data](http://mlg.ucd.ie/files/datasets/bbc-fulltext.zip), unzip it and change the name of the resulting folder to `data`.
4. (Optional) create `.env` file and set `TOKENIZERS_PARALLELISM=false`

### Model
I settled for BERTopic due to its ease of use, modularity (choices of clustering, dimensionality reduction, embeddings, etc) and good documentation. Additionally, it supports many machine learning modalities off-the-shelf, including unsupervised topic modeling, guided topic modeling and supervised document classification, among many others. 
Another reason for choosing this model/package is that it can seamlessly be used for dealing with a stream of incoming documents by using different clustering/dimensionality-reduction algorithms, as explained [here](https://maartengr.github.io/BERTopic/getting_started/online/online.html).

### Evaluation
Evaluating an unsupervised topic model is not straightforward and requires a number of choices. Due to time constraints, here I have focused on evaluation metrics as provided by `gensim` and, when BERTopic is instantiated with `HDBSCAN`, I also compute `perplexity`. I have left a notebook where I run BERTopic with k-means (with `5` centroids) so that this can be compared directly with the original categories the articles were in. 

### Dataset
To work with this repo, download the [raw data](http://mlg.ucd.ie/files/datasets/bbc-fulltext.zip), unzip it and call it `data`. The code in this repo expects the `data` folder to have sub-folders for each category `tech`, `politics`, `business`, `entertainment`, `sport`. This is exactly the structure obtained upon unzipping the data from the url above.  

### Testing
Testing is performed with `pytest` as it requires less boilerplate code and more plug-in functionalities. Due to time constraint, I have focused on testing the data loading and data formatting. In a production environment one would need to list all the features, inputs and expected outputs in the pipeline and test them exhaustively.

### Notes
The file `.env` is not being tracked. If you want to suppress the tokenizer warning simply create a `.env` file and set `TOKENIZERS_PARALLELISM=false`.
