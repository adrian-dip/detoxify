# Detoxify

Detoxify is a collection of Python linear text classifiers that filter highly inappropriate or offensive text. This repository includes a Chrome extension to serve the models. 

#### Computational resources

The models consume 10 seconds of 2.2 Ghz CPU time (one thread) for every 5 hours of regular web browsing (sequential requests). Batch processing is much faster and takes slightly over a minute to preprocess and compute 1 millon examples. 

#### Requirements

The models can be incorporated into any Python pipeline. You only need to follow the steps provided in the Flask example, `flask_server.py`. They are:

1) Tokenizing and stemming the sentence 
2) Preloading the pretrained vectorizer and vectorizing the data
3) Preloading models and doing inference. 

You can replicate the files by running the code here or download them from the following link: 

#### Dataset

The core dataset is the toxic comments database by Google's Jigsaw team augmented via highly negative and highly positive Reddit comments. Comments with a score of -7 or lower and +75 or higher were selected for pseudolabeling. `deberta-small` was pretrained on the Jigsaw dataset (97% precision, 94% recall [weighted average]), and the Reddit comments were then labeled. 

Comments identified as toxic in the negative dataset and comments identified as non-toxic in the positive dataset were used to augment the training database. The rationale was that comments in the -7 dataset were more likely to be toxic, which was confirmed by a 33% positivity rate vs. 7% in the +75 dataset. This was an attempt to minimize the compounding error that comes with pseudolabeled examples. A short qualitative review confirmed that the model was working as intended.

The dataset was augmented to avoid overfitting to the original, which contains only Wikipedia discussion pages. Around 5 GB of additional data from over 175 subreddits was included in the final training.

#### Posibilities for improvement

A larger transformer like `deberta-base`, along with more complex pooling methods, such as an LSTM before the linear head, could improve recall and reduce the error in the pseudolabeling step. A more diverse database would make the results more generalizable as well as improve accuracy across use cases, particularly recall for the toxic class. 

If compute is not an issue, a transformer will deliver much better results than linear methods. Several optimization approaches exist to make this a reality, but it must be noted that F1 scores increase greatly when making predictions on individual sentences rather than full paragraphs. The sentences can be strung together in a single forward pass to reduce the compute overhead.

#### Methodological issues

The core Jigsaw dataset has been criticized for using the average score of a pool of data labelers to determine whether a comment is toxic or not. This means that if we set the threshold at 0.5, 24 out of 50 people could think a comment is highly inappropriate and still mark it as appropriate. Conversely, if we set the threshold at 0.3, 7 people out of 10 could mark something as acceptable, but we would label it as highly toxic. As we all know, the mode and average are not accurate indicators of multi-rater labels of complex concepts like high-order semantics.

However, it must be noted that as we increase the threshold, transformers have a much easier time separating the two classes, which points to a correlation between the number of people who find something offensive and the actual offensiveness of a comment. 
