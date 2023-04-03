### My Transformer

Just writing a transformer from scratch (using Tensorflow) to understand the basic 
architecture.

Encoder works. It builds, trains, and tested on a toy data set. The test is:

1) Get a small corpuses of sentences (used Kaggle data set of yelp sentence sentiments).
2) Tokenize and pad the sentences
3) Last sentence is dropped, and classifier built on encoder to predict last word
4) Train the model
5) Write similar sentences, and predict the last words of each
