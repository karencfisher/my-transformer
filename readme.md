### My Transformer

Just writing a transformer from scratch (using Tensorflow) to understand the basic 
architecture. Following so far examples from *Natural Language Processing With Transformers*
chapter 3, but rewriting using Tensorflow rather than Pytorch.

Encoder works. It builds, trains, and tested on a toy data set. The test is:

1) Get a small corpuses of sentences (used Kaggle data set of yelp sentence sentiments).
2) Tokenize and pad the sentences
3) Last word in each sentence is dropped, and classifier built on encoder to predict last word in each
4) Train the model (end to end), training the encoder and classifier
5) Write similar sentences, and predict the last words of each

It is just a toy data set and model so far, but POC that the encoder in fact trains apprently as it should.
