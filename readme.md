### My Transformer

Just writing a transformer from scratch (using Tensorflow) to understand the basic 
architecture. Following so far examples from *Natural Language Processing With Transformers*
chapter 3, but rewriting using Tensorflow rather than Pytorch and putting it all together for
it to actually work.

Encoder works. It builds, trains, and tested on a toy data set. The test is:

1) Get a small corpus of sentences (used Kaggle data set of yelp sentence sentiments).
2) Tokenize and pad the sentences
3) Last word in each sentence is dropped, and classifier built on encoder to predict last word in each
4) Train the model (end to end), training the encoder and classifier
5) Write similar sentences, and predict the last words of each

Then, use the now pretrained encoder and use it in transfer learning for sentiment analysis.

1) Load the pretrained encoder model
2) Add new classifier head, whether positive or negative sentiments on review sentences
3) Train on the same corpus (turns out fine tuning the encoder helps)
4) Write similar sentences and predict their sentiments

It is just a toy data set and model so far, but POC that the encoder in fact trains and can be used.
