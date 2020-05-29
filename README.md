# RNN-Based-Deep-Language-Modeling
Demonstration of various cutting edge deep language modeling techniques, including variational autoencoders (VAEs), and application to a Twitter politician dataset

## Abstract

In this project we will use PyTorch to train several different flavors of RNN-based deep language models, demonstrating the merits, drawbacks, and challenges that each model faces. The first model, a vanilla RNN-LM, will be trained on the Penn Tree Bank (PTB) dataset and uses the log data likeli- hood as its training objective. We will train the second model, a variational autoencoder (VAE), on the same dataset using two different RNN architectures: a traditional RNN and a GRU. Despite being state-of-the-art language modelling tech- niques, VAEs face several that we explore solutions to, in- cluding KL-annealing and free bits (KL thresholding). In the final part of this report, we apply the variational autoen- coder to a Twitter dataset (tweets from politicians) and ex- plore what sort of information is encoded into the latent space after training.


## Introduction

Last project we trained a traditional Ngram-based language model using interpolated Kneser-Ney trigram modeling. In this project, we set out with the same objective: to maximize the likelihood of the observed data, using previous words as context for the distribution on the next word. When trying to achieve this goal with the Ngram model, we came across a few different obstacles which we will see RNNs are inherently well poised to solve.

First, the Ngram model was not flexible in the number of words it could consider in its context: for the trigram model, we were forced to always consider two words as the context for the next word. This rigid structure is a pain at the beginning of sentences when we don’t have any context, and is also inconvenient in the middle and end of sentences because it ignores out-of-context words that might have semantic value. RNN’s inherently overcome this with their ability to recurrently accept new data at each time step and thus allow contexts of varying lengths.

The other big issue that arose in the Ngram model was sparsity, which grew linearly with the size of the vocabulary and combinato- rially with the order of the Ngram. This is because Ngram models rely on counts of words and specific word contexts, and hence we cannot use word embeddings to cluster words with similar seman- tics. In contrast when we model using RNN-based techniques, we can use pre-trained word embeddings to embed words with similar semantics close to each other.

Aside from using RNNs to solve some of the big issues that Ngram models have, we can also use them as a vehicle introduce latent variables to learn high level information about the data source we are modeling. These variational autoencoders can capture tone, topic, sentiment, and many other types of information that can be viewed as having an influence on how the observed data was produced. We will see that training these VAEs comes with it’s own set of challenges, and we will explore several techniques for overcoming them. The rest of this paper is organized as follows:
- Section 2: Intuition and theory behind RNN-based LMs, the variational autoencoder, and the issue of posterior collapse
- Section 3: Dynamics of training the VAE, amortized inference, and measuring performance
- Section 4: Using the KL-annealing and free bits techniques to protect the model from posterior collapse
- Section 5: Exploration of the type of information that the latent space can encode using a novel Twitter dataset.
