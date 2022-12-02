
[mention detection missing here...]


# Overview
After the mention detection step, there are two input data sources.
1. The text to be processed, with labeled mentions $m_i$ and "local context" $c_i$.
    - Flair calculates contextual string embeddings for each word. This is used to tag the mentions, but it is not used anywere else in the software
    - <mark>*What is the local context? The 50 words around the mention? Or their embeddings? How is it in the code?*</mark>
2. Wikipedia corpus
    - $\mathbf{e}$: entity embedding vector
    - $\mathbf{w}$: word embedding vector
    - They are computed with Wikipedia2Vec and stored in the database 
    - The database only has one table with "embeddings", and they are at the word level (and not at the entity level). 


## Candidate selection
- 4 candidates as top-ranked entities from the prior $p(e|m)$
    - Intuitively, it is more likely that a word refers to a particular entity, the more the word carries a hyperlink to that entity (=wikipedia entry) -- relative to other entities. 
- 3 candidates based on the similarity to the context of the mention
    - The key term here is $\mathbf{e}^T \sum_{w \in c}\mathbf{w}$. $c$ is the 50-word context around the mention m. 
    - We compare the contextual embeddings -- all extracted from wikipedia -- of the entity and the context of the mention. This score is higher if the embeddings are more similar (in vector space). In turn, if the embedding vectors are perpendicular, the score is 0.
    - Intuitively, we want candidate entities that are often used context similar to the context of the words that are close to the mention.
    - While we could in principle extract embeddings also from the input text and use them for $\mathbf{w}$, this does not work in practice. Not only may the embeddings not have the same dimension, but also the models are trained on different underlying data, and so we cannot compare the embeddings $\mathbf{w}$ and $\mathbf{e}$ with each other.



## Entity disambiguation

For each mention and its candidate entities, we want to find the one entity to which the mention most plausibly refers. 
Remember -- whenever the paper refers to the "embedding of an entity", it is necessarily the embedding from wikipedia because the entity in the input text is not given. 

- The final classification model has two inputs. The first is the prior $p(e|m)$ from the previous step. The second is an estimated probability $\hat{q}(e_i | D)$ that the mention refers to entity $e_i$ (notation taken from Le/Titov 2018).
- The probability $\hat{q}(e_i | D)$  is calculated in another model. This is a score telling us how likely mention $m_i$ refers to entity $e_i$. The score is higher,
    1. The better entity $e_i$ fits into the local context of the mention. This is a function defined in Ganea & Hofmann 2017. *Check definition/estimation of function*
    2. The more coherent the assigned entity $e_i$ and all the assigned entities for the other mentions are "globally". In other words, oftentimes it may be that two mentions in the same document refer to entities that are somehow related to each other, ie "World Cup" refers to the football world cup, and the mention "England" may then refer to the English football team. 
    - This model uses the following inputs
        - The embeddings of mention $\mathbf{e_i}$ (from wikipedia)
        - The local context $c_i$ <mark>*-- again what is this exactly?*</mark> 
    - The global context compares two mentions $i$ and $j$ and their assigned entities. It uses two inputs
        - The similarity of the embeddings $\mathbf{e_i}$ and $\mathbf{e_j}$, and a weighting matrix $\mathbf{R_k}$ that gives weights to the embedding dimensions. Global coherence is higher if the entities from two mentions have more similar embeddings.
        - There are a given number of latent relations, and some may be more important than others. This is captured by a weight, formed by a fraction.
            - Numerator: the weight is higher if the related mentions $i$ and $j$ and their surrounding words have similar embeddings (single-layer neural network, $f(c_i, m_i)$).  <mark>*how is $c_i$ used here? is it the words, or the corresponding embeddings in wikipedia?*</mark> 
            Again, there is a diagonal weighting matrix $D_k$ giving importance to different dimensions of the output of $f$. 
            - Denumerator: normalization, here ment-norm. Mention $i$ can be related to all other mentions $j$, and we normalize within mention $i$, across mentions $j$.


## Open questions, by importance
1. What is the local context $c_i$? * -- perhaps see `mulrel_ranker.MulRelRanker()`. it refers to the G&H local model context token attention
2. What is the intuition behind the "latent" relations? What is incorporated in *one* relation? A link between two mentions? Something else? How are they implemented in the code?
3. what is the disadvantage of Kolitsas, the state of the art at the time of writing? 
    - speed?
    - not open source?
    - ...?
4. How is the whole thing estimated? As far as I understand, the three major steps in the major are independent. But then the paper refers to Kolitsas et al, which seems to estimate everything in one sweep? // How does REL differ from the model in Kolitsas et al?


