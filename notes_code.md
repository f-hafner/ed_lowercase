

### (1/12/22)
- function `EntityDisambiguation().self.emb.emb(words_filt, table_name)` extracts the embeddings from the database
- is there a function that calculates/uses embeddings from flair?
    - `MentionDetectionBase` loads the word embeddings from wikipedia as well
    - searching for the term does not yield any results
- so the program does not use *any* embeddings from the input text? 

Stef: see the function `get_data_items()` in the entity_disambiguation script.
- `__load_embeddings()`: loads 1 embedding for word, entity and snd (?), each one is of length 300.

## Database structure
- table `wiki`: store $p(e|m)$ for each word
    - can have duplicates ie for "England" and "ENGLAND". The the $p$ scores differ for the two words.
- table `embeddings`: for each word, one embedding
    - this is probably similar to what Flair does, ie the unobserved state around the word in all over wikipedia. -- the context in which the word "england" usually appears in England. (If necessary, read up the science behind Wikipedia2Vec.)
- database: need wiki table with only lower, and consolidate the $p(e|m)$ scores across mentions. Currently we using a random one. (How much would it change?)
- are the tables ever joined together? 


## Local context $c_i$
`REL.mulrel_ranker.PreRank()` is the method that multiplies word and entity vectors. It is used in `prerank_model` in the EntityDisambiguation class.
- `EntityDisambiguation.embeddings.keys()`: 
    ```python
    ['snd_seen', 'snd_voca', 'snd_embeddings', 
    'entity_seen', 'entity_voca', 'entity_embeddings', 
    'word_seen', 'word_voca', 'word_embeddings', 'word_embeddings_bag']
    ```
- In `mulrel_ranker.prerank_model.forward()`, the following are used
    - word_embeddings_bag of type torch.nn.modules.sparse.EmbeddingBag(X, 300, mean)
    - entity_embeddings of type torch.nn.modules.sparse.Embedding(X, 300)
        - 300 is the length of a single embedding vector
        - so X must be the number of embeddings? where are they defined?
    - `entity_ids.shape()`: torch.Size([X, 30]) -- X = len(content) = number of mentions (?)
        - 30 is the k=30 entities mentioned in part 2.2 of the paper for which the score is calculated


from **prerank_model.forward()**:
- multiply local context and candidate entity embeddings, and returns a probability distribution across the candidates
- `sent_vecs` are the "local context". They are extracted from `EntityDisambiguation.embeddings["word_embeddings_bag"]`, using the `token_ids`.
    - for one mention, this is a 300-dimensional embedding vector. I suspect this is calculated as the mean across all tokens (see the `mean` function attached to the object).
    - My understanding is that this embedding bag contains the embeddings of all tokens in the document (identified by flair), and the token_ids are used to extract the relevant embedding for each mention (the local context differs across mentions).
- `entity_vecs` are the embeddings of the candidate entities per mention
    - for N mentions, there are 30 candidate entities with a 300-dimensional embedding each. Thus, this is an N x 30 x 300 tensor.
    - they are extracted from `EntityDisambiguation.embeddings["entity_embeddings"]`, using the `entity_ids` 
    - Again, probably `EntityDisambiguation.embeddings["entity_embeddings"]` stores the embeddings of all candidate entities (30 per mention) for a given document (?)
- Next questions
    - where are the entities constructed (in the code)
    - test hypothesis:
        - the embeddings `entity_embeddings` and `word_embeddings_bag` are the embeddings of the unique words/tokens and candidate entity embeddings (30 per mention)
        - see the function `__load_embeddings()`? but not sure this loads anything, rather than just initializing the embedding dictonaries?
    - understand better again where the local context is coming from 
    - where is the function `f(c_i, m_i)` in the code?


from **`EntityDisambiguation.get_data_items()`** (I am not sure if any of this is correct.)
- `named_candidates` are the candidate entities
- `named_cands_filt` are the unique candidates
- `__embed_words` gets the embeddings from the db. `_embed_words()`  
    - is called several times
        - for entity embeddings
        - for word embeddings
        - for snd glove embeddings (??)
    - what does it do?
        - `embs = self.emb.emb(words_filt, table_name)`
            - `self.emb()` is a `GenericLookup` with first argument "entity_word_embedding", which means it points to the database "entity_word_embedding.db".
            - the `emb()` method calls `self.lookup_list(words, table_name)`
            - it looks up the word and embeddings from the table and stores them in memory.
        - look up the embedding. if checked, add to `seen`, and then it is not checked anymore for the next mention. does it also store some kind of id? `dir(self.embeddings["word_voca"])` gives a method `get_id`
        - add_to_vocab: what is the deal with `LOWER`? *--check where the vocabulary is used, and make a run to see what it looks like/what the lower option is doing there*
            - `vocabulary`:
                - size -- number of unique embeddings? double check with dimensions of the tensor 
                - how does `.get_id()` work? 
        - `self.__batch_embs[name]`: appends the embedding `e` as a tensor to torch.
        - should this not somehow return an index to look up the embedding vector later on?? 
        - it seems that `__batch_embs` stores the embeddings by name (word, entity, glove, ...?)
- `rctx` and `lctx` are the local contexts. They come from `m["context"]`
    - "context" is defined in the mention detection step, in `format_spans()`. `"context": (left_ctxt, right_ctxt),`, which comes from `MentionDetectionBase.get_ctxt`. But the surrounding context seems to be 100 words, not 50 now.
    - then we have up to 100 single words to the left and the right  
- what is `snd_lctx` and `snd_rctx`? context of the sentence?
- other questions
    - what does the `mask` do?
    - what is snd? secondary? what is it for?
    - what are all the ids? ie, `rctx_ids`, `lctx_ids`, `ment_ids`, 
    - where is the mapping between words and numbers created that serves to look up the embedding vectors in the estimation function (at least for the preranking?)
    - what is voca? vocabulary? where is this used later? 







## How does the training work? -- schematic view of the `train()` method
- set up optimizer (`torch.optim`)
- datasets
    - `train_data_set`: `predict=False` *-- ?*
    - `dev_dataset`: `predict=True` *-- ?*
- for `epoch` in (0, 1, ... `n_epochs`)
    - for doc in (0, .... n_documents) *iterate over train_dataset that consists of documents. in each `epoch`, the order of the documents is randomized*
        - `self.model.train()` -- what does this do exactly?
        - `self.model.zero_grad()` (?) -- and this?
        - convert data items to pytorch inputs
        - `self.model.forward(); .loss(); .backward(); ` *--see the paper by Le and Titov*
        - print out the `epoch`, progress (as % of all documents) and the `loss`
    - print out `epoch`, total loss, average loss per document
    - after each `eval_after_n_epochs`, the current performance of the model is assessed and printed (the recall, precision)
- In sum
    - tune the length of the training with the parameter `n_epochs`
    - how does the number of epochs impact the performance of the model? why are more epochs better?
        - in 1 epoch, the full data set is run once through the network (possibly in batches). for each batch, we make a forward and backward pass through the network. 
    - it still unclear to me why we call `train()` and `zero_grad()` in each batch
    - one explanation: we evaluate the model at the current parameters sequentially for each batch and calculate the loss. If this is the case, where are the parameters updated? 
        - the parameters change within epochs across datasets (minibatch) *(what is a batch then??)*
        - where does it happen? maybe in the `optimizer.step()` function? 
- It would be good to store the printed output from one full training to see what is going on and how long it takes

