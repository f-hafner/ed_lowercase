

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

