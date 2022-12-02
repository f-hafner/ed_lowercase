import numpy as np

from REL.training_datasets import TrainingEvaluationDatasets
from REL.entity_disambiguation import EntityDisambiguation

np.random.seed(seed=42)

base_url = "/home/flavio/projects/rel20/data_wip" # do whatever you want in this directory. in the worst case, delete all files in here and copy from data/.
wiki_version = "wiki_2019"

datasets = TrainingEvaluationDatasets(base_url, wiki_version).load()

config = {
    "mode": "train",
    "n_epochs": 2,
    "model_path": "{}/{}/generated/model".format(
        base_url, wiki_version
    ),
}
model = EntityDisambiguation(base_url, wiki_version, config)


model.train(
        datasets["aida_train"], # org_train_dataset -- what does it do?
        {k: v for k, v in datasets.items() if k != "aida_train"} # org_dev_datasets -- what does it do?
    )