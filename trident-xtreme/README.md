# trident-xtreme

trident-xtreme is a project to implement a modular deep learning stack to train and evaluate models with trident. 

It enables you to modularly re-use the complete functionality across NLP tasks for your model by:

- Declaratively defining the entire training pipeline
- Large re-use of functionality across
- Integrated with logger of your choice (eg weights & biases)

This highly simplifies re-producability, encourages modularity, and sharing functionality across tasks. In particular, the goal is to only provide functionality relating to your own model, while the remainder of the pipeline stays fixed.

# Status

The following tasks are currently implemented:

- Text Classification on MNLI / XNLI
- Question Anwering on datasets like SquaD
- Token Classification for NER on CoNLL
- Parallel Sentence Retrieval on tatoeba

The current focus is to implement the entire [XTREME](https://sites.research.google/xtreme/) and associated tasks, which requires providing necessary functionality in upstream `trident-core` to fully enable declaratively defining complex pipelines. For the time being it is recommended to subscribe to the repository updates as `trident(-xtreme)` is aimed to be released as an initial version during the week of 6th of December. We plan on replicating existing results on `xlm-roberta` to demonstrate our framework works functional.

# Getting started

Once more, please note that the project is a work-in-progress. In case you are curious about becoming an early adopter of `trident(-xtreme)`, you are greatly encouraged to take a deep dive into [hydra](https://hydra.cc) and [pytorch-lightning](https://pytorch-lightning.readthedocs.io/). The documentation of `trident-core` introduces how to tailor with pipeline (albeit, the documentation requires some overhaul).

For the time being, text classification based on MNLI / XNLI a great task to get introduced to the framework which is fully functional already. You'd execute training on `MNLI` with zero-shot transfer to `XNLI` (German) with

```python
# only train for a 100 steps
# do not include any callbacks
python run.py experiment=mnli_tinybert +trainer.max_steps=100 callbacks=none
```

`hydra` integrates the CLI with your declarative configuration. Any configured setting can be interactively overriden (`callbacks=none`) or added anew (`+trainer.max_steps`). For more information, please see the hydra documentation.
