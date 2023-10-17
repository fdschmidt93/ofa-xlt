from datasets import load_dataset

def load_stratified_shots(*args, **kwargs):
    shots = kwargs.pop("shots")
    seed = kwargs.pop("seed")
    dataset = load_dataset(*args, **kwargs)
    return dataset.train_test_split(train_size=shots, seed=seed)['train']
