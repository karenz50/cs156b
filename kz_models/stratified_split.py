from torch.utils.data import Subset
import random

def stratified_train_val_split(dataset, val_fraction=0.2, label_key="label", seed=42):
    """
    splits a dataset into training and validation sets ensuring at least
    one positive and one negative example in the validation set.

    Parameters:
        dataset (Dataset): The dataset to split. Must return dicts with a "label" key.
        val_fraction (float): Fraction of data to use for validation.
        label_key (str): The key to access labels in each sample.
        seed (int): Random seed for reproducibility.

    Returns:
        train_dataset (Subset), val_dataset (Subset)
    """
    random.seed(seed)

    # Separate indices by class
    positive_indices = [i for i, sample in enumerate(dataset) if sample[label_key] == 1.0]
    negative_indices = [i for i, sample in enumerate(dataset) if sample[label_key] == 0.0]

    # Shuffle
    random.shuffle(positive_indices)
    random.shuffle(negative_indices)

    # Determine validation count from each class
    val_pos_count = max(1, int(val_fraction * len(positive_indices)))
    val_neg_count = max(1, int(val_fraction * len(negative_indices)))

    val_indices = positive_indices[:val_pos_count] + negative_indices[:val_neg_count]
    train_indices = positive_indices[val_pos_count:] + negative_indices[val_neg_count:]

    random.shuffle(train_indices)
    random.shuffle(val_indices)

    return Subset(dataset, train_indices), Subset(dataset, val_indices)