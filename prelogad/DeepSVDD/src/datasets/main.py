from .bgl import BGL_Dataset

def load_dataset(data_path, encoder_path, config):
    """Loads the dataset."""

    dataset = BGL_Dataset(root=data_path, encoder_path=encoder_path, config=config)

    return dataset
