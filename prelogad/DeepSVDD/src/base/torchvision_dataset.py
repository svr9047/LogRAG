from .base_dataset import BaseADDataset
from torch.utils.data import DataLoader


class TorchvisionDataset(BaseADDataset):
    """TorchvisionDataset class for datasets already implemented in torchvision.datasets."""

    def __init__(self, root: str):
        super().__init__(root)

    def loaders(self, batch_size: int, shuffle_train=True, shuffle_test=False, num_workers: int = 0) -> (
            DataLoader, DataLoader):
        print("DL train_set type:", type(self.train_set))
        try:
            print("DL len(train_set):", len(self.train_set))
        except Exception as e:
            print("len(train_set) raised:", e)
        print("Batch size:", batch_size, "Shuffle train:", shuffle_train, "Shuffle test:", shuffle_test, "Num workers:", num_workers)
        train_loader = DataLoader(dataset=self.train_set, batch_size=batch_size, shuffle=shuffle_train,
                                  num_workers=num_workers)
        test_loader = DataLoader(dataset=self.test_set, batch_size=batch_size, shuffle=shuffle_test,
                                 num_workers=num_workers)
        return train_loader, test_loader
