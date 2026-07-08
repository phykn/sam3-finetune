from torch.utils.data import Dataset

from .sample import load


class BaseDataset(Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        return load(self.paths[index])
