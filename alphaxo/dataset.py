from torch.utils.data import Dataset


class DatasetXO(Dataset):
    def __init__(self, ):
        # nested numpy arrays of size (n, n), each represents the particular position
        self.s = []

        # nested numpy arrays of size (n * n,) each represents the probability of action in the corresponding position
        self.pi = []

        # list of the outcomes from the first player's point of view in the corresponding position
        self.z = []

    def __getitem__(self, index):
        return self.s[index], self.pi[index], self.z[index]

    def __len__(self):
        return len(self.s)
