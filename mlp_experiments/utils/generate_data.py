import torch
import numpy as np

class CustomDataset(torch.utils.data.Dataset):
        def __init__(self, data, targets, transform=None):
            """
            Custom dataset that accepts image data and labels (targets).
            
            Args:
                data (Tensor): The image data (shape: [N, C, H, W]).
                targets (Tensor): The labels (shape: [N,]).
                transform (callable, optional): A function/transform to apply to the images.
            """
            self.data = data
            self.targets = targets #torch.tensor(targets, dtype=torch.long)
            self.transform = transform  # Optional transform (such as ToTensor())

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            image = self.data[idx]
            if self.transform:
                image = self.transform(image)
            return image, self.targets[idx]
        

        
class GeneratedDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels, transform=None, device='cpu', classification=True, to_onehot = False):
        """
        Custom dataset for Gaussian-distributed covariates with labels.

        Args:
            data (np.ndarray): The generated data (shape: [n_train, d]).
            labels (np.ndarray): The labels for the data (shape: [n_train]).
            transform (callable, optional): A function/transform to apply to each sample.
        """
        self.data = torch.tensor(data, dtype=torch.float32).to(device)
        if classification:
            self.targets = torch.tensor(labels, dtype=torch.long).to(device)
        else:
            self.targets = torch.tensor(labels, dtype=torch.float32).to(device)
        self.transform = transform
        self.device = device
        self.to_onehot = to_onehot
        self.num_classes = len(torch.unique(self.targets))
        if self.to_onehot:
            self.targets = torch.nn.functional.one_hot(self.targets, num_classes=self.num_classes).float()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.targets[idx]
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample, label
    


class OneHotDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, num_classes=2):
        self.dataset = dataset
        self.num_classes = num_classes
    
    def __getitem__(self, index):
        img, label = self.dataset[index]
        
        # Convert label to one-hot encoding
        one_hot = torch.zeros(self.num_classes)
        one_hot[label] = 1.0
        
        return img, one_hot
    
    def __len__(self):
        return len(self.dataset)
    








def generate_sparse_gaussian_training_set(n_train, d, sigma):
    #generate gaussian data with diagonal covariance matrix filling up to d dimensions with sigma[-1]
    variances = np.array(sigma)
    covariances = np.diag(variances)
    if d > len(sigma):
        last_var = sigma[-1]
        extra_dims = d - len(sigma)
        extra_cov = last_var*np.eye((extra_dims))
        covariances = np.block([[covariances, np.zeros((covariances.shape[0], extra_dims))],
                                [np.zeros((extra_dims, covariances.shape[1])), extra_cov]])
    
    # Generate the training set using multivariate normal distribution
    X = np.random.multivariate_normal(np.zeros(d), covariances, size=n_train)
    
    return X
