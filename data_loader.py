import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

class MNISTDataLoader:
    def __init__(self, batch_size=32, val_ratio=0.1, data_dir='./data'):
        self.batch_size = batch_size
        self.val_ratio = val_ratio
        self.data_dir = data_dir
        self.transform = transforms.Compose([transforms.ToTensor(), ])

    def get_loaders(self):
        train_dataset = datasets.MNIST(self.data_dir, train=True, download=True, transform=self.transform)
        val_size = int(self.val_ratio * len(train_dataset))
        train_size = len(train_dataset) - val_size
        if val_size <= 0:
            train_subset = train_dataset
            val_subset = torch.utils.data.Subset(train_dataset, [])
        else:
            train_subset, val_subset = random_split(train_dataset, [train_size, val_size])

        train_loader = DataLoader(train_subset, batch_size=self.batch_size, shuffle=True)

        val_loader = DataLoader(val_subset, batch_size=self.batch_size, shuffle=False)

        test_dataset = datasets.MNIST(self.data_dir, train=False, download=True, transform=self.transform)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        return train_loader, val_loader, test_loader


def check_data(data_loader):
    batch = next(iter(data_loader))
    data, labels = batch
    print(f"数据形状: {data.shape}")
    img = data[0].squeeze().cpu().numpy()
    plt.imshow(img, cmap='gray')
    plt.title(f'label={labels[0].item()}, min={data[0].min():.3f}, max={data[0].max():.3f}')
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    dl = MNISTDataLoader(batch_size=64, val_ratio=0.2)
    train_loader, val_loader, test_loader = dl.get_loaders()
    check_data(train_loader)