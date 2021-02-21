from torchvision import datasets, transforms
import torch

class Data:
    def __init__(self):
        transformation = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.1307,), std=(0.3081,))])

    def MNIST(self):
        train = datasets.MNIST(root="./data",train=True,transform= transformation, download=True)
        test = datasets.MNIST(root="./data",train=False,transform= transformation, download=True)
        seed = 1
        cuda = torch.cuda.is_available()
        torch.manual_seed(seed)
        if cuda:
            torch.cuda.manual_seed(seed)
        dataloader_args = dict(shuffle=True, batch_size=128, num_workers=4, pin_memory=True) if cuda else dict(
            shuffle=True, batch_size=64)
        self.train_loader = torch.utils.data.DataLoader(dataset=train, **dataloader_args)
        self.test_loader = torch.utils.data.DataLoader(dataset=test, **dataloader_args)
        return self.train_loader, self.test_loader