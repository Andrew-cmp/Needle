import sys

sys.path.append("../python")
import needle as ndl
import needle.nn as nn
import numpy as np
import time
import os

np.random.seed(0)
# MY_DEVICE = ndl.backend_selection.cuda()


def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
    ### BEGIN YOUR SOLUTION
    modules = nn.Sequential(
        nn.Linear(dim, hidden_dim),
        norm(hidden_dim),
        nn.ReLU(),
        nn.Dropout(drop_prob),
        nn.Linear(hidden_dim, dim),   
        norm(dim)
    )
    return nn.Sequential(
        nn.Residual(modules),
        nn.ReLU()
    )
    

    ### END YOUR SOLUTION


def MLPResNet(
    dim,
    hidden_dim=100,
    num_blocks=3,
    num_classes=10,
    norm=nn.BatchNorm1d,
    drop_prob=0.1,
):  
    
    ### BEGIN YOUR SOLUTION
    resnet = nn.Sequential(nn.Flatten(),nn.Linear(dim, hidden_dim), nn.ReLU(), 
    # 重要的是这里的hidden_dim=hidden_dim//2
                           *[ResidualBlock(dim=hidden_dim, hidden_dim=hidden_dim//2, norm=norm, drop_prob=drop_prob) for _ in range(num_blocks)],
                           nn.Linear(hidden_dim, num_classes))
    return resnet
    ### END YOUR SOLUTION


def epoch(dataloader, model, opt=None):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    losses = []
    total_acc = 0
    if opt is not None:
        model.train()
    else:
        model.eval()
    for i, batch in enumerate(dataloader):
        x, y = batch[0], batch[1]
        if opt is not None:
            opt.reset_grad()
        y_pred = model(x)
        loss = nn.SoftmaxLoss()(y_pred, y)
        losses.append(loss.numpy())
        total_acc += (y_pred.numpy().argmax(axis=1) == y.numpy()).sum()
        
        if opt is not None:
            loss.backward()
            opt.step()
        else:
            print(loss)
            
    mis_samples_rate = 1 - total_acc / len(dataloader.dataset)
    avg_loss = np.mean(losses)
    
    # dataloader是没有len的，要用len(dataloader.dataset)
    # avg_loss /= len(dataloader)
    # mis_samples_rate /= len(dataloader)
    print("Average loss:%f for the %d epoch", avg_loss,i)
    print("Misclassified samples rate:%f for the %d epoch", mis_samples_rate,i)
    return (mis_samples_rate, avg_loss)
    ### END YOUR SOLUTION


def train_mnist(
    batch_size=100,
    epochs=10,
    optimizer=ndl.optim.Adam,
    lr=0.001,
    weight_decay=0.001,
    hidden_dim=100,
    data_dir="data",
):
    
        
    ### BEGIN YOUR SOLUTION
    
    # Load the data
    train_data_path = os.path.join(data_dir, "train-images-idx3-ubyte.gz")
    train_label_path = os.path.join(data_dir, "train-labels-idx1-ubyte.gz")
    
    test_data_path = os.path.join(data_dir, "t10k-images-idx3-ubyte.gz")
    test_label_path = os.path.join(data_dir, "t10k-labels-idx1-ubyte.gz")
    
    train_data = ndl.data.datasets.MNISTDataset(train_data_path, train_label_path)

    test_data = ndl.data.datasets.MNISTDataset(test_data_path, test_label_path)
    train_loader = ndl.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = ndl.data.DataLoader(test_data, batch_size=batch_size)
    model = MLPResNet(784, hidden_dim=hidden_dim)
    optimizer = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)
    for i in range(epochs):
        train_mis_samples_rate, train_avg_loss = epoch(train_loader, model, optimizer)
        test_mis_samples_rate, test_avg_loss = epoch(test_loader, model)
        
    return (1-train_mis_samples_rate, train_avg_loss, 1-test_mis_samples_rate, test_avg_loss)
    ### END YOUR SOLUTION


if __name__ == "__main__":
    train_mnist(data_dir="../data")
