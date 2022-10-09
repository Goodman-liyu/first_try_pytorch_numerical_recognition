from model import *
from dataset import *

learning_rate = 1e-5
batch_size = 64
epochs = 5  # 重复训练次数
loss_fn = nn.CrossEntropyLoss()
model = NeuralNetwork().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def start_train():
    for t in range(epochs):
        print(f"epoch {t + 1}\n------------------")
        train_loop(train_loader, model, loss_fn, optimizer)
    print("Done!")
    torch.save(model.state_dict(), "model_weights.pth")
# 训练的参数存放于model_weights.pth

start_train()