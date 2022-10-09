from train import *
import matplotlib.pyplot as plt

# 加载模型参数
model.load_state_dict(torch.load("model_weights.pth"))
model.eval()


def result_show():
    figure = plt.figure()
    plt.ion()  # 动态展示图片
    for i in range(len(test_data)):
        img, label = test_data[i]
        logs = model(img)
        pred_probab = nn.Softmax(dim=1)(logs)
        y_pred = pred_probab.argmax(1)
        figure.add_subplot(1, 1, 1)
        plt.title("True:" + str(label) + "   Predicted:" + str(y_pred.item()))
        plt.axis("off")
        plt.imshow(img.squeeze(), cmap="gray")
        plt.show()
        plt.pause(0.5)
        plt.clf()


def loop_test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
# 执行下序函数使用MNIST数据集进行测试
# loop_test(test_loader,model,loss_fn)
