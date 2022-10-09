import torchvision.transforms
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from PIL import Image
import os
import pandas as pd


# 定义自己的数据集，需要给出标签文件csv  存放图片的文件夹路径  对应的转换方式
class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = Image.open(img_path).convert("RGB")
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


# 1.完成将图片转为28*28的格式  2.将三通道彩色图片转换为单通道灰色图片  3.图片转换为tensor   return [1,28,28]
transform = torchvision.transforms.Compose(
    [torchvision.transforms.Resize((28, 28)),
     torchvision.transforms.Grayscale(num_output_channels=1),
     ToTensor()])
mydata = CustomImageDataset("must.csv", "D:\\python\\photo", transform)  # 实例化自己的dataset

# 使用MNIST的数据集进行训练和测试
training_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

train_loader = DataLoader(dataset=training_data, batch_size=64, shuffle=False)
test_loader = DataLoader(dataset=test_data, batch_size=64, shuffle=False)
