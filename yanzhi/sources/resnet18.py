import torch
from torchvision import transforms, models
import pandas as pd
from torchvision.models import ResNet18_Weights  # 引入ResNet18的weights枚举类
from main import FaceBeautyDataset, create_datasets_and_dataloaders, train_model, test_model


def create_model(num_classes=1):
    model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)  # 使用最新的ResNet18预训练权重
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)  # 修改最后一层的输出为num_classes
    return model

def main():
    # 超参数的设置
    batch_size = 64
    learning_rate = 0.0005
    num_epochs = 200 
    labels_df = pd.read_csv('yanzhi/name_to_label.csv') 
    image_dir = 'yanzhi/image_nobackground/' 

    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 将图像大小调整为224x224
        transforms.RandomHorizontalFlip(p=0.5),  # 数据增强：随机水平翻转
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化
    ])

    # 创建数据加载器
    train_loader, test_loader = create_datasets_and_dataloaders(labels_df, image_dir, transform, batch_size)

    # 模型构建
    model = create_model(num_classes=1)  # 根据数据集的类别数来设置num_classes
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model_save_path = 'yanzhi/models/model_resnet18.pth'  # 保存路径也改成ResNet-18

    # 模型训练
    train_model(model, train_loader, learning_rate, num_epochs, device, model_save_path, "resnet18")

    # 模型测试
    test_model(model, test_loader, device, "resnet18")

if __name__ == "__main__":
    main()
