import torch
from torchvision import transforms, models
import pandas as pd
from torchvision import models
from torchvision.models import ResNet50_Weights  # 引入weights枚举类
from main import create_datasets_and_dataloaders, train_model, test_model


def create_model(num_classes=1):
    model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)  # 使用最新的权重加载方式
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, num_classes)
    return model

def main():
    # 超参数的设置
    batch_size = 64
    learning_rate = 0.0005
    num_epochs = 100 
    labels_df = pd.read_csv('yanzhi/name_to_label.csv') 
    image_dir = 'yanzhi/image_nobackground/' 

    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 将图像大小调整为224x224
        transforms.RandomHorizontalFlip(p=0.5),  # 数据增强：随机水平翻转
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.2258, 0.1792, 0.1564], std=[0.2549, 0.2048, 0.1797])  # 如果需要，你可以调整归一化参数
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化
    ])

    # 创建数据加载器
    train_loader, test_loader = create_datasets_and_dataloaders(labels_df, image_dir, transform, batch_size)

    # 模型构建
    model = create_model(num_classes=1)  # 根据数据集的类别数来设置num_classes
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model_save_path = 'yanzhi/models/model_resnet50.pth'
    
    # 模型训练
    train_model(model, train_loader, learning_rate, num_epochs, device, model_save_path, "resnet50")

    # 模型测试
    test_model(model, test_loader, device, "resnet50")
    
if __name__ == "__main__":
    main()
