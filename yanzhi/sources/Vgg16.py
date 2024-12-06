import torch
from torchvision import transforms, models
import pandas as pd
from main import FaceBeautyDataset, create_datasets_and_dataloaders, train_model, test_model

def create_model(num_classes=1):
    # 更改模型为VGG16，并加载预训练权重
    model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    # 修改最后一层全连接层的输出节点数以适应你的任务
    model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, num_classes)
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
        transforms.Resize((224, 224)),  # 将图像大小调整为224x224，这是VGG16的要求
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
    model_save_path = 'yanzhi/models/model_vgg16.pth'  # 修改保存路径以反映模型变化
    
    # 模型训练
    train_model(model, train_loader, learning_rate, num_epochs, device, model_save_path, "vgg16")

    # 模型测试
    test_model(model, test_loader, device, "vgg16")
    
if __name__ == "__main__":
    main()