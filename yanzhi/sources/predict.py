import sys
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, models
from PIL import Image


def create_model(num_classes=1):
    model = models.resnet50(
        weights=None
    )  # 不加载预训练权重，因为我们即将加载自己的训练权重
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, num_classes)
    return model


def load_trained_model(model_path, device):
    model = create_model(num_classes=1)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # 设置为评估模式
    return model.to(device)


def scores(num):
    score = 40 + (num - 1.02) / (4.75 - 1.02) * 60
    if score > 100:
        score = score - 10
    return score.item()


def predict_single_image(model, image_path, transform, device):
    # 加载并预处理单张图片
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)  # 添加批次维度
    image_tensor = image_tensor.to(device)

    # 模型预测
    with torch.no_grad():
        output = model(image_tensor).cpu()
        output_np = output.numpy()
        prediction_score = scores(output_np[0])
        prediction_score = float(f"{prediction_score:.2f}")
    return prediction_score


if __name__ == "__main__":
    # 参数设置
    model_path = "yanzhi/models/model_resnet50.pth"  # 模型文件路径，根据需要修改
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 数据预处理
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),  # 归一化
        ]
    )

    model = load_trained_model(model_path, device)

    single_image_path = sys.argv[1]
    print(f"处理图片的路径{single_image_path}")
    prediction = predict_single_image(model, single_image_path, transform, device)
    print(f"预测分数: {prediction:.2f}")
