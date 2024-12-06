import cv2
from yanzhi.sources.predict import predict_single_image, load_trained_model
from ultralytics import YOLOv10
import os
import argparse
import time
import torch
from torchvision import transforms

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 设备检测！！


def get_color(idx):
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)
    return color
# 作用很简单，用于获得不同的颜色。


class Detector(object):
    def __init__(self, weight_path, conf_threshold=0.2, iou_threshold=0.5, save_path="save"):
        self.device = device
        self.model = YOLOv10(weight_path)
        self.conf_threshold = conf_threshold  # 置信度
        self.iou_threshold = iou_threshold  # 交并比
        self.save_path = save_path  # 新增保存路径
        self.names = self.model.names  # 获取模型中定义的类别名称列表。

    def detect_image(self, img_bgr, resnet50_model, transform, device):  # 表示输入一张rgb的图片
        results = self.model(
            img_bgr,
            verbose=True,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
        )
        # 调用 YOLOv10 模型对输入图像 img_bgr 进行目标检测。
        bboxes_cls = results[0].boxes.cls
        # 提取每个检测框对应的类别标签。
        bboxes_conf = results[0].boxes.conf
        # 提取每个检测框的置信度分数。
        bboxes_xyxy = results[0].boxes.xyxy.cpu().numpy().astype("uint32")
        # 提取每个检测框的坐标，格式为 [x_min, y_min, x_max, y_max] 将将检测结果转换为 NumPy 数组。
        list_xyxy = []
        list_scores = []

        for idx in range(len(bboxes_cls)):
            box_cls = int(
                bboxes_cls[idx]
            )  # 获取当前检测框的类别索引，并将其转换为整数。
            bbox_xyxy = bboxes_xyxy[idx]
            list_xyxy.append(bbox_xyxy)
            bbox_label = self.names[box_cls]  # 框选到的方框中的内容标签
            box_conf = f"{bboxes_conf[idx]:.2f}"
            print("置信度：", box_conf)
            
            # 初始化 scores 变量
            scores = None

            if float(box_conf) > self.conf_threshold:
                xmax, ymax, xmin, ymin = (
                    bbox_xyxy[2],
                    bbox_xyxy[3],
                    bbox_xyxy[0],
                    bbox_xyxy[1],
                )
                # 确保裁剪区域不超出原图的边界
                h, w = img_bgr.shape[:2]  # 获取原始图像的尺寸
                xmin = max(0, xmin - 10)
                ymin = max(0, ymin - 10)
                xmax = min(w, xmax + 10)
                ymax = min(h, ymax + 10)
                
                crop_img = img_bgr[ymin - 10 : ymax + 10, xmin - 10 : xmax + 10]
                # 从原始图像 img_bgr 中裁剪出当前检测框的内容，并增加上下左右各 10 像素的边距。
                if not os.path.exists(self.save_path):
                    os.makedirs(self.save_path)

                if crop_img is None or crop_img.size == 0:
                    print("Error: Cropped image is empty.")
                else:
                    if not os.path.exists(self.save_path):
                        os.makedirs(self.save_path)
                    
                    crop_img_path = os.path.join(self.save_path, f"{bbox_label}_{idx + 1}.jpg")
                    print("裁剪后的人脸保存的路径：", crop_img_path)
                    
                    cv2.imwrite(crop_img_path, crop_img)  # 将裁剪后的图像保存到指定路径。
                    crop_img = cv2.imread(crop_img_path)  # 再次读取图片以确认保存成功
                    
                    if os.path.exists(crop_img_path) and crop_img.size > 0:
                        print(f"File saved successfully: {crop_img_path}")
                        scores = predict_single_image(resnet50_model, crop_img_path, transform, device)  # 得到颜值分数
                        list_scores.append(scores)
                    else:
                        print(f"Error saving file: {crop_img_path}")
                        scores = None
                        scores = predict_single_image(resnet50_model, crop_img_path, transform, device)  # 得到颜值分数

                img_bgr = cv2.rectangle(
                    img_bgr, (xmin, ymin), (xmax, ymax), get_color(69), 2
                )
                # 在原来图像上绘制一个矩形框，并为每个框设置不同的颜色。

                cv2.putText(
                    img_bgr,
                    f"{str(bbox_label)}/{str(scores)}",  # 图像上添加文本标签
                    (xmin, ymin - 10),  # 标签卫浴图像的左上角！！
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2,  # 字体类型和字体大小！
                    get_color(30),
                    2,  # 文本的颜色，由 get_color 函数生成。
                )  # 给检测框添加表情 和 颜值得分

        return img_bgr, list_xyxy, list_scores


# Example usage
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

    resnet50_model = load_trained_model(model_path, device)

    parser = argparse.ArgumentParser()  # 作用在于读取命令行参数！

    # 检测参数
    parser.add_argument(
        "--weights",
        default=r"models/train4/weights/best.pt",
        type=str,
        help="weights path",
    )  # 权重
    parser.add_argument(
        "--source",
        default=r"dataset/videos/video1.mp4",
        type=str,
        help="img or video(.mp4)path",
    )  # 图片或者视频的地址
    parser.add_argument(
        "--save", 
        default=r"save", 
        type=str, 
        help="save img or video path"
    )  # 处理后的图片保存路径
    parser.add_argument(
        "--vis", 
        default=True, 
        action="store_true", 
        help="visualize image"
    )  # 可视化结果
    parser.add_argument(
        "--conf_thre", 
        type=float, 
        default=0.5, 
        help="conf_thre"
    )  # 置信度
    parser.add_argument(
        "--iou_thre", 
        type=float, 
        default=0.5, 
        help="iou_thre"
    )# 交并比指的是交集和并集的比值，候选框与原标记框之间的比例，表示衡量定位的精准度
    opt = parser.parse_args()  # 从命令行中解析用户输入的参数

    model = Detector(
        weight_path=opt.weights,
        conf_threshold=opt.conf_thre,
        iou_threshold=opt.iou_thre,
        save_path=opt.save,  # 传递保存路径
    )

    images_format = [".png", ".jpg", ".jpeg", ".JPG", ".PNG", ".JPEG"]
    video_format = ["mov", "MOV", "mp4", "MP4"]

    
    if opt.source.split(".")[-1].lower() not in video_format:  # 判断是否属于视频格式。
        print("注意：图片处理开始")
        image_names = [
            name
            for name in os.listdir(opt.source)
            for item in images_format
            if os.path.splitext(name)[1] == item
        ]
        # 将符合格式的图片名称存放在 image_names列表中。

        for img_name in image_names:
            img_path = os.path.join(opt.source, img_name)
            img_ori = cv2.imread(img_path)
            img_vis, _, _ = model.detect_image(img_ori, resnet50_model, transform, device)  # 检测图片，将检测的图片框选返回。
            img_vis = cv2.resize(
                img_vis, None, fx=1.0, fy=1.0, interpolation=cv2.INTER_NEAREST
            )
            cv2.imwrite(os.path.join("save", img_name), img_vis)

    else:
        print("注意：处理视频开始。")

        capture = cv2.VideoCapture(opt.source)  # 创建视频对象
        fps = capture.get(cv2.CAP_PROP_FPS)  # 获取视频的帧率
        size = (
            int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )  # 视频中图片的宽度和高度
        
        fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")
        outVideo = cv2.VideoWriter(
            os.path.join(
                "save", os.path.basename(opt.source).split(".")[-2] + "_out.mp4"
            ),
            fourcc,  # 编码器
            fps,  # 帧率
            size,  # 视频尺寸
        )

        frame_count = 0  # 初始化帧计数器
        last_detected_frame = None  # 存储最后检测到的帧
        last_detection_result = None  # 新增：存储最后检测的结果（包括边界框等）

        while True:
            ret, frame = capture.read()  # 读取一帧
            if not ret:
                break

            if frame_count % 5 == 0:  # 每10帧执行一次检测
                start_frame_time = time.perf_counter()
                detection_frame, list_xyxy, list_scores = model.detect_image(frame, resnet50_model, transform, device) # 执行检测
                last_detection_result = {
                    "frame": detection_frame,
                    "boxes": list_xyxy,
                    "scores": list_scores,
                }  # 更新最后检测的结果
                end_frame_time = time.perf_counter()
                elapsed_time = end_frame_time - start_frame_time
                fps_estimation = 1 / elapsed_time if elapsed_time > 0 else fps_estimation

            else:
                # 如果不是检测帧，则使用上次检测的结果
                if last_detection_result is not None:
                    list_xyxy = last_detection_result["boxes"]
                    list_scores = last_detection_result["scores"]
                    for coordinates in list_xyxy:
                        xmax, ymax, xmin, ymin = (
                            coordinates[2],
                            coordinates[3],
                            coordinates[0],
                            coordinates[1],
                        )
                        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), get_color(69), 2)
                        if list_scores:
                            cv2.putText(
                                frame,
                                f'{str("face")}/{int(list_scores[0]):.2f}',  # 图像上添加文本标签 和 颜值数据
                                (xmin, ymin - 10),  # 标签位于图像的左上角
                                cv2.FONT_HERSHEY_SIMPLEX,
                                2,  # 字体类型和字体大小
                                get_color(30),
                                2,  # 文本的颜色
                            )
                    fps_estimation = 0.0

            cv2.putText(
                frame,
                f"FPS: {fps_estimation:.2f}",  # 处理速度
                (10, 35),  # 文字位置
                cv2.FONT_HERSHEY_SIMPLEX,  # 文字字体
                1.3,  # 缩放比例
                (0, 0, 255),  # 文字颜色
                2,  # 线条粗细
            )

            outVideo.write(frame)  # 写入输出视频
            frame_count += 1  # 更新帧计数器

