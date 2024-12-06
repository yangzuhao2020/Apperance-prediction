import cv2
import argparse
import time
from predict import Detector, get_color
from yanzhi.sources.predict import load_trained_model
import torch
from torchvision import transforms

# 替换为 IP 摄像头地址
ip_camera_url = "rtsp://admin:202077@192.168.13.167:8554/live"  # 或者 RTSP 流，如 "rtsp://<ip-address>/stream"

# 打开 IP 摄像头
capture = cv2.VideoCapture(ip_camera_url)

# 检测参数
parser = argparse.ArgumentParser()  # 作用在于读取命令行参数！

parser.add_argument(
    "--weights",
    default=r"models/train4/weights/best.pt",
    type=str,
    help="weights path",
)  # 权重
parser.add_argument(
    "--vis", default=True, 
    action="store_true", 
    help="visualize image"
)  # 可视化结果
parser.add_argument("--conf_thre", 
    type=float, 
    default=0.20, 
    help="conf_thre"
)  # 置信度
parser.add_argument(
    "--save", 
    default=r"save", 
    type=str, 
    help="save img or video path"
)  # 处理后的图片保存路径
parser.add_argument("--iou_thre", 
    type=float, 
    default=0.6, 
    help="iou_thre"
)# 交并比指的是交集和并集的比值，候选框与原标记框之间的比例，表示衡量定位的精准度
opt = parser.parse_args()  # 从命令行中解析用户输入的参数

model = Detector(
    weight_path=opt.weights,
    conf_threshold=opt.conf_thre,
    iou_threshold=opt.iou_thre,
    save_path=opt.save,  # 传递保存路径
)

if not capture.isOpened():
    print("无法连接到 IP 摄像头")
    exit()

print("成功连接到摄像头，开启实时监测！！！")

frame_count = 0  # 初始化帧计数器
last_detected_frame = None  # 存储最后检测到的帧
last_detection_result = None  # 新增：存储最后检测的结果（包括边界框等）

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

try:
    while True:
        ret, frame = capture.read()  # 逐帧读取
        if not ret:
            print("无法读取帧，可能是网络问题")
            break

        if frame_count % 4 == 0:  # 每4帧执行一次检测
            start_frame_time = time.perf_counter()
            detection_frame, list_xyxy, list_scores = model.detect_image(frame, resnet50_model, transform, device)  # 执行检测
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
                # print(last_detection_result)
                list_xyxy = last_detection_result["boxes"]
                print(list_xyxy)
                list_scores = last_detection_result["scores"]
                if list_xyxy:  # 如果某一帧恰巧没有检测到人脸，则会跳过这一帧。
                    for coordinates in list_xyxy:
                        xmax, ymax, xmin, ymin = (
                            coordinates[2],
                            coordinates[3],
                            coordinates[0],
                            coordinates[1],
                        )
                        cv2.rectangle(
                            frame, (xmin, ymin), (xmax, ymax), get_color(69), 2
                        )
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
                # else:
                #     print("这一帧没有检测到人！！！")
                #     continue

        cv2.putText(
            frame,
            f"FPS: {fps_estimation:.2f}",  # 处理速度
            (10, 35),  # 文字位置
            cv2.FONT_HERSHEY_SIMPLEX,  # 文字字体
            1.3,  # 缩放比例
            (0, 0, 255),  # 文字颜色
            2,  # 线条粗细
        )
        cv2.imshow("IP Camera Stream", frame)

        # outVideo.write(frame)  # 写入输出视频
        frame_count += 1  # 更新帧计数器

        # 显示当前帧

        # 按 'q' 键退出
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    # 释放资源
    capture.release()
    cv2.destroyAllWindows()
