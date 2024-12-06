import cv2
import sys
import os
from PIL import Image


def cut_photos(img_dir, output_dir):
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 遍历输入目录中的所有文件
    for filename in os.listdir(img_dir):
        # 构建完整的文件路径
        image_path = os.path.join(img_dir, filename)
        
        # 读取图像
        image = cv2.imread(image_path)
        
        if image is None:
            print(f"无法读取图片: {image_path}")
            continue
        
        # 转换为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 加载预训练的人脸检测模型
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # 检测人脸
        faces = face_cascade.detectMultiScale(gray, 
                                             scaleFactor=1.1, 
                                             minNeighbors=5, 
                                             minSize=(30, 30))

        if len(faces) == 0:
            print(f"未检测到人脸: {image_path}")
            continue

        for (x, y, w, h) in faces:
            # 计算扩展后的人脸区域
            x1 = max(0, int(x - 0.3 * w)) # 扩大计算区域
            y1 = max(0, int(y - 0.6 * h))
            x2 = min(image.shape[1], x + int(w * 1.25))
            y2 = min(image.shape[0], y + int(h * 1.3))

            # 将人脸区域复制到空白图像中
            face_image = image[y1:y2, x1:x2]

        # 构建输出文件路径
        output_path = os.path.join(output_dir, filename)

        # 确保输出路径有正确的扩展名
        if not output_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            output_path += '.jpg'

        # 保存结果图片
        if not cv2.imwrite(output_path, face_image):
            print(f"无法保存图片: {output_path}")
        else:
            print(f"图片已保存: {output_path}")

# 调用函数
# cut_photos('yanzhi/image_test/after_remove_background', 'yanzhi/image_processed')

def cut_photos_single(image_path, output_dir = "yanzhi/image_processed"):
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 读取图像
    image = cv2.imread(image_path)

    # 转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 加载预训练的人脸检测模型
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # 检测人脸
    faces = face_cascade.detectMultiScale(gray, 
                                            scaleFactor=1.1, 
                                            minNeighbors=5, 
                                            minSize=(30, 30))

    if len(faces) == 0:
        print(f"在{image_path}中未检测到人脸:, 请重新选择图片！！！")
        return

    for (x, y, w, h) in faces:
        # 计算扩展后的人脸区域
        x1 = max(0, int(x - 0.3 * w)) # 扩大计算区域
        y1 = max(0, int(y - 0.6 * h))
        x2 = min(image.shape[1], x + int(w * 1.25))
        y2 = min(image.shape[0], y + int(h * 1.3))

        # 将人脸区域复制到空白图像中
        face_image = image[y1:y2, x1:x2]

    # 构建输出文件路径
    file_name = os.path.basename(image_path) # 从路径中提取文件名
    out_name = file_name.split("_")[0] + "_out2.jpg"
    output_path = os.path.join(output_dir, out_name)

    # 确保输出路径有正确的扩展名
    if not output_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        output_path += '.jpg'

    # 保存结果图片
    if not cv2.imwrite(output_path, face_image):
        print(f"无法保存图片: {output_path}")
    else:
        print(f"图片已经完成裁减，图片已保存到: {output_path}")


# image_path = "image_test/after_remove_background/AF_1.jpg"

# if __name__ == "__main__":
image_path = sys.argv[1]
cut_photos_single(image_path)