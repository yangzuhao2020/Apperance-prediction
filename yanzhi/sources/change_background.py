import sys
import cv2
from rembg import remove
from PIL import Image
import os

def change_bg(img_dir, output_dir, middle_dir='yanzhi/image_test/after_remove_background'):

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    # 确保中间目录也存在
    os.makedirs(middle_dir, exist_ok=True)  

    # 遍历输入目录中的所有文件
    for filename in os.listdir(img_dir):
        # 构建完整的文件路径
        image_path = os.path.join(img_dir, filename)
        
        # 读取图像
        image = cv2.imread(image_path)
        
        if image is None:
            print(f"无法读取图片: {image_path}")
            continue

        # 抠图处理
        output_image = remove(image)

        # 将 NumPy 数组转换为 PIL 图像
        output_image = Image.fromarray(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGBA))

        # 定义中间图片即抠图后输出文件路径
        middle_path = os.path.join(middle_dir, filename)

        # 保存抠图后的图像
        output_image.save(middle_path, "PNG")

        # 更换为黑色背景
        img = Image.open(middle_path).convert("RGBA")
        black_bg = Image.new("RGBA", img.size, (0, 0, 0, 255)) # (0, 0, 0, 255)用来定义粘贴区的透明度
        black_bg.paste(img, (0, 0), img) # (X,Y) 定义了要粘贴的位置

        # 定义最后更换背景的路径
        output_path = os.path.join(output_dir,filename)

        # 图片的保存
        black_bg.convert("RGB").save(output_path, "JPEG")

# change_bg("yanzhi/image_test/original_photos","image_test/after_remove_background")
# change_bg("yanzhi/Images","image_nobackground")

def change_background_single(img_dir, output_dir="yanzhi/image_test/after_remove_background"):
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 读取图像
    image = cv2.imread(img_dir)
    
    if image is None:
        print(f"无法读取图片: {img_dir}")
        return
    
    # 抠图处理
    output_image = remove(image)

    # 将 NumPy 数组转换为 PIL 图像
    output_image = Image.fromarray(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGBA))

    # 更换为黑色背景
    black_bg = Image.new("RGBA", output_image.size, (0, 0, 0, 255))
    black_bg.paste(output_image, (0, 0), output_image)

    # 定义最后更换背景的路径
    file_name = os.path.basename(img_dir) # 从路径中提取文件名
    out_name = file_name.split(".")[0] + "_out.jpg"
    output_path = os.path.join(output_dir, out_name)

    # 图片的保存
    black_bg.convert("RGB").save(output_path, "JPEG")

    print(f"图片背景消除已完成。消除后的照片保存到: {output_path}")


# 示例调用
# img_dir = "yanzhi/image_test/original_photos/H3M_Eason.jpg"
# if __name__ == "__main__":
img_dir = sys.argv[1]
change_background_single(img_dir)

