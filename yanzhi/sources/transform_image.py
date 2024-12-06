from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

img = Image.open("./Images/AF1.jpg") 
a7 = transforms.RandomHorizontalFlip(p=0.5)
img_randomhorizontalflip = a7(img)
plt.subplot(1,2,1),plt.imshow(img),plt.title("before")
plt.subplot(1,2,2),plt.imshow(img_randomhorizontalflip),plt.title("after")
plt.show()
