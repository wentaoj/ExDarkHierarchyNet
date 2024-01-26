# File: ExDark_imgenhancer.py
# ExDark HierarchyNet
# Author: Yuesong Huang*, Wentao Jiang*
# Date: Jan.2, 2024

import cv2
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Where do these data come from? Check above statistical analysis part
resize = 512
target_size = 256
mean_rgb = [0.15736957785604708, 0.12597658226052458, 0.10175356349007876]
std_rgb = [0.20638151823917858, 0.17567678119973223, 0.1616852394660488]
# Conventional ImageNEt: transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

image_transforms_show = transforms.Compose([
    transforms.Resize(resize),
    transforms.RandomResizedCrop(
        target_size,
        scale=(0.8, 1.0),
        ratio=(0.75, 1.33)
    ),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

def img_read(image_path):
    return cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

def img_enhance(tf, image):
    image_tensor = transforms.ToTensor()(image)
    image_tensor = image_tensor.unsqueeze(0) # (1, C, H, W)

    tf_image_tensor = tf(image_tensor)

    tf_image_tensor = tf_image_tensor.squeeze(0) # (C, H, W)
    tf_image = tf_image_tensor.permute(1, 2, 0).numpy()
    tf_image = (tf_image * 255).astype(np.uint8)

    return tf_image

def img_show(original_image, enhanced_image, tranform_name, classes):
    # Apply augmentation transform
    augmented_image = image_transforms_show(Image.fromarray(enhanced_image))

    # Convert tensor back to image for visualization
    augmented_image = augmented_image.permute(1, 2, 0).numpy()

    # Plotting the original, CLAHE applied, and augmented images
    plt.figure(figsize=(18, 6))
    plt.figtext(0.5, 0.01, 'Class: ' + ' '.join(str(classes)), ha='center', va='bottom', fontsize=12)
    plt.subplot(1, 3, 1)
    plt.imshow(original_image)
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(enhanced_image)
    plt.title(f'{tranform_name} Enhanced Image')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(augmented_image)
    plt.title('Augmented & Normalized Image')
    plt.axis('off')

    plt.show()

def img_save(enhanced_image, save_name):
    # Convert RGB to BGR
    bgr_image = cv2.cvtColor(enhanced_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_name, bgr_image)