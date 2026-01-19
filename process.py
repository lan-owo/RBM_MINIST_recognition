import torch
from PIL import Image

def crop_image(image, aspect_ratio_threshold: float = 1 / 0.6) -> Image:
    """裁剪最小边界框"""
    bbox = image.getbbox()
    cropped_image = image.crop(bbox) if bbox else image
    width, height = cropped_image.size
    
    # 判断长宽比的范围
    if 1 / aspect_ratio_threshold < width / height < aspect_ratio_threshold:
        cropped_image = cropped_image.resize((20, 20))
    else:
        if width > height:
            new_height = int(20 * height / width)
            cropped_image = cropped_image.resize((20, new_height))
        else:
            new_width = int(20 * width / height)
            cropped_image = cropped_image.resize((new_width, 20))

        # 创建 20x20 的黑色背景
        final_image = Image.new("RGB", (20, 20), color="black")
        # 将 cropped_image 居中放置到 final_image 中
        paste_x = (20 - cropped_image.width) // 2
        paste_y = (20 - cropped_image.height) // 2
        final_image.paste(cropped_image, (paste_x, paste_y))
        
        cropped_image = final_image

    return cropped_image

def calculate_centroid(image: Image) -> tuple:
    """根据亮度加权计算质心"""
    pixels = image.load()
    width, height = image.size
    total_x = total_y = total_weight = 0

    # 遍历所有像素，计算加权质心
    for y in range(height):
        for x in range(width):
            brightness = pixels[x, y]
            if not (isinstance(brightness, int) or isinstance(brightness, float)):
                r, g, b = brightness  # 获取RGB值
                brightness = 0.2989 * r + 0.5870 * g + 0.1140 * b  # 计算亮度
            total_x += x * brightness
            total_y += y * brightness
            total_weight += brightness
    
    # 计算加权质心坐标
    if total_weight > 0:
        centroid_x = total_x // total_weight
        centroid_y = total_y // total_weight
        return (int(centroid_x), int(centroid_y))
    else:
        return (width // 2, height // 2)  # 如果没有亮度值，返回图像中心

def adjust_image(image) -> Image:
    """将裁剪图像的质心放置到28x28黑色背景图像的中间"""
    # 裁剪图像并计算质心
    cropped_image = crop_image(image)
    centroid_x, centroid_y = calculate_centroid(cropped_image)

    # 创建28x28的黑色背景
    background = Image.new("RGB", (28, 28), color="black")
    
    # 对其质心和中心
    offset_x = 14 - centroid_x
    offset_y = 14 - centroid_y

    offset_x = offset_x if offset_x <= 8 else 8
    offset_y = offset_y if offset_y <= 8 else 8
    
    background.paste(cropped_image, (offset_x, offset_y))
    
    return background

def get_image_tensor(image) -> torch.Tensor:
    """得到绘制的图像对应的 tensor"""
    grayscale_image = adjust_image(image).convert("L")
    
    image_tensor = torch.tensor(list(grayscale_image.getdata()), dtype=torch.float32)
    
    image_tensor = image_tensor.view(28, 28)
    image_tensor = image_tensor.unsqueeze(0)

    return image_tensor

if __name__ == "__main__":
    image = Image.open('./images/received_processed.png')
    img = get_image_tensor(image)
    img_show = img.squeeze().numpy().astype('uint8')
    Image.fromarray(img_show).save('./images/adjusted_image.png')