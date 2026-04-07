from PIL import Image
import os


def resize_images(input_folder, output_folder, new_resolution):
    # 确保输出文件夹存在
    if os.path.exists(output_folder) is False:
        os.makedirs(output_folder)

    # 遍历输入文件夹中的所有图片
    for filename in os.listdir(input_folder):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        # 打开图片
        image = Image.open(input_path)
        # print(image.size)


        # 设置新的分辨率
        new_size = (new_resolution[0], new_resolution[1])
        # Resize the image
        image = image.resize(new_size)

        # Convert to 'RGB' mode
        image = image.convert('RGB')

        # Save the resized image as JPEG
        image.save(output_path, format='JPEG')

        # 保存处理后的图片到输出文件夹
        image.save(output_path)

# 用法示例
input_folder_path = "/home/mayi/wd/wxz/Classification/Dataset/Val"



Val_List = ['Val_d_20', 'Val_d_15', 'Val_d_12', 'Val_d_10', 'Val_d_8', 'Val_d_6', 'Val_d_5', 
            'Val_d_4',  'Val_d_3', 'Val_d_2', 'Val_d_1.5', 'Val', 'Val*1.5', 'Val*2', 
            'Val*2.5', 'Val*3']

Resolution_List = [[96, 54], [128, 72], [160,90], [192,108], [240,135], [320,180], [384,216], 
                   [480,270], [640,360], [960,540], [1280,720], [1920,1080], [2880,1620], [3840,2160], 
                   [4800,2700], [5760,3240]]


for i in range(14,16,1):
    output_folder_path = "/home/mayi/wd/wxz/Classification/Dataset/Resolution/" + Val_List[i]
    new_resolution = Resolution_List[i]
    resize_images(input_folder_path+'/fire', output_folder_path+'/fire', new_resolution)
    resize_images(input_folder_path+'/no_smokefire', output_folder_path+'/no_smokefire', new_resolution)
    resize_images(input_folder_path+'/smoke', output_folder_path+'/smoke', new_resolution)
    print(i)




