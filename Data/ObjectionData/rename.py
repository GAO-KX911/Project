import os

# 设置你的图片文件夹路径
folder_path = "/home/njust/Fire/Data/ObjectionData/data"  # 请修改为你的文件夹路径

# 获取文件夹中所有的 .jpg 文件
jpg_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".jpg")]

# 对文件进行排序，确保按文件名顺序
jpg_files.sort()

# 遍历所有 .jpg 文件并重命名
for index, file_name in enumerate(jpg_files, start=5554):
    # 生成新的文件名
    new_name = f"{index}.jpg"
    
    # 构建完整的文件路径
    old_file_path = os.path.join(folder_path, file_name)
    new_file_path = os.path.join(folder_path, new_name)
    
    # 重命名文件
    os.rename(old_file_path, new_file_path)
    print(f"Renamed: {old_file_path} -> {new_file_path}")

print("Renaming completed!")
