import os
import chardet
import shutil

def convert_to_utf8(input_file, output_file, log_file):
    """将文件转换为UTF-8编码并保存到新位置，记录结果到日志文件"""
    try:
        # 检测文件编码
        with open(input_file, 'rb') as f:
            content = f.read()
            encoding = chardet.detect(content).get('encoding', 'unknown')

        # 如果文件已经是UTF-8，直接复制
        if encoding.lower() == 'utf-8':
            shutil.copy2(input_file, output_file)
            status = "已复制(UTF-8)"
        else:
            # 使用检测到的编码读取内容并以UTF-8写入新文件
            content_str = content.decode(encoding)
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(content_str)
            status = f"已转换: {encoding} -> UTF-8"

        # 记录处理结果（仅文件名）
        log_file.write(f"{os.path.basename(input_file)}\t{status}\n")
        print(f"{input_file} -> {output_file} [{status}]")
        
    except Exception as e:
        print(f"错误: 处理文件 {input_file} 时出错: {str(e)}")
        log_file.write(f"{os.path.basename(input_file)}\t错误: {str(e)}\n")

def batch_convert_json_to_utf8(input_folder, output_folder, log_path):
    """批量转换JSON文件编码并保存到输出目录"""
    # 创建输出目录（如果不存在）
    os.makedirs(output_folder, exist_ok=True)
    
    # 创建日志文件
    with open(log_path, 'w', encoding='utf-8') as log_file:
        # 遍历输入文件夹中的所有文件
        for root, _, files in os.walk(input_folder):
            for file in files:
                if file.endswith('.json'):
                    # 构建输入文件路径
                    input_file = os.path.join(root, file)
                    
                    # 构建输出文件路径（保持原目录结构）
                    rel_path = os.path.relpath(root, input_folder)
                    output_subfolder = os.path.join(output_folder, rel_path)
                    os.makedirs(output_subfolder, exist_ok=True)
                    output_file = os.path.join(output_subfolder, file)
                    
                    # 转换文件编码
                    convert_to_utf8(input_file, output_file, log_file)

# 指定输入文件夹、输出文件夹和日志文件路径
input_folder = '/home/mayi/wd/wxz/ChemicalFire_Dataset/fire_label/'
output_folder = '/home/mayi/wd/wxz/ChemicalFire_Dataset/fire_label_utf/'  # 转换后的文件保存位置
log_path = '/home/mayi/wd/wxz/ChemicalFire_Dataset/fire_label_utf/processed_files.txt'  # 日志文件

batch_convert_json_to_utf8(input_folder, output_folder, log_path)