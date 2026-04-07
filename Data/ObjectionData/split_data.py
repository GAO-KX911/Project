#!/usr/bin/env python3
"""
图片数据集分割脚本
将指定文件夹中的.jpg文件按9:1比例分割到Train和Val子目录
使用方法: python split_images.py [图片目录] [训练集比例]
"""

import os
import sys
import shutil
import random
import argparse
from pathlib import Path
from typing import List, Tuple

def split_images(
    source_dir: str = ".",
    train_ratio: float = 0.9,
    random_seed: int = 42,
    shuffle: bool = True,
    copy_instead_of_move: bool = False
) -> Tuple[int, int]:
    """
    分割图片文件到Train和Val目录
    
    Args:
        source_dir: 源图片目录
        train_ratio: 训练集比例 (0-1之间)
        random_seed: 随机种子，保证可重复性
        shuffle: 是否随机打乱
        copy_instead_of_move: True=复制文件，False=移动文件
        
    Returns:
        (train_count, val_count): 训练集和验证集的数量
    """
    
    # 设置随机种子以保证结果可重复
    random.seed(random_seed)
    
    # 转换为Path对象
    source_path = Path(source_dir).resolve()
    train_path = source_path / "Train"
    val_path = source_path / "Val"
    
    # 检查源目录是否存在
    if not source_path.exists():
        print(f"错误: 源目录不存在 - {source_path}")
        sys.exit(1)
    
    # 检查是否为目录
    if not source_path.is_dir():
        print(f"错误: 指定的路径不是目录 - {source_path}")
        sys.exit(1)
    
    # 查找所有.jpg文件（包括子目录）
    image_extensions = ['.jpg', '.jpeg', '.JPG', '.JPEG']
    image_files = []
    
    print("正在搜索图片文件...")
    for ext in image_extensions:
        image_files.extend(source_path.rglob(f"*{ext}"))
    
    # 排除目标目录中的文件
    image_files = [
        f for f in image_files 
        if not str(f).startswith(str(train_path)) 
        and not str(f).startswith(str(val_path))
    ]
    
    total_images = len(image_files)
    
    if total_images == 0:
        print(f"错误: 在 {source_path} 中没有找到.jpg或.jpeg文件")
        print(f"支持的后缀: {', '.join(image_extensions)}")
        sys.exit(1)
    
    print(f"找到 {total_images} 个图片文件")
    
    # 确保比例在合理范围内
    if train_ratio <= 0 or train_ratio >= 1:
        print(f"警告: 训练集比例 {train_ratio} 不合理，自动调整为0.9")
        train_ratio = 0.9
    
    # 如果需要打乱
    if shuffle:
        random.shuffle(image_files)
        print("已随机打乱文件顺序")
    
    # 计算分割点
    split_index = int(total_images * train_ratio)
    train_files = image_files[:split_index]
    val_files = image_files[split_index:]
    
    print(f"\n分割结果:")
    print(f"  训练集 (Train): {len(train_files)} 个文件 ({len(train_files)/total_images*100:.1f}%)")
    print(f"  验证集 (Val):   {len(val_files)} 个文件 ({len(val_files)/total_images*100:.1f}%)")
    
    # 创建目标目录
    train_path.mkdir(exist_ok=True)
    val_path.mkdir(exist_ok=True)
    
    print(f"\n创建目录: {train_path.name}/ 和 {val_path.name}/")
    
    # 确认操作
    print("\n操作详情:")
    if copy_instead_of_move:
        print(f"  操作: 复制文件 (原文件保留)")
    else:
        print(f"  操作: 移动文件 (原文件删除)")
    
    response = input("\n是否继续？(y/n): ").strip().lower()
    if response != 'y':
        print("操作取消")
        sys.exit(0)
    
    # 处理训练集文件
    train_count = 0
    print(f"\n处理训练集文件...")
    for i, file_path in enumerate(train_files, 1):
        try:
            # 保持原始文件名，处理重名
            dest_file = train_path / file_path.name
            counter = 1
            
            # 如果目标文件已存在，添加数字后缀
            while dest_file.exists():
                name_stem = file_path.stem
                name_suffix = file_path.suffix
                dest_file = train_path / f"{name_stem}_{counter}{name_suffix}"
                counter += 1
            
            if copy_instead_of_move:
                shutil.copy2(str(file_path), str(dest_file))
            else:
                shutil.move(str(file_path), str(dest_file))
            
            train_count += 1
            
            # 显示进度
            if i % 10 == 0 or i == len(train_files):
                print(f"  训练集: {i}/{len(train_files)} ({i/len(train_files)*100:.1f}%)")
                
        except Exception as e:
            print(f"  错误处理文件 {file_path.name}: {e}")
    
    # 处理验证集文件
    val_count = 0
    print(f"\n处理验证集文件...")
    for i, file_path in enumerate(val_files, 1):
        try:
            dest_file = val_path / file_path.name
            counter = 1
            
            while dest_file.exists():
                name_stem = file_path.stem
                name_suffix = file_path.suffix
                dest_file = val_path / f"{name_stem}_{counter}{name_suffix}"
                counter += 1
            
            if copy_instead_of_move:
                shutil.copy2(str(file_path), str(dest_file))
            else:
                shutil.move(str(file_path), str(dest_file))
            
            val_count += 1
            
            if i % 10 == 0 or i == len(val_files):
                print(f"  验证集: {i}/{len(val_files)} ({i/len(val_files)*100:.1f}%)")
                
        except Exception as e:
            print(f"  错误处理文件 {file_path.name}: {e}")
    
    print(f"\n{'='*50}")
    print(f"分割完成！")
    print(f"训练集: {train_count} 个文件 -> {train_path}/")
    print(f"验证集: {val_count} 个文件 -> {val_path}/")
    print(f"总计:   {train_count + val_count} 个文件")
    print(f"{'='*50}")
    
    # 保存分割记录
    save_split_record(source_path, train_files, val_files)
    
    return train_count, val_count

def save_split_record(source_path: Path, train_files: List[Path], val_files: List[Path]):
    """保存分割记录文件"""
    
    record_file = source_path / "split_record.txt"
    
    with open(record_file, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("图片数据集分割记录\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"源目录: {source_path}\n")
        f.write(f"分割时间: {Path(__file__).stat().st_ctime}\n")
        f.write(f"训练集数量: {len(train_files)}\n")
        f.write(f"验证集数量: {len(val_files)}\n")
        f.write(f"总数量: {len(train_files) + len(val_files)}\n")
        f.write(f"训练集比例: {len(train_files)/(len(train_files)+len(val_files))*100:.1f}%\n\n")
        
        f.write("训练集文件列表:\n")
        f.write("-" * 40 + "\n")
        for file_path in train_files:
            f.write(f"{file_path.name}\n")
        
        f.write("\n验证集文件列表:\n")
        f.write("-" * 40 + "\n")
        for file_path in val_files:
            f.write(f"{file_path.name}\n")
    
    print(f"分割记录已保存到: {record_file}")

def main():
    """主函数：解析命令行参数并执行分割"""
    
    parser = argparse.ArgumentParser(
        description='将图片文件按比例分割到Train和Val目录',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  %(prog)s                    # 分割当前目录的图片
  %(prog)s ./data             # 分割指定目录的图片
  %(prog)s ./data --ratio 0.8 # 使用8:2的比例分割
  %(prog)s ./data --copy      # 复制文件而不是移动
  %(prog)s ./data --no-shuffle # 不打乱文件顺序
  %(prog)s ./data --seed 123  # 设置随机种子
        """
    )
    
    parser.add_argument(
        'source_dir',
        nargs='?',
        default='.',
        help='源图片目录（默认: 当前目录）'
    )
    
    parser.add_argument(
        '-r', '--ratio',
        type=float,
        default=0.9,
        help='训练集比例 (默认: 0.9)'
    )
    
    parser.add_argument(
        '-c', '--copy',
        action='store_true',
        help='复制文件而不是移动（默认: 移动）'
    )
    
    parser.add_argument(
        '--no-shuffle',
        action='store_true',
        help='不打乱文件顺序（默认: 随机打乱）'
    )
    
    parser.add_argument(
        '-s', '--seed',
        type=int,
        default=42,
        help='随机种子 (默认: 42)'
    )
    
    parser.add_argument(
        '-l', '--list-only',
        action='store_true',
        help='仅显示文件列表，不执行分割'
    )
    
    args = parser.parse_args()
    
    # 仅列出文件
    if args.list_only:
        source_path = Path(args.source_dir).resolve()
        image_extensions = ['.jpg', '.jpeg', '.JPG', '.JPEG']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(source_path.rglob(f"*{ext}"))
        
        print(f"在 {source_path} 中找到 {len(image_files)} 个图片文件:")
        for i, file_path in enumerate(image_files, 1):
            print(f"  {i:3d}. {file_path.relative_to(source_path)}")
        return
    
    # 执行分割
    try:
        train_count, val_count = split_images(
            source_dir=args.source_dir,
            train_ratio=args.ratio,
            random_seed=args.seed,
            shuffle=not args.no_shuffle,
            copy_instead_of_move=args.copy
        )
    except KeyboardInterrupt:
        print("\n操作被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"发生错误: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
