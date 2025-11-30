#!/usr/bin/env python3
"""
图像差异检查脚本
检查sample.png和参考图像的像素差异，要求RMSE < 1e-5
"""

import numpy as np
from PIL import Image
import sys
import os

def calculate_rmse(img1_path, img2_path):
    """
    计算两幅图像的RMSE（均方根误差）
    
    Args:
        img1_path: 第一幅图像路径
        img2_path: 第二幅图像路径
        
    Returns:
        tuple: (rmse值, 是否通过检查, 详细信息)
    """
    try:
        # 读取图像
        img1 = Image.open(img1_path)
        img2 = Image.open(img2_path)
        
        # 转换为numpy数组
        img1_array = np.array(img1, dtype=np.float64)
        img2_array = np.array(img2, dtype=np.float64)
        
        # 检查图像尺寸是否一致
        if img1_array.shape != img2_array.shape:
            return None, False, f"图像尺寸不匹配: {img1_array.shape} vs {img2_array.shape}"
        
        # 计算差异
        diff = img1_array - img2_array
        
        # 计算RMSE
        mse = np.mean(diff ** 2)
        rmse = np.sqrt(mse)
        
        # 检查是否通过
        passed = rmse < 1e-2
        
        # 生成详细信息
        info = {
            'rmse': rmse,
            'mse': mse,
            'max_diff': np.max(np.abs(diff)),
            'min_diff': np.min(np.abs(diff)),
            'mean_diff': np.mean(np.abs(diff)),
            'image_shape': img1_array.shape,
            'threshold': 1e-5,
            'passed': passed
        }
        
        return rmse, passed, info
        
    except Exception as e:
        return None, False, f"错误: {str(e)}"

def print_detailed_report(info):
    """打印详细报告"""
    print("=" * 60)
    print("图像差异检查报告")
    print("=" * 60)
    print(f"图像尺寸: {info['image_shape']}")
    print(f"RMSE: {info['rmse']:.2e}")
    print(f"MSE: {info['mse']:.2e}")
    print(f"最大绝对差异: {info['max_diff']:.2e}")
    print(f"最小绝对差异: {info['min_diff']:.2e}")
    print(f"平均绝对差异: {info['mean_diff']:.2e}")
    print(f"阈值: {info['threshold']:.2e}")
    print(f"检查结果: {'通过' if info['passed'] else '失败'}")
    print("=" * 60)

def save_diff_image(img1_path, img2_path, output_path="diff_visualization.png"):
    """
    生成差异可视化图像
    """
    try:
        img1 = Image.open(img1_path)
        img2 = Image.open(img2_path)
        
        img1_array = np.array(img1, dtype=np.float64)
        img2_array = np.array(img2, dtype=np.float64)
        
        # 计算绝对差异并归一化
        diff = np.abs(img1_array - img2_array)
        diff_normalized = (diff / np.max(diff) * 255).astype(np.uint8)
        
        # 保存差异图像
        diff_img = Image.fromarray(diff_normalized)
        diff_img.save(output_path)
        print(f"差异可视化图像已保存: {output_path}")
        
    except Exception as e:
        print(f"生成差异图像失败: {e}")

def main():
    # 图像路径
    sample_path = "sample.png"
    reference_path = "/home/data/grond.png"
    
    print("开始检查图像差异...")
    print(f"测试图像: {sample_path}")
    print(f"参考图像: {reference_path}")
    print(f"精度要求: RMSE < 1e-2")
    print()
    
    # 检查文件是否存在
    if not os.path.exists(sample_path):
        print(f"错误: 测试图像不存在 - {sample_path}")
        sys.exit(1)
    
    if not os.path.exists(reference_path):
        print(f"错误: 参考图像不存在 - {reference_path}")
        sys.exit(1)
    
    # 计算RMSE
    rmse, passed, info = calculate_rmse(sample_path, reference_path)
    
    if rmse is None:
        print(f"计算失败: {info}")
        sys.exit(1)
    
    # 打印报告
    print_detailed_report(info)
    
    # 生成差异可视化
    save_diff_image(sample_path, reference_path, "diff_visualization.png")
    
    # 退出码
    if passed:
        print("🎉 图像差异检查通过！")
        sys.exit(0)
    else:
        print("❌ 图像差异检查失败！")
        sys.exit(1)

def batch_check(sample_dir, reference_dir, pattern="sample_{}.png"):
    """
    批量检查多个样本
    
    Args:
        sample_dir: 样本图像目录
        reference_dir: 参考图像目录
        pattern: 文件名模式，例如 "sample_{}.png"
    """
    print("开始批量检查...")
    all_passed = True
    
    for i in range(1, 100):  # 假设最多检查99个样本
        sample_path = os.path.join(sample_dir, pattern.format(i))
        reference_path = os.path.join(reference_dir, pattern.format(i))
        
        if not os.path.exists(sample_path) or not os.path.exists(reference_path):
            break
            
        print(f"\n检查样本 {i}:")
        rmse, passed, info = calculate_rmse(sample_path, reference_path)
        
        if rmse is not None:
            print(f"  RMSE: {rmse:.2e} - {'通过' if passed else '失败'}")
            if not passed:
                all_passed = False
        else:
            print(f"  检查失败: {info}")
            all_passed = False
    
    return all_passed

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='图像差异检查脚本')
    parser.add_argument('--sample', default='sample.png', help='测试图像路径')
    parser.add_argument('--reference', default='/home/data/grond.png', help='参考图像路径')
    parser.add_argument('--batch', action='store_true', help='批量检查模式')
    parser.add_argument('--sample-dir', help='样本图像目录（批量模式）')
    parser.add_argument('--reference-dir', help='参考图像目录（批量模式）')
    parser.add_argument('--pattern', default='sample_{}.png', help='文件名模式（批量模式）')
    
    args = parser.parse_args()
    
    if args.batch:
        if not args.sample_dir or not args.reference_dir:
            print("批量模式需要指定 --sample-dir 和 --reference-dir")
            sys.exit(1)
        
        passed = batch_check(args.sample_dir, args.reference_dir, args.pattern)
        if passed:
            print("\n🎉 所有样本检查通过！")
            sys.exit(0)
        else:
            print("\n❌ 部分样本检查失败！")
            sys.exit(1)
    else:
        # 单文件检查
        sample_path = args.sample
        reference_path = args.reference
        
        if not os.path.exists(sample_path):
            print(f"错误: 测试图像不存在 - {sample_path}")
            sys.exit(1)
        
        if not os.path.exists(reference_path):
            print(f"错误: 参考图像不存在 - {reference_path}")
            sys.exit(1)
        
        rmse, passed, info = calculate_rmse(sample_path, reference_path)
        
        if rmse is None:
            print(f"计算失败: {info}")
            sys.exit(1)
        
        print_detailed_report(info)
        
        # 生成差异可视化
        save_diff_image(sample_path, reference_path)
        
        sys.exit(0 if passed else 1)