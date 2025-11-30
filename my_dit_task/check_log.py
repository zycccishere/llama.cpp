#!/usr/bin/env python3
"""
解析训练进度log时间的脚本
专门处理类似格式：
  0%|          | 0/250 [00:00<?, ?it/s]
 50%|█████     | 125/250 [00:48<00:48,  2.55it/s]
100%|██████████| 250/250 [02:04<00:00,  2.01it/s]
"""

import re
import sys
from datetime import timedelta
import argparse

def parse_progress_line(line):
    """
    解析训练进度log行中的时间信息
    
    Args:
        line: log行文本
        
    Returns:
        dict: 包含解析到的时间信息，None如果解析失败
    """
    # 匹配进度条格式: 50%|█████     | 125/250 [00:48<00:48,  2.55it/s]
    pattern = r'(\d+)%\|.*?\| (\d+)/(\d+) \[(\d+):(\d+)<(\d+):(\d+),\s*([\d.]+)(it)/s\]'
    
    match = re.search(pattern, line)
    if match:
        percent, current, total, elapsed_min, elapsed_sec, remaining_min, remaining_sec, speed, unit = match.groups()
        
        # 转换时间
        elapsed_time = timedelta(minutes=int(elapsed_min), seconds=int(elapsed_sec))
        remaining_time = timedelta(minutes=int(remaining_min), seconds=int(remaining_sec))
        total_estimated = elapsed_time + remaining_time
        
        return {
            'percent': int(percent),
            'current_step': int(current),
            'total_steps': int(total),
            'elapsed_time': elapsed_time,
            'remaining_time': remaining_time,
            'total_estimated_time': total_estimated,
            'speed': float(speed),
            'unit': unit,
            'elapsed_str': f"{elapsed_min}:{elapsed_sec}",
            'remaining_str': f"{remaining_min}:{remaining_sec}",
            'total_estimated_str': f"{int(total_estimated.total_seconds() // 60):02d}:{int(total_estimated.total_seconds() % 60):02d}",
            'progress': f"{current}/{total} ({percent}%)"
        }
    
    return None

def analyze_training_log(filename):
    """
    分析整个训练log文件
    """
    print(f"分析训练日志文件: {filename}")
    print("=" * 80)
    
    progress_data = []
    start_time = None
    end_time = None
    
    with open(filename, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            
            # 跳过警告信息和其他非进度行
            if 'UserWarning' in line or 'warnings.warn' in line or 'accelerate' in line:
                continue
                
            # 解析进度信息
            progress_info = parse_progress_line(line)
            if progress_info:
                progress_data.append(progress_info)
                # 记录开始和结束时间
                if progress_info['current_step'] == 0:
                    start_time = progress_info
                if progress_info['current_step'] == progress_info['total_steps']:
                    end_time = progress_info
                
  
    
    # 生成统计报告
    if progress_data:
        generate_statistics_report(progress_data, start_time, end_time)
    else:
        print("未找到训练进度信息")

def generate_statistics_report(progress_data, start_time, end_time):

    # 基本信息
    total_steps = progress_data[0]['total_steps']
    print(f"总迭代次数: {total_steps}")
    


    # 时间分析
    if len(progress_data) > 1:
        first_progress = progress_data[1]  # 跳过0%的状态
        last_progress = progress_data[-1]
        
        if first_progress and last_progress:
            actual_total_time = last_progress['elapsed_time']
            print(f"\n时间分析:")
            print(f"  实际总耗时: {str(actual_total_time).split('.')[0]}")
            
            if end_time and start_time:
                estimated_total = end_time['total_estimated_time']
                print(f"  预计总耗时: {end_time['total_estimated_str']}")
                print(f"  估计误差: {abs(actual_total_time - estimated_total).total_seconds():.1f}秒")
    
    # 性能分析
    print(f"\n性能分析:")
    total_time_seconds = progress_data[-1]['elapsed_time'].total_seconds()
    if total_time_seconds > 0:
        overall_speed = total_steps / total_time_seconds
        print(f"  整体平均速度: {overall_speed:.2f} it/s")
        print(f"  单次迭代平均时间: {total_time_seconds/total_steps:.3f} 秒/iter")
    
    # 识别性能瓶颈
    analyze_performance_bottlenecks(progress_data)

def analyze_performance_bottlenecks(progress_data):
    pass

def parse_single_line(log_line):
    """
    解析单行log
    """
    print(f"解析log行: {log_line}")
    progress_info = parse_progress_line(log_line)
    
    if progress_info:
        print("✓ 成功解析训练进度信息:")
        print(f"  进度: {progress_info['progress']}")
        print(f"  已用时间: {progress_info['elapsed_str']}")
        print(f"  剩余时间: {progress_info['remaining_str']}")
        print(f"  预计总时间: {progress_info['total_estimated_str']}")
        print(f"  当前速度: {progress_info['speed']:.2f} {progress_info['unit']}/s")
    else:
        print("✗ 未找到训练进度信息")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='解析训练进度log时间信息')
    parser.add_argument('input', nargs='?', help='log文件路径或log文本')
    parser.add_argument('-f', '--file', help='分析log文件')
    parser.add_argument('-l', '--line', help='解析单行log文本')
    parser.add_argument('-s', '--summary', action='store_true', help='只显示摘要信息')
    
    args = parser.parse_args()
    
    if args.file:
        analyze_training_log(args.file)
    elif args.line:
        parse_single_line(args.line)
    elif args.input:
        # 检查输入是文件还是文本
        try:
            with open(args.input, 'r') as f:
                analyze_training_log(args.input)
        except FileNotFoundError:
            parse_single_line(args.input)
    else:
        # 从标准输入读取
        print("训练进度log时间解析器")
        print("请输入log文本(输入空行结束):")
        
        lines = []
        while True:
            try:
                line = input().strip()
                if not line:
                    break
                lines.append(line)
            except EOFError:
                break
        
        for line in lines:
            parse_single_line(line)
            print()

if __name__ == "__main__":
    main()