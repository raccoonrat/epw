#!/usr/bin/env python3
"""
EPW-A 加载速度测试脚本
用于比较不同配置下的模型加载时间
"""

import os
import time
import subprocess
import sys

def run_test(config_name, env_vars):
    """运行指定配置的加载测试"""
    print(f"\n{'='*50}")
    print(f"测试配置: {config_name}")
    print(f"{'='*50}")
    
    # 构建环境变量
    env = os.environ.copy()
    for key, value in env_vars.items():
        env[key] = value
    
    # 记录开始时间
    start_time = time.time()
    
    try:
        # 运行EPW脚本
        result = subprocess.run(
            [sys.executable, "epw-enhance-1.py"],
            env=env,
            capture_output=True,
            text=True,
            timeout=600  # 10分钟超时
        )
        
        end_time = time.time()
        total_time = end_time - start_time
        
        print(f"配置: {config_name}")
        print(f"总运行时间: {total_time:.2f}秒")
        
        # 解析输出中的加载时间信息
        output_lines = result.stdout.split('\n')
        for line in output_lines:
            if "总加载时间:" in line:
                print(f"模型加载时间: {line.split(':')[1].strip()}")
                break
        
        if result.returncode == 0:
            print("✅ 测试成功")
        else:
            print("❌ 测试失败")
            print(f"错误信息: {result.stderr}")
            
    except subprocess.TimeoutExpired:
        print("⏰ 测试超时（超过10分钟）")
    except Exception as e:
        print(f"❌ 测试异常: {e}")

def main():
    """主函数：运行所有配置的测试"""
    print("EPW-A 加载速度测试")
    print("="*50)
    
    # 定义测试配置
    test_configs = [
        {
            "name": "标准配置",
            "env_vars": {}
        },
        {
            "name": "快速加载模式",
            "env_vars": {"EPW_FAST_LOADING": "true"}
        },
        {
            "name": "4位量化",
            "env_vars": {"EPW_LOAD_IN_4BIT": "true"}
        },
        {
            "name": "8位量化",
            "env_vars": {"EPW_LOAD_IN_8BIT": "true"}
        },
        {
            "name": "CPU模式",
            "env_vars": {"EPW_USE_CPU": "true"}
        },
        {
            "name": "4位量化+CPU",
            "env_vars": {
                "EPW_LOAD_IN_4BIT": "true",
                "EPW_USE_CPU": "true"
            }
        },
        {
            "name": "快速加载+CPU",
            "env_vars": {
                "EPW_FAST_LOADING": "true",
                "EPW_USE_CPU": "true"
            }
        }
    ]
    
    # 运行所有测试
    results = []
    for config in test_configs:
        run_test(config["name"], config["env_vars"])
        results.append(config)
    
    # 总结
    print(f"\n{'='*50}")
    print("测试总结")
    print(f"{'='*50}")
    print("建议根据您的硬件配置选择合适的加载方式：")
    print("- 如果追求最快速度且精度要求不高：使用4位量化")
    print("- 如果内存充足且使用GPU：使用快速加载模式")
    print("- 如果内存有限：使用8位量化")
    print("- 如果GPU显存不足：使用CPU模式")
    print("- 如果追求最快速度：使用4位量化+CPU模式")

if __name__ == "__main__":
    main() 