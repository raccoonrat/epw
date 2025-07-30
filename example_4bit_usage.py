#!/usr/bin/env python3
"""
EPW-A 4位量化使用示例
展示如何使用新的 EPW_LOAD_IN_4BIT 环境变量
"""

import os
import subprocess
import sys

def run_with_4bit_quantization():
    """使用4位量化运行EPW-A"""
    print("=== EPW-A 4位量化示例 ===")
    
    # 设置环境变量
    env = os.environ.copy()
    env['EPW_LOAD_IN_4BIT'] = 'true'
    
    print("设置环境变量: EPW_LOAD_IN_4BIT=true")
    print("这将强制使用4位量化，提供最快的加载速度")
    print("注意：4位量化会降低模型精度，但显著减少内存使用")
    
    try:
        # 运行EPW脚本
        result = subprocess.run(
            [sys.executable, "epw-enhance-1.py"],
            env=env,
            capture_output=True,
            text=True,
            timeout=600  # 10分钟超时
        )
        
        if result.returncode == 0:
            print("\n✅ 4位量化加载成功！")
            print("输出信息：")
            print(result.stdout)
        else:
            print("\n❌ 加载失败")
            print(f"错误信息: {result.stderr}")
            
    except subprocess.TimeoutExpired:
        print("⏰ 加载超时（超过10分钟）")
    except Exception as e:
        print(f"❌ 运行异常: {e}")

def show_quantization_options():
    """显示所有量化选项"""
    print("\n=== 量化选项说明 ===")
    print("1. 4位量化 (EPW_LOAD_IN_4BIT=true)")
    print("   - 最快加载速度")
    print("   - 最低内存使用")
    print("   - 精度损失较大")
    print("   - 适合快速测试和原型开发")
    
    print("\n2. 8位量化 (EPW_LOAD_IN_8BIT=true)")
    print("   - 较快加载速度")
    print("   - 中等内存使用")
    print("   - 精度损失较小")
    print("   - 适合生产环境")
    
    print("\n3. 快速加载模式 (EPW_FAST_LOADING=true)")
    print("   - 使用默认4位量化")
    print("   - 启用低内存使用模式")
    print("   - 适合标准使用场景")
    
    print("\n4. 标准模式 (无环境变量)")
    print("   - 使用默认4位量化")
    print("   - 标准加载流程")
    print("   - 适合大多数场景")

def main():
    """主函数"""
    print("EPW-A 4位量化使用示例")
    print("="*50)
    
    show_quantization_options()
    
    print("\n" + "="*50)
    print("开始4位量化测试...")
    print("="*50)
    
    run_with_4bit_quantization()
    
    print("\n" + "="*50)
    print("使用说明：")
    print("1. 设置环境变量: export EPW_LOAD_IN_4BIT=true")
    print("2. 运行脚本: python epw-enhance-1.py")
    print("3. 或者在PowerShell中: $env:EPW_LOAD_IN_4BIT='true'; python epw-enhance-1.py")
    print("="*50)

if __name__ == "__main__":
    main() 