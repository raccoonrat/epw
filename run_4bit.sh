#!/bin/bash

# EPW-A 4位量化运行脚本 (Bash)
# 适用于Linux/macOS系统

echo "=== EPW-A 4位量化运行脚本 ==="
echo ""

# 设置环境变量
export EPW_LOAD_IN_4BIT=true
echo "已设置环境变量: EPW_LOAD_IN_4BIT=true"
echo "这将使用4位量化，提供最快的加载速度"
echo "注意：4位量化会降低模型精度，但显著减少内存使用"
echo ""

# 检查Python是否可用
if ! command -v python &> /dev/null; then
    echo "错误：未找到Python，请确保Python已安装并添加到PATH"
    exit 1
fi

# 显示Python版本
python_version=$(python --version 2>&1)
echo "Python版本: $python_version"

# 检查脚本文件是否存在
if [ ! -f "epw-enhance-1.py" ]; then
    echo "错误：未找到 epw-enhance-1.py 文件"
    echo "请确保在正确的目录中运行此脚本"
    exit 1
fi

echo "开始运行EPW-A（4位量化模式）..."
echo ""

# 运行EPW脚本
if python epw-enhance-1.py; then
    echo ""
    echo "✅ EPW-A运行完成！"
else
    echo ""
    echo "❌ EPW-A运行失败"
    exit 1
fi

echo ""
echo "=== 其他量化选项 ==="
echo "8位量化: export EPW_LOAD_IN_8BIT=true; python epw-enhance-1.py"
echo "快速加载: export EPW_FAST_LOADING=true; python epw-enhance-1.py"
echo "CPU模式: export EPW_USE_CPU=true; python epw-enhance-1.py"
echo "组合使用: export EPW_LOAD_IN_4BIT=true; export EPW_USE_CPU=true; python epw-enhance-1.py" 