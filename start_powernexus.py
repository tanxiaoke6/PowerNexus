#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PowerNexus - 一键启动脚本

本脚本执行以下检查和操作：
1. 检查 data/ 目录是否存在，如不存在则运行数据生成器
2. 检查核心依赖是否已安装
3. 启动 Streamlit 仪表板

运行方式:
    python start_powernexus.py

作者: PowerNexus Team
日期: 2025-12-18
"""

import os
import sys
import subprocess
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent

# 颜色代码
class Colors:
    """终端颜色"""
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BLUE = "\033[94m"
    BOLD = "\033[1m"
    END = "\033[0m"


def print_banner():
    """打印启动横幅"""
    banner = f"""
{Colors.BLUE}{Colors.BOLD}
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║     ⚡ PowerNexus - 智能电网巡检与决策系统                   ║
║                                                              ║
║        Powered by Qwen2.5 | Streamlit Dashboard              ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
{Colors.END}
"""
    print(banner)


def check_python_version():
    """检查 Python 版本"""
    print(f"{Colors.BLUE}[1/5]{Colors.END} 检查 Python 版本...")
    
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"  {Colors.RED}✗ Python 版本过低: {sys.version}{Colors.END}")
        print(f"  {Colors.YELLOW}需要 Python 3.8+{Colors.END}")
        return False
    
    print(f"  {Colors.GREEN}✓ Python {version.major}.{version.minor}.{version.micro}{Colors.END}")
    return True


def check_dependencies():
    """检查核心依赖"""
    print(f"\n{Colors.BLUE}[2/5]{Colors.END} 检查核心依赖...")
    
    dependencies = {
        "streamlit": "Streamlit (Web UI)",
        "numpy": "NumPy (数值计算)",
        "PIL": "Pillow (图像处理)",
        "openai": "OpenAI (API 访问)",
    }
    
    optional_deps = {
        "torch": "PyTorch (深度学习)",
        "transformers": "Transformers (模型库)",
        "chromadb": "ChromaDB (向量库)",
    }
    
    missing = []
    
    # 必需依赖
    for module, name in dependencies.items():
        try:
            if module == "PIL":
                __import__("PIL")
            else:
                __import__(module)
            print(f"  {Colors.GREEN}✓ {name}{Colors.END}")
        except ImportError:
            print(f"  {Colors.RED}✗ {name} - 未安装{Colors.END}")
            missing.append(module)
    
    # 可选依赖
    print(f"\n  {Colors.BOLD}可选依赖:{Colors.END}")
    for module, name in optional_deps.items():
        try:
            __import__(module)
            print(f"  {Colors.GREEN}✓ {name}{Colors.END}")
        except ImportError:
            print(f"  {Colors.YELLOW}○ {name} - 未安装 (Mock 模式可用){Colors.END}")
    
    if missing:
        print(f"\n  {Colors.YELLOW}缺少必需依赖，请运行:{Colors.END}")
        print(f"  {Colors.BOLD}pip install {' '.join(missing)}{Colors.END}")
        return False
    
    return True


def check_config():
    """检查配置文件"""
    print(f"\n{Colors.BLUE}[3/5]{Colors.END} 检查配置文件...")
    
    config_path = PROJECT_ROOT / "config" / "config.yaml"
    if not config_path.exists():
        print(f"  {Colors.RED}✗ 配置文件不存在: {config_path}{Colors.END}")
        print(f"  {Colors.YELLOW}请参考 config/config_example.yaml 创建配置文件{Colors.END}")
        return False
    
    print(f"  {Colors.GREEN}✓ 配置文件存在{Colors.END}")
    return True


def check_data_directory():
    """检查数据目录"""
    print(f"\n{Colors.BLUE}[4/5]{Colors.END} 检查数据目录...")
    
    data_dir = PROJECT_ROOT / "data"
    images_dir = data_dir / "images"
    
    # 检查 data/images 是否存在且有文件
    if not images_dir.exists() or not list(images_dir.glob("*.jpg")):
        print(f"  {Colors.YELLOW}○ Mock 数据不存在，正在生成...{Colors.END}")
        
        generator_script = PROJECT_ROOT / "tools" / "generate_mock_data.py"
        
        if generator_script.exists():
            try:
                result = subprocess.run(
                    [sys.executable, str(generator_script)],
                    capture_output=True,
                    text=True,
                    cwd=str(PROJECT_ROOT),
                )
                
                if result.returncode == 0:
                    print(f"  {Colors.GREEN}✓ Mock 数据已生成{Colors.END}")
                else:
                    print(f"  {Colors.RED}✗ 数据生成失败{Colors.END}")
                    print(f"    {result.stderr[:200] if result.stderr else ''}")
                    
            except Exception as e:
                print(f"  {Colors.RED}✗ 运行生成器失败: {e}{Colors.END}")
        else:
            print(f"  {Colors.RED}✗ 找不到生成器脚本{Colors.END}")
    else:
        # 列出现有图像
        images = list(images_dir.glob("*.jpg"))
        print(f"  {Colors.GREEN}✓ 数据目录存在 ({len(images)} 张图像){Colors.END}")
        for img in images[:3]:
            print(f"    - {img.name}")


def check_cuda():
    """检查 CUDA 可用性"""
    print(f"\n{Colors.BLUE}[5/5]{Colors.END} 检查 GPU 状态...")
    
    try:
        import torch
        
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"  {Colors.GREEN}✓ CUDA 可用: {gpu_name} ({vram:.1f} GB){Colors.END}")
        else:
            print(f"  {Colors.YELLOW}○ CUDA 不可用，将使用 CPU 模式{Colors.END}")
            
    except ImportError:
        print(f"  {Colors.YELLOW}○ PyTorch 未安装，无法检测 GPU{Colors.END}")


def launch_streamlit():
    """启动 Streamlit"""
    print(f"\n{Colors.BLUE}{'='*60}{Colors.END}")
    print(f"{Colors.GREEN}{Colors.BOLD}启动 Streamlit 仪表板...{Colors.END}")
    print(f"{Colors.BLUE}{'='*60}{Colors.END}")
    print(f"\n{Colors.BOLD}访问地址: http://localhost:8501{Colors.END}")
    print(f"{Colors.YELLOW}按 Ctrl+C 停止服务器{Colors.END}\n")
    
    app_path = PROJECT_ROOT / "src" / "app.py"
    
    if not app_path.exists():
        print(f"{Colors.RED}错误: 找不到 {app_path}{Colors.END}")
        return False
    
    try:
        # 启动 Streamlit
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", str(app_path)],
            cwd=str(PROJECT_ROOT),
        )
        return True
        
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}服务器已停止{Colors.END}")
        return True
        
    except Exception as e:
        print(f"{Colors.RED}启动失败: {e}{Colors.END}")
        return False


def main():
    """主函数"""
    print_banner()
    
    # 执行检查
    if not check_python_version():
        sys.exit(1)
    
    deps_ok = check_dependencies()
    config_ok = check_config()
    check_data_directory()
    check_cuda()
    
    if not deps_ok or not config_ok:
        if not deps_ok:
            print(f"\n{Colors.RED}请先安装缺少的依赖：{Colors.END}")
            print(f"{Colors.BOLD}pip install -r requirements.txt{Colors.END}")
        if not config_ok:
            print(f"\n{Colors.RED}请先检查配置文件{Colors.END}")
        sys.exit(1)
    
    # 启动服务
    print(f"\n{Colors.GREEN}所有检查通过!{Colors.END}")
    
    # 询问是否启动
    try:
        response = input(f"\n{Colors.BOLD}是否启动 Streamlit? [Y/n]: {Colors.END}").strip().lower()
        if response in ("", "y", "yes"):
            launch_streamlit()
        else:
            print(f"\n{Colors.YELLOW}已取消启动{Colors.END}")
            print(f"手动启动命令: {Colors.BOLD}streamlit run src/app.py{Colors.END}")
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}已取消{Colors.END}")


if __name__ == "__main__":
    main()
