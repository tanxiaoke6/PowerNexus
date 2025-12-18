#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PowerNexus - Mock 数据生成器

本脚本生成用于测试和演示的模拟数据：
1. 模拟缺陷图像 (绝缘子裂纹)
2. 模拟技术标准 PDF 文档

运行方式:
    python tools/generate_mock_data.py

作者: PowerNexus Team
日期: 2025-12-18
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ============================================================================
# 依赖检测
# ============================================================================

PIL_AVAILABLE = False
REPORTLAB_AVAILABLE = False

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
    print("✓ Pillow 已导入")
except ImportError:
    print("✗ Pillow 未安装，无法生成图像")
    print("  安装: pip install pillow")

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.lib.units import cm
    REPORTLAB_AVAILABLE = True
    print("✓ ReportLab 已导入")
except ImportError:
    print("✗ ReportLab 未安装，无法生成 PDF")
    print("  安装: pip install reportlab")


# ============================================================================
# 输出目录
# ============================================================================

DATA_DIR = PROJECT_ROOT / "data"
IMAGES_DIR = DATA_DIR / "images"
MANUALS_DIR = DATA_DIR / "manuals"


def ensure_dirs():
    """确保输出目录存在"""
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    MANUALS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"✓ 目录已创建: {DATA_DIR}")


# ============================================================================
# 图像生成
# ============================================================================

def generate_insulator_image(output_path: Path = None) -> Path:
    """
    生成模拟绝缘子缺陷图像
    
    绘制一个灰色圆柱形绝缘子，带有红色裂纹。
    
    Args:
        output_path: 输出文件路径
        
    Returns:
        生成的图像路径
    """
    if not PIL_AVAILABLE:
        print("✗ 无法生成图像 (Pillow 未安装)")
        return None
    
    output_path = output_path or IMAGES_DIR / "insulator_defect.jpg"
    
    # 图像尺寸
    width, height = 640, 480
    
    # 创建图像
    img = Image.new('RGB', (width, height), color=(200, 220, 240))  # 天空蓝背景
    draw = ImageDraw.Draw(img)
    
    # 绘制背景 (天空和远景)
    for y in range(height):
        # 渐变天空
        r = int(150 + (y / height) * 50)
        g = int(180 + (y / height) * 40)
        b = int(220 + (y / height) * 35)
        draw.line([(0, y), (width, y)], fill=(r, g, b))
    
    # 绘制塔结构 (简化)
    tower_color = (70, 70, 80)
    draw.polygon([(280, 400), (360, 400), (340, 100), (300, 100)], fill=tower_color)
    draw.polygon([(250, 300), (270, 300), (280, 100), (260, 100)], fill=tower_color)
    draw.polygon([(370, 300), (390, 300), (380, 100), (360, 100)], fill=tower_color)
    
    # 绘制横担
    draw.rectangle([150, 120, 490, 135], fill=tower_color)
    
    # 绘制绝缘子串 (多个盘状)
    insulator_x = 200
    insulator_y_start = 140
    disc_height = 25
    disc_width = 50
    
    for i in range(6):
        y = insulator_y_start + i * disc_height
        
        # 绝缘子盘片 (灰色/棕色陶瓷)
        disc_color = (120, 100, 90)
        highlight_color = (160, 140, 130)
        shadow_color = (80, 60, 50)
        
        # 盘片主体
        draw.ellipse([insulator_x - disc_width//2, y, 
                      insulator_x + disc_width//2, y + disc_height - 5], 
                     fill=disc_color, outline=shadow_color)
        
        # 高光
        draw.ellipse([insulator_x - disc_width//3, y + 2, 
                      insulator_x + disc_width//4, y + disc_height//2], 
                     fill=highlight_color)
        
        # 连接金具
        if i < 5:
            draw.rectangle([insulator_x - 8, y + disc_height - 5, 
                           insulator_x + 8, y + disc_height + 5], 
                          fill=(100, 100, 110))
    
    # 绘制裂纹 (红色，明显)
    crack_y = insulator_y_start + 2 * disc_height + 10
    crack_points = [
        (insulator_x - 20, crack_y),
        (insulator_x - 15, crack_y + 5),
        (insulator_x - 8, crack_y + 2),
        (insulator_x, crack_y + 8),
        (insulator_x + 5, crack_y + 3),
        (insulator_x + 15, crack_y + 10),
        (insulator_x + 20, crack_y + 5),
    ]
    
    # 粗裂纹
    for i in range(len(crack_points) - 1):
        draw.line([crack_points[i], crack_points[i+1]], 
                  fill=(200, 0, 0), width=4)
    
    # 裂纹边缘 (黑色描边)
    for i in range(len(crack_points) - 1):
        draw.line([crack_points[i], crack_points[i+1]], 
                  fill=(100, 0, 0), width=2)
    
    # 添加分支裂纹
    draw.line([(insulator_x, crack_y + 8), (insulator_x - 5, crack_y + 15)], 
              fill=(180, 0, 0), width=2)
    draw.line([(insulator_x + 5, crack_y + 3), (insulator_x + 10, crack_y - 5)], 
              fill=(180, 0, 0), width=2)
    
    # 绘制第二串绝缘子 (正常)
    insulator_x2 = 440
    for i in range(6):
        y = insulator_y_start + i * disc_height
        disc_color = (120, 100, 90)
        draw.ellipse([insulator_x2 - disc_width//2, y, 
                      insulator_x2 + disc_width//2, y + disc_height - 5], 
                     fill=disc_color, outline=(80, 60, 50))
        if i < 5:
            draw.rectangle([insulator_x2 - 8, y + disc_height - 5, 
                           insulator_x2 + 8, y + disc_height + 5], 
                          fill=(100, 100, 110))
    
    # 绘制导线
    draw.line([(0, 300), (insulator_x, insulator_y_start + 6 * disc_height)], 
              fill=(50, 50, 50), width=3)
    draw.line([(insulator_x, insulator_y_start + 6 * disc_height), 
               (insulator_x2, insulator_y_start + 6 * disc_height)], 
              fill=(50, 50, 50), width=3)
    draw.line([(insulator_x2, insulator_y_start + 6 * disc_height), (640, 300)], 
              fill=(50, 50, 50), width=3)
    
    # 添加标签
    try:
        # 尝试使用系统字体
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()
    
    # 红色标注框
    draw.rectangle([insulator_x - 35, crack_y - 15, insulator_x + 35, crack_y + 25], 
                   outline=(255, 0, 0), width=2)
    draw.text((insulator_x - 30, crack_y + 30), "DEFECT", fill=(255, 0, 0), font=font)
    
    # 时间戳
    draw.text((10, height - 25), "2024-12-18 14:32:15 | 无人机巡检影像 | 220kV L1 线路 #35 塔", 
              fill=(255, 255, 255), font=font)
    
    # 保存图像
    img.save(output_path, "JPEG", quality=95)
    print(f"✓ 图像已生成: {output_path}")
    
    return output_path


def generate_normal_insulator_image(output_path: Path = None) -> Path:
    """生成正常绝缘子图像 (无缺陷)"""
    if not PIL_AVAILABLE:
        return None
    
    output_path = output_path or IMAGES_DIR / "insulator_normal.jpg"
    
    width, height = 640, 480
    img = Image.new('RGB', (width, height), color=(180, 200, 220))
    draw = ImageDraw.Draw(img)
    
    # 渐变天空
    for y in range(height):
        r = int(140 + (y / height) * 60)
        g = int(170 + (y / height) * 50)
        b = int(210 + (y / height) * 40)
        draw.line([(0, y), (width, y)], fill=(r, g, b))
    
    # 简化的塔和绝缘子
    draw.rectangle([300, 100, 340, 400], fill=(70, 70, 80))
    draw.rectangle([200, 115, 440, 130], fill=(70, 70, 80))
    
    # 绝缘子 (无缺陷)
    for x_pos in [220, 420]:
        for i in range(5):
            y = 135 + i * 22
            draw.ellipse([x_pos - 22, y, x_pos + 22, y + 18], 
                        fill=(110, 95, 85), outline=(70, 55, 45))
    
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except:
        font = ImageFont.load_default()
    
    draw.text((10, height - 22), "2024-12-18 14:35:22 | 无人机巡检 | 设备状态正常", 
              fill=(255, 255, 255), font=font)
    
    img.save(output_path, "JPEG", quality=90)
    print(f"✓ 图像已生成: {output_path}")
    
    return output_path


def generate_rust_image(output_path: Path = None) -> Path:
    """生成锈蚀设备图像"""
    if not PIL_AVAILABLE:
        return None
    
    output_path = output_path or IMAGES_DIR / "metal_rust.jpg"
    
    width, height = 640, 480
    img = Image.new('RGB', (width, height), color=(150, 150, 160))
    draw = ImageDraw.Draw(img)
    
    # 金属底板
    draw.rectangle([50, 100, 590, 380], fill=(180, 180, 190))
    
    # 锈蚀斑块
    import random
    random.seed(42)
    for _ in range(30):
        x = random.randint(60, 580)
        y = random.randint(110, 370)
        size = random.randint(10, 40)
        rust_r = random.randint(120, 180)
        rust_g = random.randint(60, 90)
        rust_b = random.randint(30, 50)
        draw.ellipse([x, y, x + size, y + size * 0.7], 
                    fill=(rust_r, rust_g, rust_b))
    
    # 标签
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except:
        font = ImageFont.load_default()
    
    draw.text((10, height - 22), "2024-12-18 15:10:33 | 金具锈蚀检测", 
              fill=(50, 50, 50), font=font)
    
    img.save(output_path, "JPEG", quality=90)
    print(f"✓ 图像已生成: {output_path}")
    
    return output_path


# ============================================================================
# PDF 生成
# ============================================================================

def generate_standards_pdf(output_path: Path = None) -> Path:
    """
    生成电力技术标准 PDF 文档
    
    包含 GB/T 50150 相关内容和高温处理规程。
    
    Args:
        output_path: 输出文件路径
        
    Returns:
        生成的 PDF 路径
    """
    if not REPORTLAB_AVAILABLE:
        print("✗ 无法生成 PDF (ReportLab 未安装)")
        return None
    
    output_path = output_path or MANUALS_DIR / "grid_standards.pdf"
    
    # 创建 PDF 画布
    c = canvas.Canvas(str(output_path), pagesize=A4)
    width, height = A4
    
    # 尝试注册中文字体 (如果有)
    try:
        # 尝试系统字体
        pdfmetrics.registerFont(TTFont('SimHei', 'simhei.ttf'))
        chinese_font = 'SimHei'
    except:
        chinese_font = 'Helvetica'
    
    # ===== 第一页：标题和目录 =====
    c.setFont(chinese_font, 24)
    c.drawCentredString(width/2, height - 80, "电力设备检修技术标准汇编")
    
    c.setFont(chinese_font, 14)
    c.drawCentredString(width/2, height - 120, "PowerNexus 知识库参考文档")
    
    c.setFont(chinese_font, 12)
    y = height - 180
    
    # 目录
    c.drawString(72, y, "目录")
    y -= 25
    c.setFont(chinese_font, 10)
    
    contents = [
        "一、GB/T 50150-2016 电气设备交接试验标准",
        "二、DL/T 596-2018 电力设备预防性试验规程",
        "三、绝缘子缺陷检测与处理",
        "四、高温环境设备运行规程",
        "五、电网拓扑调整操作规程",
    ]
    
    for item in contents:
        c.drawString(90, y, item)
        y -= 20
    
    c.showPage()
    
    # ===== 第二页：GB/T 50150 =====
    c.setFont(chinese_font, 16)
    c.drawString(72, height - 60, "一、GB/T 50150-2016 电气设备交接试验标准")
    
    c.setFont(chinese_font, 10)
    y = height - 100
    
    gb50150_content = """
1. 总则

1.1 本标准适用于新建、扩建和改建工程中电气设备的交接试验。
1.2 电气设备交接试验应在设备安装完毕后进行。
1.3 试验条件应符合相关规定。

2. 绝缘电阻测量

2.1 测量前应将被试设备各端子短接并接地放电。
2.2 测量应在环境温度 10~40°C 条件下进行。
2.3 绝缘电阻值不应低于制造厂规定值。
2.4 吸收比（R60s/R15s）应不小于 1.3（对于油纸绝缘）。

3. 介质损失角正切值测量

3.1 测量应在良好天气条件下进行。
3.2 空气相对湿度不应大于 80%。
3.3 测量结果应与同类型设备或前次测量值比较。

4. 变压器试验

4.1 变压器绕组的绝缘电阻值应符合表 4.1 规定。
4.2 变压器绕组的吸收比应不小于 1.3。
4.3 变压器油的击穿电压应不低于：
    - 110kV 及以下：35kV/2.5mm
    - 220kV：50kV/2.5mm
    - 330kV 及以上：60kV/2.5mm
"""
    
    for line in gb50150_content.strip().split('\n'):
        c.drawString(72, y, line)
        y -= 14
        if y < 80:
            c.showPage()
            y = height - 60
    
    c.showPage()
    
    # ===== 第三页：高温处理 =====
    c.setFont(chinese_font, 16)
    c.drawString(72, height - 60, "四、高温环境设备运行规程")
    
    c.setFont(chinese_font, 10)
    y = height - 100
    
    hightemp_content = """
1. 高温预警与响应

1.1 当环境温度超过 35°C 时，应启动高温预警。
1.2 高温期间应加强设备测温监控。
1.3 导线温度超过 70°C 时应采取降负荷措施。

2. 变压器高温运行

2.1 变压器油温不应超过 95°C（顶层油温）。
2.2 绕组温度不应超过 105°C。
2.3 高温期间应增加油色谱分析频次。

3. 线路高温处理

3.1 线路负载率超过 80% 时应考虑负荷转移。
3.2 导线弧垂增大时应检查对地距离。
3.3 高温期间应暂停线路带电作业。

4. 应急措施

4.1 设备温度异常升高时应立即降低负荷。
4.2 必要时可启动备用设备分担负荷。
4.3 严重过热设备应紧急停运检修。

5. 负荷转移操作

5.1 负荷转移前应确认目标线路容量余量。
5.2 操作应按调度指令执行。
5.3 转移后应监控各节点电压变化。
"""
    
    for line in hightemp_content.strip().split('\n'):
        c.drawString(72, y, line)
        y -= 14
        if y < 80:
            c.showPage()
            y = height - 60
    
    # 保存 PDF
    c.save()
    print(f"✓ PDF 已生成: {output_path}")
    
    return output_path


def generate_insulator_manual_pdf(output_path: Path = None) -> Path:
    """生成绝缘子维护手册 PDF"""
    if not REPORTLAB_AVAILABLE:
        return None
    
    output_path = output_path or MANUALS_DIR / "insulator_maintenance.pdf"
    
    c = canvas.Canvas(str(output_path), pagesize=A4)
    width, height = A4
    
    try:
        pdfmetrics.registerFont(TTFont('SimHei', 'simhei.ttf'))
        chinese_font = 'SimHei'
    except:
        chinese_font = 'Helvetica'
    
    c.setFont(chinese_font, 18)
    c.drawCentredString(width/2, height - 80, "绝缘子缺陷检测与维护手册")
    
    c.setFont(chinese_font, 10)
    y = height - 130
    
    content = """
1. 绝缘子常见缺陷类型

1.1 机械损伤
    - 裂纹：表面或内部裂纹，可能导致闪络
    - 破损：边缘缺损、掉瓷
    - 断裂：严重机械损伤导致的完全断裂

1.2 电气性能劣化
    - 污闪：污秽积累导致的表面放电
    - 老化：材料老化导致绝缘性能下降
    - 零值：绝缘电阻趋近于零

2. 缺陷检测方法

2.1 目视检测
    - 利用无人机高清摄像进行外观检查
    - 检测裂纹、破损、锈蚀、异物

2.2 红外测温
    - 检测绝缘子温度分布
    - 发热点可能预示缺陷

2.3 紫外检测
    - 检测电晕放电
    - 判断绝缘劣化程度

3. 缺陷处理规程

3.1 裂纹处理
    - 发现裂纹后立即上报
    - 评估裂纹深度和长度
    - 深度裂纹应安排更换
    - 更换前应做好安全措施

3.2 更换流程
    - 确认线路已停电
    - 拆除故障绝缘子
    - 安装新绝缘子
    - 进行耐压试验
    - 恢复运行
"""
    
    for line in content.strip().split('\n'):
        c.drawString(72, y, line)
        y -= 13
        if y < 80:
            c.showPage()
            y = height - 60
    
    c.save()
    print(f"✓ PDF 已生成: {output_path}")
    
    return output_path


# ============================================================================
# 主函数
# ============================================================================

def main():
    """生成所有 Mock 数据"""
    print("=" * 60)
    print("        PowerNexus Mock 数据生成器")
    print("=" * 60)
    
    # 确保目录存在
    ensure_dirs()
    
    print("\n" + "-" * 40)
    print("生成图像...")
    print("-" * 40)
    
    # 生成图像
    generate_insulator_image()
    generate_normal_insulator_image()
    generate_rust_image()
    
    print("\n" + "-" * 40)
    print("生成 PDF 文档...")
    print("-" * 40)
    
    # 生成 PDF
    generate_standards_pdf()
    generate_insulator_manual_pdf()
    
    print("\n" + "=" * 60)
    print("✓ Mock 数据生成完成!")
    print("=" * 60)
    
    # 列出生成的文件
    print("\n生成的文件:")
    
    if IMAGES_DIR.exists():
        for f in IMAGES_DIR.iterdir():
            print(f"  📷 {f.relative_to(PROJECT_ROOT)}")
    
    if MANUALS_DIR.exists():
        for f in MANUALS_DIR.iterdir():
            print(f"  📄 {f.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
