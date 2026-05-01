import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os
from datetime import datetime


class NestedAprilTagGenerator:
    """
    生成嵌套AprilTag的类
    支持多层嵌套，可调节输出尺寸和分辨率
    """

    def __init__(self, output_size_px=2000, dpi=300):
        """
        初始化生成器

        参数：
        - output_size_px: 最终输出图像的像素大小（默认2000px，建议300dpi下为2000-4000）
        - dpi: 输出分辨率（默认300dpi，用于PDF和高质量输出）
        """
        self.output_size_px = output_size_px
        self.dpi = dpi
        self.output_size_inches = output_size_px / dpi

    def create_nested_apriltag(self, tag65, tag66, tag67):
        """
        创建嵌套AprilTag

        步骤：
        1. 创建最外层10x10的tag65
        2. 将tag65中心的2x2区域替换为缩放的tag66
        3. 将tag66中心的2x2区域替换为缩放的tag67

        参数：
        - tag65, tag66, tag67: 10x10的numpy数组（0为黑色，1为白色）

        返回：
        - PIL Image对象
        """
        # 第一步：创建基础的tag65层
        # 每个tag单元的像素大小
        base_unit_size = self.output_size_px // 10

        # 创建最外层画布
        canvas = np.zeros((self.output_size_px, self.output_size_px), dtype=np.uint8)

        # 填充tag65
        for i in range(10):
            for j in range(10):
                if tag65[i, j] == 1:  # 白色
                    canvas[i * base_unit_size:(i + 1) * base_unit_size,
                    j * base_unit_size:(j + 1) * base_unit_size] = 255

        # 第二步：处理tag65的中心2x2区域（第4-5行，第4-5列，索引从0开始）
        # 中心位置：4,5行和4,5列
        center_start_px = 4 * base_unit_size
        center_end_px = 6 * base_unit_size

        # 清空中心2x2区域
        canvas[center_start_px:center_end_px, center_start_px:center_end_px] = 255

        # 缩放tag66到中心2x2区域的大小
        tag66_size = center_end_px - center_start_px
        tag66_resized = np.zeros((tag66_size, tag66_size), dtype=np.uint8)

        tag66_unit_size = tag66_size // 10
        for i in range(10):
            for j in range(10):
                if tag66[i, j] == 1:  # 白色
                    tag66_resized[i * tag66_unit_size:(i + 1) * tag66_unit_size,
                    j * tag66_unit_size:(j + 1) * tag66_unit_size] = 255

        # 将缩放后的tag66放入中心区域
        canvas[center_start_px:center_end_px, center_start_px:center_end_px] = tag66_resized

        # 第三步：处理tag66的中心2x2区域（在缩放后的坐标中也是4-5行4-5列）
        # 相对于tag66缩放图像的中心
        tag66_center_start_px = 4 * tag66_unit_size
        tag66_center_end_px = 6 * tag66_unit_size

        # 在canvas中的绝对位置
        canvas_tag66_center_start = center_start_px + tag66_center_start_px
        canvas_tag66_center_end = center_start_px + tag66_center_end_px

        # 清空tag66的中心2x2区域
        canvas[canvas_tag66_center_start:canvas_tag66_center_end,
        canvas_tag66_center_start:canvas_tag66_center_end] = 255

        # 缩放tag67到tag66的中心2x2区域的大小
        tag67_size = canvas_tag66_center_end - canvas_tag66_center_start
        tag67_resized = np.zeros((tag67_size, tag67_size), dtype=np.uint8)

        tag67_unit_size = tag67_size // 10
        for i in range(10):
            for j in range(10):
                if tag67[i, j] == 1:  # 白色
                    tag67_resized[i * tag67_unit_size:(i + 1) * tag67_unit_size,
                    j * tag67_unit_size:(j + 1) * tag67_unit_size] = 255

        # 将缩放后的tag67放入tag66的中心区域
        canvas[canvas_tag66_center_start:canvas_tag66_center_end,
        canvas_tag66_center_start:canvas_tag66_center_end] = tag67_resized

        # 转换为PIL Image（0为黑色，255为白色）
        # 由于numpy数组默认是黑色背景，我们需要反转
        image = Image.fromarray(canvas, mode='L')

        return image

    def save_png(self, image, output_path='nested_apriltag.png'):
        """
        保存为高质量PNG文件

        参数：
        - image: PIL Image对象
        - output_path: 输出文件路径
        """
        # 使用高质量设置保存PNG
        image.save(output_path, 'PNG', quality=95)
        print(f"✓ PNG文件已保存: {output_path}")
        return output_path

    def save_pdf(self, image, output_path='nested_apriltag.pdf', paper_size='A4', margin_mm=10):
        """
        保存为高质量PDF文件（A4纸打印）

        参数：
        - image: PIL Image对象
        - output_path: 输出文件路径
        - paper_size: 纸张大小（'A4', 'Letter'等）
        - margin_mm: 边距（毫米）
        """
        # A4尺寸
        a4_width_inches = 8.27
        a4_height_inches = 11.69
        margin_inches = margin_mm / 25.4

        # 计算实际可用空间
        usable_width = a4_width_inches - 2 * margin_inches
        usable_height = a4_height_inches - 2 * margin_inches

        # 创建图表
        fig = plt.figure(figsize=(a4_width_inches, a4_height_inches), dpi=self.dpi)
        ax = fig.add_axes([margin_inches / a4_width_inches,
                           margin_inches / a4_height_inches,
                           usable_width / a4_width_inches,
                           usable_height / a4_height_inches])

        # 显示图像
        ax.imshow(image, cmap='gray', vmin=0, vmax=255)
        ax.axis('off')

        # 保存为PDF
        fig.savefig(output_path, format='pdf', dpi=self.dpi, bbox_inches='tight', pad_inches=margin_mm / 25.4)
        plt.close(fig)
        print(f"✓ PDF文件已保存: {output_path} (A4纸，边距{margin_mm}mm)")
        return output_path

    def save_svg(self, canvas_array, output_path='nested_apriltag.svg'):
        """
        保存为SVG矢量格式（可无限缩放）

        参数：
        - canvas_array: numpy数组
        - output_path: 输出文件路径
        """
        unit_size = self.output_size_px // 10

        svg_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg width="{self.output_size_px}" height="{self.output_size_px}" xmlns="http://www.w3.org/2000/svg">
  <rect width="{self.output_size_px}" height="{self.output_size_px}" fill="white"/>
'''

        # 遍历所有像素块
        for i in range(canvas_array.shape[0]):
            for j in range(canvas_array.shape[1]):
                if canvas_array[i, j] == 0:  # 黑色
                    x = j
                    y = i
                    svg_content += f'  <rect x="{x}" y="{y}" width="1" height="1" fill="black"/>\n'

        svg_content += '</svg>'

        with open(output_path, 'w') as f:
            f.write(svg_content)
        print(f"✓ SVG文件已保存: {output_path}")
        return output_path

    def generate_and_save(self, tag65, tag66, tag67, output_dir='./output', basename='nested_apriltag'):
        """
        一次性生成并保存所有格式的文件

        参数：
        - tag65, tag66, tag67: 10x10的numpy数组
        - output_dir: 输出目录
        - basename: 输出文件的基础名称
        """
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        print(f"\n{'=' * 60}")
        print(f"生成嵌套AprilTag")
        print(f"{'=' * 60}")
        print(f"输出尺寸: {self.output_size_px}x{self.output_size_px} 像素")
        print(f"分辨率: {self.dpi} DPI")
        print(f"输出目录: {output_dir}")
        print(f"{'=' * 60}\n")

        # 创建嵌套AprilTag
        print("正在生成嵌套结构...")
        image = self.create_nested_apriltag(tag65, tag66, tag67)

        # 保存各种格式
        files_saved = []

        png_path = os.path.join(output_dir, f'{basename}.png')
        self.save_png(image, png_path)
        files_saved.append(png_path)

        pdf_path = os.path.join(output_dir, f'{basename}.pdf')
        self.save_pdf(image, pdf_path)
        files_saved.append(pdf_path)

        # 额外保存SVG（可选）
        # svg_path = os.path.join(output_dir, f'{basename}.svg')
        # canvas = np.array(image)
        # self.save_svg(canvas, svg_path)
        # files_saved.append(svg_path)

        print(f"\n{'=' * 60}")
        print(f"✓ 所有文件生成完成！")
        print(f"{'=' * 60}")
        for file_path in files_saved:
            print(f"  • {file_path}")
        print(f"{'=' * 60}\n")

        return files_saved


# ============ 使用示例 ============
if __name__ == '__main__':
    # 定义AprilTag矩阵
    tag65 = np.array([
        [1, 1, 0, 1, 0, 1, 1, 0, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 1, 1, 1, 1, 0, 1],
        [0, 0, 1, 1, 0, 1, 1, 1, 0, 1],
        [1, 0, 1, 1, 0, 0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0, 0, 1, 1, 0, 0],
        [1, 0, 1, 1, 1, 1, 0, 1, 0, 1],
        [1, 0, 1, 1, 1, 1, 1, 1, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 0, 1, 1, 0, 1],
    ])

    tag66 = np.array([
        [1, 1, 0, 1, 0, 1, 1, 0, 1, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 1, 1, 1, 1, 0, 1],
        [0, 0, 1, 1, 0, 1, 0, 1, 0, 1],
        [0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 1, 1, 0, 0, 0, 1, 0, 1],
        [1, 0, 1, 0, 1, 1, 0, 1, 0, 1],
        [0, 0, 1, 1, 1, 1, 1, 1, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 1, 1, 0, 0, 0, 1, 0],
    ])

    tag67 = np.array([
        [1, 1, 0, 1, 0, 1, 1, 0, 1, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 1, 1, 1, 1, 0, 1],
        [0, 0, 1, 1, 0, 1, 0, 1, 0, 1],
        [1, 0, 1, 1, 0, 0, 1, 1, 0, 1],
        [1, 0, 1, 1, 0, 0, 0, 1, 0, 0],
        [0, 0, 1, 1, 1, 1, 0, 1, 0, 1],
        [1, 0, 1, 1, 1, 1, 1, 1, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 0, 1, 0, 1, 0, 1, 1],
    ])

    # 创建生成器（可调节参数）
    # output_size_px: 最终输出的像素大小（推荐2000-4000用于高质量打印）
    # dpi: 分辨率（推荐300DPI用于A4打印）
    generator = NestedAprilTagGenerator(output_size_px=20000, dpi=1200)

    # 生成并保存所有格式
    generator.generate_and_save(tag65, tag66, tag67,
                                output_dir='./nested_apriltag_output',
                                basename='nested_apriltag_tag65_66_67')