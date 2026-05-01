"""
生成 A3 版 AprilTag + 形状编码彩色标记复合降落标志。

坐标约定：
- 纸面向右为 Tag +X。
- 纸面向下为 Tag +Y。
- Tag +Y 安装时朝无人车车头，Tag +X 安装时朝无人车左侧。

输出：
- nested_apriltag_output/nested_apriltag_color_board_a3.png
- nested_apriltag_output/nested_apriltag_color_board_a3.pdf
- nested_apriltag_output/color_marker_layout.json
"""

import json
import os
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw


# =========================================================
# 页面与打印参数
# =========================================================
A3_WIDTH_MM = 420.0
A3_HEIGHT_MM = 297.0
DPI = 1200

TAG_SIZE_MM = 200.0
MARKER_SIZE_MM = 32.0
MARKER_CENTER_DISTANCE_MM = 118.5

OUTPUT_DIR = "./nested_apriltag_output"
OUTPUT_BASENAME = "nested_apriltag_color_board_a3"


@dataclass(frozen=True)
class ColorMarker:
    name: str
    axis: str
    color_class: str
    shape: str
    color_hex: str
    x_mm: float
    y_mm: float
    description: str


COLOR_MARKERS = [
    ColorMarker(
        name="positive_x",
        axis="+X",
        color_class="purple",
        shape="circle",
        color_hex="#7A00FF",
        x_mm=MARKER_CENTER_DISTANCE_MM,
        y_mm=0.0,
        description="Tag +X，紫色圆形，安装后位于无人车左侧",
    ),
    ColorMarker(
        name="negative_x",
        axis="-X",
        color_class="purple",
        shape="square",
        color_hex="#7A00FF",
        x_mm=-MARKER_CENTER_DISTANCE_MM,
        y_mm=0.0,
        description="Tag -X，紫色正方形，安装后位于无人车右侧",
    ),
    ColorMarker(
        name="positive_y",
        axis="+Y",
        color_class="green",
        shape="circle",
        color_hex="#00B83F",
        x_mm=0.0,
        y_mm=MARKER_CENTER_DISTANCE_MM,
        description="Tag +Y，绿色圆形，安装后朝无人车车头",
    ),
    ColorMarker(
        name="negative_y",
        axis="-Y",
        color_class="green",
        shape="square",
        color_hex="#00B83F",
        x_mm=0.0,
        y_mm=-MARKER_CENTER_DISTANCE_MM,
        description="Tag -Y，绿色正方形，安装后朝无人车车尾",
    ),
    ColorMarker(
        name="redundant_top_right",
        axis="+X/-Y",
        color_class="yellow",
        shape="circle",
        color_hex="#FFFF00",
        x_mm=MARKER_CENTER_DISTANCE_MM,
        y_mm=-MARKER_CENTER_DISTANCE_MM,
        description="右上冗余点，纯黄色圆形，用于彩色 PnP 和误检校验",
    ),
    ColorMarker(
        name="redundant_bottom_left",
        axis="-X/+Y",
        color_class="yellow",
        shape="square",
        color_hex="#FFFF00",
        x_mm=-MARKER_CENTER_DISTANCE_MM,
        y_mm=MARKER_CENTER_DISTANCE_MM,
        description="左下冗余点，纯黄色正方形，用于彩色 PnP 和误检校验",
    ),
]


def mm_to_px(mm):
    """毫米转像素。"""
    return int(round(mm / 25.4 * DPI))


def hex_to_rgb(hex_color):
    """#RRGGBB 转 RGB 元组。"""
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))


def get_grid_bounds(size_px, grid_size=10):
    """返回均分网格边界，保证非整除像素尺寸也能完整覆盖画布。"""
    return np.rint(np.linspace(0, size_px, grid_size + 1)).astype(int)


def draw_tag_matrix(tag_matrix, output_size_px):
    """把 10x10 AprilTag 矩阵绘制成完整灰度图。"""
    bounds = get_grid_bounds(output_size_px)
    canvas = np.zeros((output_size_px, output_size_px), dtype=np.uint8)

    for i in range(10):
        for j in range(10):
            if tag_matrix[i, j] == 1:
                canvas[
                    bounds[i]:bounds[i + 1],
                    bounds[j]:bounds[j + 1],
                ] = 255

    return canvas


def create_nested_apriltag(tag65, tag66, tag67, output_size_px):
    """
    创建嵌套 AprilTag 灰度图。

    结构沿用原始脚本：
    - 外层 tag65 为 10x10。
    - tag65 中心 2x2 区域嵌入 tag66。
    - tag66 中心 2x2 区域嵌入 tag67。
    """
    canvas = draw_tag_matrix(tag65, output_size_px)
    tag65_bounds = get_grid_bounds(output_size_px)
    center_start_px = tag65_bounds[4]
    center_end_px = tag65_bounds[6]
    tag66_size = center_end_px - center_start_px
    tag66_resized = draw_tag_matrix(tag66, tag66_size)
    tag66_bounds = get_grid_bounds(tag66_size)
    tag66_center_start_px = tag66_bounds[4]
    tag66_center_end_px = tag66_bounds[6]
    tag67_size = tag66_center_end_px - tag66_center_start_px
    tag67_resized = draw_tag_matrix(tag67, tag67_size)

    tag66_resized[
        tag66_center_start_px:tag66_center_end_px,
        tag66_center_start_px:tag66_center_end_px,
    ] = tag67_resized

    canvas[center_start_px:center_end_px, center_start_px:center_end_px] = tag66_resized

    return Image.fromarray(canvas, mode="L")


def get_tag_arrays():
    """返回当前毕设使用的嵌套 AprilTag 三层矩阵。"""
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

    return tag65, tag66, tag67


def board_to_page_px(x_mm, y_mm, page_width_px, page_height_px):
    """
    Tag 中心坐标系转页面像素坐标。

    x_mm 向右为正，y_mm 向下为正。
    """
    return (
        int(round(page_width_px * 0.5 + mm_to_px(x_mm))),
        int(round(page_height_px * 0.5 + mm_to_px(y_mm))),
    )


def validate_layout():
    """检查彩色标记是否位于 A3 内，且不覆盖 AprilTag 区域。"""
    page_half_w = A3_WIDTH_MM * 0.5
    page_half_h = A3_HEIGHT_MM * 0.5
    tag_half = TAG_SIZE_MM * 0.5
    marker_radius = MARKER_SIZE_MM * 0.5

    errors = []
    for marker in COLOR_MARKERS:
        if abs(marker.x_mm) + marker_radius > page_half_w:
            errors.append(f"{marker.name} 超出 A3 横向范围")
        if abs(marker.y_mm) + marker_radius > page_half_h:
            errors.append(f"{marker.name} 超出 A3 纵向范围")

        overlaps_tag_x = abs(marker.x_mm) - marker_radius < tag_half
        overlaps_tag_y = abs(marker.y_mm) - marker_radius < tag_half
        if overlaps_tag_x and overlaps_tag_y:
            errors.append(f"{marker.name} 与 AprilTag 区域重叠")

    if errors:
        raise ValueError("布局检查失败：\n" + "\n".join(errors))


def create_color_board():
    """创建 A3 复合标志 RGB 图像。"""
    validate_layout()

    page_width_px = mm_to_px(A3_WIDTH_MM)
    page_height_px = mm_to_px(A3_HEIGHT_MM)
    tag_size_px = mm_to_px(TAG_SIZE_MM)
    marker_radius_px = mm_to_px(MARKER_SIZE_MM * 0.5)

    board = Image.new("RGB", (page_width_px, page_height_px), "white")

    tag65, tag66, tag67 = get_tag_arrays()
    tag_image = create_nested_apriltag(tag65, tag66, tag67, tag_size_px)
    tag_image = tag_image.convert("RGB")

    tag_left = (page_width_px - tag_size_px) // 2
    tag_top = (page_height_px - tag_size_px) // 2
    board.paste(tag_image, (tag_left, tag_top))

    draw = ImageDraw.Draw(board)
    for marker in COLOR_MARKERS:
        center_x, center_y = board_to_page_px(
            marker.x_mm,
            marker.y_mm,
            page_width_px,
            page_height_px,
        )
        bbox = [
            center_x - marker_radius_px,
            center_y - marker_radius_px,
            center_x + marker_radius_px,
            center_y + marker_radius_px,
        ]
        if marker.shape == "circle":
            draw.ellipse(bbox, fill=hex_to_rgb(marker.color_hex))
        elif marker.shape == "square":
            draw.rectangle(bbox, fill=hex_to_rgb(marker.color_hex))
        else:
            raise ValueError(f"未知标记形状: {marker.shape}")

    return board


def build_layout_metadata():
    """生成供后续彩色 PnP 检测使用的布局元数据。"""
    markers = []
    for marker in COLOR_MARKERS:
        markers.append({
            "name": marker.name,
            "axis": marker.axis,
            "color_class": marker.color_class,
            "shape": marker.shape,
            "color_hex": marker.color_hex,
            "size_m": MARKER_SIZE_MM / 1000.0,
            "object_point_m": [
                marker.x_mm / 1000.0,
                marker.y_mm / 1000.0,
                0.0,
            ],
            "description": marker.description,
        })

    return {
        "name": "nested_apriltag_color_board_a3",
        "dpi": DPI,
        "page": {
            "paper": "A3",
            "width_m": A3_WIDTH_MM / 1000.0,
            "height_m": A3_HEIGHT_MM / 1000.0,
        },
        "coordinate_convention": {
            "origin": "AprilTag center",
            "x_positive": "paper right / Tag +X / vehicle left after installation",
            "y_positive": "paper down / Tag +Y / vehicle front after installation",
            "z": "0 for all printed color marker centers",
            "unit": "meter",
        },
        "apriltag": {
            "family": "tagCustom48h12",
            "outer_tag_id": 65,
            "outer_tag_size_m": TAG_SIZE_MM / 1000.0,
            "nested_tag_ids": [65, 66, 67],
        },
        "color_markers": markers,
        "print_note": "A3 纸，打印时关闭适合页面/缩放到纸张，按 100% 原始尺寸打印。",
    }


def save_outputs(board):
    """保存 PNG、PDF 和 JSON 元数据。"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    png_path = os.path.join(OUTPUT_DIR, f"{OUTPUT_BASENAME}.png")
    pdf_path = os.path.join(OUTPUT_DIR, f"{OUTPUT_BASENAME}.pdf")
    json_path = os.path.join(OUTPUT_DIR, "color_marker_layout.json")

    board.save(png_path, "PNG", dpi=(DPI, DPI))
    board.save(pdf_path, "PDF", resolution=DPI)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(build_layout_metadata(), f, ensure_ascii=False, indent=2)
        f.write("\n")

    return png_path, pdf_path, json_path


def main():
    print("=" * 60)
    print("生成 A3 AprilTag + 形状编码彩色标记复合降落标志")
    print("=" * 60)
    print(f"A3 页面: {A3_WIDTH_MM:.1f} mm x {A3_HEIGHT_MM:.1f} mm")
    print(f"DPI: {DPI}")
    print(f"外层 AprilTag 边长: {TAG_SIZE_MM:.1f} mm")
    print(f"彩色标记尺寸: {MARKER_SIZE_MM:.1f} mm")
    print(f"彩色标记中心距 Tag 中心: {MARKER_CENTER_DISTANCE_MM:.1f} mm")
    print("=" * 60)

    board = create_color_board()
    png_path, pdf_path, json_path = save_outputs(board)

    print("生成完成：")
    print(f"  PNG : {png_path}")
    print(f"  PDF : {pdf_path}")
    print(f"  JSON: {json_path}")
    print("打印提示：使用 A3，关闭页面缩放，按 100% 原始尺寸打印。")


if __name__ == "__main__":
    main()
