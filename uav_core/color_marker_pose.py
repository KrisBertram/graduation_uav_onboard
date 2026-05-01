"""
形状编码彩色标记检测与 PnP 位姿解算。

作为 AprilTag 检测失败时的备用视觉观测：
- 彩色点坐标系与 AprilTag 坐标系保持一致。
- 原点为 AprilTag 中心，纸面向右为 +X，纸面向下为 +Y。
- 所有彩色标记位于同一平面，Z=0。
"""

from dataclasses import dataclass
from itertools import combinations, product

import cv2
import numpy as np
from loguru import logger


# =========================================================
# 默认 HSV 阈值（OpenCV HSV: H=0~179, S/V=0~255）
# 新图案只调 3 个颜色类；后续用机载摄像头实测颜色后，优先替换这里。
# =========================================================
COLOR_CLASS_HSV_RANGES = {
    "green": [((50, 55, 45), (86, 255, 255))],    # 深绿色 #00B83F
    "purple": [((118, 40, 65), (150, 227, 224))], # 紫色 / 蓝紫色 #7A00FF
    "yellow": [((22, 100, 80), (38, 255, 255))],  # 纯黄色 #FFFF00
}

# 用户实测三类地板 HSV 范围。默认从彩色候选 mask 中扣除，降低误检地板概率。
FLOOR_HSV_RANGES = {
    "floor_1": [((164, 69, 110), (179, 149, 195)), ((0, 69, 110), (0, 149, 195))], # 粉红色地板块
    "floor_2": [((6, 19, 138), (22, 90, 213))], # 米黄色地板块
    "floor_3": [((99, 55, 102), (115, 144, 195))], # 蓝色地板块
}
FLOOR_REJECTION_ENABLED = True


# =========================================================
# 当前 A3 复合标志的理想物理尺寸
# =========================================================
OUTER_TAG_SIZE_M = 0.200
COLOR_MARKER_CENTER_DISTANCE_M = 0.1185
COLOR_MARKER_SIZE_M = 0.032
COLOR_MARKER_DIAMETER_M = COLOR_MARKER_SIZE_M  # 兼容旧命名：圆形为直径，方形为边长
COLOR_MARKER_AXIS_LENGTH_M = 0.08


@dataclass(frozen=True)
class ColorMarkerSpec:
    name: str
    axis: str
    color_class: str
    shape: str
    object_point: np.ndarray


COLOR_MARKER_SPECS = {
    "positive_x": ColorMarkerSpec(
        name="positive_x",
        axis="+X",
        color_class="purple",
        shape="circle",
        object_point=np.array([ COLOR_MARKER_CENTER_DISTANCE_M, 0.0, 0.0], dtype=np.float32),
    ),
    "negative_x": ColorMarkerSpec(
        name="negative_x",
        axis="-X",
        color_class="purple",
        shape="square",
        object_point=np.array([-COLOR_MARKER_CENTER_DISTANCE_M, 0.0, 0.0], dtype=np.float32),
    ),
    "positive_y": ColorMarkerSpec(
        name="positive_y",
        axis="+Y",
        color_class="green",
        shape="circle",
        object_point=np.array([0.0,  COLOR_MARKER_CENTER_DISTANCE_M, 0.0], dtype=np.float32),
    ),
    "negative_y": ColorMarkerSpec(
        name="negative_y",
        axis="-Y",
        color_class="green",
        shape="square",
        object_point=np.array([0.0, -COLOR_MARKER_CENTER_DISTANCE_M, 0.0], dtype=np.float32),
    ),
    "redundant_top_right": ColorMarkerSpec(
        name="redundant_top_right",
        axis="+X/-Y",
        color_class="yellow",
        shape="circle",
        object_point=np.array([ COLOR_MARKER_CENTER_DISTANCE_M, -COLOR_MARKER_CENTER_DISTANCE_M, 0.0], dtype=np.float32),
    ),
    "redundant_bottom_left": ColorMarkerSpec(
        name="redundant_bottom_left",
        axis="-X/+Y",
        color_class="yellow",
        shape="square",
        object_point=np.array([-COLOR_MARKER_CENTER_DISTANCE_M,  COLOR_MARKER_CENTER_DISTANCE_M, 0.0], dtype=np.float32),
    ),
}

COLOR_MARKER_OBJECT_POINTS = {
    name: spec.object_point for name, spec in COLOR_MARKER_SPECS.items()
}

# 兼容旧调试脚本/外部读取；检测主逻辑使用 COLOR_CLASS_HSV_RANGES。
COLOR_MARKER_HSV_RANGES = {
    name: COLOR_CLASS_HSV_RANGES[spec.color_class]
    for name, spec in COLOR_MARKER_SPECS.items()
}

COLOR_CLASS_DRAW_BGR = {
    "green": (63, 184, 0),
    "purple": (255, 0, 122),
    "yellow": (0, 255, 255),
}
COLOR_MARKER_DRAW_BGR = {
    name: COLOR_CLASS_DRAW_BGR[spec.color_class]
    for name, spec in COLOR_MARKER_SPECS.items()
}


# =========================================================
# 检测与 PnP 参数
# =========================================================
MIN_MARKER_AREA_PX = 35
MAX_MARKER_AREA_FRACTION = 0.08
MORPH_KERNEL_SIZE = 3
PNP_MIN_POINTS = 4
PNP_RANSAC_REPROJECTION_ERROR_PX = 10.0
PNP_MAX_REPROJECTION_ERROR_PX = 12.0
PNP_RANSAC_ITERATIONS = 100
PNP_RANSAC_CONFIDENCE = 0.99
MAX_CANDIDATES_PER_MARKER = 3
MIN_SHAPE_SCORE = 0.60
MIN_LAYOUT_SCORE = 0.35


@dataclass
class ColorMarkerDetection:
    name: str
    center: np.ndarray
    area: float
    bbox: tuple
    color_class: str
    shape: str
    shape_score: float
    aspect_ratio: float
    extent: float
    circularity: float
    vertices: int
    candidate_id: str


@dataclass
class ColorMarkerPoseObservation:
    source: str
    rvec: np.ndarray
    tvec: np.ndarray
    image_points: np.ndarray
    object_points: np.ndarray
    marker_names: list
    detections: dict
    confidence: float
    reprojection_error: float
    inlier_count: int


@dataclass
class ContourCandidate:
    color_class: str
    center: np.ndarray
    area: float
    bbox: tuple
    aspect_ratio: float
    extent: float
    circularity: float
    vertices: int
    contour_index: int


def _score_close(value, target, tolerance):
    return float(np.clip(1.0 - abs(value - target) / tolerance, 0.0, 1.0))


def _make_raw_hsv_mask(hsv_frame, ranges):
    mask = np.zeros(hsv_frame.shape[:2], dtype=np.uint8)
    for lower, upper in ranges:
        lower_arr = np.array(lower, dtype=np.uint8)
        upper_arr = np.array(upper, dtype=np.uint8)
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv_frame, lower_arr, upper_arr))
    return mask


def _make_floor_mask(hsv_frame):
    mask = np.zeros(hsv_frame.shape[:2], dtype=np.uint8)
    for ranges in FLOOR_HSV_RANGES.values():
        mask = cv2.bitwise_or(mask, _make_raw_hsv_mask(hsv_frame, ranges))
    return mask


def _make_hsv_mask(hsv_frame, ranges, floor_mask=None):
    mask = _make_raw_hsv_mask(hsv_frame, ranges)
    if FLOOR_REJECTION_ENABLED and floor_mask is not None:
        mask = cv2.bitwise_and(mask, cv2.bitwise_not(floor_mask))

    if MORPH_KERNEL_SIZE > 1:
        kernel = np.ones((MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def _contour_to_candidate(contour, color_class, contour_index):
    area = float(cv2.contourArea(contour))
    perimeter = float(cv2.arcLength(contour, True))
    if perimeter < 1e-6:
        return None

    moments = cv2.moments(contour)
    if abs(moments["m00"]) < 1e-6:
        return None

    x, y, w, h = cv2.boundingRect(contour)
    bbox_area = max(1.0, float(w * h))
    aspect_ratio = float(w / max(1, h))
    extent = float(area / bbox_area)
    circularity = float(4.0 * np.pi * area / (perimeter * perimeter))
    approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
    center = np.array([moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]], dtype=np.float32)

    return ContourCandidate(
        color_class=color_class,
        center=center,
        area=area,
        bbox=(int(x), int(y), int(w), int(h)),
        aspect_ratio=aspect_ratio,
        extent=extent,
        circularity=circularity,
        vertices=int(len(approx)),
        contour_index=contour_index,
    )


def _shape_score(candidate, shape):
    aspect_score = _score_close(candidate.aspect_ratio, 1.0, 0.55)
    if shape == "circle":
        circularity_score = float(np.clip((candidate.circularity - 0.50) / 0.40, 0.0, 1.0))
        vertex_score = float(np.clip((candidate.vertices - 4) / 4.0, 0.0, 1.0))
        extent_score = _score_close(candidate.extent, 0.76, 0.35)
        return 0.35 * circularity_score + 0.25 * vertex_score + 0.25 * aspect_score + 0.15 * extent_score

    if shape == "square":
        vertex_score = _score_close(candidate.vertices, 4.0, 3.0)
        extent_score = float(np.clip((candidate.extent - 0.55) / 0.35, 0.0, 1.0))
        circularity_score = _score_close(candidate.circularity, 0.78, 0.35)
        return 0.40 * vertex_score + 0.25 * extent_score + 0.20 * aspect_score + 0.15 * circularity_score

    raise ValueError(f"未知标记形状: {shape}")


def _detect_color_class_candidates(hsv_frame, frame_area):
    floor_mask = _make_floor_mask(hsv_frame) if FLOOR_REJECTION_ENABLED else None
    candidates = []
    max_area = frame_area * MAX_MARKER_AREA_FRACTION

    for color_class, ranges in COLOR_CLASS_HSV_RANGES.items():
        mask = _make_hsv_mask(hsv_frame, ranges, floor_mask=floor_mask)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour_index, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area < MIN_MARKER_AREA_PX or area > max_area:
                continue

            candidate = _contour_to_candidate(contour, color_class, contour_index)
            if candidate is not None:
                candidates.append(candidate)

    return candidates


def _build_marker_candidates(candidates):
    marker_candidates = {}
    for name, spec in COLOR_MARKER_SPECS.items():
        matches = []
        for candidate in candidates:
            if candidate.color_class != spec.color_class:
                continue
            score = _shape_score(candidate, spec.shape)
            if score < MIN_SHAPE_SCORE:
                continue
            matches.append(ColorMarkerDetection(
                name=name,
                center=candidate.center,
                area=candidate.area,
                bbox=candidate.bbox,
                color_class=candidate.color_class,
                shape=spec.shape,
                shape_score=score,
                aspect_ratio=candidate.aspect_ratio,
                extent=candidate.extent,
                circularity=candidate.circularity,
                vertices=candidate.vertices,
                candidate_id=f"{candidate.color_class}:{candidate.contour_index}",
            ))

        matches.sort(key=lambda item: (item.shape_score, item.area), reverse=True)
        if matches:
            marker_candidates[name] = matches[:MAX_CANDIDATES_PER_MARKER]
    return marker_candidates


def detect_color_markers(frame_bgr):
    """
    检测每个形状编码彩色标记的候选质心。

    返回 dict: marker_name -> [ColorMarkerDetection, ...]。
    """
    hsv_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    frame_area = float(frame_bgr.shape[0] * frame_bgr.shape[1])
    candidates = _detect_color_class_candidates(hsv_frame, frame_area)
    return _build_marker_candidates(candidates)


def _make_pnp_points(detections):
    marker_names = []
    object_points = []
    image_points = []

    for name, object_point in COLOR_MARKER_OBJECT_POINTS.items():
        detection = detections.get(name)
        if detection is None:
            continue
        marker_names.append(name)
        object_points.append(object_point)
        image_points.append(detection.center)

    if len(marker_names) < PNP_MIN_POINTS:
        return marker_names, None, None

    return (
        marker_names,
        np.asarray(object_points, dtype=np.float32),
        np.asarray(image_points, dtype=np.float32),
    )


def _mean_reprojection_error(object_points, image_points, rvec, tvec, camera_matrix, dist_coeffs):
    projected, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, dist_coeffs)
    projected = projected.reshape(-1, 2)
    return float(np.mean(np.linalg.norm(projected - image_points, axis=1)))


def _solve_color_marker_pnp(object_points, image_points, camera_matrix, dist_coeffs):
    if len(object_points) >= 5:
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            object_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            iterationsCount=PNP_RANSAC_ITERATIONS,
            reprojectionError=PNP_RANSAC_REPROJECTION_ERROR_PX,
            confidence=PNP_RANSAC_CONFIDENCE,
            flags=cv2.SOLVEPNP_EPNP,
        )
        if ok and inliers is not None and len(inliers) >= PNP_MIN_POINTS:
            inlier_idx = inliers.reshape(-1)
            ok_refine, rvec, tvec = cv2.solvePnP(
                object_points[inlier_idx],
                image_points[inlier_idx],
                camera_matrix,
                dist_coeffs,
                rvec,
                tvec,
                useExtrinsicGuess=True,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
            if ok_refine:
                return True, rvec, tvec, inlier_idx

    for flag in (cv2.SOLVEPNP_IPPE, cv2.SOLVEPNP_EPNP, cv2.SOLVEPNP_ITERATIVE):
        try:
            ok, rvec, tvec = cv2.solvePnP(
                object_points,
                image_points,
                camera_matrix,
                dist_coeffs,
                flags=flag,
            )
        except cv2.error:
            ok = False

        if ok:
            return True, rvec, tvec, np.arange(len(object_points))

    return False, None, None, None


def _layout_score(detections):
    score = 0.35
    checks = 0
    opposite_pair_count = 0

    def point(name):
        detection = detections.get(name)
        return None if detection is None else detection.center

    px = point("positive_x")
    nx = point("negative_x")
    py = point("positive_y")
    ny = point("negative_y")

    if px is not None and nx is not None:
        checks += 1
        opposite_pair_count += 1
        score += 0.20
    if py is not None and ny is not None:
        checks += 1
        opposite_pair_count += 1
        score += 0.20

    if opposite_pair_count == 0 and len(detections) < 5:
        return 0.0

    if px is not None and nx is not None and py is not None and ny is not None:
        x_vec = px - nx
        y_vec = py - ny
        x_len = float(np.linalg.norm(x_vec))
        y_len = float(np.linalg.norm(y_vec))
        if x_len > 1e-6 and y_len > 1e-6:
            checks += 1
            cos_abs = abs(float(np.dot(x_vec, y_vec) / (x_len * y_len)))
            perpendicular_score = float(np.clip(1.0 - cos_abs / 0.65, 0.0, 1.0))
            length_ratio = min(x_len, y_len) / max(x_len, y_len)
            ratio_score = float(np.clip((length_ratio - 0.25) / 0.5, 0.0, 1.0))
            midpoint_gap = float(np.linalg.norm((px + nx) * 0.5 - (py + ny) * 0.5))
            midpoint_score = float(np.clip(1.0 - midpoint_gap / max(x_len, y_len, 1.0), 0.0, 1.0))
            score += 0.25 * perpendicular_score + 0.20 * ratio_score + 0.15 * midpoint_score

    if point("redundant_top_right") is not None:
        checks += 1
        score += 0.10
    if point("redundant_bottom_left") is not None:
        checks += 1
        score += 0.10

    if checks == 0:
        return 0.0
    return float(np.clip(score, 0.0, 1.0))


def _iter_detection_combinations(marker_candidates):
    available_names = [name for name in COLOR_MARKER_SPECS if name in marker_candidates]
    if len(available_names) < PNP_MIN_POINTS:
        return

    for count in range(len(available_names), PNP_MIN_POINTS - 1, -1):
        for names in combinations(available_names, count):
            for candidates in product(*(marker_candidates[name] for name in names)):
                candidate_ids = [candidate.candidate_id for candidate in candidates]
                if len(set(candidate_ids)) != len(candidate_ids):
                    continue
                yield dict(zip(names, candidates))


def _choose_best_pose(marker_candidates, camera_matrix, dist_coeffs):
    best = None

    for detections in _iter_detection_combinations(marker_candidates):
        layout_score = _layout_score(detections)
        if layout_score < MIN_LAYOUT_SCORE:
            continue

        marker_names, object_points, image_points = _make_pnp_points(detections)
        ok, rvec, tvec, inlier_idx = _solve_color_marker_pnp(
            object_points,
            image_points,
            camera_matrix,
            dist_coeffs,
        )
        if not ok or float(tvec.reshape(-1)[2]) <= 0.0:
            continue

        inlier_object_points = object_points[inlier_idx]
        inlier_image_points = image_points[inlier_idx]
        inlier_marker_names = [marker_names[i] for i in inlier_idx]
        if len(inlier_marker_names) < PNP_MIN_POINTS:
            continue

        reprojection_error = _mean_reprojection_error(
            inlier_object_points,
            inlier_image_points,
            rvec,
            tvec,
            camera_matrix,
            dist_coeffs,
        )
        if reprojection_error > PNP_MAX_REPROJECTION_ERROR_PX:
            continue

        selected_detections = {name: detections[name] for name in inlier_marker_names}
        avg_shape_score = float(np.mean([item.shape_score for item in selected_detections.values()]))
        point_ratio = len(inlier_marker_names) / len(COLOR_MARKER_SPECS)
        error_score = max(0.0, 1.0 - reprojection_error / PNP_MAX_REPROJECTION_ERROR_PX)
        confidence = float(np.clip(
            0.35 * point_ratio + 0.25 * avg_shape_score + 0.25 * error_score + 0.15 * layout_score,
            0.0,
            1.0,
        ))
        rank_score = confidence + 0.08 * len(inlier_marker_names) - 0.01 * reprojection_error

        candidate_result = {
            "rank_score": rank_score,
            "confidence": confidence,
            "rvec": rvec,
            "tvec": tvec,
            "object_points": inlier_object_points,
            "image_points": inlier_image_points,
            "marker_names": inlier_marker_names,
            "detections": selected_detections,
            "reprojection_error": reprojection_error,
        }
        if best is None or candidate_result["rank_score"] > best["rank_score"]:
            best = candidate_result

    return best


def estimate_color_marker_pose(frame_bgr, camera_matrix, dist_coeffs):
    """
    从 BGR 图像中检测形状编码彩色标记并解算位姿。

    成功返回 ColorMarkerPoseObservation；失败返回 None。
    """
    marker_candidates = detect_color_markers(frame_bgr)
    if len(marker_candidates) < PNP_MIN_POINTS:
        return None

    best = _choose_best_pose(marker_candidates, camera_matrix, dist_coeffs)
    if best is None:
        logger.debug(f"[ColorMarker] 未找到满足形状/几何/PnP 约束的组合，候选标记数={len(marker_candidates)}")
        return None

    return ColorMarkerPoseObservation(
        source="color_board",
        rvec=best["rvec"],
        tvec=best["tvec"],
        image_points=best["image_points"],
        object_points=best["object_points"],
        marker_names=best["marker_names"],
        detections=best["detections"],
        confidence=best["confidence"],
        reprojection_error=best["reprojection_error"],
        inlier_count=len(best["marker_names"]),
    )


def draw_color_marker_debug(frame_bgr, observation):
    """在图像上绘制彩色标记质心、名称、形状和 PnP 质量信息。"""
    for name, detection in observation.detections.items():
        center = tuple(np.round(detection.center).astype(int))
        color = COLOR_MARKER_DRAW_BGR.get(name, (255, 255, 255))
        x, y, w, h = detection.bbox
        cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), color, 1)
        cv2.circle(frame_bgr, center, 5, color, -1)
        cv2.putText(
            frame_bgr,
            f"{name} {detection.shape} s={detection.shape_score:.2f}",
            (center[0] + 6, center[1] - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
        )

    cv2.putText(
        frame_bgr,
        f"COLOR PnP | points={observation.inlier_count} "
        f"err={observation.reprojection_error:.1f}px conf={observation.confidence:.2f}",
        (10, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 0),
        2,
    )
