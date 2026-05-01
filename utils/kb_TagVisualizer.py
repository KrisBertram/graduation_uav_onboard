import cv2
import numpy as np

class TagVisualizer:
    ''' AprilTag 可视化工具类 '''

    @staticmethod
    def compute_pixel_scale_on_tag(tag_corners_px, tag_size_m):
        """
        计算 AprilTag 所在平面上：
        1 像素 ≈ 多少米

        tag_corners_px: (4,2) AprilTag 四个角点（像素）
        tag_size_m: AprilTag 实际边长（米）
        返 回：米 / 像素
        """
        # 计算像素边长（取两条边平均，更稳）
        edge1 = np.linalg.norm(tag_corners_px[0] - tag_corners_px[1])
        edge2 = np.linalg.norm(tag_corners_px[1] - tag_corners_px[2])
        pixel_edge = 0.5 * (edge1 + edge2)

        meter_per_pixel = tag_size_m / pixel_edge
        return meter_per_pixel


    @staticmethod
    def draw_image_center(img, color=(255, 255, 0)):
        '''
        在图像中心画十字标记：

        img: 输入图像
        color: 标记颜色
        返 回：图像中心点坐标 (cx, cy)
        '''
        h, w = img.shape[:2]
        cx, cy = w // 2, h // 2

        cv2.drawMarker(
            img,
            (cx, cy),
            color,
            markerType=cv2.MARKER_CROSS,
            markerSize=20,
            thickness=2
        )

        cv2.putText(
            img,
            "Image Center",
            (cx + 10, cy - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1
        )

        return (cx, cy)


    @staticmethod
    def draw_tvec_vector_scaled(
        img,
        origin_px,
        tvec,
        meter_per_pixel,
        scale=1.0,
        color=(0, 0, 255)
    ):
        """
        在图像中按真实比例绘制 tvec 向量 (仅 X / Y)
        
        img: 输入图像
        origin_px: 图像中的起点（像素）
        tvec: solvePnP 得到的 (3,1)
        meter_per_pixel: 米 / 像素
        scale: 视觉放大倍率（不影响物理比例，只影响显示）
        """

        # 相机坐标系：X 右，Y 下
        dx_m = tvec[0][0]
        dy_m = tvec[1][0]

        dx_px = int(dx_m / meter_per_pixel * scale)
        dy_px = int(dy_m / meter_per_pixel * scale)

        end_px = (
            int(origin_px[0] + dx_px),
            int(origin_px[1] + dy_px)
        )

        cv2.arrowedLine(
            img,
            origin_px,
            end_px,
            color,
            2,
            tipLength=0.15
        )

        cv2.putText(
            img,
            f"tvec XY = ({dx_m:.6f}, {dy_m:.6f}) m",
            (origin_px[0] + 10, origin_px[1] + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1
        )


    @staticmethod
    def draw_z_bar(img, z_m, max_z=3.0):
        """
        用竖条表示 Z 距离（高度）
        """
        h, w = img.shape[:2]
        bar_x = w - 40
        bar_y0 = int(h * 0.2)
        bar_y1 = int(h * 0.8)

        z_norm = np.clip(z_m / max_z, 0.0, 1.0)
        bar_height = int((bar_y1 - bar_y0) * z_norm)

        cv2.rectangle(img, (bar_x, bar_y0), (bar_x + 10, bar_y1), (200, 200, 200), 1)
        cv2.rectangle(
            img,
            (bar_x, bar_y1 - bar_height),
            (bar_x + 10, bar_y1),
            (0, 0, 255),
            -1
        )

        cv2.putText(
            img,
            f"Z={z_m:.2f}m",
            (bar_x - 20, bar_y1 + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1
        )


    @staticmethod
    def draw_reprojected_center(
        img,
        rvec,
        tvec,
        cameraMatrix,
        distCoeffs,
        color=(255, 128, 0)
    ):
        """
        将 3D 点 (0,0,0) 投影回图像并绘制
        输入:
            img: 输入图像
            rvec, tvec: solvePnP 输出的旋转向量和平移向量
            cameraMatrix, distCoeffs: 相机内参和畸变参数
            color: 绘制颜色
        返回: reproj: 重投影像素坐标
        """
        reprojected_center, _ = cv2.projectPoints(
            np.array([[0, 0, 0]], dtype=np.float32),
            rvec,
            tvec,
            cameraMatrix,
            distCoeffs
        )
        reproj = tuple(reprojected_center[0][0].astype(int))
        cv2.circle(img, reproj, 6, color, -1)
        return reproj


    @staticmethod
    def compute_reprojection_error(
        object_points,
        image_points,
        rvec,
        tvec,
        cameraMatrix,
        distCoeffs
    ):
        """
        计算所有角点的重投影误差，如果误差 < 0.5px 说明模型是健康的
        输入：
            object_points: (N,3)    3D 角点坐标
            image_points: (N,2)     2D 角点像素坐标
            rvec, tvec: (vector)    solvePnP 输出的旋转向量和平移向量
            cameraMatrix, distCoeffs: 相机内参和畸变参数
        返回：
            mean_error (float)      平均重投影误差（像素）
            per_point_error (N,)    每个角点的重投影误差（像素）
        """
        projected, _ = cv2.projectPoints(
            object_points,
            rvec,
            tvec,
            cameraMatrix,
            distCoeffs
        )
        projected = projected.reshape(-1, 2)
        error = np.linalg.norm(projected - image_points, axis=1)
        mean_error = np.mean(error)
        return mean_error, error