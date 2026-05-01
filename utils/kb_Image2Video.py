import os
import cv2
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple, Union, Dict
import glob
import re
from datetime import datetime
import logging
from dataclasses import dataclass
from enum import Enum
import time

class VideoCodec(Enum):
    """视频编码器枚举"""
    XVID = "XVID"
    MP4V = "mp4v"
    H264 = "H264"
    X264 = "X264"
    MJPG = "MJPG"
    DIVX = "DIVX"
    WMV1 = "WMV1"
    WMV2 = "WMV2"
    VP80 = "VP80"
    VP90 = "VP90"

class SortMethod(Enum):
    """排序方法枚举"""
    NATURAL = "natural"       # 自然排序（考虑数字）
    ALPHABETIC = "alphabetic" # 字母排序
    MODIFIED_TIME = "mtime"   # 修改时间
    CREATED_TIME = "ctime"    # 创建时间
    FRAME_NUMBER = "frame"    # 根据帧号排序

@dataclass
class VideoConfig:
    """视频配置类"""
    fps: float = 30.0
    codec: VideoCodec = VideoCodec.XVID
    frame_size: Optional[Tuple[int, int]] = None
    quality: int = 95
    is_color: bool = True
    fourcc: Optional[str] = None
    
    def get_fourcc(self) -> str:
        """获取fourcc编码"""
        if self.fourcc:
            return self.fourcc
        return self.codec.value

class Image2Video:
    """
    将图片序列合成为视频的工具类
    
    Attributes:
        input_dir (Path): 输入图片目录
        output_path (Path): 输出视频路径
        config (VideoConfig): 视频配置
        logger (logging.Logger): 日志记录器
        supported_formats (list): 支持的图片格式
    """
    
    def __init__(self, 
                 input_dir: Union[str, Path],
                 output_path: Union[str, Path],
                 fps: float = 30.0,
                 codec: Union[str, VideoCodec] = VideoCodec.XVID,
                 frame_size: Optional[Tuple[int, int]] = None,
                 quality: int = 95,
                 log_level: int = logging.INFO):
        """
        初始化Image2Video
        
        Args:
            input_dir: 输入图片目录
            output_path: 输出视频路径
            fps: 帧率
            codec: 视频编码器
            frame_size: 视频尺寸 (宽, 高)，如果为None则使用第一张图片的尺寸
            quality: 视频质量 (0-100)
            log_level: 日志级别
        """
        
        # 初始化路径
        self.input_dir = Path(input_dir).resolve()
        self.output_path = Path(output_path).resolve()
        
        # 初始化配置
        if isinstance(codec, str):
            try:
                codec = VideoCodec(codec.upper())
            except ValueError:
                codec = VideoCodec.XVID
                self._log_warning(f"未知的编码器 {codec}，使用默认编码器 XVID")
        
        self.config = VideoConfig(
            fps=fps,
            codec=codec,
            frame_size=frame_size,
            quality=quality,
            is_color=True
        )
        
        # 初始化日志
        self.logger = self._setup_logger(log_level)
        
        # 支持的图片格式
        self.supported_formats = [
            '.jpg', '.jpeg', '.png', '.bmp', '.tiff', 
            '.tif', '.webp', '.ppm', '.pgm', '.pbm'
        ]
        
        self.logger.info(f"初始化完成: 输入目录={self.input_dir}, 输出={self.output_path}")
    
    def _setup_logger(self, level: int) -> logging.Logger:
        """设置日志记录器"""
        logger = logging.getLogger(f"Image2Video_{datetime.now().timestamp()}")
        logger.setLevel(level)
        
        if not logger.handlers:
            # 控制台处理器
            ch = logging.StreamHandler()
            ch.setLevel(level)
            
            # 格式化器
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            ch.setFormatter(formatter)
            logger.addHandler(ch)
        
        return logger
    
    def _log_info(self, message: str):
        """记录信息日志"""
        self.logger.info(message)
    
    def _log_warning(self, message: str):
        """记录警告日志"""
        self.logger.warning(message)
    
    def _log_error(self, message: str):
        """记录错误日志"""
        self.logger.error(message)
    
    def _log_debug(self, message: str):
        """记录调试日志"""
        self.logger.debug(message)
    
    def _get_image_files(self, 
                        pattern: str = "*",
                        recursive: bool = False,
                        sort_method: SortMethod = SortMethod.NATURAL) -> List[Path]:
        """
        获取图片文件列表
        
        Args:
            pattern: 文件名模式，例如 "*.jpg" 或 "frame_*.png"
            recursive: 是否递归搜索子目录
            sort_method: 排序方法
            
        Returns:
            图片文件路径列表
        """
        
        if not self.input_dir.exists():
            raise FileNotFoundError(f"输入目录不存在: {self.input_dir}")
        
        if not self.input_dir.is_dir():
            raise NotADirectoryError(f"输入路径不是目录: {self.input_dir}")
        
        # 构建搜索模式
        if pattern == "*":
            # 搜索所有支持的格式
            patterns = [f"*{fmt}" for fmt in self.supported_formats]
            files = []
            for p in patterns:
                if recursive:
                    files.extend(self.input_dir.rglob(p))
                else:
                    files.extend(self.input_dir.glob(p))
        else:
            # 使用指定的模式
            if recursive:
                files = list(self.input_dir.rglob(pattern))
            else:
                files = list(self.input_dir.glob(pattern))
        
        # 过滤掉目录，只保留文件
        files = [f for f in files if f.is_file()]
        
        if not files:
            self._log_warning(f"在目录 {self.input_dir} 中未找到匹配 {pattern} 的图片文件")
            return []
        
        # 排序
        files = self._sort_files(files, sort_method)
        
        self._log_info(f"找到 {len(files)} 个图片文件")
        return files
    
    def _sort_files(self, files: List[Path], method: SortMethod) -> List[Path]:
        """对文件进行排序"""
        
        if method == SortMethod.ALPHABETIC:
            return sorted(files, key=lambda x: x.name)
        
        elif method == SortMethod.MODIFIED_TIME:
            return sorted(files, key=lambda x: x.stat().st_mtime)
        
        elif method == SortMethod.CREATED_TIME:
            return sorted(files, key=lambda x: x.stat().st_ctime)
        
        elif method == SortMethod.FRAME_NUMBER:
            # 尝试从文件名中提取帧号
            def extract_frame_number(filepath: Path) -> int:
                name = filepath.stem
                # 查找连续的数字
                numbers = re.findall(r'\d+', name)
                return int(numbers[-1]) if numbers else 0
            
            return sorted(files, key=extract_frame_number)
        
        else:  # NATURAL
            return self._natural_sort(files)
    
    def _natural_sort(self, files: List[Path]) -> List[Path]:
        """自然排序（考虑数字）"""
        
        def convert(text):
            return int(text) if text.isdigit() else text.lower()
        
        def alphanum_key(key):
            return [convert(c) for c in re.split(r'(\d+)', key.name)]
        
        return sorted(files, key=alphanum_key)
    
    def _get_video_writer(self, 
                         frame_size: Tuple[int, int],
                         output_path: Path) -> cv2.VideoWriter:
        """
        创建视频写入器
        
        Args:
            frame_size: 帧尺寸 (宽, 高)
            output_path: 输出路径
            
        Returns:
            VideoWriter对象
        """
        
        # 确保输出目录存在
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 根据文件扩展名选择编码器
        suffix = output_path.suffix.lower()
        
        # 设置fourcc编码
        fourcc_str = self.config.get_fourcc()
        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
        
        # 创建VideoWriter
        writer = cv2.VideoWriter(
            str(output_path),
            fourcc,
            self.config.fps,
            frame_size,
            self.config.is_color
        )
        
        if not writer.isOpened():
            raise RuntimeError(f"无法创建视频文件: {output_path}")
        
        return writer
    
    def _validate_and_resize_image(self, 
                                 img: np.ndarray, 
                                 target_size: Optional[Tuple[int, int]]) -> np.ndarray:
        """
        验证并调整图片尺寸
        
        Args:
            img: 输入图片
            target_size: 目标尺寸 (宽, 高)
            
        Returns:
            调整后的图片
        """
        
        if img is None:
            raise ValueError("读取图片失败")
        
        if target_size:
            current_height, current_width = img.shape[:2]
            target_width, target_height = target_size
            
            if (current_width, current_height) != (target_width, target_height):
                img = cv2.resize(img, (target_width, target_height))
                self._log_debug(f"调整图片尺寸: {current_width}x{current_height} -> {target_width}x{target_height}")
        
        return img
    
    def create_video(self,
                    pattern: str = "*",
                    recursive: bool = False,
                    sort_method: Union[str, SortMethod] = SortMethod.NATURAL,
                    start_frame: int = 0,
                    end_frame: Optional[int] = None,
                    frame_interval: int = 1,
                    batch_size: int = 100,
                    progress_callback: Optional[callable] = None,
                    metadata: Optional[Dict] = None) -> Dict:
        """
        创建视频
        
        Args:
            pattern: 文件名模式
            recursive: 是否递归搜索
            sort_method: 排序方法
            start_frame: 起始帧索引
            end_frame: 结束帧索引
            frame_interval: 帧间隔（每隔几帧取一帧）
            batch_size: 批量处理大小
            progress_callback: 进度回调函数
            metadata: 元数据
            
        Returns:
            处理统计信息
        """
        
        start_time = time.time()
        
        # 转换排序方法
        if isinstance(sort_method, str):
            try:
                sort_method = SortMethod(sort_method.lower())
            except ValueError:
                sort_method = SortMethod.NATURAL
                self._log_warning(f"未知的排序方法，使用自然排序")
        
        # 获取图片文件列表
        try:
            image_files = self._get_image_files(pattern, recursive, sort_method)
        except Exception as e:
            self._log_error(f"获取图片文件失败: {e}")
            return {"success": False, "error": str(e)}
        
        if not image_files:
            return {"success": False, "error": "未找到图片文件"}
        
        # 应用帧范围
        if end_frame is None:
            end_frame = len(image_files) - 1
        
        # 验证范围
        if start_frame < 0:
            start_frame = 0
        if end_frame >= len(image_files):
            end_frame = len(image_files) - 1
        if start_frame > end_frame:
            return {"success": False, "error": "起始帧大于结束帧"}
        
        # 读取第一张图片以获取尺寸
        first_image_path = image_files[start_frame]
        first_img = cv2.imread(str(first_image_path))
        
        if first_img is None:
            self._log_error(f"无法读取第一张图片: {first_image_path}")
            return {"success": False, "error": f"无法读取图片: {first_image_path}"}
        
        # 确定视频尺寸
        if self.config.frame_size:
            frame_size = self.config.frame_size
        else:
            height, width = first_img.shape[:2]
            frame_size = (width, height)
        
        self._log_info(f"视频设置: {frame_size[0]}x{frame_size[1]}, {self.config.fps} FPS, 编码器: {self.config.get_fourcc()}")
        
        # 创建视频写入器
        try:
            video_writer = self._get_video_writer(frame_size, self.output_path)
        except Exception as e:
            self._log_error(f"创建视频写入器失败: {e}")
            return {"success": False, "error": str(e)}
        
        # 统计信息
        stats = {
            "total_frames": len(image_files),
            "processed_frames": 0,
            "skipped_frames": 0,
            "failed_frames": 0,
            "output_path": str(self.output_path),
            "frame_size": frame_size,
            "fps": self.config.fps,
            "codec": self.config.get_fourcc(),
            "start_time": datetime.now().isoformat()
        }
        
        # 处理图片序列
        frame_count = 0
        saved_count = 0
        
        for i in range(start_frame, end_frame + 1, frame_interval):
            img_path = image_files[i]
            
            try:
                # 读取图片
                img = cv2.imread(str(img_path))
                
                if img is None:
                    self._log_warning(f"无法读取图片，跳过: {img_path}")
                    stats["failed_frames"] += 1
                    continue
                
                # 调整尺寸
                img = self._validate_and_resize_image(img, frame_size)
                
                # 写入视频帧
                video_writer.write(img)
                saved_count += 1
                
                # 更新进度
                frame_count += 1
                if progress_callback and frame_count % 10 == 0:
                    progress = (i - start_frame) / (end_frame - start_frame + 1) * 100
                    progress_callback(progress, frame_count, str(img_path))
                
                # 批量处理日志
                if frame_count % batch_size == 0:
                    self._log_info(f"已处理 {frame_count}/{len(range(start_frame, end_frame + 1, frame_interval))} 帧")
                
            except Exception as e:
                self._log_error(f"处理图片失败 {img_path}: {e}")
                stats["failed_frames"] += 1
                continue
        
        # 释放资源
        video_writer.release()
        
        # 计算处理时间
        end_time = time.time()
        processing_time = end_time - start_time
        
        # 更新统计信息
        stats.update({
            "processed_frames": frame_count,
            "saved_frames": saved_count,
            "success": True,
            "processing_time_seconds": processing_time,
            "processing_time_minutes": processing_time / 60,
            "average_fps_processed": frame_count / processing_time if processing_time > 0 else 0,
            "end_time": datetime.now().isoformat(),
            "frame_interval_used": frame_interval,
            "actual_output_fps": saved_count / (saved_count / self.config.fps) if saved_count > 0 else 0
        })
        
        self._log_info(f"视频创建完成: {self.output_path}")
        self._log_info(f"统计信息: 处理了 {frame_count} 帧，保存了 {saved_count} 帧，耗时 {processing_time:.2f} 秒")
        
        return stats
    
    def create_video_from_frames(self,
                               frames: List[np.ndarray],
                               output_path: Optional[Union[str, Path]] = None,
                               frame_size: Optional[Tuple[int, int]] = None) -> Dict:
        """
        从内存中的帧列表创建视频
        
        Args:
            frames: 帧列表
            output_path: 输出路径，如果为None则使用初始化时的路径
            frame_size: 帧尺寸
            
        Returns:
            处理统计信息
        """
        
        if not frames:
            return {"success": False, "error": "帧列表为空"}
        
        # 使用指定的输出路径或默认路径
        if output_path:
            output_path = Path(output_path).resolve()
        else:
            output_path = self.output_path
        
        # 确定视频尺寸
        if frame_size:
            target_size = frame_size
        elif self.config.frame_size:
            target_size = self.config.frame_size
        else:
            height, width = frames[0].shape[:2]
            target_size = (width, height)
        
        # 创建视频写入器
        try:
            video_writer = self._get_video_writer(target_size, output_path)
        except Exception as e:
            self._log_error(f"创建视频写入器失败: {e}")
            return {"success": False, "error": str(e)}
        
        start_time = time.time()
        saved_count = 0
        
        for i, frame in enumerate(frames):
            try:
                # 调整尺寸
                frame = self._validate_and_resize_image(frame, target_size)
                
                # 写入视频
                video_writer.write(frame)
                saved_count += 1
                
                # 进度日志
                if i % 100 == 0:
                    self._log_debug(f"已处理 {i}/{len(frames)} 帧")
                    
            except Exception as e:
                self._log_error(f"处理第 {i} 帧失败: {e}")
                continue
        
        # 释放资源
        video_writer.release()
        
        # 统计信息
        processing_time = time.time() - start_time
        
        stats = {
            "success": True,
            "total_frames": len(frames),
            "saved_frames": saved_count,
            "failed_frames": len(frames) - saved_count,
            "output_path": str(output_path),
            "frame_size": target_size,
            "fps": self.config.fps,
            "processing_time_seconds": processing_time,
            "average_fps_processed": saved_count / processing_time if processing_time > 0 else 0
        }
        
        self._log_info(f"从内存帧创建视频完成: {output_path}")
        
        return stats
    
    def create_timelapse(self,
                        pattern: str = "*",
                        speed_factor: float = 10.0,
                        output_suffix: str = "_timelapse") -> Dict:
        """
        创建延时摄影视频
        
        Args:
            pattern: 文件名模式
            speed_factor: 加速倍数
            output_suffix: 输出文件后缀
            
        Returns:
            处理统计信息
        """
        
        # 计算延时摄影的帧率
        timelapse_fps = self.config.fps * speed_factor
        
        # 修改输出路径
        original_output = self.output_path
        timelapse_output = original_output.with_name(
            f"{original_output.stem}{output_suffix}{original_output.suffix}"
        )
        
        self._log_info(f"创建延时摄影视频，加速倍数: {speed_factor}, 输出帧率: {timelapse_fps}")
        
        # 保存原始配置
        original_fps = self.config.fps
        
        # 临时修改配置
        self.config.fps = timelapse_fps
        
        # 创建视频
        result = self.create_video(pattern=pattern)
        
        # 恢复配置
        self.config.fps = original_fps
        
        # 修改输出路径
        if result.get("success"):
            result["output_path"] = str(timelapse_output)
            result["speed_factor"] = speed_factor
            result["timelapse_fps"] = timelapse_fps
        
        return result
    
    def create_multiple_resolutions(self,
                                  pattern: str = "*",
                                  resolutions: List[Tuple[int, int]] = None) -> Dict:
        """
        创建多种分辨率的视频
        
        Args:
            pattern: 文件名模式
            resolutions: 分辨率列表，如果为None则使用常用分辨率
            
        Returns:
            处理结果统计
        """
        
        if resolutions is None:
            resolutions = [
                (1920, 1080),  # 1080p
                (1280, 720),   # 720p
                (854, 480),    # 480p
                (640, 360),    # 360p
            ]
        
        results = {}
        
        for width, height in resolutions:
            self._log_info(f"创建 {width}x{height} 分辨率视频")
            
            # 修改输出路径
            original_output = self.output_path
            res_output = original_output.with_name(
                f"{original_output.stem}_{width}x{height}{original_output.suffix}"
            )
            
            # 临时修改配置
            original_size = self.config.frame_size
            self.config.frame_size = (width, height)
            
            # 创建视频
            result = self.create_video(pattern=pattern)
            
            # 恢复配置
            self.config.frame_size = original_size
            
            # 重命名输出文件
            if result.get("success"):
                result["output_path"] = str(res_output)
            
            results[f"{width}x{height}"] = result
        
        return results
    
    def get_preview_info(self,
                        pattern: str = "*",
                        max_samples: int = 10) -> Dict:
        """
        获取预览信息
        
        Args:
            pattern: 文件名模式
            max_samples: 最大采样数
            
        Returns:
            预览信息
        """
        
        try:
            image_files = self._get_image_files(pattern)
        except Exception as e:
            return {"success": False, "error": str(e)}
        
        if not image_files:
            return {"success": False, "error": "未找到图片文件"}
        
        # 采样一些图片获取信息
        sample_indices = np.linspace(0, len(image_files)-1, min(max_samples, len(image_files)), dtype=int)
        sample_files = [image_files[i] for i in sample_indices]
        
        resolutions = []
        channels_list = []
        
        for img_path in sample_files[:5]:  # 只检查前5张
            img = cv2.imread(str(img_path))
            if img is not None:
                height, width = img.shape[:2]
                channels = img.shape[2] if len(img.shape) == 3 else 1
                resolutions.append((width, height))
                channels_list.append(channels)
        
        if not resolutions:
            return {"success": False, "error": "无法读取任何图片"}
        
        # 统计信息
        unique_resolutions = set(resolutions)
        unique_channels = set(channels_list)
        
        info = {
            "success": True,
            "total_images": len(image_files),
            "file_list_sample": [str(f) for f in sample_files],
            "resolutions_found": list(unique_resolutions),
            "channels_found": list(unique_channels),
            "suggested_resolution": resolutions[0] if resolutions else None,
            "suggested_fps": self.config.fps,
            "input_directory": str(self.input_dir),
            "first_image": str(image_files[0]),
            "last_image": str(image_files[-1]),
            "date_range": {
                "first_modified": datetime.fromtimestamp(image_files[0].stat().st_mtime).isoformat(),
                "last_modified": datetime.fromtimestamp(image_files[-1].stat().st_mtime).isoformat()
            }
        }
        
        return info


# 使用示例和工具函数
def example_usage():
    """使用示例"""
    
    print("=== Image2Video 使用示例 ===")
    
    # 示例1: 基本使用
    print("\n1. 基本使用:")
    converter = Image2Video(
        input_dir="./output/basic",  # 图片目录
        output_path="./videos/basic_output.avi",  # 输出视频路径
        fps=30.0,
        codec=VideoCodec.XVID,
        frame_size=(640, 480)
    )
    
    # 创建视频
    stats = converter.create_video(
        pattern="*.jpg",
        recursive=False,
        sort_method=SortMethod.NATURAL,
        frame_interval=1
    )
    
    if stats.get("success"):
        print(f"视频创建成功: {stats['output_path']}")
        print(f"处理了 {stats['processed_frames']} 帧，耗时 {stats['processing_time_seconds']:.2f} 秒")
    
    # 示例2: 创建延时摄影
    print("\n2. 创建延时摄影:")
    converter2 = Image2Video(
        input_dir="./output/timed",
        output_path="./videos/timelapse.avi",
        fps=30.0
    )
    
    timelapse_stats = converter2.create_timelapse(
        pattern="*.jpg",
        speed_factor=20.0  # 20倍速度
    )
    
    # 示例3: 获取预览信息
    print("\n3. 获取预览信息:")
    preview = converter.get_preview_info(pattern="*.jpg")
    if preview.get("success"):
        print(f"找到 {preview['total_images']} 张图片")
        print(f"分辨率: {preview['resolutions_found']}")
        print(f"建议分辨率: {preview['suggested_resolution']}")
    
    # 示例4: 多种分辨率输出
    print("\n4. 创建多种分辨率:")
    converter3 = Image2Video(
        input_dir="./output/basic",
        output_path="./videos/multi_resolution.mp4",
        fps=25.0,
        codec=VideoCodec.H264
    )
    
    multi_res_stats = converter3.create_multiple_resolutions(
        pattern="*.jpg",
        resolutions=[(1920, 1080), (1280, 720), (640, 480)]
    )
    
    for res, result in multi_res_stats.items():
        if result.get("success"):
            print(f"{res}: 成功")
        else:
            print(f"{res}: 失败 - {result.get('error')}")


def create_video_from_images(input_dir: str, 
                           output_file: str, 
                           fps: float = 30.0,
                           pattern: str = "*.jpg",
                           sort_by: str = "natural") -> bool:
    """
    快速创建视频的便捷函数
    
    Args:
        input_dir: 输入图片目录
        output_file: 输出视频文件
        fps: 帧率
        pattern: 文件模式
        sort_by: 排序方式
        
    Returns:
        是否成功
    """
    
    try:
        # 创建转换器
        converter = Image2Video(
            input_dir=input_dir,
            output_path=output_file,
            fps=fps
        )
        
        # 转换排序方法
        if sort_by == "name":
            sort_method = SortMethod.ALPHABETIC
        elif sort_by == "time":
            sort_method = SortMethod.MODIFIED_TIME
        elif sort_by == "number":
            sort_method = SortMethod.FRAME_NUMBER
        else:
            sort_method = SortMethod.NATURAL
        
        # 创建视频
        stats = converter.create_video(
            pattern=pattern,
            sort_method=sort_method
        )
        
        return stats.get("success", False)
        
    except Exception as e:
        print(f"创建视频失败: {e}")
        return False


def batch_convert_images_to_videos(input_dirs: List[str],
                                  output_dir: str,
                                  fps: float = 30.0,
                                  pattern: str = "*.jpg") -> Dict:
    """
    批量将多个目录的图片转换为视频
    
    Args:
        input_dirs: 输入目录列表
        output_dir: 输出目录
        fps: 帧率
        pattern: 文件模式
        
    Returns:
        批量处理结果
    """
    
    results = {}
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for input_dir in input_dirs:
        input_path = Path(input_dir)
        if not input_path.exists():
            results[str(input_path)] = {"success": False, "error": "目录不存在"}
            continue
        
        # 生成输出文件名
        output_file = output_path / f"{input_path.name}_output.mp4"
        
        try:
            # 创建转换器
            converter = Image2Video(
                input_dir=input_dir,
                output_path=output_file,
                fps=fps,
                codec=VideoCodec.H264
            )
            
            # 创建视频
            stats = converter.create_video(pattern=pattern)
            
            results[str(input_path)] = {
                "success": stats.get("success", False),
                "output_file": str(output_file),
                "stats": stats
            }
            
            print(f"转换完成: {input_dir} -> {output_file}")
            
        except Exception as e:
            results[str(input_path)] = {
                "success": False,
                "error": str(e)
            }
            print(f"转换失败 {input_dir}: {e}")
    
    return results


if __name__ == "__main__":
    # 运行示例
    example_usage()
    
    # 快速使用示例
    print("\n=== 快速使用示例 ===")
    
    # 检查示例目录是否存在
    sample_dir = "./output/basic"
    if Path(sample_dir).exists():
        success = create_video_from_images(
            input_dir=sample_dir,
            output_file="./videos/quick_output.mp4",
            fps=25.0,
            pattern="*.jpg",
            sort_by="natural"
        )
        
        if success:
            print("快速创建视频成功!")
        else:
            print("快速创建视频失败!")
    else:
        print(f"示例目录不存在，请先运行ImageDumper示例或创建目录: {sample_dir}")
    
    # 批量转换示例
    print("\n=== 批量转换示例 ===")
    
    # 假设有多个图片目录
    test_dirs = ["./output/basic", "./output/timed", "./output/metadata"]
    existing_dirs = [d for d in test_dirs if Path(d).exists()]
    
    if existing_dirs:
        batch_results = batch_convert_images_to_videos(
            input_dirs=existing_dirs,
            output_dir="./videos/batch_output",
            fps=30.0,
            pattern="*.jpg"
        )
        
        print(f"批量转换完成，处理了 {len(existing_dirs)} 个目录")
    else:
        print("没有找到示例目录，跳过批量转换示例")