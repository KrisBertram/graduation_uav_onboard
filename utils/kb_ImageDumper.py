import os
import cv2
import time
import shutil
from datetime import datetime
from pathlib import Path
from typing import Union, Optional, Dict, Callable
import numpy as np
import json

class ImageDumper:
    """
    图像转储工具类，支持多种存储模式和配置选项
    
    Attributes:
        base_path (str): 基础存储路径
        current_path (str): 当前实际使用的存储路径
        interval_type (str): 间隔类型，'frame'或'time'
        interval_value (int/float): 间隔值
        storage_mode (str): 存储模式
        filename_format (str): 文件名格式
        counter (int): 帧计数器
        last_save_time (float): 上次保存时间
        created_subdirs (list): 创建的临时子文件夹列表
    """
    
    # 定义存储模式常量
    MODE_CREATE_IF_NOT_EXIST = "create_if_not_exist"      # 不存在则创建，存在则使用
    MODE_OVERWRITE = "overwrite"                          # 覆盖模式，删除所有内容
    MODE_CLEAN_BEFORE_DUMP = "clean_before_dump"          # 每次转储前清理旧文件
    MODE_TIMESTAMP_SUBDIR = "timestamp_subdir"            # 每次创建带时间戳的子文件夹
    MODE_INCREMENTAL_SUBDIR = "incremental_subdir"        # 创建递增编号的子文件夹
    MODE_SESSION_BASED = "session_based"                  # 基于会话，每次运行创建新文件夹
    
    # 文件名格式常量
    FORMAT_SEQUENTIAL = "sequential"      # 顺序编号：img_0001.jpg
    FORMAT_TIMESTAMP = "timestamp"        # 时间戳：img_20231201_143025_001.jpg
    FORMAT_FRAMECOUNT = "framecount"      # 帧计数：frame_001234.jpg
    FORMAT_CUSTOM = "custom"              # 自定义格式
    
    def __init__(self, 
                 base_path: str,
                 interval_type: str = "frame",
                 interval_value: Union[int, float] = 1,
                 storage_mode: str = "create_if_not_exist",
                 filename_format: str = "sequential",
                 time_unit: str = "seconds",
                 auto_create: bool = True,
                 image_format: str = "jpg",
                 quality: int = 95,
                 max_files_per_dir: int = 1000,
                 enable_logging: bool = True,
                 custom_namer: Optional[Callable] = None):
        """
        初始化ImageDumper
        
        Args:
            base_path: 基础存储路径
            interval_type: 间隔类型，'frame'或'time'
            interval_value: 间隔值，帧数或时间
            storage_mode: 存储模式，见MODE_*常量
            filename_format: 文件名格式，见FORMAT_*常量
            time_unit: 时间单位，'seconds', 'minutes', 'hours'
            auto_create: 是否自动创建目录
            image_format: 图像格式，'jpg', 'png', 'bmp'
            quality: 图像质量（仅对jpg有效），1-100
            max_files_per_dir: 每个目录最大文件数
            enable_logging: 是否启用日志
            custom_namer: 自定义命名函数
        """
        
        # 基础配置
        self.base_path = Path(base_path).resolve()
        self.interval_type = interval_type
        self.interval_value = interval_value
        self.storage_mode = storage_mode
        self.filename_format = filename_format
        self.time_unit = time_unit
        self.auto_create = auto_create
        self.image_format = image_format.lower()
        self.quality = quality
        self.max_files_per_dir = max_files_per_dir
        self.enable_logging = enable_logging
        self.custom_namer = custom_namer
        
        # 内部状态
        self.counter = 0
        self.save_counter = 0
        self.last_save_time = 0
        self.current_path = None
        self.created_subdirs = []
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 初始化路径
        self._initialize_storage()
        
        # 日志
        if self.enable_logging:
            print(f"[ImageDumper] 初始化完成")
            print(f"  存储路径: {self.current_path}")
            print(f"  存储模式: {self.storage_mode}")
            print(f"  间隔设置: {interval_value} {interval_type}({time_unit})")
    
    def _initialize_storage(self):
        """初始化存储路径"""
        
        # 会话模式：在基础路径下创建会话文件夹
        if self.storage_mode == self.MODE_SESSION_BASED:
            self.current_path = self.base_path / f"session_{self.session_id}"
            self._ensure_directory(self.current_path, clear=False)
            
        # 时间戳子文件夹模式：创建带时间戳的子文件夹
        elif self.storage_mode == self.MODE_TIMESTAMP_SUBDIR:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.current_path = self.base_path / f"dump_{timestamp}"
            self._ensure_directory(self.current_path, clear=False)
            self.created_subdirs.append(self.current_path)
            
        # 递增子文件夹模式：查找下一个可用编号
        elif self.storage_mode == self.MODE_INCREMENTAL_SUBDIR:
            self.current_path = self._find_next_available_dir()
            self._ensure_directory(self.current_path, clear=False)
            self.created_subdirs.append(self.current_path)
            
        # 覆盖模式：清空目录
        elif self.storage_mode == self.MODE_OVERWRITE:
            self.current_path = self.base_path
            self._ensure_directory(self.current_path, clear=True)
            
        # 清理模式：只设置路径，在转储时清理
        elif self.storage_mode == self.MODE_CLEAN_BEFORE_DUMP:
            self.current_path = self.base_path
            self._ensure_directory(self.current_path, clear=False)
            
        # 默认模式：不存在则创建
        else:  # MODE_CREATE_IF_NOT_EXIST
            self.current_path = self.base_path
            self._ensure_directory(self.current_path, clear=False)
    
    def _ensure_directory(self, path: Path, clear: bool = False):
        """确保目录存在，可选是否清空"""
        
        if self.auto_create and not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            if self.enable_logging:
                print(f"[ImageDumper] 创建目录: {path}")
        
        if clear and path.exists():
            self._clean_directory(path)
    
    def _clean_directory(self, path: Path):
        """清空目录内容"""
        for item in path.iterdir():
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)
        
        if self.enable_logging:
            print(f"[ImageDumper] 已清空目录: {path}")
    
    def _find_next_available_dir(self) -> Path:
        """查找下一个可用的递增目录名"""
        
        index = 1
        while True:
            dir_name = f"dump_{index:03d}"
            dir_path = self.base_path / dir_name
            if not dir_path.exists():
                return dir_path
            index += 1
    
    def _should_save(self, frame: np.ndarray) -> bool:
        """判断是否应该保存当前帧"""
        
        self.counter += 1
        
        if self.interval_type == "frame":
            return self.counter % self.interval_value == 0
        
        elif self.interval_type == "time":
            current_time = time.time()
            
            # 转换时间单位
            multiplier = 1
            if self.time_unit == "minutes":
                multiplier = 60
            elif self.time_unit == "hours":
                multiplier = 3600
            elif self.time_unit == "milliseconds":
                multiplier = 0.001
            
            interval_seconds = self.interval_value * multiplier
            
            if current_time - self.last_save_time >= interval_seconds:
                self.last_save_time = current_time
                return True
        
        return False
    
    def _generate_filename(self, frame: np.ndarray) -> str:
        """生成文件名"""
        
        if self.custom_namer:
            return self.custom_namer(frame, self.save_counter, self.counter)
        
        if self.filename_format == self.FORMAT_SEQUENTIAL:
            filename = f"img_{self.save_counter:06d}.{self.image_format}"
            
        elif self.filename_format == self.FORMAT_TIMESTAMP:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"img_{timestamp}_{self.save_counter:03d}.{self.image_format}"
            
        elif self.filename_format == self.FORMAT_FRAMECOUNT:
            filename = f"frame_{self.counter:08d}.{self.image_format}"
            
        else:  # 默认使用顺序编号
            filename = f"img_{self.save_counter:06d}.{self.image_format}"
        
        return filename
    
    def _get_save_params(self):
        """获取保存参数"""
        if self.image_format == "jpg" or self.image_format == "jpeg":
            return [cv2.IMWRITE_JPEG_QUALITY, self.quality]
        elif self.image_format == "png":
            return [cv2.IMWRITE_PNG_COMPRESSION, min(9, self.quality // 10)]
        else:
            return []
    
    def dump(self, 
             image: Union[np.ndarray, str],
             metadata: Optional[Dict] = None,
             force_save: bool = False) -> Optional[str]:
        """
        转储图像
        
        Args:
            image: 图像数据（numpy数组）或图像文件路径
            metadata: 元数据字典，将保存为JSON文件
            force_save: 是否强制保存，忽略间隔设置
            
        Returns:
            保存的文件路径，如果未保存则返回None
        """
        
        # 如果是文件路径，读取图像
        if isinstance(image, str):
            if not os.path.exists(image):
                print(f"[ImageDumper] 错误: 文件不存在 - {image}")
                return None
            image_data = cv2.imread(image)
            if image_data is None:
                print(f"[ImageDumper] 错误: 无法读取图像 - {image}")
                return None
        else:
            image_data = image
        
        # 检查是否需要保存
        if not force_save and not self._should_save(image_data):
            return None
        
        # 清理模式：每次转储前清理
        if self.storage_mode == self.MODE_CLEAN_BEFORE_DUMP:
            self._clean_directory(self.current_path)
        
        # 检查目录文件数限制
        if self.max_files_per_dir > 0:
            file_count = len(list(self.current_path.glob(f"*.{self.image_format}")))
            if file_count >= self.max_files_per_dir:
                # 创建新子目录
                new_dir = self.current_path / f"part_{(file_count // self.max_files_per_dir) + 1:03d}"
                new_dir.mkdir(exist_ok=True)
                self.current_path = new_dir
                self.created_subdirs.append(self.current_path)
        
        # 生成文件名
        filename = self._generate_filename(image_data)
        filepath = self.current_path / filename
        
        # 保存图像
        save_params = self._get_save_params()
        success = cv2.imwrite(str(filepath), image_data, save_params)
        
        if not success:
            print(f"[ImageDumper] 错误: 保存图像失败 - {filepath}")
            return None
        
        # 保存元数据
        if metadata:
            metadata_path = filepath.with_suffix('.json')
            metadata['save_time'] = datetime.now().isoformat()
            metadata['frame_count'] = self.counter
            metadata['save_count'] = self.save_counter
            
            try:
                with open(metadata_path, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, indent=2, ensure_ascii=False)
            except Exception as e:
                print(f"[ImageDumper] 警告: 保存元数据失败 - {e}")
        
        self.save_counter += 1
        
        if self.enable_logging:
            print(f"[ImageDumper] 保存图像: {filename}")
        
        return str(filepath)
    
    def dump_batch(self, 
                   images: list,
                   metadata_list: Optional[list] = None) -> list:
        """
        批量转储图像
        
        Args:
            images: 图像列表
            metadata_list: 元数据列表
            
        Returns:
            保存的文件路径列表
        """
        saved_paths = []
        
        for i, image in enumerate(images):
            metadata = metadata_list[i] if metadata_list and i < len(metadata_list) else None
            saved_path = self.dump(image, metadata, force_save=True)
            if saved_path:
                saved_paths.append(saved_path)
        
        return saved_paths
    
    def set_new_base_path(self, new_path: str, mode: Optional[str] = None):
        """
        设置新的基础路径
        
        Args:
            new_path: 新路径
            mode: 可选的存储模式，如果为None则使用当前模式
        """
        self.base_path = Path(new_path).resolve()
        self.save_counter = 0
        
        if mode:
            self.storage_mode = mode
        
        self._initialize_storage()
        
        if self.enable_logging:
            print(f"[ImageDumper] 切换到新路径: {self.current_path}")
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        total_files = 0
        total_size = 0
        
        for subdir in [self.current_path] + self.created_subdirs:
            if subdir.exists():
                for file in subdir.glob(f"*.{self.image_format}"):
                    total_files += 1
                    total_size += file.stat().st_size
        
        return {
            'base_path': str(self.base_path),
            'current_path': str(self.current_path),
            'total_frames': self.counter,
            'saved_frames': self.save_counter,
            'total_files': total_files,
            'total_size_bytes': total_size,
            'total_size_mb': total_size / (1024 * 1024),
            'interval_setting': f"{self.interval_value} {self.interval_type}",
            'storage_mode': self.storage_mode
        }
    
    def create_summary(self, summary_file: str = "dump_summary.txt"):
        """创建转储摘要文件"""
        summary_path = self.base_path / summary_file
        stats = self.get_stats()
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("图像转储摘要\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"基础路径: {stats['base_path']}\n")
            f.write(f"当前路径: {stats['current_path']}\n")
            f.write(f"存储模式: {stats['storage_mode']}\n")
            f.write(f"间隔设置: {stats['interval_setting']}\n\n")
            
            f.write(f"总处理帧数: {stats['total_frames']}\n")
            f.write(f"已保存帧数: {stats['saved_frames']}\n")
            f.write(f"保存的文件数: {stats['total_files']}\n")
            f.write(f"总大小: {stats['total_size_mb']:.2f} MB\n\n")
            
            f.write("创建的子文件夹:\n")
            for i, subdir in enumerate(self.created_subdirs, 1):
                f.write(f"  {i}. {subdir}\n")
            
            f.write("\n" + "=" * 60 + "\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        print(f"[ImageDumper] 摘要已保存到: {summary_path}")
    
    def cleanup(self, keep_latest: int = 3):
        """
        清理旧的转储文件夹
        
        Args:
            keep_latest: 保留最新的几个文件夹
        """
        if self.storage_mode in [self.MODE_TIMESTAMP_SUBDIR, self.MODE_INCREMENTAL_SUBDIR]:
            dirs = list(self.base_path.glob("dump_*"))
            dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            
            for old_dir in dirs[keep_latest:]:
                if old_dir != self.current_path:
                    shutil.rmtree(old_dir)
                    if self.enable_logging:
                        print(f"[ImageDumper] 清理旧文件夹: {old_dir}")
    
    def __del__(self):
        """析构函数，清理资源"""
        if self.enable_logging:
            print(f"[ImageDumper] 销毁，共保存 {self.save_counter} 张图像")


# 使用示例
def example_usage():
    """使用示例"""
    
    # 示例1: 基本使用 - 每5帧保存一张
    print("=== 示例1: 基本使用 ===")
    dumper1 = ImageDumper(
        base_path="./output/basic",
        interval_type="frame",
        interval_value=5,
        storage_mode=ImageDumper.MODE_CREATE_IF_NOT_EXIST
    )
    
    # 模拟保存一些图像
    for i in range(100):
        # 创建模拟图像
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.putText(img, f"Frame {i}", (10, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 转储图像
        dumper1.dump(img)
    
    # 获取统计信息
    stats = dumper1.get_stats()
    print(f"保存了 {stats['saved_frames']} 张图像")
    
    # 示例2: 时间间隔模式 - 每0.5秒保存一张
    print("\n=== 示例2: 时间间隔模式 ===")
    dumper2 = ImageDumper(
        base_path="./output/timed",
        interval_type="time",
        interval_value=0.5,  # 0.5秒
        time_unit="seconds",
        storage_mode=ImageDumper.MODE_TIMESTAMP_SUBDIR,
        filename_format=ImageDumper.FORMAT_TIMESTAMP
    )
    
    # 示例3: 每次创建新文件夹
    print("\n=== 示例3: 递增子文件夹 ===")
    dumper3 = ImageDumper(
        base_path="./output/incremental",
        interval_type="frame",
        interval_value=10,
        storage_mode=ImageDumper.MODE_INCREMENTAL_SUBDIR,
        max_files_per_dir=10  # 每目录最多10个文件
    )
    
    # 示例4: 带元数据的保存
    print("\n=== 示例4: 带元数据的保存 ===")
    dumper4 = ImageDumper(
        base_path="./output/metadata",
        storage_mode=ImageDumper.MODE_CLEAN_BEFORE_DUMP
    )
    
    # 保存带元数据的图像
    metadata = {
        "source": "camera_1",
        "resolution": "1920x1080",
        "timestamp": datetime.now().isoformat()
    }
    
    img = np.ones((200, 200, 3), dtype=np.uint8) * 128
    saved_path = dumper4.dump(img, metadata=metadata, force_save=True)
    print(f"保存到: {saved_path}")
    
    # 创建摘要
    dumper4.create_summary()
    
    return [dumper1, dumper2, dumper3, dumper4]


# 高级使用示例：从视频中提取帧
def extract_frames_from_video(video_path: str, output_dir: str, fps: float = 1.0):
    """
    从视频中按指定帧率提取帧
    
    Args:
        video_path: 视频文件路径
        output_dir: 输出目录
        fps: 提取的帧率（帧/秒）
    """
    if not os.path.exists(video_path):
        print(f"视频文件不存在: {video_path}")
        return
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频文件: {video_path}")
        return
    
    # 创建dumper，按时间间隔提取
    dumper = ImageDumper(
        base_path=output_dir,
        interval_type="time",
        interval_value=1.0/fps,  # 转换为时间间隔
        storage_mode=ImageDumper.MODE_TIMESTAMP_SUBDIR,
        filename_format=ImageDumper.FORMAT_TIMESTAMP,
        enable_logging=True
    )
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 添加帧号到图像
        cv2.putText(frame, f"Frame: {frame_count}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # 添加时间戳
        timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        cv2.putText(frame, f"Time: {timestamp:.2f}s", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # 创建元数据
        metadata = {
            "video_source": os.path.basename(video_path),
            "frame_index": frame_count,
            "timestamp_seconds": timestamp,
            "video_time": time.strftime('%H:%M:%S', time.gmtime(timestamp))
        }
        
        # 转储帧
        dumper.dump(frame, metadata=metadata)
        
        frame_count += 1
    
    cap.release()
    
    # 输出统计信息
    stats = dumper.get_stats()
    print(f"\n视频提取完成:")
    print(f"  总帧数: {frame_count}")
    print(f"  保存帧数: {stats['saved_frames']}")
    print(f"  输出目录: {stats['current_path']}")
    
    # 创建摘要
    dumper.create_summary()
    
    return dumper


if __name__ == "__main__":
    # 运行示例
    dumpers = example_usage()
    
    print("\n=== 所有示例完成 ===")
    for i, dumper in enumerate(dumpers, 1):
        stats = dumper.get_stats()
        print(f"示例{i}: 保存了 {stats['saved_frames']} 张图像到 {stats['current_path']}")