"""
调试图像、视频转换和 UDP 图传工具初始化。
"""

from loguru import logger

from utils.kb_Image2Video import Image2Video, SortMethod, VideoCodec
from utils.kb_ImageDumper import ImageDumper
from utils.udp_video_sender import VideoSender


def init_debug_tools(udp_sender_enabled, udp_receiver_ip, udp_sender_quality):
    """初始化调试工具: 图像转储器、视频转换器、UDP 视频发送器"""
    # 图像转储器
    image_dumper = ImageDumper(
        base_path="./image_output",
        interval_type="time",
        interval_value=0.08,  # 0.04 秒
        time_unit="seconds",
        storage_mode=ImageDumper.MODE_OVERWRITE,
        filename_format=ImageDumper.FORMAT_SEQUENTIAL
    )

    # 视频转换器
    video_converter = Image2Video(
        input_dir="./image_output",  # 图片目录
        output_path="./image_output/video/output.mp4",  # 输出视频路径
        fps=12.0,
        codec=VideoCodec.XVID
    )

    # UDP 高速图传
    udp_sender = VideoSender(dest_ip=udp_receiver_ip, jpeg_quality=udp_sender_quality) if udp_sender_enabled else None
    if udp_sender_enabled:
        udp_sender.start()

    return image_dumper, video_converter, udp_sender


def finish_debug_tools(debug_mode_enabled, image_dumper, video_converter):
    """在程序退出时保存调试统计，并按需生成视频。"""
    if debug_mode_enabled:
        # 获取统计信息
        stats = image_dumper.get_stats()
        logger.info(f"保存了 {stats['saved_frames']} 张图像")

        # 创建视频
        stats = video_converter.create_video(
            pattern="*.jpg",
            recursive=False,
            sort_method=SortMethod.NATURAL,
            frame_interval=1
        )

        if stats.get("success"):
            logger.info(f"视频创建成功: {stats['output_path']}")
            logger.info(f"处理了 {stats['processed_frames']} 帧，耗时 {stats['processing_time_seconds']:.2f} 秒")
    else:
        logger.info("调试模式关闭，直接退出")
