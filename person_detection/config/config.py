#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
人员外出检测系统配置文件
"""
import os

# 获取包根目录的绝对路径
PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class Config:
    """配置参数类"""
    
    # ==================== ROS配置 ====================
    NODE_NAME = "person_detection_node"
    SWITCH_TOPIC = "/person_detection_switch"
    
    # ==================== 检测参数 ====================
    # YOLO模型配置
    MODEL_PATH = os.path.join(PACKAGE_ROOT, "model", "people_openvino_model")
    YOLO_CONFIDENCE = 0.8
    YOLO_DEVICE = "cpu"  # 可以是 "cpu" 或 "cuda"
    
    # 检测阈值配置
    DETECTION_THRESHOLD = 2  # 连续检测帧数阈值
    DETECTION_INTERVAL = 2.0  # 检测间隔(秒)
    
    # 时间范围配置 (24小时制)
    START_TIME_HOUR = 8
    START_TIME_MINUTE = 0
    END_TIME_HOUR = 20
    END_TIME_MINUTE = 0
    
    # ==================== 豆包API配置 ====================
    DOUBAO_API_KEY = ""  # 替换为实际API密钥
    DOUBAO_API_URL = "https://ark.cn-beijing.volces.com/api/v3/chat/completions"
    DOUBAO_MODEL_NAME = "doubao-seed-1-6-flash-250828"  # 根据实际模型名称调整
    # API请求参数
    API_TIMEOUT = 30
    API_TEMPERATURE = 0.1
    API_MAX_TOKENS = 500
    
    # ==================== 通知配置 ====================
    PHONE_NUMBER = "YOUR_PHONE_NUMBER"  # 替换为实际手机号
    
    # 邮件通知配置（推荐方案）
    EMAIL_ENABLED = True
    EMAIL_SMTP_SERVER = "smtp.qq.com"  # QQ邮箱SMTP服务器
    EMAIL_SMTP_PORT = 587  # TLS端口
    EMAIL_USERNAME = ""  # 发送方邮箱
    EMAIL_PASSWORD = ""  # 邮箱授权码（不是登录密码）
    EMAIL_TO = ""  # 接收方邮箱（可以是同一个）
    EMAIL_SUBJECT = "人员外出检测警告"
    
    # 短信通知配置（备选方案）
    SMS_ENABLED = False  # 默认关闭，需要额外配置
    SMS_API_KEY = "YOUR_SMS_API_KEY"  # 短信服务API密钥
    
    # ==================== 腾讯云语音合成配置 ====================
    # 腾讯云API配置
    TENCENT_SECRET_ID = ""
    TENCENT_SECRET_KEY = ""
    TENCENT_REGION = "ap-guangzhou"  # 腾讯云区域 (广州区域支持粤语)
    
    # 语音合成参数
    TTS_VOICE_TYPE = 102  # 粤语男声 (101: 粤语女声, 102: 粤语男声)
    TTS_SPEED = 0  # 语速 (-2到2，0为正常语速)
    TTS_VOLUME = 0  # 音量 (-10到10，0为正常音量)
    TTS_SAMPLE_RATE = 16000  # 采样率
    
    # 音频文件配置
    AUDIO_CACHE_DIR = os.path.join(PACKAGE_ROOT, "audio_cache")
    AUDIO_FORMAT = "mp3"  # 音频格式
    
    # 备选语音合成配置（espeak）
    ESPEAK_VOICE = "zh+f5"
    ESPEAK_SPEED = 150
    
    # ==================== 图像处理配置 ====================
    # 图像调整大小配置（用于API传输）
    RESIZE_WIDTH = 640
    RESIZE_HEIGHT = 480
    JPEG_QUALITY = 80
    
    # ==================== 显示配置 ====================
    WINDOW_NAME = "人员外出检测"
    DISPLAY_FONT_SCALE = 0.6
    DISPLAY_FONT_THICKNESS = 2
    
    # ==================== 摄像头配置 ====================
    CAMERA_INDEX = 0  # 默认摄像头索引

# 创建配置实例
config = Config()
