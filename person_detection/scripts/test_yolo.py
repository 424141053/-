#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import time
import threading
import os
import sys
import json
import base64
import requests
import subprocess
from datetime import datetime, time as dt_time
from ultralytics import YOLO
import rospy
from std_msgs.msg import Bool

# 导入腾讯云语音合成和音频播放
try:
    from tencentcloud.common import credential
    from tencentcloud.common.profile.client_profile import ClientProfile
    from tencentcloud.common.profile.http_profile import HttpProfile
    from tencentcloud.tts.v20190823 import tts_client, models
    import simpleaudio as sa
    TENCENT_TTS_AVAILABLE = True
    print("✅ 腾讯云SDK和simpleaudio导入成功")
except ImportError as e:
    TENCENT_TTS_AVAILABLE = False
    print(f"⚠️ 腾讯云SDK或simpleaudio导入失败: {e}")
    print("请运行: pip install tencentcloud-sdk-python simpleaudio")

# 添加包路径以便导入config
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'config'))
try:
    from config import config
    print("配置文件导入成功")
except ImportError as e:
    print(f"配置文件导入失败: {e}")
    sys.exit(1)

class YOLOTester:
    def __init__(self):
        """初始化YOLO测试器"""
        # ROS节点初始化
        rospy.init_node('yolo_tester_node', anonymous=True)
        
        # 控制开关
        self.detection_event = threading.Event()
        self.detection_event.clear()  # 默认开启
        self.switch_lock = threading.Lock()
        
        # API调用控制
        self.last_api_call_time = 0
        self.api_call_interval = 5  # 5秒间隔，便于测试
        self.api_lock = threading.Lock()
        
        # 多帧缓存机制
        self.frame_buffer = []  # 缓存最近N帧
        self.max_buffer_size = 5  # 最大缓存帧数
        self.buffer_lock = threading.Lock()
        
        # 语音播报控制
        self.audio_lock = threading.Lock()
        self.audio_threads = []  # 存储活跃的音频线程
        
        # 获取包路径
        self.package_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        print(f"包路径: {self.package_path}")
        
        # 创建音频缓存目录
        self.audio_cache_dir = config.AUDIO_CACHE_DIR
        if not os.path.exists(self.audio_cache_dir):
            os.makedirs(self.audio_cache_dir)
            print(f"创建音频缓存目录: {self.audio_cache_dir}")
        
        # YOLO模型初始化
        self.model = None
        try:
            print(f"尝试加载模型路径: {config.MODEL_PATH}")
            
            # 检查模型路径是否存在
            if not os.path.exists(config.MODEL_PATH):
                print(f"模型路径不存在: {config.MODEL_PATH}")
                print("请检查模型文件是否放置在正确位置")
                return
            
            # 检查模型文件是否存在
            model_files = os.listdir(config.MODEL_PATH)
            print(f"模型目录下的文件: {model_files}")
            
            self.model = YOLO(config.MODEL_PATH)
            print(f"YOLO OpenVINO模型加载成功: {config.MODEL_PATH}")
        except Exception as e:
            print(f"YOLO模型加载失败: {e}")
            import traceback
            print(f"详细错误信息: {traceback.format_exc()}")
            return
        
        # 摄像头初始化
        self.cap = cv2.VideoCapture(config.CAMERA_INDEX)
        if not self.cap.isOpened():
            print(f"错误：无法打开摄像头索引 {config.CAMERA_INDEX}")
            # 尝试其他摄像头索引
            for i in range(1, 3):
                self.cap = cv2.VideoCapture(i)
                if self.cap.isOpened():
                    print(f"成功打开摄像头索引 {i}")
                    break
            else:
                print("所有摄像头尝试均失败")
                return
        
        # 创建显示窗口
        try:
            cv2.namedWindow(config.WINDOW_NAME, cv2.WINDOW_NORMAL)
            print("显示窗口创建成功")
        except Exception as e:
            print(f"显示窗口创建失败: {e}")
        
        # 订阅开关话题
        rospy.Subscriber(config.SWITCH_TOPIC, Bool, self.switch_callback)
        
        print("YOLO测试器初始化完成")
        print(f"开关话题: {config.SWITCH_TOPIC}")
        print(f"检测时间范围: {config.START_TIME_HOUR:02d}:{config.START_TIME_MINUTE:02d} - {config.END_TIME_HOUR:02d}:{config.END_TIME_MINUTE:02d}")
        print(f"豆包API调用间隔: {self.api_call_interval}秒")
        print(f"豆包API密钥状态: {'已配置' if config.DOUBAO_API_KEY != 'YOUR_DOUBAO_API_KEY' else '未配置'}")

    def switch_callback(self, msg):
        """处理开关状态更新"""
        with self.switch_lock:
            old_state = self.detection_event.is_set()
            if msg.data:
                self.detection_event.set()
            else:
                self.detection_event.clear()
            
            if old_state != self.detection_event.is_set():
                status = "开启" if msg.data else "关闭"
                print(f"ROS话题控制 - YOLO检测: {status}")

    def is_time_in_range(self):
        """检查当前时间是否在配置的时间范围内"""
        now = datetime.now().time()
        start_time = dt_time(config.START_TIME_HOUR, config.START_TIME_MINUTE, 0)
        end_time = dt_time(config.END_TIME_HOUR, config.END_TIME_MINUTE, 0)
        return start_time <= now <= end_time

    def image_to_base64(self, image_frame):
        """将图像转换为base64编码"""
        try:
            # 调整图像大小以减少传输数据量
            resized_frame = cv2.resize(image_frame, (config.RESIZE_WIDTH, config.RESIZE_HEIGHT))
            _, buffer = cv2.imencode('.jpg', resized_frame, [cv2.IMWRITE_JPEG_QUALITY, config.JPEG_QUALITY])
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            return img_base64
        except Exception as e:
            print(f"图像编码失败: {e}")
            return None

    def add_frame_to_buffer(self, image_frame):
        """将检测到人的帧添加到缓存中"""
        with self.buffer_lock:
            current_time = time.time()
            cache_cleared = False
            
            # 检查时间间隔：如果缓存不为空且新帧与前一帧间隔超过5秒，清空缓存
            if len(self.frame_buffer) > 0:
                last_frame_time = self.frame_buffer[-1]['timestamp']
                time_diff = current_time - last_frame_time
                
                if time_diff > 5.0:  # 超过5秒
                    print(f"⏰ 新帧与前一帧间隔 {time_diff:.1f} 秒，超过5秒阈值，清空缓存重新开始")
                    self.frame_buffer.clear()
                    cache_cleared = True
            
            # 将图像转换为base64并添加到缓存
            img_base64 = self.image_to_base64(image_frame)
            if img_base64:
                self.frame_buffer.append({
                    'timestamp': current_time,
                    'image_base64': img_base64
                })
                
                # 保持缓存大小不超过最大值
                if len(self.frame_buffer) > self.max_buffer_size:
                    self.frame_buffer.pop(0)  # 移除最旧的帧
                
                print(f"📸 帧已添加到缓存，当前缓存帧数: {len(self.frame_buffer)}")
                
                # 如果缓存被清空，立即触发下一次推理循环
                if cache_cleared:
                    print("🔄 缓存已清空，立即触发下一次推理循环")
                    return True  # 返回True表示需要立即进行下一次推理
        
        return False  # 返回False表示正常添加帧

    def get_buffer_frames(self):
        """获取当前缓存的所有帧"""
        with self.buffer_lock:
            return self.frame_buffer.copy()

    def clear_frame_buffer(self):
        """清空帧缓存"""
        with self.buffer_lock:
            self.frame_buffer.clear()
            print("🗑️ 帧缓存已清空")

    def save_frames_to_folder(self, frame_buffer):
        """将帧缓存中的图片保存到新文件夹"""
        try:
            # 创建基于时间戳的文件夹名称
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            folder_name = f"api_images_{timestamp}"
            folder_path = os.path.join(os.getcwd(), folder_name)
            
            # 创建文件夹
            os.makedirs(folder_path, exist_ok=True)
            print(f"📁 创建图片保存文件夹: {folder_path}")
            
            # 保存每帧图片
            for i, frame_data in enumerate(frame_buffer):
                # 解码base64图像
                img_data = base64.b64decode(frame_data['image_base64'])
                
                # 保存图片
                img_filename = f"frame_{i+1:02d}_{frame_data['timestamp']:.3f}.jpg"
                img_path = os.path.join(folder_path, img_filename)
                
                with open(img_path, 'wb') as f:
                    f.write(img_data)
                
                print(f"💾 保存图片: {img_filename}")
            
            # 保存帧信息到JSON文件
            frame_info = {
                "timestamp": timestamp,
                "folder_path": folder_path,
                "frame_count": len(frame_buffer),
                "frames": []
            }
            
            for i, frame_data in enumerate(frame_buffer):
                frame_info["frames"].append({
                    "index": i + 1,
                    "timestamp": frame_data['timestamp'],
                    "filename": f"frame_{i+1:02d}_{frame_data['timestamp']:.3f}.jpg"
                })
            
            info_file = os.path.join(folder_path, "frame_info.json")
            with open(info_file, 'w', encoding='utf-8') as f:
                json.dump(frame_info, f, ensure_ascii=False, indent=2)
            
            print(f"📄 保存帧信息文件: frame_info.json")
            print(f"✅ 成功保存 {len(frame_buffer)} 帧图片到文件夹: {folder_path}")
            
            return folder_path
            
        except Exception as e:
            print(f"❌ 保存图片失败: {e}")
            return None

    def call_doubao_api_multi_frame(self, frame_buffer):
        """调用豆包API进行多帧图像分析"""
        # 检查API密钥是否配置
        if config.DOUBAO_API_KEY == "YOUR_DOUBAO_API_KEY":
            print("豆包API密钥未配置，跳过API调用")
            return None
            
        try:
            # 检查是否有足够的帧进行分析
            if len(frame_buffer) < self.max_buffer_size:
                print(f"帧缓存不足，当前只有 {len(frame_buffer)} 帧，需要{self.max_buffer_size}帧")
                return None
            
            # 保存图片到文件夹
            saved_folder = self.save_frames_to_folder(frame_buffer)
            if saved_folder:
                print(f"📸 图片已保存到: {saved_folder}")
            
            # 构建请求头
            headers = {
                "Authorization": f"Bearer {config.DOUBAO_API_KEY}",
                "Content-Type": "application/json"
            }
            
            # 构建系统提示词
            system_prompt = """你是一位专业的视频分析智能体。你的任务是根据提供的连续多帧图像准确判断是否有人员正在出门（离开房间），此应用场景为医院、养老院或家庭，用于监控是否有人正在离开房间。当前在判断进出门情况时容易出现误判，尽管已经告知你摄像头视角，但判断仍不准确。同时，图像的处理逻辑是当yolo在当前帧检测到人时就将其加入缓存队列，当缓存队列超过5帧就调用一次api来进行判断，存在卡顿情况。

## 重要说明:
- 你将收到连续的多帧图像，这些图像按时间顺序排列
- 摄像头视角：你当前位于房间内部，摄像头安装在房间内，对着大门/门口进行监控
- 你的视角是从房间内部向外看，能够看到门、门把手、门框等室内设施
- 请综合分析多帧图像中的动作、位置变化和时序信息
- 必须看到明确的推门出门动作才能判定为出门，不能仅凭趋势或意图判断
- 严格区分进门和出门：只关注人员从房间内走向房间外的出门行为
- 人物朝向判断：在确定进出门行为后，通过人物朝向进行二次验证
- 根据分析结果，在结构化输出中正确填写"是否有人员出门"和"是否需要调用工具"字段
- 如果检测到人员出门，在"调用工具列表"中列出需要调用的工具
- 如果没有检测到人员出门，"调用工具列表"应为空数组"""

            # 构建用户提示词
            user_prompt = f"""请分析这 {len(frame_buffer)} 张连续图像中是否有人员正在出门（离开房间）。

## 图像说明:
- 这些图像是连续拍摄的视频帧，按时间顺序排列
- 摄像头位置：房间内部，正对着大门/门口进行监控
- 你的视角是从房间内部向外看，能够看到门、门把手、门框等室内设施
- 人物可能在不同帧中出现在不同位置
- 请综合动作和时序信息进行判断

## 判定标准:
- 人员正在出门: 综合多帧分析，必须同时满足以下所有条件：
  * 门的状态发生明显变化：门把手被转动、门被推开、门保持开启状态
  * 人物实际踏出门槛：人物从房间内移动到房间外，跨越门的位置
  * 时序连贯性：门开启和人物踏出的动作在时间上有逻辑顺序
  * 明确的出门动作：不是仅仅靠近门或经过门，而是真正走出房间
  * 必须看到推门动作：人物必须主动推开门或拉开门把手
  * 必须看到跨出门槛：人物必须完全踏出门外，不能只是站在门口
  * 方向确认：人物必须是从房间内走向房间外，不是从房间外进入房间内
  * 完整序列：必须看到"靠近门→推门→踏出门槛→人不在室内"的完整序列
  * 位置验证：人物在最后一帧中应该完全消失在门外，不再出现在室内画面中
  * 人物朝向验证：在确定出门行为后，人物身体朝向应面向门外方向（背对室内/摄像头），这是出门的辅助判断依据

- 非出门情况: 以下情况都不算出门：
  * 有人但门保持关闭状态
  * 有人靠近门但没有开门
  * 有人开门但没有踏出门槛
  * 有人从门外进入房间（这是进门，不是出门）
  * 有人在房间内正常活动（坐着、站着、工作、休息等）
  * 门开着但没有人踏出
  * 有人只是走向门但没有推门动作
  * 有人只是站在门口但没有跨出门槛
  * 门是开着的但人物没有出去门口的动作
  * 人物从房间外进入房间内（进门行为）
  * 人物从门外出现并进入房间（进门行为）
  * 人物在门口徘徊但没有真正踏出门外
  * 人物推开门但没有完全踏出门槛
  * 人物在最后一帧中仍然出现在室内画面中
  * 人物从画面边缘进入（可能是进门）
  * 人物在门附近活动但没有明确的出门动作
  * 人物朝向验证：如果已确定是进门行为，人物身体朝向应面向室内/摄像头方向，这是进门的辅助判断依据

## 特别关注点:
- 门的状态变化：门把手是否被触碰、门是否被推开、门是否保持开启
- 人物位置变化：是否从房间内移动到房间外，跨越门的位置
- 动作时序：开门动作和踏出动作的时间顺序
- 视角确认：你当前在房间内部，摄像头对着大门，人物是向外走而不是向内走
- 推门动作：必须看到人物主动推门或拉门把手的动作
- 跨门槛动作：必须看到人物完全踏出门外，不能只是站在门口
- 方向判断：严格区分进门和出门，只关注出门行为
- 室内视角：你看到的是房间内部环境，门、门把手、门框等室内设施
- 完整序列验证：必须看到完整的出门序列，不能缺少任何关键步骤
- 进门行为识别：特别注意人物从门外出现或从画面边缘进入的情况
- 人物朝向分析：在确定进出门行为后，通过人物身体朝向进行二次验证
  * 出门行为：人物身体朝向应面向门外方向（背对室内/摄像头）
  * 进门行为：人物身体朝向应面向室内/摄像头方向
  * 朝向一致性：在多帧中人物朝向应保持一致，不应出现朝向变化
- 严格判定：宁可误判为"未出门"也不要误判为"出门"

## 输出要求:
- 如果检测到人员正在出门: "是否有人员出门"填"是"，"是否需要调用工具"填"是"，"调用工具列表"包含play_audio和send_sms
- 如果没有检测到人员出门: "是否有人员出门"填"否"，"是否需要调用工具"填"否"，"调用工具列表"为空数组

请仔细分析多帧图像内容，特别关注门的状态变化和人物是否真正踏出门槛。必须看到明确的推门出门动作才能判定为出门，不要仅凭趋势或意图判断。严格区分进门和出门，只关注出门行为。记住你当前在房间内部，摄像头对着大门，看到的是室内环境。

## 关键验证步骤:
1. 检查人物是否从房间内开始（不是从门外出现）
2. 验证是否看到完整的出门序列：靠近门→推门→踏出门槛→人不在室内
3. 确认人物在最后一帧中不在室内
4. 排除任何从门外出现或从画面边缘进入的情况
5. 人物朝向二次验证：在确定进出门行为后，通过人物身体朝向进行辅助判断
   * 如果判断为出门：人物身体朝向应面向门外方向（背对室内/摄像头）
   * 如果判断为进门：人物身体朝向应面向室内/摄像头方向
   * 朝向一致性：在多帧中人物朝向应保持一致
6. 如果任何一步不满足，则判定为"未出门"

按照结构化输出格式返回结果。"""
            
            
            # 定义结构化输出schema
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": "person_detection_analysis",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "是否有人员出门": {
                                "type": "string",
                                "enum": ["是", "否"],
                                "description": "是否检测到人员正在出门（离开房间）"
                            },
                            "是否需要调用工具": {
                                "type": "string",
                                "enum": ["是", "否"],
                                "description": "是否需要调用工具（play_audio和send_sms）"
                            },
                            "调用工具列表": {
                                "type": "array",
                                "description": "需要调用的工具列表",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "工具名称": {
                                            "type": "string",
                                            "enum": ["play_audio", "send_sms"],
                                            "description": "工具名称"
                                        },
                                        "参数": {
                                            "type": "object",
                                            "properties": {
                                                "message": {
                                                    "type": "string",
                                                    "description": "消息内容"
                                                }
                                            },
                                            "required": ["message"]
                                        }
                                    },
                                    "required": ["工具名称", "参数"]
                                }
                            }
                        },
                        "required": ["是否有人员出门", "是否需要调用工具", "调用工具列表"]
                    }
                }
            }
            
            # 构建多帧图像内容
            content_items = []
            
            # 添加所有帧的图像
            for i, frame_data in enumerate(frame_buffer):
                content_items.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{frame_data['image_base64']}"
                    }
                })
            
            # 添加文本提示
            content_items.append({
                "type": "text",
                "text": user_prompt
            })
            
            # 构建请求数据
            payload = {
                "model": config.DOUBAO_MODEL_NAME,
                "messages": [
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": content_items
                    }
                ],
                "response_format": response_format,
                "temperature": config.API_TEMPERATURE,
                "max_tokens": config.API_MAX_TOKENS
            }
            
            # 发送请求
            print(f"调用豆包API进行多帧图像分析... (共{len(frame_buffer)}帧)")
            response = requests.post(config.DOUBAO_API_URL, headers=headers, json=payload, timeout=config.API_TIMEOUT)
            response.raise_for_status()
            
            # 解析响应
            result = response.json()
            print(f"豆包API多帧分析原始响应数据: {json.dumps(result, ensure_ascii=False, indent=2)}")
            return result
                
        except requests.exceptions.RequestException as e:
            print(f"API请求失败: {e}")
            return None
        except Exception as e:
            print(f"调用豆包API异常: {e}")
            return None

    def create_tencent_tts_client(self):
        """创建腾讯云TTS客户端"""
        try:
            # 创建认证对象
            cred = credential.Credential(
                config.TENCENT_SECRET_ID, 
                config.TENCENT_SECRET_KEY
            )
            
            # 实例化一个http选项
            httpProfile = HttpProfile()
            httpProfile.endpoint = "tts.tencentcloudapi.com"
            
            # 实例化一个client选项
            clientProfile = ClientProfile()
            clientProfile.httpProfile = httpProfile
            
            # 实例化要请求产品的client对象
            client = tts_client.TtsClient(cred, config.TENCENT_REGION, clientProfile)
            return client
        except Exception as e:
            print(f"❌ 创建腾讯云客户端失败: {e}")
            return None

    def text_to_speech_tencent(self, text, audio_file):
        """使用腾讯云进行语音合成"""
        client = self.create_tencent_tts_client()
        if not client:
            return False
        
        try:
            # 实例化一个请求对象
            req = models.TextToVoiceRequest()
            
            # 设置参数
            params = {
                "Text": text,
                "SessionId": "person_detection_session",
                "VoiceType": config.TTS_VOICE_TYPE,  # 粤语男声
                "Codec": "mp3",
                "Speed": config.TTS_SPEED,
                "Volume": config.TTS_VOLUME,
                "SampleRate": config.TTS_SAMPLE_RATE
            }
            req.from_json_string(str(params).replace("'", '"'))
            
            # 调用接口
            resp = client.TextToVoice(req)
            
            # 保存音频文件 - 腾讯云返回的是base64编码的字符串
            import base64
            audio_data = base64.b64decode(resp.Audio)
            with open(audio_file, 'wb') as f:
                f.write(audio_data)
                
                return True
        except Exception as e:
            print(f"❌ 腾讯云TTS失败: {e}")
            return False

    def play_audio_file(self, audio_file):
        """播放音频文件"""
        try:
            # 检查文件扩展名
            if audio_file.endswith('.mp3'):
                # 使用pydub播放MP3文件
                from pydub import AudioSegment
                from pydub.playback import play
                
                audio = AudioSegment.from_mp3(audio_file)
                play(audio)
            else:
                # 使用simpleaudio播放WAV文件
                wave_obj = sa.WaveObject.from_wave_file(audio_file)
                play_obj = wave_obj.play()
                play_obj.wait_done()
            return True
        except Exception as e:
            print(f"❌ 播放音频失败: {e}")
            return False

    def play_audio(self, text):
        """播放粤语语音通知（使用腾讯云）"""
        def audio_thread():
            try:
                with self.audio_lock:
                    print(f"🔊 开始播放语音: {text}")
                
                # 检查腾讯云TTS是否可用
                if not TENCENT_TTS_AVAILABLE:
                    print("⚠️ 腾讯云TTS不可用，回退到文本输出")
                    print(f"🔊 语音警告: {text}")
                    return
                
                # 检查API密钥配置
                if config.TENCENT_SECRET_ID == "your_secret_id_here":
                    print("⚠️ 腾讯云API密钥未配置，回退到文本输出")
                    print(f"🔊 语音警告: {text}")
                    return
                
                # 生成音频文件名（基于文本内容的哈希）
                import hashlib
                text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
                audio_file = os.path.join(self.audio_cache_dir, f"{text_hash}.mp3")
                
                # 如果音频文件不存在，生成新的
                if not os.path.exists(audio_file):
                    print("🎵 生成新的粤语语音文件...")
                    if not self.text_to_speech_tencent(text, audio_file):
                        print(f"🔊 文本输出: {text}")
                        return
                    print(f"✅ 粤语语音文件已保存: {audio_file}")
                else:
                    print(f"♻️ 使用缓存的语音文件: {audio_file}")
                
                # 播放音频文件
                print("🔊 开始播放粤语音频...")
                if self.play_audio_file(audio_file):
                    with self.audio_lock:
                        print(f"✅ 粤语语音播报完成: {text}")
                else:
                    print(f"🔊 文本输出: {text}")
                    
            except Exception as e:
                print(f"语音播放异常: {e}")
                print(f"🔊 文本输出: {text}")
            finally:
                # 从活跃线程列表中移除
                with self.audio_lock:
                    if threading.current_thread() in self.audio_threads:
                        self.audio_threads.remove(threading.current_thread())
        
        # 创建并启动音频线程
        audio_thread_obj = threading.Thread(target=audio_thread)
        audio_thread_obj.daemon = True
        
        with self.audio_lock:
            self.audio_threads.append(audio_thread_obj)
        
        audio_thread_obj.start()
        print(f"🔊 语音播报线程已启动: {text}")

    def send_email(self, message):
        """发送邮件通知"""
        try:
            # 检查邮箱功能是否启用
            if not config.EMAIL_ENABLED:
                print("邮箱功能未启用，跳过邮件发送")
                return
                
            # 检查邮箱配置是否完整
            if (config.EMAIL_USERNAME == "YOUR_EMAIL@qq.com" or 
                config.EMAIL_PASSWORD == "YOUR_EMAIL_PASSWORD"):
                print("邮箱配置不完整，跳过邮件发送")
                return
            
            import smtplib
            from email.mime.text import MIMEText
            from email.mime.multipart import MIMEMultipart
            from email.header import Header
            
            # 创建邮件对象
            msg = MIMEMultipart()
            msg['From'] = config.EMAIL_USERNAME
            msg['To'] = config.EMAIL_TO
            msg['Subject'] = Header(config.EMAIL_SUBJECT, 'utf-8')
            
            # 构建邮件内容
            email_body = f"""
人员出门检测警告

检测时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
警告信息: {message}

此邮件由人员出门检测系统自动发送。
请及时关注相关人员的安全状况。

---
系统信息:
- 检测系统: YOLO + 豆包AI
- 发送时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
            
            # 添加邮件正文
            msg.attach(MIMEText(email_body, 'plain', 'utf-8'))
            
            # 连接SMTP服务器并发送邮件
            print(f"正在发送邮件到 {config.EMAIL_TO}...")
            server = smtplib.SMTP(config.EMAIL_SMTP_SERVER, config.EMAIL_SMTP_PORT)
            server.starttls()  # 启用TLS加密
            server.login(config.EMAIL_USERNAME, config.EMAIL_PASSWORD)
            
            # 发送邮件
            text = msg.as_string()
            server.sendmail(config.EMAIL_USERNAME, config.EMAIL_TO, text)
            server.quit()
            
            print(f"✅ 邮件发送成功到 {config.EMAIL_TO}")
            
        except Exception as e:
            print(f"邮件发送失败: {e}")
            import traceback
            print(f"详细错误信息: {traceback.format_exc()}")

    def send_sms(self, message):
        """发送短信通知（同时发送邮件）"""
        try:
            # 检查手机号是否配置
            if config.PHONE_NUMBER == "YOUR_PHONE_NUMBER":
                print("手机号未配置，跳过短信发送")
            else:
                print(f"发送短信到 {config.PHONE_NUMBER}: {message}")
                
                # 模拟短信发送（实际使用时需要替换为真实的短信API）
                # 例如使用twilio:
                # from twilio.rest import Client
                # client = Client(account_sid, auth_token)
                # message = client.messages.create(
                #     body=message,
                #     from_='YOUR_TWILIO_NUMBER',
                #     to=config.PHONE_NUMBER
                # )
            
            # 同时发送邮件通知
            self.send_email(message)
            
        except Exception as e:
            print(f"短信发送失败: {e}")

    def process_doubao_response(self, response):
        """处理豆包API的响应结果"""
        if not response:
            print("API响应为空")
            return
        
        try:
            # 获取消息内容
            message = response["choices"][0]["message"]
            
            # 打印推理内容（如果有的话）
            if "reasoning_content" in message:
                print(f"豆包推理过程: {message['reasoning_content']}")
            
            # 优先处理结构化输出
            structured_output = self.parse_structured_output(message)
            if structured_output:
                self.process_structured_output(structured_output)
                return
            
            # 回退到原有的工具调用逻辑
            print("⚠️ 未找到结构化输出，回退到工具调用逻辑")
            self.process_tool_calls_fallback(message)
                
        except Exception as e:
            print(f"解析API响应失败: {e}")
            print(f"响应内容: {response}")

    def parse_structured_output(self, message):
        """解析结构化输出"""
        try:
            # 检查是否有content字段且包含JSON
            content = message.get("content", "")
            if not content:
                return None
            
            # 尝试解析JSON
            import re
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                structured_data = json.loads(json_str)
                
                # 验证必需字段
                required_fields = ["是否有人员出门", "是否需要调用工具", "调用工具列表"]
                if all(field in structured_data for field in required_fields):
                    print("✅ 成功解析结构化输出")
                    return structured_data
                else:
                    print("⚠️ 结构化输出缺少必需字段")
                    return None
            else:
                return None
                
        except json.JSONDecodeError as e:
            print(f"结构化输出JSON解析失败: {e}")
            return None
        except Exception as e:
            print(f"解析结构化输出异常: {e}")
            return None

    def process_structured_output(self, structured_data):
        """处理结构化输出结果"""
        try:
            has_person_leaving = structured_data.get("是否有人员出门", "否")
            need_tools = structured_data.get("是否需要调用工具", "否")
            tool_list = structured_data.get("调用工具列表", [])
            
            print(f"📊 结构化分析结果:")
            print(f"   - 是否有人员出门: {has_person_leaving}")
            print(f"   - 是否需要调用工具: {need_tools}")
            
            # 检查逻辑一致性
            if has_person_leaving == "是" and need_tools == "否":
                print("⚠️ 检测到人员出门但不需要调用工具，可能存在逻辑不一致")
            elif has_person_leaving == "否" and need_tools == "是":
                print("⚠️ 未检测到人员出门但需要调用工具，可能存在逻辑不一致")
            
            # 核心判断逻辑：只有当"是否有人员出门"和"是否需要调用工具"都为"是"时才执行工具调用
            if has_person_leaving == "是" and need_tools == "是":
                print(f"🔔 检测到人员出门且需要调用工具，执行 {len(tool_list)} 个工具")
                self.execute_tools_from_list(tool_list)
            elif has_person_leaving == "是" and need_tools == "否":
                print("✅ 检测到人员出门但不需要调用工具，跳过工具执行")
            elif has_person_leaving == "否" and need_tools == "否":
                print("✅ 未检测到人员出门且不需要调用工具，确认无出门情况")
            else:
                print("⚠️ 逻辑异常：未检测到人员出门但需要调用工具，跳过工具执行")
                
        except Exception as e:
            print(f"处理结构化输出失败: {e}")

    def execute_tools_from_list(self, tool_list):
        """从工具列表中执行工具"""
        executed_tools = []
        
        for tool_info in tool_list:
            try:
                tool_name = tool_info.get("工具名称", "")
                tool_params = tool_info.get("参数", {})
                
                if tool_name == "play_audio":
                    message = tool_params.get("message", "检测到人员出门")
                    self.play_audio(message)
                    executed_tools.append("play_audio")
                elif tool_name == "send_sms":
                    message = tool_params.get("message", "检测到人员出门")
                    self.send_sms(message)
                    executed_tools.append("send_sms")
                else:
                    print(f"⚠️ 未知工具名称: {tool_name}")
                    
            except Exception as e:
                print(f"执行工具 {tool_name} 失败: {e}")
        
        if executed_tools:
            print(f"🔔 已执行工具: {', '.join(executed_tools)}")

    def process_tool_calls_fallback(self, message):
        """回退处理工具调用（原有逻辑）"""
        # 检查是否有工具调用
        if message.get("tool_calls"):
            tool_calls = message["tool_calls"]
            print(f"检测到工具调用: {len(tool_calls)} 个工具")
            
            # 记录调用的工具列表
            called_tools = []
            valid_tool_calls = 0
            
            # 处理每个工具调用
            for tool_call in tool_calls:
                function_name = tool_call["function"]["name"]
                arguments_str = tool_call["function"]["arguments"]
                
                # 检查工具名称和参数是否为空（豆包API bug：有时会返回空的工具调用）
                if not function_name or not arguments_str:
                    print(f"忽略无效工具调用 - 名称: '{function_name}', 参数: '{arguments_str}'")
                    continue
                
                try:
                    arguments = json.loads(arguments_str)
                    print(f"调用工具: {function_name}, 参数: {arguments}")
                    called_tools.append(function_name)
                    valid_tool_calls += 1
                    
                    # 根据工具名称调用相应函数
                    if function_name == "play_audio":
                        self.play_audio(arguments.get('message', '检测到人员出门'))
                    elif function_name == "send_sms":
                        self.send_sms(arguments.get('message', '检测到人员出门'))
                except json.JSONDecodeError as e:
                    print(f"工具调用参数解析失败: {e}, 参数内容: '{arguments_str}'")
                    continue
            
            # 记录工具调用总结
            if valid_tool_calls == 0:
                print("✅ 所有工具调用都无效，确认未检测到人员出门")
            elif valid_tool_calls == 1:
                print(f"⚠️ 模型只调用了单个工具: {called_tools[0]}，不符合要求（需要同时调用两个工具）")
            elif valid_tool_calls == 2:
                # 检查是否同时调用了两个必需的工具
                required_tools = {"play_audio", "send_sms"}
                called_tools_set = set(called_tools)
                if called_tools_set == required_tools:
                    print(f"🔔 模型正确同时调用两个工具: {', '.join(called_tools)}")
                else:
                    print(f"⚠️ 模型调用了两个工具但不符合要求: {', '.join(called_tools)}，需要同时调用play_audio和send_sms")
            else:
                print(f"⚠️ 模型调用了 {valid_tool_calls} 个工具，不符合要求（需要同时调用两个工具）")
                    
        else:
            # 没有工具调用，说明未检测到人员出门
            content = message.get("content", "")
            print(f"✅ 未检测到人员出门: {content}")

    def call_doubao_api_async_multi_frame(self):
        """异步调用豆包API进行多帧分析（非阻塞）"""
        current_time = time.time()
        
        # 检查是否满足调用间隔
        with self.api_lock:
            if current_time - self.last_api_call_time < self.api_call_interval:
                remaining_time = self.api_call_interval - (current_time - self.last_api_call_time)
                print(f"API调用间隔未到，还需等待 {remaining_time:.1f} 秒")
                return
            
            # 更新最后调用时间
            self.last_api_call_time = current_time
        
        # 获取当前帧缓存
        frame_buffer = self.get_buffer_frames()
        if len(frame_buffer) < self.max_buffer_size:
            print(f"帧缓存不足，当前只有 {len(frame_buffer)} 帧，跳过API调用")
            return
        
        # 立即清空原始缓存，让系统开始收集新的帧
        self.clear_frame_buffer()
        print("🗑️ 已获取帧缓存副本，原始缓存已清空，开始收集新帧")
        
        # 在单独线程中执行API调用
        def api_thread():
            try:
                print(f"开始异步调用豆包API进行多帧分析... (共{len(frame_buffer)}帧)")
                response = self.call_doubao_api_multi_frame(frame_buffer)
                self.process_doubao_response(response)
                print("豆包API多帧分析调用完成")
            except Exception as e:
                print(f"异步多帧API调用失败: {e}")
        
        # 启动线程
        thread = threading.Thread(target=api_thread)
        thread.daemon = True
        thread.start()
        print("豆包API多帧分析调用线程已启动")

    def yolo_inference(self):
        """YOLO推理主循环"""
        print("开始YOLO推理循环")
        
        # 检查摄像头状态
        if not self.cap.isOpened():
            print("摄像头未正确初始化，无法开始推理循环")
            return
        
        frame_count = 0
        last_frame_time = time.time()
        
        while not rospy.is_shutdown():
            try:
                # 检查时间和开关状态
                time_ok = self.is_time_in_range()
                switch_ok = self.detection_event.is_set()
                
                # 如果条件不满足，等待
                if not (time_ok and switch_ok):
                    if not time_ok:
                        print(f"当前时间不在检测范围内 ({config.START_TIME_HOUR:02d}:{config.START_TIME_MINUTE:02d}-{config.END_TIME_HOUR:02d}:{config.END_TIME_MINUTE:02d})，等待1秒...")
                    elif not switch_ok:
                        print("检测已关闭，等待1秒...")
                    time.sleep(1)
                    continue
                
                # 读取帧
                ret, frame = self.cap.read()
                if not ret:
                    print("无法读取摄像头帧")
                    # 检查摄像头是否仍然可用
                    if not self.cap.isOpened():
                        print("摄像头连接已断开，尝试重新连接...")
                        self.cap = cv2.VideoCapture(config.CAMERA_INDEX)
                        if not self.cap.isOpened():
                            print("摄像头重连失败")
                            time.sleep(5)
                            continue
                        else:
                            print("摄像头重连成功")
                    else:
                        time.sleep(0.1)
                    continue
                
                # 更新帧计数和时间
                frame_count += 1
                current_time = time.time()
                
                # 每10帧输出一次状态信息
                if frame_count % 10 == 0:
                    fps = 10 / (current_time - last_frame_time)
                    print(f"推理状态 - 帧数: {frame_count}, FPS: {fps:.2f}, 开关: {switch_ok}, 时间: {time_ok}")
                    last_frame_time = current_time
                
                # YOLO推理
                results = self.model(frame, device=config.YOLO_DEVICE, conf=config.YOLO_CONFIDENCE,verbose=False)
                result = results[0]
                
                # 检查是否检测到人
                person_detected = False
                person_count = 0
                if result.boxes is not None:
                    for box in result.boxes:
                        cls_id = int(box.cls.item())
                        if cls_id == 1:  # person类ID为0
                            person_detected = True
                            person_count += 1
                
                # 如果检测到人，将帧添加到缓存并检查是否需要调用API
                if person_detected:
                    print(f"✅ 检测到 {person_count} 个人，添加到帧缓存")
                    cache_cleared = self.add_frame_to_buffer(frame)
                    
                    # 如果缓存被清空，立即触发下一次推理循环
                    if cache_cleared:
                        print("🔄 缓存已清空，立即触发下一次推理循环")
                        continue  # 立即进入下一次循环
                    
                    # 检查缓存是否足够进行多帧分析
                    current_buffer = self.get_buffer_frames()
                    if len(current_buffer) >= self.max_buffer_size:  # 缓存满了才开始分析
                        print(f"📊 帧缓存已满 ({len(current_buffer)}帧)，准备进行多帧分析")
                        self.call_doubao_api_async_multi_frame()
                else:
                    print("❌ 未检测到人员")
                    # 如果连续多帧没有检测到人，清空缓存
                    current_buffer = self.get_buffer_frames()
                    if len(current_buffer) > 0:
                        print("🔄 连续未检测到人员，清空帧缓存")
                        self.clear_frame_buffer()
                
                # 显示结果
                try:
                    annotated_frame = result.plot()
                    
                    # 添加状态信息
                    status_text = f"Detection: {'ON' if switch_ok else 'OFF'}, Time: {'OK' if time_ok else 'OUT'}"
                    cv2.putText(annotated_frame, status_text, (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, config.DISPLAY_FONT_SCALE, (0, 255, 0), config.DISPLAY_FONT_THICKNESS)
                    cv2.putText(annotated_frame, f"ROS Topic: {config.SWITCH_TOPIC}", 
                               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, config.DISPLAY_FONT_SCALE, (0, 255, 0), config.DISPLAY_FONT_THICKNESS)
                    cv2.putText(annotated_frame, f"Time Range: {config.START_TIME_HOUR:02d}:{config.START_TIME_MINUTE:02d}-{config.END_TIME_HOUR:02d}:{config.END_TIME_MINUTE:02d}", 
                               (10, 90), cv2.FONT_HERSHEY_SIMPLEX, config.DISPLAY_FONT_SCALE, (0, 255, 0), config.DISPLAY_FONT_THICKNESS)
                    cv2.putText(annotated_frame, f"Person Count: {person_count}", 
                               (10, 120), cv2.FONT_HERSHEY_SIMPLEX, config.DISPLAY_FONT_SCALE, (0, 255, 0), config.DISPLAY_FONT_THICKNESS)
                    cv2.putText(annotated_frame, f"Frame: {frame_count}", 
                               (10, 150), cv2.FONT_HERSHEY_SIMPLEX, config.DISPLAY_FONT_SCALE, (0, 255, 0), config.DISPLAY_FONT_THICKNESS)
                    
                    # 显示帧缓存状态
                    buffer_size = len(self.get_buffer_frames())
                    buffer_status = f"Buffer: {buffer_size}/{self.max_buffer_size}"
                    buffer_color = (0, 255, 0) if buffer_size >= self.max_buffer_size else (255, 255, 0)  # 绿色表示准备分析，黄色表示缓存中
                    cv2.putText(annotated_frame, buffer_status, 
                               (10, 180), cv2.FONT_HERSHEY_SIMPLEX, config.DISPLAY_FONT_SCALE, buffer_color, config.DISPLAY_FONT_THICKNESS)
                    
                    # 显示API调用状态
                    time_since_last_api = time.time() - self.last_api_call_time
                    if time_since_last_api < self.api_call_interval:
                        remaining_time = self.api_call_interval - time_since_last_api
                        api_status = f"API Cooldown: {remaining_time:.1f}s"
                        color = (0, 255, 255)  # 黄色
                    else:
                        api_status = "API Ready"
                        color = (0, 255, 0)  # 绿色
                    cv2.putText(annotated_frame, api_status, 
                               (10, 210), cv2.FONT_HERSHEY_SIMPLEX, config.DISPLAY_FONT_SCALE, color, config.DISPLAY_FONT_THICKNESS)
                    
                    # 显示时间
                    time_str = time.strftime("%Y-%m-%d %H:%M:%S")
                    cv2.putText(annotated_frame, time_str, (10, annotated_frame.shape[0] - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                    # 添加控制提示
                    cv2.putText(annotated_frame, "Press 's' to pause/resume, 'q' to quit", 
                               (10, annotated_frame.shape[0] - 40), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                    cv2.putText(annotated_frame, f"ROS Control: rostopic pub {config.SWITCH_TOPIC} std_msgs/Bool", 
                               (10, annotated_frame.shape[0] - 70), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                    
                    cv2.imshow(config.WINDOW_NAME, annotated_frame)
                    
                    # 处理按键
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("用户退出")
                        rospy.signal_shutdown("用户退出")
                        break
                    elif key == ord('s'):
                        if self.detection_event.is_set():
                            self.detection_event.clear()
                            print("检测已暂停")
                        else:
                            self.detection_event.set()
                            print("检测已恢复")
                            
                except Exception as e:
                    print(f"显示图像失败: {e}")
                
                # 控制循环频率
                time.sleep(0.1)
                
            except Exception as e:
                print(f"推理循环出错: {e}")
                import traceback
                print(f"详细错误信息: {traceback.format_exc()}")
                time.sleep(1)
                # 如果出现严重错误，检查ROS是否还在运行
                if rospy.is_shutdown():
                    print("ROS已关闭，退出推理循环")
                    break

    def start_detection(self):
        """启动检测"""
        self.detection_event.set()
        print("检测已启动")

    def stop_detection(self):
        """停止检测"""
        self.detection_event.clear()
        print("检测已停止")
    
    def cleanup_audio_threads(self):
        """清理音频线程"""
        with self.audio_lock:
            if self.audio_threads:
                print(f"等待 {len(self.audio_threads)} 个音频线程完成...")
                for thread in self.audio_threads[:]:  # 创建副本避免修改列表时出错
                    if thread.is_alive():
                        thread.join(timeout=2)  # 等待最多2秒
                        if thread.is_alive():
                            print(f"音频线程 {thread.name} 仍在运行，强制结束")
                self.audio_threads.clear()
                print("音频线程清理完成")

    def run(self):
        """运行测试器"""
        try:
            print("YOLO测试器启动成功")
            print("控制说明:")
            print("- 按 's' 键暂停/继续检测")
            print("- 按 'q' 键退出程序")
            print("- 检测状态会在窗口上显示")
            print(f"- ROS话题控制: rostopic pub {config.SWITCH_TOPIC} std_msgs/Bool \"data: true/false\"")
            print("- 多帧融合分析: 缓存最近5帧，当缓存满5帧时进行多帧时序分析")
            print("- 检测到人员时会自动调用豆包API进行多帧分析")
            print(f"- API调用间隔: {self.api_call_interval}秒")
            print(f"- 帧缓存大小: 需要{self.max_buffer_size}帧才开始分析")
            
            # 直接运行推理循环，ROS回调在后台处理
            self.yolo_inference()
            
        except KeyboardInterrupt:
            print("接收到中断信号")
        except Exception as e:
            print(f"测试器运行异常: {e}")
        finally:
            # 清理资源
            print("正在清理资源...")
            self.cleanup_audio_threads()
            if hasattr(self, 'cap') and self.cap.isOpened():
                self.cap.release()
            cv2.destroyAllWindows()
            rospy.signal_shutdown("程序退出")
            print("测试器已关闭")

if __name__ == "__main__":
    try:
        tester = YOLOTester()
        tester.run()
    except Exception as e:
        print(f"测试器运行失败: {e}")

