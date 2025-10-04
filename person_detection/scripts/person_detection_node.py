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

    def call_doubao_api(self, image_frame):
        """调用豆包API进行图像分析"""
        # 检查API密钥是否配置
        if config.DOUBAO_API_KEY == "YOUR_DOUBAO_API_KEY":
            print("豆包API密钥未配置，跳过API调用")
            return None
            
        try:
            # 将图像转换为base64
            img_base64 = self.image_to_base64(image_frame)
            if not img_base64:
                return None
            
            # 构建请求头
            headers = {
                "Authorization": f"Bearer {config.DOUBAO_API_KEY}",
                "Content-Type": "application/json"
            }
            
            # 构建系统提示词
            system_prompt = """你是一位专业的图像分析智能体。你的任务是根据提供的图片判断是否有人员准备外出，此应用场景为医院、养老院或家庭，用于监控是否有人准备离开。

## 重要说明:
- 请仔细分析图像内容，准确判断是否有人员准备外出
- 根据分析结果，在结构化输出中正确填写"是否有人员外出"和"是否需要调用工具"字段
- 如果检测到人员外出，在"调用工具列表"中列出需要调用的工具
- 如果没有检测到人员外出，"调用工具列表"应为空数组"""

            # 构建用户提示词
            user_prompt = """请分析这张图片中是否有人员准备外出。

## 判定标准:
- 人员准备外出: 图像中有人且表现出要离开的迹象（如走向门口、携带物品准备离开、穿外套准备出门等）
- 非外出情况: 图像中无人，或有人但没有表现出离开迹象（如坐着工作、休息、聊天等）

## 输出要求:
- 如果检测到人员外出: "是否有人员外出"填"是"，"是否需要调用工具"填"是"，"调用工具列表"包含play_audio和send_sms
- 如果没有检测到人员外出: "是否有人员外出"填"否"，"是否需要调用工具"填"否"，"调用工具列表"为空数组

请仔细分析图像内容并按照结构化输出格式返回结果。"""
            
            
            # 定义结构化输出schema
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": "person_detection_analysis",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "是否有人员外出": {
                                "type": "string",
                                "enum": ["是", "否"],
                                "description": "是否检测到人员准备外出"
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
                        "required": ["是否有人员外出", "是否需要调用工具", "调用工具列表"]
                    }
                }
            }
            
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
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{img_base64}"
                                }
                            },
                            {
                                "type": "text",
                                "text": user_prompt
                            }
                        ]
                    }
                ],
                "response_format": response_format,
                "temperature": config.API_TEMPERATURE,
                "max_tokens": config.API_MAX_TOKENS
            }
            
            # 发送请求
            print("调用豆包API进行图像分析...")
            response = requests.post(config.DOUBAO_API_URL, headers=headers, json=payload, timeout=config.API_TIMEOUT)
            response.raise_for_status()
            
            # 解析响应
            result = response.json()
            print(f"豆包API原始响应数据: {json.dumps(result, ensure_ascii=False, indent=2)}")
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
人员外出检测警告

检测时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
警告信息: {message}

此邮件由人员外出检测系统自动发送。
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
                required_fields = ["是否有人员外出", "是否需要调用工具", "调用工具列表"]
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
            has_person_leaving = structured_data.get("是否有人员外出", "否")
            need_tools = structured_data.get("是否需要调用工具", "否")
            tool_list = structured_data.get("调用工具列表", [])
            
            print(f"📊 结构化分析结果:")
            print(f"   - 是否有人员外出: {has_person_leaving}")
            print(f"   - 是否需要调用工具: {need_tools}")
            
            # 检查逻辑一致性
            if has_person_leaving == "是" and need_tools == "否":
                print("⚠️ 检测到人员外出但不需要调用工具，可能存在逻辑不一致")
            elif has_person_leaving == "否" and need_tools == "是":
                print("⚠️ 未检测到人员外出但需要调用工具，可能存在逻辑不一致")
            
            # 核心判断逻辑：只有当"是否有人员外出"和"是否需要调用工具"都为"是"时才执行工具调用
            if has_person_leaving == "是" and need_tools == "是":
                print(f"🔔 检测到人员外出且需要调用工具，执行 {len(tool_list)} 个工具")
                self.execute_tools_from_list(tool_list)
            elif has_person_leaving == "是" and need_tools == "否":
                print("✅ 检测到人员外出但不需要调用工具，跳过工具执行")
            elif has_person_leaving == "否" and need_tools == "否":
                print("✅ 未检测到人员外出且不需要调用工具，确认无外出情况")
            else:
                print("⚠️ 逻辑异常：未检测到人员外出但需要调用工具，跳过工具执行")
                
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
                    message = tool_params.get("message", "检测到人员外出")
                    self.play_audio(message)
                    executed_tools.append("play_audio")
                elif tool_name == "send_sms":
                    message = tool_params.get("message", "检测到人员外出")
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
                        self.play_audio(arguments.get('message', '检测到人员外出'))
                    elif function_name == "send_sms":
                        self.send_sms(arguments.get('message', '检测到人员外出'))
                except json.JSONDecodeError as e:
                    print(f"工具调用参数解析失败: {e}, 参数内容: '{arguments_str}'")
                    continue
            
            # 记录工具调用总结
            if valid_tool_calls == 0:
                print("✅ 所有工具调用都无效，确认未检测到人员外出")
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
            # 没有工具调用，说明未检测到人员外出
            content = message.get("content", "")
            print(f"✅ 未检测到人员外出: {content}")

    def call_doubao_api_async(self, image_frame):
        """异步调用豆包API（非阻塞）"""
        current_time = time.time()
        
        # 检查是否满足调用间隔
        with self.api_lock:
            if current_time - self.last_api_call_time < self.api_call_interval:
                remaining_time = self.api_call_interval - (current_time - self.last_api_call_time)
                print(f"API调用间隔未到，还需等待 {remaining_time:.1f} 秒")
                return
            
            # 更新最后调用时间
            self.last_api_call_time = current_time
        
        # 在单独线程中执行API调用
        def api_thread():
            try:
                print("开始异步调用豆包API...")
                response = self.call_doubao_api(image_frame)
                self.process_doubao_response(response)
                print("豆包API调用完成")
            except Exception as e:
                print(f"异步API调用失败: {e}")
        
        # 启动线程
        thread = threading.Thread(target=api_thread)
        thread.daemon = True
        thread.start()
        print("豆包API调用线程已启动")

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
                
                # 如果检测到人，异步调用豆包API
                if person_detected:
                    print(f"✅ 检测到 {person_count} 个人，准备调用豆包API分析")
                    self.call_doubao_api_async(frame)
                else:
                    print("❌ 未检测到人员")
                
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
                               (10, 180), cv2.FONT_HERSHEY_SIMPLEX, config.DISPLAY_FONT_SCALE, color, config.DISPLAY_FONT_THICKNESS)
                    
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
            print("- 检测到人员时会自动调用豆包API分析")
            print(f"- API调用间隔: {self.api_call_interval}秒")
            
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

