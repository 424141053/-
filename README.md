# 人员外出检测系统 (Person Detection System)

一个基于YOLO和豆包AI的智能人员外出检测系统，适用于医院、养老院或家庭环境，用于监控人员是否准备离开房间。

## 🌟 功能特性

- **实时人员检测**: 使用YOLO模型进行实时人员检测
- **智能外出判断**: 结合豆包AI进行多帧时序分析，准确判断人员是否正在出门
- **多模态通知**: 支持语音播报（粤语）和邮件通知
- **ROS集成**: 支持ROS话题控制，可远程开启/关闭检测
- **时间控制**: 可配置检测时间段，避免误报
- **多帧融合**: 缓存多帧图像进行时序分析，提高判断准确性

## 🏗️ 系统架构

```
摄像头 → YOLO检测 → 帧缓存 → 豆包AI分析 → 通知系统
   ↓         ↓         ↓         ↓         ↓
 实时视频   人员检测   多帧融合   外出判断   语音/邮件
```

## 📋 系统要求

### 硬件要求
- 摄像头设备
- 支持OpenVINO的CPU或GPU

### 软件要求
- Python 3.8+
- ROS (可选)
- OpenCV
- YOLO (ultralytics)

## 🚀 快速开始

### 1. 克隆项目
```bash
git clone https://github.com/424141053/-.git
cd person_detection
```

### 2. 安装依赖
```bash
pip install -r requirements.txt
```

### 3. 配置API密钥
编辑 `person_detection/config/config.py` 文件：

```python
# 豆包API配置
DOUBAO_API_KEY = "your_doubao_api_key_here"

# 腾讯云语音合成配置
TENCENT_SECRET_ID = "your_tencent_secret_id"
TENCENT_SECRET_KEY = "your_tencent_secret_key"

# 邮件通知配置
EMAIL_USERNAME = "your_email@qq.com"
EMAIL_PASSWORD = "your_email_password"
EMAIL_TO = "recipient@example.com"
```

### 4. 运行系统

#### 方式一：直接运行（推荐）
```bash
cd person_detection/scripts
python test_yolo.py
```

#### 方式二：ROS环境运行
```bash
# 启动ROS
roscore

# 运行检测节点
rosrun person_detection person_detection_node.py
```

## 🎮 使用说明

### 基本控制
- **按 's' 键**: 暂停/继续检测
- **按 'q' 键**: 退出程序

### ROS话题控制
```bash
# 开启检测
rostopic pub /person_detection_switch std_msgs/Bool "data: true"

# 关闭检测
rostopic pub /person_detection_switch std_msgs/Bool "data: false"
```

### 配置参数

在 `config/config.py` 中可以调整以下参数：

```python
# 检测时间范围 (24小时制)
START_TIME_HOUR = 8    # 开始时间：8:00
END_TIME_HOUR = 20     # 结束时间：20:00

# YOLO检测参数
YOLO_CONFIDENCE = 0.8  # 置信度阈值
YOLO_DEVICE = "cpu"    # 设备类型：cpu/cuda

# API调用间隔
API_CALL_INTERVAL = 5  # 秒

# 帧缓存大小
MAX_BUFFER_SIZE = 5    # 帧数
```

## 📁 项目结构

```
person_detection/
├── config/
│   └── config.py          # 配置文件
├── scripts/
│   ├── person_detection_node.py  # 主检测节点
│   ├── test_yolo.py             # 测试脚本（多帧分析）
│   └── test.py                  # 简单测试脚本
├── model/
│   ├── people_openvino_model/   # OpenVINO模型
│   └── yolov8n.pt              # YOLO模型文件
├── audio_cache/                 # 音频缓存目录
├── CMakeLists.txt              # ROS构建配置
├── package.xml                 # ROS包配置
└── README.md                   # 项目说明
```

## 🔧 核心功能详解

### 1. 多帧时序分析
- 缓存最近5帧检测到人员的图像
- 当缓存满时，调用豆包AI进行多帧时序分析
- 通过分析门的开关状态和人员位置变化判断是否外出

### 2. 智能判断逻辑
系统会分析以下关键要素：
- 门的状态变化（门把手转动、门被推开）
- 人员位置变化（从室内移动到室外）
- 动作时序（开门→踏出门槛→离开室内）
- 人物朝向（出门时背对摄像头）

### 3. 通知系统
- **语音播报**: 使用腾讯云TTS生成粤语语音警告
- **邮件通知**: 发送详细的外出检测报告
- **缓存机制**: 相同文本的语音会被缓存，避免重复生成

## 🛠️ 开发说明

### 添加新的通知方式
在 `person_detection_node.py` 中添加新的通知函数：

```python
def send_custom_notification(self, message):
    """自定义通知方式"""
    # 实现你的通知逻辑
    pass
```

### 调整检测逻辑
修改 `config.py` 中的相关参数，或直接修改检测脚本中的判断逻辑。

## 📊 性能优化

- **帧缓存机制**: 避免频繁API调用
- **异步处理**: API调用在独立线程中执行
- **图像压缩**: 传输前压缩图像减少带宽
- **音频缓存**: 避免重复生成相同语音

## 🐛 故障排除

### 常见问题

1. **摄像头无法打开**
   - 检查摄像头索引是否正确
   - 确认摄像头未被其他程序占用

2. **YOLO模型加载失败**
   - 检查模型文件路径
   - 确认OpenVINO环境配置正确

3. **API调用失败**
   - 检查网络连接
   - 验证API密钥是否正确
   - 查看API调用频率限制

4. **语音播报无声音**
   - 检查音频设备
   - 确认腾讯云API配置
   - 查看音频文件是否生成成功

## 📝 更新日志

### v1.0.0
- 初始版本发布
- 支持YOLO人员检测
- 集成豆包AI多帧分析
- 实现语音和邮件通知
- 支持ROS话题控制

## 🤝 贡献指南

欢迎提交Issue和Pull Request来改进这个项目！

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 📞 联系方式

如有问题或建议，请通过以下方式联系：
- 提交GitHub Issue
- 发送邮件至项目维护者

---

**注意**: 使用前请确保已正确配置所有API密钥，并在合法合规的环境中使用本系统。