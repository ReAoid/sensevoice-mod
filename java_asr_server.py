#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""
Java集成专用ASR服务器
基于SenseVoice模型，专为Java聊天系统提供实时语音识别

功能特性：
- 健康检查API (HTTP)
- WebSocket实时音频流处理
- VAD语音活动检测
- 自动语音结束检测
- 与Java聊天系统无缝集成
- 完整的错误处理和日志记录

使用方法：
1. 启动服务器: python java_asr_server.py
2. Java客户端连接: ws://localhost:8767/asr
3. 健康检查: GET http://localhost:8768/health
"""

import asyncio
import websockets
import json
import base64
import os
import tempfile
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import time
from datetime import datetime
import threading
import queue
from concurrent.futures import ThreadPoolExecutor
import io
import wave
import numpy as np
from aiohttp import web, WSMsgType
import aiohttp_cors
import signal
import sys

try:
    from funasr import AutoModel
    from funasr.utils.postprocess_utils import rich_transcription_postprocess
    import librosa
    import soundfile as sf
except ImportError as e:
    print(f"错误：缺少必要的依赖库: {e}")
    print("请运行: pip install funasr>=1.1.3 websockets aiohttp librosa soundfile numpy")
    exit(1)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class JavaASRServer:
    """Java集成专用ASR服务器"""
    
    def __init__(self, 
                 model_dir: str = "iic/SenseVoiceSmall",
                 device: str = "auto",
                 websocket_port: int = 8767,
                 http_port: int = 8768,
                 host: str = "localhost",
                 sample_rate: int = 16000,
                 silence_duration: float = 2.0,
                 vad_threshold: float = 0.5):
        """
        初始化Java集成专用ASR服务器
        
        Args:
            model_dir: SenseVoice模型目录
            device: 计算设备
            websocket_port: WebSocket端口
            http_port: HTTP健康检查端口
            host: 服务器地址
            sample_rate: 音频采样率
            silence_duration: 静音持续时间判断输入结束(秒)
            vad_threshold: VAD阈值
        """
        self.model_dir = model_dir
        self.device = self._get_device(device)
        self.websocket_port = websocket_port
        self.http_port = http_port
        self.host = host
        self.sample_rate = sample_rate
        self.silence_duration = silence_duration
        self.vad_threshold = vad_threshold
        
        # 模型相关
        self.model = None
        
        # 客户端管理
        self.clients = {}  # websocket -> client_info
        self.audio_buffers = {}  # client_id -> audio_buffer
        self.vad_states = {}  # client_id -> vad_state
        
        # 线程池
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # 服务器状态
        self.server_ready = False
        self.websocket_server = None
        self.http_server = None
        self._start_time = time.time()
        
        # 语言映射
        self.language_map = {
            "auto": "自动检测", "zh": "中文", "en": "英文", 
            "yue": "粤语", "ja": "日语", "ko": "韩语", "nospeech": "无语音"
        }
        
        # 支持的音频格式
        self.supported_formats = {'.wav', '.mp3', '.flac', '.m4a', '.aac', '.ogg', '.wma'}
    
    def _get_device(self, device: str) -> str:
        """自动选择计算设备"""
        if device == "auto":
            try:
                import torch
                return "cuda:0" if torch.cuda.is_available() else "cpu"
            except ImportError:
                return "cpu"
        return device
    
    async def load_models(self):
        """异步加载模型"""
        if self.model is not None:
            return
            
        logger.info(f"🔄 正在加载SenseVoice模型 ({self.model_dir})...")
        logger.info(f"📱 使用设备: {self.device}")
        
        try:
            # 在线程池中加载模型
            loop = asyncio.get_event_loop()
            self.model = await loop.run_in_executor(None, self._load_model_sync)
            logger.info("✅ SenseVoice模型加载成功！")
            self.server_ready = True
        except Exception as e:
            logger.error(f"❌ 模型加载失败: {e}")
            raise
    
    def _load_model_sync(self):
        """同步加载模型"""
        try:
            return AutoModel(
                model=self.model_dir,
                trust_remote_code=True,
                remote_code="./model.py",
                vad_model="fsmn-vad",
                vad_kwargs={"max_single_segment_time": 30000},
                device=self.device,
            )
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            # 尝试不使用VAD的简化版本
            logger.info("尝试加载简化版本（不使用VAD）...")
            return AutoModel(
                model=self.model_dir,
                trust_remote_code=True,
                remote_code="./model.py",
                device=self.device,
            )
    
    # ==================== WebSocket处理 ====================
    
    async def handle_websocket(self, websocket, path="/asr"):
        """处理WebSocket连接"""
        client_id = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}_{int(time.time())}"
        logger.info(f"🔗 Java客户端连接: {client_id}")
        
        # 初始化客户端状态
        self.clients[websocket] = {
            "id": client_id,
            "connected_at": datetime.now(),
            "session_active": False,
            "language": "auto",
            "total_chunks": 0,
            "last_activity": time.time()
        }
        
        self.audio_buffers[client_id] = []
        self.vad_states[client_id] = {
            "is_speaking": False,
            "silence_start": None,
            "last_speech_end": None,
            "accumulated_audio": [],
            "speech_detected": False
        }
        
        try:
            # 发送欢迎消息
            await self.send_welcome_message(websocket, client_id)
            
            # 处理消息
            async for message in websocket:
                try:
                    if isinstance(message, str):
                        # JSON控制消息
                        data = json.loads(message)
                        await self.process_control_message(websocket, data)
                    elif isinstance(message, bytes):
                        # 二进制音频数据
                        await self.process_audio_stream(websocket, message)
                except json.JSONDecodeError:
                    await self.send_error(websocket, "无效的JSON格式")
                except Exception as e:
                    logger.error(f"处理消息时出错: {e}")
                    await self.send_error(websocket, f"处理消息时出错: {str(e)}")
        
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"🔌 Java客户端断开连接: {client_id}")
        except Exception as e:
            logger.error(f"处理Java客户端 {client_id} 时出错: {e}")
        finally:
            # 清理资源
            await self.cleanup_client(websocket, client_id)
    
    async def send_welcome_message(self, websocket, client_id):
        """发送欢迎消息"""
        welcome_msg = {
            "type": "welcome",
            "message": "Java ASR语音识别服务已就绪",
            "client_id": client_id,
            "server_info": {
                "model": self.model_dir,
                "device": self.device,
                "sample_rate": self.sample_rate,
                "vad_enabled": True,
                "silence_duration": self.silence_duration,
                "supported_formats": list(self.supported_formats),
                "java_integration": True
            },
            "timestamp": datetime.now().isoformat()
        }
        await websocket.send(json.dumps(welcome_msg, ensure_ascii=False))
    
    async def process_control_message(self, websocket, data: Dict[str, Any]):
        """处理控制消息"""
        msg_type = data.get("type")
        
        # 对于start_session消息，允许未注册的客户端（会在handle_start_session中自动注册）
        if msg_type == "start_session":
            await self.handle_start_session(websocket, data)
            return
        
        # 其他消息需要客户端已注册
        if websocket not in self.clients:
            await self.send_error(websocket, "客户端未注册，请先发送start_session消息")
            return
        
        if msg_type == "end_session":
            await self.handle_end_session(websocket, data)
        elif msg_type == "audio_chunk":
            await self.handle_audio_chunk(websocket, data)
        elif msg_type == "ping":
            await self.handle_ping(websocket, data)
        elif msg_type == "get_status":
            await self.handle_get_status(websocket, data)
        else:
            await self.send_error(websocket, f"未知的消息类型: {msg_type}")
    
    async def handle_start_session(self, websocket, data: Dict[str, Any]):
        """开始ASR会话"""
        # 检查客户端是否已注册，如果没有则先注册（兼容立即发送start_session的客户端）
        if websocket not in self.clients:
            client_id = f"auto_client_{int(time.time())}_{id(websocket)}"
            logger.info(f"🔧 自动注册客户端: {client_id} (兼容模式)")
            
            # 自动注册客户端
            self.clients[websocket] = {
                "id": client_id,
                "connected_at": datetime.now(),
                "session_active": False,
                "language": "auto",
                "total_chunks": 0,
                "last_activity": time.time()
            }
            
            # 初始化相关状态
            self.audio_buffers[client_id] = []
            self.vad_states[client_id] = {
                "is_speaking": False,
                "silence_start": None,
                "last_speech_end": None,
                "accumulated_audio": [],
                "speech_detected": False
            }
        
        client_info = self.clients[websocket]
        client_id = client_info["id"]
        
        # 更新会话配置
        client_info["session_active"] = True
        client_info["language"] = data.get("language", "auto")
        client_info["session_id"] = data.get("session_id", client_id)
        
        # 重置状态
        self.audio_buffers[client_id] = []
        self.vad_states[client_id] = {
            "is_speaking": False,
            "silence_start": None,
            "last_speech_end": None,
            "accumulated_audio": [],
            "speech_detected": False
        }
        
        logger.info(f"🎙️ 开始ASR会话: {client_id}, 语言: {client_info['language']}")
        
        response = {
            "type": "session_started",
            "client_id": client_id,
            "session_id": client_info["session_id"],
            "language": client_info["language"],
            "message": "ASR会话已开始，请开始说话",
            "timestamp": datetime.now().isoformat()
        }
        await websocket.send(json.dumps(response, ensure_ascii=False))
    
    async def handle_end_session(self, websocket, data: Dict[str, Any]):
        """结束ASR会话"""
        client_info = self.clients[websocket]
        client_id = client_info["id"]
        
        if client_info.get("session_active"):
            # 处理剩余的音频数据
            await self.process_final_audio(websocket)
            
            client_info["session_active"] = False
            logger.info(f"🛑 结束ASR会话: {client_id}")
            
            response = {
                "type": "session_ended",
                "client_id": client_id,
                "total_chunks": client_info.get("total_chunks", 0),
                "message": "ASR会话已结束",
                "timestamp": datetime.now().isoformat()
            }
            await websocket.send(json.dumps(response, ensure_ascii=False))
    
    async def handle_audio_chunk(self, websocket, data: Dict[str, Any]):
        """处理音频数据块"""
        client_info = self.clients[websocket]
        client_id = client_info["id"]
        
        if not client_info.get("session_active"):
            await self.send_error(websocket, "会话未激活，请先开始会话")
            return
        
        audio_data = data.get("audio_data")
        if not audio_data:
            await self.send_error(websocket, "缺少音频数据")
            return
        
        try:
            # 解码音频数据
            audio_bytes = base64.b64decode(audio_data)
            await self.process_audio_data(websocket, audio_bytes)
            
        except Exception as e:
            await self.send_error(websocket, f"处理音频数据失败: {str(e)}")
    
    async def process_audio_stream(self, websocket, audio_bytes: bytes):
        """处理音频流数据"""
        # 检查客户端是否已注册
        if websocket not in self.clients:
            logger.warning("收到未注册客户端的音频数据，忽略")
            return
            
        client_info = self.clients[websocket]
        client_id = client_info["id"]
        
        if not client_info.get("session_active"):
            logger.warning(f"客户端 {client_id} 会话未激活，忽略音频数据")
            return
        
        await self.process_audio_data(websocket, audio_bytes)
    
    async def process_audio_data(self, websocket, audio_bytes: bytes):
        """处理音频数据并执行VAD+ASR"""
        client_info = self.clients[websocket]
        client_id = client_info["id"]
        
        try:
            # 更新活动时间
            client_info["last_activity"] = time.time()
            client_info["total_chunks"] += 1
            
            # 添加到缓冲区
            self.audio_buffers[client_id].append(audio_bytes)
            
            # VAD检测和ASR处理
            await self.process_vad_and_asr(websocket, audio_bytes)
            
        except Exception as e:
            logger.error(f"处理音频数据失败: {e}")
            await self.send_error(websocket, f"音频处理失败: {str(e)}")
    
    async def process_vad_and_asr(self, websocket, audio_bytes: bytes):
        """VAD检测和ASR处理"""
        client_info = self.clients[websocket]
        client_id = client_info["id"]
        vad_state = self.vad_states[client_id]
        
        try:
            # 将音频数据添加到累积缓冲区
            vad_state["accumulated_audio"].append(audio_bytes)
            
            # 在线程池中执行VAD检测
            loop = asyncio.get_event_loop()
            has_speech = await loop.run_in_executor(
                self.executor,
                self._detect_speech_sync,
                audio_bytes
            )
            
            current_time = time.time()
            
            if has_speech:
                # 检测到语音
                if not vad_state["is_speaking"]:
                    # 开始说话
                    vad_state["is_speaking"] = True
                    vad_state["speech_detected"] = True
                    vad_state["silence_start"] = None
                    
                    logger.info(f"🎤 检测到语音开始: {client_id}")
                    
                    # 通知Java端开始说话
                    await self.send_speech_status(websocket, "speech_started")
                
                # 重置静音开始时间
                vad_state["silence_start"] = None
                
            else:
                # 未检测到语音
                if vad_state["is_speaking"]:
                    # 正在说话但当前帧无语音，开始计时
                    if vad_state["silence_start"] is None:
                        vad_state["silence_start"] = current_time
                    
                    # 检查静音持续时间
                    silence_duration = current_time - vad_state["silence_start"]
                    if silence_duration >= self.silence_duration:
                        # 静音时间足够长，认为输入结束
                        await self.handle_input_complete(websocket)
            
            # 定期处理累积的音频（每秒处理一次）
            if len(vad_state["accumulated_audio"]) >= 10:  # 假设每100ms一个音频块
                await self.process_accumulated_audio_for_asr(websocket)
                
        except Exception as e:
            logger.error(f"VAD处理失败: {e}")
    
    def _detect_speech_sync(self, audio_bytes: bytes) -> bool:
        """同步语音检测"""
        try:
            # 检查音频数据长度
            if len(audio_bytes) == 0:
                return False
            
            # 确保音频数据长度是2的倍数（16位音频）
            if len(audio_bytes) % 2 != 0:
                audio_bytes = audio_bytes[:-1]  # 移除最后一个字节
            
            # 将音频字节转换为numpy数组
            audio_data = np.frombuffer(audio_bytes, dtype=np.int16)
            
            if len(audio_data) == 0:
                return False
            
            # 简单的能量检测作为VAD
            energy = np.sum(audio_data.astype(np.float32) ** 2) / len(audio_data)
            
            # 动态阈值调整
            threshold = 1000000  # 可以根据实际情况调整
            
            return energy > threshold
            
        except Exception as e:
            logger.error(f"语音检测失败: {e}")
            return False
    
    async def process_accumulated_audio_for_asr(self, websocket):
        """处理累积的音频进行ASR识别"""
        client_info = self.clients[websocket]
        client_id = client_info["id"]
        vad_state = self.vad_states[client_id]
        
        if not vad_state["accumulated_audio"]:
            return
        
        try:
            # 合并音频数据
            combined_audio = b''.join(vad_state["accumulated_audio"])
            
            # 清空累积缓冲区
            vad_state["accumulated_audio"] = []
            
            # 如果音频数据足够长，进行ASR识别
            if len(combined_audio) > self.sample_rate * 0.5:  # 至少0.5秒的音频
                # 在线程池中执行ASR
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    self.executor,
                    self._recognize_audio_sync,
                    combined_audio,
                    client_info.get("language", "auto")
                )
                
                # 如果有识别结果，发送给Java客户端
                if result and result.get("transcription") and result["transcription"].strip():
                    await self.send_partial_result(websocket, result)
        
        except Exception as e:
            logger.error(f"ASR处理失败: {e}")
    
    async def handle_input_complete(self, websocket):
        """处理输入完成 - 关键函数，通知Java端语音输入结束"""
        client_info = self.clients[websocket]
        client_id = client_info["id"]
        vad_state = self.vad_states[client_id]
        
        logger.info(f"🏁 检测到输入结束: {client_id}")
        
        # 更新状态
        vad_state["is_speaking"] = False
        vad_state["last_speech_end"] = time.time()
        
        # 处理剩余的音频数据
        if vad_state["accumulated_audio"]:
            await self.process_final_accumulated_audio(websocket)
        
        # 通知Java客户端输入结束 - 这是关键消息，Java端需要接收此消息来触发AI对话
        await self.send_speech_status(websocket, "input_complete")
    
    async def process_final_accumulated_audio(self, websocket):
        """处理最终的累积音频"""
        client_info = self.clients[websocket]
        client_id = client_info["id"]
        vad_state = self.vad_states[client_id]
        
        if not vad_state["accumulated_audio"]:
            return
        
        try:
            # 合并所有音频数据
            combined_audio = b''.join(vad_state["accumulated_audio"])
            vad_state["accumulated_audio"] = []
            
            # 执行最终ASR识别
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor,
                self._recognize_audio_sync,
                combined_audio,
                client_info.get("language", "auto")
            )
            
            # 发送最终结果给Java端
            if result and result.get("transcription"):
                await self.send_final_result(websocket, result)
        
        except Exception as e:
            logger.error(f"最终ASR处理失败: {e}")
    
    def _recognize_audio_sync(self, audio_bytes: bytes, language: str = "auto") -> Dict[str, Any]:
        """同步音频识别"""
        try:
            # 检查音频数据
            if len(audio_bytes) == 0:
                logger.warning("音频数据为空")
                return {"transcription": "", "confidence": 0.0}
            
            # 确保音频数据长度是2的倍数（16位音频）
            if len(audio_bytes) % 2 != 0:
                audio_bytes = audio_bytes[:-1]
            
            # 创建正确格式的WAV文件
            temp_path = None
            try:
                # 创建临时文件
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                    temp_path = temp_file.name
                
                # 使用wave模块创建正确的WAV文件
                with wave.open(temp_path, 'wb') as wav_file:
                    wav_file.setnchannels(1)  # 单声道
                    wav_file.setsampwidth(2)  # 16位 = 2字节
                    wav_file.setframerate(self.sample_rate)  # 采样率
                    wav_file.writeframes(audio_bytes)
                
                # 验证文件是否创建成功
                if not os.path.exists(temp_path) or os.path.getsize(temp_path) == 0:
                    logger.error("临时WAV文件创建失败")
                    return {"transcription": "", "confidence": 0.0}
                
                logger.debug(f"创建临时WAV文件: {temp_path}, 大小: {os.path.getsize(temp_path)} bytes")
                
                # 使用SenseVoice进行识别
                res = self.model.generate(
                    input=temp_path,
                    cache={},
                    language=language,
                    use_itn=True,
                    batch_size_s=60,
                    merge_vad=True,
                    merge_length_s=15
                )
                
                if res and len(res) > 0:
                    result = res[0]
                    text = rich_transcription_postprocess(result["text"])
                    
                    # 解析结果
                    parsed_result = self._parse_recognition_result(text)
                    logger.debug(f"识别结果: {parsed_result['transcription']}")
                    return parsed_result
                
                return {"transcription": "", "confidence": 0.0}
                
            finally:
                # 清理临时文件
                if temp_path and os.path.exists(temp_path):
                    try:
                        os.unlink(temp_path)
                    except Exception as cleanup_error:
                        logger.warning(f"清理临时文件失败: {cleanup_error}")
                
        except Exception as e:
            logger.error(f"音频识别失败: {e}")
            return {"transcription": "", "confidence": 0.0, "error": str(e)}
    
    def _parse_recognition_result(self, text: str) -> Dict[str, Any]:
        """解析识别结果"""
        import re
        
        result = {
            "transcription": "",
            "language": "unknown",
            "language_zh": "未知",
            "confidence": 0.8,
            "timestamp": datetime.now().isoformat()
        }
        
        # 提取语言信息
        lang_match = re.search(r'<\|([^|]+)\|>', text)
        if lang_match:
            lang = lang_match.group(1).lower()
            result["language"] = lang
            result["language_zh"] = self.language_map.get(lang, lang)
        
        # 清理转录文本
        clean_text = re.sub(r'<\|[^|]+\|>', '', text).strip()
        result["transcription"] = clean_text
        
        return result
    
    async def send_partial_result(self, websocket, result: Dict[str, Any]):
        """发送部分识别结果"""
        client_info = self.clients[websocket]
        
        response = {
            "type": "partial_result",
            "client_id": client_info["id"],
            "session_id": client_info.get("session_id"),
            "result": result,
            "is_final": False,
            "timestamp": datetime.now().isoformat()
        }
        
        await websocket.send(json.dumps(response, ensure_ascii=False))
        logger.info(f"📝 部分识别结果 [{client_info['id']}]: {result['transcription']}")
    
    async def send_final_result(self, websocket, result: Dict[str, Any]):
        """发送最终识别结果 - Java端接收的完整语音识别结果"""
        client_info = self.clients[websocket]
        
        response = {
            "type": "final_result",
            "client_id": client_info["id"],
            "session_id": client_info.get("session_id"),
            "result": result,
            "is_final": True,
            "speech_complete": True,  # 标记语音输入完成
            "timestamp": datetime.now().isoformat()
        }
        
        await websocket.send(json.dumps(response, ensure_ascii=False))
        logger.info(f"🎯 最终识别结果发送给Java端 [{client_info['id']}]: {result['transcription']}")
    
    async def send_speech_status(self, websocket, status: str):
        """发送语音状态"""
        client_info = self.clients[websocket]
        
        response = {
            "type": "speech_status",
            "client_id": client_info["id"],
            "session_id": client_info.get("session_id"),
            "status": status,
            "timestamp": datetime.now().isoformat()
        }
        
        await websocket.send(json.dumps(response, ensure_ascii=False))
        
        # 特殊处理输入完成状态
        if status == "input_complete":
            logger.info(f"🚨 通知Java端语音输入完成: {client_info['id']}")
    
    async def process_final_audio(self, websocket):
        """处理会话结束时的剩余音频"""
        await self.process_final_accumulated_audio(websocket)
    
    async def handle_ping(self, websocket, data: Dict[str, Any]):
        """处理ping请求"""
        pong_msg = {
            "type": "pong",
            "timestamp": datetime.now().isoformat(),
            "request_id": data.get("request_id", ""),
            "server_status": "ready" if self.server_ready else "loading"
        }
        await websocket.send(json.dumps(pong_msg, ensure_ascii=False))
    
    async def handle_get_status(self, websocket, data: Dict[str, Any]):
        """获取服务器状态"""
        client_info = self.clients[websocket]
        
        status_msg = {
            "type": "server_status",
            "server_ready": self.server_ready,
            "model_loaded": self.model is not None,
            "device": self.device,
            "active_clients": len(self.clients),
            "client_info": {
                "id": client_info["id"],
                "session_active": client_info.get("session_active", False),
                "total_chunks": client_info.get("total_chunks", 0)
            },
            "timestamp": datetime.now().isoformat()
        }
        await websocket.send(json.dumps(status_msg, ensure_ascii=False))
    
    async def send_error(self, websocket, message: str):
        """发送错误消息"""
        error_msg = {
            "type": "error",
            "message": message,
            "timestamp": datetime.now().isoformat()
        }
        await websocket.send(json.dumps(error_msg, ensure_ascii=False))
    
    async def cleanup_client(self, websocket, client_id):
        """清理客户端资源"""
        if websocket in self.clients:
            del self.clients[websocket]
        if client_id in self.audio_buffers:
            del self.audio_buffers[client_id]
        if client_id in self.vad_states:
            del self.vad_states[client_id]
    
    # ==================== HTTP健康检查 ====================
    
    async def health_check(self, request):
        """健康检查接口 - Java端调用此接口检查ASR服务状态"""
        try:
            # 检查模型状态
            model_status = "loaded" if self.model is not None else "not_loaded"
            
            # 检查设备状态
            device_status = "available"
            if self.device.startswith("cuda"):
                try:
                    import torch
                    if not torch.cuda.is_available():
                        device_status = "cuda_unavailable"
                except ImportError:
                    device_status = "torch_not_available"
            
            # 计算服务运行时间
            uptime_seconds = int(time.time() - self._start_time)
            
            status = {
                "status": "healthy" if self.server_ready else "loading",
                "server_ready": self.server_ready,
                "model_loaded": self.model is not None,
                "model_status": model_status,
                "device": self.device,
                "device_status": device_status,
                "active_clients": len(self.clients),
                "websocket_url": f"ws://{self.host}:{self.websocket_port}/asr",
                "http_url": f"http://{self.host}:{self.http_port}",
                "uptime_seconds": uptime_seconds,
                "timestamp": datetime.now().isoformat(),
                "version": "1.0.0",
                "service_name": "Java集成ASR服务器",
                "java_integration": True,
                "supported_features": [
                    "real_time_asr",
                    "vad_detection", 
                    "multi_language",
                    "websocket_streaming",
                    "speech_end_detection",
                    "java_integration"
                ]
            }
            
            status_code = 200 if self.server_ready else 503
            return web.json_response(status, status=status_code)
            
        except Exception as e:
            logger.error(f"健康检查失败: {e}")
            error_status = {
                "status": "error",
                "server_ready": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "version": "1.0.0"
            }
            return web.json_response(error_status, status=500)
    
    async def server_info(self, request):
        """服务器信息接口"""
        info = {
            "name": "Java集成ASR服务器",
            "version": "1.0.0",
            "model": self.model_dir,
            "device": self.device,
            "sample_rate": self.sample_rate,
            "vad_enabled": True,
            "silence_duration": self.silence_duration,
            "supported_formats": list(self.supported_formats),
            "websocket_port": self.websocket_port,
            "http_port": self.http_port,
            "java_integration": True,
            "endpoints": {
                "websocket": f"ws://{self.host}:{self.websocket_port}/asr",
                "health": f"http://{self.host}:{self.http_port}/health",
                "info": f"http://{self.host}:{self.http_port}/info"
            },
            "integration_guide": {
                "java_websocket_url": f"ws://{self.host}:{self.websocket_port}/asr",
                "health_check_url": f"http://{self.host}:{self.http_port}/health",
                "message_types": [
                    "welcome", "session_started", "session_ended", 
                    "partial_result", "final_result", "speech_status", "error"
                ],
                "speech_status_values": [
                    "speech_started", "input_complete"
                ]
            }
        }
        return web.json_response(info)
    
    # ==================== 服务器启动 ====================
    
    async def start_http_server(self):
        """启动HTTP服务器"""
        app = web.Application()
        
        # 添加CORS支持
        cors = aiohttp_cors.setup(app, defaults={
            "*": aiohttp_cors.ResourceOptions(
                allow_credentials=True,
                expose_headers="*",
                allow_headers="*",
                allow_methods="*"
            )
        })
        
        # 添加路由
        app.router.add_get('/health', self.health_check)
        app.router.add_get('/info', self.server_info)
        
        # 添加CORS
        for route in list(app.router.routes()):
            cors.add(route)
        
        # 启动HTTP服务器
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, self.host, self.http_port)
        await site.start()
        
        logger.info(f"✅ HTTP服务器已启动: http://{self.host}:{self.http_port}")
        return runner
    
    async def start_websocket_server(self):
        """启动WebSocket服务器"""
        # 创建WebSocket处理器包装函数
        async def websocket_handler(websocket):
            await self.handle_websocket(websocket, "/asr")  # 直接使用/asr路径
        
        # 启动WebSocket服务器
        server = await websockets.serve(
            websocket_handler,
            self.host,
            self.websocket_port,
            ping_interval=30,
            ping_timeout=10,
            max_size=10 * 1024 * 1024,  # 10MB
            compression=None
        )
        
        logger.info(f"✅ WebSocket服务器已启动: ws://{self.host}:{self.websocket_port}/asr")
        return server
    
    async def start_server(self):
        """启动完整服务器"""
        logger.info("🚀 正在启动Java集成ASR服务器...")
        
        try:
            # 预加载模型
            await self.load_models()
            
            # 启动HTTP服务器
            self.http_server = await self.start_http_server()
            
            # 启动WebSocket服务器
            self.websocket_server = await self.start_websocket_server()
            
            logger.info("🎉 Java集成ASR服务器启动完成！")
            logger.info(f"🔗 WebSocket地址: ws://{self.host}:{self.websocket_port}/asr")
            logger.info(f"🏥 健康检查: http://{self.host}:{self.http_port}/health")
            logger.info(f"ℹ️  服务信息: http://{self.host}:{self.http_port}/info")
            
            return self.websocket_server, self.http_server
            
        except Exception as e:
            logger.error(f"启动服务器失败: {e}")
            raise
    
    async def stop_server(self):
        """停止服务器"""
        logger.info("🛑 正在停止ASR服务器...")
        
        if self.websocket_server:
            self.websocket_server.close()
            await self.websocket_server.wait_closed()
        
        if self.http_server:
            await self.http_server.cleanup()
        
        logger.info("✅ ASR服务器已停止")


async def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Java集成专用ASR WebSocket服务器")
    parser.add_argument("--host", default="localhost", help="服务器地址")
    parser.add_argument("--websocket-port", type=int, default=8767, help="WebSocket端口")
    parser.add_argument("--http-port", type=int, default=8768, help="HTTP端口")
    parser.add_argument("--device", default="auto", help="计算设备")
    parser.add_argument("--model-dir", default="iic/SenseVoiceSmall", help="模型目录")
    parser.add_argument("--sample-rate", type=int, default=16000, help="音频采样率")
    parser.add_argument("--silence-duration", type=float, default=2.0, help="静音持续时间(秒)")
    
    args = parser.parse_args()
    
    # 创建服务器
    asr_server = JavaASRServer(
        model_dir=args.model_dir,
        device=args.device,
        websocket_port=args.websocket_port,
        http_port=args.http_port,
        host=args.host,
        sample_rate=args.sample_rate,
        silence_duration=args.silence_duration
    )
    
    # 设置信号处理
    def signal_handler(signum, frame):
        logger.info("收到中断信号，正在关闭服务器...")
        asyncio.create_task(asr_server.stop_server())
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # 启动服务器
        websocket_server, http_server = await asr_server.start_server()
        
        # 保持服务器运行
        await websocket_server.wait_closed()
        
    except KeyboardInterrupt:
        logger.info("收到中断信号，正在关闭服务器...")
    except Exception as e:
        logger.error(f"服务器运行异常: {e}")
    finally:
        await asr_server.stop_server()


if __name__ == "__main__":
    asyncio.run(main())
