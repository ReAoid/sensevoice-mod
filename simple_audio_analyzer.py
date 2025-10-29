#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""
简化版本地音频分析工具
直接在代码中指定音频文件路径，无需命令行参数

使用方法：
1. 在下面的配置区域填写您的音频文件路径
2. 运行: python simple_audio_analyzer.py
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, Any, List
import time
from datetime import datetime

# ==================== 配置区域 - 在这里填写您的设置 ====================

# 音频文件路径配置 - 请修改为您的本地音频文件路径
AUDIO_FILES = [
    # 单个文件示例（取消注释并修改路径）
    # "/Users/mingy/Documents/audio/test.wav",
    # "/Users/mingy/Documents/audio/speech.mp3",
    
    # 或者指定一个包含音频文件的目录
    # "/Users/mingy/Documents/audio/",
    
    # 示例路径（请替换为您的实际路径）
    # "/Users/mingy/Documents/python/SenseVoice/【默认】很害怕的表情呢。放心，既然此事与你无关，我不会为难你的。就当陪我走走吧。.wav",  # 当前目录下的音频文件
    # "/Users/mingy/Documents/python/SenseVoice/【默认】诶呀，最近总是见到你嘛…我在哪儿看见的？这能让你知道吗，嘻嘻～.wav",  # 当前目录下的音频文件
    # "/Users/mingy/Documents/python/SenseVoice/【默认】那就是裂界侵蚀的痕迹越弱，附近的情况就会越安全！.wav",  # 当前目录下的音频文件
    "/Users/mingy/Documents/python/SenseVoice/派蒙.wav",  # 当前目录下的音频文件
]

# 输出设置
OUTPUT_FILE = "analysis_results.json"  # 结果保存文件名
OUTPUT_FORMAT = "json"  # 输出格式: "json", "csv", "txt"

# 分析设置
LANGUAGE = "auto"  # 语言设置: "auto", "zh", "en", "yue", "ja", "ko"
DEVICE = "auto"    # 设备设置: "auto", "cpu", "cuda:0"
USE_TIMESTAMP = False  # 是否输出时间戳
USE_ITN = True     # 是否使用标点符号
USE_VAD = True     # 是否使用语音活动检测

# ================================================================

try:
    from funasr import AutoModel
    from funasr.utils.postprocess_utils import rich_transcription_postprocess
except ImportError:
    print("错误：请先安装funasr库")
    print("运行: pip install funasr>=1.1.3")
    sys.exit(1)


class SimpleAudioAnalyzer:
    """简化的音频分析器"""
    
    def __init__(self, device="auto", use_vad=True):
        self.device = self._get_device(device)
        self.use_vad = use_vad
        self.model = None
        self.supported_formats = {'.wav', '.mp3', '.flac', '.m4a', '.aac', '.ogg', '.wma'}
        
        # 中文映射
        self.language_map = {
            "auto": "自动检测", "zh": "中文", "en": "英文", 
            "yue": "粤语", "ja": "日语", "ko": "韩语", "nospeech": "无语音"
        }
        
        self.emotion_map = {
            "HAPPY": "快乐", "SAD": "悲伤", "ANGRY": "愤怒", "NEUTRAL": "中性",
            "FEARFUL": "恐惧", "DISGUSTED": "厌恶", "SURPRISED": "惊讶"
        }
        
        self.event_map = {
            "Speech": "语音", "BGM": "背景音乐", "Applause": "掌声", "Laughter": "笑声",
            "Cry": "哭声", "Sneeze": "喷嚏", "Breath": "呼吸声", "Cough": "咳嗽"
        }
    
    def _get_device(self, device):
        """自动选择设备"""
        if device == "auto":
            try:
                import torch
                return "cuda:0" if torch.cuda.is_available() else "cpu"
            except ImportError:
                return "cpu"
        return device
    
    def load_model(self):
        """加载模型"""
        print(f"正在加载SenseVoice模型...")
        print(f"使用设备: {self.device}")
        
        try:
            if self.use_vad:
                self.model = AutoModel(
                    model="iic/SenseVoiceSmall",
                    trust_remote_code=True,
                    remote_code="./model.py",
                    vad_model="fsmn-vad",
                    vad_kwargs={"max_single_segment_time": 30000},
                    device=self.device,
                )
            else:
                self.model = AutoModel(
                    model="iic/SenseVoiceSmall",
                    trust_remote_code=True,
                    remote_code="./model.py",
                    device=self.device
                )
            print("✓ 模型加载成功！")
        except Exception as e:
            print(f"✗ 模型加载失败: {e}")
            sys.exit(1)
    
    def analyze_audio(self, audio_path, language="auto", use_itn=True, output_timestamp=False):
        """分析音频文件"""
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"音频文件不存在: {audio_path}")
        
        if Path(audio_path).suffix.lower() not in self.supported_formats:
            raise ValueError(f"不支持的音频格式: {Path(audio_path).suffix}")
        
        if self.model is None:
            self.load_model()
        
        print(f"\n正在分析: {Path(audio_path).name}")
        start_time = time.time()
        
        try:
            # 执行分析
            if self.use_vad:
                res = self.model.generate(
                    input=audio_path,
                    cache={},
                    language=language,
                    use_itn=use_itn,
                    batch_size_s=60,
                    merge_vad=True,
                    merge_length_s=15,
                    output_timestamp=output_timestamp
                )
            else:
                res = self.model.generate(
                    input=audio_path,
                    cache={},
                    language=language,
                    use_itn=use_itn,
                    batch_size=1,
                    output_timestamp=output_timestamp
                )
            
            # 处理结果
            result = res[0]
            text = rich_transcription_postprocess(result["text"])
            
            # 解析富文本信息
            analysis_result = self._parse_rich_text(text)
            
            # 添加基本信息
            analysis_result.update({
                "file_path": audio_path,
                "file_name": Path(audio_path).name,
                "processing_time": round(time.time() - start_time, 2),
                "timestamp": datetime.now().isoformat(),
                "raw_text": result["text"],
                "processed_text": text
            })
            
            # 添加时间戳（如果启用）
            if output_timestamp and "timestamp" in result:
                analysis_result["word_timestamps"] = result["timestamp"]
            
            return analysis_result
            
        except Exception as e:
            raise RuntimeError(f"音频分析失败: {e}")
    
    def _parse_rich_text(self, text):
        """解析富文本信息"""
        import re
        
        result = {
            "transcription": "",
            "language": "unknown",
            "language_zh": "未知",
            "emotion": "unknown", 
            "emotion_zh": "未知",
            "events": [],
            "events_zh": []
        }
        
        # 提取语言
        lang_match = re.search(r'<\|([^|]+)\|>', text)
        if lang_match:
            lang = lang_match.group(1).lower()
            result["language"] = lang
            result["language_zh"] = self.language_map.get(lang, lang)
        
        # 提取情感
        emotion_match = re.search(r'<\|([A-Z]+)\|>', text)
        if emotion_match:
            emotion = emotion_match.group(1)
            if emotion in self.emotion_map:
                result["emotion"] = emotion
                result["emotion_zh"] = self.emotion_map[emotion]
        
        # 提取事件
        event_matches = re.findall(r'<\|([A-Za-z]+)\|>', text)
        for event in event_matches:
            if event in self.event_map:
                result["events"].append(event)
                result["events_zh"].append(self.event_map[event])
        
        # 清理转录文本
        clean_text = re.sub(r'<\|[^|]+\|>', '', text).strip()
        result["transcription"] = clean_text
        
        return result
    
    def save_results(self, results, output_path, format_type="json"):
        """保存结果"""
        if isinstance(results, dict):
            results = [results]
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format_type == "json":
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
        
        elif format_type == "csv":
            import csv
            if results:
                fieldnames = ['file_name', 'transcription', 'language_zh', 'emotion_zh', 'events_zh', 'processing_time']
                with open(output_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    for result in results:
                        if 'error' not in result:
                            row = {
                                'file_name': result.get('file_name', ''),
                                'transcription': result.get('transcription', ''),
                                'language_zh': result.get('language_zh', ''),
                                'emotion_zh': result.get('emotion_zh', ''),
                                'events_zh': ', '.join(result.get('events_zh', [])),
                                'processing_time': result.get('processing_time', '')
                            }
                            writer.writerow(row)
        
        elif format_type == "txt":
            with open(output_path, 'w', encoding='utf-8') as f:
                for result in results:
                    if 'error' not in result:
                        f.write(f"文件: {result.get('file_name', '')}\n")
                        f.write(f"转录: {result.get('transcription', '')}\n")
                        f.write(f"语言: {result.get('language_zh', '')}\n")
                        f.write(f"情感: {result.get('emotion_zh', '')}\n")
                        f.write(f"事件: {', '.join(result.get('events_zh', []))}\n")
                        f.write(f"处理时间: {result.get('processing_time', '')}秒\n")
                        f.write("-" * 50 + "\n")
        
        print(f"✓ 结果已保存到: {output_path}")


def find_audio_files(directory):
    """查找目录中的音频文件"""
    audio_files = []
    supported_formats = {'.wav', '.mp3', '.flac', '.m4a', '.aac', '.ogg', '.wma'}
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if Path(file).suffix.lower() in supported_formats:
                audio_files.append(os.path.join(root, file))
    
    return sorted(audio_files)


def main():
    """主函数"""
    print("SenseVoice 简化版音频分析工具")
    print("=" * 50)
    
    # 检查配置
    if not AUDIO_FILES:
        print("❌ 错误: 请在代码顶部的配置区域填写音频文件路径")
        print("\n请修改 AUDIO_FILES 变量，例如：")
        print('AUDIO_FILES = ["/path/to/your/audio.wav"]')
        return
    
    # 收集所有音频文件
    all_audio_files = []
    
    for path in AUDIO_FILES:
        if os.path.isfile(path):
            # 单个文件
            all_audio_files.append(path)
        elif os.path.isdir(path):
            # 目录
            found_files = find_audio_files(path)
            all_audio_files.extend(found_files)
            print(f"在目录 {path} 中找到 {len(found_files)} 个音频文件")
        else:
            print(f"⚠️  警告: 路径不存在: {path}")
    
    if not all_audio_files:
        print("❌ 错误: 未找到任何音频文件")
        return
    
    print(f"\n总共找到 {len(all_audio_files)} 个音频文件")
    
    # 创建分析器
    analyzer = SimpleAudioAnalyzer(device=DEVICE, use_vad=USE_VAD)
    
    # 分析音频文件
    results = []
    
    for i, audio_file in enumerate(all_audio_files, 1):
        print(f"\n进度: {i}/{len(all_audio_files)}")
        try:
            result = analyzer.analyze_audio(
                audio_path=audio_file,
                language=LANGUAGE,
                use_itn=USE_ITN,
                output_timestamp=USE_TIMESTAMP
            )
            results.append(result)
            
            # 显示结果
            print(f"✓ 分析完成")
            print(f"  转录: {result['transcription']}")
            print(f"  语言: {result['language_zh']}")
            print(f"  情感: {result['emotion_zh']}")
            print(f"  事件: {', '.join(result['events_zh']) if result['events_zh'] else '无'}")
            print(f"  处理时间: {result['processing_time']}秒")
            
        except Exception as e:
            error_result = {
                "file_path": audio_file,
                "file_name": Path(audio_file).name,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
            results.append(error_result)
            print(f"✗ 分析失败: {e}")
    
    # 保存结果
    if results:
        analyzer.save_results(results, OUTPUT_FILE, OUTPUT_FORMAT)
        
        # 统计
        success_count = sum(1 for r in results if 'error' not in r)
        error_count = len(results) - success_count
        
        print(f"\n" + "=" * 50)
        print(f"分析完成!")
        print(f"总文件数: {len(results)}")
        print(f"成功: {success_count}")
        print(f"失败: {error_count}")
        print(f"结果文件: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
