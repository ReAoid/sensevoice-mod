#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""
本地音频分析工具
基于SenseVoice模型进行语音识别、情感识别、事件检测和语言识别

功能特性：
- 多语言语音识别 (中文、英文、粤语、日语、韩语)
- 情感识别 (快乐、悲伤、愤怒、中性、恐惧、厌恶、惊讶)
- 事件检测 (背景音乐、掌声、笑声、哭声、咳嗽、喷嚏等)
- 支持多种音频格式
- 批量处理
- 时间戳输出
- 多种输出格式
"""

import os
import sys
import json
import csv
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import time
from datetime import datetime

try:
    from funasr import AutoModel
    from funasr.utils.postprocess_utils import rich_transcription_postprocess
except ImportError:
    print("错误：请先安装funasr库")
    print("运行: pip install funasr>=1.1.3")
    sys.exit(1)


class AudioAnalyzer:
    """音频分析器类"""
    
    def __init__(self, 
                 model_dir: str = "iic/SenseVoiceSmall",
                 device: str = "auto",
                 use_vad: bool = True,
                 max_segment_time: int = 30000):
        """
        初始化音频分析器
        
        Args:
            model_dir: 模型目录或名称
            device: 设备 ("auto", "cpu", "cuda:0" 等)
            use_vad: 是否使用VAD语音活动检测
            max_segment_time: VAD最大分段时间(毫秒)
        """
        self.model_dir = model_dir
        self.device = self._get_device(device)
        self.use_vad = use_vad
        self.max_segment_time = max_segment_time
        self.model = None
        
        # 支持的音频格式
        self.supported_formats = {'.wav', '.mp3', '.flac', '.m4a', '.aac', '.ogg', '.wma'}
        
        # 语言映射
        self.language_map = {
            "auto": "自动检测",
            "zh": "中文",
            "en": "英文", 
            "yue": "粤语",
            "ja": "日语",
            "ko": "韩语",
            "nospeech": "无语音"
        }
        
        # 情感映射
        self.emotion_map = {
            "HAPPY": "快乐",
            "SAD": "悲伤", 
            "ANGRY": "愤怒",
            "NEUTRAL": "中性",
            "FEARFUL": "恐惧",
            "DISGUSTED": "厌恶",
            "SURPRISED": "惊讶"
        }
        
        # 事件映射
        self.event_map = {
            "Speech": "语音",
            "BGM": "背景音乐",
            "Applause": "掌声",
            "Laughter": "笑声", 
            "Cry": "哭声",
            "Sneeze": "喷嚏",
            "Breath": "呼吸声",
            "Cough": "咳嗽"
        }
        
    def _get_device(self, device: str) -> str:
        """自动选择设备"""
        if device == "auto":
            try:
                import torch
                if torch.cuda.is_available():
                    return "cuda:0"
                else:
                    return "cpu"
            except ImportError:
                return "cpu"
        return device
    
    def load_model(self):
        """加载模型"""
        print(f"正在加载SenseVoice模型 ({self.model_dir})...")
        print(f"使用设备: {self.device}")
        
        try:
            if self.use_vad:
                self.model = AutoModel(
                    model=self.model_dir,
                    trust_remote_code=True,
                    remote_code="./model.py",
                    vad_model="fsmn-vad",
                    vad_kwargs={"max_single_segment_time": self.max_segment_time},
                    device=self.device,
                )
            else:
                self.model = AutoModel(
                    model=self.model_dir,
                    trust_remote_code=True,
                    remote_code="./model.py",
                    device=self.device
                )
            print("模型加载成功！")
        except Exception as e:
            print(f"模型加载失败: {e}")
            sys.exit(1)
    
    def is_audio_file(self, file_path: str) -> bool:
        """检查是否为支持的音频文件"""
        return Path(file_path).suffix.lower() in self.supported_formats
    
    def analyze_audio(self, 
                     audio_path: str,
                     language: str = "auto",
                     use_itn: bool = True,
                     output_timestamp: bool = False,
                     ban_emo_unk: bool = False) -> Dict[str, Any]:
        """
        分析单个音频文件
        
        Args:
            audio_path: 音频文件路径
            language: 语言 ("auto", "zh", "en", "yue", "ja", "ko", "nospeech")
            use_itn: 是否使用逆文本正则化（标点符号）
            output_timestamp: 是否输出时间戳
            ban_emo_unk: 是否禁用未知情感标签
            
        Returns:
            分析结果字典
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"音频文件不存在: {audio_path}")
        
        if not self.is_audio_file(audio_path):
            raise ValueError(f"不支持的音频格式: {Path(audio_path).suffix}")
        
        if self.model is None:
            self.load_model()
        
        print(f"正在分析音频: {audio_path}")
        start_time = time.time()
        
        try:
            # 生成分析结果
            if self.use_vad:
                res = self.model.generate(
                    input=audio_path,
                    cache={},
                    language=language,
                    use_itn=use_itn,
                    batch_size_s=60,
                    merge_vad=True,
                    merge_length_s=15,
                    ban_emo_unk=ban_emo_unk,
                    output_timestamp=output_timestamp
                )
            else:
                res = self.model.generate(
                    input=audio_path,
                    cache={},
                    language=language,
                    use_itn=use_itn,
                    batch_size=1,
                    ban_emo_unk=ban_emo_unk,
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
            
            # 添加时间戳信息（如果可用）
            if output_timestamp and "timestamp" in result:
                analysis_result["word_timestamps"] = result["timestamp"]
            
            return analysis_result
            
        except Exception as e:
            raise RuntimeError(f"音频分析失败: {e}")
    
    def _parse_rich_text(self, text: str) -> Dict[str, Any]:
        """解析富文本，提取语言、情感、事件信息"""
        result = {
            "transcription": "",
            "language": "unknown",
            "language_zh": "未知",
            "emotion": "unknown", 
            "emotion_zh": "未知",
            "events": [],
            "events_zh": []
        }
        
        # 提取转录文本（移除标签）
        import re
        
        # 提取语言标签
        lang_match = re.search(r'<\|([^|]+)\|>', text)
        if lang_match:
            lang = lang_match.group(1).lower()
            result["language"] = lang
            result["language_zh"] = self.language_map.get(lang, lang)
        
        # 提取情感标签
        emotion_match = re.search(r'<\|([A-Z]+)\|>', text)
        if emotion_match:
            emotion = emotion_match.group(1)
            if emotion in self.emotion_map:
                result["emotion"] = emotion
                result["emotion_zh"] = self.emotion_map[emotion]
        
        # 提取事件标签
        event_matches = re.findall(r'<\|([A-Za-z]+)\|>', text)
        for event in event_matches:
            if event in self.event_map:
                result["events"].append(event)
                result["events_zh"].append(self.event_map[event])
        
        # 清理转录文本
        clean_text = re.sub(r'<\|[^|]+\|>', '', text).strip()
        result["transcription"] = clean_text
        
        return result
    
    def batch_analyze(self, 
                     input_paths: List[str],
                     language: str = "auto",
                     use_itn: bool = True,
                     output_timestamp: bool = False,
                     ban_emo_unk: bool = False) -> List[Dict[str, Any]]:
        """
        批量分析音频文件
        
        Args:
            input_paths: 音频文件路径列表
            language: 语言设置
            use_itn: 是否使用逆文本正则化
            output_timestamp: 是否输出时间戳
            ban_emo_unk: 是否禁用未知情感标签
            
        Returns:
            分析结果列表
        """
        results = []
        total_files = len(input_paths)
        
        print(f"开始批量分析 {total_files} 个音频文件...")
        
        for i, audio_path in enumerate(input_paths, 1):
            print(f"\n进度: {i}/{total_files}")
            try:
                result = self.analyze_audio(
                    audio_path=audio_path,
                    language=language,
                    use_itn=use_itn,
                    output_timestamp=output_timestamp,
                    ban_emo_unk=ban_emo_unk
                )
                results.append(result)
                print(f"✓ 分析完成: {Path(audio_path).name}")
            except Exception as e:
                error_result = {
                    "file_path": audio_path,
                    "file_name": Path(audio_path).name,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
                results.append(error_result)
                print(f"✗ 分析失败: {Path(audio_path).name} - {e}")
        
        return results
    
    def save_results(self, 
                    results: Union[Dict, List[Dict]], 
                    output_path: str,
                    format: str = "json"):
        """
        保存分析结果
        
        Args:
            results: 分析结果
            output_path: 输出文件路径
            format: 输出格式 ("json", "csv", "txt")
        """
        # 确保results是列表格式
        if isinstance(results, dict):
            results = [results]
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format.lower() == "json":
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
        
        elif format.lower() == "csv":
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
        
        elif format.lower() == "txt":
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
        
        print(f"结果已保存到: {output_path}")


def find_audio_files(directory: str) -> List[str]:
    """递归查找目录中的音频文件"""
    audio_files = []
    supported_formats = {'.wav', '.mp3', '.flac', '.m4a', '.aac', '.ogg', '.wma'}
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if Path(file).suffix.lower() in supported_formats:
                audio_files.append(os.path.join(root, file))
    
    return sorted(audio_files)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="SenseVoice本地音频分析工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 分析单个音频文件
  python audio_analyzer.py -i audio.wav
  
  # 批量分析目录中的音频文件
  python audio_analyzer.py -i /path/to/audio/dir -o results.json
  
  # 指定语言和输出格式
  python audio_analyzer.py -i audio.wav -l zh -f csv -o results.csv
  
  # 启用时间戳输出
  python audio_analyzer.py -i audio.wav --timestamp -o results.json
        """
    )
    
    parser.add_argument('-i', '--input', required=True,
                       help='输入音频文件或目录路径')
    parser.add_argument('-o', '--output', 
                       help='输出文件路径 (默认: results.json)')
    parser.add_argument('-l', '--language', default='auto',
                       choices=['auto', 'zh', 'en', 'yue', 'ja', 'ko', 'nospeech'],
                       help='语言设置 (默认: auto)')
    parser.add_argument('-f', '--format', default='json',
                       choices=['json', 'csv', 'txt'],
                       help='输出格式 (默认: json)')
    parser.add_argument('--device', default='auto',
                       help='计算设备 (默认: auto)')
    parser.add_argument('--no-vad', action='store_true',
                       help='禁用VAD语音活动检测')
    parser.add_argument('--no-itn', action='store_true',
                       help='禁用逆文本正则化（标点符号）')
    parser.add_argument('--timestamp', action='store_true',
                       help='输出词级时间戳')
    parser.add_argument('--ban-emo-unk', action='store_true',
                       help='禁用未知情感标签')
    parser.add_argument('--model-dir', default='iic/SenseVoiceSmall',
                       help='模型目录 (默认: iic/SenseVoiceSmall)')
    
    args = parser.parse_args()
    
    # 检查输入路径
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"错误: 输入路径不存在: {args.input}")
        sys.exit(1)
    
    # 获取音频文件列表
    if input_path.is_file():
        if not AudioAnalyzer().is_audio_file(str(input_path)):
            print(f"错误: 不支持的音频格式: {input_path.suffix}")
            sys.exit(1)
        audio_files = [str(input_path)]
    else:
        audio_files = find_audio_files(str(input_path))
        if not audio_files:
            print(f"错误: 在目录 {args.input} 中未找到音频文件")
            sys.exit(1)
    
    print(f"找到 {len(audio_files)} 个音频文件")
    
    # 设置输出路径
    if args.output:
        output_path = args.output
    else:
        if len(audio_files) == 1:
            output_path = f"{Path(audio_files[0]).stem}_analysis.{args.format}"
        else:
            output_path = f"batch_analysis.{args.format}"
    
    # 创建分析器
    analyzer = AudioAnalyzer(
        model_dir=args.model_dir,
        device=args.device,
        use_vad=not args.no_vad
    )
    
    try:
        # 执行分析
        if len(audio_files) == 1:
            print(f"\n开始分析音频文件...")
            result = analyzer.analyze_audio(
                audio_path=audio_files[0],
                language=args.language,
                use_itn=not args.no_itn,
                output_timestamp=args.timestamp,
                ban_emo_unk=args.ban_emo_unk
            )
            
            # 显示结果
            print(f"\n=== 分析结果 ===")
            print(f"文件: {result['file_name']}")
            print(f"转录: {result['transcription']}")
            print(f"语言: {result['language_zh']}")
            print(f"情感: {result['emotion_zh']}")
            print(f"事件: {', '.join(result['events_zh']) if result['events_zh'] else '无'}")
            print(f"处理时间: {result['processing_time']}秒")
            
            # 保存结果
            analyzer.save_results(result, output_path, args.format)
            
        else:
            print(f"\n开始批量分析...")
            results = analyzer.batch_analyze(
                input_paths=audio_files,
                language=args.language,
                use_itn=not args.no_itn,
                output_timestamp=args.timestamp,
                ban_emo_unk=args.ban_emo_unk
            )
            
            # 统计结果
            success_count = sum(1 for r in results if 'error' not in r)
            error_count = len(results) - success_count
            
            print(f"\n=== 批量分析完成 ===")
            print(f"总文件数: {len(results)}")
            print(f"成功: {success_count}")
            print(f"失败: {error_count}")
            
            # 保存结果
            analyzer.save_results(results, output_path, args.format)
    
    except KeyboardInterrupt:
        print("\n用户中断操作")
        sys.exit(1)
    except Exception as e:
        print(f"错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
