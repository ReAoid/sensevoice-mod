#!/usr/bin/env python3
# coding=utf-8
"""
集成3D-Speaker声纹识别和SenseVoice语音识别的统一系统
功能：
1. 可选的说话人识别（基于3D-Speaker）
2. 语音识别、情感识别、事件检测（基于SenseVoice）
3. 提供统一的API接口
"""

import os
import sys
import json
import time
import argparse
from typing import Dict, Optional, List, Tuple
import numpy as np
import torch
import torchaudio

from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess

# 导入声纹管理器
from register_voiceprint import VoiceprintManager


class IntegratedASRSystem:
    """集成的ASR系统（3D-Speaker + SenseVoice）"""
    
    def __init__(self,
                 enable_speaker_verification: bool = False,
                 voiceprint_dir: str = "./voiceprint_db",
                 speaker_model_dir: str = "./pretrained_models",
                 speaker_model_type: str = "eres2net",
                 speaker_threshold: float = 0.5,
                 sensevoice_model: str = "iic/SenseVoiceSmall",
                 vad_model: str = "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
                 device: str = "cuda:0"):
        """
        初始化集成ASR系统
        
        Args:
            enable_speaker_verification: 是否启用说话人识别
            voiceprint_dir: 声纹数据库目录
            speaker_model_dir: 说话人识别模型目录
            speaker_model_type: 说话人识别模型类型
            speaker_threshold: 说话人识别阈值
            sensevoice_model: SenseVoice模型名称或路径
            vad_model: VAD模型名称或路径
            device: 计算设备
        """
        self.enable_speaker_verification = enable_speaker_verification
        self.speaker_threshold = speaker_threshold
        self.device = device
        
        print("="*80)
        print("集成ASR系统初始化中...")
        print("="*80)
        
        # 初始化说话人识别模块
        self.speaker_manager = None
        if enable_speaker_verification:
            print("\n[1/2] 初始化3D-Speaker说话人识别模块...")
            self.speaker_manager = VoiceprintManager(
                voiceprint_dir=voiceprint_dir,
                model_dir=speaker_model_dir,
                model_type=speaker_model_type,
                device=device
            )
            # 预加载模型
            self.speaker_manager.load_model()
            print("✓ 3D-Speaker模块初始化完成")
        else:
            print("\n[提示] 说话人识别功能已关闭")
        
        # 初始化SenseVoice模块
        print("\n[2/2] 初始化SenseVoice语音识别模块...")
        self.sensevoice = AutoModel(
            model=sensevoice_model,
            vad_model=vad_model,
            vad_kwargs={"max_single_segment_time": 30000},
            trust_remote_code=True,
            device=device,
        )
        print("✓ SenseVoice模块初始化完成")
        
        print("\n" + "="*80)
        print("系统初始化完成!")
        print("="*80)
        print(f"说话人识别: {'启用' if enable_speaker_verification else '禁用'}")
        print(f"计算设备: {device}")
        print("="*80 + "\n")
    
    def toggle_speaker_verification(self, enable: bool):
        """
        切换说话人识别功能
        
        Args:
            enable: True=启用, False=禁用
        """
        if enable and self.speaker_manager is None:
            print("[WARNING] 说话人识别模块未初始化，无法启用")
            return False
        
        self.enable_speaker_verification = enable
        status = "启用" if enable else "禁用"
        print(f"[INFO] 说话人识别功能已{status}")
        return True
    
    def identify_speaker(self, audio_path: str, return_all_scores: bool = False) -> Optional[Dict]:
        """
        识别说话人
        
        Args:
            audio_path: 音频文件路径
            return_all_scores: 是否返回所有说话人的相似度
            
        Returns:
            说话人信息字典 或 None
        """
        if not self.enable_speaker_verification or self.speaker_manager is None:
            return None
        
        try:
            # 如果需要详细信息，使用扩展版本
            if return_all_scores:
                return self._identify_speaker_detailed(audio_path)
            else:
                result = self.speaker_manager.identify_speaker(
                    audio_path=audio_path,
                    threshold=self.speaker_threshold
                )
                return result
        except Exception as e:
            print(f"[ERROR] 说话人识别失败: {e}")
            return None
    
    def _identify_speaker_detailed(self, audio_path: str) -> Optional[Dict]:
        """
        识别说话人（详细版本，返回所有相似度）
        """
        import numpy as np
        
        # 获取所有已注册的说话人
        speakers = self.speaker_manager.list_speakers(verbose=False)
        
        if not speakers:
            return {
                'matched': False,
                'message': '声纹库为空',
                'all_scores': []
            }
        
        # 提取测试音频特征
        try:
            query_embedding = self.speaker_manager.extract_embedding(audio_path)
        except Exception as e:
            return {
                'matched': False,
                'message': f'特征提取失败: {e}',
                'all_scores': []
            }
        
        # 计算与所有说话人的相似度
        all_scores = []
        for speaker in speakers:
            speaker_id = speaker['speaker_id']
            embedding_file = speaker['embedding_file']
            registered_embedding = np.load(embedding_file)
            
            # 计算余弦相似度
            similarity = np.dot(query_embedding, registered_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(registered_embedding)
            )
            
            all_scores.append({
                'speaker_id': speaker_id,
                'speaker_name': speaker['speaker_name'],
                'similarity': float(similarity)
            })
        
        # 按相似度排序
        all_scores.sort(key=lambda x: x['similarity'], reverse=True)
        
        # 找出最佳匹配
        best_match = all_scores[0]
        
        # 判断是否超过阈值
        if best_match['similarity'] >= self.speaker_threshold:
            return {
                'matched': True,
                'speaker_id': best_match['speaker_id'],
                'speaker_name': best_match['speaker_name'],
                'similarity': best_match['similarity'],
                'threshold': self.speaker_threshold,
                'all_scores': all_scores
            }
        else:
            return {
                'matched': False,
                'best_match': best_match,
                'threshold': self.speaker_threshold,
                'message': f"最高相似度 {best_match['similarity']:.4f} < 阈值 {self.speaker_threshold}",
                'all_scores': all_scores
            }
    
    
    def recognize_speech(self,
                        audio_path: str,
                        language: str = "auto",
                        use_itn: bool = True,
                        batch_size_s: int = 60,
                        merge_vad: bool = True,
                        merge_length_s: int = 15) -> List[Dict]:
        """
        语音识别
        
        Args:
            audio_path: 音频文件路径
            language: 语言 ("auto", "zh", "en", "yue", "ja", "ko")
            use_itn: 是否使用逆文本正则化
            batch_size_s: 批处理大小（秒）
            merge_vad: 是否合并VAD片段
            merge_length_s: 合并长度（秒）
            
        Returns:
            识别结果列表
        """
        try:
            res = self.sensevoice.generate(
                input=audio_path,
                cache={},
                language=language,
                use_itn=use_itn,
                batch_size_s=batch_size_s,
                merge_vad=merge_vad,
                merge_length_s=merge_length_s,
            )
            
            # 后处理文本
            for item in res:
                item['text'] = rich_transcription_postprocess(item['text'])
            
            return res
        except Exception as e:
            print(f"[ERROR] 语音识别失败: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def process_audio(self,
                     audio_path: str,
                     language: str = "auto",
                     use_itn: bool = True,
                     identify_speaker: bool = None) -> Dict:
        """
        完整处理音频（说话人识别 + 语音识别）
        
        Args:
            audio_path: 音频文件路径
            language: 语言
            use_itn: 是否使用逆文本正则化
            identify_speaker: 是否进行说话人识别（None=使用系统默认设置）
            
        Returns:
            处理结果字典
        """
        print("\n" + "="*80)
        print(f"开始处理音频: {audio_path}")
        print("="*80)
        
        result = {
            'audio_path': audio_path,
            'speaker_info': None,
            'asr_results': [],
            'processing_time': 0,
        }
        
        start_time = time.time()
        
        # 1. 说话人识别
        if identify_speaker is None:
            identify_speaker = self.enable_speaker_verification
        
        if identify_speaker:
            print("\n[步骤 1/2] 执行说话人识别...")
            speaker_info = self.identify_speaker(audio_path)
            result['speaker_info'] = speaker_info
            
            if speaker_info:
                print(f"✓ 识别到说话人: {speaker_info['speaker_name']} "
                      f"(相似度: {speaker_info['similarity']:.4f})")
            else:
                print("✗ 未识别到注册的说话人")
        else:
            print("\n[步骤 1/2] 跳过说话人识别")
        
        # 2. 语音识别
        print("\n[步骤 2/2] 执行语音识别...")
        asr_results = self.recognize_speech(
            audio_path=audio_path,
            language=language,
            use_itn=use_itn,
        )
        result['asr_results'] = asr_results
        
        if asr_results:
            print(f"✓ 语音识别完成，共 {len(asr_results)} 个片段")
            for i, item in enumerate(asr_results):
                print(f"  片段 {i+1}: {item['text']}")
        else:
            print("✗ 语音识别失败")
        
        # 统计处理时间
        result['processing_time'] = time.time() - start_time
        
        print(f"\n处理完成! 总耗时: {result['processing_time']:.2f}秒")
        print("="*80 + "\n")
        
        return result
    
    def batch_process(self,
                     audio_list: List[str],
                     language: str = "auto",
                     use_itn: bool = True,
                     output_file: str = None) -> List[Dict]:
        """
        批量处理音频文件
        
        Args:
            audio_list: 音频文件路径列表
            language: 语言
            use_itn: 是否使用逆文本正则化
            output_file: 输出结果文件（JSON格式）
            
        Returns:
            处理结果列表
        """
        print(f"\n批量处理开始: 共 {len(audio_list)} 个音频文件")
        
        results = []
        for i, audio_path in enumerate(audio_list):
            print(f"\n[{i+1}/{len(audio_list)}] 处理: {audio_path}")
            result = self.process_audio(
                audio_path=audio_path,
                language=language,
                use_itn=use_itn,
            )
            results.append(result)
        
        # 保存结果
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"\n结果已保存到: {output_file}")
        
        return results
    
    def get_system_info(self) -> Dict:
        """获取系统信息"""
        info = {
            'speaker_verification_enabled': self.enable_speaker_verification,
            'speaker_threshold': self.speaker_threshold,
            'device': self.device,
        }
        
        if self.speaker_manager:
            speakers = self.speaker_manager.list_speakers(verbose=False)
            info['registered_speakers_count'] = len(speakers)
            info['registered_speakers'] = [s['speaker_id'] for s in speakers]
        else:
            info['registered_speakers_count'] = 0
            info['registered_speakers'] = []
        
        return info


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="集成3D-Speaker和SenseVoice的语音识别系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 基础语音识别（不启用说话人识别）
  python integrated_asr.py --audio test.wav
  
  # 启用说话人识别
  python integrated_asr.py --audio test.wav --enable-speaker
  
  # 批量处理
  python integrated_asr.py --audio-list file1.wav file2.wav file3.wav --enable-speaker
  
  # 指定语言
  python integrated_asr.py --audio test.wav --language zh
  
  # 保存结果到文件
  python integrated_asr.py --audio test.wav --output result.json
        """
    )
    
    parser.add_argument('--audio', type=str,
                       help='音频文件路径')
    
    parser.add_argument('--audio-list', nargs='+', type=str,
                       help='音频文件列表（批量处理）')
    
    parser.add_argument('--enable-speaker', action='store_true',
                       help='启用说话人识别')
    
    parser.add_argument('--voiceprint-dir', type=str, default='./voiceprint_db',
                       help='声纹数据库目录')
    
    parser.add_argument('--speaker-model-dir', type=str, default='./pretrained_models',
                       help='说话人识别模型目录')
    
    parser.add_argument('--speaker-model-type', type=str, default='eres2net',
                       choices=['campplus', 'eres2net', 'eres2netv2'],
                       help='说话人识别模型类型')
    
    parser.add_argument('--speaker-threshold', type=float, default=0.5,
                       help='说话人识别阈值（0-1）')
    
    parser.add_argument('--sensevoice-model', type=str, default='iic/SenseVoiceSmall',
                       help='SenseVoice模型名称或路径')
    
    parser.add_argument('--vad-model', type=str, 
                       default='iic/speech_fsmn_vad_zh-cn-16k-common-pytorch',
                       help='VAD模型名称或路径')
    
    parser.add_argument('--language', type=str, default='auto',
                       choices=['auto', 'zh', 'en', 'yue', 'ja', 'ko'],
                       help='语言')
    
    parser.add_argument('--use-itn', action='store_true', default=True,
                       help='使用逆文本正则化')
    
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='计算设备')
    
    parser.add_argument('--output', type=str,
                       help='输出结果文件（JSON格式）')
    
    parser.add_argument('--show-info', action='store_true',
                       help='显示系统信息')
    
    args = parser.parse_args()
    
    # 创建系统
    system = IntegratedASRSystem(
        enable_speaker_verification=args.enable_speaker,
        voiceprint_dir=args.voiceprint_dir,
        speaker_model_dir=args.speaker_model_dir,
        speaker_model_type=args.speaker_model_type,
        speaker_threshold=args.speaker_threshold,
        sensevoice_model=args.sensevoice_model,
        vad_model=args.vad_model,
        device=args.device,
    )
    
    # 显示系统信息
    if args.show_info:
        info = system.get_system_info()
        print("\n系统信息:")
        print(json.dumps(info, indent=2, ensure_ascii=False))
        return
    
    # 处理音频
    if args.audio_list:
        # 批量处理
        results = system.batch_process(
            audio_list=args.audio_list,
            language=args.language,
            use_itn=args.use_itn,
            output_file=args.output,
        )
    elif args.audio:
        # 单个处理
        result = system.process_audio(
            audio_path=args.audio,
            language=args.language,
            use_itn=args.use_itn,
        )
        
        # 保存结果
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"\n结果已保存到: {args.output}")
    else:
        print("[ERROR] 请提供 --audio 或 --audio-list 参数")
        parser.print_help()


if __name__ == '__main__':
    main()

