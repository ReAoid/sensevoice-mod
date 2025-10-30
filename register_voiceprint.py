#!/usr/bin/env python3
# coding=utf-8
"""
声纹注册管理系统
功能：
1. 注册本地声音文件到声纹库
2. 管理已注册的声纹
3. 从声纹库中删除声纹
4. 列出所有已注册的声纹
"""

import os
import sys
import json
import argparse
import pathlib
import numpy as np
import torch
import torchaudio
from typing import List, Dict, Optional
from datetime import datetime

# 添加speakerlab到路径
sys.path.append(os.path.join(os.path.dirname(__file__)))

from speakerlab.process.processor import FBank
from speakerlab.utils.builder import dynamic_import


class VoiceprintManager:
    """声纹管理器"""
    
    # 支持的模型配置
    MODELS = {
        'campplus': {
            'model_id': 'iic/speech_campplus_sv_zh-cn_16k-common',
            'revision': 'v1.0.0',
            'obj': 'speakerlab.models.campplus.DTDNN.CAMPPlus',
            'args': {'feat_dim': 80, 'embedding_size': 192},
            'model_file': 'campplus_cn_common.bin',
        },
        'eres2net': {
            'model_id': 'iic/speech_eres2net_sv_zh-cn_16k-common',
            'revision': 'v1.0.5',
            'obj': 'speakerlab.models.eres2net.ERes2Net_huge.ERes2Net',
            'args': {'feat_dim': 80, 'embedding_size': 192},
            'model_file': 'pretrained_eres2net_aug.ckpt',
        },
        'eres2netv2': {
            'model_id': 'iic/speech_eres2netv2_sv_zh-cn_16k-common',
            'revision': 'v1.0.1',
            'obj': 'speakerlab.models.eres2net.ERes2NetV2.ERes2NetV2',
            'args': {'feat_dim': 80, 'embedding_size': 192, 'baseWidth': 26, 'scale': 2, 'expansion': 2},
            'model_file': 'pretrained_eres2netv2.ckpt',
        },
    }
    
    def __init__(self, 
                 voiceprint_dir: str = "./voiceprint_db",
                 model_dir: str = "./pretrained_models",
                 model_type: str = "eres2net",
                 device: str = None):
        """
        初始化声纹管理器
        
        Args:
            voiceprint_dir: 声纹数据库目录
            model_dir: 预训练模型目录
            model_type: 模型类型 ('campplus', 'eres2net', 'eres2netv2')
            device: 计算设备 ('cuda' 或 'cpu')
        """
        self.voiceprint_dir = pathlib.Path(voiceprint_dir)
        self.voiceprint_dir.mkdir(exist_ok=True, parents=True)
        
        self.model_dir = pathlib.Path(model_dir)
        self.model_dir.mkdir(exist_ok=True, parents=True)
        
        self.embeddings_dir = self.voiceprint_dir / "embeddings"
        self.embeddings_dir.mkdir(exist_ok=True, parents=True)
        
        self.metadata_file = self.voiceprint_dir / "metadata.json"
        self.metadata = self._load_metadata()
        
        # 设置设备
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # 初始化模型
        self.model_type = model_type
        if model_type not in self.MODELS:
            raise ValueError(f"不支持的模型类型: {model_type}. 支持的类型: {list(self.MODELS.keys())}")
        
        self.model_config = self.MODELS[model_type]
        self.model = None
        self.feature_extractor = FBank(80, sample_rate=16000, mean_nor=True)
        
        print(f"[INFO] 声纹管理器初始化完成")
        print(f"[INFO] 声纹库目录: {self.voiceprint_dir}")
        print(f"[INFO] 模型类型: {model_type}")
        print(f"[INFO] 计算设备: {self.device}")
    
    def _load_metadata(self) -> Dict:
        """加载元数据"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def _save_metadata(self):
        """保存元数据"""
        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)
    
    def _download_model(self):
        """下载模型"""
        try:
            from modelscope.hub.snapshot_download import snapshot_download
            
            model_id = self.model_config['model_id']
            revision = self.model_config['revision']
            
            print(f"[INFO] 正在从ModelScope下载模型: {model_id}")
            cache_dir = snapshot_download(model_id, revision=revision)
            print(f"[INFO] 模型下载完成: {cache_dir}")
            return pathlib.Path(cache_dir)
        except Exception as e:
            print(f"[ERROR] 下载模型失败: {e}")
            raise
    
    def load_model(self):
        """加载声纹识别模型"""
        if self.model is not None:
            return
        
        # 检查本地是否有模型
        model_name = self.model_config['model_id'].split('/')[-1]
        local_model_dir = self.model_dir / model_name
        
        if not local_model_dir.exists():
            # 下载模型
            cache_dir = self._download_model()
            # 创建符号链接
            try:
                local_model_dir.symlink_to(cache_dir)
            except:
                # 如果符号链接失败，复制文件
                import shutil
                shutil.copytree(cache_dir, local_model_dir)
        
        # 加载模型权重
        model_file = local_model_dir / self.model_config['model_file']
        if not model_file.exists():
            raise FileNotFoundError(f"模型文件不存在: {model_file}")
        
        pretrained_state = torch.load(model_file, map_location='cpu')
        
        # 创建模型
        model_obj = self.model_config['obj']
        model_args = self.model_config['args']
        
        self.model = dynamic_import(model_obj)(**model_args)
        self.model.load_state_dict(pretrained_state)
        self.model.to(self.device)
        self.model.eval()
        
        print(f"[INFO] 模型加载完成: {model_file}")
    
    def _load_audio(self, audio_path: str, target_sr: int = 16000) -> torch.Tensor:
        """加载音频文件"""
        wav, sr = torchaudio.load(audio_path)
        
        # 重采样
        if sr != target_sr:
            print(f"[INFO] 重采样音频: {sr}Hz -> {target_sr}Hz")
            resampler = torchaudio.transforms.Resample(sr, target_sr)
            wav = resampler(wav)
        
        # 转换为单声道
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        
        return wav
    
    def extract_embedding(self, audio_path: str) -> np.ndarray:
        """
        从音频文件中提取声纹特征
        
        Args:
            audio_path: 音频文件路径
            
        Returns:
            声纹特征向量 (numpy array)
        """
        # 确保模型已加载
        if self.model is None:
            self.load_model()
        
        # 加载音频
        wav = self._load_audio(audio_path)
        
        # 提取特征
        feat = self.feature_extractor(wav).unsqueeze(0).to(self.device)
        
        # 提取embedding
        with torch.no_grad():
            embedding = self.model(feat).detach().squeeze(0).cpu().numpy()
        
        return embedding
    
    def register_speaker(self, 
                        speaker_id: str,
                        audio_path: str,
                        speaker_name: str = None,
                        metadata: Dict = None) -> bool:
        """
        注册说话人声纹
        
        Args:
            speaker_id: 说话人唯一ID
            audio_path: 音频文件路径
            speaker_name: 说话人名称（可选）
            metadata: 额外的元数据（可选）
            
        Returns:
            是否注册成功
        """
        try:
            # 检查音频文件是否存在
            if not os.path.exists(audio_path):
                print(f"[ERROR] 音频文件不存在: {audio_path}")
                return False
            
            # 检查是否已注册
            if speaker_id in self.metadata:
                print(f"[WARNING] 说话人ID已存在: {speaker_id}")
                overwrite = input("是否覆盖已有声纹? (y/n): ").strip().lower()
                if overwrite != 'y':
                    print("[INFO] 取消注册")
                    return False
            
            print(f"[INFO] 正在提取声纹特征: {audio_path}")
            # 提取声纹特征
            embedding = self.extract_embedding(audio_path)
            
            # 保存embedding
            embedding_file = self.embeddings_dir / f"{speaker_id}.npy"
            np.save(embedding_file, embedding)
            
            # 更新元数据
            self.metadata[speaker_id] = {
                'speaker_id': speaker_id,
                'speaker_name': speaker_name or speaker_id,
                'audio_path': str(audio_path),
                'embedding_file': str(embedding_file),
                'register_time': datetime.now().isoformat(),
                'model_type': self.model_type,
                'metadata': metadata or {}
            }
            
            self._save_metadata()
            
            print(f"[SUCCESS] 声纹注册成功!")
            print(f"  - 说话人ID: {speaker_id}")
            print(f"  - 说话人名称: {speaker_name or speaker_id}")
            print(f"  - 特征维度: {embedding.shape}")
            print(f"  - 保存位置: {embedding_file}")
            
            return True
            
        except Exception as e:
            print(f"[ERROR] 注册声纹失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def unregister_speaker(self, speaker_id: str) -> bool:
        """
        注销说话人声纹
        
        Args:
            speaker_id: 说话人ID
            
        Returns:
            是否注销成功
        """
        if speaker_id not in self.metadata:
            print(f"[ERROR] 说话人ID不存在: {speaker_id}")
            return False
        
        # 删除embedding文件
        embedding_file = pathlib.Path(self.metadata[speaker_id]['embedding_file'])
        if embedding_file.exists():
            embedding_file.unlink()
        
        # 删除元数据
        del self.metadata[speaker_id]
        self._save_metadata()
        
        print(f"[SUCCESS] 声纹注销成功: {speaker_id}")
        return True
    
    def list_speakers(self, verbose: bool = True) -> List[Dict]:
        """
        列出所有已注册的说话人
        
        Args:
            verbose: 是否打印详细信息
            
        Returns:
            说话人信息列表
        """
        if not self.metadata:
            print("[INFO] 声纹库为空")
            return []
        
        speakers = []
        for speaker_id, info in self.metadata.items():
            speakers.append(info)
            
            if verbose:
                print(f"\n{'='*60}")
                print(f"说话人ID: {speaker_id}")
                print(f"说话人名称: {info['speaker_name']}")
                print(f"音频文件: {info['audio_path']}")
                print(f"注册时间: {info['register_time']}")
                print(f"模型类型: {info['model_type']}")
                if info['metadata']:
                    print(f"额外信息: {info['metadata']}")
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"[INFO] 共有 {len(speakers)} 个已注册的说话人")
        
        return speakers
    
    def identify_speaker(self, audio_path: str, threshold: float = 0.5) -> Optional[Dict]:
        """
        识别说话人
        
        Args:
            audio_path: 音频文件路径
            threshold: 相似度阈值
            
        Returns:
            识别结果 (speaker_id, speaker_name, similarity)
        """
        if not self.metadata:
            print("[WARNING] 声纹库为空，无法识别")
            return None
        
        # 提取音频特征
        print(f"[INFO] 正在识别说话人: {audio_path}")
        query_embedding = self.extract_embedding(audio_path)
        
        # 计算与所有已注册声纹的相似度
        max_similarity = -1
        best_match = None
        
        for speaker_id, info in self.metadata.items():
            embedding_file = info['embedding_file']
            registered_embedding = np.load(embedding_file)
            
            # 计算余弦相似度
            similarity = np.dot(query_embedding, registered_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(registered_embedding)
            )
            
            if similarity > max_similarity:
                max_similarity = similarity
                best_match = {
                    'speaker_id': speaker_id,
                    'speaker_name': info['speaker_name'],
                    'similarity': float(similarity)
                }
        
        if max_similarity < threshold:
            print(f"[INFO] 未找到匹配的说话人 (最高相似度: {max_similarity:.4f}, 阈值: {threshold})")
            return None
        
        print(f"[SUCCESS] 识别成功!")
        print(f"  - 说话人ID: {best_match['speaker_id']}")
        print(f"  - 说话人名称: {best_match['speaker_name']}")
        print(f"  - 相似度: {best_match['similarity']:.4f}")
        
        return best_match
    
    def batch_register(self, audio_dir: str, pattern: str = "*.wav"):
        """
        批量注册声纹
        
        Args:
            audio_dir: 音频文件目录
            pattern: 文件匹配模式
        """
        audio_dir = pathlib.Path(audio_dir)
        audio_files = list(audio_dir.glob(pattern))
        
        if not audio_files:
            print(f"[ERROR] 未找到音频文件: {audio_dir}/{pattern}")
            return
        
        print(f"[INFO] 找到 {len(audio_files)} 个音频文件")
        
        success_count = 0
        for audio_file in audio_files:
            speaker_id = audio_file.stem
            speaker_name = speaker_id
            
            print(f"\n正在处理: {audio_file.name}")
            if self.register_speaker(speaker_id, str(audio_file), speaker_name):
                success_count += 1
        
        print(f"\n[INFO] 批量注册完成: {success_count}/{len(audio_files)}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="声纹注册管理系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 注册单个声纹
  python register_voiceprint.py --action register --speaker-id user001 --audio path/to/audio.wav --speaker-name "张三"
  
  # 批量注册声纹
  python register_voiceprint.py --action batch-register --audio-dir path/to/audio_dir
  
  # 识别说话人
  python register_voiceprint.py --action identify --audio path/to/test.wav
  
  # 列出所有已注册的说话人
  python register_voiceprint.py --action list
  
  # 注销说话人
  python register_voiceprint.py --action unregister --speaker-id user001
        """
    )
    
    parser.add_argument('--action', type=str, required=True,
                       choices=['register', 'unregister', 'list', 'identify', 'batch-register'],
                       help='操作类型')
    
    parser.add_argument('--speaker-id', type=str,
                       help='说话人ID')
    
    parser.add_argument('--speaker-name', type=str,
                       help='说话人名称')
    
    parser.add_argument('--audio', type=str,
                       help='音频文件路径')
    
    parser.add_argument('--audio-dir', type=str,
                       help='音频文件目录（用于批量注册）')
    
    parser.add_argument('--pattern', type=str, default='*.wav',
                       help='音频文件匹配模式（用于批量注册）')
    
    parser.add_argument('--voiceprint-dir', type=str, default='./voiceprint_db',
                       help='声纹数据库目录')
    
    parser.add_argument('--model-dir', type=str, default='./pretrained_models',
                       help='预训练模型目录')
    
    parser.add_argument('--model-type', type=str, default='eres2net',
                       choices=['campplus', 'eres2net', 'eres2netv2'],
                       help='模型类型')
    
    parser.add_argument('--device', type=str, default=None,
                       choices=['cuda', 'cpu'],
                       help='计算设备')
    
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='识别阈值（0-1之间）')
    
    args = parser.parse_args()
    
    # 创建声纹管理器
    manager = VoiceprintManager(
        voiceprint_dir=args.voiceprint_dir,
        model_dir=args.model_dir,
        model_type=args.model_type,
        device=args.device
    )
    
    # 执行操作
    if args.action == 'register':
        if not args.speaker_id or not args.audio:
            print("[ERROR] 注册声纹需要提供 --speaker-id 和 --audio 参数")
            return
        
        manager.register_speaker(
            speaker_id=args.speaker_id,
            audio_path=args.audio,
            speaker_name=args.speaker_name
        )
    
    elif args.action == 'batch-register':
        if not args.audio_dir:
            print("[ERROR] 批量注册需要提供 --audio-dir 参数")
            return
        
        manager.batch_register(args.audio_dir, args.pattern)
    
    elif args.action == 'unregister':
        if not args.speaker_id:
            print("[ERROR] 注销声纹需要提供 --speaker-id 参数")
            return
        
        manager.unregister_speaker(args.speaker_id)
    
    elif args.action == 'list':
        manager.list_speakers(verbose=True)
    
    elif args.action == 'identify':
        if not args.audio:
            print("[ERROR] 识别说话人需要提供 --audio 参数")
            return
        
        manager.identify_speaker(args.audio, threshold=args.threshold)


if __name__ == '__main__':
    main()

