#!/usr/bin/env python3
# coding=utf-8
"""
声纹测试诊断脚本
用于测试注册的声纹和测试音频的相似度
"""

import sys
import argparse
from register_voiceprint import VoiceprintManager

def test_similarity(audio_path: str, 
                   voiceprint_dir: str = "./voiceprint_db",
                   model_dir: str = "./pretrained_models"):
    """
    测试音频与所有已注册声纹的相似度
    """
    print("="*80)
    print("声纹相似度诊断测试")
    print("="*80)
    
    # 创建声纹管理器
    print("\n[1/3] 初始化声纹管理器...")
    manager = VoiceprintManager(
        voiceprint_dir=voiceprint_dir,
        model_dir=model_dir,
        model_type="eres2net",
        device="cpu"
    )
    
    # 检查声纹库
    print("\n[2/3] 检查声纹库...")
    speakers = manager.list_speakers(verbose=False)
    
    if not speakers:
        print("❌ 声纹库为空！请先注册声纹。")
        return
    
    print(f"✓ 找到 {len(speakers)} 个已注册的声纹：")
    for speaker in speakers:
        print(f"  - {speaker['speaker_id']}: {speaker['speaker_name']}")
    
    # 提取测试音频特征
    print(f"\n[3/3] 测试音频: {audio_path}")
    print("正在提取特征...")
    
    try:
        import numpy as np
        query_embedding = manager.extract_embedding(audio_path)
        print(f"✓ 特征提取完成，维度: {query_embedding.shape}")
        
        # 计算与每个声纹的相似度
        print("\n" + "="*80)
        print("相似度测试结果：")
        print("="*80)
        
        similarities = []
        for speaker in speakers:
            speaker_id = speaker['speaker_id']
            embedding_file = speaker['embedding_file']
            registered_embedding = np.load(embedding_file)
            
            # 计算余弦相似度
            similarity = np.dot(query_embedding, registered_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(registered_embedding)
            )
            
            similarities.append({
                'speaker_id': speaker_id,
                'speaker_name': speaker['speaker_name'],
                'similarity': similarity
            })
            
            print(f"\n说话人: {speaker['speaker_name']} (ID: {speaker_id})")
            print(f"相似度: {similarity:.6f} ({similarity*100:.2f}%)")
            
            # 给出判断建议
            if similarity >= 0.7:
                print("判断: 🟢 非常匹配")
            elif similarity >= 0.5:
                print("判断: 🟡 可能匹配")
            elif similarity >= 0.3:
                print("判断: 🟠 弱匹配")
            else:
                print("判断: 🔴 不匹配")
        
        # 找出最佳匹配
        best_match = max(similarities, key=lambda x: x['similarity'])
        
        print("\n" + "="*80)
        print("最佳匹配结果：")
        print("="*80)
        print(f"说话人: {best_match['speaker_name']}")
        print(f"相似度: {best_match['similarity']:.6f} ({best_match['similarity']*100:.2f}%)")
        
        # 推荐阈值
        print("\n" + "="*80)
        print("阈值建议：")
        print("="*80)
        
        max_similarity = best_match['similarity']
        
        if max_similarity >= 0.7:
            print(f"✅ 当前最高相似度 {max_similarity:.3f} >= 0.7")
            print("   建议阈值: 0.6 或 0.7 (严格模式)")
        elif max_similarity >= 0.5:
            print(f"⚠️  当前最高相似度 {max_similarity:.3f} 在 0.5-0.7 之间")
            print("   建议阈值: 0.4 或 0.5 (适中)")
        elif max_similarity >= 0.3:
            print(f"⚠️  当前最高相似度 {max_similarity:.3f} 较低 (0.3-0.5)")
            print("   建议阈值: 0.3 (宽松)")
            print("   💡 提示: 建议使用更高质量的音频重新注册")
        else:
            print(f"❌ 当前最高相似度 {max_similarity:.3f} < 0.3")
            print("   建议:")
            print("   1. 使用相同说话人的音频重新注册")
            print("   2. 确保音频质量良好（清晰、无噪音）")
            print("   3. 使用3-10秒的纯净人声")
        
        # 配置文件修改建议
        print("\n" + "="*80)
        print("配置文件修改建议：")
        print("="*80)
        
        if max_similarity < 0.5:
            recommended_threshold = max(0.3, max_similarity - 0.05)
            print(f"编辑 config_3dspeaker.yaml，将阈值改为 {recommended_threshold:.2f}:")
            print("\nspeaker_verification:")
            print(f"  threshold: {recommended_threshold:.2f}  # 当前是 0.5")
        else:
            print("当前阈值 0.5 应该可以正常识别")
            print("如果仍然无法识别，可以尝试降低到 0.4")
        
        print("\n" + "="*80)
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(
        description="声纹相似度测试工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 测试音频与已注册声纹的相似度
  python test_voiceprint.py --audio test.wav
  
  # 指定声纹库目录
  python test_voiceprint.py --audio test.wav --voiceprint-dir ./voiceprint_db
        """
    )
    
    parser.add_argument('--audio', type=str, required=True,
                       help='测试音频文件路径')
    
    parser.add_argument('--voiceprint-dir', type=str, default='./voiceprint_db',
                       help='声纹数据库目录')
    
    parser.add_argument('--model-dir', type=str, default='./pretrained_models',
                       help='模型目录')
    
    args = parser.parse_args()
    
    test_similarity(
        audio_path=args.audio,
        voiceprint_dir=args.voiceprint_dir,
        model_dir=args.model_dir
    )


if __name__ == '__main__':
    main()

