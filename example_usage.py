#!/usr/bin/env python3
# coding=utf-8
"""
使用示例脚本
演示如何使用集成的3D-Speaker + SenseVoice系统
"""

from integrated_asr import IntegratedASRSystem
from register_voiceprint import VoiceprintManager


def example_1_basic_asr():
    """示例1: 基础语音识别（不使用说话人识别）"""
    print("\n" + "="*80)
    print("示例1: 基础语音识别")
    print("="*80)
    
    # 创建系统（关闭说话人识别）
    system = IntegratedASRSystem(enable_speaker_verification=False)
    
    # 处理音频
    audio_file = "example/zh.mp3"  # 使用项目自带的示例音频
    result = system.process_audio(audio_file, language="zh")
    
    # 显示结果
    print("\n识别结果：")
    for item in result['asr_results']:
        print(f"  文本: {item['text']}")
    print(f"  处理时间: {result['processing_time']:.2f}秒")


def example_2_register_speaker():
    """示例2: 注册说话人声纹"""
    print("\n" + "="*80)
    print("示例2: 注册说话人声纹")
    print("="*80)
    
    # 创建声纹管理器
    manager = VoiceprintManager(
        voiceprint_dir="./voiceprint_db",
        model_type="eres2net"
    )
    
    # 注册声纹
    # 注意：这里使用示例音频，实际使用时应该用真实的人声录音
    success = manager.register_speaker(
        speaker_id="example_user",
        audio_path="example/zh.mp3",
        speaker_name="示例用户"
    )
    
    if success:
        print("\n✓ 声纹注册成功！")
        
        # 列出所有已注册的声纹
        print("\n当前声纹库：")
        speakers = manager.list_speakers(verbose=False)
        for speaker in speakers:
            print(f"  - {speaker['speaker_id']}: {speaker['speaker_name']}")
    else:
        print("\n✗ 声纹注册失败")


def example_3_asr_with_speaker():
    """示例3: 语音识别 + 说话人识别"""
    print("\n" + "="*80)
    print("示例3: 语音识别 + 说话人识别")
    print("="*80)
    
    # 创建系统（启用说话人识别）
    system = IntegratedASRSystem(enable_speaker_verification=True)
    
    # 处理音频
    audio_file = "example/zh.mp3"
    result = system.process_audio(audio_file, language="zh")
    
    # 显示结果
    print("\n识别结果：")
    if result['speaker_info']:
        print(f"  说话人: {result['speaker_info']['speaker_name']}")
        print(f"  相似度: {result['speaker_info']['similarity']:.4f}")
    else:
        print("  说话人: 未识别到注册的说话人")
    
    for item in result['asr_results']:
        print(f"  文本: {item['text']}")
    
    print(f"  处理时间: {result['processing_time']:.2f}秒")


def example_4_toggle_feature():
    """示例4: 动态切换说话人识别功能"""
    print("\n" + "="*80)
    print("示例4: 动态切换说话人识别功能")
    print("="*80)
    
    # 创建系统
    system = IntegratedASRSystem(enable_speaker_verification=False)
    
    audio_file = "example/zh.mp3"
    
    # 第一次：不使用说话人识别
    print("\n[1] 不使用说话人识别：")
    result1 = system.process_audio(audio_file)
    print(f"  文本: {result1['asr_results'][0]['text']}")
    print(f"  说话人信息: {result1['speaker_info']}")
    
    # 启用说话人识别
    system.toggle_speaker_verification(True)
    
    # 第二次：使用说话人识别
    print("\n[2] 启用说话人识别：")
    result2 = system.process_audio(audio_file)
    print(f"  文本: {result2['asr_results'][0]['text']}")
    if result2['speaker_info']:
        print(f"  说话人: {result2['speaker_info']['speaker_name']}")
    else:
        print(f"  说话人: 未识别")


def example_5_batch_process():
    """示例5: 批量处理多个音频文件"""
    print("\n" + "="*80)
    print("示例5: 批量处理")
    print("="*80)
    
    # 创建系统
    system = IntegratedASRSystem(enable_speaker_verification=False)
    
    # 批量处理
    audio_list = [
        "example/zh.mp3",
        "example/en.mp3",
        "example/yue.mp3",
    ]
    
    results = system.batch_process(
        audio_list=audio_list,
        language="auto",
        output_file="batch_results.json"
    )
    
    print(f"\n处理完成！共处理 {len(results)} 个文件")
    print("结果已保存到: batch_results.json")


def example_6_different_languages():
    """示例6: 多语言识别"""
    print("\n" + "="*80)
    print("示例6: 多语言识别")
    print("="*80)
    
    # 创建系统
    system = IntegratedASRSystem(enable_speaker_verification=False)
    
    # 测试不同语言
    languages = [
        ("example/zh.mp3", "zh", "中文"),
        ("example/en.mp3", "en", "英文"),
        ("example/yue.mp3", "yue", "粤语"),
    ]
    
    for audio_file, lang, lang_name in languages:
        print(f"\n{lang_name}识别：")
        result = system.process_audio(audio_file, language=lang)
        if result['asr_results']:
            print(f"  {result['asr_results'][0]['text']}")


def example_7_system_info():
    """示例7: 获取系统信息"""
    print("\n" + "="*80)
    print("示例7: 系统信息")
    print("="*80)
    
    # 创建系统
    system = IntegratedASRSystem(enable_speaker_verification=True)
    
    # 获取系统信息
    info = system.get_system_info()
    
    print("\n系统配置：")
    print(f"  说话人识别: {'启用' if info['speaker_verification_enabled'] else '禁用'}")
    print(f"  识别阈值: {info['speaker_threshold']}")
    print(f"  计算设备: {info['device']}")
    print(f"  已注册说话人数: {info['registered_speakers_count']}")
    
    if info['registered_speakers']:
        print("\n已注册的说话人：")
        for speaker_id in info['registered_speakers']:
            print(f"  - {speaker_id}")


def main():
    """主函数"""
    print("\n" + "="*80)
    print("  SenseVoice + 3D-Speaker 集成系统使用示例")
    print("="*80)
    print("\n本脚本演示如何使用集成系统的各种功能")
    print("请按照提示运行示例\n")
    
    examples = [
        ("1", "基础语音识别", example_1_basic_asr),
        ("2", "注册说话人声纹", example_2_register_speaker),
        ("3", "语音识别 + 说话人识别", example_3_asr_with_speaker),
        ("4", "动态切换功能", example_4_toggle_feature),
        ("5", "批量处理", example_5_batch_process),
        ("6", "多语言识别", example_6_different_languages),
        ("7", "查看系统信息", example_7_system_info),
        ("0", "运行所有示例", None),
    ]
    
    # 显示菜单
    print("请选择要运行的示例：")
    for num, desc, _ in examples:
        print(f"  [{num}] {desc}")
    
    choice = input("\n请输入选项 (0-7): ").strip()
    
    if choice == "0":
        # 运行所有示例
        for num, desc, func in examples:
            if func is not None:
                try:
                    func()
                    input("\n按回车继续下一个示例...")
                except Exception as e:
                    print(f"\n错误: {e}")
                    import traceback
                    traceback.print_exc()
    else:
        # 运行指定示例
        for num, desc, func in examples:
            if num == choice and func is not None:
                try:
                    func()
                except Exception as e:
                    print(f"\n错误: {e}")
                    import traceback
                    traceback.print_exc()
                break
        else:
            print("无效的选项！")
    
    print("\n" + "="*80)
    print("示例运行完成！")
    print("="*80)
    print("\n更多使用方法请参考:")
    print("  - README_3DSpeaker.md (详细文档)")
    print("  - 快速使用指南.md (快速上手)")
    print("  - config_3dspeaker.yaml (配置说明)")
    print("\n启动Web界面: python webui_integrated.py")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()

