#!/usr/bin/env python3
# coding=utf-8
"""
集成3D-Speaker和SenseVoice的Web UI界面
功能：
1. 语音识别界面
2. 说话人识别界面
3. 声纹注册管理界面
4. 系统配置界面
"""

import os
import sys
import gradio as gr
import yaml
from pathlib import Path
from typing import Dict, Tuple, List

# 导入自定义模块
from integrated_asr import IntegratedASRSystem
from register_voiceprint import VoiceprintManager


# 加载配置
def load_config(config_file: str = "config_3dspeaker.yaml") -> Dict:
    """加载配置文件"""
    if os.path.exists(config_file):
        with open(config_file, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    return {}


# 全局配置和系统实例
CONFIG = load_config()
SYSTEM = None
SPEAKER_MANAGER = None


def initialize_system(enable_speaker: bool):
    """初始化系统"""
    global SYSTEM, SPEAKER_MANAGER
    
    if SYSTEM is None:
        device = CONFIG.get('system', {}).get('device', 'cpu')  # 默认CPU
        
        SYSTEM = IntegratedASRSystem(
            enable_speaker_verification=enable_speaker,
            voiceprint_dir=CONFIG.get('speaker_verification', {}).get('voiceprint_dir', './voiceprint_db'),
            speaker_model_dir=CONFIG.get('speaker_verification', {}).get('model_dir', './pretrained_models'),
            speaker_model_type=CONFIG.get('speaker_verification', {}).get('model_type', 'eres2net'),
            speaker_threshold=CONFIG.get('speaker_verification', {}).get('threshold', 0.5),
            sensevoice_model=CONFIG.get('sensevoice', {}).get('model', 'iic/SenseVoiceSmall'),
            vad_model=CONFIG.get('sensevoice', {}).get('vad_model', 'iic/speech_fsmn_vad_zh-cn-16k-common-pytorch'),
            device=device,
        )
        
        if enable_speaker:
            SPEAKER_MANAGER = SYSTEM.speaker_manager
    
    return SYSTEM


def asr_inference(audio_input, 
                 language: str,
                 enable_speaker: bool,
                 use_itn: bool) -> Tuple[str, str]:
    """
    语音识别推理
    
    Returns:
        (识别文本, 说话人详细信息)
    """
    if audio_input is None:
        return "请上传音频文件", ""
    
    # 初始化系统
    system = initialize_system(enable_speaker)
    
    # 如果输入是元组（采样率，音频数据），保存为临时文件
    if isinstance(audio_input, tuple):
        import tempfile
        import soundfile as sf
        
        sr, audio_data = audio_input
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            sf.write(tmp.name, audio_data, sr)
            audio_path = tmp.name
    else:
        audio_path = audio_input
    
    # 处理音频（获取详细的说话人信息）
    if enable_speaker:
        # 使用详细识别模式
        speaker_result = system.identify_speaker(audio_path, return_all_scores=True)
    else:
        speaker_result = None
    
    # 执行语音识别
    asr_result = system.recognize_speech(
        audio_path=audio_path,
        language=language,
        use_itn=use_itn,
    )
    
    # 提取识别文本
    text_results = []
    if asr_result:
        for item in asr_result:
            text_results.append(item['text'])
    
    text = "\n".join(text_results) if text_results else "未识别到语音"
    
    # 构建详细的说话人信息
    speaker_info = ""
    if enable_speaker and speaker_result:
        speaker_info = format_speaker_info(speaker_result)
    elif enable_speaker:
        speaker_info = "未识别到注册的说话人（声纹库可能为空）"
    
    # 清理临时文件
    if isinstance(audio_input, tuple):
        try:
            os.unlink(audio_path)
        except:
            pass
    
    return text, speaker_info


def format_speaker_info(speaker_result: dict) -> str:
    """格式化说话人识别结果，显示详细信息"""
    
    if speaker_result.get('matched'):
        # 识别成功
        info = f"✅ 识别成功！\n\n"
        info += f"说话人: {speaker_result['speaker_name']}\n"
        info += f"ID: {speaker_result['speaker_id']}\n"
        info += f"相似度: {speaker_result['similarity']:.4f} ({speaker_result['similarity']*100:.2f}%)\n"
        info += f"阈值: {speaker_result['threshold']:.4f}\n"
        
        # 显示所有相似度
        all_scores = speaker_result.get('all_scores', [])
        if len(all_scores) > 1:
            info += f"\n{'='*40}\n"
            info += f"所有说话人相似度排名：\n"
            info += f"{'='*40}\n"
            for i, score in enumerate(all_scores, 1):
                status = "✓" if i == 1 else " "
                percentage = score['similarity'] * 100
                info += f"{status} {i}. {score['speaker_name']}: {score['similarity']:.4f} ({percentage:.2f}%)\n"
        
        return info
    
    else:
        # 识别失败
        info = f"❌ 未识别到注册的说话人\n\n"
        
        best_match = speaker_result.get('best_match')
        if best_match:
            info += f"最接近的说话人:\n"
            info += f"  名称: {best_match['speaker_name']}\n"
            info += f"  相似度: {best_match['similarity']:.4f} ({best_match['similarity']*100:.2f}%)\n"
            info += f"  阈值要求: {speaker_result['threshold']:.4f}\n"
            info += f"  差距: {(speaker_result['threshold'] - best_match['similarity']):.4f}\n\n"
            
            # 给出建议
            info += f"💡 建议:\n"
            if best_match['similarity'] >= 0.3:
                suggested_threshold = max(0.25, best_match['similarity'] - 0.05)
                info += f"  • 当前相似度 {best_match['similarity']:.3f} 接近阈值\n"
                info += f"  • 可以尝试降低阈值到 {suggested_threshold:.2f}\n"
                info += f"  • 或使用更清晰的音频重新注册\n"
            else:
                info += f"  • 相似度较低，建议重新注册声纹\n"
                info += f"  • 使用3-10秒的纯净人声\n"
                info += f"  • 确保录音环境安静\n"
        
        # 显示所有相似度
        all_scores = speaker_result.get('all_scores', [])
        if all_scores:
            info += f"\n{'='*40}\n"
            info += f"所有说话人相似度排名：\n"
            info += f"{'='*40}\n"
            for i, score in enumerate(all_scores, 1):
                percentage = score['similarity'] * 100
                info += f"{i}. {score['speaker_name']}: {score['similarity']:.4f} ({percentage:.2f}%)\n"
        
        return info



def register_speaker_ui(speaker_id: str,
                       speaker_name: str,
                       audio_input) -> tuple:
    """注册说话人声纹，返回(注册结果, 更新后的列表)"""
    global SPEAKER_MANAGER
    
    if not speaker_id:
        return "❌ 请输入说话人ID", ""
    
    if audio_input is None:
        return "❌ 请上传音频文件", ""
    
    # 初始化声纹管理器
    if SPEAKER_MANAGER is None:
        device = CONFIG.get('system', {}).get('device', 'cpu')  # 默认CPU
        SPEAKER_MANAGER = VoiceprintManager(
            voiceprint_dir=CONFIG.get('speaker_verification', {}).get('voiceprint_dir', './voiceprint_db'),
            model_dir=CONFIG.get('speaker_verification', {}).get('model_dir', './pretrained_models'),
            model_type=CONFIG.get('speaker_verification', {}).get('model_type', 'eres2net'),
            device=device,
        )
    
    # 处理音频输入
    if isinstance(audio_input, tuple):
        import tempfile
        import soundfile as sf
        
        sr, audio_data = audio_input
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            sf.write(tmp.name, audio_data, sr)
            audio_path = tmp.name
    else:
        audio_path = audio_input
    
    # 注册声纹
    success = SPEAKER_MANAGER.register_speaker(
        speaker_id=speaker_id,
        audio_path=audio_path,
        speaker_name=speaker_name or speaker_id,
    )
    
    # 清理临时文件
    if isinstance(audio_input, tuple):
        try:
            os.unlink(audio_path)
        except:
            pass
    
    # 获取更新后的列表
    updated_list = list_speakers_ui()
    
    if success:
        result_msg = f"✓ 声纹注册成功!\n说话人ID: {speaker_id}\n说话人名称: {speaker_name or speaker_id}\n\n自动刷新列表如下："
        return result_msg, updated_list
    else:
        return "❌ 声纹注册失败，请查看控制台日志", updated_list


def list_speakers_ui() -> str:
    """列出所有已注册的说话人"""
    global SPEAKER_MANAGER
    
    if SPEAKER_MANAGER is None:
        device = CONFIG.get('system', {}).get('device', 'cpu')
        SPEAKER_MANAGER = VoiceprintManager(
            voiceprint_dir=CONFIG.get('speaker_verification', {}).get('voiceprint_dir', './voiceprint_db'),
            model_dir=CONFIG.get('speaker_verification', {}).get('model_dir', './pretrained_models'),
            model_type=CONFIG.get('speaker_verification', {}).get('model_type', 'eres2net'),
            device=device,
        )
    
    speakers = SPEAKER_MANAGER.list_speakers(verbose=False)
    
    if not speakers:
        return "声纹库为空"
    
    result = f"共有 {len(speakers)} 个已注册的说话人:\n\n"
    for i, info in enumerate(speakers, 1):
        result += f"{i}. ID: {info['speaker_id']}\n"
        result += f"   名称: {info['speaker_name']}\n"
        result += f"   注册时间: {info['register_time']}\n"
        result += f"   模型: {info['model_type']}\n\n"
    
    return result


def delete_speaker_ui(speaker_id: str) -> str:
    """删除说话人声纹，返回更新后的列表"""
    global SPEAKER_MANAGER
    
    if not speaker_id:
        return "❌ 请输入说话人ID"
    
    if SPEAKER_MANAGER is None:
        device = CONFIG.get('system', {}).get('device', 'cpu')
        SPEAKER_MANAGER = VoiceprintManager(
            voiceprint_dir=CONFIG.get('speaker_verification', {}).get('voiceprint_dir', './voiceprint_db'),
            model_dir=CONFIG.get('speaker_verification', {}).get('model_dir', './pretrained_models'),
            model_type=CONFIG.get('speaker_verification', {}).get('model_type', 'eres2net'),
            device=device,
        )
    
    success = SPEAKER_MANAGER.unregister_speaker(speaker_id)
    
    # 获取更新后的列表
    updated_list = list_speakers_ui()
    
    if success:
        result_msg = f"✓ 成功删除说话人: {speaker_id}\n\n更新后的列表：\n{updated_list}"
        return result_msg
    else:
        return f"❌ 删除失败，说话人ID不存在: {speaker_id}\n\n当前列表：\n{updated_list}"


def create_webui():
    """创建Web UI"""
    
    # HTML标题
    html_title = """
    <div style="text-align: center;">
        <h1>🎙️ SenseVoice + 3D-Speaker 集成系统</h1>
        <p style="font-size: 16px;">
            支持多语言语音识别、情感识别、事件检测和说话人识别
        </p>
    </div>
    """
    
    # 系统信息
    html_info = """
    <div style="background-color: #f0f0f0; padding: 15px; border-radius: 5px; margin: 10px 0;">
        <h3>📋 系统功能</h3>
        <ul>
            <li><b>语音识别</b>：支持中文、英文、粤语、日语、韩语</li>
            <li><b>情感识别</b>：识别开心😊、悲伤😔、愤怒😡等情绪</li>
            <li><b>事件检测</b>：检测音乐🎼、掌声👏、笑声😀等声音事件</li>
            <li><b>说话人识别</b>：识别注册过的说话人身份（可选功能）</li>
        </ul>
        <h3>🔧 使用说明</h3>
        <ul>
            <li>在<b>"语音识别"</b>标签页进行语音转文字</li>
            <li>在<b>"声纹管理"</b>标签页注册和管理说话人声纹</li>
            <li>勾选<b>"启用说话人识别"</b>可同时识别说话人身份</li>
        </ul>
    </div>
    """
    
    with gr.Blocks(theme=gr.themes.Soft(), title="SenseVoice + 3D-Speaker") as demo:
        gr.HTML(html_title)
        gr.HTML(html_info)
        
        with gr.Tabs():
            # 标签页1：语音识别
            with gr.TabItem("🎯 语音识别"):
                with gr.Row():
                    with gr.Column():
                        audio_input = gr.Audio(
                            label="上传音频或使用麦克风录音",
                            type="filepath"
                        )
                        
                        with gr.Accordion("⚙️ 配置选项", open=True):
                            language_dropdown = gr.Dropdown(
                                choices=["auto", "zh", "en", "yue", "ja", "ko"],
                                value="auto",
                                label="语言 (auto=自动识别)"
                            )
                            
                            enable_speaker_check = gr.Checkbox(
                                label="启用说话人识别 (需要先在'声纹管理'中注册说话人)",
                                value=False
                            )
                            
                            use_itn_check = gr.Checkbox(
                                label="使用文本规范化（添加标点等）",
                                value=True
                            )
                        
                        recognize_btn = gr.Button("🎙️ 开始识别", variant="primary")
                    
                    with gr.Column():
                        text_output = gr.Textbox(
                            label="识别结果",
                            lines=10,
                            placeholder="识别的文本将显示在这里..."
                        )
                        
                        speaker_output = gr.Textbox(
                            label="说话人信息",
                            lines=5,
                            placeholder="说话人识别信息（如果启用）..."
                        )
                
                recognize_btn.click(
                    fn=asr_inference,
                    inputs=[audio_input, language_dropdown, enable_speaker_check, use_itn_check],
                    outputs=[text_output, speaker_output]
                )
                
                # 添加示例
                gr.Examples(
                    examples=[
                        ["example/zh.mp3", "zh", False, True],
                        ["example/en.mp3", "en", False, True],
                        ["example/yue.mp3", "yue", False, True],
                    ],
                    inputs=[audio_input, language_dropdown, enable_speaker_check, use_itn_check],
                    label="📝 示例音频"
                )
            
            # 标签页2：声纹管理
            with gr.TabItem("👤 声纹管理"):
                gr.Markdown("## 注册说话人声纹")
                
                with gr.Row():
                    with gr.Column():
                        speaker_id_input = gr.Textbox(
                            label="说话人ID (唯一标识，不可重复)",
                            placeholder="例如: user001"
                        )
                        
                        speaker_name_input = gr.Textbox(
                            label="说话人名称 (可选，便于识别)",
                            placeholder="例如: 张三"
                        )
                        
                        register_audio_input = gr.Audio(
                            label="上传声纹音频 (建议：3-10秒纯净人声)",
                            type="filepath"
                        )
                        
                        register_btn = gr.Button("✨ 注册声纹", variant="primary")
                    
                    with gr.Column():
                        register_output = gr.Textbox(
                            label="注册结果",
                            lines=5
                        )
                        
                        register_list_output = gr.Textbox(
                            label="已注册的声纹列表",
                            lines=10
                        )
                
                register_btn.click(
                    fn=register_speaker_ui,
                    inputs=[speaker_id_input, speaker_name_input, register_audio_input],
                    outputs=[register_output, register_list_output]
                )
                
                gr.Markdown("---")
                gr.Markdown("## 管理已注册的声纹")
                
                with gr.Row():
                    with gr.Column():
                        list_btn = gr.Button("📋 查看所有声纹")
                        delete_speaker_id = gr.Textbox(
                            label="要删除的说话人ID",
                            placeholder="例如: user001"
                        )
                        delete_btn = gr.Button("🗑️ 删除声纹", variant="stop")
                    
                    with gr.Column():
                        management_output = gr.Textbox(
                            label="操作结果",
                            lines=15
                        )
                
                list_btn.click(
                    fn=list_speakers_ui,
                    inputs=[],
                    outputs=management_output
                )
                
                delete_btn.click(
                    fn=delete_speaker_ui,
                    inputs=delete_speaker_id,
                    outputs=management_output
                )
            
            # 标签页3：使用说明
            with gr.TabItem("📖 使用说明"):
                gr.Markdown("""
                # 完整使用指南
                
                ## 1️⃣ 基础语音识别（不使用说话人识别）
                
                ### 步骤：
                1. 在"语音识别"标签页上传音频或使用麦克风录音
                2. 选择语言（建议使用"auto"自动识别）
                3. 点击"开始识别"按钮
                4. 查看识别结果
                
                ### 支持的功能：
                - ✅ 多语言识别：中文、英文、粤语、日语、韩语
                - ✅ 情感识别：😊开心 😔悲伤 😡愤怒 😰害怕 🤢厌恶 😮惊讶
                - ✅ 事件检测：🎼音乐 👏掌声 😀笑声 😭哭声 🤧咳嗽/喷嚏
                
                ---
                
                ## 2️⃣ 使用说话人识别功能
                
                ### 第一步：注册声纹
                
                1. 切换到"声纹管理"标签页
                2. 填写说话人ID（例如：user001）
                3. 填写说话人名称（例如：张三）
                4. 上传该说话人的音频文件（建议3-10秒纯净人声）
                5. 点击"注册声纹"按钮
                6. 等待注册完成
                
                **注意事项：**
                - 说话人ID必须唯一，不可重复
                - 音频质量越好，识别准确率越高
                - 建议使用安静环境录制的音频
                - 首次运行需要下载模型，请耐心等待
                
                ### 第二步：识别说话人
                
                1. 回到"语音识别"标签页
                2. 勾选"启用说话人识别"选项
                3. 上传要识别的音频
                4. 点击"开始识别"
                5. 同时查看语音识别结果和说话人识别结果
                
                ### 第三步：管理声纹库
                
                在"声纹管理"标签页可以：
                - 📋 查看所有已注册的说话人
                - 🗑️ 删除不需要的声纹
                
                ---
                
                ## 3️⃣ 命令行使用
                
                ### 基础语音识别
                ```bash
                python integrated_asr.py --audio test.wav
                ```
                
                ### 启用说话人识别
                ```bash
                python integrated_asr.py --audio test.wav --enable-speaker
                ```
                
                ### 批量处理
                ```bash
                python integrated_asr.py --audio-list file1.wav file2.wav file3.wav --enable-speaker --output results.json
                ```
                
                ### 注册声纹
                ```bash
                python register_voiceprint.py --action register --speaker-id user001 --speaker-name "张三" --audio voice.wav
                ```
                
                ### 批量注册声纹
                ```bash
                python register_voiceprint.py --action batch-register --audio-dir ./voice_samples
                ```
                
                ### 识别说话人
                ```bash
                python register_voiceprint.py --action identify --audio test.wav
                ```
                
                ### 列出所有声纹
                ```bash
                python register_voiceprint.py --action list
                ```
                
                ---
                
                ## 4️⃣ 高级配置
                
                编辑 `config_3dspeaker.yaml` 文件可以调整：
                
                - 说话人识别阈值（threshold）：越高要求越严格
                - 模型类型（model_type）：campplus, eres2net, eres2netv2
                - 计算设备（device）：cuda:0 或 cpu
                - 其他参数...
                
                ---
                
                ## 5️⃣ 常见问题
                
                ### Q: 首次运行很慢？
                A: 首次运行需要下载模型文件，之后会直接使用缓存的模型。
                
                ### Q: 说话人识别准确率不高？
                A: 建议：
                - 使用更长的音频注册（5-10秒）
                - 确保音频质量好，背景噪音少
                - 调高识别阈值（在配置文件中修改threshold）
                
                ### Q: 支持实时识别吗？
                A: 当前版本主要支持文件输入，实时流式识别功能开发中。
                
                ### Q: GPU内存不够？
                A: 可以在配置文件中设置 device: "cpu" 使用CPU进行推理。
                
                ---
                
                ## 6️⃣ 技术支持
                
                - GitHub Issues: 在项目页面提交问题
                - 钉钉交流群: 扫描README中的二维码
                
                """)
        
        return demo


def launch_webui():
    """启动Web UI"""
    config = CONFIG.get('webui', {})
    
    demo = create_webui()
    demo.launch(
        server_name=config.get('server_name', '0.0.0.0'),
        server_port=config.get('port', 7860),
        share=config.get('share', False),
        inbrowser=True,
    )


if __name__ == '__main__':
    launch_webui()

