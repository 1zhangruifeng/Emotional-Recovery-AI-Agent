import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from inference import OmniInference
import sounddevice as sd
import soundfile as sf
import tempfile
import numpy as np
import torch


def record_audio(duration=5, sample_rate=16000):
    """录制音频"""
    print(f"\n🎤 Recording for {duration} seconds...")
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype=np.float32)
    sd.wait()
    print("✅ Recording complete!")

    temp_path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    sf.write(temp_path, recording, sample_rate)
    return temp_path


def play_audio(audio_data, samplerate=24000):
    """播放音频"""
    # 确保是1D数组
    if isinstance(audio_data, torch.Tensor):
        audio_data = audio_data.cpu().numpy()

    if audio_data.ndim > 1:
        audio_data = audio_data.squeeze()

    # 归一化
    if np.abs(audio_data).max() > 0:
        audio_data = audio_data / np.abs(audio_data).max()

    sd.play(audio_data, samplerate)
    sd.wait()


def main():
    print("=" * 50)
    print("🎙️ Mini-Omni CLI Voice Assistant")
    print("=" * 50)

    print("Loading model...")
    model = OmniInference(ckpt_dir='./checkpoint', device='cuda:0')
    print("✅ Model ready!\n")

    response_count = 0

    while True:
        print("\n" + "-" * 30)
        print("1. Speak to AI (record 5 seconds)")
        print("2. Exit")
        choice = input("Choose option: ")

        if choice == "1":
            audio_path = record_audio(duration=5)
            print("🤔 AI is thinking...")

            # 收集响应 - 直接保存整个响应
            output_dir = "./output"
            os.makedirs(output_dir, exist_ok=True)

            # 使用模型的现有方法生成完整音频
            try:
                # 方法1：直接使用 run_AT_batch_stream 并收集
                audio_chunks = []
                for i, chunk in enumerate(model.run_AT_batch_stream(audio_path)):
                    print(
                        f"  Received chunk {i + 1}, type: {type(chunk)}, shape: {np.array(chunk).shape if hasattr(chunk, '__len__') else 'scalar'}")

                    # 转换 chunk 为 numpy 数组
                    if isinstance(chunk, torch.Tensor):
                        chunk = chunk.cpu().numpy()

                    chunk = np.array(chunk)

                    # 过滤有效的音频块（应该有多个值）
                    if chunk.size > 100:  # 音频块应该至少有100个样本
                        audio_chunks.append(chunk)
                        print(f"    Added chunk with {chunk.size} samples")

                if audio_chunks:
                    # 打印每个块的信息
                    print(f"\nTotal chunks: {len(audio_chunks)}")
                    for i, chunk in enumerate(audio_chunks):
                        print(f"  Chunk {i}: shape {chunk.shape}, dtype {chunk.dtype}")

                    # 尝试合并
                    try:
                        # 确保所有块都是一维
                        flat_chunks = []
                        for chunk in audio_chunks:
                            if chunk.ndim > 1:
                                chunk = chunk.flatten()
                            flat_chunks.append(chunk)

                        full_audio = np.concatenate(flat_chunks)
                        print(f"Full audio shape: {full_audio.shape}")

                        # 保存文件
                        output_path = os.path.join(output_dir, f"response_{response_count}.wav")
                        sf.write(output_path, full_audio, 24000)
                        print(f"💾 Saved to {output_path}")

                        print("🔊 Playing response...")
                        play_audio(full_audio, 24000)
                        print("✅ Done!")
                        response_count += 1

                    except Exception as e:
                        print(f"Error concatenating: {e}")
                        # 逐个保存每个块
                        for i, chunk in enumerate(audio_chunks):
                            output_path = os.path.join(output_dir, f"response_chunk_{response_count}_{i}.wav")
                            sf.write(output_path, chunk, 24000)
                            print(f"💾 Saved chunk to {output_path}")
                else:
                    print("❌ No valid audio chunks received")
                    print("Trying alternative method...")

                    # 方法2：使用 A1_A2 函数直接生成
                    from inference import load_audio, get_input_ids_whisper, A1_A2

                    mel, leng = load_audio(audio_path)
                    audio_feature, input_ids = get_input_ids_whisper(mel, leng, model.whispermodel, model.device)

                    text = A1_A2(
                        model.fabric,
                        audio_feature,
                        input_ids,
                        leng,
                        model.model,
                        model.text_tokenizer,
                        response_count,
                        model.snacmodel,
                        out_dir=output_dir
                    )
                    print(f"💾 Audio saved to {output_dir}/{response_count:02d}.wav")
                    print(f"📝 Text response: {text}")
                    response_count += 1

            except Exception as e:
                print(f"Error: {e}")
                import traceback
                traceback.print_exc()

        elif choice == "2":
            print("Goodbye!")
            break


if __name__ == "__main__":
    main()