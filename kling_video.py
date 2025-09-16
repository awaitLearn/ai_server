import torch
from diffusers import DiffusionPipeline
import imageio
import os
import time
import numpy as np
from datetime import datetime

print("🔍 Проверка системы...")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA доступна: {torch.cuda.is_available()}")

if not torch.cuda.is_available():
    print("❌ CUDA не доступна!")
    exit(1)

gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
print(f"💾 GPU память: {gpu_memory:.1f} GB")

print("🔄 Загружаем ModelScope модель...")
try:
    pipeline = DiffusionPipeline.from_pretrained(
        "damo-vilab/modelscope-damo-text-to-video-synthesis",
        torch_dtype=torch.float16,
        variant="fp16"
    ).to("cuda")
    print("✅ Модель успешно загружена!")
except Exception as e:
    print(f"❌ Ошибка загрузки модели: {e}")
    exit(1)

# Ваш промпт
prompt = "a man leaves the room and teleports to the world of ANIME NARUTO, fighting with enemies in a beautiful field with flowers in rainy weather, cinematic, high quality, dynamic action, epic battle, anime style"

print(f"🎬 Начинаем генерацию видео...")
print(f"📝 Промпт: {prompt}")

try:
    print("🔨 Генерируем видео... (это займет 3-8 минут)")
    
    # Генерация видео - 8 секунд, 64 кадра
    result = pipeline(
        prompt=prompt,
        num_frames=64,
        num_inference_steps=25,
        guidance_scale=7.5,
        generator=torch.Generator("cuda").manual_seed(int(time.time())),
        height=384,
        width=384,
    )
    
    video_frames = result.frames[0]
    print(f"✅ Видео сгенерировано: {len(video_frames)} кадров")
    
    # Сохраняем результат
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"naruto_battle_{timestamp}.mp4"
    
    print("💾 Сохраняем видео...")
    
    # Конвертируем в numpy arrays
    frames_np = [np.array(frame) for frame in video_frames]
    
    # Сохраняем с помощью imageio
    with imageio.get_writer(
        output_path, 
        fps=8, 
        codec='libx264',
        quality=8,
        pixelformat='yuv420p'
    ) as writer:
        for frame in frames_np:
            writer.append_data(frame)
    
    if os.path.exists(output_path):
        file_size = os.path.getsize(output_path) / (1024 * 1024)
        print(f"🎉 ВИДЕО ГОТОВО! 🎉")
        print(f"📁 Файл: {output_path}")
        print(f"📊 Размер: {file_size:.1f} MB")
        print(f"\n📥 Для скачивания:")
        print(f"scp root@192.165.134.28:{output_path} .")
    else:
        print("❌ Файл не создался")
        
except Exception as e:
    print(f"❌ Ошибка генерации: {e}")
