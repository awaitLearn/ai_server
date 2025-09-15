import torch
import PIL.Image
import imageio
import os
import time
import numpy as np
from datetime import datetime

try:
    from diffusers import KlingImgToVidPipeline
    print("✅ KlingImgToVidPipeline импортирован успешно")
except ImportError:
    print("❌ Не удалось импортировать KlingImgToVidPipeline")
    print("Попробуйте: pip install git+https://github.com/huggingface/diffusers.git")
    exit(1)

class KlingVideoGenerator:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🚀 Используется устройство: {self.device}")
        
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"💾 GPU память: {gpu_memory:.1f} GB")
            
        print("🔄 Загружаем Kling AI модель... Это займет время (10-20 минут)...")
        self._load_model()
    
    def _load_model(self):
        """Загружаем Kling AI модель"""
        try:
            self.pipeline = KlingImgToVidPipeline.from_pretrained(
                "kling/Kling",
                torch_dtype=torch.float16,
                variant="fp16",
            ).to(self.device)
            print("✅ Модель успешно загружена!")
        except Exception as e:
            print(f"❌ Ошибка загрузки модели: {e}")
            print("\n🔧 Решение проблем:")
            print("1. Убедитесь что выполнили: git lfs install")
            print("2. Попробуйте: pip install git+https://github.com/huggingface/diffusers.git")
            print("3. Установите: pip install decord einops omegaconf timm requests av")
            raise

    def _clear_memory(self):
        """Очистка памяти CUDA"""
        torch.cuda.empty_cache()
        print("🧹 Память очищена")

    def generate_video(self, image_path: str, prompt: str, num_frames: int = 50) -> str:
        """Генерируем видео из изображения и промпта"""
        print(f"🎬 Начинаем генерацию видео ({num_frames} кадров)...")
        print(f"📝 Промпт: {prompt}")
        
        # Загружаем изображение
        try:
            image = PIL.Image.open(image_path).convert("RGB")
            image = image.resize((1024, 576))
            print(f"📸 Изображение загружено: {image.size}")
        except Exception as e:
            print(f"❌ Ошибка загрузки изображения: {e}")
            return None

        print(f"📊 Генерируем {num_frames} кадров")

        try:
            print("🔨 Генерируем видео... (это займет 5-15 минут)")
            
            # Генерация видео
            result = self.pipeline(
                image=image,
                prompt=prompt,
                num_frames=num_frames,
                fps=10,
                num_inference_steps=20,
                guidance_scale=7.5,
                generator=torch.Generator(self.device).manual_seed(int(time.time())),
            )
            
            video_frames = result.frames[0]
            print(f"✅ Видео сгенерировано: {len(video_frames)} кадров")
                
        except Exception as e:
            print(f"❌ Ошибка генерации: {e}")
            return None

        # Сохраняем результат
        output_path = self._save_video(video_frames, fps=10)
        return output_path

    def _save_video(self, frames, fps: int):
        """Сохраняем кадры как видео"""
        if not frames:
            print("❌ Нет кадров для сохранения")
            return None

        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"kling_video_{timestamp}.mp4"
            
            print("💾 Сохраняем видео...")
            
            # Конвертируем PIL Image в numpy arrays
            frames_np = [np.array(frame) for frame in frames]
            
            # Сохраняем с помощью imageio
            with imageio.get_writer(
                output_path, 
                fps=fps, 
                codec='libx264',
                quality=9,
                pixelformat='yuv420p'
            ) as writer:
                for frame in frames_np:
                    writer.append_data(frame)
            
            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path) / (1024 * 1024)
                print(f"✅ Видео сохранено: {output_path} ({file_size:.1f} MB)")
                return output_path
            else:
                print("❌ Файл не создался")
                return None
                
        except Exception as e:
            print(f"❌ Ошибка сохранения: {e}")
            return None

if __name__ == "__main__":
    print("🔍 Проверка системы...")
    
    # Проверяем CUDA
    if not torch.cuda.is_available():
        print("❌ CUDA не доступна!")
        exit(1)
        
    # Проверяем память
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"💾 GPU память: {gpu_memory:.1f} GB")
    
    # Инициализируем генератор
    generator = KlingVideoGenerator()
    
    # Ваши параметры
    image_path = "face.jpg"
    prompt = "a man leaves the room and teleports to the world of ANIME NARUTO, fighting with enemies in a beautiful field with flowers in rainy weather, cinematic, high quality, dynamic action, epic battle"
    
    # Генерируем видео (начнем с 30 кадров для теста)
    print("\n⭐ СТАРТ ГЕНЕРАЦИИ ⭐")
    result = generator.generate_video(
        image_path=image_path,
        prompt=prompt,
        num_frames=30  # 3 секунды для теста
    )
    
    if result:
        print(f"\n🎉 ВИДЕО ГОТОВО! 🎉")
        print(f"📁 Файл: {result}")
        print("\n📥 Для скачивания:")
        print(f"scp root@192.165.134.28:{result} .")
    else:
        print("\n💥 Генерация не удалась")
