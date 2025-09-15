import torch
from diffusers import KlingImgToVidPipeline
from PIL import Image
import imageio
import os
import time
from datetime import datetime

class KlingVideoGenerator:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🚀 Используется устройство: {self.device}")
        
        # Проверяем доступную память
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"💾 GPU память: {gpu_memory:.1f} GB")
            
        print("🔄 Загружаем Kling AI модель... Это займет несколько минут...")
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
            print("Попробуйте запустить: git lfs install")
            raise

    def _clear_memory(self):
        """Очистка памяти CUDA"""
        torch.cuda.empty_cache()
        print("🧹 Память очищена")

    def generate_video(self, image_path: str, prompt: str, duration_sec: int = 60) -> str:
        """Генерируем видео из изображения и промпта"""
        print(f"🎬 Начинаем генерацию видео ({duration_sec} секунд)...")
        print(f"📝 Промпт: {prompt}")
        
        # Загружаем изображение
        try:
            image = Image.open(image_path).convert("RGB")
            print(f"📸 Изображение загружено: {image.size}")
        except Exception as e:
            print(f"❌ Ошибка загрузки изображения: {e}")
            return None

        # Рассчитываем параметры для длинного видео
        fps = 10  # FPS для плавного видео
        total_frames = duration_sec * fps
        
        # Разбиваем на сегменты (Kling может генерировать сегментами)
        segment_frames = min(100, total_frames)  # Максимум 100 кадров за один проход
        num_segments = max(1, total_frames // segment_frames)
        
        print(f"📊 Параметры: {fps} FPS, {total_frames} кадров, {num_segments} сегментов")

        all_frames = []
        
        try:
            for segment in range(num_segments):
                print(f"🔨 Генерируем сегмент {segment + 1}/{num_segments}...")
                
                # Генерация сегмента видео
                video_frames = self.pipeline(
                    image=image,
                    prompt=prompt,
                    num_frames=segment_frames,
                    fps=fps,
                    num_inference_steps=30,
                    guidance_scale=7.5,
                    generator=torch.Generator(self.device).manual_seed(int(time.time()) + segment),
                    height=576,  # Оптимальное разрешение
                    width=1024,
                ).frames[0]
                
                all_frames.extend(video_frames)
                print(f"✅ Сегмент {segment + 1} готов: {len(video_frames)} кадров")
                
                # Очищаем память между сегментами
                self._clear_memory()
                
        except Exception as e:
            print(f"❌ Ошибка генерации: {e}")
            return None

        # Сохраняем результат
        output_path = self._save_video(all_frames, fps)
        return output_path

    def _save_video(self, frames, fps: int):
        """Сохраняем кадры как видео"""
        if not frames:
            print("❌ Нет кадров для сохранения")
            return None

        try:
            # Создаем имя файла с timestamp
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
                quality=10,
                pixelformat='yuv420p'
            ) as writer:
                for frame in frames_np:
                    writer.append_data(frame)
            
            # Проверяем размер файла
            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path) / (1024 * 1024)  # в MB
                print(f"✅ Видео сохранено: {output_path} ({file_size:.1f} MB)")
                return output_path
            else:
                print("❌ Файл не создался")
                return None
                
        except Exception as e:
            print(f"❌ Ошибка сохранения: {e}")
            return None

# Функция для проверки памяти
def check_gpu_memory():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"📊 Память: {allocated:.1f} GB allocated, {reserved:.1f} GB reserved")

if __name__ == "__main__":
    # Проверяем GPU
    print("🔍 Проверка системы...")
    check_gpu_memory()
    
    # Инициализируем генератор
    generator = KlingVideoGenerator()
    
    # Ваши параметры
    image_path = "face.jpg"  # Ваше фото на сервере
    prompt = "a man I leave the room and then find myself in the world of ANIME NARUTO, and start fighting with enemies in a beautiful field with flowers in rainy weather, cinematic, high quality, best quality, dynamic action, epic battle"
    
    # Генерируем видео (60 секунд)
    print("\n⭐ СТАРТ ГЕНЕРАЦИИ ⭐")
    result = generator.generate_video(
        image_path=image_path,
        prompt=prompt,
        duration_sec=60  # 1 минута
    )
    
    if result:
        print(f"\n🎉 ВИДЕО ГОТОВО! 🎉")
        print(f"📁 Файл: {result}")
        print("🚀 Запустите: scp user@your-server:{result} .  для скачивания")
    else:
        print("\n💥 Генерация не удалась")
    
    check_gpu_memory()