import torch
from diffusers import StableVideoDiffusionPipeline, AnimateDiffPipeline, StableDiffusionXLPipeline
from PIL import Image
import cv2
import numpy as np
import time
import gc

class LocalVideoGenerator:
    def __init__(self, model_type="svd"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_type = model_type
        
        print(f"🔄 Загружаем модель {model_type}...")
        self._load_model()
    
    def _load_model(self):
        """Загружаем выбранную модель"""
        if self.model_type == "svd":
            self.pipeline = StableVideoDiffusionPipeline.from_pretrained(
                "stabilityai/stable-video-diffusion-img2vid-xt",
                torch_dtype=torch.float16,
                variant="fp16"
            ).to(self.device)
            
        elif self.model_type == "animatediff":
            # Загружаем SDXL для генерации кадров
            self.sdxl_pipeline = StableDiffusionXLPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0",
                torch_dtype=torch.float16
            ).to(self.device)
            
            # И AnimateDiff для анимации
            self.animate_pipeline = AnimateDiffPipeline.from_pretrained(
                "emilianJR/epiCRealism",
                torch_dtype=torch.float16
            ).to(self.device)
        
        print("✅ Модель загружена")

    def _clear_memory(self):
        """Очистка памяти CUDA"""
        torch.cuda.empty_cache()
        gc.collect()
        print("🧹 Память очищена")

    def generate_from_image(self, image_path: str, prompt: str, duration: int = 10) -> str:
        """Генерируем видео из изображения"""
        print(f"🎬 Генерируем {duration} сек видео...")
        
        # Загружаем и подготавливаем изображение
        image = Image.open(image_path).convert("RGB")
        image = image.resize((1024, 576))  # 16:9
        
        try:
            if self.model_type == "svd":
                return self._generate_svd(image, duration)
            else:
                return self._generate_animatediff(image, prompt, duration)
        finally:
            self._clear_memory()

    def _generate_svd(self, image: Image.Image, duration: int) -> str:
        """Генерация через Stable Video Diffusion"""
        # Конвертируем длительность в кадры (уменьшаем FPS для экономии памяти)
        num_frames = min(75, duration * 15)  # Уменьшили с 25 до 15 FPS
        
        # Генерируем видео с оптимизированными параметрами
        video_frames = self.pipeline(
            image,
            num_frames=num_frames,
            num_inference_steps=20,  # Уменьшили количество шагов
            motion_bucket_id=150,    # Уменьшили motion для меньшей сложности
            fps=15,                  # Уменьшили FPS
            decode_chunk_size=4,     # Добавим chunking для декодирования
            generator=torch.Generator(self.device).manual_seed(int(time.time()))
        ).frames[0]
        
        # Сохраняем результат
        output_path = f"svd_video_{int(time.time())}.mp4"
        self._save_video(video_frames, output_path, fps=15)
        
        return output_path

    def _generate_animatediff(self, image: Image.Image, prompt: str, duration: int) -> str:
        """Генерация через AnimateDiff + SDXL"""
        # Сначала генерируем базовый кадр через SDXL
        print("🎨 Генерируем базовый кадр...")
        base_frame = self.sdxl_pipeline(
            prompt=prompt,
            image=image,
            strength=0.6,           # Уменьшили strength
            num_inference_steps=25,  # Уменьшили шаги
            guidance_scale=7.0       # Уменьшили guidance scale
        ).images[0]
        
        # Очищаем память после генерации кадра
        self._clear_memory()
        
        # Затем анимируем через AnimateDiff
        print("🎬 Анимируем кадр...")
        num_frames = min(80, duration * 20)  # Уменьшили количество кадров
        
        video_frames = self.animate_pipeline(
            prompt=prompt,
            image=base_frame,
            num_frames=num_frames,
            num_inference_steps=20,  # Уменьшили шаги
            fps=20,                  # Уменьшили FPS
            guidance_scale=7.0       # Уменьшили guidance scale
        ).frames[0]
        
        output_path = f"animatediff_video_{int(time.time())}.mp4"
        self._save_video(video_frames, output_path, fps=20)
        
        return output_path

    def _save_video(self, frames, output_path: str, fps: int = 24):
        """Сохраняем кадры как видео"""
        if not frames:
            return None
        
        height, width = frames[0].size
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        for frame in frames:
            out.write(cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR))
        
        out.release()
        return output_path

# Пример использования
if __name__ == "__main__":
    # Инициализируем генератор
    generator = LocalVideoGenerator(model_type="svd")  # или "animatediff"
    
    # Генерируем видео
    result = generator.generate_from_image(
        image_path="face.jpg",
        prompt="beautiful man, cinematic portrait, high quality",
        duration=10  # секунд
    )
    
    print(f"✅ Видео готово: {result}")
