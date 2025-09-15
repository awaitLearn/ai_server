def _save_video(self, frames, output_path: str, fps: int = 24):
    """Сохраняем кадры как видео"""
    if not frames:
        print("❌ Нет кадров для сохранения")
        return None
    
    try:
        height, width = frames[0].size
        print(f"📏 Размер кадра: {width}x{height}")
        
        # Пробуем разные кодеки
        codecs = ['mp4v', 'XVID', 'MJPG']
        
        for codec in codecs:
            try:
                fourcc = cv2.VideoWriter_fourcc(*codec)
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                
                if out.isOpened():
                    print(f"✅ Используем кодек: {codec}")
                    for frame in frames:
                        out.write(cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR))
                    out.release()
                    
                    # Проверяем что файл создался
                    if os.path.exists(output_path):
                        print(f"✅ Видео сохранено: {output_path}")
                        return output_path
                    else:
                        print("❌ Файл не создался")
                        break
                        
            except Exception as e:
                print(f"❌ Ошибка с кодеком {codec}: {e}")
        
        # Сохраняем как PNG если видео не получилось
        print("💾 Сохраняем кадры как PNG...")
        os.makedirs("frames", exist_ok=True)
        for i, frame in enumerate(frames):
            frame.save(f"frames/frame_{i:04d}.png")
        return "frames/"
        
    except Exception as e:
        print(f"❌ Критическая ошибка сохранения: {e}")
        return None
