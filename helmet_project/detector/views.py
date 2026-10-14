import os
import time
import cv2
import imageio
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from django.conf import settings
from ultralytics import YOLO

# Load model (fall back to yolov8n.pt if custom weights aren't found)
MODEL_PATH = os.path.join(settings.BASE_DIR, 'detector', 'ml_model', 'best.pt')

if os.path.exists(MODEL_PATH):
    model = YOLO(MODEL_PATH)
else:
    model = YOLO('yolov8n.pt')


def process_video_with_h264(input_path, output_path):
    """
    Reads a video frame-by-frame, runs YOLO detection, and writes
    the output with H.264 encoding so browsers can play it natively.
    """
    cap = cv2.VideoCapture(input_path)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps > 60:
        fps = 25.0

    # Use imageio to write H.264 encoded MP4 (playable in all browsers)
    writer = imageio.get_writer(output_path, fps=fps, codec='libx264', quality=8)

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Run YOLO prediction (RGB conversion for correct color rendering)
            results = model.predict(frame, verbose=False)
            annotated_frame = results[0].plot()

            # OpenCV uses BGR, imageio expects RGB
            rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            writer.append_data(rgb_frame)
    finally:
        cap.release()
        writer.close()


def index(request):
    context = {}
    if request.method == 'POST' and request.FILES.get('media'):
        uploaded_file = request.FILES['media']
        fs = FileSystemStorage()

        # Save uploaded file
        filename = fs.save(uploaded_file.name, uploaded_file)
        file_path = fs.path(filename)

        # Check if video or image
        ext = os.path.splitext(filename)[1].lower()
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
        unique_timestamp = int(time.time())

        if ext in video_extensions:
            output_filename = f"detected_{unique_timestamp}.mp4"
            output_path = os.path.join(settings.MEDIA_ROOT, output_filename)

            # Process video with browser-compatible H.264
            process_video_with_h264(file_path, output_path)

            context['is_video'] = True
            context['detected_url'] = f"{fs.url(output_filename)}?v={unique_timestamp}"
        else:
            output_filename = f"detected_{unique_timestamp}{ext}"
            output_path = os.path.join(settings.MEDIA_ROOT, output_filename)

            # Process image
            results = model.predict(file_path)
            annotated_img = results[0].plot()
            cv2.imwrite(output_path, annotated_img)

            context['is_video'] = False
            context['detected_url'] = f"{fs.url(output_filename)}?v={unique_timestamp}"

    return render(request, 'detector/index.html', context)