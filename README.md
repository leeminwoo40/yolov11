### YOLOv11을 이용해서 분석하기
```bash
!pip install ultralytics yt-dlp opencv-python --quiet
import yt_dlp

video_url = "https://www.youtube.com/shorts/0N0iZudj0R4"
ydl_opts = {
    'format': 'best',
    'outtmpl': 'input_video.%(ext)s',
}

with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    ydl.download([video_url])
from ultralytics import YOLO

# YOLOv11 small model (다른 모델로 바꿔도 됨: yolov11n.pt, yolov11m.pt 등)
model = YOLO("yolov8s.pt")
import cv2
import os

# 비디오 열기
input_video_path = "input_video.mp4"
cap = cv2.VideoCapture(input_video_path)

# 비디오 저장 설정
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 저장 형식: mp4
out = cv2.VideoWriter("output_yolov8.mp4", fourcc, fps, (width, height))

frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 객체 인식
    results = model.predict(source=frame, save=False, imgsz=640)
    annotated_frame = results[0].plot()  # 결과 그리기

    out.write(annotated_frame)  # 결과 저장
    frame_count += 1

cap.release()
out.release()
print(f"✅ 객체 인식 완료! 총 {frame_count} 프레임 처리됨.")
from IPython.display import Video
Video("output_yolov11.mp4", embed=True)

# 6. Colab → 로컬로 영상 다운로드
from google.colab import files
files.download("output_yolov11.mp4")
```
