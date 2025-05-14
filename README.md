### YOLOv11을 이용해서 분석하기
```bash
# 1. 필수 패키지 설치
!pip install ultralytics yt-dlp opencv-python --quiet

# 2. 유튜브 영상 다운로드
import yt_dlp
import glob

video_url = "https://www.youtube.com/shorts/0N0iZudj0R4"
ydl_opts = {
    'format': 'bestvideo+bestaudio/best',
    'outtmpl': 'input_video.%(ext)s',
}

with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    ydl.download([video_url])

# 3. 다운로드된 파일명 자동 인식
video_files = glob.glob("input_video.*")
if video_files:
    input_video_path = video_files[0]
    print(f"🎥 다운로드 완료: {input_video_path}")
else:
    raise FileNotFoundError("❌ 다운로드된 비디오 파일을 찾을 수 없습니다.")

# 4. YOLOv8 모델 불러오기
from ultralytics import YOLO
model = YOLO("yolov8n.pt")  # 가장 작은 YOLOv8 모델

# 5. 비디오 읽고 처리
import cv2

cap = cv2.VideoCapture(input_video_path)
if not cap.isOpened():
    raise IOError(f"❌ 비디오 파일 열기에 실패했습니다: {input_video_path}")

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"🔍 해상도: {width}x{height}, FPS: {fps}")

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("output_yolov8.mp4", fourcc, fps, (width, height))

frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(source=frame, save=False, imgsz=640)
    annotated_frame = results[0].plot()
    out.write(annotated_frame)
    frame_count += 1

cap.release()
out.release()
print(f"✅ 객체 인식 완료! 총 {frame_count} 프레임 처리됨.")

# 6. Colab에서 영상 표시
from IPython.display import Video
Video("output_yolov8.mp4", embed=True)

# 7. Colab → 로컬로 다운로드
from google.colab import files
files.download("output_yolov8.mp4")

```
