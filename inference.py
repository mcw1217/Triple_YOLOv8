import cv2
import time
from ultralytics import YOLO

# YOLOv8 모델 로드
model = YOLO('EX/new_data/train/yolov8n34/weights/best.pt')

image_paths = [
    "dataset/new_data/test/RGB/000037.jpg",
    "dataset/new_data/test/RGB/000038.jpg",
    "dataset/new_data/test/RGB/000039.jpg",
    "dataset/new_data/test/RGB/000040.jpg"
]
depth_image_paths = [
    "dataset/new_data/test/D/000037.jpg",
    "dataset/new_data/test/D/000038.jpg",
    "dataset/new_data/test/D/000039.jpg",
    "dataset/new_data/test/D/000040.jpg"
]
thermo_image_paths = [
    "dataset/new_data/test/Thermo/000037.jpg",
    "dataset/new_data/test/Thermo/000038.jpg",
    "dataset/new_data/test/Thermo/000039.jpg",
    "dataset/new_data/test/Thermo/000040.jpg"
]

total_inference_time = 0
num_images = len(image_paths)

for i in range(num_images):
    # 이미지 로드
    img = cv2.imread(image_paths[i])
    img2 = cv2.imread(depth_image_paths[i])
    img3 = cv2.imread(thermo_image_paths[i])
    
    # 추론 시간 측정 시작
    start_time = time.time()
    
    # YOLOv8 추론 수행
    results = model(img, img2, img3)
    
    # 추론 시간 측정 종료
    end_time = time.time()
    
    # 각 이미지의 추론 시간 누적
    inference_time = end_time - start_time
    total_inference_time += inference_time
    
    
    
    # 결과 시각화
    for j, result in enumerate(results):
        if hasattr(result, 'boxes'):
            # 이미지에 바운딩 박스 그리기
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0]  # 바운딩 박스의 좌표
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            
            # 바운딩 박스가 그려진 이미지 표시
            cv2.imshow(f"YOLOv8 추론 - 이미지 {i+1}", img)
        else:
            print(f"Result {j+1} does not have bounding boxes.")

# 평균 FPS 계산
average_inference_time = total_inference_time / num_images
fps = 1 / average_inference_time
print(f"Average Inference Time: {average_inference_time:.4f} seconds")
print(f"Average FPS: {fps:.2f}")

# 키 입력 대기 후 창 닫기
cv2.waitKey(0)
cv2.destroyAllWindows()
