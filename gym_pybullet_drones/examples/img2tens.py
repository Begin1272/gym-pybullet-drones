import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

# --- 1. 이미지 파일 불러오기 ---
# 알려주신 경로로 수정했습니다.
image_path = 'C:/Users/SAMSUNG/Desktop/intern/pybullet_model/gym-pybullet-drones/materials/testimg3.png'
try:
    # OpenCV로 이미지 읽기 (결과는 NumPy 배열)
    original_image_np = cv2.imread(image_path)
    # OpenCV는 색상 채널을 BGR(파랑, 초록, 빨강) 순서로 불러오므로,
    # 일반적인 RGB(빨강, 초록, 파랑) 순서로 바꿔줍니다.
    original_image_np = cv2.cvtColor(original_image_np, cv2.COLOR_BGR2RGB)
except Exception as e:
    print(f"이미지를 불러오는 데 실패했습니다: {e}")
    print(f"'{image_path}' 파일 경로가 올바른지 다시 확인해주세요.")
    # 예외 발생 시, 임의의 100x100 검은색 이미지 생성
    original_image_np = np.zeros((100, 100, 3), dtype=np.uint8)


# --- 2. NumPy 배열을 PyTorch 텐서로 변환 ---
# NumPy 배열을 기반으로 텐서를 생성합니다.
# 픽셀 값의 타입을 float으로 바꾸고 255로 나누어 0~1 사이로 정규화합니다.
image_tensor = torch.from_numpy(original_image_np).float() / 255.0
# PyTorch 모델은 보통 (채널, 높이, 너비) 순서를 기대하므로 차원을 변경합니다.
# 현재: (높이, 너비, 채널) -> 변경 후: (채널, 높이, 너비)
image_tensor = image_tensor.permute(2, 0, 1)


# --- 3. 확인을 위해 텐서를 다시 시각화 가능한 NumPy 배열로 변환 ---
# 화면에 표시하기 위해 차원 순서를 다시 (높이, 너비, 채널)로 되돌립니다.
tensor_to_image_np = image_tensor.permute(1, 2, 0).numpy()


# --- 4. 원본 이미지와 변환된 이미지를 나란히 시각화 ---
# 두 개의 이미지를 보여줄 창을 생성
plt.figure(figsize=(10, 5))

# 첫 번째 이미지 (원본)
plt.subplot(1, 2, 1)
plt.title("Original Image (NumPy)")
plt.imshow(original_image_np)
plt.axis('off') # 축 번호 숨기기

# 두 번째 이미지 (텐서에서 변환)
plt.subplot(1, 2, 2)
plt.title("Tensor -> Image (NumPy)")
plt.imshow(tensor_to_image_np)
plt.axis('off') # 축 번호 숨기기

# 이미지 창 보여주기
plt.show()

print("Original NumPy shape:", original_image_np.shape)
print("Tensor shape (C, H, W):", image_tensor.shape)