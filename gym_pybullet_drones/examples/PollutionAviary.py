import os
import numpy as np
import cv2
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.env_util import make_vec_env

# BaseAviary 클래스를 상속받아 환경을 구축합니다.
# gym_pybullet_drones/envs/BaseAviary.py 파일을 참고하여 필요한 모듈을 임포트하세요.
from gym_pybullet_drones.envs.BaseAviary import BaseAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType

class PollutionAviary(BaseAviary):
    """
    DQN을 위한 오염원 탐지 및 경로 최적화 환경
    """
    def __init__(self,
                 drone_model: DroneModel=DroneModel.CF2X,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics=Physics.PYB,
                 obs: ObservationType=ObservationType.KIN,
                 act: ActionType=ActionType.RPM,
                 record_video=False,
                 output_folder='results',
                 ):
        
        # HYSPLIT 오염 지도 이미지 로드 및 전처리
        # 파일 경로를 직접 지정해 주세요.
        image_path = 'C:\\Users\\SAMSUNG\\Desktop\\intern\\pybullet_model\\gym-pybullet-drones\\materials\\testimg3.png'
        self.pollution_map = self.load_and_preprocess_map(image_path)
        self.map_height, self.map_width = self.pollution_map.shape[:2]

        # 상태 및 행동 공간을 정의합니다.
        # 드론의 위치(x, y, z) + 오염 농도를 관측값으로 사용합니다.
        # 드론의 최대, 최소 위치를 설정하여 오염 지도 범위 내에 있도록 합니다.
        self.max_x = self.map_width / 2.0
        self.max_y = self.map_height / 2.0
        self.observation_space = spaces.Box(
            low=np.array([-self.max_x, -self.max_y, 0, 0]),
            high=np.array([self.max_x, self.max_y, 1, 1]),
            dtype=np.float32
        )
        
        # 부모 클래스의 생성자를 호출합니다.
        super().__init__(drone_model=drone_model,
                         initial_xyzs=initial_xyzs,
                         physics=physics,
                         obs=obs,
                         act=act,
                         record_video=record_video,
                         output_folder=output_folder,
                         initial_rpys=initial_rpys,
                         )
        
        # 이전 스텝의 오염 농도를 저장하기 위한 변수
        self.previous_pollution_density = 0

    def load_and_preprocess_map(self, image_path):
        """
        HYSPLIT 이미지를 불러와 오염 농도 맵으로 전처리합니다.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"오염 지도 파일을 찾을 수 없습니다: {image_path}")

        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"오염 지도 이미지를 불러올 수 없습니다: {image_path}")
        
        # 픽셀 값을 0-1 사이로 정규화
        return image / 255.0

    def _computeReward(self):
        """
        현재 오염 농도에 기반한 보상 계산
        """
        # 드론의 현재 위치를 가져옵니다.
        current_pos = self.get_pos()[0] # 첫 번째 드론의 위치
        
        # 위치를 이미지 픽셀 좌표로 변환합니다.
        x_pixel = int((current_pos[0] + self.max_x) / (2 * self.max_x) * self.map_width)
        y_pixel = int((-current_pos[1] + self.max_y) / (2 * self.max_y) * self.map_height)
        
        # 유효한 픽셀 좌표 범위 확인
        if 0 <= x_pixel < self.map_width and 0 <= y_pixel < self.map_height:
            current_pollution_density = self.pollution_map[y_pixel, x_pixel]
        else:
            current_pollution_density = 0 # 맵 밖으로 벗어나면 보상 0
            
        # 오염 밀도 변화량으로 보상을 계산합니다.
        # 오염도가 높아지면 양의 보상, 낮아지면 음의 보상
        reward = (current_pollution_density - self.previous_pollution_density) * 10.0
        
        # 매 스텝마다 작은 시간 패널티를 줘서 빠른 탐색을 유도
        reward -= 0.01 
        
        self.previous_pollution_density = current_pollution_density
        
        return reward

    def _computeObs(self):
        """
        현재 상태를 계산하고 반환합니다.
        """
        current_pos = self.get_pos()[0]
        x_pixel = int((current_pos[0] + self.max_x) / (2 * self.max_x) * self.map_width)
        y_pixel = int((-current_pos[1] + self.max_y) / (2 * self.max_y) * self.map_height)

        if 0 <= x_pixel < self.map_width and 0 <= y_pixel < self.map_height:
            current_pollution_density = self.pollution_map[y_pixel, x_pixel]
        else:
            current_pollution_density = 0
            
        # 드론의 위치(x, y)와 오염 농도를 관측값으로 반환
        return np.array([current_pos[0], current_pos[1], current_pos[2], current_pollution_density])

    def _computeInfo(self):
        """
        보조 정보 딕셔너리를 반환합니다.
        """
        return {"current_pollution_density": self.previous_pollution_density}

    def _computeDone(self):
        """
        에피소드 종료 여부를 판단합니다.
        """
        # 현재 오염 밀도가 특정 임계값 이상이면 에피소드 종료
        if self.previous_pollution_density >= 0.95:
            print("목표 오염원에 도달! 에피소드 종료.")
            return True
        return False