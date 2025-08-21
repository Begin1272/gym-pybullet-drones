import os
import numpy as np
import cv2
import gymnasium as gym
from gymnasium import spaces

from gym_pybullet_drones.envs.BaseAviary import BaseAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics

class PollutionAviary(BaseAviary):
    """
    DQN을 위한 오염원 탐지 및 경로 최적화 환경
    """
    def __init__(self,
                 drone_model: DroneModel=DroneModel.CF2X,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics=Physics.PYB,
                 ):
        
        # HYSPLIT 오염 지도 이미지 로드 및 전처리
        image_path = 'C:/Users/SAMSUNG/Desktop/intern/pybullet_model/gym-pybullet-drones/materials/testimg3.png'
        self.pollution_map = self.load_and_preprocess_map(image_path)
        self.map_height, self.map_width = self.pollution_map.shape[:2]

        self.max_x = self.map_width / 20.0
        self.max_y = self.map_height / 20.0
        
        # 이전 스텝의 오염 농도를 저장하기 위한 변수
        self.previous_pollution_density = 0

        # 부모 클래스의 생성자를 호출합니다.
        # 이 과정에서 아래에 정의된 _observationSpace와 _actionSpace가 자동으로 호출됩니다.
        super().__init__(drone_model=drone_model,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics
                         )

    # ----------------------------------------------------
    # vvvv 여기가 새로 추가/수정된 부분입니다! vvvv
    # ----------------------------------------------------

    def _observationSpace(self):
        """환경의 관측 공간을 정의합니다."""
        return spaces.Box(
            low=np.array([-self.max_x, -self.max_y, 0, 0]),
            high=np.array([self.max_x, self.max_y, 1, 1]),
            dtype=np.float32
        )

    def _actionSpace(self):
        """환경의 행동 공간을 정의합니다."""
        # 4개의 모터에 대한 RPM 값을 행동으로 정의 (0 ~ MAX_RPM)
        return spaces.Box(low=np.zeros(4),
                          high=np.full(4, self.MAX_RPM),
                          dtype=np.float32
                          )

    # ----------------------------------------------------
    # ^^^^ 여기가 새로 추가/수정된 부분입니다! ^^^^
    # ----------------------------------------------------

    def load_and_preprocess_map(self, image_path):
        """
        HYSPLIT 이미지를 불러와 오염 농도 맵으로 전처리합니다.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"오염 지도 파일을 찾을 수 없습니다: {image_path}")

        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"오염 지도 이미지를 불러올 수 없습니다: {image_path}")
        
        return image / 255.0

    def _computeReward(self):
        """
        현재 오염 농도에 기반한 보상 계산
        """
        current_pos = self.pos[0]
        x_pixel = int((current_pos[0] + self.max_x) / (2 * self.max_x) * (self.map_width - 1))
        y_pixel = int((-current_pos[1] + self.max_y) / (2 * self.max_y) * (self.map_height - 1))
        
        if 0 <= x_pixel < self.map_width and 0 <= y_pixel < self.map_height:
            current_pollution_density = self.pollution_map[y_pixel, x_pixel]
        else:
            current_pollution_density = 0
            
        reward = (current_pollution_density - self.previous_pollution_density) * 10.0
        reward -= 0.01 
        
        self.previous_pollution_density = current_pollution_density
        return reward

    def _computeObs(self):
        """
        현재 상태를 계산하고 반환합니다.
        """
        current_pos = self.pos[0]
        x_pixel = int((current_pos[0] + self.max_x) / (2 * self.max_x) * (self.map_width - 1))
        y_pixel = int((-current_pos[1] + self.max_y) / (2 * self.max_y) * (self.map_height - 1))

        if 0 <= x_pixel < self.map_width and 0 <= y_pixel < self.map_height:
            current_pollution_density = self.pollution_map[y_pixel, x_pixel]
        else:
            current_pollution_density = 0
            
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
        if self.previous_pollution_density >= 0.95:
            print("목표 오염원에 도달! 에피소드 종료.")
            return True
        return False