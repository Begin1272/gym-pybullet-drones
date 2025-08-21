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
        
        image_path = 'C:/Users/SAMSUNG/Desktop/intern/pybullet_model/gym-pybullet-drones/materials/testimg3.png'
        self.pollution_map = self.load_and_preprocess_map(image_path)
        self.map_height, self.map_width = self.pollution_map.shape[:2]

        self.max_x = self.map_width / 20.0
        self.max_y = self.map_height / 20.0
        
        self.previous_pollution_density = 0

        super().__init__(drone_model=drone_model,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics
                         )

    def _observationSpace(self):
        return spaces.Box(
            low=np.array([-self.max_x, -self.max_y, 0, 0]),
            high=np.array([self.max_x, self.max_y, 1, 1]),
            dtype=np.float32
        )

    def _actionSpace(self):
        return spaces.Box(low=np.zeros(4),
                          high=np.full(4, self.MAX_RPM),
                          dtype=np.float32
                          )
                          
    def _preprocessAction(self, action):
        return np.clip(action, 0, self.MAX_RPM)

    def load_and_preprocess_map(self, image_path):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"오염 지도 파일을 찾을 수 없습니다: {image_path}")
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"오염 지도 이미지를 불러올 수 없습니다: {image_path}")
        return image / 255.0

    def _computeReward(self):
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
        current_pos = self.pos[0]
        x_pixel = int((current_pos[0] + self.max_x) / (2 * self.max_x) * (self.map_width - 1))
        y_pixel = int((-current_pos[1] + self.max_y) / (2 * self.max_y) * (self.map_height - 1))

        if 0 <= x_pixel < self.map_width and 0 <= y_pixel < self.map_height:
            current_pollution_density = self.pollution_map[y_pixel, x_pixel]
        else:
            current_pollution_density = 0
            
        return np.array([current_pos[0], current_pos[1], current_pos[2], current_pollution_density])

    def _computeInfo(self):
        return {"current_pollution_density": self.previous_pollution_density}

    def _computeTerminated(self):
        """에피소드 종료 여부(성공/실패)를 판단합니다."""
        if self.previous_pollution_density >= 0.95:
            print("목표 오염원에 도달! 에피소드 종료.")
            return True
        return False

    def _computeTruncated(self):
        """에피소드 강제 중단 여부(시간 초과 등)를 판단합니다."""
        return False