import gymnasium as gym
import time
import numpy as np  # <--- NumPy를 임포트합니다!
from pollutionaviary import PollutionAviary

# 1. 드론의 초기 위치를 NumPy 배열로 생성합니다.
initial_positions = np.array([[0, 0, 1]])

# 2. 환경 생성 시, NumPy 배열로 만든 초기 위치를 전달합니다.
env = PollutionAviary(initial_xyzs=initial_positions) 

# 3. 환경 초기화 및 상태 확인
obs, info = env.reset()
print(f"초기 상태(Observation): {obs}")

# 4. 100 스텝 동안 무작위로 행동하며 테스트
for i in range(100):
    # 행동 공간에서 무작위 행동 하나를 샘플링합니다.
    action = env.action_space.sample() 
    
    # 행동을 실행하고 다음 상태, 보상, 종료 여부 등을 받습니다.
    obs, reward, terminated, truncated, info = env.step(action)
    
    # 콘솔에 상태 출력
    print(f"스텝 {i+1}: 보상(Reward): {reward:.4f}, 오염농도: {info.get('current_pollution_density', 0):.4f}")
    
    # 에피소드가 종료되면 루프를 빠져나옵니다.
    if terminated or truncated:
        print("\n에피소드가 종료되었습니다!")
        break
    
    # 시뮬레이션 GUI를 실시간으로 보기 위해 약간의 딜레이를 줍니다.
    time.sleep(1/30)

# 5. 환경 종료
env.close()
print("\n테스트 완료!")