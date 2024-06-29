from CattleWeight.Cattle_inference import CattleInference
import CattleWeight
import os

path = os.path.dirname(CattleWeight.__file__)
model = CattleInference(cwd=path)
side_img_path = "/Users/aftaabhussain/Work/Data Accumulator and prediction/Images/Side/127_b4-1_s_131_M.jpg"
rear_img_path = "/Users/aftaabhussain/Work/Data Accumulator and prediction/Images/Rear/127_b4-1_r_131_M.jpg"

weight = model.predict(side_img_path, rear_img_path)
print(f"The weight is {weight} kg")
