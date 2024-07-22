import pandas as pd
import random
import os
from datetime import datetime, timedelta

# 确认当前工作目录是否允许写入
script_dir = os.path.dirname(__file__)
os.chdir(script_dir)

# 动物和食物列表
animals = ['Dog', 'Cat', 'bird']
foods = ['Apple', 'Banana', 'Orange', 'Grape', 'Watermelon']

# 偏好评分范围
preference_range = [1, 5]

# 样本数量
num_samples = 10000

# 时间段列表
times_of_day = ['Morning', 'Afternoon', 'Evening']

# 日期范围
start_date = datetime(2024, 1, 1)
end_date = datetime.now()

def random_date(start, end):
    return start + timedelta(
        seconds=random.randint(0, int((end - start).total_seconds())),
    )

# 生成数据
data = {
    'animal': [random.choice(animals) for _ in range(num_samples)],
    'food': [random.choice(foods) for _ in range(num_samples)],
    'preference': [random.randint(preference_range[0], preference_range[1]) for _ in range(num_samples)],
    'time_of_day': [random.choice(times_of_day) for _ in range(num_samples)],
    'timestamp': [random_date(start_date, end_date) for _ in range(num_samples)]
}

# 创建DataFrame
df = pd.DataFrame(data)

# 确认保存路径是否允许写入
save_path = os.path.join(script_dir, 'animal_food_preferences.csv')
df.to_csv(save_path, index=False)

print("数据集已生成并保存到 'animal_food_preferences.csv'")
