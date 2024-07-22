import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from datetime import datetime
import os

# 确定当前工作目录
script_dir = os.path.dirname(__file__)
os.chdir(script_dir)

# 加载数据
save_path = os.path.join(script_dir, 'animal_food_preferences.csv')
df = pd.read_csv(save_path)

# 编码
animal_tokenizer = Tokenizer()
animal_tokenizer.fit_on_texts(df['animal'])
food_tokenizer = Tokenizer()
food_tokenizer.fit_on_texts(df['food'])
time_tokenizer = Tokenizer()
time_tokenizer.fit_on_texts(df['time_of_day'])

# 加载模型
model_save_path = os.path.join(script_dir, 'animal_food_time_recommendation_model.h5')
model = tf.keras.models.load_model(model_save_path)
print("模型已加载")

# 推荐函数
def recommend_foods(animal, time_of_day, top_k=1):
    animal_id = animal_tokenizer.texts_to_sequences([animal])[0][0]
    time_id = time_tokenizer.texts_to_sequences([time_of_day])[0][0]
    food_ids = np.array(range(1, len(food_tokenizer.word_index) + 1))
    animal_array = np.array([animal_id] * len(food_ids))
    time_array = np.array([time_id] * len(food_ids))
    predictions = model.predict([animal_array, food_ids, time_array])
    
    # 打印预测值以检查
    print(f"Predictions for {animal} at {time_of_day}: {predictions.flatten()}")
    
    # 获取评分最高的前k个食物
    top_foods_idx = predictions.flatten().argsort()[-top_k:][::-1]
    top_foods = [food_tokenizer.index_word[food_ids[i]] for i in top_foods_idx]
    
    return top_foods

# 当前时间段
current_hour = datetime.now().hour
if current_hour < 12:
    current_time_of_day = 'Morning'
elif current_hour < 18:
    current_time_of_day = 'Afternoon'
else:
    current_time_of_day = 'Evening'

# 打印当前时间
print(f"当前时间: {datetime.now()}")

# 打印所有动物在当前时间段的推荐结果
animals = ['Dog', 'Cat', 'bird']
all_recommendations = {animal: recommend_foods(animal, current_time_of_day, top_k=3) for animal in animals}

for animal, foods in all_recommendations.items():
    print(f"Recommended foods for {animal} at {current_time_of_day}: {foods}")
