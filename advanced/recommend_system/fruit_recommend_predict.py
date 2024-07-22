import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Concatenate, Dense, Dropout
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

df['animal_id'] = animal_tokenizer.texts_to_sequences(df['animal'])
df['food_id'] = food_tokenizer.texts_to_sequences(df['food'])
df['time_id'] = time_tokenizer.texts_to_sequences(df['time_of_day'])

df['animal_id'] = df['animal_id'].apply(lambda x: x[0])
df['food_id'] = df['food_id'].apply(lambda x: x[0])
df['time_id'] = df['time_id'].apply(lambda x: x[0])

# 模型定义
embedding_dim = 10
animal_input = Input(shape=(1,), name='animal_input')
food_input = Input(shape=(1,), name='food_input')
time_input = Input(shape=(1,), name='time_input')

animal_embedding = Embedding(input_dim=len(animal_tokenizer.word_index) + 1, output_dim=embedding_dim, name='animal_embedding')(animal_input)
food_embedding = Embedding(input_dim=len(food_tokenizer.word_index) + 1, output_dim=embedding_dim, name='food_embedding')(food_input)
time_embedding = Embedding(input_dim=len(time_tokenizer.word_index) + 1, output_dim=embedding_dim, name='time_embedding')(time_input)

animal_flat = Flatten()(animal_embedding)
food_flat = Flatten()(food_embedding)
time_flat = Flatten()(time_embedding)

concatenated = Concatenate()([animal_flat, food_flat, time_flat])
dense_1 = Dense(128, activation='relu')(concatenated)
dropout_1 = Dropout(0.5)(dense_1)
dense_2 = Dense(64, activation='relu')(dropout_1)
dropout_2 = Dropout(0.5)(dense_2)
output = Dense(1, activation='linear')(dropout_2)

model = Model([animal_input, food_input, time_input], output)
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# 数据准备
animal_ids = df['animal_id'].values
food_ids = df['food_id'].values
time_ids = df['time_id'].values
preferences = df['preference'].values

# 训练模型
model.fit([animal_ids, food_ids, time_ids], preferences, epochs=50, batch_size=32, validation_split=0.2)

# 保存模型
model_save_path = os.path.join(script_dir, 'animal_food_time_recommendation_model.h5')
model.save(model_save_path)
print(f"模型已保存为 '{model_save_path}'")
