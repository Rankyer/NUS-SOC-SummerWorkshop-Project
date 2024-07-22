from flask import Flask, render_template, request, jsonify, Response
import paho.mqtt.client as mqtt
import threading
import time
import json
import serial
import queue
from picamera2 import Picamera2, Preview
import io
import cv2
import numpy as np
import os
from PIL import Image 
from tensorflow.keras.models import load_model
from datetime import datetime
from collections import Counter, defaultdict
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from datetime import datetime
import os




app = Flask(__name__)
ser = serial.Serial('/dev/ttyACM0', 115200, timeout=1)


# Class Dict
food_dict = {0: 'apple', 1: 'banana', 2: 'grape', 3: 'orange', 4: 'watermelon'}
animal_dict = {0: 'bird',1: 'cat',2: 'dog'}

# Path of models
FOOD_MODEL_PATH = "/home/group2/Desktop/demo/fruit.keras"
ANIMAL_MODEL_PATH = "/home/group2/Desktop/demo/cifar_animals.keras"

# Loading models
print("Loading models...")
food_model = load_model(FOOD_MODEL_PATH)
animal_model = load_model(ANIMAL_MODEL_PATH)
print("Done loading...")

# queues
ser_color_curr = None
#queue_data = {'food': [], 'animal': []}
# queue_data = {'food': ['香蕉' for _ in range(100)], 'animal': ['猫' for _ in range(100)]}
queue_data = {'food': [], 'animal': []}
# MQTT Broker 的地址和端口号
broker_address = "172.25.97.132"
broker_port = 1883
# MQTT 主题
mqtt_topic = "camera/video"


# Communicate with RPi
def ser_send_data(msg):
    ser.write(msg.encode('utf-8'))
    time.sleep(1)

def ser_receive_data():
    global ser_color_curr
    while True:
        if ser.in_waiting > 0:
            line = ser.readline().decode('utf-8').rstrip()
            ser_color_curr = str(line)
            print(ser_color_curr)
                

# classify function
def classify_food(model, image):
    result = model.predict(image)
    themax = np.argmax(result)
    return food_dict[themax], result[0][themax], themax

def classify_animal(model, image):
    result = model.predict(image)
    themax = np.argmax(result)
    return animal_dict[themax], result[0][themax], themax

def detect_and_crop_subimages(image, save_dir=None, save_with_boxes=False, min_area=40000):
    img = image.copy()

    # convert to GRAY
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # applying GaussianBlur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply meadianBlur to diminish errors
    blurred = cv2.medianBlur(blurred, 5)

    # Canny Algor for side detect
    edges = cv2.Canny(blurred, 50, 150)

    # Moving Detect
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.erode(edges, kernel, iterations=1)

    # find Contours
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    sub_images = []
    bounding_boxes = []
    for contour in contours:
        if cv2.contourArea(contour) > min_area:  # Rule out img with smaller area
            x, y, w, h = cv2.boundingRect(contour)
            sub_img = img[y:y+h, x:x+w]  
            sub_images.append(sub_img)
            bounding_boxes.append((x, y, w, h))
            # drawing bbox
            if save_with_boxes:
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

    if not save_dir is None:
        date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for i, sub_image in enumerate(sub_images):
            sub_image_path = save_dir + "/" + date + "/cropped_"+ str(i+1) + ".png"
            cv2.imwrite(sub_image_path, sub_image)

    return sub_images

def img_preprocessing(image, size):
    img = image.copy()
    img = cv2.resize(img, (size, size))
    img_array = img / 255.0
    final = np.expand_dims(img_array, axis=0)
    return final

# processing function--------------------
def predict_and_send_msg(image):
    global ser_color_curr
    global food_lock
    global animal_lock
    subimages = detect_and_crop_subimages(image)

    if ser_color_curr == 'b':
        label_counts = Counter()
        label_probs = defaultdict(float)
        if len(subimages) == 0:
            print("No sub_images detected.")
        else:
            for idx, subimage in enumerate(subimages):
                sub_img_ = img_preprocessing(subimage, 249)
                label, prob, _ = classify_food(food_model, sub_img_)
                label_counts[label] += 1
                label_probs[label] += prob

            most_frequent_labels = [label for label, count in label_counts.items() if count == max(label_counts.values())]
            if len(most_frequent_labels) > 1:
                most_probable_label = max(most_frequent_labels, key=lambda label: label_probs[label])
            else:
                most_probable_label = most_frequent_labels[0]

            print(f"The most frequent and highest probability label is: {most_probable_label}")

            # 判断与队列中的食物是否匹配
            if food_lock and queue_data['food'][0] == most_probable_label:
                ser_send_data('R')
                ser_color_curr = None
                queue_data['food'].pop(0)

                global food
                food = most_probable_label

                food_lock = 0
                animal_lock = 1

            else:
                ser_send_data('F')
                ser_color_curr = None



    elif ser_color_curr == 'g':
        label_counts = Counter()
        label_probs = defaultdict(float)
        if len(subimages) == 0:
            print("No sub_images detected.")
        else:
            for idx, subimage in enumerate(subimages):
                sub_img_ = img_preprocessing(subimage, 249)
                label, prob, _ = classify_animal(animal_model, sub_img_)
                label_counts[label] += 1
                label_probs[label] += prob

            most_frequent_labels = [label for label, count in label_counts.items() if count == max(label_counts.values())]
            if len(most_frequent_labels) > 1:
                most_probable_label = max(most_frequent_labels, key=lambda label: label_probs[label])
            else:
                most_probable_label = most_frequent_labels[0]

            print(f"The most frequent and highest probability label is: {most_probable_label}")

            # 判断与队列中的animal是否匹配
            if animal_lock and queue_data['animal'][0] == most_probable_label:
                ser_send_data('R')
                print(111111)
                ser_color_curr = None
                queue_data['animal'].pop(0)
                global flag
                flag=1
                global animal
                animal = most_probable_label

                animal_lock = 0
                food_lock = 1


            else:
                ser_send_data('F')
                print(22222)
                ser_color_curr = None



completed_queue = []
def queue_worker():
    while True:
        global flag
        global animal
        global food
        if flag == 1:
            completed_item = f"{animal} and {food}"
            completed_queue.append(completed_item)
            notify_clients(completed_item)
            flag = 0
        time.sleep(1)

clients = []

def notify_clients(item):
    for client in clients:
        client.put(item)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/add_to_queue', methods=['POST'])
def add_to_queue():
    data = request.get_json()
    animal = data.get('animal')
    food = data.get('food')
    queue_data['food'].append(food)
    queue_data['animal'].append(animal)

    queue_item = f"{food} and {food}"
    return jsonify({'status': 'success', 'queue_item': queue_item})

@app.route('/queue_update')
def queue_update():
    def generate():
        q = queue.Queue()
        clients.append(q)
        try:
            while True:
                item = q.get()
                yield f'data: {json.dumps({"type": "completed", "item": item})}\n\n'
        except GeneratorExit:  # 当客户端断开连接时，移除该客户端
            clients.remove(q)

    return Response(generate(), content_type='text/event-stream')









def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Connected to MQTT Broker!")
    else:
        print(f"Failed to connect, return code {rc}")

def publish_video():
    client = mqtt.Client()
    client.on_connect = on_connect
    client.connect(broker_address, broker_port)
    client.loop_start()

    try:
        picam2 = Picamera2()
        camera_config = picam2.create_video_configuration(main={"size": (640, 480)})
        picam2.configure(camera_config)
        picam2.start()

        time.sleep(2)  # 等待摄像头初始化

        stream = io.BytesIO()
        while True:
            stream.seek(0)
            picam2.capture_file(stream, format='jpeg')
            frame = stream.getvalue()

            nparr = np.frombuffer(frame, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            # 开始预测
            t1 = threading.Thread(target=predict_and_send_msg, args=(img))
            t2 = threading.Thread(target=publish, args=(client, frame))

            t1.start()
            t2.start()

            t1.join()
            t2.join()    
            # if status == 0:
            #     print(f"Sent frame to topic {mqtt_topic}")
            # else:
            #     print(f"Failed to send message to topic {mqtt_topic}")

            # 清空缓冲区准备下一帧
            stream.seek(0)
            stream.truncate()

            time.sleep(0.0333)  # 控制发布频率

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        client.disconnect()
        client.loop_stop()
        picam2.stop()

def publish(client, frame):
    result = client.publish(mqtt_topic, frame, qos=1)
    status = result.rc

def camera():
    picam2 = Picamera2()
    camera_config = picam2.create_video_configuration(main={"size": (640, 480)})
    picam2.configure(camera_config)
    picam2.start()

    time.sleep(2)  # 等待摄像头初始化

    stream = io.BytesIO()
    while True:
        stream.seek(0)
        picam2.capture_file(stream, format='jpeg')
        frame = stream.getvalue()
        
        nparr = np.frombuffer(frame, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # 开始预测
        predict_and_send_msg(img)

        # 清空缓冲区准备下一帧
        stream.seek(0)
        stream.truncate()

        time.sleep(0.0333)  # 控制发布频率
    
    # picam2.stop()





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

@app.route('/get_recommendation', methods=['POST'])
def get_recommendation():
    data = request.get_json()
    animal = data.get('animal')
    current_time = datetime.now()
    current_hour = current_time.hour
    if current_hour < 12:
        time_of_day = 'Morning'
    elif current_hour < 18:
        time_of_day = 'Afternoon'
    else:
        time_of_day = 'Evening'
    
    recommendations = recommend_foods(animal, time_of_day, top_k=3)

    return jsonify({'status': 'success', 'recommendations': recommendations})


if __name__ == '__main__':
    global flag
    global animal
    global food
    global food_lock
    global animal_lock

    food_lock = 1
    animal_lock = 1

    flag = 0
    animal = None
    food = None

    t1 = threading.Thread(target=app.run, args=('0.0.0.0', 5000))
    t2 = threading.Thread(target=camera)
    t3 = threading.Thread(target=ser_receive_data)
    t4 = threading.Thread(target=queue_worker, daemon=True).start()
    #t4 = threading.Thread(target=publish_video)

    t1.start()
    t2.start()
    t3.start()
    t4.start()

    t1.join()
    t2.join()
    t3.join()
    t4.join()
