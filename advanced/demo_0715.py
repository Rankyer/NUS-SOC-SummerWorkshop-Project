import paho.mqtt.client as mqtt
import cv2
import numpy as np
import os
from PIL import Image
from tensorflow.keras.models import load_model # type: ignore
from datetime import datetime
from collections import Counter, defaultdict
import json
import threading
import queue
import time
script_dir = os.path.dirname(__file__)
os.chdir(script_dir)



# 类别字典
# food_dict = {0: 'apple', 1: 'banana', 2: 'grape', 3: 'orange', 4: 'watermelon'}
# animal_dict = {0: 'bird',1: 'cat',2: 'deer',3: 'dog',4: 'frog'}
food_dict = {0: '苹果', 1: '香蕉', 2: 'grape', 3: 'orange', 4: 'watermelon'}
animal_dict = {0: '鸟',1: '猫',2: 'deer',3: 'dog',4: 'frog'}

# 模型文件名
# FOOD_MODEL = "D:\\Shanghaitech\\NUS_SOC\\project_baseline\\NUS-Robotics-DL-project\\advanced\\fruit.keras"
# ANIMAL_MODEL = "D:\\Shanghaitech\\NUS_SOC\\project_baseline\\NUS-Robotics-DL-project\\advanced\\cifar_animals.keras"

FOOD_MODEL = "./fruit.keras"
ANIMAL_MODEL = "./cifar_animals.keras"

# MQTT Broker 的地址和端口号
broker_address = "172.25.100.123"
broker_port = 1883
# MQTT 主题
mqtt_topic_video = "camera/video"
mqtt_topic_color = "camera/color"
mqtt_topic_queue = "transport/queue"
mqtt_topic_result = "transport/result"
mqtt_completion_topic = "transport/completion"  # 用于通知任务完成


# 图片保存目录
save_dir = "received_images_without_yolo"
os.makedirs(save_dir, exist_ok=True)

# 分类函数
def classify_food(model, image):
    result = model.predict(image)
    themax = np.argmax(result)
    return food_dict[themax], result[0][themax], themax

def classify_animal(model, image):
    result = model.predict(image)
    themax = np.argmax(result)
    return animal_dict[themax], result[0][themax], themax

# 加载图像
def load_image(image_fname,size):
    img = Image.open(image_fname).convert('RGB')
    img = img.resize((size, size))
    imgarray = np.array(img) / 255.0
    final = np.expand_dims(imgarray, axis=0)
    return final

def detect_and_crop_subimages(image_path, save_with_boxes=False, min_area=40000):
    # 读取图像
    image = cv2.imread(image_path)
    original_image = image.copy()

    # 转换为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 应用高斯模糊来减少噪声
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 使用中值滤波来减少噪声
    blurred = cv2.medianBlur(blurred, 5)

    # 使用Canny算法进行边缘检测
    edges = cv2.Canny(blurred, 50, 150)

    # 应用形态学操作
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.erode(edges, kernel, iterations=1)

    # 查找边缘的轮廓
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    subimages = []
    bounding_boxes = []
    for contour in contours:
        if cv2.contourArea(contour) > min_area:  # 筛选掉面积小的轮廓
            x, y, w, h = cv2.boundingRect(contour)
            subimage = image[y:y+h, x:x+w]  # 裁剪出子图
            subimages.append(subimage)
            bounding_boxes.append((x, y, w, h))
            # 在原图上画出边框
            if save_with_boxes:
                cv2.rectangle(original_image, (x, y), (x+w, y+h), (0, 255, 0), 2)

    save_dir = "./cropped_images"
    os.makedirs(save_dir, exist_ok=True)
    for i, subimage in enumerate(subimages):
        subimage_path = os.path.join(save_dir, f"cropped_{i+1}.png")
        cv2.imwrite(subimage_path, subimage)
        print(f"Saved cropped image: {subimage_path}")

    return subimages

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Connected to MQTT Broker!")
        client.subscribe(mqtt_topic_video)
        client.subscribe(mqtt_topic_color)
        client.subscribe(mqtt_topic_queue)
    else:
        print(f"Failed to connect, return code {rc}")

def on_message(client, userdata, msg):
    global current_frame
    global last_color
    # global processing_color
    global current_color

    # processing_color = False
    try:
        if msg.topic == mqtt_topic_video:
            # 将消息转换为图像
            np_arr = np.frombuffer(msg.payload, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if frame is not None:
                process_video_frame(frame)

            else:
                print("Failed to decode video frame.")

        elif msg.topic == mqtt_topic_color:
            color = msg.payload.decode('utf-8')
            
            # 检查颜色是否变化
            if color != last_color:   
                print(f"Received color: {color}")             
                current_color = color
                if (current_color!=last_color):
                    print(current_color)
                    print(last_color)
                    print(color)
                    last_color = color
                    print(111)
                    print(current_color)
                    print(last_color)
                    print(color)
                    processing_thread = threading.Thread(target=process_frame_with_color, args=(client,))
                    processing_thread.start()

        elif msg.topic == mqtt_topic_queue:
            queue_message = msg.payload.decode('utf-8')
            print(f"Received queue message: {queue_message}")
            queue_thread = threading.Thread(target=process_queue_message, args=(queue_message,))
            queue_thread.start()

    except Exception as e:
        print(f"An error occurred: {e}")

def process_video_frame(frame):
    global current_frame
    try:
        # 显示接收到的视频帧
        cv2.imshow("Received Video", frame)
        current_frame = frame
        cv2.waitKey(1)
    except Exception as e:
        print(f"An error occurred while processing video frame: {e}")

def process_queue_message(queue_message):
    global queue_data
    global tmp

    try:
        animal, food = queue_message.split(" 和 ")
        queue_data = {"animal": animal, "food": food}
        tmp = 1

        tmp=1
    except Exception as e:
        print(f"Failed to process queue message: {e}")

def process_frame_with_color(client):
    global current_frame
    global current_color
    global queue_data
    print(tmp)

    global queue_data

    if queue_data:
        print(queue_data)

    try:
        while(1):   
            if queue_data:
                print(queue_data)
                print(f"Processing frame with color: {last_color}, queue item: {queue_data}")

                current_time = datetime.now()
                formatted_time = current_time.strftime('%Y-%m-%d_%H-%M-%S')
                filename = os.path.join(save_dir, f"image_{formatted_time}.jpg")
                os.makedirs(os.path.dirname(filename), exist_ok=True)
                cv2.imwrite(filename, current_frame)
                print(f"Saved image: {filename}")
                image = Image.open(filename)
                subimages = detect_and_crop_subimages(filename)
                print(current_color)

                if current_color.lower() == 'b':

                    # client.publish(mqtt_topic_result, str(True))
                    # print("published!")
                    # break

                    global food_model
                    print("Start classifying food.")
                    label_counts = Counter()
                    label_probs = defaultdict(float)
                    if(len(subimages)==0):
                        print("No subimages detected.")
                    else:
                        for idx, subimage in enumerate(subimages):
                            print(1)
                            crop_filename = os.path.join(save_dir, f'{formatted_time}_subimage_{idx}.png')
                            os.makedirs(os.path.dirname(crop_filename), exist_ok=True)
                            cv2.imwrite(crop_filename, subimage)
                            print(f'Saved {crop_filename}')
                            img = load_image(crop_filename, 249)
                            label, prob, _ = classify_food(food_model, img)
                            print("We think with certainty %3.2f that image %s is %s." % (prob, crop_filename, label))
                            label_counts[label] += 1
                            label_probs[label] += prob

                        most_frequent_labels = [label for label, count in label_counts.items() if count == max(label_counts.values())]
                        if len(most_frequent_labels) > 1:
                            most_probable_label = max(most_frequent_labels, key=lambda label: label_probs[label])
                        else:
                            most_probable_label = most_frequent_labels[0]

                        print(f"The most frequent and highest probability label is: {most_probable_label}")

                        # 判断与队列中的食物是否匹配
                        if queue_data['food'] == most_probable_label:
                            client.publish(mqtt_topic_result, str(True))
                        else:
                            client.publish(mqtt_topic_result, str(False))
                        break

                elif current_color.lower() == 'g':

                    # client.publish(mqtt_topic_result, str(True))
                    # break

                    global animal_model
                    print("Start classifying animals.")
                    label_counts = Counter()
                    label_probs = defaultdict(float)
                    if(len(subimages)==0):
                        print("No subimages detected.")
                    else:
                        for idx, subimage in enumerate(subimages):
                            crop_filename = os.path.join(save_dir, f'{formatted_time}_subimage_{idx}.png')
                            os.makedirs(os.path.dirname(crop_filename), exist_ok=True)
                            cv2.imwrite(crop_filename, subimage)
                            print(f'Saved {crop_filename}')
                            img = load_image(crop_filename, 32)
                            label, prob, _ = classify_animal(animal_model, img)
                            print("We think with certainty %3.2f that image %s is %s." % (prob, crop_filename, label))
                            label_counts[label] += 1
                            label_probs[label] += prob

                        most_frequent_labels = [label for label, count in label_counts.items() if count == max(label_counts.values())]
                        if len(most_frequent_labels) > 1:
                            most_probable_label = max(most_frequent_labels, key=lambda label: label_probs[label])
                        else:
                            most_probable_label = most_frequent_labels[0]

                        print(f"The most frequent and highest probability label is: {most_probable_label}")

                        # 判断与队列中的动物是否匹配
                        if queue_data['animal'] == most_probable_label:
                            client.publish(mqtt_topic_result, str(True))
                            queue_data.pop(0)  # 移除已处理的队列项
                        else:
                            client.publish(mqtt_topic_result, str(False))
                        break
            
                
    except Exception as e:
            print(f"An error occurred during processing: {e}")
    # finally:
    #     processing_color = False

def receive():
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message

    client.connect(broker_address, broker_port)
    client.loop_forever()

def main():
    global food_model
    global animal_model
    global current_color
    global queue_data
    global last_color
    global tmp
    tmp = 0

    current_color = None
    queue_data = None
    last_color = None

    print("Loading classification model from ", FOOD_MODEL)
    food_model = load_model(FOOD_MODEL)
    print("Loading classification model from ", ANIMAL_MODEL)
    animal_model = load_model(ANIMAL_MODEL)

    print("Done")
    
    receive()

if __name__ == '__main__':
    main()