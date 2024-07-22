import paho.mqtt.client as mqtt
import cv2
import numpy as np
import io
import time
import os
from PIL import Image
from tensorflow.keras.models import load_model
import torch
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from datetime import datetime

current_time = datetime.now()
formatted_time = current_time.strftime('%Y-%m-%d_%H-%M-%S')
script_dir = os.path.dirname(__file__)
os.chdir(script_dir)

# ########################################################################
# This file can be used to test model accuracy when receiving specific key

# 类别字典
food_dict = {0: 'apple', 1: 'banana', 2: 'grape', 3: 'orange', 4: 'watermelon'}
# animal_dict = {0: 'bird', 1: 'cat', 2: 'deer', 3: 'dog', 4: 'frog'}
animal_dict = {0: 'bird', 1: 'cat', 3: 'dog'}
num_dict = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9'}

# 模型文件名
# FOOD_MODEL = "./food_recognition_plus/fruit.keras"
# ANIMAL_MODEL = "./animal_recognition/cifar.keras"
# NUM_MODEL = "./num_recognition/mnist-cnn.keras"
FOOD_MODEL = "./fruit.keras"
ANIMAL_MODEL = "./animals.keras"
NUM_MODEL = "./num_recognition/mnist-cnn.keras"


# MQTT Broker 的地址和端口号
broker_address = "172.25.100.123"
broker_port = 1883
# MQTT 主题
mqtt_topic = "camera/video"

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

def classify_num(model, image):
    result = model.predict(image)
    themax = np.argmax(result)
    return num_dict[themax], result[0][themax], themax

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

    if save_with_boxes:
        current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        save_dir = "received_images_with_boxes"
        os.makedirs(save_dir, exist_ok=True)
        filename_with_boxes = os.path.join(save_dir, f"boxed_image_{current_time}.jpg")
        cv2.imwrite(filename_with_boxes, original_image)
        print(f"Saved boxed image: {filename_with_boxes}")

    return subimages



def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Connected to MQTT Broker!")
        client.subscribe(mqtt_topic)
    else:
        print(f"Failed to connect, return code {rc}")

def on_message(client, userdata, msg):
    try:
        # 将消息转换为图像
        np_arr = np.frombuffer(msg.payload, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        if frame is not None:
            cv2.imshow("Received Video", frame)
            
            # 按下 's' 键保存当前帧为图像文件
            key = cv2.waitKey(1) & 0xFF
            if key == ord('p'):
                filename = os.path.join(save_dir, f"image_{formatted_time}.jpg")
                os.makedirs(os.path.dirname(filename), exist_ok=True)
                cv2.imwrite(filename, frame)
                print(frame.shape)
                print(f"Saved image: {filename}")
                ########################
                image = Image.open(filename)
                subimages = detect_and_crop_subimages(filename)

                global food_model
                print("Start classifying.")
                for idx, subimage in enumerate(subimages):
                    crop_filename = os.path.join(save_dir, f'{formatted_time}_subimage_{idx}.png')
                    os.makedirs(os.path.dirname(crop_filename), exist_ok=True)
                    cv2.imwrite(crop_filename, subimage)
                    print(f'Saved {crop_filename}')
                    img = load_image(crop_filename, 249)
                    label, prob, _ = classify_food(food_model, img)
                    print("We think with certainty %3.2f that image %s is %s." % (prob, crop_filename, label))

            # 识别动物
            elif key == ord('o'):
                filename = os.path.join(save_dir, f"image_{time.time()}.jpg")
                os.makedirs(os.path.dirname(filename), exist_ok=True)
                cv2.imwrite(filename, frame)
                print(f"Saved image: {filename}")
                ########################
                image = Image.open(filename)
                subimages = detect_and_crop_subimages(filename)

                global animal_model
                print("Start classifying.")
                for idx, subimage in enumerate(subimages):
                    crop_filename = os.path.join(save_dir, f'{formatted_time}_subimage_{idx}.png')
                    os.makedirs(os.path.dirname(crop_filename), exist_ok=True)
                    cv2.imwrite(crop_filename, subimage)
                    print(f'Saved {crop_filename}')
                    img = load_image(crop_filename, 249)
                    label, prob, _ = classify_animal(animal_model, img)
                    print("We think with certainty %3.2f that image %s is %s." % (prob, crop_filename, label))

            elif key == ord('q'):
                client.disconnect()
                cv2.destroyAllWindows()
    except Exception as e:
        print(f"An error occurred: {e}")

def receive_video():
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message

    client.connect(broker_address, broker_port)
    client.loop_forever()

def main():

    global food_model
    global animal_model
    global num_model

    print("Loading classification model from ", FOOD_MODEL)
    food_model = load_model(FOOD_MODEL)
    print("Loading classification model from ", ANIMAL_MODEL)
    animal_model = load_model(ANIMAL_MODEL)
    print("Loading classification model from ", NUM_MODEL)
    num_model = load_model(NUM_MODEL)

    print("Done")
    
    receive_video()

if __name__ == '__main__':
    main()