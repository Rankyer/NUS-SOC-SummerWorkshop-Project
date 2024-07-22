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

script_dir = os.path.dirname(__file__)
os.chdir(script_dir)

# 类别字典
food_dict = {0: 'apple', 1: 'banana', 2: 'grape', 3: 'orange', 4: 'watermelon'}
animal_dict = {0: 'bird', 1: 'cat', 2: 'deer', 3: 'dog', 4: 'frog'}
num_dict = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9'}

# 模型文件名
FOOD_MODEL = "./food_recognition_plus/fruit.keras"
ANIMAL_MODEL = "./animal_recognition/cifar.keras"
NUM_MODEL = "./num_recognition/mnist-cnn.keras"

# MQTT Broker 的地址和端口号
broker_address = "172.25.97.190"
broker_port = 1883
# MQTT 主题
mqtt_topic = "camera/video"

# 图片保存目录
save_dir = "received_images_without_yolo"
os.makedirs(save_dir, exist_ok=True)


# YOLOv7 模型文件名
YOLOV7_MODEL_PATH = "yolov7.pt"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
yolo_model = attempt_load(YOLOV7_MODEL_PATH, map_location=device)
yolo_model.to(device).eval()


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

# YOLOv7 目标检测
def detect_objects_yolo(image):
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (640, 640))  # YOLOv7模型默认输入尺寸
    img = img.transpose((2, 0, 1))  # HWC to CHW
    img = np.expand_dims(img, axis=0)
    img = torch.from_numpy(img).to(device)
    img = img.half() if device.type != 'cpu' else img.float()  # uint8 to fp16/32 if using CUDA
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    
    with torch.no_grad():
        pred = yolo_model(img)[0]
    pred = non_max_suppression(pred, 0.25, 0.45)
    return pred

# MQTT 连接回调
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Connected to MQTT Broker!")
        client.subscribe(mqtt_topic)
    else:
        print(f"Failed to connect, return code {rc}")

# MQTT 消息回调
def on_message(client, userdata, msg):
    try:
        np_arr = np.frombuffer(msg.payload, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        if frame is not None:
            cv2.imshow("Received Video", frame)
            
            key = cv2.waitKey(1) & 0xFF

            # 识别食物
            if key == ord('p'):
                filename = os.path.join(save_dir, f"image_{time.time()}.jpg")
                cv2.imwrite(filename, frame)
                print(f"Saved image: {filename}")
                
                # 使用 YOLOv7 进行目标检测
                # 使用 YOLOv7 进行目标检测
                image = Image.open(filename)
                detections = detect_objects_yolo(image)

                for i, det in enumerate(detections):
                    if det is not None and len(det):
                        det[:, :4] = scale_coords(torch.tensor([640, 640]), det[:, :4], frame.shape).round()
                        for *xyxy, conf, cls in det:
                            x1, y1, x2, y2 = map(int, xyxy)
                            cropped_image = frame[y1:y2, x1:x2]
                            crop_filename = os.path.join(save_dir, f"crop_{time.time()}_{i}.jpg")
                            cv2.imwrite(crop_filename, cropped_image)
                            print(f"Cropped and saved: {crop_filename}")
                            
                            # 分类
                            img = load_image(crop_filename,249)
                            label, prob, _ = classify_food(food_model, img)
                            print("We think with certainty %3.2f that image %s is %s." % (prob, crop_filename, label))
            # 识别动物
            elif key == ord('o'):
                filename = os.path.join(save_dir, f"image_{time.time()}.jpg")
                cv2.imwrite(filename, frame)
                print(f"Saved image: {filename}")
                
                # 使用 YOLOv7 进行目标检测
                image = Image.open(filename)
                detections = detect_objects_yolo(image)
                
                for i, det in enumerate(detections):
                    if det is not None and len(det):
                        det[:, :4] = scale_coords(torch.tensor([640, 640]), det[:, :4], frame.shape).round()
                        for *xyxy, conf, cls in det:
                            x1, y1, x2, y2 = map(int, xyxy)
                            cropped_image = frame[y1:y2, x1:x2]
                            crop_filename = os.path.join(save_dir, f"crop_{time.time()}_{i}.jpg")
                            cv2.imwrite(crop_filename, cropped_image)
                            print(f"Cropped and saved: {crop_filename}")
                            
                            # 分类
                            img = load_image(crop_filename,32)
                            label, prob, _ = classify_animal(animal_model, img)
                            print("We think with certainty %3.2f that image %s is %s." % (prob, crop_filename, label))

            # 识别货架
            elif key == ord('i'):
                filename = os.path.join(save_dir, f"image_{time.time()}.jpg")
                cv2.imwrite(filename, frame)
                print(f"Saved image: {filename}")
                
                # 使用 YOLOv7 进行目标检测
                image = Image.open(filename)
                detections = detect_objects_yolo(image)
                
                for i, det in enumerate(detections):
                    if det is not None and len(det):
                        det[:, :4] = scale_coords(torch.tensor([640, 640]), det[:, :4], frame.shape).round()
                        for *xyxy, conf, cls in det:
                            x1, y1, x2, y2 = map(int, xyxy)
                            cropped_image = frame[y1:y2, x1:x2]
                            crop_filename = os.path.join(save_dir, f"crop_{time.time()}_{i}.jpg")
                            cv2.imwrite(crop_filename, cropped_image)
                            print(f"Cropped and saved: {crop_filename}")
                            
                            # 分类
                            img = load_image(crop_filename,28)
                            label, prob, _ = classify_num(num_model, img)
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
