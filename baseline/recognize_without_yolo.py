import paho.mqtt.client as mqtt
import cv2
import numpy as np
import io
import time
import os
from PIL import Image
from tensorflow.keras.models import load_model
script_dir = os.path.dirname(__file__)
os.chdir(script_dir)

# classes = ["daisy", "dandelion", "roses", "sunflowers", "tulips"]
# classes = ["Pallas cats", "Persian cats", "Ragdolls", "Singapore cats", "Sphynx cats"]
dict={0:'Pallas_cats', 1:'Persian_cat', 2:'Ragdoll', 3:'Singapura_cats', 4:'Sphynx_cats'}

# Change this to the correct model name
# FILENAME="flowers.keras"
FILENAME="cats_plus1.keras"
model = None


# MQTT Broker 的地址和端口号
broker_address = "172.25.97.132"
broker_port = 1883
# MQTT 主题
mqtt_topic = "camera/video"

# 图片保存目录
save_dir = "received_images"

# 创建保存目录
os.makedirs(save_dir, exist_ok=True)

def classify(model, image):
    result = model.predict(image)
    themax = np.argmax(result)
    return (dict[themax], result[0][themax], themax)

# Load image
def load_image(image_fname):
    img = Image.open(image_fname)
    img = img.resize((249, 249))
    imgarray = np.array(img)/255.0
    final = np.expand_dims(imgarray, axis=0)
    return final

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
            if key == ord('s'):
                filename = os.path.join(save_dir, f"image_{time.time()}.jpg")
                cv2.imwrite(filename, frame)
                print(frame.shape)
                print(f"Saved image: {filename}")
                ########################
                global model
                print("Start classifying.")
                img = load_image(filename)
                label, prob, _ = classify(model, img)

                print("We think with certainty %3.2f that image %s is %s." % (prob, filename, label))

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

    global model

    print("Loading model from ", FILENAME)
    model = load_model(FILENAME)
    print("Done")
    receive_video()

if __name__ == '__main__':
    main()