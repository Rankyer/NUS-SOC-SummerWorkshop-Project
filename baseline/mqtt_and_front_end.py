from flask import Flask, render_template, request, jsonify
import serial
import paho.mqtt.client as mqtt
from picamera2 import Picamera2, Preview
import io
import threading
import time

# MQTT Broker 的地址和端口号
broker_address = "172.25.97.132"
broker_port = 1883
# MQTT 主题
mqtt_topic = "camera/video"


ser = serial.Serial('/dev/ttyACM0', 115200, timeout=1)

mapping = {0: 'n', 1: 'v', 2: 'r', 3: 'y', 4: 't',
5: 'f', 6: 'h', 7: 'g' , 8: ' ', 9: '0'}
app = Flask(__name__)

# 定义按键映射关系
key_mapping = {
    'e': 0,
    'q': 1,
    'a': 2,
    'd': 3,
    'w': 4,
    'z': 5,
    'c': 6,
    's': 7,
    ' ': 8,  # 默认情况，即没有按键按下时输出空格键（8）
    'default': 9
}

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
                
            # 发布视频帧到 MQTT 主题
            result = client.publish(mqtt_topic, frame, qos=1)
            status = result.rc
            if status == 0:
                print(f"Sent frame to topic {mqtt_topic}")
            else:
                print(f"Failed to send message to topic {mqtt_topic}")

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


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/check_keys', methods=['POST'])
def check_keys():
    direction = request.form.get('direction', '')

    # 根据按键组合查找映射关系
    current_output = key_mapping.get(direction, key_mapping['default'])
    print(current_output)
    key_button = False
    try:
        if(key_button == True):
                            # 两种自转 前进转弯 后退转弯 停车
            ser.write(mapping[current_output].encode('utf-8'))
        else:
            # 两种自转 前进转弯 后退转弯 停车
            ser.write(mapping[current_output].encode('utf-8'))

            #print(key_value)
        response = ser.readline()
            #print(response.decode('utf-8'))
    except KeyboardInterrupt:
        ser.close()
    # 返回当前输出
    return jsonify({'current_output': current_output})


if __name__ == '__main__':
    #app.run(debug=True, host='0.0.0.0', port=5000)
    t1 = threading.Thread(target=publish_video)
    t2 = threading.Thread(target=app.run, args=('0.0.0.0', 5000))
    
    t1.start()
    t2.start()
    
    t1.join()
    t2.join()
