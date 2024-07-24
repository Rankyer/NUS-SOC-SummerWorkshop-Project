#include "car.h"

// RGB 传感器引脚
#define S0 28
#define S1 30
#define S2 32
#define S3 34
#define OUT 36

// 传感器引脚定义
const int leftSensor = 22;
const int centerSensor = 24;
const int rightSensor = 26;

// 设置白平衡
int red_0, blue_0, green_0;

// 设置停车标识
bool stop_more = false;
bool direction = true;
bool spin_stop = false;
bool normal_track = true;

// 初始化函数
void setup() {
  Serial.begin(115200);   //串口波特率

  pinMode(leftSensor, INPUT);
  pinMode(centerSensor, INPUT);
  pinMode(rightSensor, INPUT);
  
  pinMode(S0, OUTPUT);
  pinMode(S1, OUTPUT);
  pinMode(S2, OUTPUT);
  pinMode(S3, OUTPUT);
  pinMode(OUT, INPUT);

  // RGB传感器
  // 设置频率标度为20%
  digitalWrite(S0, HIGH);
  digitalWrite(S1, LOW);
  // 初始化白平衡
  digitalWrite(S2, LOW);
  digitalWrite(S3, LOW);
  delay(2); // 增加延迟以确保设置生效
  red_0 = pulseIn(OUT, LOW);

  digitalWrite(S2, LOW);
  digitalWrite(S3, HIGH);
  delay(2); // 增加延迟以确保设置生效
  blue_0 = pulseIn(OUT, LOW);

  digitalWrite(S2, HIGH);
  digitalWrite(S3, HIGH);
  delay(2); // 增加延迟以确保设置生效
  green_0 = pulseIn(OUT, LOW);
  // 等待传感器稳定
  delay(2);
}

void loop() {
  int leftValue = digitalRead(leftSensor);
  int centerValue = digitalRead(centerSensor);
  int rightValue = digitalRead(rightSensor);

  if (normal_track && leftValue == LOW && centerValue == HIGH && rightValue == LOW) {
    move_forward(110);  // 中间传感器在黑线
    stop_more = false;
  } else if (normal_track && leftValue == HIGH && centerValue == LOW && rightValue == LOW) {
    turn_spin(130, 100, false);  // 左传感器在黑线
    stop_more = false;
  } else if (normal_track && leftValue == LOW && centerValue == LOW && rightValue == HIGH) {
    turn_spin(100, 130, true);  // 右传感器在黑线
    stop_more = false;
  } else if (normal_track && leftValue == HIGH && centerValue == HIGH && rightValue == HIGH) {
    stop_more = true;           //全黑
    if (!stop_more) {           //一般情况 向左转进入主干道
//    move_backward(100);
//    delay(10);
      turn_spin(100,100,false);

    }
//    stop_motors();
    if (spin_stop) {           //特殊停车
      stop_motors();
      delay(1000);
      move_backward(120); //设置后退距离
      delay(600);
      turn_spin(140, 140, false);
      delay(600);          // 设置转弯角度
      normal_track = false;   //开启归中判断
      spin_stop = false;
    }
  } else if (normal_track && leftValue == HIGH && centerValue == HIGH && rightValue == LOW) {
    turn_spin(130, 0, false);
    stop_more = false;
  } else if (normal_track && leftValue == LOW && centerValue == HIGH && rightValue == HIGH) {
    turn_spin(0, 130, true);
    stop_more = false;
  } else if(!normal_track){
        turn_spin(180, 150, false);
      if(leftValue == HIGH || centerValue == HIGH || rightValue == HIGH){
        normal_track = true;        //寻到黑线 结束归中
        stop_motors();
      }

  }

  // 选择红色过滤器
  digitalWrite(S2, LOW);
  digitalWrite(S3, LOW);
  delay(2); // 增加延迟以确保设置生效
  int red = pulseIn(OUT, LOW) - red_0;

  // 选择蓝色过滤器
  digitalWrite(S2, LOW);
  digitalWrite(S3, HIGH);
  delay(2); // 增加延迟以确保设置生效
  int blue = pulseIn(OUT, LOW) - blue_0;

  // 选择绿色过滤器
  digitalWrite(S2, HIGH);
  digitalWrite(S3, HIGH);
  delay(2); // 增加延迟以确保设置生效
  int green = pulseIn(OUT, LOW) - green_0;

  // 蓝色代表取货
  if (blue < red - 60 && blue < green - 60) { // (R,G,B) = (180-255, 80-120, 10-50)
    move_backward(100);
    delay(100);
    stop_stably();
    wait_resp('b');
  }

  // 红色表示到了对应的动物位置
  if (red < blue - 60 && red < green - 60) { // (R,G,B) = 160-255,70-130,90-150
    move_backward(100);
    delay(100);
    stop_stably();
    wait_resp('g');
  }


}  //主loop

void serialEvent() {    //串口loop
  while (Serial.available()) {
    char inChar = (char)Serial.read();
    if (inChar == 'F') {
      color_forward();
    } else if (inChar == 'R') {
      color_spin();
      normal_track = false;     //开启归中判断
    }
    clearSerialBuffer();
  }
}

void stop_stably() {
  move_backward(100);
  delay(10);
  stop_motors();
}

void wait_resp(char cmd) {
  //turn_spin(120, 120, direction);
  //delay(15);
  stop_motors();
  Serial.println(cmd);  //需要运行水果识别模型
  delay(1000);
  
  int N = 1; // 分N次旋转
  while (Serial.available() == 0) {
 //   direction = !direction;
 //   for (int i = 0; i < N; i++) {
 //     turn_spin(200, 200, direction);   //调整位置，放置识别一直无结果
 //     delay(30 / N);
      stop_motors();
      delay(1000);  // 稳定参数
 //   }
  }
}

void color_forward() {
  move_forward(120);
  delay(600);
}

void color_spin() {

  move_forward(120);
  delay(400);       // 设置前进距离
  turn_spin(120, 180, true); 
  delay(300);
  //move_forward(120);
  //delay(600);       // 设置前进距离
  int leftValue = digitalRead(leftSensor);
  int centerValue = digitalRead(centerSensor);
  while(leftValue == LOW && centerValue == LOW){
    turn_spin(200, 200, true);
    delay(80);
    leftValue = digitalRead(leftSensor);
    centerValue = digitalRead(centerSensor);
    move_forward(90);
  }
  move_backward(100); // 震荡停止
  delay(10);
  stop_motors();
  spin_stop = true;




//  move_backward(120); //设置后退距离
//  delay(550);

//  turn_spin(200, 200);
//  delay(800);          // 设置转弯角度
}

void clearSerialBuffer() {
  while (Serial.available() > 0) {
    Serial.read();
  }
}
