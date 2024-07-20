#include <AFMotor.h>

// Declare four motors
AF_DCMotor motor4(4);
AF_DCMotor motor3(3);
AF_DCMotor motor2(2);
AF_DCMotor motor1(1);

void stop_motors() {
  motor1.run(RELEASE);
  motor2.run(RELEASE);
  motor3.run(RELEASE);
  motor4.run(RELEASE);
}

void move_forward(int speed) {
  motor1.run(FORWARD);
  motor2.run(FORWARD);
  motor3.run(FORWARD);
  motor4.run(FORWARD);

  motor1.setSpeed(speed);
  motor2.setSpeed(speed);
  motor3.setSpeed(speed);
  motor4.setSpeed(speed);
}

void move_backward(int speed) {
  motor1.run(BACKWARD);
  motor2.run(BACKWARD);
  motor3.run(BACKWARD);
  motor4.run(BACKWARD);

  motor1.setSpeed(speed);
  motor2.setSpeed(speed);
  motor3.setSpeed(speed);
  motor4.setSpeed(speed);
}

// if is_forward = 1, means the car will move forward, otherwise will go backward
void turn_spin(int speed_left, int speed_right, bool is_forward=true) {
  if (is_forward) {
  motor1.run(FORWARD);
  motor2.run(BACKWARD);
  motor3.run(BACKWARD);
  motor4.run(FORWARD);
  }

  else {
  motor1.run(BACKWARD);
  motor2.run(FORWARD);
  motor3.run(FORWARD);
  motor4.run(BACKWARD);
  }

  motor1.setSpeed(speed_right);
  motor2.setSpeed(speed_left);
  motor3.setSpeed(speed_left);
  motor4.setSpeed(speed_right);
}
void turn_dir(int speed_left, int speed_right, bool is_forward=true) {
  if (is_forward) {
  motor1.run(FORWARD);
  motor2.run(FORWARD);
  motor3.run(FORWARD);
  motor4.run(FORWARD);
  }

  else {
  motor1.run(BACKWARD);
  motor2.run(BACKWARD);
  motor3.run(BACKWARD);
  motor4.run(BACKWARD);
  }

  motor1.setSpeed(speed_right);
  motor2.setSpeed(speed_left);
  motor3.setSpeed(speed_left);
  motor4.setSpeed(speed_right);
}
