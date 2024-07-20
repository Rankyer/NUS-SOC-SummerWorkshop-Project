#include "car.h"

void setup() {
  // put your setup code here, to run once:
   //move_forward(0);
  stop_motors();
  Serial.begin(115200);
}

void clearInputBuffer() {
  while (Serial.available() > 0) {
    Serial.read();
  }
}
 
void loop()
{
//  static unsigned long last_time = 0;
//  unsigned long curr_time = millis();
//  if (curr_time - last_time > 100) {
//    clearInputBuffer();
//    last_time = curr_time;
//  }
  if ( Serial.available()){
    switch (Serial.read()){
      stop_motors();
      case 'c':
        turn_spin(100,0,true);
//        Serial.write('c');///
        break;
      case 'z':
        turn_spin(0,100,false);
//        Serial.write('z');/
        break;
      case 'q':
        turn_dir(100,100/2,true);
//        Serial.write('q');/
        break;
      case 'e':
        turn_dir(100/2,100,true);
//        Serial.write('e');/
        break;
      case 'w':
        move_forward(100);
//        Serial.write('w');
        break;
      case 'a':
        turn_dir(100,100/2,false);
//        Serial.write('a');/
        break;
      case 'd':
        turn_dir(100/2,100,false);
//        Serial.write('d');/
        break;
      case 's':
        move_backward(100);
//        Serial.write('s');/
        break;
      case 'n':
        turn_spin(255,255,true);
//        Serial.write('n');/
        break;
      case 'v':
        turn_spin(255,255,false);
//        Serial.write('v');/
        break;
      case 'r':
        turn_dir(255,100,true);
//        Serial.write('r');/
        break;
      case 'y':
        turn_dir(100,255,true);
//        Serial.write('y');/
        break;
      case 't':
        move_forward(255);
//        Serial.write('t');/
        break;
      case 'f':
        turn_dir(255,100,false);
//        Serial.write('f');/
        break;
      case 'h':
        turn_dir(100,255,false);
//        Serial.write('h');/
        break;
      case 'g':
        move_backward(255);
//        Serial.write('g');/
        break;        
      case ' ':
        stop_motors();
//        Serial.write(' ');
        break;
    }

  }
}
