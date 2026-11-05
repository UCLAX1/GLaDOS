


#include <Servo.h> 
int servoPin = 3; //the pin that the servo will be attached to 
Servo servo1; // creates a sevo object
// 180: fully up
// 120: fully down
int val = 180; 
void setup() { 
  Serial.begin(9600);
  servo1.attach(servoPin); 
  Serial.setTimeout(-1);
}
void loop(){ 
  if(Serial.available() > 0){
    val = Serial.parseInt();
    Serial.print("setting to ");
    Serial.println(val);

  }

  servo1.write(val);
}
// end of code 
