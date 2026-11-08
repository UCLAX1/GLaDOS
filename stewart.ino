#include <Servo.h>

#define MAX 180
#define MIN 120
#define INTIAL 2//TODO: change this value to the length of the ball joint rod minus the length of the push rod in mm
#define BASE 30 //mm from ball joint to rod pivot point 
#define ARMLENGTH 15 //TODO: change this value to the length of the servo arm in mm 
#define PI 3.1415926535897932384626433832795
#define SERVOOFFSET 150 //TODO: change this value to the angle of the servo when the platform is level (in degrees)
#define SERVOPIN1 9
#define SERVOPIN2 10
#define SERVOPIN3 11

Servo servo1;  // create servo object to control a servo
Servo servo2;
Servo servo3;

float val1;
float val2;
float val3;
float Amplitude;    
float Offset = (2 * PI) / 3;

void setup() {
  servo1.attach(SERVOPIN1);
  servo2.attach(SERVOPIN2);
  servo3.attach(SERVOPIN3);  // attaches the servo pins to the respective servo objects
  Serial.begin(9600);

  // Serial.setTimeout(-1);

  moveTo(90, 10);
}

void loop() {


  float t = (float)millis() / 1000;
  
  // int theta = 0;
  int theta = 10 * cos(t / PI);

  int phi = 10 * sin(t / PI);

  // if(Serial.available() > 0){
  //   val = Serial.parseInt();
  //   Serial.print("setting theta to ");
  //   Serial.println(val);

  //   theta = val;
  // }


  moveTo(theta, phi);
 
}

// Function to move the platform to a specific theta and phi angle
void moveTo(int theta, int phi) {
  Amplitude = sin((PI/180) * phi) * BASE/ ARMLENGTH;
  val1 = asin(Amplitude * sin((PI/180) * theta) + ((float)INTIAL/ARMLENGTH))* (180/PI);
  val1 = constrain(val1 + SERVOOFFSET, 120, 180);  // constain it to use it with the servo (value between 120 and 180)
  servo1.write(val1);                // sets the servo position according to the scaled value

  val2 = asin(Amplitude * sin((PI/180) * theta + Offset) + ((float)INTIAL/ARMLENGTH))* (180/PI);
  val2 = constrain(val2 + SERVOOFFSET, 120, 180);  // scale it to use it with the servo (value between 0 and 180)
  servo2.write(val2);                // sets the servo position according to the scaled value

  val3 = asin(Amplitude * sin((PI/180) * theta + (2 * Offset)) + ((float)INTIAL/ARMLENGTH)) * (180/PI);
  val3 = constrain(val3 + SERVOOFFSET, 120, 180);  // scale it to use it with the servo (value between 0 and 180)
  servo3.write(val3);  

  // Print the servo angles to the serial monitor for debugging
  Serial.print(val1);
  Serial.print(", ");
  Serial.print(val2);
  Serial.print(", ");
  Serial.println(val3);
}