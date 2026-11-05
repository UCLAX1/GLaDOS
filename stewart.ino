#include <Servo.h>

#define MAX 180
#define MIN 120
#define INTIAL 2
#define BASE 30 //mm from ball joint to rod pivot point
#define ARMLENGTH 15  
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


int theta = 0;
int phi = 0; 
char buffer[50];

void setup() {
  servo1.attach(SERVOPIN1);
  servo1.attach(SERVOPIN2);
  servo1.attach(SERVOPIN1);  // attaches the servo pins to the respective servo objects
  Serial.begin(9600);

  moveTo(90, 10);
}

void loop() {
 
}

void moveTo(int theta, int phi){
  Amplitude = sin((PI/180) * phi) * BASE/ ARMLENGTH;
  val1 = asin(Amplitude * sin((PI/180) * theta) + ((float)INTIAL/ARMLENGTH))* (180/PI);
  val1 = constrain(val1 + 150, 120, 180);  // constain it to use it with the servo (value between 120 and 180)
  servo1.write(val1);                // sets the servo position according to the scaled value

  val2 = asin(Amplitude * sin((PI/180) * theta + Offset) + ((float)INTIAL/ARMLENGTH))* (180/PI);
  val2 = constrain(val2 + 150, 120, 180);  // scale it to use it with the servo (value between 0 and 180)
  servo2.write(val2);                // sets the servo position according to the scaled value

  val3 = asin(Amplitude * sin((PI/180) * theta + (2 * Offset)) + ((float)INTIAL/ARMLENGTH)) * (180/PI);
  val3 = constrain(val3 + 150, 120, 180);  // scale it to use it with the servo (value between 0 and 180)
  servo3.write(val3);  

  Serial.print(val1);
  Serial.print(", ");
  Serial.print(val2);
  Serial.print(", ");
  Serial.println(val3);
}