#include <LiquidCrystal.h>
#include <Servo.h>
#include <EEPROM.h>

// LCD pins
LiquidCrystal lcd(2, 3, 4, 5, 6, 7);

// Buzzer
const int buzzer_Pin = 8;

// RGB LED
const int RED_LED = 9;
const int GREEN_LED = 10;
const int BLUE_LED = 11;

// Servo door lock
Servo doorServo;
const int servoPin = 12;

char incoming_char = 0;
String incoming_str = "";

// Door Open = 90°, Door Closed = 0°
int doorOpenAngle = 90;
int doorCloseAngle = 0;

void setup() {
  Serial.begin(9600);

  // Pins
  pinMode(buzzer_Pin, OUTPUT);

  pinMode(RED_LED, OUTPUT);
  pinMode(GREEN_LED, OUTPUT);
  pinMode(BLUE_LED, OUTPUT);

  doorServo.attach(servoPin);
  doorServo.write(doorCloseAngle); // door locked initially

  lcd.begin(16, 2);
  lcd.print("Student Face");
  lcd.setCursor(0, 1);
  lcd.print("Recognition...");
  delay(2000);
  lcd.clear();
  lcd.print("Ready!");
  delay(1000);
  lcd.clear();
}

// RGB helper
void setColor(int r, int g, int b) {
  digitalWrite(RED_LED, r);
  digitalWrite(GREEN_LED, g);
  digitalWrite(BLUE_LED, b);
}

void openDoor() {
  doorServo.write(doorOpenAngle);
  delay(1500);
  doorServo.write(doorCloseAngle);
}

void handleStudent(String studentName, int eepromAddr) {
  EEPROM.write(eepromAddr, 1); // Save attendance

  lcd.clear();
  lcd.print(studentName);
  lcd.setCursor(0, 1);
  lcd.print("Attendance Saved");

  setColor(0, 1, 0);  // GREEN
  digitalWrite(buzzer_Pin, LOW);

  openDoor();  // servo opens door

  delay(2000);
  setColor(0, 0, 0); // LED off
  lcd.clear();
}

void loop() {

  if (Serial.available() > 0) {
    incoming_str = Serial.readStringUntil('\n');
    incoming_str.trim();  // Remove whitespace
    incoming_str.toLowerCase();  // Convert to lowercase

    // Student: Zahir
    if (incoming_str == "zahir") {
      handleStudent("Zahir", 0);
    }

    // Student: Mehran
    else if (incoming_str == "mehran") {
      handleStudent("Mehran", 1);
    }

    // Student: Yousaf
    else if (incoming_str == "yousaf") {
      handleStudent("Yousaf", 2);
    }

    // Unknown Person
    else if (incoming_str == "unknown") {
      EEPROM.write(3, EEPROM.read(3) + 1); // count unknown attempts

      lcd.clear();
      lcd.print("Unknown Person");
      lcd.setCursor(0, 1);
      lcd.print("Access Denied!");

      setColor(1, 0, 0); // RED
      digitalWrite(buzzer_Pin, HIGH);
      delay(2000);

      digitalWrite(buzzer_Pin, LOW);
      setColor(0, 0, 0);
      lcd.clear();
    }

    // Ask to adjust face (no face detected)
    else if (incoming_str == "noface") {
      lcd.clear();
      lcd.print("Adjust Your Face");
      setColor(1, 1, 0); // YELLOW
      delay(2000);
      setColor(0, 0, 0);
      lcd.clear();
    }
  }
}
