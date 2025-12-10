"""
COMPLETE EMBEDDED DOOR CONTROL SYSTEM
For Face Recognition Attendance with Automatic Door Access

Hardware Control:
- Known Student: Door opens, Green LED, Success beep, Mark attendance
- Unknown Person: Door locked, Red LED, Warning beep, No access
"""

import time
import sys

# Try importing GPIO for real hardware
try:
    import RPi.GPIO as GPIO
    HARDWARE_MODE = True
except ImportError:
    HARDWARE_MODE = False
    print("Running in SIMULATION mode (Proteus/Testing)")

class EmbeddedDoorSystem:
    """
    Complete embedded system for automatic door control
    Compatible with Proteus simulation and Raspberry Pi hardware
    """
    
    def __init__(self, mode='auto'):
        """
        Initialize embedded door control system
        
        Args:
            mode: 'auto', 'simulation', or 'hardware'
        """
        # Auto-detect mode
        if mode == 'auto':
            self.mode = 'hardware' if HARDWARE_MODE else 'simulation'
        else:
            self.mode = mode
        
        # ===== GPIO PIN CONFIGURATION =====
        # For Proteus: Use Arduino/ATmega328 pins
        # For Pi: Use BCM GPIO numbering
        
        self.RELAY_DOOR_LOCK = 17    # Relay for solenoid door lock
        self.LED_GREEN = 27          # Success indicator (access granted)
        self.LED_RED = 22            # Failure indicator (access denied)
        self.BUZZER = 23             # Audio feedback
        self.LCD_RS = 25             # LCD Register Select (optional)
        self.LCD_EN = 24             # LCD Enable (optional)
        
        # ===== TIMING CONFIGURATION =====
        self.DOOR_UNLOCK_DURATION = 5    # Seconds door stays unlocked
        self.DENIED_LED_DURATION = 3     # Seconds red LED stays on
        self.BEEP_DURATION = 0.2        # Beep length
        
        # ===== SYSTEM STATE =====
        self.door_locked = True
        self.access_log = []
        
        # Initialize hardware
        if self.mode == 'hardware':
            self._init_gpio()
        else:
            print("\n" + "="*60)
            print("🔧 PROTEUS SIMULATION MODE ACTIVE")
            print("="*60)
            print("Connect components in Proteus as follows:")
            print(f"  Pin {self.RELAY_DOOR_LOCK} → Relay → Door Lock Solenoid")
            print(f"  Pin {self.LED_GREEN} → Green LED → 220Ω → GND")
            print(f"  Pin {self.LED_RED} → Red LED → 220Ω → GND")
            print(f"  Pin {self.BUZZER} → Buzzer → GND")
            print("="*60 + "\n")
    
    def _init_gpio(self):
        """Initialize GPIO pins for Raspberry Pi"""
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        
        # Configure output pins
        pins = [self.RELAY_DOOR_LOCK, self.LED_GREEN, self.LED_RED, self.BUZZER]
        for pin in pins:
            GPIO.setup(pin, GPIO.OUT, initial=GPIO.LOW)
        
        print("✅ GPIO Initialized - Hardware Ready")
    
    def grant_access(self, student_name, student_id=None):
        """
        GRANT ACCESS to known student
        1. Unlock door
        2. Turn on green LED
        3. Success beep
        4. Log attendance
        5. Lock door after timeout
        
        Args:
            student_name: Name of recognized student
            student_id: Optional student ID number
        """
        print("\n" + "="*60)
        print(f"✅ ACCESS GRANTED - Welcome {student_name}!")
        print("="*60)
        
        # Step 1: Unlock door (activate relay)
        self._control_relay(True)
        self.door_locked = False
        print(f"🔓 DOOR: UNLOCKED")
        
        # Step 2: Green LED ON
        self._control_led('green', True)
        print(f"💚 GREEN LED: ON")
        
        # Step 3: Success beeps (2 short beeps)
        self._beep_pattern('success')
        print(f"🔊 BUZZER: Beep-Beep (Welcome)")
        
        # Step 4: Log access
        self._log_access(student_name, 'GRANTED', student_id)
        
        # Step 5: Display on LCD (if available)
        self._display_message(f"Welcome {student_name}", "Access Granted")
        
        # Step 6: Keep door open for configured time
        print(f"⏱️  Door will remain unlocked for {self.DOOR_UNLOCK_DURATION} seconds...")
        time.sleep(self.DOOR_UNLOCK_DURATION)
        
        # Step 7: Lock door again
        self._control_relay(False)
        self.door_locked = True
        print(f"🔒 DOOR: LOCKED")
        
        # Step 8: Turn off green LED
        self._control_led('green', False)
        print(f"⚫ GREEN LED: OFF")
        
        print("="*60 + "\n")
        
        return True
    
    def deny_access(self, reason="Unknown person"):
        """
        DENY ACCESS to unknown person
        1. Keep door locked
        2. Turn on red LED
        3. Warning beeps
        4. Log attempt
        
        Args:
            reason: Reason for access denial
        """
        print("\n" + "="*60)
        print(f"🚫 ACCESS DENIED - {reason}")
        print("="*60)
        
        # Step 1: Ensure door is locked
        self._control_relay(False)
        self.door_locked = True
        print(f"🔒 DOOR: LOCKED (Secure)")
        
        # Step 2: Red LED ON
        self._control_led('red', True)
        print(f"❌ RED LED: ON (Warning)")
        
        # Step 3: Warning beeps (3 long beeps)
        self._beep_pattern('denied')
        print(f"🔊 BUZZER: Beep-Beep-Beep (Alert)")
        
        # Step 4: Log denied access
        self._log_access("UNKNOWN", 'DENIED', reason=reason)
        
        # Step 5: Display warning
        self._display_message("ACCESS DENIED", "Unknown Person")
        
        # Step 6: Keep red LED on for warning duration
        print(f"⏱️  Warning indicator active for {self.DENIED_LED_DURATION} seconds...")
        time.sleep(self.DENIED_LED_DURATION)
        
        # Step 7: Turn off red LED
        self._control_led('red', False)
        print(f"⚫ RED LED: OFF")
        
        print("="*60 + "\n")
        
        return False
    
    def _control_relay(self, activate):
        """Control relay (door lock)"""
        if self.mode == 'hardware':
            GPIO.output(self.RELAY_DOOR_LOCK, GPIO.HIGH if activate else GPIO.LOW)
        else:
            state = "ACTIVATED (Door Unlocked)" if activate else "DEACTIVATED (Door Locked)"
            print(f"   [RELAY] {state}")
    
    def _control_led(self, color, state):
        """Control LED indicators"""
        pin = self.LED_GREEN if color == 'green' else self.LED_RED
        
        if self.mode == 'hardware':
            GPIO.output(pin, GPIO.HIGH if state else GPIO.LOW)
        else:
            status = "ON" if state else "OFF"
            print(f"   [{color.upper()} LED] {status}")
    
    def _beep_pattern(self, pattern_type):
        """
        Control buzzer with different patterns
        
        Args:
            pattern_type: 'success' or 'denied'
        """
        if pattern_type == 'success':
            # 2 short beeps
            self._beep(0.2)
            time.sleep(0.1)
            self._beep(0.2)
        elif pattern_type == 'denied':
            # 3 longer beeps
            for _ in range(3):
                self._beep(0.5)
                time.sleep(0.1)
    
    def _beep(self, duration):
        """Single beep"""
        if self.mode == 'hardware':
            GPIO.output(self.BUZZER, GPIO.HIGH)
            time.sleep(duration)
            GPIO.output(self.BUZZER, GPIO.LOW)
        else:
            print(f"   [BUZZER] Beep ({duration}s)")
    
    def _log_access(self, name, status, student_id=None, reason=None):
        """Log access attempt"""
        log_entry = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'name': name,
            'status': status,
            'student_id': student_id,
            'reason': reason
        }
        self.access_log.append(log_entry)
        
        # Also log to console
        if status == 'GRANTED':
            print(f"📝 LOGGED: {name} accessed at {log_entry['timestamp']}")
        else:
            print(f"📝 LOGGED: Access denied - {reason} at {log_entry['timestamp']}")
    
    def _display_message(self, line1, line2):
        """Display message on LCD (if available)"""
        if self.mode == 'simulation':
            print(f"   [LCD Display]")
            print(f"   ┌────────────────┐")
            print(f"   │ {line1[:16]:16s} │")
            print(f"   │ {line2[:16]:16s} │")
            print(f"   └────────────────┘")
    
    def get_access_log(self):
        """Return access log"""
        return self.access_log
    
    def get_system_status(self):
        """Get current system status"""
        return {
            'door_locked': self.door_locked,
            'mode': self.mode,
            'total_access_attempts': len(self.access_log)
        }
    
    def cleanup(self):
        """Cleanup and shutdown"""
        print("\n🧹 Cleaning up embedded system...")
        
        if self.mode == 'hardware':
            # Turn off all outputs
            GPIO.output(self.RELAY_DOOR_LOCK, GPIO.LOW)
            GPIO.output(self.LED_GREEN, GPIO.LOW)
            GPIO.output(self.LED_RED, GPIO.LOW)
            GPIO.output(self.BUZZER, GPIO.LOW)
            GPIO.cleanup()
        
        print("✅ System shutdown complete\n")


# ============================================================================
# DEMO/TEST PROGRAM
# ============================================================================

def test_embedded_system():
    """Test the embedded door control system"""
    print("\n" + "="*60)
    print("🎓 EMBEDDED DOOR CONTROL SYSTEM - TEST MODE")
    print("="*60 + "\n")
    
    # Initialize system
    door_system = EmbeddedDoorSystem(mode='auto')
    
    print("\n📋 Running System Tests...\n")
    
    # Test 1: Known student access
    print("TEST 1: Known Student (Zahir)")
    door_system.grant_access("Zahir", student_id="CS-2021-001")
    
    time.sleep(1)
    
    # Test 2: Another known student
    print("TEST 2: Known Student (Mehran)")
    door_system.grant_access("Mehran", student_id="CS-2021-002")
    
    time.sleep(1)
    
    # Test 3: Unknown person
    print("TEST 3: Unknown Person Detected")
    door_system.deny_access("Face not in database")
    
    time.sleep(1)
    
    # Test 4: Another unknown attempt
    print("TEST 4: Unrecognized Face")
    door_system.deny_access("Low confidence match")
    
    # Show access log
    print("\n" + "="*60)
    print("📊 ACCESS LOG SUMMARY")
    print("="*60)
    log = door_system.get_access_log()
    for entry in log:
        print(f"{entry['timestamp']} | {entry['name']:15s} | {entry['status']}")
    
    # Show system status
    print("\n" + "="*60)
    print("🔧 SYSTEM STATUS")
    print("="*60)
    status = door_system.get_system_status()
    print(f"Mode: {status['mode']}")
    print(f"Door Locked: {status['door_locked']}")
    print(f"Total Access Attempts: {status['total_access_attempts']}")
    print("="*60 + "\n")
    
    # Cleanup
    door_system.cleanup()


if __name__ == "__main__":
    test_embedded_system()
