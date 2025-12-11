"""
Quick test script to verify serial communication with Proteus
This sends test characters to the virtual COM port to verify setup
"""
import serial
import serial.tools.list_ports
import time
import sys

def list_available_ports():
    """List all available COM ports"""
    ports = serial.tools.list_ports.comports()
    print("\n" + "="*60)
    print("AVAILABLE COM PORTS:")
    print("="*60)
    if ports:
        for i, port in enumerate(ports, 1):
            print(f"{i}. {port.device} - {port.description}")
    else:
        print("No COM ports found!")
    print("="*60 + "\n")
    return [port.device for port in ports]

def test_serial_connection(com_port, baud_rate=9600):
    """Test serial connection by sending test characters"""
    print(f"\n[TEST] Attempting to connect to {com_port}...")
    
    try:
        ser = serial.Serial(
            port=com_port,
            baudrate=baud_rate,
            timeout=1,
            write_timeout=1
        )
        time.sleep(2)  # Wait for connection to stabilize
        print(f"[OK] Connected to {com_port} @ {baud_rate} baud")
        
        # Test sequence
        test_sequence = [
            ('A', 'Student A (Zahir)'),
            ('B', 'Student B (Mehran)'),
            ('C', 'Student C (Yousaf)'),
            ('U', 'Unknown Person'),
            ('X', 'Adjust Face')
        ]
        
        print("\n" + "="*60)
        print("TESTING SERIAL COMMUNICATION")
        print("="*60)
        print("Watch Proteus LCD display for responses...")
        print("-"*60)
        
        for char, description in test_sequence:
            print(f"\nSending '{char}' - {description}")
            ser.write(char.encode())
            print(f"  ✓ Sent successfully")
            time.sleep(3)  # Wait to observe Arduino response
        
        print("\n" + "="*60)
        print("TEST COMPLETE!")
        print("="*60)
        print("\nDid you see the responses on Proteus LCD?")
        print("If yes: ✓ Serial connection is working!")
        print("If no: Check Proteus COM port configuration")
        print("="*60 + "\n")
        
        ser.close()
        return True
        
    except serial.SerialException as e:
        print(f"[ERROR] Failed to connect: {e}")
        print("\nPossible issues:")
        print("  1. Virtual COM port not created")
        print("  2. Port already in use by another program")
        print("  3. Incorrect port number")
        print("  4. Need administrator privileges")
        return False
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        return False

def main():
    print("\n" + "="*60)
    print("PROTEUS SERIAL CONNECTION TEST")
    print("="*60)
    print("This script tests serial communication with Proteus/Arduino")
    print("="*60)
    
    # List available ports
    available_ports = list_available_ports()
    
    if not available_ports:
        print("[ERROR] No COM ports available!")
        print("\nPlease set up virtual COM ports first:")
        print("  1. Install VSPE or com0com")
        print("  2. Create virtual port pair (e.g., COM3 ↔ COM4)")
        print("  3. Run this script again")
        return
    
    # Get COM port from user or use default
    if len(sys.argv) > 1:
        com_port = sys.argv[1]
    else:
        print("Usage: python test_proteus_serial.py [COM_PORT]")
        print(f"Example: python test_proteus_serial.py COM3")
        print(f"\nUsing default: COM3")
        com_port = "COM3"
    
    # Verify port exists
    if com_port not in available_ports:
        print(f"\n[WARNING] {com_port} not found in available ports!")
        response = input(f"Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return
    
    # Get baud rate
    baud_rate = 9600
    if len(sys.argv) > 2:
        baud_rate = int(sys.argv[2])
    
    print(f"\nConfiguration:")
    print(f"  COM Port: {com_port}")
    print(f"  Baud Rate: {baud_rate}")
    
    # Make sure Proteus is ready
    print("\n" + "="*60)
    print("BEFORE CONTINUING:")
    print("="*60)
    print("1. Open Proteus project")
    print("2. Configure Arduino to use the PAIRED COM port")
    print(f"   (If Python uses {com_port}, Proteus should use COM4)")
    print("3. Start Proteus simulation (click Play)")
    print("4. Verify LCD shows 'Ready!'")
    print("="*60)
    input("\nPress ENTER when ready to start test...")
    
    # Run test
    success = test_serial_connection(com_port, baud_rate)
    
    if success:
        print("\n✓ Test completed successfully!")
        print("\nYou can now run the full face recognition system:")
        print("  cd d:\\face_det\\src")
        print(f"  python proteus_integration.py --com-port {com_port}")
    else:
        print("\n✗ Test failed!")
        print("\nTroubleshooting steps:")
        print("  1. Verify virtual COM ports are created")
        print("  2. Check port numbers match in VSPE/com0com")
        print("  3. Close other programs using the COM port")
        print("  4. Restart Proteus simulation")
        print("  5. Try different COM ports")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[EXIT] Test cancelled by user")
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
