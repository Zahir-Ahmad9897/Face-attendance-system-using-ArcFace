# VSPE Virtual COM Port Setup for Face Recognition + Proteus

## Configuration

**Virtual Port Pair:** COM2 ↔ COM4

- **COM2** - Used by Python face recognition script (`webcam_recognition.py`)
- **COM4** - Used by Proteus Arduino simulation

## How to Setup VSPE

1. **Download VSPE**
   - URL: https://www.eterlogic.com/Products.VSPE.html
   - Install and run VSPE

2. **Create Port Pair**
   - Click "Device" → "Create"
   - Select "Pair"
   - Configure:
     - Port 1: COM2
     - Port 2: COM4
   - Click "Create"

3. **Keep VSPE Running**
   - VSPE must stay open while using the system
   - Minimize to system tray if desired

## Data Transmission

- **Frequency**: Every 3 seconds
- **Format**: Lowercase name + newline (`\n`)
- **Examples**:
  - `zahir\n`
  - `mehran\n`
  - `yousaf\n`
  - `unknown\n`

## Troubleshooting

### Port Already in Use
- Close any programs using COM2 or COM4
- Restart VSPE
- Recreate the port pair

### No Data Received in Proteus
1. Check VSPE is running
2. Verify Proteus Arduino is set to COM4
3. Check webcam_recognition.py shows `[SERIAL] Connected to COM2`
4. Look for `[PROTEUS] Sent: <name>` messages in console

### Permission Denied
- Run VSPE as Administrator
- Ensure no other virtual COM port software is running

## Configuration File

The VSPE configuration is saved in: `v ports.vspe`

You can load this configuration in VSPE:
- File → Load → Select `v ports.vspe`

## Notes

- Baud rate: 9600
- Data bits: 8
- Stop bits: 1
- Parity: None
- Flow control: None
