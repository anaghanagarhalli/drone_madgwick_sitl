# drone_madgwick_sitl

# 1. WSL Ubuntu setup (one-time)
sudo apt update && sudo apt install git python3-pip python3-venv build-essential -y

# 2. ArduPilot clone
cd ~
git clone https://github.com/ArduPilot/ardupilot.git --recursive
cd ardupilot

# 3. Virtual environment
python3 -m venv ardupilot_venv
source ardupilot_venv/bin/activate
pip install empy numpy future lxml pymavlink MAVProxy dronekit pandas matplotlib pyserial pexpect wxPython pyglet

# 4. Build & launch SITL
cd ArduCopter
../Tools/autotest/sim_vehicle.py -v ArduCopter --console --map --speedup=10

# 5. SITL flight (at MAV> prompt)
mode GUIDED
rc 1 1500; rc 2 1500; rc 3 1000; rc 4 1500
arm throttle
rc 3 1500; rc 1 1600; rc 4 1650; rc 3 1000
disarm
# Ctrl+C to stop

# 6. Extract IMU data
cd ..
python3 extract_imu.py

# 7. Run Madgwick analysis
python3 madgwick_analysis.py
