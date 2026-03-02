#!/usr/bin/env python3
"""
Extract IMU data from ArduPilot SITL .bin logs
"""
from pymavlink import mavutil
import numpy as np
import glob, sys

logs = glob.glob('ArduCopter/logs/*.bin')
if not logs:
    print("❌ No SITL logs found. Run sim_vehicle.py first!")
    sys.exit(1)

print(f"✅ Processing {logs[0]}")
m = mavutil.mavlink_connection(logs[0])

data = {'TimeUS':[], 'GyrX':[], 'GyrY':[], 'GyrZ':[], 
        'AccX':[], 'AccY':[], 'AccZ':[], 'Roll':[], 'Pitch':[]}
count = 0

while True:
    msg = m.recv_match(type=['IMU','ATT'])
    if not msg: break
    
    if msg.get_type() == 'IMU' and count < 10000:
        data['TimeUS'].append(msg.TimeUS)
        data['GyrX'].append(msg.GyrX); data['GyrY'].append(msg.GyrY); data['GyrZ'].append(msg.GyrZ)
        data['AccX'].append(msg.AccX); data['AccY'].append(msg.AccY); data['AccZ'].append(msg.AccZ)
        count += 1
    elif msg.get_type() == 'ATT':
        data['Roll'].append(msg.Roll*57.3)  # deg
        data['Pitch'].append(msg.Pitch*57.3)

np.savetxt('imu_log.csv', np.column_stack([data[k] for k in data]), 
           delimiter=',', header=','.join(data.keys()), comments='')
print(f"✅ {len(data['TimeUS'])} IMU samples saved!")
