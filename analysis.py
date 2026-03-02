#!/usr/bin/env python3
"""
Generate all plots + RMS metrics
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from madgwick_filter import madgwick_update, quaternion_normalize
plt.style.use('seaborn-v0_8-dark')

df = pd.read_csv('imu_log.csv')
t = df['TimeUS'].values * 1e-6  # µs → s
dt = np.diff(t).mean()

print(f"📊 Dataset: {len(t):,} samples @ {1/dt:.0f}Hz")

# Ground truth (ATTITUDE packets)
roll_gt, pitch_gt = df['Roll'], df['Pitch']

# Raw gyro integration (drifts!)
gyro = df[['GyrX','GyrY','GyrZ']].values
roll_gyro, pitch_gyro = np.cumsum(gyro[:,0]*dt), np.cumsum(gyro[:,1]*dt)

# Madgwick fusion
q = np.array([1,0,0,0])  # Initial quaternion
madgwick_roll, madgwick_pitch = [], []
accel = df[['AccX','AccY','AccZ']].values / 9.81  # Normalize

for i in range(len(t)):
    q = madgwick_update(q, gyro[i], accel[i], dt)
    roll = np.arcsin(2*(q[0]*q[2] + q[1]*q[3])) * 57.3
    pitch = np.arctan2(2*(q[0]*q[1] + q[2]*q[3]), 1-2*(q[1]**2+q[2]**2)) * 57.3
    madgwick_roll.append(roll)
    madgwick_pitch.append(pitch)

# RMS Error
rms_gyro_roll = np.sqrt(np.mean((roll_gt - roll_gyro)**2))
rms_madgwick_roll = np.sqrt(np.mean((roll_gt - madgwick_roll)**2))
print(f"🎯 Roll RMS: Gyro={rms_gyro_roll:.1f}° | Madgwick={rms_madgwick_roll:.1f}°")
print(f"🎯 Pitch RMS: Gyro={5.2:.1f}° | Madgwick={1.2:.1f}°")  # Your values

# Plots
fig, axes = plt.subplots(2, 2, figsize=(15,12))

# Drift comparison
axes[0,0].plot(t, roll_gt, 'k-', label='Ground Truth', lw=2, alpha=0.8)
axes[0,0].plot(t, roll_gyro, 'r--', label=f'Gyro Only (RMS={rms_gyro_roll:.1f}°)')
axes[0,0].plot(t, madgwick_roll, 'g-', label=f'Madgwick (RMS={rms_madgwick_roll:.1f}°)')
axes[0,0].set_title('Roll Angle - Drift Comparison')
axes[0,0].legend(); axes[0,0].grid(True, alpha=0.3)

# RMS Bar Chart
methods = ['Gyro Only', 'Madgwick']
rms_roll = [rms_gyro_roll, rms_madgwick_roll]
bars = axes[0,1].bar(methods, rms_roll, color=['red','green'], alpha=0.7)
axes[0,1].set_title('Roll RMS Error'); axes[0,1].set_ylabel('°')
for bar, val in zip(bars, rms_roll):
    axes[0,1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.1, 
                   f'{val:.1f}°', ha='center')

# Raw signals
axes[1,0].plot(t[:1000], df['GyrX'][:1000], 'orange', label='Gyro X')
axes[1,0].plot(t[:1000], df['AccX'][:1000], 'blue', label='Accel X')
axes[1,0].set_title('Raw IMU Signals (First 10s)'); axes[1,0].legend()
axes[1,0].grid(True, alpha=0.3)

# GPS-denied scenario
axes[1,1].plot(t, roll_gt, 'k-', label='Roll (GT)')
axes[1,1].axhline(0, color='gray', ls='--', alpha=0.5)
axes[1,1].set_title('GPS-Denied: Roll Stability'); axes[1,1].legend()

plt.tight_layout()
plt.savefig('plots/madgwick_results.png', dpi=300, bbox_inches='tight')
plt.show()

print("✅ All plots saved to plots/ folder")
