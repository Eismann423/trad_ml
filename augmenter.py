import numpy as np
import tsaug as ts
import os
import pandas as pd

path_v1 = r"C:\Users\Eisma\thesis\nationals_data\calcs\qual\processing\white_proc\impulse_by_mass\csv\V8-V9"
path_v2 = r"C:\Users\Eisma\thesis\nationals_data\calcs\qual\processing\white_proc\impulse_by_mass\csv\V10-V11"
path_v3 = r"C:\Users\Eisma\thesis\nationals_data\calcs\qual\processing\white_proc\impulse_by_mass\csv\V12-V13"
path_v4 = r"C:\Users\Eisma\thesis\nationals_data\calcs\qual\processing\white_proc\impulse_by_mass\csv\V14-V15"
path_dest = r"C:\Users\Eisma\thesis\nationals_data\calcs\qual\processing\white_proc\impulse_by_mass_aug\csv\multiaug5"
path_dest_cnn = r"C:\Users\Eisma\thesis\nationals_data\calcs\qual\processing\white_proc\impulse_by_mass_aug\csv\multiaug5_cnn/csv"


# Augmenter used to create augmented data files
# augmenter = ts.AddNoise(scale=(0.01, 0.05)) @ 0.75
# augmenter = ts.Drift(max_drift=(0.5, 0.99)) @ 0.75
# augmenter = ts.TimeWarp(n_speed_change=3, max_speed_ratio=8, seed=42)
augmenter = (
    ts.TimeWarp(n_speed_change=3, max_speed_ratio=3, seed=42)
    + ts.AddNoise(scale=(0.01, 0.05)) @ 0.75
    + ts.Drift(max_drift=(0.5, 0.99)) @ 0.25
)

total_files = 1000
num_classes = 4
files_per_class = total_files / num_classes
c1_original_size = 10
c2_original_size = 24
c3_original_size = 34
c4_original_size = 5

c1_new_size = files_per_class - c1_original_size
c2_new_size = files_per_class - c2_original_size
c3_new_size = files_per_class - c3_original_size
c4_new_size = files_per_class - c4_original_size

# Create enough augmented files so that each class has equal number of samples
count = 201
inc = 0
break_out_flag = False
# Handles V9-V9
for x in range(24):
    for subdir, dirs, files in os.walk(path_v1):
        for filename in files:
            if count == 201 + (c1_new_size / 2):
                break_out_flag = True
                break
            xls = pd.read_csv(os.path.join(subdir, filename))
            x = np.array(xls["iX"])
            y = np.array(xls["iY"])
            z = np.array(xls["iZ"])
            x_aug = augmenter.augment(x)
            y_aug = augmenter.augment(y)
            z_aug = augmenter.augment(z)

            forces = np.vstack((x_aug, y_aug, z_aug)).T
            xname = os.path.join(path_dest, str(count) + " " + filename)
            xname_cnn = os.path.join(path_dest_cnn, 'V8-V9' + '\\' + str(count) + " " + filename)
            np.savetxt(xname, forces, delimiter=",")
            np.savetxt(xname_cnn, forces, delimiter=",")
            inc = inc + 1
            if inc % 2 == 0:
                count = count + 1
            print(filename + " -- Done")
        if break_out_flag:
            break
    if break_out_flag:
        break

break_out_flag = False
inc = 0
# Handles V10-V11
for x in range(24):
    for subdir, dirs, files in os.walk(path_v2):
        for filename in files:
            if count == 201 + (c1_new_size / 2) + (c2_new_size / 2):
                break_out_flag = True
                break
            xls = pd.read_csv(os.path.join(subdir, filename))
            x = np.array(xls["iX"])
            y = np.array(xls["iY"])
            z = np.array(xls["iZ"])
            x_aug = augmenter.augment(x)
            y_aug = augmenter.augment(y)
            z_aug = augmenter.augment(z)

            forces = np.vstack((x_aug, y_aug, z_aug)).T
            xname = os.path.join(path_dest, str(count) + " " + filename)
            xname_cnn = os.path.join(path_dest_cnn, 'V10-V11' + '\\' + str(count) + " " + filename)
            np.savetxt(xname, forces, delimiter=",")
            np.savetxt(xname_cnn, forces, delimiter=",")
            inc = inc + 1
            if inc % 2 == 0:
                count = count + 1
            print(filename + " -- Done")
        if break_out_flag:
            break
    if break_out_flag:
        break

break_out_flag = False
inc = 0
# Handles V12-V13
for x in range(24):
    for subdir, dirs, files in os.walk(path_v3):
        for filename in files:
            if count == 201 + (c1_new_size / 2) + (c2_new_size / 2) + (c3_new_size / 2):
                break_out_flag = True
                break
            xls = pd.read_csv(os.path.join(subdir, filename))
            x = np.array(xls["iX"])
            y = np.array(xls["iY"])
            z = np.array(xls["iZ"])
            x_aug = augmenter.augment(x)
            y_aug = augmenter.augment(y)
            z_aug = augmenter.augment(z)

            forces = np.vstack((x_aug, y_aug, z_aug)).T
            xname = os.path.join(path_dest, str(count) + " " + filename)
            xname_cnn = os.path.join(path_dest_cnn, 'V12-V13' + '\\' + str(count) + " " + filename)
            np.savetxt(xname, forces, delimiter=",")
            np.savetxt(xname_cnn, forces, delimiter=",")
            inc = inc + 1
            if inc % 2 == 0:
                count = count + 1
            print(filename + " -- Done")
        if break_out_flag:
            break
    if break_out_flag:
        break

break_out_flag = False
inc = 0
count = 542
# Handles V14-V15
for x in range(100):
    for subdir, dirs, files in os.walk(path_v4):
        for filename in files:
            if count == 689:
                break_out_flag = True
                break
            xls = pd.read_csv(os.path.join(subdir, filename))
            x = np.array(xls["iX"])
            y = np.array(xls["iY"])
            z = np.array(xls["iZ"])
            x_aug = augmenter.augment(x)
            y_aug = augmenter.augment(y)
            z_aug = augmenter.augment(z)

            forces = np.vstack((x_aug, y_aug, z_aug)).T
            xname = os.path.join(path_dest, str(count) + " " + filename)
            xname_cnn = os.path.join(path_dest_cnn, 'V14-V15' + '\\' + str(count) + " " + filename)
            np.savetxt(xname, forces, delimiter=",")
            np.savetxt(xname_cnn, forces, delimiter=",")
            inc = inc + 1
            if "48" in filename:
                inc = inc + 1
                count = count + 1
            elif inc % 2 == 0:
                count = count + 1
            print(filename + " -- Done")
        if break_out_flag:
            break
    if break_out_flag:
        break
