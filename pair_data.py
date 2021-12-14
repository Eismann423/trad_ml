import numpy as np
import pandas as pd
import os

file_number = 5
aug_type = "timewarp"
aug_file_name = aug_type + str(file_number)


# file paths for augmented data
path_forces = r"C:\Users\Eisma\thesis\nationals_data\calcs\qual\processing\white_proc\impulse_by_mass_aug\csv\multiaug5_cnn\csv"
path_dest = r"C:\Users\Eisma\thesis\nationals_data\calcs\qual\processing\white_proc\impulse_by_mass_paired_aug\multiaug5_cnn\csv"


def combine_two_csvs(first, second, subdir, path_dest):
    file_one = pd.read_csv(os.path.join(subdir, first), header=None, usecols=(0, 1, 2), names=['x', 'y', 'z'])
    file_two = pd.read_csv(os.path.join(subdir, second), header=None, usecols=(0, 1, 2), names=['x', 'y', 'z'])

    x_one = np.array(file_one['x'])
    y_one = np.array(file_one['y'])
    z_one = np.array(file_one['z'])

    x_two = np.array(file_two['x'])
    y_two = np.array(file_two['y'])
    z_two = np.array(file_two['z'])

    # for unaugmented files
    # file_one = pd.read_csv(os.path.join(subdir, first))
    # file_two = pd.read_csv(os.path.join(subdir, second))
    #
    # x_one = np.array(file_one["iX"])
    # y_one = np.array(file_one["iY"])
    # z_one = np.array(file_one["iZ"])
    #
    # x_two = np.array(file_two["iX"])
    # y_two = np.array(file_two["iY"])
    # z_two = np.array(file_two["iZ"])

    one_len = len(x_one)
    two_len = len(x_two)

    if (one_len > two_len):
        n = one_len - two_len
        x_two = np.pad(x_two, (0, n), 'constant')
        y_two = np.pad(y_two, (0, n), 'constant')
        z_two = np.pad(z_two, (0, n), 'constant')

    else:
        n = two_len - one_len
        x_one = np.pad(x_one, (0, n), 'constant')
        y_one = np.pad(y_one, (0, n), 'constant')
        z_one = np.pad(z_one, (0, n), 'constant')

    forces = np.vstack((x_one, y_one, z_one, x_two, y_two, z_two)).T

    filename = os.path.join(path_dest, os.path.split(subdir)[1], first)
    np.savetxt(filename, forces, delimiter=",")
    return


for subdir, dirs, files in os.walk(path_forces):
    i = 0
    if os.path.split(subdir)[1] == "V14-V15":
        first48 = None
        second48 = None
        flipper = 1
        while i < len(files):
            if " 48 " in files[i]:
                if flipper > 0:
                    first48 = files[i]
                    flipper *= -1
                else:
                    second48 = files[i]
                    flipper *= -1
                    combine_two_csvs(first48, second48, subdir, path_dest)
                i += 1

            else:
                combine_two_csvs(files[i], files[i+1], subdir, path_dest)
                i += 2

        print("V14-V15 finished")
        print("-------------")
    while i < len(files):
        combine_two_csvs(files[i], files[i + 1], subdir, path_dest)
        i = i + 2

    print(os.path.split(subdir)[1] + " finished")
    print("--------------")



