import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import pre_processing  # Custom module for preprocessing
import models  # Custom module for model definition
from sklearn.metrics import mean_absolute_error

# Load the datasets from the 'Dataset' directory
B0005 = loadmat('Dataset/B0005.mat')['B0005']
B0006 = loadmat('Dataset/B0006.mat')['B0006']
B0007 = loadmat('Dataset/B0007.mat')['B0007']
B0018 = loadmat('Dataset/B0018.mat')['B0018']

# Visualize charging profiles for Battery B0005
vCharge, iCharge, tCharge, timeCharge = pre_processing.visualise_charging_profiles(B0005)

# Extract and flatten charging profiles for the first cycle (Fresh Cell) and 168th cycle (Aged Cell)
FTime = np.array(timeCharge[0][:]).flatten()
FreshCell_V = np.array(vCharge[0][:]).flatten()
FreshCell_I = np.array(iCharge[0][:]).flatten()
FreshCell_T = np.array(tCharge[0][:]).flatten()

ATime = np.array(timeCharge[54][:]).flatten()
AgedCell_V = np.array(vCharge[54][:]).flatten()
AgedCell_I = np.array(iCharge[54][:]).flatten()
AgedCell_T = np.array(tCharge[54][:]).flatten()

# Plot charging profiles for Fresh and Aged Cells
fig, a = plt.subplots(3, 1)
a[0].plot(FTime, FreshCell_V, label='Fresh Cell (1st Cycle)')
a[0].plot(ATime, AgedCell_V, label='Aged Cell (168th Cycle)')
a[0].set_ylabel('Voltage (V)')
a[0].legend()

a[1].plot(FTime, FreshCell_I, label='Fresh Cell (1st Cycle)')
a[1].plot(ATime, AgedCell_I, label='Aged Cell (168th Cycle)')
a[1].set_ylabel('Current (A)')

a[2].plot(FTime, FreshCell_T, label='Fresh Cell (1st Cycle)')
a[2].plot(ATime, AgedCell_T, label='Aged Cell (168th Cycle)')
a[2].set_ylabel('Temperature (°C)')
a[2].set_xlabel('Time (Minutes)')

# Extract discharge capacities for all batteries
cap5 = pre_processing.extract_discharge(B0005)
cap6 = pre_processing.extract_discharge(B0006)
cap7 = pre_processing.extract_discharge(B0007)
cap18 = pre_processing.extract_discharge(B0018)

# Plot discharge capacities for all batteries and failure threshold
plt.figure(2)
plt.plot(cap5, label="Battery #5")
plt.plot(cap6, label="Battery #6")
plt.plot(cap7, label="Battery #7")
plt.plot(cap18, label="Battery #18")
plt.plot(1.4 * np.ones(len(cap7)), label="Failure Threshold")
plt.legend()

# Preprocess charging data for all batteries
charInput5 = pre_processing.extract_charge_preprocessing(B0005)
charInput6 = pre_processing.extract_charge_preprocessing(B0006)
charInput7 = pre_processing.extract_charge_preprocessing(B0007)
charInput18 = pre_processing.extract_charge_preprocessing(B0018)

# Initial capacities for normalization
InitC5 = 1.86
InitC6 = 2.04
InitC7 = 1.89
InitC18 = 1.86

# Normalize and prepare training and testing datasets
xB5, yB5, ym5, yr5 = pre_processing.minmax_norm(charInput5, InitC5, cap5)
xB6, yB6, ym6, yr6 = pre_processing.minmax_norm(charInput6, InitC6, cap6)
xB7, yB7, ym7, yr7 = pre_processing.minmax_norm(charInput7, InitC7, cap7)
xB18, yB18, ym18, yr18 = pre_processing.minmax_norm(charInput18, InitC18, cap18)

# Concatenate training data
Train_Input = np.concatenate((xB5, xB6, xB7, xB18))
Train_Output = np.concatenate((yB5, yB6, yB7, yB18))


# Print shapes of training data for verification
print(Train_Input.shape)
print(Train_Output.shape)


# Initialize and train the BiLSTM model
model_bilstm = models.bilstm()
model_bilstm.fit(Train_Input.reshape(Train_Input.shape[0], 1, Train_Input.shape[1]), 
                 Train_Output, epochs=500, batch_size=1)

