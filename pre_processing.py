import numpy as np

# Visualize charging profiles for a given battery dataset
def visualise_charging_profiles(B):
    vCharge, iCharge, tCharge, timeCharge = list(), list(), list(), list()
    cycle = B['cycle'][0][0]
    for i in range(cycle.shape[0]):
        for j in range(cycle.shape[1]):
            if cycle[i,j]['type'] == ['charge']:
                # Extract voltage, current, temperature, and time data for charging cycles
                vTemp = cycle[i,j]['data'][0][0]['Voltage_measured']
                vCharge.append(vTemp)

                iTemp = cycle[i,j]['data'][0][0]['Current_measured']
                iCharge.append(iTemp)

                tTemp = cycle[i,j]['data'][0][0]['Temperature_measured']
                tCharge.append(tTemp)
                
                time = cycle[i,j]['data'][0][0]['Time']
                timeCharge.append(time/60)  # Convert time to minutes
            
    return vCharge, iCharge, tCharge, timeCharge

# Extract discharge capacities for a given battery dataset
def extract_discharge(B):
    cycle = B['cycle'][0][0]
    cap = list()
    for i in range(cycle.shape[0]):
        for j in range(cycle.shape[1]):
            if cycle[i,j]['type'] == ['discharge']:
                # Extract capacity data for discharge cycles
                c = cycle[i,j]['data'][0][0]['Capacity']
                c = c.flatten()
                cap.append(c)

    return cap

# Remove NaN values from a charging profile
def remove_nan_values(charging_profile):
    for i in range(charging_profile.shape[0]):
        for j in range(charging_profile.shape[1]):
            if np.isnan(charging_profile[i][j]):
                charging_profile[i][j] = (charging_profile[i][j-1] + charging_profile[i][j+1])/2        

    return charging_profile

# Preprocess and extract charge data from a given battery dataset
def extract_charge_preprocessing(B):
    cycle = B['cycle'][0][0]
    charInput = list()
    for i in range(cycle.shape[0]):
        for j in range(cycle.shape[1]-1):
            if cycle[i,j]['type'] == ['charge']:
                # Extract and preprocess voltage, current, and temperature data for charging cycles
                vTemp = cycle[i,j]['data'][0][0]['Voltage_measured']
                vTemp = remove_nan_values(vTemp)
                le = np.mod(vTemp.shape[1], 10)
                vTemp = vTemp[:, 0:vTemp.shape[1]-le]
                vTemp = np.reshape(vTemp, (int(vTemp.shape[1]/10), 10))
                vTemp = np.mean(vTemp, axis=0)
                
                iTemp = cycle[i,j]['data'][0][0]['Current_measured']
                iTemp = remove_nan_values(iTemp)
                iTemp = iTemp[:, 0:iTemp.shape[1]-le]
                iTemp = np.reshape(iTemp, (int(iTemp.shape[1]/10), 10))
                iTemp = np.mean(iTemp, axis=0)
                
                tTemp = cycle[i,j]['data'][0][0]['Temperature_measured']
                tTemp = remove_nan_values(tTemp)
                tTemp = tTemp[:, 0:tTemp.shape[1]-le]
                tTemp = np.reshape(tTemp, (int(tTemp.shape[1]/10), 10))
                tTemp = np.mean(tTemp, axis=0)
                
                charInput.append(np.concatenate([vTemp, iTemp, tTemp]))
                
    print(np.array(charInput).shape)          
    return charInput

# Normalize charge data and corresponding capacities using min-max normalization
def minmax_norm(charInput, InitC, cap):
    # Normalize input features
    r = np.amax(charInput, axis=0) - np.amin(charInput, axis=0)
    xData = (charInput - np.amin(charInput, axis=0))
    xData = np.true_divide(xData, r)
    
    # Normalize output capacities
    comp = len(charInput) - len(cap)
    yData = np.concatenate((InitC * np.ones((comp, 1)), np.array(cap)))
    ym = np.min(yData)
    yr = np.max(yData) - np.min(yData)
    yData = (yData - ym) / yr
    
    return xData, yData, ym, yr
