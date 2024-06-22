from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dense

# Define a function to build a Bidirectional LSTM model
def bilstm():
    # Initialize a sequential model
    model = Sequential()

    # Add a bidirectional LSTM layer with 40 units
    # Activation function is ReLU and kernel initializer is He uniform
    # Input shape is (1, 30), indicating 1 timestep with 30 features
    model.add(Bidirectional(LSTM(40, activation='relu', kernel_initializer='he_uniform'), input_shape=(1, 30)))

    # Add a dense layer with 10 units
    # Activation function is ReLU and kernel initializer is He uniform
    model.add(Dense(10, activation='relu', kernel_initializer='he_uniform'))

    # Add the output dense layer with 1 unit
    # Kernel initializer is He uniform
    model.add(Dense(1, kernel_initializer='he_uniform'))

    # Print the model summary to show the structure
    #model.summary()

    # Compile the model
    # Loss function is mean squared error
    # Optimizer is Adam and metric is mean squared error
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])

    return model
