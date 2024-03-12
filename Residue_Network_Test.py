import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from Residue_Network import NN
from Residue_Network import getData

if __name__ == "__main__":

    Col_model = NN(2, 1)
    Col_model.load_weights('./policies/' + 'COLL_Mar-11-2024_1826')
    weights = Col_model.get_weights()
    print("Model Wights: ", weights)

    # Get the data
    INPUT_ARRAY_TESTING, COLL_TESTING, CYC_TESTING = getData("Test")
    INPUT_ARRAY, COLL, CYC, DeltaNCs, thetas = getData("All")

    # Predict the output
    COLL_PRED = Col_model(INPUT_ARRAY_TESTING)
    error = tf.keras.losses.MSE(COLL_TESTING, COLL_PRED)
    print("Root Mean Sq. Error: ", np.sqrt(np.rad2deg(error)))
    
    count = 0
    for i in range(len(DeltaNCs)):
        if i%(159//6) == 0:

            mask = (INPUT_ARRAY[:, 0] == DeltaNCs[i])
            inputs = INPUT_ARRAY[mask]
            outputs = COLL[mask]
            predictions = Col_model(inputs)

            # Calcualte RMSE
            error = tf.keras.losses.MSE(outputs, predictions)
            print(f"Root Mean Sq. Error for DeltaNC = {DeltaNCs[i]:.2f}: ", np.sqrt(error))
            color = 'C'+ str(count)

            plt.plot(np.rad2deg(inputs[:, 1]), np.rad2deg(outputs), color+'.', alpha=0.2)
            plt.plot(np.rad2deg(inputs[:, 1]), np.rad2deg(predictions), color, label=f'{DeltaNCs[i]:.2f}mm')
            count += 1

    plt.xlabel('Theta')
    plt.ylabel('Collective')
    plt.legend()
    plt.show()



    
    