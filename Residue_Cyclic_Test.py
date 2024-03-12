import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from Residue_Network import NN
from Residue_Network import getData

if __name__ == "__main__":

    Cyc_model = NN(2, 1)
    Cyc_model.load_weights('./policies/' + 'CYC_Mar-11-2024_1842')
    weights = Cyc_model.get_weights()
    print("Model Wights: ", weights)

    # Get the data
    INPUT_ARRAY_TESTING, COLL_TESTING, CYC_TESTING = getData("Test")
    INPUT_ARRAY, COLL, CYC, DeltaNCs, thetas = getData("All")

    # Predict the output
    Cyc_PRED = Cyc_model(INPUT_ARRAY_TESTING)
    error = tf.keras.losses.MSE(CYC_TESTING, Cyc_PRED)
    print("Root Mean Sq. Error: ", np.sqrt(np.rad2deg(error)))
    
    count = 0
    for i in range(len(thetas)):
        if i%(60//6) == 0 or i == 59:

            mask = (INPUT_ARRAY[:, 1] == thetas[i])
            inputs = INPUT_ARRAY[mask]
            outputs = CYC[mask]
            predictions = Cyc_model(inputs)

            # Calcualte RMSE
            error = tf.keras.losses.MSE(outputs, predictions)
            print(f"Root Mean Sq. Error for Theta = {np.rad2deg(thetas[i]):.2f}: ", np.sqrt(error))
            color = 'C'+ str(count)

            plt.plot((inputs[:, 0]), np.rad2deg(outputs), color+'.', alpha=0.2)
            plt.plot((inputs[:, 0]), np.rad2deg(predictions), color, label=f'{np.rad2deg(thetas[i]):.2f} deg')
            count += 1

    plt.xlabel('Swashplate Offsets (mm)')
    plt.ylabel('Cyclic (deg)')
    plt.title('Cyclic Angle vs Swashplate Offsets for Different Thetas')
    plt.legend()
    plt.show()



    
    