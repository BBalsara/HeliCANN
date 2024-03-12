import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from Dom_Func_Network import NN as NN_Dom
from Dom_Func_Network import getData_Dom
from Residue_Network import NN as NN_Res
from Residue_Network import getData

if __name__ == "__main__":

    # Load the models
    Col_DOM_Model = NN_Dom(1, 1)
    Col_DOM_Model.load_weights('./policies/' + 'COLL_DOM_Mar-11-2024_1900')
    weights_d = Col_DOM_Model.get_weights()
    print("Model Wights: ", weights_d)


    Col_RES_Model = NN_Res(2, 1)
    Col_RES_Model.load_weights('./policies/' + 'COLL_Mar-11-2024_1826')
    weights_r = Col_RES_Model.get_weights()
    print("Model Wights: ", weights_r)

    Cyc_RES_Model = NN_Res(2, 1)
    Cyc_RES_Model.load_weights('./policies/' + 'CYC_Mar-11-2024_1842')
    weights_r = Cyc_RES_Model.get_weights()
    print("Model Wights: ", weights_r)



    # Get the data
    INPUT_ARRAY, AVG_COLL_RAW, AVG_CYC_RAW, DeltaNCs, thetas= getData("FULL")

    # Predict the output
    COLL_DOM_PRED = Col_DOM_Model(INPUT_ARRAY[:, 0])
    COLL_RES_PRED = Col_RES_Model(INPUT_ARRAY)
    CYC_DOM_PRED = INPUT_ARRAY[:, 1]
    CYC_RES_PRED = Cyc_RES_Model(INPUT_ARRAY)

    COLL_PRED = COLL_DOM_PRED + COLL_RES_PRED
    CYC_PRED = CYC_DOM_PRED + CYC_RES_PRED

    error = tf.keras.losses.MSE(AVG_COLL_RAW, COLL_PRED)
    print("Collective Root Mean Sq. Error: ", np.rad2deg(np.sqrt(error)))

    error = tf.keras.losses.MSE(AVG_CYC_RAW, CYC_PRED)
    print("Cyclic Root Mean Sq. Error: ", np.rad2deg(np.sqrt(error)))
    
    count = 0
    for i in range(len(DeltaNCs)):
        if i == 10: # Choose which one to plot

            mask = (INPUT_ARRAY[:, 0] == DeltaNCs[i])
            inputs = INPUT_ARRAY[mask]
            outputs = AVG_COLL_RAW[mask]
            predictions = COLL_PRED[mask]

            # Calcualte RMSE
            error = tf.keras.losses.MSE(outputs, predictions)
            print(f"Root Mean Sq. Error for DeltaNC = {DeltaNCs[i]:.2f}: ", np.rad2deg(np.sqrt(error)))
            color = 'C'+ str(count)

            plt.plot(np.rad2deg(inputs[:, 1]), np.rad2deg(outputs), color+'.', alpha=0.2)
            plt.plot(np.rad2deg(inputs[:, 1]), np.rad2deg(predictions), color, label=f'{DeltaNCs[i]:.2f}mm')
            count += 1


    plt.xlabel('DeltaNC')
    plt.ylabel('Collective')
    plt.legend()
    plt.show()



    
    