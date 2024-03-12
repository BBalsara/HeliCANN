# Importing stuff
import numpy as np
import tensorflow as tf
import time
import matplotlib.pyplot as plt

# Hyperparameters
DIFF = True
LEARNING_RATE = 0.00001
EPOCHS = 15000
WEIGHT_SPREAD = 0.01
ALPHA = 0.1

tf.config.run_functions_eagerly(True) # This is to make sure that the code runs in eager mode

class NN(tf.keras.Model):
    def __init__(self, in_size, out_size):
        super(NN, self).__init__()

        self.w41 = tf.Variable(tf.cast(WEIGHT_SPREAD*np.random.rand(1)[0], dtype=tf.float32), trainable=True)
        self.w42 = tf.Variable(tf.cast(WEIGHT_SPREAD*np.random.rand(1)[0], dtype=tf.float32), trainable=True)
        self.w43 = tf.Variable(tf.cast(WEIGHT_SPREAD*np.random.rand(1)[0], dtype=tf.float32), trainable=True)
        self.w51 = tf.Variable(tf.cast(WEIGHT_SPREAD*np.random.rand(1)[0], dtype=tf.float32), trainable=True)
        self.w52 = tf.Variable(tf.cast(WEIGHT_SPREAD*np.random.rand(1)[0], dtype=tf.float32), trainable=True)
        self.w53 = tf.Variable(tf.cast(WEIGHT_SPREAD*np.random.rand(1)[0], dtype=tf.float32), trainable=True)


    def call(self, x):
        x = tf.cast(x, dtype=tf.float32)    
        # Forward pass
        inter14 = tf.nn.relu(x) 
        inter24 = self.w41*inter14 + self.w42*tf.pow(inter14,2) + self.w43*tf.pow(inter14,3)
        inter15 = -tf.nn.relu(-x)
        inter25 = self.w51*inter15 + self.w52*tf.pow(inter15,2) + self.w53*tf.pow(inter15,3)
        output = inter24 + inter25
        return output

def loss(y_est, y, weights):
    y = tf.cast(y, dtype=tf.float32)
    # Compute loss
    l = tf.norm(y_est - y) * 180 / np.pi
    # l += tf.norm(weights, 1) * ALPHA
    return l    

def nn(data, mode):
    """
    Trains a feedforward NN. 
    """
    params = {
        'train_batch_size': 4096,
    }
    in_size = data['x_train'].shape[-1]
    out_size = data['y_train'].shape[-1]
    
    nn_model = NN(in_size, out_size)

    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    train_loss = tf.keras.metrics.Mean(name='train_loss')

    @tf.function
    def train_step(x, y):
        with tf.GradientTape() as tape:
            y_est = nn_model.call(x)
            current_loss = loss(y_est, y, nn_model.trainable_variables)
        grads = tape.gradient(current_loss, nn_model.trainable_variables)
        optimizer.apply_gradients(zip(grads, nn_model.trainable_variables))
        train_loss(current_loss)

    @tf.function
    def train(train_data):
        for x, y in train_data:
            train_step(x, y)

    train_data = tf.data.Dataset.from_tensor_slices((data['x_train'], data['y_train'])).shuffle(100000).batch(params['train_batch_size'])
    for epoch in range(EPOCHS):
        train_loss.reset_states()
        train(train_data)
        template = 'Epoch {}, Loss: {}'
        if (epoch+1) % 50 == 0:
            print(template.format(epoch + 1, train_loss.result()))

        # Print the weights every 1000 epochs
        if (epoch+1) % 1000 == 0:
            weights = nn_model.get_weights()
            print("Model Wights: ", weights)
        
    t = time.localtime()
    timestamp = time.strftime('%b-%d-%Y_%H%M', t)
    nn_model.save_weights('./policies/' + mode + '_' + timestamp)

# Data formulation
def getData_Dom(cond='train'):
    # Loading simulated data
    col_M = "coll_matrix.npy"
    cyc_M = "cyc_matrix.npy"
    col_Input = "col_input.npy"
    cyc_Input = "cyc_input.npy"

    AVG_COLL = np.load("Final_Project_Data/" + col_M)# average collective output
    AVG_CYC = np.load("Final_Project_Data/" + cyc_M) # average cyclic output
    DeltaNCs = np.load("Final_Project_Data/" + col_Input) # Delta NC input swashplate rise
    thetas = np.load("Final_Project_Data/" + cyc_Input) # Theta input swashplate angle

    # Turn these tables into input and output pairs
    # Define possible values for each input
    INPUT_ARRAY_COL = DeltaNCs
    INPUT_ARRAY_CYC = thetas



    condition_col = (np.abs(INPUT_ARRAY_COL) <= 6.5)
    condition_cyc = (INPUT_ARRAY_CYC <= np.deg2rad(12))
    INPUT_ARRAY_TRAINING_COL = INPUT_ARRAY_COL[condition_col]
    INPUT_ARRAY_TESTING_COL = INPUT_ARRAY_COL[~condition_col]
    INPUT_ARRAY_TRAINING_CYC = INPUT_ARRAY_CYC[condition_cyc]
    INPUT_ARRAY_TESTING_CYC = INPUT_ARRAY_CYC[~condition_cyc]
    # Example data table (3x4) with output values
    data_table1 = AVG_COLL[:, 0]
    zero_ind = np.argmin(np.abs(DeltaNCs))
    data_table2 = AVG_CYC[zero_ind,:]

    # Flatten the data table to create output array
    COLL = data_table1.flatten()
    COLL_TRAINING = COLL[condition_col]
    COLL_TESTING = COLL[~condition_col]
    CYC = data_table2.flatten()
    CYC_TRAINING = CYC[condition_cyc]
    CYC_TESTING = CYC[~condition_cyc]


    if cond.lower() == "train":
        return INPUT_ARRAY_TRAINING_COL, INPUT_ARRAY_TRAINING_CYC, COLL_TRAINING, CYC_TRAINING
    elif cond.lower() == "test":
        return INPUT_ARRAY_TESTING_COL, INPUT_ARRAY_TESTING_CYC, COLL_TESTING, CYC_TESTING
    elif cond.lower() == "all":
        return INPUT_ARRAY_COL, INPUT_ARRAY_CYC, COLL, CYC
    elif cond.lower() == "full":
        return INPUT_ARRAY_COL, INPUT_ARRAY_CYC, AVG_COLL, AVG_CYC
    else:
        raise ValueError("Invalid condition")
    
if __name__ == '__main__':
    
    INPUT_ARRAY_TRAINING_COL, INPUT_ARRAY_TRAINING_CYC, COLL_TRAINING, CYC_TRAINING = getData_Dom('train')

    # create dictionary for data input and output
    data_coll = {'x_train': INPUT_ARRAY_TRAINING_COL, 'y_train': COLL_TRAINING}
    data_cyc = {'x_train': INPUT_ARRAY_TRAINING_CYC, 'y_train': CYC_TRAINING}

    nn(data_coll, 'COLL_DOM')
    # nn(data_cyc, 'CYC_DOM')

    # print(np.linalg.norm(INPUT_ARRAY_TRAINING_CYC - CYC_TRAINING))
    # plt.plot(INPUT_ARRAY_TRAINING_CYC, CYC_TRAINING, 'b.', alpha=0.2)
    # plt.show()

    

