# Importing stuff
import numpy as np
import tensorflow as tf
import time

# Hyperparameters
DIFF = True
LEARNING_RATE = 0.00001
EPOCHS = 50000
WEIGHT_SPREAD = 0.02
ALPHA = 0.1


tf.config.run_functions_eagerly(True) # This is to make sure that the code runs in eager mode

class NN(tf.keras.Model):
    def __init__(self, in_size, out_size):
        super(NN, self).__init__()


        # Define weights and biases        
        self.w11 = tf.Variable(tf.cast(WEIGHT_SPREAD*np.random.rand(1)[0], dtype=tf.float32), trainable=True)
        self.w12 = tf.Variable(tf.cast(0., dtype=tf.float32), trainable=False)
        self.w13 = tf.Variable(tf.cast(0., dtype=tf.float32), trainable=False)
        self.w21 = tf.Variable(tf.cast(WEIGHT_SPREAD*np.random.rand(1)[0], dtype=tf.float32), trainable=True)
        self.w22 = tf.Variable(tf.cast(WEIGHT_SPREAD*np.random.rand(1)[0], dtype=tf.float32), trainable=True)
        self.w23 = tf.Variable(tf.cast(WEIGHT_SPREAD*np.random.rand(1)[0], dtype=tf.float32), trainable=True)
        self.w31 = tf.Variable(tf.cast(WEIGHT_SPREAD*np.random.rand(1)[0], dtype=tf.float32), trainable=True)
        self.w32 = tf.Variable(tf.cast(WEIGHT_SPREAD*np.random.rand(1)[0], dtype=tf.float32), trainable=True)
        self.w33 = tf.Variable(tf.cast(0., dtype=tf.float32), trainable=False)



    def call(self, x):
        x = tf.cast(x, dtype=tf.float32)    
        # Forward pass
        inter11 = tf.nn.relu(x[:,0]) 
        inter21 = self.w11*inter11 + self.w12*tf.pow(inter11,2) + self.w13*tf.pow(inter11,3)
        inter12 = -tf.nn.relu(-x[:,0])
        inter22 = self.w21*inter12 + self.w22*tf.pow(inter12,2) + self.w23*tf.pow(inter12,3)
        inter32 = self.w31*x[:,1] + self.w32*tf.pow(x[:,1],2) + self.w33*tf.pow(x[:,1],3)
        output = tf.multiply(inter21+inter22,inter32)

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
def getData(cond='train'):
    # Loading simulated data
    col_M = "coll_matrix.npy"
    cyc_M = "cyc_matrix.npy"
    col_Input = "col_input.npy"
    cyc_Input = "cyc_input.npy"

    AVG_COLL = np.load("Final_Project_Data/" + col_M)# average collective output
    AVG_CYC = np.load("Final_Project_Data/" + cyc_M) # average cyclic output
    DeltaNCs = np.load("Final_Project_Data/" + col_Input) # Delta NC input swashplate rise
    thetas = np.load("Final_Project_Data/" + cyc_Input) # Theta input swashplate angle


    # If we want to model residule
    normCol = AVG_COLL[:,0].reshape((-1,1))
    zero_ind = np.argmin(np.abs(DeltaNCs))
    normCyc = AVG_CYC[zero_ind,:].reshape((1,-1))
    AVG_COLL_RAW = AVG_COLL
    AVG_CYC_RAW = AVG_CYC

    AVG_COLL = AVG_COLL - normCol
    AVG_CYC = AVG_CYC - normCyc

    # Turn these tables into input and output pairs
    # Define possible values for each input
    input1_values = DeltaNCs
    input2_values = thetas

    # Create meshgrids for each input
    input1_mesh, input2_mesh = np.meshgrid(input1_values, input2_values, indexing='ij')

    # Flatten the meshgrids to create input array
    INPUT_ARRAY = np.stack((input1_mesh.flatten(), input2_mesh.flatten()), axis=-1)
    condition = (np.abs(INPUT_ARRAY[:,0]) <= 6.5) & (INPUT_ARRAY[:,1] <= np.deg2rad(12))
    INPUT_ARRAY_TRAINING = INPUT_ARRAY[condition]
    INPUT_ARRAY_TESTING = INPUT_ARRAY[~condition]
    # Example data table (3x4) with output values
    data_table1 = AVG_COLL
    data_table2 = AVG_CYC

    # Flatten the data table to create output array
    COLL = data_table1.flatten()
    COLL_TRAINING = COLL[condition]
    COLL_TESTING = COLL[~condition]
    CYC = data_table2.flatten()
    CYC_TRAINING = CYC[condition]
    CYC_TESTING = CYC[~condition]

    # Flatten the AVG_COLL_RAW and AVG_CYC_RAW tables to create output arrays
    AVG_COLL_RAW = AVG_COLL_RAW.flatten()
    AVG_CYC_RAW = AVG_CYC_RAW.flatten()

    


    if cond.lower() == "train":
        return INPUT_ARRAY_TRAINING, COLL_TRAINING, CYC_TRAINING
    elif cond.lower() == "test":
        return INPUT_ARRAY_TESTING, COLL_TESTING, CYC_TESTING
    elif cond.lower() == "all":
        return INPUT_ARRAY, COLL, CYC, DeltaNCs, thetas
    else:
        return INPUT_ARRAY, AVG_COLL_RAW, AVG_CYC_RAW, DeltaNCs, thetas
    
if __name__ == '__main__':
    
    INPUT_ARRAY_TRAINING, COLL_TRAINING, CYC_TRAINING = getData('train')

    # create dictionary for data input and output
    data_coll = {'x_train': INPUT_ARRAY_TRAINING, 'y_train': COLL_TRAINING}
    data_cyc = {'x_train': INPUT_ARRAY_TRAINING, 'y_train': CYC_TRAINING}

    nn(data_coll, 'COLL')
    # nn(data_cyc, 'CYC')

    

