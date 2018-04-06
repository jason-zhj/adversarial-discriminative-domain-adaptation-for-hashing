########################################
# parameters for image preprocessing
########################################
dataset_mean_value = 0.5
dataset_std_value = 0.5
dataset_mean = (dataset_mean_value, dataset_mean_value, dataset_mean_value)
dataset_std = (dataset_std_value, dataset_std_value, dataset_std_value)

########################################
# settings for training
########################################
# image-scale: 100 for office, 28 for mnist
image_scale = 28
shuffle_batch = True
train_data_path = {
    "source": "F:/data/mnist/mini/training",
    "target": "F:/data/USPS/train-sampled" # "F:/data/mnist_m/mini/train-10-percent"
}

iterations = 100
batch_size = 50
learning_rate = 1e-4
num_classes = 10

# settings for discriminator
dcd_input_dims = 1000
dcd_hidden_dims = 500
dcd_output_dims = 4

# loss coefficients
gamma = 0.5

########################################
# settings for ml_test
########################################

# test data path
test_data_path = "F:/data/USPS/test" # "F:/data/mnist_m/test"

test_batch_size = 100
# in ml_test, retrieval precision within hamming radius `precision_radius` will be calculated
precision_radius = 2