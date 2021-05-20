import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn.utils import shuffle

# workaround for MacOS/jupyter notebook bug w/ tensorflow
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# adapted from original code in project 2
def divide_data(weather):
    '''divide dataset into two sets: 90% train and 10% test'''
    n = weather.shape[0]
    
    # shuffle data for test/train so no patterns
    weather = shuffle(weather)
    
    # take out 10% of the data for validation
    ind_test = np.random.choice(n, size = n // 10, replace = False)
    weather_test = weather.iloc[ind_test]

    # take the other 90% for building the model
    ind_train = [x for x in range(n) if x not in ind_test] # not in index
    weather_train = weather.iloc[ind_train]

    return weather_test, weather_train

# adapted from labs 22 and 23
def separate_targets(weather):
    '''separate dataset into features and targets'''
    # target: whether next day rains
    target = weather[['RAIN']].iloc[1:, :]
    target = np.round(target.to_numpy().reshape(-1))

    # feature: today's weather (array of 14 vars)
    feature = weather.iloc[:-1].to_numpy()
    
    return feature, target

def build_model(first_layer, num_output_vals):
    ''' build NN model'''
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(first_layer, activation='sigmoid', dtype='float64'),
        tf.keras.layers.Dropout(0.2, dtype='float64'),
        tf.keras.layers.Dense(num_output_vals, activation='sigmoid', dtype='float64')
    ])
    
    model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
    return model

# input week is days 0-6, target week is days 1-7
def split_input_target(week):
    '''duplicate and shift weeks to form input and target days'''
    input_days = week[:-1]
    target_days = week[1:]
    return input_days, target_days

def build_model_RNN(num_output_vals, embedding_dim, rnn_units, batch_size):
    '''build an RNN'''
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(num_output_vals, embedding_dim,
                                  batch_input_shape=[batch_size, None]),
        tf.keras.layers.GRU(rnn_units,
                            return_sequences=True,
                            stateful=True,
                            recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dense(num_output_vals)
    ])
    return model

def restore_model():
    '''restore model from save'''
    tf.train.latest_checkpoint(checkpoint_dir)

    modelRNN = build_model(num_output_vals, embedding_dim, rnn_units, batch_size=1)
    modelRNN.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
    modelRNN.build(tf.TensorShape([1, None]))
    
    return modelRNN

def generate_weather(model, start_weather):
    '''generate rain predictions for the next 7 days'''
    # Evaluation step (generating weather using the learned model)

    # Number of days to generate
    num_generate = 7

    # Empty string to store our results
    days_generated = []

    # Low temperatures results in more predictable results.
    # Higher temperatures results in more surprising results.
    # Experiment to find the best setting.
    temperature = 1.0

    # Here batch size == 1
    model.reset_states()
    for i in range(num_generate):
        predictions = model(start_weather)
        # remove the batch dimension
        predictions = tf.squeeze(predictions, 0)

        # using a categorical distribution to predict the temperature returned by the model
        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()
        
        # we pass the predicted weather as the next input to the model
        # along with the previous hidden state
        start_weather = tf.expand_dims([predicted_id], 0)

        days_generated.append(predicted_id)

    return days_generated

def part1():
    # import data
    np.random.seed(4471)
    weather_pd = pd.read_csv('../data/weather.csv', index_col = 0)
    weather_pd = weather_pd.drop(['DAY', 'STP', 'GUST'], axis=1)

    # whether it rained that day
    weather_pd['RAIN'] = (weather_pd['PRCP'] > 0).astype(int)

    #  divide into test/train and feature/target
    weather_test, weather_train = divide_data(weather_pd)
    feature_test, target_test = separate_targets(weather_test)
    feature_train, target_train = separate_targets(weather_train)

    # model specs
    # how many possible outputs
    num_output_vals = len(np.unique(target_test))
    # num nodes in first layer
    first_layer = 64
    model = build_model(first_layer, num_output_vals)

    # train model
    EPOCHS = 30
    history = model.fit(feature_train, target_train, epochs=EPOCHS)

    # predict for example day
    predicted = model.predict(feature_test[88].reshape(1,14))

    # evaluate model performance
    model.evaluate(feature_test, target_test, verbose=2)
    return predicted

def part2():
    # extract rain column
    rain = weather_pd[['RAIN']].to_numpy().reshape(-1)
    rain.shape

    # week length
    seq_length = 7
    examples_per_epoch = len(rain)//(seq_length+1)

    # convert to tf
    rain_tf = tf.data.Dataset.from_tensor_slices(rain)

    # example data
    for i in rain_tf.take(5):
      print(i.numpy())

    # turn individual days into week sequences
    weeks = rain_tf.batch(seq_length+1, drop_remainder=True)

    dataset = weeks.map(split_input_target)

    # create training batches of size 4
    BATCH_SIZE = 4
    # buffer size to shuffle dataset
    BUFFER_SIZE = 10000
    dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
    dataset

    # how many possible outputs
    num_output_vals = len(np.unique(rain))
    # embedding dimension
    embedding_dim = 256
    # number of RNN units
    rnn_units = 16
    modelRNN = build_model_RNN(num_output_vals, embedding_dim, rnn_units, BATCH_SIZE)

    # set up loss function
    loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
    # compile the model
    modelRNN.compile(optimizer='adam', loss=loss)

    # configure checkpoints to save during training
    # directory where the checkpoints will be saved
    checkpoint_dir = './training-checkpoints'
    # name checkpoint files
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_prefix,
        save_weights_only=True)

    # train the model
    # EPOCHS = 30
    # history = modelRNN.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])

    # restore model from save
    modelRNN = restore_model()

    # form predictions for next week
    example_week = np.array([[0, 0, 0, 0, 0, 0, 0]])
    predicted = generate_weather(modelRNN, start_weather=example_week)
    return predicted
