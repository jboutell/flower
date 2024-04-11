import tensorflow as tf
import numpy as np
import time
import random
from glob import glob
from tensorflow import keras
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.utils import *
import decord
from decord import VideoReader
import tensorflow_hub as hub
import tensorflow_addons as tfa

def all_from_dir(local_path):
    path = np.sort(np.array(glob(local_path)))
    samples = list()
    labels = list()
    for j in range(len(path)):
        v = np.sort(np.array(glob(path[j]+'/*')))  
        random.shuffle(v)
        for i in range(len(v)):
            samples.append(v[i]) 
            labels.append(j) 
    return np.array(samples), np.array(labels)

def format_frames(frame, resolution):
    output_size = [resolution, resolution]
    frame = tf.image.convert_image_dtype(frame, tf.uint8)
    frame = tf.image.resize(frame, size=list(output_size))
    frame = tf.image.per_image_standardization(frame)  
    return frame
        
def get_n_frames(x, num_samples):
    # create an array of indices 0 .. num_samples with stepping 1 in tf.float32
    indices = tf.linspace(tf.cast(0, tf.float32), tf.cast(num_samples-1, tf.float32), num_samples)
	# take at max as many frames as there are in the video clip
    indices = tf.clip_by_value(indices, 0, tf.cast(tf.shape(x)[0] - 1, tf.float32))
    # convert indices to tf.int32 -- important for cases where we have less than num_samples frames!
    indices = tf.cast(tf.round(indices), tf.int32)
    # return as many frames from input video as 'indices' tells
    return tf.gather(x, indices, axis=0)

# return video from file_path given frame_count and resolution
def get_clip(file_path, frame_count, resolution):
    vr = VideoReader(file_path)
    frames = vr.get_batch(range(len(vr))).asnumpy()
    video_tensor = format_frames(frames, resolution)
    frames = get_n_frames(video_tensor, frame_count)
    return frames

def ran_crop(vido):
    n1,n2,n3,n4 =tf.shape(vido).numpy() 
    dcrop =  tf.image.random_crop(vido, size=(n1,n2//2, n2//2 ,3))
    return np.array (tf.image.resize(dcrop,[n2,n3]))

def video_augment(vio):
    ff1 = np.random.rand(1) 
    if 0.0 <= ff1 <= .4:
        vio = tf.image.flip_left_right(vio)
        vio = tf.image.random_brightness(vio, 0.2)
        vio = tf.image.random_saturation(vio, 5, 10)
    elif 0.41 <= ff1 <= .8:
        vio = tf.image.flip_up_down(vio)
        vio =  tf.image.random_hue(vio, 0.2)
        vio = tf.image.random_contrast(vio, 0.2, 0.5)
    else:
        vio = ran_crop(vio)
    return vio

def prepare_dataset(data, batch_size, frame_count, resolution, n_class, shuffle=True, augment=False):
    num_samples = len(data)
    if shuffle:
        random.shuffle(data)
    # return video content in snippets of batch_size for up to num_samples of videos
    for offset in range(0, num_samples, batch_size):
        # *_batch is a list of batch_size file names
        video_batch = np.array(data[offset:offset+batch_size]) [:,0]
        label_batch = np.array(data[offset:offset+batch_size]) [:,1]
        # Initialise X_train and y_train arrays for this batch
        X_train = []
        y_train = []
        # load each video by filename (get_clip), trainsform each label to categorical type
        for (sample, label) in zip(video_batch, label_batch):
            label = to_categorical(label, n_class) 
            video_clip = get_clip(sample, frame_count, resolution)
            if augment:
                video_clip = video_augment(video_clip) 
            X_train.append(video_clip) 
            y_train.append(label)
        X_train = np.float32(np.array(X_train))
        y_train = np.array(y_train)
        yield X_train, y_train
                
def create_network(net1, input_img, class_count):
    y0 = Lambda(lambda x: tf.transpose(x, perm=(0,4,1,2,3)))(input_img)
    y0 = net1(y0)
    y0 = Lambda(lambda x: tf.transpose(x, perm=(0,2,3,4,1)))(y0)
    y0 = tfa.layers.AdaptiveAveragePooling3D((1, 1, 1))(y0)
    y0 = Flatten()(y0)
    y0 = Dense(512, activation='relu')(y0)
    y0 = Dropout(.3)(y0)
    y = Dense(class_count , activation='softmax')(y0)
    return y
        
# declaring the training step as a separate @tf.function can save from out-of-memory error
@tf.function
def video_train_step(x_batch_train, y_batch_train, model, optimizer, loss_fn, train_acc_metric):
    with tf.GradientTape() as tape:
        logits = model(x_batch_train, training=True)
        loss_value = loss_fn(y_batch_train, logits)
    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))

    train_acc_metric.update_state(y_batch_train, logits)
    return loss_value

def video_val_step(val_dataset, model, val_acc_metric):
    for x_batch_val, y_batch_val in val_dataset:
        val_logits = model(x_batch_val, training=False)
        val_acc_metric.update_state(y_batch_val, val_logits)
    return val_acc_metric.result()

def video_fit(train, val, model, optimizer, loss_fn, batch_size, epochs, frame_count, class_count):
    train_acc_metric = tf.keras.metrics.CategoricalAccuracy()
    val_acc_metric   = tf.keras.metrics.CategoricalAccuracy()
    for epoch in range(epochs):
        train_dataset = prepare_dataset(train, batch_size, frame_count, resolution, class_count, shuffle=True, augment=False)
        val_dataset = prepare_dataset(val, batch_size, frame_count, resolution, class_count, shuffle=True, augment=False)
        print("\nStart of epoch %d" % (epoch,))
        start_time = time.time()

        # Iterate over the batches of the dataset.
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            print(x_batch_train.shape)
            loss_value = video_train_step(x_batch_train, y_batch_train, model, optimizer, loss_fn, train_acc_metric)
            print(
                "Training loss (for one batch) at step %d: %.4f"
                % (step, float(loss_value))
            )

        # Display metrics at the end of each epoch.
        train_acc = train_acc_metric.result()
        print("Training acc over epoch: %.4f" % (float(train_acc),))

        # Reset training metrics at the end of each epoch
        train_acc_metric.reset_states()

        # Run a validation loop at the end of each epoch.
        val_acc = video_val_step(val_dataset, model, val_acc_metric)
        print("Validation acc: %.4f" % (float(val_acc),))
        val_acc_metric.reset_states()
        print("Time taken: %.2fs" % (time.time() - start_time))
    return model

# --- load training and validation data ---
train_path = './kin6/train/*'
val_path = './kin6/test/*'

iv = 10 # small portion of the val set for after epoch testing
train_samples, train_labels = all_from_dir(train_path)
train_set = list(zip(train_samples, train_labels))

tmp_samples, tmp_labels = all_from_dir(val_path)

total_test_samples = len(tmp_labels)
val_samples = tmp_samples[0:iv]
val_labels = tmp_labels[0:iv]
test_samples = tmp_samples[iv:total_test_samples]
test_labels = tmp_labels[iv:total_test_samples]

val_set = list(zip(val_samples, val_labels))
test_set = list(zip(test_samples, test_labels))

class_count = np.max(train_labels)+1

print("Number of train videos: " + str(len(train_samples)))
print("Number of val videos: " + str(len(val_samples)))
print("Number of test videos: " + str(len(test_samples)))
print("Classes to learn: " + str(class_count))

# --- load base models and create network ---
resolution = 224 # Load SwinFormer feature extractor
epochs = 1
batch_size = 4
frame_count = 32

swin = hub.KerasLayer("https://tfhub.dev/shoaib6174/swin_base_patch244_window877_kinetics600_22k/1", trainable=False) 
input_img = tf.keras.layers.Input(shape=(frame_count, resolution, resolution, 3))
y = create_network(swin, input_img, class_count)

model = Model(input_img, y)
optimizer = tf.keras.optimizers.Adam(learning_rate=3e-4)
loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])

model = video_fit(train_set, val_set, model, optimizer, loss_fn, batch_size, epochs, frame_count, class_count)

test_dataset = prepare_dataset(test_set, batch_size, frame_count, resolution, class_count, shuffle=True, augment=False)
test_acc = video_val_step(test_dataset, model, tf.keras.metrics.CategoricalAccuracy())
print("Test acc: %.4f" % (float(test_acc),))

