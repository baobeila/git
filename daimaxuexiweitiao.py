#coding=utf-8
#Version:python3.5.2
import os
# CUDA_VISIBLE_DEVICES=0
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import tensorflow as tf

from daimaxuexi import AlexNet
# from datagenerator import ImageDataGenerator
from datetime import datetime
from tensorflow.contrib.data import Iterator
import nnprocess1
"""
Configuration Part.
"""

# Path to the textfiles for the trainings and validation set
train_file = '/path/to/train.txt'
val_file = '/path/to/val.txt'

# Learning params
learning_rate = 0.0001
num_epochs =200
batch_size = 64
train_batches_per_epoch = 20
# Network params
dropout_rate = 0.5
num_classes = 6
train_layers = ['fc8', 'fc7', 'fc6']
TFRECORD_FILE = r'D:\\2345Downloads\\tfrecord\\train.tfrecords'
# How often we want to write the tf.summary data to disk
display_step = 2

# Path for tf.summary.FileWriter and to store model checkpoints
filewriter_path = "/tmp/finetune_alexnet/tensorboard"
checkpoint_path = "/tmp/finetune_alexnet/checkpoints"

"""
Main Part of the finetuning Script.
"""

# Create parent path if it doesn't exist
# if not os.path.isdir(checkpoint_path):
#     os.mkdir(checkpoint_path)
if not os.path.exists(checkpoint_path):
      os.makedirs(checkpoint_path)

deimage, label= nnprocess1.read_and_decode(TFRECORD_FILE)
# image = tf.reshape(deimage, [224, 224])
image = tf.concat([deimage,deimage,deimage], axis=2)#(224, 224, 3)
min_after_dequeue = 1000
#当一次出列操作完成后,队列中元素的最小数量,往往用于定义元素的混合级别.

#定义了随机取样的缓冲区大小,此参数越大表示更大级别的混合但是会导致启动更加缓慢,并且会占用更多的内存
capacity = min_after_dequeue + 3 * batch_size
image_batch0, label_batch0 = tf.train.shuffle_batch(
        [image, label], batch_size=batch_size,
        capacity=capacity, min_after_dequeue=min_after_dequeue,
        num_threads=32)
# Place data loading and preprocessing on the cpu
# with tf.device('/cpu:0'):
#     tr_data = ImageDataGenerator(train_file,
#                                  mode='training',
#                                  batch_size=batch_size,
#                                  num_classes=num_classes,
#                                  shuffle=True)
#     val_data = ImageDataGenerator(val_file,
#                                   mode='inference',
#                                   batch_size=batch_size,
#                                   num_classes=num_classes,
#                                   shuffle=False)
#
#     # create an reinitializable iterator given the dataset structure
#     iterator = Iterator.from_structure(tr_data.data.output_types,
#                                        tr_data.data.output_shapes)
#     next_batch = iterator.get_next()
#
# # Ops for initializing the two different iterators
# training_init_op = iterator.make_initializer(tr_data.data)
# validation_init_op = iterator.make_initializer(val_data.data)

# TF placeholder for graph input and output
x = tf.placeholder(tf.float32, [batch_size, 224, 224, 3])
y = tf.placeholder(tf.float32, [batch_size, num_classes])
keep_prob = tf.placeholder(tf.float32)
one_hot_labels0 = tf.one_hot(indices=tf.cast(label_batch0, tf.int32), depth=num_classes)

# Initialize model
model = AlexNet(x, keep_prob, num_classes, train_layers)

# Link variable to model output
score = model.fc8

# List of trainable variables of the layers we want to train
var_list = [v for v in tf.trainable_variables() if v.name.split('/')[0] in train_layers]

# Op for calculating the loss
with tf.name_scope("cross_ent"):
    #该空间是为了在可视化显示
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=score,
                                                                  labels=y))

# Train op
with tf.name_scope("train"):
    # Get gradients of all trainable variables
    gradients = tf.gradients(loss, var_list)#计算梯度
    gradients = list(zip(gradients, var_list))

    # Create optimizer and apply gradient descent to the trainable variables
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizer.apply_gradients(grads_and_vars=gradients)#进行梯度更新

# Add gradients to summary
for gradient, var in gradients:
    tf.summary.histogram(var.name + '/gradient', gradient)

# Add the variables we train to the summary
for var in var_list:
    tf.summary.histogram(var.name, var)

# Add the loss to summary
tf.summary.scalar('cross_entropy', loss)


# Evaluation op: Accuracy of the model
with tf.name_scope("accuracy"):
    correct_pred = tf.equal(tf.argmax(score, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Add the accuracy to the summary
tf.summary.scalar('accuracy', accuracy)

# Merge all summaries together
merged_summary = tf.summary.merge_all()
# config = tf.ConfigProto(allow_soft_placement=True)
# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
# config.gpu_options.allow_growth = True

# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.7 # 占用GPU90%的显存

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
# Initialize the FileWriter
writer = tf.summary.FileWriter(filewriter_path)

# Initialize an saver for store model checkpoints
saver = tf.train.Saver()

# Get the number of training/validation steps per epoch
# train_batches_per_epoch = int(np.floor(tr_data.data_size/batch_size))
# val_batches_per_epoch = int(np.floor(val_data.data_size / batch_size))

# Start Tensorflow session
with tf.Session(config=config) as sess:

    # Initialize all variables
    sess.run(tf.global_variables_initializer())

    # Add the model graph to TensorBoard
    writer.add_graph(sess.graph)

    # Load the pretrained weights into the non-trainable layer
    model.load_initial_weights(sess)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    print("{} Start training...".format(datetime.now()))
    print("{} Open Tensorboard at --logdir {}".format(datetime.now(),
                                                      filewriter_path))

    # Loop over number of epochs
    for epoch in range(num_epochs):

        print("{} Epoch number: {}".format(datetime.now(), epoch+1))

        # Initialize iterator with the training dataset
        # sess.run(training_init_op)

        for step in range(train_batches_per_epoch):

            # get next batch of data
            img_batch, one_hot_labels = sess.run([image_batch0, one_hot_labels0])

            # And run the training op
            sess.run(train_op, feed_dict={x: img_batch,
                                          y: one_hot_labels ,
                                          keep_prob: dropout_rate})

            # Generate summary with the current batch of data and write to file
            if step % display_step == 0:
                train_acc = sess.run(accuracy, feed_dict={x: img_batch,y: one_hot_labels, keep_prob: dropout_rate})
                s = sess.run(merged_summary, feed_dict={x: img_batch,
                                                        y: one_hot_labels,
                                                        keep_prob: 1.})
                print("{} Train Accuracy = {:.4f}".format(datetime.now(),
                                                          train_acc))

                writer.add_summary(s, epoch*train_batches_per_epoch + step)
    coord.request_stop()
    coord.join(threads)

        # Validate the model on the entire validation set
        # print("{} Start validation".format(datetime.now()))
        # sess.run(validation_init_op)
        # test_acc = 0.
        # test_count = 0
        # for _ in range(val_batches_per_epoch):
        #
        #     img_batch, label_batch = sess.run([image_batch, label_batch0])
        #     acc = sess.run(accuracy, feed_dict={x: img_batch,
        #                                         y: label_batch,
        #                                         keep_prob: 1.})
        #     test_acc += acc
        #     test_count += 1
        # test_acc /= test_count#计算在验证集上的平均准确率
        # print("{} Validation Accuracy = {:.4f}".format(datetime.now(),
        #                                                test_acc))
        # print("{} Saving checkpoint of model...".format(datetime.now()))
        #
        # # save checkpoint of the model
        # checkpoint_name = os.path.join(checkpoint_path,
        #                                'model_epoch'+str(epoch+1)+'.ckpt')
        # save_path = saver.save(sess, checkpoint_name)
        #
        # print("{} Model checkpoint saved at {}".format(datetime.now(),
        #                                                checkpoint_name))

