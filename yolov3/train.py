import os
import shutil

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from config import cfg
import utils
from dataset import Dataset
import yolov3

train_dataset = Dataset('train')
logdir = './data/log'
step_per_epoch = len(train_dataset)
global_steps = tf.Variable(1, trainable=False, dtype=tf.int64)
warmup_steps = cfg.YOLO.TRAIN.WARMUP_EPOCHS * step_per_epoch
total_steps = cfg.YOLO.TRAIN.EPOCHS * step_per_epoch

input_tensor = tf.keras.layers.Input([416, 416, 3])
conv_tensors = yolov3.YOLOv3(input_tensor)

output_tensors = []
for i, conv_tensor in enumerate(conv_tensors):
    pred_tensor = yolov3.decode(conv_tensor, i)
    output_tensors.append(conv_tensor)
    output_tensors.append(pred_tensor)

model = tf.keras.Model(input_tensor, output_tensors)
optimizer = tf.keras.optimizers.Adam()

if os.path.exists(logdir):
    shutil.rmtree(logdir)

writer = tf.summary.create_file_writer(logdir)


def train_step(image_data, target):
    with tf.GradientTape() as tape:
        pred_result = model(image_data, training=True)
        bbox_loss = conf_loss = prob_loss = 0

        # optimizing process
        for i in range(3):
            conv, pred = pred_result[i * 2], pred_result[i * 2 + 1]
            loss_items = yolov3.compute_loss(conv, pred, *target[i], i)
            bbox_loss += loss_items[0]
            conf_loss += loss_items[1]
            prob_loss += loss_items[2]

        total_loss = bbox_loss + conf_loss + prob_loss

        gradients = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        tf.print("=> STEP %4d   lr: %.6f   giou_loss: %4.2f   conf_loss: %4.2f   "
                 "prob_loss: %4.2f   total_loss: %4.2f" % (global_steps.nummpy(), optimizer.lr.numpy(),
                                                           bbox_loss, conf_loss,
                                                           prob_loss, total_loss))

        # update learning rate
        global_steps.assign_add(1)
        if global_steps.nummpy() < warmup_steps:
            lr = (global_steps.nummpy() / warmup_steps) * cfg.YOLO.TRAIN.LEARNING_RATE_INIT
        else:
            lr = cfg.YOLO.TRAIN.LEARNING_RATE_END + 0.5 * (
                    cfg.YOLO.TRAIN.LEARNING_RATE_INIT - cfg.YOLO.TRAIN.LEARNING_RATE_END) * (
                     (1 + tf.cos((global_steps.nummpy() - warmup_steps) / (total_steps - warmup_steps) * np.pi)))

        optimizer.lr.assign(lr.numpy())

        # writing summary data
        with writer.as_default():
            tf.summary.scalar("lr", optimizer.lr, step=global_steps)
            tf.summary.scalar("loss/total_loss", total_loss, step=global_steps)
            tf.summary.scalar("loss/bbox_loss", bbox_loss, step=global_steps)
            tf.summary.scalar("loss/conf_loss", conf_loss, step=global_steps)
            tf.summary.scalar("loss/prob_loss", prob_loss, step=global_steps)
        writer.flush()


for epoch in range(cfg.YOLO.TRAIN.EPOCHS):
    for image_data, target in train_dataset:
        train_step(image_data, target)
    model.save_weights("./yolov3")