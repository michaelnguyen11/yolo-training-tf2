import numpy as np
import tensorflow as tf
from config import cfg
import utils
import backbone
import common

NUM_CLASS = len(utils.read_class_name(cfg.YOLO.CLASSES))
ANCHORS = utils.get_anchors(cfg.YOLO.ANCHORS)
STRIDES = np.array(cfg.YOLO.STRIDES)
IOU_LOSS_THRESH = cfg.YOLO.IOU_LOSS_THRESH


def YOLOv3(inputs):
    route_1, route_2, conv = backbone.darknet53(inputs)

    conv = common.convolution(conv, (1, 1, 1024, 512))  # conv52
    conv = common.convolution(conv, (3, 3, 512, 1024))  # conv53
    conv = common.convolution(conv, (1, 1, 1024, 512))  # conv54
    conv = common.convolution(conv, (3, 3, 512, 1024))  # conv55
    conv = common.convolution(conv, (1, 1, 1024, 512))  # conv56

    conv_large_obj_branch = common.convolution(conv, (3, 3, 512, 1024))
    # conv_lbbox is used to predict large sizes Object, shape = [None, 13, 13, 255]
    conv_lbbox = common.convolution(conv_large_obj_branch, (1, 1, 1024, 3 * (NUM_CLASS + 5)), activate=False, bn=False)

    conv = common.convolution(conv, (1, 1, 512, 256))  # conv57
    conv = common.upsample(conv)

    conv = tf.concat([conv, route_2], axis=-1)
    conv = common.convolution(conv, (1, 1, 768, 256))  # conv58
    conv = common.convolution(conv, (3, 3, 256, 512))  # conv59
    conv = common.convolution(conv, (1, 1, 512, 256))  # conv60
    conv = common.convolution(conv, (3, 3, 256, 512))  # conv61
    conv = common.convolution(conv, (1, 1, 512, 256))  # conv62

    conv_medium_obj_branch = common.convolution(conv, (3, 3, 256, 512))
    # conv_mbbox is used to predict medium-sized objects, shape = [None, 26, 26, 255]
    conv_mbbox = common.convolution(conv_medium_obj_branch, (1, 1, 512, 3 * (NUM_CLASS + 5)), activate=False, bn=False)

    conv = common.convolution(conv, (1, 1, 256, 128))  # conv63
    conv = common.upsample(conv)

    conv = tf.concat([conv, route_1], axis=-1)
    conv = common.convolution(conv, (1, 1, 384, 128))  # conv64
    conv = common.convolution(conv, (3, 3, 128, 256))  # conv65
    conv = common.convolution(conv, (1, 1, 256, 128))  # conv66
    conv = common.convolution(conv, (3, 3, 128, 256))  # conv67
    conv = common.convolution(conv, (1, 1, 256, 128))  # conv68

    conv_small_obj_branch = common.convolution(conv, (3, 3, 128, 256))
    # conv_sbbox is used to predict small-sized objects, shape = [None, 52, 52, 255]
    conv_sbbox = common.convolution(conv_small_obj_branch, (1, 1, 256, 3 * (NUM_CLASS + 5)), activate=False, bn=False)

    return [conv_sbbox, conv_mbbox, conv_lbbox]


def decode(conv_output, i=0):
    # where i = [0, 1, 2] corresponds to three grid sizes respectively
    """
    return tensor of shape [batch_size, output_size, output_size, anchor_per_scale, 5 + num_classes]
            contains (x, y, w, h, score, probability)
    """
    conv_shape = tf.shape(conv_output)
    batch_size = conv_shape[0]
    output_size = conv_shape[1]

    conv_output = tf.reshape(conv_output, (batch_size, output_size, output_size, 3, 5 + NUM_CLASS))

    # The offset of the center position
    conv_raw_dxdy = conv_output[:, :, :, :, 0:2]
    # Prediction box length and width offset
    conv_raw_dwdh = conv_output[:, :, :, :, 2:4]
    # The confidence of the prediction box
    conv_raw_confident = conv_output[:, :, :, :, 4:5]
    # The category probability of the prediction box
    conv_raw_prob = conv_output[:, :, :, :, 5:]

    # Draw the grid, among them, output size is equal to 13, 25 or 52
    y = tf.tile(tf.range(output_size, dtype=tf.int32)[:, tf.newaxis], [1, output_size])
    x = tf.tile(tf.range(output_size, dtype=tf.int32)[tf.newaxis, :], [output_size, 1])

    xy_grid = tf.concat([x[:, :, tf.newaxis], y[:, :, tf.newaxis]], axis=-1)
    xy_grid = tf.tile(xy_grid[tf.newaxis, :, :, tf.newaxis, :], [batch_size, 1, 1, 3, 1])
    # Calculate the position of the upper left corner of the grid
    xy_grid = tf.cast(xy_grid, tf.float32)

    # Calculate the center position of the prediction box
    pred_xy = (tf.sigmoid(conv_raw_dxdy) + xy_grid) * STRIDES[i]
    # Calculate the length and width of the prediction box
    pred_wh = (tf.exp(conv_raw_dwdh) * ANCHORS[i]) * STRIDES[i]
    pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)
    # Calculate the confidence of the object in the prediction box
    pred_confident = tf.sigmoid(conv_raw_confident)
    # Calculate the category probability of the object in the prediction box
    pred_prob = tf.sigmoid(conv_raw_prob)

    return tf.concat([pred_xywh, pred_confident, pred_prob], axis=-1)


def bbox_iou(boxes1, boxes2):
    boxes1_area = boxes1[..., 2] * boxes1[..., 3]
    boxes2_area = boxes2[..., 2] * boxes2[..., 3]

    boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                        boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
    boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                        boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

    left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = tf.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area

    return 1.0 * inter_area / union_area


def bbox_giou(boxes1, boxes2):
    boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                        boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
    boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                        boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

    boxes1 = tf.concat([tf.minimum(boxes1[..., :2], boxes1[..., 2:]),
                        tf.maximum(boxes1[..., :2], boxes1[..., 2:])], axis=-1)
    boxes2 = tf.concat([tf.minimum(boxes2[..., :2], boxes2[..., 2:]),
                        tf.maximum(boxes2[..., :2], boxes2[..., 2:])], axis=-1)

    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = tf.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area
    iou = inter_area / union_area

    enclose_left_up = tf.minimum(boxes1[..., :2], boxes2[..., :2])
    enclose_right_down = tf.maximum(boxes1[..., 2:], boxes2[..., 2:])
    enclose = tf.maximum(enclose_right_down - enclose_left_up, 0.0)
    enclose_area = enclose[..., 0] * enclose[..., 1]
    giou = iou - 1.0 * (enclose_area - union_area) / enclose_area

    return giou


def bbox_iou2(boxes1, boxes2):
    boxes1 = np.array(boxes1)
    boxes2 = np.array(boxes2)

    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxes1[0], boxes2[0])
    yA = max(boxes1[1], boxes2[1])
    xB = min(boxes1[2], boxes2[2])
    yB = min(boxes1[3], boxes2[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxes1[2] - boxes1[0] + 1) * (boxes1[3] - boxes1[1] + 1)
    boxBArea = (boxes2[2] - boxes2[0] + 1) * (boxes2[3] - boxes2[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou


def bbox_giou2(boxes1, boxes2):
    boxes1 = np.array(boxes1)
    boxes2 = np.array(boxes2)

    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxes1[0], boxes2[0])
    yA = max(boxes1[1], boxes2[1])
    xB = min(boxes1[2], boxes2[2])
    yB = min(boxes1[3], boxes2[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxes1[2] - boxes1[0] + 1) * (boxes1[3] - boxes1[1] + 1)
    boxBArea = (boxes2[2] - boxes2[0] + 1) * (boxes2[3] - boxes2[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    union_area = boxAArea + boxBArea - interArea
    iou = interArea / float(union_area)
    # return the intersection over union value

    enclose_left_up = tf.minimum(boxes1[..., :2], boxes2[..., :2])
    enclose_right_down = tf.maximum(boxes1[..., 2:], boxes2[..., 2:])
    enclose = tf.maximum(enclose_right_down - enclose_left_up, 0.0)
    enclose_area = enclose[..., 0] * enclose[..., 1]
    giou = iou - 1.0 * (enclose_area - union_area) / enclose_area

    return giou


def compute_loss(pred, conv, label, bboxes, i=0):
    conv_shape = tf.shape(conv)
    batch_size = conv_shape[0]
    output_size = conv_shape[1]
    input_size = STRIDES[i] * output_size
    conv = tf.reshape(conv, (batch_size, output_size, output_size, 3, 5 + NUM_CLASS))

    conv_raw_conf = conv[:, :, :, :, 4:5]
    conv_raw_prob = conv[:, :, :, :, 5:]

    pred_xywh = pred[:, :, :, :, 0:4]
    pred_conf = pred[:, :, :, :, 4:5]

    label_xywh = label[:, :, :, :, 0:4]
    label_conf = label[:, :, :, :, 4:5]
    label_prob = label[:, :, :, :, 5:]

    giou = tf.expand_dims(bbox_giou(pred_xywh, label_xywh), axis=-1)
    input_size = tf.cast(input_size, tf.float32)

    # The smaller the size of the bounding box, the larger the value of bbox_loss_scale
    bbox_loss_scale = 2.0 - 1.0 * label_xywh[:, :, :, :, 2:3] * label_xywh[:, :, :, :, 3:4] / (input_size ** 2)
    bbox_loss = label_conf * bbox_loss_scale * (1 - giou)

    iou = bbox_iou(pred_xywh[:, :, :, :, np.newaxis, :], bboxes[:, np.newaxis, np.newaxis, np.newaxis, :, :])
    max_iou = tf.expand_dims(tf.reduce_max(iou, axis=-1), axis=-1)

    response_background = (1.0 - label_conf) * tf.cast(max_iou < IOU_LOSS_THRESH, tf.float32)
    conf_focal = tf.pow(label_conf - pred_conf, 2)

    conf_loss = conf_focal * (
            label_conf * tf.nn.sigmoid_cross_entropy_with_logits(labels=label_conf, logits=conv_raw_conf)
            + response_background * tf.nn.sigmoid_cross_entropy_with_logits(labels=label_conf,
                                                                            logits=conv_raw_conf))

    prob_loss = label_conf * tf.nn.sigmoid_cross_entropy_with_logits(labels=label_prob, logits=conv_raw_prob)

    bbox_loss = tf.reduce_mean(tf.reduce_sum(bbox_loss, axis=[1, 2, 3, 4]))
    conf_loss = tf.reduce_mean(tf.reduce_sum(conf_loss, axis=[1, 2, 3, 4]))
    prob_loss = tf.reduce_mean(tf.reduce_sum(prob_loss, axis=[1, 2, 3, 4]))

    return bbox_loss, conf_loss, prob_loss
