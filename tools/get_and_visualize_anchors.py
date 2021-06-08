import glob
import xml.etree.ElementTree as ET
import argparse
import os
import re

import numpy as np
import matplotlib.pyplot as plt


class KMeans(object):

    def iou(self, boxes, clusters):
        """
        Calculates the Intersection over Union (IoU) between N boxes and K clusters.
        :param boxes: numpy array of shape (n, 2) where n is the number of box, shifted to the origin (i. e. width and height)
        :param clusters: numpy array of shape (k, 2) where k is the number of clusters
        :return: numpy array of shape (n, k) where k is the number of clusters
        """
        N = boxes.shape[0]
        K = clusters.shape[0]
        iw = np.minimum(
            np.broadcast_to(boxes[:, np.newaxis, 0], (N, K)),  # (N, 1) -> (N, K)
            np.broadcast_to(clusters[np.newaxis, :, 0], (N, K))  # (1, K) -> (N, K)
        )
        ih = np.minimum(
            np.broadcast_to(boxes[:, np.newaxis, 1], (N, K)),
            np.broadcast_to(clusters[np.newaxis, :, 1], (N, K))
        )
        if np.count_nonzero(iw == 0) > 0 or np.count_nonzero(ih == 0) > 0:
            raise ValueError("Some box has no area")

        intersection = iw * ih  # (N, K)
        boxes_area = np.broadcast_to((boxes[:, np.newaxis, 0] * boxes[:, np.newaxis, 1]), (N, K))
        clusters_area = np.broadcast_to((clusters[np.newaxis, :, 0] * clusters[np.newaxis, :, 1]), (N, K))

        iou_ = intersection / (boxes_area + clusters_area - intersection + 1e-7)

        return iou_

    def avg_iou(self, boxes, clusters):
        """
        Calculates the average Intersection over Union (IoU) between a numpy array of boxes and k clusters.
        :param boxes: numpy array of shape (r, 2), where r is the number of rows
        :param clusters: numpy array of shape (k, 2) where k is the number of clusters
        :return: average IoU as a single float
        """
        return np.mean(np.max(self.iou(boxes, clusters), axis=1))

    def translate_boxes(self, boxes):
        """
        Translates all the boxes to the origin.
        :param boxes: numpy array of shape (r, 4)
        :return: numpy array of shape (r, 2)
        """
        new_boxes = boxes.copy()
        for row in range(new_boxes.shape[0]):
            new_boxes[row][2] = np.abs(new_boxes[row][2] - new_boxes[row][0])
            new_boxes[row][3] = np.abs(new_boxes[row][3] - new_boxes[row][1])
        return np.delete(new_boxes, [0, 1], axis=1)

    def kmeans(self, boxes, k, dist=np.median):
        """
        Calculates k-means clustering with the Intersection over Union (IoU) metric.
        :param boxes: numpy array of shape (r, 2), where r is the number of rows
        :param k: number of clusters
        :param dist: distance function
        :return: numpy array of shape (k, 2)
        """
        rows = boxes.shape[0]

        distances = np.empty((rows, k))
        last_clusters = np.zeros((rows,))

        np.random.seed()

        # the Forgy method will fail if the whole array contains the same rows
        clusters = boxes[np.random.choice(rows, k, replace=False)]

        iter_num = 1
        while True:
            iter_num += 1

            distances = 1 - self.iou(boxes, clusters)
            nearest_clusters = np.argmin(distances, axis=1)

            if (last_clusters == nearest_clusters).all():
                break

            for cluster in range(k):
                if len(boxes[nearest_clusters == cluster]) == 0:
                    print("Cluster %d is zero size" % cluster)
                    # to avoid empty cluster
                    clusters[cluster] = boxes[np.random.choice(rows, 1, replace=False)]
                    continue

                clusters[cluster] = dist(boxes[nearest_clusters == cluster], axis=0)

            last_clusters = nearest_clusters

        return clusters


class KMeansCluster(object):
    def __init__(self, annotation_path, cluster_number, bbox_normalize=False):
        self.annotation_path = annotation_path
        self.cluster_number = cluster_number
        self.bbox_normalize = bbox_normalize

    def load_dataset(self):
        """
        Load dataset from PascalVOC format xml files
        """

        dataset = []
        for xml_file in glob.glob("{}/*xml".format(self.annotation_path)):
            tree = ET.parse(xml_file)
            height = int(tree.findtext("./size/height"))
            width = int(tree.findtext("./size/width"))

            for obj in tree.iter("object"):
                if self.bbox_normalize:
                    xmin = int(obj.findtext("bndbox/xmin")) / float(width)
                    xmax = int(obj.findtext("bndbox/xmax")) / float(width)
                    ymin = int(obj.findtext("bndbox/ymin")) / float(height)
                    ymax = int(obj.findtext("bndbox/ymax")) / float(height)
                else:
                    xmin = int(obj.findtext("bndbox/xmin"))
                    xmax = int(obj.findtext("bndbox/xmax"))
                    ymin = int(obj.findtext("bndbox/ymin"))
                    ymax = int(obj.findtext("bndbox/ymax"))

                if (xmax - xmin) == 0 or (ymax - ymin) == 0:
                    continue
                dataset.append([xmax - xmin, ymax - ymin])

        return np.array(dataset)

    def show_cluster(self, data, cluster, max_points=2000):
        '''
        Display bouding box's size distribution and anchor generated in scatter.
        '''
        if len(data) > max_points:
            idx = np.random.choice(len(data), max_points)
            data = data[idx]
        plt.scatter(data[:, 0], data[:, 1], s=5, c='black')
        plt.scatter(cluster[:, 0], cluster[:, 1], c='red', s=100, marker="^")
        plt.xlabel("Width")
        plt.ylabel("Height")
        plt.title("Bounding and anchor distribution")
        plt.savefig("cluster.png")
        plt.show()

    def show_width_height(self, data, cluster, bins=50):
        '''
        Display bouding box distribution with histgram.
        '''
        if data.dtype != np.float32:
            data = data.astype(np.float32)
        width = data[:, 0]
        height = data[:, 1]
        ratio = height / width

        plt.figure(1, figsize=(20, 6))
        plt.subplot(131)
        plt.hist(width, bins=bins, color='green')
        plt.xlabel('width')
        plt.ylabel('number')
        plt.title('Distribution of Width')

        plt.subplot(132)
        plt.hist(height, bins=bins, color='blue')
        plt.xlabel('Height')
        plt.ylabel('Number')
        plt.title('Distribution of Height')

        plt.subplot(133)
        plt.hist(ratio, bins=bins, color='magenta')
        plt.xlabel('Height / Width')
        plt.ylabel('number')
        plt.title('Distribution of aspect ratio(Height / Width)')
        plt.savefig("shape-distribution.png")
        plt.show()

    def sort_cluster(self, cluster):
        '''
        Sort the cluster to with area small to big.
        '''
        if cluster.dtype != np.float32:
            cluster = cluster.astype(np.float32)
        area = cluster[:, 0] * cluster[:, 1]
        cluster = cluster[area.argsort()]
        ratio = cluster[:, 1:2] / cluster[:, 0:1]
        return np.concatenate([cluster, ratio], axis=-1)

    def write_result_to_txt(self, sorted_result):
        if re.search("train", self.annotation_path):
            name = "train"
        else:
            name = "val"

        anchor_name = "yolo_anchors_" + name + ".txt"
        f = open(anchor_name, 'w')
        row = np.shape(sorted_result)[0]
        for i in range(row):
            if i == 0:
                x_y = "%d,%d" % (sorted_result[i][0], sorted_result[i][1])
            else:
                x_y = ", %d,%d" % (sorted_result[i][0], sorted_result[i][1])
            f.write(x_y)
        f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='''
                                    --dataset Input your WIDER FACE dataset path''')
    parser.add_argument('--dataset', default='', help='Path to WIDER FACE Annotation dataset')

    args = parser.parse_args()

    wider_face_dir = args.dataset

    train_annotation = "WIDER_train/Annotations"
    val_annotation = "WIDER_val/Annotations"

    wider_train_anno_dir = os.path.join(wider_face_dir, train_annotation)
    wider_val_anno_dir = os.path.join(wider_face_dir, val_annotation)

    datasets = [wider_train_anno_dir, wider_val_anno_dir]

    num_cluster = 25
    kmeans = KMeans()

    for ds in datasets:
        kmeansCluster = KMeansCluster(ds, num_cluster)
        print("Start to load data annotations on: %s" % ds)
        data = kmeansCluster.load_dataset()
        print("Start to do kmeans, please wait for a moment.")
        out = kmeans.kmeans(data, num_cluster)
        out_sorted = kmeansCluster.sort_cluster(out)
        print("Accuracy: {:.2f}%".format(kmeans.avg_iou(data, out) * 100))
        kmeansCluster.write_result_to_txt(out_sorted)
        kmeansCluster.show_cluster(data, out, max_points=2000)

        if out.dtype != np.float32:
            out = out.astype(np.float32)

        print("Recommanded aspect ratios(width/height)")
        print("Width    Height   Height/Width")
        for i in range(len(out_sorted)):
            print("%.3f      %.3f     %.1f" % (out_sorted[i, 0], out_sorted[i, 1], out_sorted[i, 2]))
        kmeansCluster.show_width_height(data, out, bins=50)
