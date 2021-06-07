import numpy as np
import re


class YOLO_Kmeans:

    def __init__(self, cluster_number, filename):
        self.cluster_number = cluster_number
        self.filename = filename

    def iou(self, boxes, clusters):  # 1 box -> k clusters
        """
        Calculates the Intersection over Union (IoU) between a box and k clusters.
        param:
            box: tuple or array, shifted to the origin (i. e. width and height)
            clusters: numpy array of shape (k, 2) where k is the number of clusters
        return:
            numpy array of shape (k, 0) where k is the number of clusters
        """
        x = np.minimum(clusters[:, 0], boxes[0])
        y = np.minimum(clusters[:, 1], boxes[1])
        if np.count_nonzero(x == 0) > 0 or np.count_nonzero(y == 0) > 0:
            raise ValueError("Box has no area")

        intersection = x * y
        box_area = boxes[0] * boxes[1]
        cluster_area = clusters[:, 0] * clusters[:, 1]

        iou_ = np.true_divide(intersection, box_area + cluster_area - intersection + 1e-10)
        # iou_ = intersection / (box_area + cluster_area - intersection + 1e-10)

        return iou_

    def avg_iou(self, boxes, clusters):
        """
        Calculates the average Intersection over Union (IoU) between a numpy array of boxes and k clusters.
        param:
            boxes: numpy array of shape (r, 2), where r is the number of rows
            clusters: numpy array of shape (k, 2) where k is the number of clusters
        return:
            average IoU as a single float
        """
        return np.mean([np.max(self.iou(boxes[i], clusters)) for i in range(boxes.shape[0])])

    def kmeans(self, boxes, k, dist=np.median):
        """
        Calculates k-means clustering with the Intersection over Union (IoU) metric.
        param:
            boxes: numpy array of shape (r, 2), where r is the number of rows
            k: number of clusters
            dist: distance function
        return:
            numpy array of shape (k, 2)
        """
        rows = boxes.shape[0]

        distances = np.empty((rows, k))
        last_clusters = np.zeros((rows,))

        np.random.seed()

        # the Forgy method will fail if the whole array contains the same rows
        clusters = boxes[np.random.choice(rows, k, replace=False)]
        while True:
            for row in range(rows):
                distances[row] = 1 - self.iou(boxes[row], clusters)

            nearest_clusters = np.argmin(distances, axis=1)

            if (last_clusters == nearest_clusters).all():
                break

            for cluster in range(k):
                clusters[cluster] = dist(boxes[nearest_clusters == cluster], axis=0)

            last_clusters = nearest_clusters

        return clusters

    def result2txt(self, data):
        if re.search("train", self.filename):
            name = "train"
        else:
            name = "val"

        anchor_name = "data/yolo_anchors_" + name + ".txt"
        f = open(anchor_name, 'w')
        row = np.shape(data)[0]
        for i in range(row):
            if i == 0:
                x_y = "%d,%d" % (data[i][0], data[i][1])
            else:
                x_y = ", %d,%d" % (data[i][0], data[i][1])
            f.write(x_y)
        f.close()

    def txt2boxes(self):
        f = open(self.filename, 'r')
        dataSet = []
        for line in f:
            infos = line.split(" ")
            length = len(infos)
            for i in range(1, length):
                width = int(infos[i].split(",")[2]) - \
                        int(infos[i].split(",")[0])
                height = int(infos[i].split(",")[3]) - \
                         int(infos[i].split(",")[1])
                dataSet.append([width, height])
        result = np.array(dataSet)
        f.close()
        return result

    def txt2clusters(self):
        all_boxes = self.txt2boxes()
        result = self.kmeans(all_boxes, k=self.cluster_number)
        result = result[np.lexsort(result.T[0, None])]
        self.result2txt(result)

        print("K anchors:\n {}".format(result))
        print("Accuracy: {:.2f}%".format(
            self.avg_iou(all_boxes, result) * 100))


if __name__ == "__main__":
    cluster_number = 9
    filenames = ["data/WIDER_train_1.txt", "data/WIDER_val_1.txt"]
    for filename in filenames:
        kmeans = YOLO_Kmeans(cluster_number, filename)
        kmeans.txt2clusters()
