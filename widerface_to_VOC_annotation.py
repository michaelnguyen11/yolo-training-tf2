import os
from PIL import Image

data_dir = "/media/hiep/Ubuntu/datasets/wider-face/original/wider_face_split"

val_bbox_file = os.path.join(data_dir, "wider_face_val_bbx_gt.txt")
train_bbox_file = os.path.join(data_dir, "wider_face_train_bbx_gt.txt")

image_dir = "/media/hiep/Ubuntu/datasets/wider-face/original"
val_data_folder = os.path.join(image_dir, "WIDER_val")
train_data_folder = os.path.join(image_dir, "WIDER_train")

out_file_train = os.path.join(image_dir, "WIDER_train_1.txt")
out_file_val = os.path.join(image_dir, "WIDER_val_1.txt")


def generate_train_file(bbx_file, data_folder, out_file):
    paths_list, names_list = traverse_dir_files(data_folder)
    name_dict = dict()
    for path, name in zip(paths_list, names_list):
        name_dict[name] = path

    data_lines = read_file(bbx_file)

    sub_count = 0
    item_count = 0
    out_list = []
    for data_line in data_lines:
        item_count += 1
        if item_count % 1000 == 0:
            print('item_count: ' + str(item_count))

        data_line = data_line.strip()
        l_names = data_line.split('/')
        if len(l_names) == 2:
            if out_list:
                out_line = ' '.join(out_list)
                write_line(out_file, out_line)
                out_list = []

            name = l_names[-1]
            img_path = name_dict[name]
            img = Image.open(name_dict[name])
            # print(img.size[0], img.size[1])
            sub_count = 1
            out_list.append(img_path)
            continue

        if sub_count == 1:
            sub_count += 1
            continue

        if sub_count >= 2:
            n_list = data_line.split(' ')
            x_min = n_list[0]
            y_min = n_list[1]
            x_max = str(int(n_list[0]) + int(n_list[2]))
            y_max = str(int(n_list[1]) + int(n_list[3]))
            if int(x_max) - int(x_min) == 0 or int(y_max) - int(y_min) == 0:
                out_list.pop()
                continue

            p_list = ','.join([x_min, y_min, x_max, y_max, "0"])  # Tags are all 0, face
            out_list.append(p_list)
            continue


def traverse_dir_files(root_dir, ext=None):
    """
    List files in a folder, deep traversal
     :param root_dir: root directory
     :param ext: suffix name
     :return: [file path list, file name list]
    """
    names_list = []
    paths_list = []
    for parent, _, fileNames in os.walk(root_dir):
        for name in fileNames:
            if name.startswith('.'):  # Remove hidden files
                continue
            if ext:  # Search by suffix
                if name.endswith(tuple(ext)):
                    names_list.append(name)
                    paths_list.append(os.path.join(parent, name))
            else:
                names_list.append(name)
                paths_list.append(os.path.join(parent, name))
    paths_list, names_list = sort_two_list(paths_list, names_list)
    return paths_list, names_list


def sort_two_list(list1, list2):
    """
    Sort two lists
     :param list1: list 1
     :param list2: list 2
     :return: two lists after sorting
    """
    list1, list2 = (list(t) for t in zip(*sorted(zip(list1, list2))))
    return list1, list2


def read_file(data_file, mode='more'):
    """
    Read files, original files and data files
     :return: single row or array
    """
    try:
        with open(data_file, 'r') as f:
            if mode == 'one':
                output = f.read()
                return output
            elif mode == 'more':
                output = f.readlines()
                # return map(str.strip, output)
                return output
            else:
                return list()
    except IOError:
        return list()


def write_line(file_name, line):
    """
    Write row data to file
     :param file_name: file name
     :param line: line data
     :return: None
    """
    if file_name == "":
        return
    with open(file_name, "a+") as fs:
        if type(line) is (tuple or list):
            fs.write("%s\n" % ", ".join(line))
        else:
            fs.write("%s\n" % line)

if __name__ == '__main__':
    generate_train_file(val_bbox_file, val_data_folder, out_file_val)
    generate_train_file(train_bbox_file, train_data_folder, out_file_train)