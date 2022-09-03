import cv2
import numpy as np
import os
import random

import torch
from torchvision import transforms
from torch.utils import data

from library.File import *

from .ClassAverages import ClassAverages

# TODO: clean up where this is
def generate_bins(bins):
    angle_bins = np.zeros(bins)
    interval = 2 * np.pi / bins
    for i in range(1,bins):
        angle_bins[i] = i * interval
    angle_bins += interval / 2 # center of the bin

    return angle_bins

class Dataset(data.Dataset):
    def __init__(self, path, bins=2, overlap=0.1):
        self.top_label_path = path + "/label_2/"
        self.top_img_path = path + "/image_2/"
        self.top_calib_path = path + "/calib/"

        # # TODO: which camera cal to use, per frame or global one?
        # self.proj_matrix = get_P(os.path.abspath(os.path.dirname(os.path.dirname(__file__)) + '/camera_cal/calib_cam_to_cam.txt'))

        self.ids = [x.split('.')[0] for x in sorted(os.listdir(self.top_img_path))] # name of file
        self.num_images = len(self.ids)

        # create angle bins
        self.bins = bins
        self.angle_bins = np.zeros(bins)
        self.interval = 2 * np.pi / bins
        for i in range(1,bins):
            self.angle_bins[i] = i * self.interval
        self.angle_bins += self.interval / 2 # center of the bin

        self.overlap = overlap
        # ranges for confidence
        # [(min angle in bin, max angle in bin), ... ]
        self.bin_ranges = []
        for i in range(0,bins):
            self.bin_ranges.append(( (i*self.interval - overlap) % (2*np.pi), \
                                (i*self.interval + self.interval + overlap) % (2*np.pi)) )

        # Hold average dimensions
        class_list = ['Car', 'Pedestrian', 'Cyclist']
        self.averages = ClassAverages(class_list)

        self.object_list = self.get_objects(self.ids)

        # Pre-fetch all labels
        self.labels = {}
        last_id = ""
        for obj in self.object_list:
            id = obj[0]
            line_num = obj[1]
            label = self.get_label(id, line_num)
            if id != last_id:
                self.labels[id] = {}
                last_id = id

            self.labels[id][str(line_num)] = label

        # Hold one image at a time
        self.curr_id = ""
        self.curr_img = None


    # Return (Input, Label)
    def __getitem__(self, index):
        id = self.object_list[index][0]
        line_num = self.object_list[index][1]

        if id != self.curr_id:
            self.curr_id = id
            self.curr_img = cv2.imread(self.top_img_path + '%s.png'%id)

        label = self.labels[id][str(line_num)]
        
        # obj = DetectedObject(self.curr_img, label['Class'], label['Box_2D'], proj_matrix, label=label)
        bbox = [label['Box_2D'][0][0], label['Box_2D'][0][1], label['Box_2D'][1][0], label['Box_2D'][1][1]]
        cropped_img = format_img(self.curr_img, bbox)

        return cropped_img, label

    def __len__(self):
        return len(self.object_list)

    def get_objects(self, ids):
        objects = []
        for id in ids:
            with open(self.top_label_path + '%s.txt'%id) as file:
                for line_num, line in enumerate(file):
                    line = line[:-1].split(' ')
                    obj_class = line[0]

                    if self.averages.recognized_class(obj_class):                    
                        # 3D bounding box dimension
                        dimension = np.array([float(line[8]), float(line[9]), float(line[10])], dtype=np.double)
                        self.averages.add_item(obj_class, dimension)

                        objects.append((id, line_num))

        self.averages.dump_to_file()
        return objects


    def get_label(self, id, line_num):
        lines = open(self.top_label_path + '%s.txt'%id).read().splitlines()
        label = self.format_label(lines[line_num])
        return label

    def get_bin(self, angle):

        bin_idxs = []

        def is_between(min, max, angle):
            max = (max - min) if (max - min) > 0 else (max - min) + 2*np.pi
            angle = (angle - min) if (angle - min) > 0 else (angle - min) + 2*np.pi
            return angle < max

        for bin_idx, bin_range in enumerate(self.bin_ranges):
            if is_between(bin_range[0], bin_range[1], angle):
                bin_idxs.append(bin_idx)

        return bin_idxs

    def format_label(self, line):
        line = line[:-1].split(' ')

        Class = line[0]

        for i in range(1, len(line)):
            line[i] = float(line[i])

        Alpha = line[3] # what we will be regressing
        Ry = line[14]
        top_left = (int(round(line[4])), int(round(line[5])))
        bottom_right = (int(round(line[6])), int(round(line[7])))
        Box_2D = [top_left, bottom_right]

        Dimension = np.array([line[8], line[9], line[10]], dtype=np.double) # height, width, length
        # modify for the average
        Dimension -= self.averages.get_item(Class)

        Location = [line[11], line[12], line[13]] # x, y, z
        Location[1] -= Dimension[0] / 2 # bring the KITTI center up to the middle of the object

        Orientation = np.zeros((self.bins, 2))
        Confidence = np.zeros(self.bins)

        # alpha is [-pi..pi], shift it to be [0..2pi]
        angle = Alpha + np.pi

        bin_idxs = self.get_bin(angle)

        for bin_idx in bin_idxs:
            angle_diff = angle - self.angle_bins[bin_idx]

            Orientation[bin_idx,:] = np.array([np.cos(angle_diff), np.sin(angle_diff)])
            Confidence[bin_idx] = 1

        label = {'Class': Class,
                'Box_2D': Box_2D,
                'Dimensions': Dimension,
                'Alpha': Alpha,
                'Orientation': Orientation,
                'Confidence': Confidence}

        return label

    # will be deprc soon
    def parse_label(self, label_path):
        buf = []
        with open(label_path, 'r') as f:
            for line in f:
                line = line[:-1].split(' ')

                Class = line[0]
                if Class == "DontCare":
                    continue

                for i in range(1, len(line)):
                    line[i] = float(line[i])

                Alpha = line[3] # what we will be regressing
                Ry = line[14]
                top_left = (int(round(line[4])), int(round(line[5])))
                bottom_right = (int(round(line[6])), int(round(line[7])))
                Box_2D = [top_left, bottom_right]

                Dimension = [line[8], line[9], line[10]] # height, width, length
                Location = [line[11], line[12], line[13]] # x, y, z
                Location[1] -= Dimension[0] / 2 # bring the KITTI center up to the middle of the object

                buf.append({'Class': Class,
                        'Box_2D': Box_2D,
                        'Dimensions': Dimension,
                        'Location': Location,
                        'Alpha': Alpha,
                        'Ry': Ry})
        return buf

    # will be deprc soon
    # def all_objects(self):
    #     data = {}
    #     for id in self.ids:
    #         data[id] = {}
    #         img_path = self.top_img_path + '%s.png'%id
    #         img = cv2.imread(img_path)
    #         data[id]['Image'] = img

    #         # Get projection matrix
    #         calib_file = os.path.join(self.top_calib_path, id + '.txt')
    #         proj_matrix = get_P_mat(calib_file)

    #         # # using P_rect from global calib file
    #         # proj_matrix = proj_matrix

    #         data[id]['Calib'] = proj_matrix

    #         label_path = self.top_label_path + '%s.txt'%id
    #         labels = self.parse_label(label_path)
    #         objects = []
    #         for label in labels:
    #             box_2d = label['Box_2D']
    #             detection_class = label['Class']
    #             objects.append(DetectedObject(img, detection_class, box_2d, proj_matrix, label=label))

    #         data[id]['Objects'] = objects
    #     return data


"""
Input to the neural net. Will hold the cropped image and the angle to that image, 
and (optionally) the label for the object. 
"""
class DetectedObject:
    def __init__(self, img, detection_class, box_2d, calib_file, label=None):
        self.xmin = box_2d[0]
        self.ymin = box_2d[1]
        self.xmax = box_2d[2]
        self.ymax = box_2d[3]
        self.box_2d = [int(self.xmin), int(self.ymin), \
                        int(self.xmax), int(self.ymax)]     # 2D bounding box 
        self.cropped_img = format_img(img, self.box_2d)     # Cropped image
        
        self.class_name = detection_class

        # if isinstance(proj_matrix, str):                # Projection matrix file name
        #     proj_matrix = get_P(proj_matrix)
        # self.proj_matrix = proj_matrix
        self.set_project_matrix(calib_file)

        self.theta_ray = self.calc_theta_ray(img)

        self.label = label

    def calc_theta_ray(self, img):
        width = img.shape[1]
        fovx = 2 * np.arctan(width / (2 * self.proj_matrix[0][0]))
        center = (self.xmax + self.xmin) / 2
        dx = center - (width / 2)

        mult = np.sign(dx)        
        dx = abs(dx)
        return np.arctan((2*dx*np.tan(fovx/2)) / width) * mult

    def set_project_matrix(self, calib_file):
        self.proj_matrix = get_P_mat(calib_file)

    def set_score(self, score):
        "Set the confidence score of the detection"
        self.score = float(score)

    def set_alpha(self, alpha):
        "Set the rotation ry around Y-axis in camera coordinates"
        self.alpha = alpha
        self.rotation_y = alpha + self.theta_ray
    
    def set_3d_dim(self, dim):
        "Set the dimensions for the 3D object"
        self.height = dim[0]
        self.width = dim[1]
        self.lenght = dim[2]

    def set_3d_location(self, location):
        "Set the location for the 3D object (x, y, z)"
        self.x = location[0]
        self.y = location[1]
        self.z = location[2]

    def get_kitti_label(self):
        label = [None] * 16
        label[0] = self.class_name                      # Type/class
        label[1] = 0                                    # Truncated value 
        label[2] = 0                                    # Occulated value
        label[3] = self.alpha                           # Observation angle of the object (alpha)
        label[4] = self.xmin                            # 2D bounding box coordinates 
        label[5] = self.ymin
        label[6] = self.xmax
        label[7] = self.ymax
        label[8] = self.height                          # 3D bounding box dimensions
        label[9] = self.width
        label[10] = self.lenght
        label[11] = self.x                               # 3D bounding box location
        label[12] = self.y
        label[13] = self.z
        label[14] = self.rotation_y                     # Rotation ry
        label[15] = self.score                          # Confidence score
        return " ".join(str(item) for item in label) 


def get_P_mat(file_name):
    for line in open(file_name, 'r'):
        if 'P2:' in line:
            p_mat = line.strip().split(' ')
            p_mat = np.asarray([float(num) for num in p_mat[1:]]).reshape((3, 4))
            return p_mat

def format_img(img, box_2d):
    # Torch transforms
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    process = transforms.Compose([transforms.ToTensor(), normalize])

    # Crop detection and resize
    crop = img[box_2d[1]:box_2d[3]+1, box_2d[0]:box_2d[2]+1]
    crop = cv2.resize(src=crop, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)

    # Recolor and reformat
    return process(crop)
