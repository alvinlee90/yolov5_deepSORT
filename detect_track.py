"""
Uses YOLO to obtain 2D box, PyTorch to get 3D box
"""

from torch_lib.Dataset import *
from library.Math import *
from library.Plotting import *
from torch_lib import Model, ClassAverages

import os
import argparse
import numpy as np
import cv2

import torch
from torchvision.models import vgg


def main(opts):
    # Load YOLOv5 model
    yolo_model = torch.hub.load('ultralytics/yolov5', 'custom', path=opts.yolo_model)
 
    # Load torch model
    weights_path = os.path.abspath(os.path.dirname(__file__)) + '/weights'
    model_lst = [x for x in sorted(os.listdir(weights_path)) if x.endswith('.pkl')]
    
    if len(model_lst) == 0:
        print('No previous model found, please train first!')
        exit()
    else:
        print('Using previous model %s'%model_lst[-1])
        my_vgg = vgg.vgg19_bn(pretrained=True)

        bbox_model = Model.Model(features=my_vgg.features, bins=2).cuda()
        checkpoint = torch.load(weights_path + '/%s'%model_lst[-1])
        bbox_model.load_state_dict(checkpoint['model_state_dict'])
        bbox_model.eval()

    # Class averages 
    averages = ClassAverages.ClassAverages()
    angle_bins = generate_bins(2)

    # Path for evaluation images     
    img_path = os.path.abspath(os.path.dirname(__file__)) + "/" + opts.image_dir
    
    # Using P2_rect from calibration file
    calib_dir = os.path.abspath(os.path.dirname(__file__)) + "/" + opts.cal_dir

    # Path for KITTI label outputs
    os.makedirs(opts.output_dir, exist_ok=True)
    output_path = os.path.abspath(os.path.dirname(__file__)) + "/" + opts.output_dir

    # Get frame id for all test images
    try:
        ids = [x.split('.')[0] for x in sorted(os.listdir(img_path))]
    except:
        print("\nError: no images in %s"%img_path)
        exit()

    # Evaluate each image 
    for img_id in ids:
        # Test image file 
        img_file = img_path + img_id + ".png"
        test_img = cv2.imread(img_file)        

        # Evaluate 2D bounding boxes - detection head
        results =  yolo_model(test_img)
        detections = results.pandas().xyxy[0].reset_index()
        label_output = []

        # Evaluate 3D bounding box for each 2D detection 
        for idx, detect in detections.iterrows():
            # Filter classes 
            if not averages.recognized_class(detect['name']):
                continue
            
            # Crop 2D bounding box and resize to [224,224]
            box_2d = [detect['xmin'], detect['ymin'], detect['xmax'], detect['ymax']]

            try:
                class_type = detect['name']
                calib_file = os.path.join(calib_dir, img_id + ".txt")
                detected_object = DetectedObject(test_img, class_type, box_2d, calib_file)
                detected_object.set_score(detect['confidence'])
            except:
                print("\nError with detection in %s"%img_id)
                continue
            
            input_tensor = torch.zeros([1, 3, 224, 224]).cuda()
            input_tensor[0,:,:,:] = detected_object.cropped_img

            [orient, conf, dim_3d] = bbox_model(input_tensor)
            orient = orient.cpu().data.numpy()[0, :, :]
            conf = conf.cpu().data.numpy()[0, :]
            dim_3d = dim_3d.cpu().data.numpy()[0, :]
            dim_3d += averages.get_item(class_type)
            detected_object.set_3d_dim(dim_3d)

            # Calculate 3D bounding box orientation
            argmax = np.argmax(conf)
            orient = orient[argmax, :]
            cos = orient[0]
            sin = orient[1]
            alpha = np.arctan2(sin, cos)
            alpha += angle_bins[argmax]
            alpha -= np.pi
            detected_object.set_alpha(alpha)

            # Calculate the center location of the 3D bounding box
            location = calc_location(detected_object, alpha)
            detected_object.set_3d_location(location)

            label_output.append(detected_object.get_kitti_label())
        
        # Export labels to .txt file
        with open(output_path + img_id + ".txt", 'w') as f:
            for line in label_output:
                f.write(f"{line}\n")
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", default="dataset/image_2_val/",
                        help="Relative path to the directory containing images to detect.")
    parser.add_argument("--cal-dir", default="dataset/calib/",
                        help="Relative path to the directory containing camera calibration form KITTI.")
    parser.add_argument("--output-dir", default="inferences/",
                    help="Relative path to the directory for KITTI output labels.")
    parser.add_argument("--yolo-model", default="/home/alvinlee/masters_project/yolov5_3d/yolov5/runs/train/freeze10_kitti/weights/best.pt", 
                    help="Weights of yolo model to be used for 2D detection")
    opts = parser.parse_args()

    main(opts)
