"""
Mask R-CNN
Train on the bread dataset.
------------------------------------------------------------
Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:
    # Train a new model starting from pre-trained COCO weights
    python3 bread.py train --dataset=/path/to/bread/dataset --weights=coco
    # Resume training a model that you had trained earlier
    python3 bread.py train --dataset=/path/to/bread/dataset --weights=last
    # Train a new model starting from ImageNet weights
    python3 bread.py train --dataset=/path/to/bread/dataset --weights=imagenet
"""

import os
import sys
import json
import datetime
import numpy as np
import skimage.draw

# Import Mask RCNN
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights file
ROOT_DIR = os.getcwd()
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

LABEL_CLASS = {
  'red-bean-bread': 1,
  'read-bean-bread': 1,
  'soblo': 2,
  'cream-bread': 3,
  'pizza-bread': 4,
  'tart': 5,
  'walnut-pie': 6,
  'walnut-bread': 6
}

############################################################
#  Configurations
############################################################


class BreadConfig(Config):
    """Configuration for training on the bread dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "bread"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2
    GPU_COUNT = 1

    # Number of training and validation steps per epoch
    STEPS_PER_EPOCH = 500
    VALIDATION_STEPS = 40

    # Backbone network architecture
    # Supported values are: resnet50, resnet101
    BACKBONE = "resnet101"

    # Number of classes (including background)
    NUM_CLASSES = 1 + 6  # Background + breades

    # Length of square anchor side in pixels
    RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)

    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.8

    # How many anchors per image to use for RPN training
    RPN_TRAIN_ANCHORS_PER_IMAGE = 128

    # ROIs kept after non-maximum supression (training and inference)
    POST_NMS_ROIS_TRAINING = 2000
    POST_NMS_ROIS_INFERENCE = 1000

    # If enabled, resizes instance masks to a smaller size to reduce
    # memory load. Recommended when using high-resolution images.
    USE_MINI_MASK = True
    MINI_MASK_SHAPE = (56, 56)  # (height, width) of the mini-mask

    # Input image resizing
    # Random crops of size 512x512
    IMAGE_RESIZE_MODE = "square"
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 1024
    # IMAGE_MIN_SCALE = 2.0

    # Image mean (RGB)
    # MEAN_PIXEL = np.array([43.53, 39.56, 48.22])

    # Number of ROIs per image to feed to classifier/mask heads
    # The Mask RCNN paper uses 512 but often the RPN doesn't generate
    # enough positive proposals to fill this and keep a positive:negative
    # ratio of 1:3. You can increase the number of proposals by adjusting
    # the RPN NMS threshold.
    TRAIN_ROIS_PER_IMAGE = 128

    # Maximum number of ground truth instances to use in one image
    MAX_GT_INSTANCES = 100

    # Max number of final detections per image
    DETECTION_MAX_INSTANCES = 200

    # Skip detections with < 50% confidence
    DETECTION_MIN_CONFIDENCE = 0.3
    
    def add_args(self, **kwargs):
        self.__dict__.update(kwargs)


class BreadInferenceConfig(BreadConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    # Don't resize imager for inferencing
    # IMAGE_RESIZE_MODE = 'pad64'
    RPN_NMS_THRESHOLD = 0.8
    DETECTION_MIN_CONFIDENCE = 0.7


############################################################
#  Dataset
############################################################

class BreadDataset(utils.Dataset):

    def load_bread(self, dataset_dir, subset):
        """Load a subset of the bread dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        
        # Add classes. We have only one class to add.
        self.add_class("bread", 1, "red-bean-bread")
        self.add_class("bread", 2, "soblo")
        self.add_class("bread", 3, "cream-bread")
        self.add_class("bread", 4, "pizza-bread")
        self.add_class("bread", 5, "tart")
        self.add_class("bread", 6, "walnut-pie")
        
        # Train or validation dataset?
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)

        # get all images that end with '.jpg'
        image_list = [x for x in os.listdir(dataset_dir) if x.endswith('.jpg')]

        # Load annotations and Add images
        # LabelImg Annotator saves each image in the form:
        # { 'flags': {},
        #   'shapes': [
        #     {
        #       'label': 'string',
        #       'points': 
        #       [
        #         [
        #           y0, x0
        #         ],
        #         [
        #           y1, x1
        #         ],
        #         ...
        #       ]
        #     },
        #     ... more regions ...
        #   ],
        #   'imagePath': '/path/to/img'
        # }
        # (left top is (0, 0))
        for image in image_list:
            image_name = image.split('.jpg')[0]

            # get image size and annotation
            image_path = os.path.join(dataset_dir, image)
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]
            annotation = json.load(open(os.path.join(dataset_dir, image_name + '.json')))

            self.add_image(
              'bread',
              image_id=image_name,
              path=image_path,
              width=width, height=height,
              shapes=annotation['shapes']
            )

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a bread dataset image, delegate to parent class.
            # image_info = self.image_info[image_id]
            # if image_info["source"] != "bread":
            #     return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["shapes"])],
                        dtype=np.uint8)
        class_ids = []
        for i, a in enumerate(info["shapes"]):
            # Get indexes of pixels inside the polygon and set them to 1
            poly = np.array(a['points'])
            rr, cc = skimage.draw.polygon(poly[:, 1], poly[:, 0])
            mask[rr, cc, i] = 1
            if a['label'] in LABEL_CLASS:
                class_ids.append(LABEL_CLASS[a['label']])
            else:
                print('skip:', image_id, a['label'])

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return mask.astype(np.bool), class_ids

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "bread":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


def train(model):
    from imgaug import augmenters as iaa

    """Train the model."""
    # Training dataset.
    dataset_train = BreadDataset()
    dataset_train.load_bread(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = BreadDataset()
    dataset_val.load_bread(args.dataset, "val")
    dataset_val.prepare()

    # Augment
    augmentation = iaa.Sequential([
        iaa.Fliplr(0.5),
        iaa.OneOf([iaa.Affine(rotate=90),
                   iaa.Affine(rotate=180),
                   iaa.Affine(rotate=270),
                   iaa.Affine(rotate=45),
                   iaa.Affine(rotate=135)]),
        iaa.Multiply((0.8, 1.5)),
        iaa.Add((-10, 10), per_channel=0.5),
        iaa.ContrastNormalization((0.5, 2.0))
    ])

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=20,
                layers='heads',
                augmentation=augmentation)

    print("Fine tune Resnet stage 4 and up")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=40,
                layers='4+',
                augmentation=augmentation)


def color_splash(image, mask):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]
    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # Copy color pixels from the original color image where mask is set
    if mask.shape[-1] > 0:
        # We're treating all instances as one, so collapse the mask into one layer
        mask = (np.sum(mask, -1, keepdims=True) >= 1)
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray.astype(np.uint8)
    return splash


def detect_and_color_splash(model, image_path=None, video_path=None):
    assert image_path or video_path

    # Image or video?
    if image_path:
        # Run model detection and generate the color splash effect
        print("Running on {}".format(args.image))
        # Read image
        image = skimage.io.imread(args.image)
        # Detect objects
        r = model.detect([image], verbose=1)[0]
        # Color splash
        splash = color_splash(image, r['masks'])
        # Save output
        file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
        skimage.io.imsave(file_name, splash)
    elif video_path:
        import cv2
        # Video capture
        vcapture = cv2.VideoCapture(video_path)
        width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vcapture.get(cv2.CAP_PROP_FPS)

        # Define codec and create video writer
        file_name = "splash_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())
        vwriter = cv2.VideoWriter(file_name,
                                  cv2.VideoWriter_fourcc(*'MJPG'),
                                  fps, (width, height))

        count = 0
        success = True
        while success:
            print("frame: ", count)
            # Read next image
            success, image = vcapture.read()
            if success:
                # OpenCV returns images as BGR, convert to RGB
                image = image[..., ::-1]
                # Detect objects
                r = model.detect([image], verbose=0)[0]
                # Color splash
                splash = color_splash(image, r['masks'])
                # RGB -> BGR to save image to video
                splash = splash[..., ::-1]
                # Add image to video writer
                vwriter.write(splash)
                count += 1
        vwriter.release()
    print("Saved to ", file_name)


############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect balloons.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'splash'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/bread/dataset/",
                        help='Directory of the bread dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')
    parser.add_argument('--video', required=False,
                        metavar="path or URL to video",
                        help='Video to apply the color splash effect on')
    # parser.add_argument('--pairs', required=False,
    #                     metavar="key=value",
    #                     help="Key values to apply config",
    #                     action='append',
    #                     type=lambda kv: kv.split("="))
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "splash":
        assert args.image or args.video,\
               "Provide --image or --video to apply color splash"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)
    # print("Keypairs: ", args.pairs)

    # Configurations
    if args.command == "train":
        bread_config = BreadConfig()
        # bread_config.add_args(**dict(args.pairs))
        config = BreadConfig()
    else:
        config = BreadInferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])

    # Train or evaluate
    if args.command == "train":
        train(model)
    elif args.command == "splash":
        detect_and_color_splash(model, image_path=args.image,
                                video_path=args.video)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'splash'".format(args.command))

