import os
import xml.etree.ElementTree as ET
import numpy as np

# from data package
from .data_util import read_image

"""Custom dataset을 불러오는 모듈
이미지뿐 아니라 annotation에 xml 파일로 저장되있는
bbox 좌표나 label 정보 또한 함께 불러온다."""

class MyBboxDataset:
    def __init__(self, data_dir, split='trainval',use_difficult=False, return_difficult=False,):

        id_list_file = os.path.join(data_dir, 'ImageSets/Main/{0}.txt'.format(split))

        self.ids = [id_.strip() for id_ in open(id_list_file)]

        self.data_dir = data_dir

        self.use_difficult = use_difficult
        self.return_difficult = return_difficult

        self.label_names = VOC_BBOX_LABEL_NAMES

    def __len__(self):
        return len(self.ids)

    # Returns: tuple of an image and bounding boxes
    def get_example(self, i):
        id_ = self.ids[i]
        anno = ET.parse(os.path.join(self.data_dir, 'Annotations', id_ + '.xml'))
        
        bbox = list()
        label = list()
        difficult = list()
        
        for obj in anno.findall('object'):
            # when in not using difficult split, and the object is
            # difficult, skipt it.
            if not self.use_difficult and int(obj.find('difficult').text) == 1:
                continue

            difficult.append(int(obj.find('difficult').text))
            bndbox_anno = obj.find('bndbox')
            
            # subtract 1 to make pixel indexes 0-based
            bbox.append([
                int(bndbox_anno.find(tag).text) - 1
                for tag in ('ymin', 'xmin', 'ymax', 'xmax')])
            
            name = obj.find('name').text.lower().strip()
            
            label.append(VOC_BBOX_LABEL_NAMES.index(name))
        
        bbox = np.stack(bbox).astype(np.float32)
        label = np.stack(label).astype(np.int32)
        
        # When `use_difficult==False`, all elements in `difficult` are False.
        difficult = np.array(difficult, dtype=np.bool).astype(np.uint8)  # PyTorch don't support np.bool

        # Load a image
        img_file = os.path.join(self.data_dir, 'JPEGImages', id_ + '.jpg')
        img = read_image(img_file, color=True)

        # if self.return_difficult:
        #     return img, bbox, label, difficult
        return img, bbox, label, difficult

    __getitem__ = get_example


VOC_BBOX_LABEL_NAMES = (
    'aeroplane',
    'bicycle',
    'bird',
    'boat',
    'nest',
    'bus',
    'car',
    'cat',
    'chair',
    'cow',
    'diningtable',
    'dog',
    'horse',
    'motorbike',
    'person',
    'pottedplant',
    'sheep',
    'sofa',
    'train',
    'tvmonitor')