import glob
import os
import random
import torch

from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from metric import sake_metric


def get_transform(split='train'):
    if split == 'train':
        return transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()])
    else:
        return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()])

def get_transform_SHREC(split='train'):
    if split == 'train':
        return transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    else:
        return transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])


class DomainDataset(Dataset):
    def __init__(self, data_root, data_name, split='train',flag = "target"):
        super(DomainDataset, self).__init__()

        images = []
        self.flag = flag

        if data_name == 'IM2MN':
            self.query_folder = 'images'
            self.target_folder = 'views'
            self.query_pic_ext = '*.JPEG' 
            self.target_pic_ext = '*.png' 
        if data_name == 'MI3DOR':
            self.query_folder = 'image'
            self.target_folder = 'view'
            self.query_pic_ext = '*.JPEG' 
            self.target_pic_ext = '*.png' 
        if data_name == 'SHREC13_ZS_DDGAN' or data_name == 'SHREC14_ZS_DDGAN':
            self.query_folder = 'sketch'
            self.target_folder = 'shape'
            self.query_pic_ext = '*.png' 
            self.target_pic_ext = '*.jpg' 
        if data_name == 'SHREC13_ZS_DTO' or data_name == 'SHREC14_ZS_DTO':
            self.query_folder = 'sketch'
            self.target_folder = 'shape'
            self.query_pic_ext = '*.png' 
            self.target_pic_ext = '*.jpg' 
        if data_name == 'ImageModel'or data_name == 'ImageModel2':
            self.query_folder = 'sketch'
            self.target_folder = 'shape'
            self.query_pic_ext = '*.JPEG' 
            self.target_pic_ext = '*.jpg' 
        if data_name == 'GZS_STO':
            self.query_folder = 'sketch'
            self.target_folder = 'shape'
            self.query_pic_ext = '*.JPEG' 
            self.target_pic_ext = '*.jpg' 


        images = []
        for classes in os.listdir(os.path.join(data_root, data_name, self.query_folder, split)):
            query_data = glob.glob(os.path.join(data_root, data_name, self.query_folder, split, str(classes), self.query_pic_ext))
            target_data = glob.glob(os.path.join(data_root, data_name, self.target_folder,split,  str(classes), '*'))
            images += query_data

        self.images = sorted(images)
        self.transform = get_transform(split)
            

        self.domains, self.labels, self.classes = [], [], {}
        i = 0
        for img in self.images:
            temp = os.path.dirname(img).split('\\')
            domain,_,label = os.path.dirname(img).split('\\')[-3:]
            self.domains.append(0 if domain == self.target_folder else 1)
            if label not in self.classes:
                self.classes[label] = i
                i += 1
            self.labels.append(self.classes[label])
        # store photos for each class to easy sample for sketch in training period
        if split == 'train':
            self.refs = {}
            for key, value in self.classes.items():
                self.refs[value] = sorted(glob.glob(os.path.join(data_root, data_name,self.target_folder,split, key, '*')))
        # if (split == 'test' or split == 'val') and flag == 'target':
        if (split == 'test' or split == 'val' or split == 'test_SBSR' or split == 'test_ZS_SBSR' or split == 'test_GZS_SBSR') and flag == 'target':
        # if split == 'val' and flag == 'target':
            images = []
            self.images = []
            for classes in os.listdir(os.path.join(data_root, data_name, self.target_folder, split)):
                target_data = glob.glob(os.path.join(data_root, data_name, self.target_folder,split,  str(classes), '*'))
                images += target_data
            self.images = sorted(images)
            self.domains, self.labels, self.classes = [], [], {}
            i = 0
            for img in self.images:
                temp = os.path.dirname(img).split('\\')
                domain,_,label = os.path.dirname(img).split('\\')[-3:]
                self.domains.append(0 if domain == self.target_folder else 1)
                if label not in self.classes:
                    self.classes[label] = i
                    i += 1
                self.labels.append(self.classes[label])

        self.split = split

    def __getitem__(self, index):
        path = self.images[index]
        label = self.labels[index]
        query_path = []
        target_path = []
        # if (self.split == 'test' or self.split == 'val') and self.flag == 'query':
        if (self.split == 'test' or self.split == 'val' or self.split == 'test_SBSR' or self.split == 'test_ZS_SBSR' or self.split == 'test_GZS_SBSR') and self.flag == 'query':
        # if self.split == 'val' and self.flag == 'query':
            # if domain == 1:
                img = Image.open(path)
                img = img.convert('RGB')
                img = self.transform(img)
                query_path.append(path)

                return img, label#, query_path
        elif (self.split == 'test' or self.split == 'val' or self.split == 'test_SBSR' or self.split == 'test_ZS_SBSR' or self.split == 'test_GZS_SBSR') and self.flag == 'target':
        # elif self.split == 'val' and self.flag == 'target':
            # if domain == 0:
                img = []
                for view_path in glob.glob(path+'/view12/'+ self.target_pic_ext):
                # for view_path in glob.glob(os.path.join(path, self.target_pic_ext)):
                    im = Image.open(view_path)
                    im = im.convert('RGB')
                    im = self.transform(im)
                    img.append(im)
                img = torch.stack(img)
                target_path.append(path)


                return img, label#, target_path
        elif self.split == 'train':
            path = self.images[index]
            label = self.labels[index]
            img = Image.open(path)
            img = img.convert('RGB')
            img = self.transform(img)
            ref_path = random.choice(self.refs[label])
            ref = []
            for view_path  in glob.glob(ref_path+'/view12/'+ self.target_pic_ext):
            # for view_path in glob.glob(os.path.join(ref_path, self.target_pic_ext)):
                im = Image.open(view_path)
                im = im.convert('RGB')
                im = self.transform(im)
                ref.append(im)
            ref = torch.stack(ref)

            return img, ref, label,ref_path

    def __len__(self):
        return len(self.images)


def compute_metric(vectors, domains, labels):
    acc = {}

    photo_vectors = vectors[domains == 0].numpy()
    sketch_vectors = vectors[domains == 1].numpy()
    photo_labels = labels[domains == 0].numpy()
    sketch_labels = labels[domains == 1].numpy()
    map_all, p_100 = sake_metric(photo_vectors, photo_labels, sketch_vectors, sketch_labels)
    map_200, p_200 = sake_metric(photo_vectors, photo_labels, sketch_vectors, sketch_labels,
                                 {'precision': 200, 'map': 200})

    acc['P@100'], acc['P@200'], acc['mAP@200'], acc['mAP@all'] = p_100, p_200, map_200, map_all
    # the mean value is chosen as the representative of precise
    acc['precise'] = (acc['P@100'] + acc['P@200'] + acc['mAP@200'] + acc['mAP@all']) / 4
    return acc

