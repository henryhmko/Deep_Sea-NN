import torch
from PIL import Image
from torch.utils.data import Dataset
import os
from torchvision import transforms
from torch.utils.data import DataLoader
from TRAINING_CONFIG import * 

"Code from [Shallow-UWnet/dataloader.py] from 'mkartik' github with smol changes"


def get_image_list(raw_image_path, clear_image_path, is_train):
  image_list = []
  raw_image_list = [raw_image_path + i for i in os.listdir(raw_image_path)]
  if is_train:
    for raw_image in raw_image_list:
      image_file = raw_image.split('/')[-1]
      image_list.append([raw_image, os.path.join(clear_image_path + image_file), image_file])
  else:
    for raw_image in raw_image_list:
      image_file = raw_image.split('/')[-1]
      image_list.append([raw_image, None, image_file])
  return image_list


class UWDataset(Dataset):
  def __init__(self, raw_image_path, clear_image_path, transform, is_train=False):
    self.raw_image_path = raw_image_path
    self.clear_image_path = clear_image_path
    self.is_train = is_train
    self.transform = transform
    self.image_list = get_image_list(self.raw_image_path, self.clear_image_path, is_train)
  
  def __len__(self):
    return len(self.image_list)

  def __getitem__(self, index):
    raw_image, clear_image, image_name = self.image_list[index]
    raw_image = Image.open(raw_image)
    if self.is_train:
      clear_image = Image.open(clear_image)
      return self.transform(raw_image), self.transform(clear_image), "_"
    return self.transform(raw_image), "_", image_name



def data_prep(testing=False):

  if testing:
    transform = transforms.Compose([transforms.Resize((test_img_size,test_img_size)), transforms.ToTensor()])
    test_dataset = UWDataset(test_image_path, None, transform, False)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    print("Test Dataset Reading Complete")
    return test_dataloader

  transform = transforms.Compose([transforms.Resize((train_img_size,train_img_size)), transforms.ToTensor()]) #Resize to (256,256) and then ToTensor
  train_dataset = UWDataset(raw_image_path, clear_image_path, transform, True)
  train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
  print("Train Dataset Reading Complete")

  return train_dataloader


