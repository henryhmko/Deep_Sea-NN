from dataloader import *
from TRAINING_CONFIG import *
import torchvision


def eval(test_model):
    
  test_dataloader = data_prep(testing=True)

  test_model.eval()

  if not os.path.exists(output_images_path):
    os.mkdir(output_images_path)

  for i, (img, _, name) in enumerate(test_dataloader):
    with torch.no_grad():
      if img.size()[1] == 4:    #if alpha channel exists in test images, remove alpha channel
        img = img[:, :3, :, :]
      img = img.to(device)
      generate_img = test_model(img)
      torchvision.utils.save_image(generate_img, output_images_path + name[0])

  print("Evaluation of Given Test Images Completed!")

def run_testing():
  model_test = torch.load(test_model_path).to(device)
  eval(model_test)


### START TESTING ###
#run_testing()
