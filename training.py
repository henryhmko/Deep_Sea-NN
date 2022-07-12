from TRAINING_CONFIG import *
from loss import *
from model import *
from dataloader import *
from tqdm.autonotebook import tqdm, trange



# SETUP - MODEL, OPT, LOSS
def setup(resume_train = False, filepath = False):
  # returns model, opt, and loss
  
  print('Device is set to {}'.format(device))

  model = Unet(in_channels=3, out_channels=3, init_features=32).to(device) 

  loss = MS_SSIM_L1_LOSS()
  opt = torch.optim.Adam(model.parameters(), lr=lr)

  if resume_train: #if resuming training...
    if os.path.isfile(filepath):
      print("==> loading checkpoint '{}'".format(filepath))
      checkpoint = torch.load(filepath)
      start_epoch = checkpoint['epoch'] + 1 #training should start from the next epoch
      model.load_state_dict(checkpoint['state_dict'])
      opt.load_state_dict(checkpoint['optimizer'])
      print("==> loaded checkpoint '{}' (epoch {})".format(filepath, start_epoch - 1))
    else:
        print("==> no checkpoint found. Check File Path!")
    return model, opt, loss, start_epoch
  
  return model, opt, loss



def train(train_dataloader, save_model=False, resume_train=False, ckpt_path=False):

  if resume_train:
    model, opt, ssimloss, start_epoch = setup(resume_train=True, filepath = ckpt_path)
    print("RESUMING TRAINING FROM EPOCH {}.".format(start_epoch))
    print("REMAINING EPOCHS TO TRAIN: {} EPOCHS".format(num_epochs - start_epoch))
  
  else:
    model, opt, ssimloss = setup() #Initialize MODEL, OPT, LOSS
    start_epoch = 0

  for epoch in trange(start_epoch, num_epochs, desc=f'[Full Loop]', leave = True):

    total_loss_tmp = 0
    
    for inp, label, _ in tqdm(train_dataloader, desc = f'[Train]', leave = False):  # Erase progress bar once this epoch is done training
      inp = inp.to(device)
      label = label.to(device)

      model.train() #model = UWNet and this is under a nn.Module class, so I can call .train()

      opt.zero_grad()
      out = model(inp)
      loss = ssimloss(out, label)

      loss.backward()
      opt.step()

      total_loss_tmp += loss.item()

    #Printing epoch results BELOW.
    print('epoch: [{}]/[{}], image loss: {}'.format(epoch, (num_epochs-1), str(total_loss_tmp)))
    #Printing epoch results ABOVE

    if not os.path.exists(snapshots_folder): #make dir for storing snapshots
      os.mkdir(snapshots_folder)
    
    #Saving model states [NOTE: saving only epoch number, model parameters and optimizer paramters]
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': opt.state_dict()
    }
    
    
    if epoch % snapshot_freq == 0: #Save checkpoint every [snapshot_freq] epochs
      torch.save(state, snapshots_folder + 'model_epoch_{}_{}.ckpt'.format(epoch, model_name))
    
    if epoch == (num_epochs - 1): #For Last Epoch, just save the entire model
        torch.save(model, snapshots_folder + 'model_epoch_{}_{}_MODEL.ckpt'.format(epoch, model_name))


def run_training():
  train_dataloader = data_prep()
  train(train_dataloader, save_model=True)


### START TRAINING ###
#run_training()
