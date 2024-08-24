import ever as er
import torch
from ever.core.checkpoint import remove_module_prefix
from ever.core.config import import_config
from ever.core.builder import make_model, make_dataloader
from tqdm import tqdm

from utils import generate_mask

#---------------------------------------------------------
# Load HRNet
#---------------------------------------------------------
# register
er.registry.register_all()

# load the config
config_path = 'baseline.hrnetw32'
cfg = import_config(config_path)

# load the model
statedict = torch.load('./log/hrnetw32.pth', map_location=lambda storage, loc: storage)
model_state_dict = remove_module_prefix(statedict)
model = make_model(cfg['model'])
model.load_state_dict(model_state_dict)
model.cuda()
model.eval()

#---------------------------------------------------------
# Training images
#---------------------------------------------------------
# we use test because test dataloarder do not have augmentation
cfg['data']['test']['params']['image_dir'] = ['./LoveDA/train/image/'] # the image directory for inference
cfg['data']['test']['params']['mask_dir'] = ['./LoveDA/train/image/'] # we dont have mask, so just put the image directory
cfg['data']['test']['params']['batch_size'] = 1

# make the data loader
dataloader = make_dataloader(cfg['data']['test'])

# source dirs
image_dir = "LoveDA/train/image"
label_dir = "LoveDA/train/label"

# generate mask
generate_mask(image_dir, label_dir, dataloader, mode='train', edge=False, plot=False, save_dir='diffusion_data')


#---------------------------------------------------------
# Validation images
#---------------------------------------------------------
# we use test because test dataloarder do not have augmentation
cfg['data']['test']['params']['image_dir'] = ['./LoveDA/val/image/'] # the image directory for inference
cfg['data']['test']['params']['mask_dir'] = ['./LoveDA/val/image/'] # we dont have mask, so just put the image directory
cfg['data']['test']['params']['batch_size'] = 1

# make the data loader
dataloader = make_dataloader(cfg['data']['test'])

# source dirs
image_dir = "LoveDA/val/image"
label_dir = "LoveDA/val/label"

# generate mask
generate_mask(image_dir, label_dir, dataloader, mode='val', edge=False, plot=False, save_dir='diffusion_data')
