"""standard libraries"""
import torch
import pdb
import os
"""third party imports"""
pdb.set_trace()

"""local specific imports"""
from train.utils import reconstruction

model_path = os.path.join(
    os.path.dirname(os.getcwd()),
    trained_models,
    'dulcet-bush-116.pth'
)
parser = argparse.ArgumentParser(
    description = "Radiance Fields"
)
parser.add_argument(
    "--conf",
    default=None,
    help="path of configuration file"
)
args = parser.parse_args()
params = yaml.full_load(open(args.conf, 'r')).get('dict')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

train_dataset, valid_dataset, test_dataset = data_splitter(
    os.path.join(os.getcwd(),params['data']['conf']),
    params['data']['desgn'],
    device
)
# train_dataset[1087]
#train_dataloader = DataLoader(
#    train_dataset, 
#    batch_size=params['batch_size']['train'], 
#    shuffle=True,
#    drop_last=True
#)
## next(iter(train_dataloader))
#valid_dataloader = DataLoader(
#    valid_dataset, 
#    batch_size=params['batch_size']['valid'], 
#    shuffle=True,
#    drop_last=True
#)

test_dataloader = DataLoader(
    test_dataset, 
    batch_size=params['batch_size']['test'], 
    shuffle=True,
    drop_last=True
)

models = torch.load(model_path)
pdb.set_trace()
media = reconstruction(
    models,
    test_dataloader,
    params['inv_tr_conf'],
    params['net_conf']['model_info'],
    device
)
pdb.set_trace() 



