# %%
'''
Make T=10 updates and save internal statistics such as effective updates Delta W^l x^{l-1} for each layer l for in-depth analysis of the effect of width scaling on the training dynamics.
'''

import time
from absl import logging
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.nn import functional as F

import utils
from utils.mlp import MLP, Scale_Corrected_MLP
from mlp_experiments.trainer_mlp import MLPTrainer
from utils.generate_data import GeneratedDataset
#from utils import get_filename_from_args, mysave, myload, find


def identity(x):
    return x

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data, targets, transform=None):
        """
        Custom dataset that accepts image data and labels (targets).
        
        Args:
            data (Tensor): The image data (shape: [N, C, H, W]).
            targets (Tensor): The labels (shape: [N,]).
            transform (callable, optional): A function/transform to apply to the images.
        """
        self.data = data#.float() if isinstance(data,torch.Tensor) else torch.tensor(targets, dtype=torch.float32)
        self.targets = targets#.long() if isinstance(targets,torch.Tensor) else torch.tensor(targets, dtype=torch.long)
        self.transform = transform  # Optional transform (such as ToTensor())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        if self.transform:
            image = self.transform(image)
        return image, self.targets[idx]

class OneHotCIFAR10(torch.utils.data.Dataset):
    def __init__(self, root='../data', train=True, transform=None, download=True, num_classes=10):
        self.cifar10 = torchvision.datasets.CIFAR10(
            root=root, train=train, transform=transform, download=download
        )
        self.num_classes = num_classes
    
    def __getitem__(self, index):
        img, label = self.cifar10[index]
        
        # Convert label to one-hot encoding
        one_hot = torch.zeros(self.num_classes)
        one_hot[label] = 1.0
        
        return img, one_hot
    
    def __len__(self):
        return len(self.cifar10)
    
dataset = 'cifar10' # 'cifar10_mse', 'cifar10_mse_softmax', 'teacher' (from chizat and kunin papers), 'teacher_ce' (classification task), 'teacher_softmax'
T = 10 # number of update steps
param = 'sp' # 'mup' 'mup_spllit' 'mup_largelr' 'mup_spllit_largelr  # 'llm'  # 'sp_largeinput'
rmsnormlayer = False
llzeroinit = False # True
optim = 'SGD' # 'SGD' 'ADAM'
if optim == 'SGD':
    base_lr = 0.03
else:
    base_lr = 0.0003
#lr_expon = 0.0 # 0 for shallow SGD or ADAM # -0.5 for deep SGD, -1.0 for ADAM
nepochs = 1
fine_widths = True
extendedevaliter = 1
evaliter = 1
bs = 64
small_cifar_size = n_samples = T * bs
N_RUNS = 4
seed = 42
#nhiddenlayers = 1
linear = False  
resnet = False  

if optim == 'SGD':
    experiments = [('SGD', 0.0, 2), ('SGD',-0.5,2), ('SGD',0.0,1), ('SGD',-0.5,1), ('LL-SGD',0.0,1)]
else:
    experiments = [('ADAM',-1.0,2),('ADAM',-0.5,2), ('ADAM',0.0,2), ('ADAM',0.0,1), ('LL-ADAM',0.0,1)]

# width = 'all'
rho = 0
perturb = 'naive'
scheduler = 'None'
nomultipliers = True  
use_bias = False  
finaleval = False  
large_bias_init = False  
indepgradnorms = False  
gn0 = []  # Empty list for the layers
initvarnorm = False  
weightdecay = 0  # Default to 0
momentum = 0  # Default to 0
numfeaturerankevals = 0
spectral_mup = False 
small_input_mup = False
multiplier_mup = False
yangmult = False  
label_noise = 0
savebestmodel = False  
prefix = 'coord_check'
n_samples_test = 1000

if 'mup' in param:
    experiments = [('SGD', 0.0, 2), ('SGD',0.0,1), ('LL-SGD',0.0,1), ('ADAM',0.0,2), ('ADAM',0.0,1), ('LL-ADAM',0.0,1)]


# Set the GPU device
if torch.cuda.is_available():
    torch.set_default_device('cuda')
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device('cpu')
    #raise ValueError('Cuda is not available!')

start_time = time.time()

# using passed hyperparameters
width = 'all'
if fine_widths:
    WIDTHS = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
else:
    WIDTHS = [256, 1024, 4096, 16384]

#LR = base_lr
RHO = rho
PARAMETERIZATION = param #'mup' | 'sp' | 'ntp'
PERTURBATION = perturb # 'mpp' | 'global' | 'gb' | 'naive'
BATCH_SIZE = bs

for (optim, lr_expon, nhiddenlayers) in experiments:
    OPTIM_ALGO = optim
    seed = seed

    EXP_NAME = f'{prefix}_{param}_{dataset}_{OPTIM_ALGO}_lr{base_lr}_lrexp{lr_expon}_allwidths_nlay{nhiddenlayers}{"_lin" if linear else ""}{"_llit" if llzeroinit else ""}{"_rmsnorm" if rmsnormlayer else ""}_seed{seed}_nruns{N_RUNS}' 
    all_stats = {}
    SUBFOLDER = 'cifar10/' if 'cifar10' in dataset else 'teacher/'
    corresponding_statfiles = utils.find(f'stats_mlp' + EXP_NAME,'./stats/'+SUBFOLDER)
    # if len(corresponding_statfiles)>0:
    #     all_stats = myload(corresponding_statfiles[0])
    #     print(f'Stats already computed: {EXP_NAME}')
    # else:
    for irun in range(N_RUNS):
        for WIDTH in WIDTHS:
            width = WIDTH
            LR = base_lr * (WIDTH/256)**lr_expon
            utils.seed_everything(seed+irun)
            
            if OPTIM_ALGO == 'SGD' and prefix == 'onlybest': RHO = 0
            if OPTIM_ALGO == 'SGD' and RHO>0:
                print('No need for running SGD with RHO>0.')
            
            
            # Set hyperparameters
            used_dataset = dataset # 'cifar10'

            N_HIDDEN_LAYERS = nhiddenlayers
            N_EPOCHS = nepochs
            
            logging.set_verbosity(logging.INFO)
            
            if 'cifar10' in used_dataset:
                transform = transforms.Compose(
                    [transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

                if 'mse' in used_dataset:
                    train_set = OneHotCIFAR10(root='../data', train=True, download=True, transform=transform)
                    test_set = OneHotCIFAR10(root='../data', train=False, download=True, transform=transform)
                else:
                    if small_cifar_size > 0:
                        train_set = torchvision.datasets.CIFAR10(root='../data', train=True,download=True, transform=None)

                        class_0_indices = [i for i, label in enumerate(train_set.targets) if label == 0]
                        class_1_indices = [i for i, label in enumerate(train_set.targets) if label == 1]

                        # Combine the indices of class 0 and class 1
                        selected_indices = np.array(class_0_indices[:int(small_cifar_size/2)] + class_1_indices[:int(small_cifar_size/2)])

                        # Create a subset of the dataset using these indices
                        #train_set = Subset(train_set, selected_indices)# Subset does not allow to replace labels, it just points to original dataset!!! -> nasty bugs
                        data_subset = train_set.data[selected_indices]  # Convert indices to numpy array
                        targets_subset = torch.tensor(train_set.targets)[selected_indices]
                        train_set = CustomDataset(data_subset, targets_subset, transform=transform)
                    else:
                        train_set = torchvision.datasets.CIFAR10(root='../data', train=True,download=True, transform=transform)

                if small_cifar_size > 0:
                    test_set = torchvision.datasets.CIFAR10(root='../data', train=False,download=True,transform=None)

                    class_0_indices = [i for i, label in enumerate(test_set.targets) if label == 0]
                    class_1_indices = [i for i, label in enumerate(test_set.targets) if label == 1]
                    selected_indices = np.array(class_0_indices + class_1_indices)
                    #test_set = Subset(test_set, selected_indices)
                    data_subset = test_set.data[selected_indices]  # Convert indices to numpy array
                    targets_subset = torch.tensor(test_set.targets)[selected_indices]
                    test_set = CustomDataset(data_subset, targets_subset,transform=transform)
                else:
                    test_set = torchvision.datasets.CIFAR10(root='../data', train=False,download=True, transform=transform)
                if label_noise > 0:
                    if 'mse' in used_dataset:
                        raise ValueError('Label noise not implemented for MSE.')
                    num_classes = 10
                    class_indices = {i: [] for i in range(num_classes)}

                    # Organize the indices by their labels (class)
                    for idx, label in enumerate(train_set.targets):
                        class_indices[label].append(idx)

                    # Randomly change label_noise of labels in each class
                    for label in range(num_classes):
                        num_class_samples = len(class_indices[label])
                        noisy_indices = np.random.choice(class_indices[label], int(label_noise * num_class_samples), replace=False)
                        
                        for idx in noisy_indices:
                            new_label = np.random.randint(0, num_classes)
                            train_set.targets[idx] = new_label


                train_sampler = torch.utils.data.RandomSampler(train_set,generator=torch.Generator(device=DEVICE))
                train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE,sampler=train_sampler)

                test_sampler = torch.utils.data.RandomSampler(test_set,generator=torch.Generator(device=DEVICE))
                test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE, sampler=test_sampler)

                DIM_X = 32 * 32 * 3
                DIM_Y = 10
                SUBFOLDER = 'cifar10/'
                loss_fn = nn.CrossEntropyLoss() if 'mse' not in used_dataset else nn.MSELoss()
            elif 'teacher' in used_dataset:
                # inspired by 'get rich quick'-paper https://github.com/allanraventos/getrichquick/blob/main/two-layer-relu/plot_fig1a.ipynb
                # but using sgd, not gd, and ensuring balanced classes
                
                DIM_X = 100 #2
                k = 4
                min_spread = 0.5 # all target vectors at least 0.5*pi radians apart (in dim=2, min_spread=0.5 only works for k<=3...)
                from utils.kunin_utils import TeacherNetwork
                teacher = TeacherNetwork(DIM_X, k, min_spread=min_spread,classification=True,unit_teachers = True) if 'teacher_ce' in used_dataset else TeacherNetwork(DIM_X, k, min_spread=min_spread,unit_teachers = True)
                inputs = torch.randn(n_samples, DIM_X).float()
                inputs = inputs / torch.norm(inputs, dim=1, keepdim=True)
                inputs_test = torch.randn(n_samples_test, DIM_X).float()
                inputs_test = inputs_test / torch.norm(inputs_test, dim=1, keepdim=True)
                with torch.no_grad():
                    if 'teacher_ce' in used_dataset:
                        flip_indices = np.random.choice(n_samples, size=int(n_samples*label_noise), replace=False)
                        labels = teacher(inputs)
                        labels[flip_indices] = 1-labels[flip_indices]
                        flip_indices_test = np.random.choice(n_samples_test, size=int(n_samples_test*label_noise), replace=False)
                        labels_test = teacher(inputs_test)
                        labels_test[flip_indices_test] = 1-labels_test[flip_indices_test]
                        labels = labels.flatten().long()
                        labels_test = labels_test.flatten().long()
                    else:
                        labels = teacher(inputs) + label_noise * torch.randn((n_samples,1))
                        labels_test = teacher(inputs_test) + label_noise * torch.randn((n_samples_test,1))
                    
                print(f"input mean: {inputs.mean()}, label mean: {labels.float().mean()}")

                train_set = GeneratedDataset(data=inputs, labels=labels, transform=None, device=DEVICE,classification=('teacher_ce' in used_dataset))
                train_sampler = torch.utils.data.RandomSampler(train_set, generator=torch.Generator(device=DEVICE))
                train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, sampler=train_sampler)  

                test_set = GeneratedDataset(data=inputs_test, labels=labels_test, transform=None, device=DEVICE,classification=('teacher_ce' in used_dataset))
                test_sampler = torch.utils.data.RandomSampler(test_set, generator=torch.Generator(device=DEVICE))
                test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE, sampler=test_sampler)            
                
                loss_fn = nn.CrossEntropyLoss() if 'teacher_ce' in used_dataset else nn.MSELoss()
                DIM_Y = 2 if 'teacher_ce' in used_dataset else 1
                SUBFOLDER = 'teacher/'
            else:
                raise ValueError('Implement data loading for this dataset.')

                
            if numfeaturerankevals == 0:
                feature_rank_iter = None
            else:
                if evaliter == 0:
                    feature_rank_iter = np.linspace(0, N_EPOCHS-1, numfeaturerankevals,dtype=int)
                else:
                    feature_rank_iter = np.linspace(0, N_EPOCHS * len(train_dataloader) - 1, numfeaturerankevals, dtype=int)
                
            logging.info(f"Width={WIDTH}, LR={LR}, RHO={RHO}")

            if not nomultipliers:
                if yangmult:
                    multipl_inout = [0.00390625, 32]
                else:
                    raise NotImplementedError('First tune multipliers for this parameterization.')
            else:
                multipl_inout = None
            

            if not rmsnormlayer:
                model = MLP(in_size=DIM_X,
                            out_size=DIM_Y,
                            hidden_size=WIDTH,
                            n_hidden_layers=N_HIDDEN_LAYERS,
                            parameterization=PARAMETERIZATION,
                            perturbation=PERTURBATION,
                            activation = identity if linear else F.relu,
                            out_activation=nn.Softmax(dim=1) if 'softmax' in used_dataset else None,
                            res_connections = resnet,
                            flat_indim = DIM_X,
                            multipl_inout=multipl_inout,
                            ll_zero_init=llzeroinit,
                            use_bias=use_bias,
                            large_bias_init = large_bias_init,
                            multiplier_mup = multiplier_mup,
                            initvarnorm=initvarnorm)
            else:
                model = Scale_Corrected_MLP(in_size=DIM_X,
                            out_size=DIM_Y,
                            hidden_size=WIDTH,
                            n_hidden_layers=N_HIDDEN_LAYERS,
                            parameterization=PARAMETERIZATION,
                            perturbation=PERTURBATION,
                            activation = identity if linear else F.relu,
                            out_activation=nn.Softmax(dim=1) if 'softmax' in used_dataset else None,
                            res_connections = resnet,
                            flat_indim = DIM_X,
                            multipl_inout=multipl_inout,
                            ll_zero_init=llzeroinit,
                            use_bias=use_bias,
                            large_bias_init = large_bias_init,
                            multiplier_mup = multiplier_mup,
                            norm_layer='rms',
                            initvarnorm=initvarnorm)
            model.to(DEVICE)
            
            
            # Construct trainer
            if savebestmodel:
                savebest=EXP_NAME
            else:
                savebest=None
            if scheduler == 'None':
                scheduler = None
                
            mlp_trainer = MLPTrainer(model=model,
                                train_dataloader=train_dataloader,
                                eval_dataloader=test_dataloader,
                                lr=LR,
                                rho=RHO,
                                optim_algo=OPTIM_ALGO,
                                weight_decay=weightdecay,
                                momentum=momentum,
                                scheduler=scheduler,
                                exp_name=EXP_NAME,
                                device= DEVICE,
                                loss_fn=loss_fn,
                                classification=False if used_dataset == 'teacher' else True,
                                width_indep_gradnorms=indepgradnorms,
                                gn0=gn0,
                                extended_eval_iter = extendedevaliter,
                                eval_iter = evaliter,
                                feature_rank_iter=feature_rank_iter,
                                final_eval=finaleval,
                                save_best_file = savebest,
                                spectral_mup=spectral_mup,
                                small_input_mup=small_input_mup,
                                only_track_norms=True,
                                tag_dict={'W':WIDTH, 'LR':float(LR), 'RHO':float(RHO)})
            mlp_trainer.train(N_EPOCHS, resample_data=False)
            all_stats[(WIDTH, LR,seed+irun)] = mlp_trainer.history

            print(f'width {WIDTH} took time {time.time()-start_time}')
    timestamp = time.strftime('_%Y%m%d-%H%M%S')
    args = {'dataset':dataset, 'nhiddenlayers':nhiddenlayers,'optim':OPTIM_ALGO,'param':param, 'lr':base_lr, 'lr_expon': lr_expon, 'widths':WIDTHS, 'linear':linear, 'seed':seed, 'nruns': N_RUNS, 'T':T,'nepochs':N_EPOCHS, 'bs':bs,'linear':linear,'resnet':resnet,'llzeroinit':llzeroinit, 'nomultipliers':nomultipliers}
    utils.mysave('./stats/'+SUBFOLDER,f'config_mlp_nomult' + EXP_NAME + timestamp+'.txt', args)
    utils.mysave('./stats/'+SUBFOLDER,f'stats_mlp_nomult' + EXP_NAME + timestamp+'.txt', all_stats)


# %%







