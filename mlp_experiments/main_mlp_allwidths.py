"""
Train a MLP of varying width with many optional evaluation metrics using different datasets, parameterizations, losses, optimizers, and architecture hyper parameters.
"""

import argparse
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
from utils.generate_data import GeneratedDataset, OneHotDataset, generate_sparse_gaussian_training_set

def identity(x):
    return x

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
    
    

def main(args):
    # Set the GPU device
    if torch.cuda.is_available():
        torch.set_default_device('cuda')
        DEVICE = torch.device("cuda")
    else:
        DEVICE = torch.device('cpu')
        #raise ValueError('Cuda is not available!')

    start_time = time.time()

    # using passed hyperparameters
    args.width = 'all'
    if args.width_choice=='fine':
        WIDTHS = [256, 512, 1024, 2048, 4096, 8192, 16384]
    elif args.width_choice=='few':
        WIDTHS = [256, 1024, 4096]
    elif args.width_choice == 'small':
        WIDTHS = [64, 256, 1024]
    elif args.width_choice == '4k':
        WIDTHS = [4096]
    elif args.width_choice=='8k':
        WIDTHS = [8192]
    elif args.width_choice=='wide':
        WIDTHS = [16384]
    elif args.width_choice=='standard':
        WIDTHS = [256, 1024, 4096, 16384]
    else:
        raise ValueError('Unknown width choice. Use standard, fine, few or wide.')
    
    RHO = args.rho
    PARAMETERIZATION = args.param #'mup' | 'sp' | 'ntp'
    PERTURBATION = args.perturb # 'mpp' | 'global' | 'gb' | 'naive'
    BATCH_SIZE = args.bs
    OPTIM_ALGO = args.optim
    seed = args.seed
    
    for WIDTH in WIDTHS:
        args.width = WIDTH
        utils.seed_everything(seed)
        LR = args.lr if args.lr_log == 9999 else 2.0**(-args.lr_log)
        if args.lrexp != 0:
            LR=np.round(LR * (WIDTH/256)**args.lrexp,8)
        
        if OPTIM_ALGO == 'SGD' and args.prefix == 'onlybest': RHO = 0
        if OPTIM_ALGO == 'SGD' and RHO>0:
            print('No need for running SGD with RHO>0.')
            return 0
        
        EXP_NAME = utils.get_expname_from_args(args,prefix=args.prefix,join='-')
        
        # Set hyperparameters
        used_dataset = args.dataset # 'cifar10'

        N_HIDDEN_LAYERS = args.nhiddenlayers
        N_EPOCHS = args.nepochs
        
        logging.set_verbosity(logging.INFO)
        
        if 'cifar10' in used_dataset:
            transform = transforms.Compose(
                [transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

            if 'mse' in used_dataset:
                train_set = OneHotCIFAR10(root='../data', train=True, download=True, transform=transform)
                test_set = OneHotCIFAR10(root='../data', train=False, download=True, transform=transform)
            else:
                train_set = torchvision.datasets.CIFAR10(root='../data', train=True,download=True, transform=transform)
                test_set = torchvision.datasets.CIFAR10(root='../data', train=False,download=True, transform=transform)
            if args.label_noise > 0:
                if 'mse' in used_dataset:
                    raise ValueError('Label noise not implemented for MSE.')
                num_classes = 10
                class_indices = {i: [] for i in range(num_classes)}

                # Organize the indices by their labels (class)
                for idx, label in enumerate(train_set.targets):
                    class_indices[label].append(idx)

                # Randomly change args.label_noise of labels in each class
                for label in range(num_classes):
                    num_class_samples = len(class_indices[label])
                    noisy_indices = np.random.choice(class_indices[label], int(args.label_noise * num_class_samples), replace=False)
                    
                    for idx in noisy_indices:
                        #current_label = train_set.targets[idx]
                        # Assign a new random label different from the current one
                        #new_label = current_label
                        #while new_label == current_label:
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
        elif 'mnist' in used_dataset:
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
                ])
            train_set = torchvision.datasets.MNIST('../data', train=True, download=False,transform=transform)
            test_set = torchvision.datasets.MNIST('../data', train=False,download=False,transform=transform)
            if args.label_noise>0:
                raise NotImplementedError('Label noise not implemented for MNIST.')
            if 'mse' in used_dataset:
                train_set = OneHotDataset(train_set, num_classes=10)
                test_set = OneHotDataset(test_set, num_classes=10)
                #train_set = GeneratedDataset(data=train_set.data.view(-1, 28*28).float()/255.0, labels=train_set.targets, transform=None, device=DEVICE,classification=False, to_onehot=True)
                #test_set = GeneratedDataset(data=test_set.data.view(-1, 28*28).float()/255.0, labels=test_set.targets, transform=None, device=DEVICE,classification=False, to_onehot=True)

            train_sampler = torch.utils.data.RandomSampler(train_set,generator=torch.Generator(device=DEVICE))
            train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE,sampler=train_sampler)

            test_sampler = torch.utils.data.RandomSampler(test_set,generator=torch.Generator(device=DEVICE))
            test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE, sampler=test_sampler)

            DIM_X = 28 * 28
            DIM_Y = 10
            SUBFOLDER = 'mnist/'
            loss_fn = nn.CrossEntropyLoss() if 'mse' not in used_dataset else nn.MSELoss()

        elif 'teacher' in used_dataset:
            # copied from 'get rich quick'-paper https://github.com/allanraventos/getrichquick/blob/main/two-layer-relu/plot_fig1a.ipynb
            # but using sgd, not gd
            n_samples, n_samples_test = 10000, 1000
            DIM_X = 100 #2
            k = 4
            min_spread = 0.5 # all target vectors at least 0.5*pi radians apart (in dim=2, min_spread=0.5 only works for k<=3...)
            from utils.kunin_utils import TeacherNetwork
            teacher = TeacherNetwork(DIM_X, k, min_spread=min_spread,classification=True,unit_teachers = True) if used_dataset != 'teacher' else TeacherNetwork(DIM_X, k, min_spread=min_spread,unit_teachers = True)
            inputs = torch.randn(n_samples, DIM_X).float()
            if 'decay' in used_dataset:
                sigma = [(k+1)**(-1.0) for k in range(DIM_X)]
                inputs = generate_sparse_gaussian_training_set(n_samples, DIM_X, sigma)
                inputs = torch.tensor(inputs).float()
            inputs = inputs / torch.norm(inputs, dim=1, keepdim=True)
            inputs_test = torch.randn(n_samples_test, DIM_X).float()
            if 'decay' in used_dataset:
                sigma = [(k+1)**(-1.0) for k in range(DIM_X)]
                inputs_test = generate_sparse_gaussian_training_set(n_samples, DIM_X, sigma)
                inputs_test = torch.tensor(inputs_test).float()
            inputs_test = inputs_test / torch.norm(inputs_test, dim=1, keepdim=True)
            with torch.no_grad():
                if used_dataset != 'teacher':
                    flip_indices = np.random.choice(n_samples, size=int(n_samples*args.label_noise), replace=False)
                    labels = teacher(inputs)
                    labels[flip_indices] = 1-labels[flip_indices]
                    flip_indices_test = np.random.choice(n_samples_test, size=int(n_samples_test*args.label_noise), replace=False)
                    labels_test = teacher(inputs_test)
                    labels_test[flip_indices_test] = 1-labels_test[flip_indices_test]
                    labels = labels.flatten().long()
                    labels_test = labels_test.flatten().long()
                else:
                    labels = teacher(inputs) + args.label_noise * torch.randn((n_samples,1))
                    labels_test = teacher(inputs_test) + args.label_noise * torch.randn((n_samples_test,1))
                
            print(f"input mean: {inputs.mean()}, label mean: {labels.float().mean()}")

            train_set = GeneratedDataset(data=inputs, labels=labels, transform=None, device=DEVICE,classification=(used_dataset != 'teacher'), to_onehot=('softmax' in used_dataset))
            train_sampler = torch.utils.data.RandomSampler(train_set, generator=torch.Generator(device=DEVICE))
            train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, sampler=train_sampler)  

            test_set = GeneratedDataset(data=inputs_test, labels=labels_test, transform=None, device=DEVICE,classification=(used_dataset != 'teacher'), to_onehot=('softmax' in used_dataset))
            test_sampler = torch.utils.data.RandomSampler(test_set, generator=torch.Generator(device=DEVICE))
            test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE, sampler=test_sampler)            
            

            loss_fn = nn.CrossEntropyLoss() if 'teacher_ce' in used_dataset else nn.MSELoss()
            DIM_Y = 2 if used_dataset != 'teacher' else 1
            SUBFOLDER = 'teacher/'
        else:
            raise ValueError('Implement data loading for this dataset.')

            
        if args.numfeaturerankevals == 0:
            feature_rank_iter = None
        else:
            if args.evaliter == 0:
                feature_rank_iter = np.linspace(0, N_EPOCHS-1, args.numfeaturerankevals,dtype=int)
            else:
                feature_rank_iter = np.linspace(0, N_EPOCHS * len(train_dataloader) - 1, args.numfeaturerankevals, dtype=int)
                
        if len(utils.find(EXP_NAME+'.txt','stats/'))>0: # is already broken through different random seed
            all_stats = utils.myload('stats/'+SUBFOLDER+EXP_NAME+'.txt')
            return f'Stats already computed: {EXP_NAME}'
        else:
            all_stats = {}
            
            logging.info(f"Width={WIDTH}, LR={LR}, RHO={RHO}")

            if not args.nomultipliers:
                if args.yangmult:
                    multipl_inout = [0.00390625, 32]
                else:
                    raise NotImplementedError('First tune multipliers for this parameterization.')
            else:
                multipl_inout = None
            
            if not (args.rmsnormlayer or args.outnormlayer):
                model = MLP(in_size=DIM_X,
                            out_size=DIM_Y,
                            hidden_size=WIDTH,
                            n_hidden_layers=N_HIDDEN_LAYERS,
                            parameterization=PARAMETERIZATION,
                            perturbation=PERTURBATION,
                            activation = identity if args.linear else F.relu,
                            out_activation=nn.Softmax(dim=1) if 'softmax' in used_dataset else None,
                            res_connections = args.resnet,
                            flat_indim = DIM_X,
                            multipl_inout=multipl_inout,
                            ll_zero_init=args.llzeroinit,
                            use_bias=args.use_bias,
                            large_bias_init = args.large_bias_init,
                            multiplier_mup = args.multiplier_mup,
                            initvarnorm=args.initvarnorm)
            else:
                if args.rmsnormlayer:
                    model = Scale_Corrected_MLP(in_size=DIM_X,
                                out_size=DIM_Y,
                                hidden_size=WIDTH,
                                n_hidden_layers=N_HIDDEN_LAYERS,
                                parameterization=PARAMETERIZATION,
                                perturbation=PERTURBATION,
                                activation = identity if args.linear else F.relu,
                                out_activation=nn.Softmax(dim=1) if 'softmax' in used_dataset else None,
                                res_connections = args.resnet,
                                flat_indim = DIM_X,
                                multipl_inout=multipl_inout,
                                ll_zero_init=args.llzeroinit,
                                use_bias=args.use_bias,
                                large_bias_init = args.large_bias_init,
                                multiplier_mup = args.multiplier_mup,
                                norm_layer='rms_hidden',
                                initvarnorm=args.initvarnorm)
                elif args.outnormlayer:
                    model = Scale_Corrected_MLP(in_size=DIM_X,
                                out_size=DIM_Y,
                                hidden_size=WIDTH,
                                n_hidden_layers=N_HIDDEN_LAYERS,
                                parameterization=PARAMETERIZATION,
                                perturbation=PERTURBATION,
                                activation = identity if args.linear else F.relu,
                                out_activation=nn.Softmax(dim=1) if 'softmax' in used_dataset else None,
                                res_connections = args.resnet,
                                flat_indim = DIM_X,
                                multipl_inout=multipl_inout,
                                ll_zero_init=args.llzeroinit,
                                use_bias=args.use_bias,
                                large_bias_init = args.large_bias_init,
                                multiplier_mup = args.multiplier_mup,
                                norm_layer=None,
                                out_norm_layer='layernorm',
                                initvarnorm=args.initvarnorm)
            model.to(DEVICE)
            
            # Construct trainer
            if args.savebestmodel:
                savebest=EXP_NAME
            else:
                savebest=None
            if args.scheduler == 'None':
                args.scheduler = None
                
            mlp_trainer = MLPTrainer(model=model,
                                train_dataloader=train_dataloader,
                                eval_dataloader=test_dataloader,
                                lr=LR,
                                rho=RHO,
                                optim_algo=OPTIM_ALGO,
                                weight_decay=args.weightdecay,
                                momentum=args.momentum,
                                scheduler=args.scheduler,
                                exp_name=EXP_NAME,
                                device= DEVICE,
                                loss_fn=loss_fn,
                                classification=False if used_dataset == 'teacher' else True,
                                width_indep_gradnorms=args.indepgradnorms,
                                gn0=args.gn0,
                                extended_eval_iter = args.extendedevaliter,
                                eval_iter = args.evaliter,
                                feature_rank_iter=feature_rank_iter,
                                final_eval=args.finaleval,
                                save_best_file = savebest,
                                spectral_mup=args.spectral_mup,
                                small_input_mup=args.small_input_mup,
                                only_track_norms=args.only_track_norms,
                                del_last_step=args.del_last_step,
                                tag_dict={'W':WIDTH, 'LR':float(LR), 'RHO':float(RHO)})
            mlp_trainer.train(N_EPOCHS, resample_data=False)
            all_stats[(WIDTH, LR, float(RHO))] = mlp_trainer.history

        import datetime
        timestamp = datetime.datetime.now().strftime("_%Y%m%d_%H%M%S")
        utils.mysave('./stats/'+SUBFOLDER,f'config_mlp{"_"+args.dataset}_lr{LR}_rho{RHO}_with{WIDTH}_nlay{args.nhiddenlayers}{"_lin" if args.linear else ""}{"_normlayer" if args.rmsnormlayer else ""}_seed{seed}_'+timestamp+'.txt', args)
        utils.mysave('./stats/'+SUBFOLDER,f'stats_mlp{"_"+args.dataset}_lr{LR}_rho{RHO}_with{WIDTH}_nlay{args.nhiddenlayers}{"_lin" if args.linear else ""}{"_normlayer" if args.rmsnormlayer else ""}_seed{seed}_'+timestamp+'.txt', all_stats)
        
        if isinstance(all_stats[(WIDTH, LR, float(RHO))]["Loss/val"][-1], tuple):
            val_accs = [thisacc[1] for thisacc in all_stats[(WIDTH, LR, float(RHO))]["Loss/val"]]
            train_accs = [thisacc[1] for thisacc in all_stats[(WIDTH, LR, float(RHO))]["Loss/train"]]
        else:
            raise ValueError(f'Unknown acc saving format {all_stats[(WIDTH, LR, float(RHO))]["Loss/val"][-1]}')
        if isinstance(loss_fn, nn.CrossEntropyLoss):
            new_line = f'{WIDTH};{LR};{seed};{timestamp};{N_EPOCHS};{val_accs[-1]};{train_accs[-1]};{np.max(val_accs)};{np.max(train_accs)}'
        else:
            new_line = f'{WIDTH};{LR};{seed};{timestamp};{N_EPOCHS};{val_accs[-1]};{train_accs[-1]};{np.min(val_accs)};{np.min(train_accs)}'

        with open('./stats/' + f'accs_{args.dataset}_nlay{args.nhiddenlayers}{"_lin" if args.linear else ""}{"_rf" if "LL" in args.optim else ""}{"_adam" if "ADAM" in args.optim else "_sgd"}{"_normlayer" if args.rmsnormlayer else ""}.txt', 'a') as file:
            file.write(new_line + '\n')
        
        print(f'width {WIDTH}, epoch {N_EPOCHS}, final train;val for lr {LR}: {train_accs[-1]};{val_accs[-1]}, took time {time.time()-start_time}')
    return 0
    

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run single training run and save stats.")
    parser.add_argument("--gpu_index", type=int, default=0, help="GPU index to use")
    parser.add_argument("--dataset", type=str, default='cifar10', help="Dataset") # 'cifar10_mse', 'cifar10_mse_softmax', 'teacher' (from chizat and kunin papers), 'teacher_ce' (classification task), 'teacher_softmax', 'mnist', 'mnist_mse', 'mnist_mse_softmax'  'teacher_ce_decay'(for better generalization without feature learning)
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--width",type=int, default=256, help="Width placeholder, will be overwritten by width_choice")
    parser.add_argument("--width_choice", type=str, default='standard', help="array of widths to sweep") # 'standard' [256,1024,4096,16384], 'fine', 'few' [256,1024,4096], 'wide' [16384]
    parser.add_argument("--optim", type=str, default='SGD', help="Optimization algorithm")
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate")
    parser.add_argument("--lr_log", type=float, default=9999, help="lr=2**lr_log2")
    parser.add_argument("--lrexp", type=float, default=0.0, help="Learning rate scaling exponent")
    parser.add_argument("--rho", type=float, default=0.0, help="Perturbation radius")
    parser.add_argument("--nepochs", type=int, default=300, help="Number of training epochs")
    parser.add_argument("--param", type=str, default='sp', help="bc-Parametrization")
    parser.add_argument("--perturb", type=str, default='naive', help="Perturbation scaling")
    parser.add_argument("--scheduler", type=str, default='None', help="Optimization algorithm")
    parser.add_argument("--bs", type=int, default=64, help="Batch size")
    parser.add_argument('--weightdecay', type=float, default=0)# 5e-4
    parser.add_argument('--momentum', type=float, default=0)# 0.9
    parser.add_argument("--nomultipliers", action='store_true', help="Use the no input/output multipliers")
    parser.add_argument("--yangmult", action='store_true', help="Use Yang's width-indep input/output multipliers for MLPs")         
    
    #architecture
    parser.add_argument("--nhiddenlayers", type=int, default=2, help="number of hidden layers")
    parser.add_argument("--linear", action='store_true', help="No ReLUs, just linear network.")
    parser.add_argument("--llzeroinit", action='store_true', help="Initialize last layer to 0")
    parser.add_argument("--initvarnorm", action='store_true', help="He init variance irrespective of init multiplier")
    parser.add_argument("--resnet", action='store_true', help="Residual connections except in first and last layer. So makes no difference in 2 layer nets.")
    parser.add_argument("--rmsnormlayer", action='store_true', help="MLP with RMS norm after every layer.")
    parser.add_argument("--outnormlayer", action='store_true', help="MLP with RMS norm after the output layer.")
    parser.add_argument("--use_bias", action='store_true', help="Should MLPs have trainable biases?")

    # parameterization variants
    parser.add_argument("--spectral_mup", action='store_true', help="Use LR scaling fan_out/fan_in (results in different layerwise constants)")
    parser.add_argument("--small_input_mup", action='store_true', help="Use input layer LR scaling 1/sqrt(width) smaller than mup (as in u-mup)")
    parser.add_argument("--multiplier_mup", action='store_true', help="Use multiplier version of mup with naive learning rate scaling")
    parser.add_argument("--large_bias_init", action='store_true', help="In SP, initialize biases with fan_in=1, not fan_in(weights)")
    parser.add_argument("--indepgradnorms", action='store_true', help="If True, gradnorms are scaled independently layerwise to Theta(1)")
    parser.add_argument('-gn0','--gn0', nargs='+', help='List of layers which should NOT be perturbed')
    
    # evaluation
    parser.add_argument("--extendedevaliter", type=int, default=0, help="Extended evaluation every ... steps")
    parser.add_argument("--evaliter", type=int, default=0, help="eval every ... batches instead of every epoch.")
    parser.add_argument("--numfeaturerankevals", type=int, default=0, help="How many feature rank evals across training")
    parser.add_argument("--finaleval", action='store_true', help="Perform a final extended evaluation at the end of training.")
    parser.add_argument("--only_track_norms", action='store_true', help="In extended eval, only track norm statistics, not Hessian, feature ranks, ...")         
    parser.add_argument("--del_last_step", action='store_true', help="Instead of comparing to init, compare to last step")         
    parser.add_argument("--label_noise", type=float, default=0)         
    parser.add_argument("--savebestmodel", action='store_true', help="Save model state with best test accuracy")
    parser.add_argument("--prefix", type=str, default='mlp', help="Prefix for the filename")

    
    args = parser.parse_args()
    main(args)
