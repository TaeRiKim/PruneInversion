import torch
import torchvision

import numpy as np
from PIL import Image

torch.backends.cudnn.benchmark = True

import matplotlib.pyplot as plt
from collections import defaultdict
import datetime
import time
import os

import neptune

from options import option
from prune import pruning
from optimization_strategy import training_strategy
from data_processing import construct_dataloaders
from reconstruction_algorithms import GradientReconstructor
from reconstruction_algorithms import loss_steps
from reconstruction_algorithms import FedAvgReconstructor
from metrics import psnr
import consts


neptune.init(project_qualified_name='tigaelyu/PRNN', api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiOGEzOTI3MGMtNzVjYi00YzMxLThkMjMtMTYzZTk3NmY1ODdlIn0=')

args = option().parse_args()

defs = training_strategy('conservative')

defs.epochs = args.epochs

cifar10_mean = [0.4914672374725342, 0.4822617471218109, 0.4467701315879822]
cifar10_std = [0.24703224003314972, 0.24348513782024384, 0.26158785820007324]


if args.deterministic:
    image2graph2vec.utils.set_deterministic()

if __name__ == "__main__":
    # Choose GPU device and print status information:   
    device = torch.device('cuda:%d' %args.gpu) if torch.cuda.is_available() else torch.device('cpu')
   
    setup = dict(device=device, dtype=torch.float)
    
    start_time = time.time()

    # Prepare for training

    # Get data:
    loss_fn, trainloader, validloader = construct_dataloaders(args.dataset, defs, data_path=args.data_path)
    idx = args.target_id
    dm = torch.as_tensor(getattr(consts, f'{args.dataset.lower()}_mean'), **setup)[:, None, None]
    ds = torch.as_tensor(getattr(consts, f'{args.dataset.lower()}_std'), **setup)[:, None, None]

    if args.dataset == 'ImageNet':
        if args.model == 'ResNet152':
            model = torchvision.models.resnet152(pretrained=args.trained_model)
            pruning(args.model, model, args.pruning_rate)
        else:
            model = torchvision.models.resnet18(pretrained=args.trained_model)
            pruning(args.model, model, args.pruning_rate)
        model_seed = None
    '''
    else:
        model, model_seed = inversefed.construct_model(args.model, num_classes=10, num_channels=3)
    '''
    
    model.to(**setup)
    model.eval()

    # Sanity check: Validate model accuracy
    training_stats = defaultdict(list)
    # inversefed.training.training_routine.validate(model, loss_fn, validloader, defs, setup, training_stats)
    # name, format = loss_fn.metric()
    # print(f'Val loss is {training_stats["valid_losses"][-1]:6.4f}, Val {name}: {training_stats["valid_" + name][-1]:{format}}.')

    # Choose example images from the validation set or from third-party sources
    if args.num_images == 1:
        if args.target_id == -1:  # demo image
            # Specify PIL filter for lower pillow versions
            ground_truth = torch.as_tensor(np.array(Image.open("auto.jpg").resize((32, 32), Image.BICUBIC)) / 255, **setup)
            ground_truth = ground_truth.permute(2, 0, 1).sub(dm).div(ds).unsqueeze(0).contiguous()
            if not args.label_flip:
                labels = torch.as_tensor((1,), device=setup['device'])
            else:
                labels = torch.as_tensor((5,), device=setup['device'])
            target_id = -1
        else:
            if args.target_id is None:
                target_id = np.random.randint(len(validloader.dataset))
            else:
                target_id = args.target_id
            ground_truth, labels = validloader.dataset[target_id]
            if args.label_flip:
                labels = torch.randint((10,))
            ground_truth, labels = ground_truth.unsqueeze(0).to(**setup), torch.as_tensor((labels,), device=setup['device'])
        img_shape = (3, ground_truth.shape[2], ground_truth.shape[3])
        ground_truth_denormalized = torch.clamp(ground_truth * ds + dm, 0, 1)
        torchvision.utils.save_image(ground_truth_denormalized, f'{idx}_ImageNet_input.png')

    else:
        ground_truth, labels = [], []
        if args.target_id is None:
            target_id = np.random.randint(len(validloader.dataset))
        else:
            target_id = args.target_id
        while len(labels) < args.num_9images:
            img, label = validloader.dataset[target_id]
            target_id += 1
            if label not in labels:
                labels.append(torch.as_tensor((label,), device=setup['device']))
                ground_truth.append(img.to(**setup))

        ground_truth = torch.stack(ground_truth)
        labels = torch.cat(labels)
        if args.label_flip:
            labels = torch.permute(labels)
        img_shape = (3, ground_truth.shape[2], ground_truth.shape[3])

    # Run reconstruction
    if args.accumulation == 0:
        model.zero_grad()
        target_loss, _, _ = loss_fn(model(ground_truth), labels)
        input_gradient = torch.autograd.grad(target_loss, model.parameters())
        input_gradient = [grad.detach() for grad in input_gradient]
        full_norm = torch.stack([g.norm() for g in input_gradient]).mean()
        print(f'Full gradient norm is {full_norm:e}.')

        # Run reconstruction in different precision?
        if args.dtype != 'float':
            if args.dtype in ['double', 'float64']:
                setup['dtype'] = torch.double
            elif args.dtype in ['half', 'float16']:
                setup['dtype'] = torch.half
            else:
                raise ValueError(f'Unknown data type argument {args.dtype}.')
            print(f'Model and input parameter moved to {args.dtype}-precision.')
            dm = torch.as_tensor(cifar10_mean, **setup)[:, None, None]
            ds = torch.as_tensor(cifar10_std, **setup)[:, None, None]
            ground_truth = ground_truth.to(**setup)
            input_gradient = [g.to(**setup) for g in input_gradient]
            model.to(**setup)
            model.eval()

        if args.optim == 'ours':
            config = dict(signed=args.signed,
                          boxed=args.boxed,
                          cost_fn=args.cost_fn,
                          indices='def',
                          weights='equal',
                          lr=0.1,
                          optim=args.optimizer,
                          restarts=args.restarts,
                          max_iterations=24_000,
                          total_variation=args.tv,
                          init='randn',
                          filter='none',
                          lr_decay=True,
                          scoring_choice='loss')
        elif args.optim == 'zhu':
            config = dict(signed=False,
                          boxed=False,
                          cost_fn='l2',
                          indices='def',
                          weights='equal',
                          lr=1e-4,
                          optim='LBFGS',
                          restarts=args.restarts,
                          max_iterations=300,
                          total_variation=args.tv,
                          init=args.init,
                          filter='none',
                          lr_decay=False,
                          scoring_choice=args.scoring_choice)

        neptune.create_experiment(tags=['maxiter: %d'%config["max_iterations"],'restart: %d'%args.restarts, 'imagelabel: %d'%args.target_id, 'model: %s'%args.model, 'pruningrate: %f'%args.pruning_rate, 'imagenum: %d'%args.num_images, 'tv: %f'%args.tv, 'lr: %f'%config["lr"]])

        rec_machine = GradientReconstructor(model, (dm, ds), config, num_images=args.num_images)
        output, stats = rec_machine.reconstruct(input_gradient, labels, img_shape=img_shape, dryrun=args.dryrun)

    else:
        local_gradient_steps = args.accumulation
        local_lr = 1e-4
        input_parameters = loss_steps(model, ground_truth, labels,
                                                                           lr=local_lr, local_steps=local_gradient_steps)
        input_parameters = [p.detach() for p in input_parameters]

        # Run reconstruction in different precision?
        if args.dtype != 'float':
            if args.dtype in ['double', 'float64']:
                setup['dtype'] = torch.double
            elif args.dtype in ['half', 'float16']:
                setup['dtype'] = torch.half
            else:
                raise ValueError(f'Unknown data type argument {args.dtype}.')
            print(f'Model and input parameter moved to {args.dtype}-precision.')
            ground_truth = ground_truth.to(**setup)
            dm = torch.as_tensor(cifar10_mean, **setup)[:, None, None]
            ds = torch.as_tensor(cifar10_std, **setup)[:, None, None]
            input_parameters = [g.to(**setup) for g in input_parameters]
            model.to(**setup)
            model.eval()

        config = dict(signed=args.signed,
                      boxed=args.boxed,
                      cost_fn=args.cost_fn,
                      indices=args.indices,
                      weights=args.weights,
                      lr=0.1,
                      optim=args.optimizer,
                      restarts=args.restarts,
                      max_iterations=24_000,
                      total_variation=args.tv,
                      init=args.init,
                      filter='none',
                      lr_decay=True,
                      scoring_choice=args.scoring_choice)

        rec_machine = FedAvgReconstructor(model, (dm, ds), local_gradient_steps, local_lr, config,
                                                     num_images=args.num_images, use_updates=True)
        output, stats = rec_machine.reconstruct(input_parameters, labels, img_shape=img_shape, dryrun=args.dryrun)

    pr = args.pruning_rate
    num = args.num_images
    output_denormalized = torch.clamp(output * ds + dm, 0, 1)
    torchvision.utils.save_image(output_denormalized, f'{idx}_{pr}_{num}_{config["max_iterations"]}_ImageNet_output.png')
    ground_truth_denormalized = torch.clamp(ground_truth * ds + dm, 0, 1)
    torchvision.utils.save_image(ground_truth_denormalized, f'{idx}_{num}_ImageNet_input.png')
    final = output.cpu().numpy()
    plt.imsave('result.png',final)

    # Compute stats
    test_mse = (output - ground_truth).pow(2).mean().item()
    feat_mse = (model(output) - model(ground_truth)).pow(2).mean().item()
    test_psnr = psnr(output, ground_truth, factor=1 / ds)


    # Save the resulting image
    if args.save_image and not args.dryrun:
        os.makedirs(args.image_path, exist_ok=True)
        output_denormalized = torch.clamp(output * ds + dm, 0, 1)
        rec_filename = (f'{validloader.dataset.classes[labels][0]}_{"trained" if args.trained_model else ""}'
                        f'{args.model}_{args.cost_fn}-{args.target_id}.png')
        torchvision.utils.save_image(output_denormalized, os.path.join(args.image_path, rec_filename))

        gt_denormalized = torch.clamp(ground_truth * ds + dm, 0, 1)
        gt_filename = (f'{validloader.dataset.classes[labels][0]}_ground_truth-{args.target_id}.png')
        torchvision.utils.save_image(gt_denormalized, os.path.join(args.image_path, gt_filename))
    else:
        rec_filename = None
        gt_filename = None



    # Print final timestamp
    print(datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p"))
    print('---------------------------------------------------')
    print(f'Finished computations with time: {str(datetime.timedelta(seconds=time.time() - start_time))}')
    print('-------------Job finished.-------------------------')