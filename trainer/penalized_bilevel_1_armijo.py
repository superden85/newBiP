import time

import torch

from utils.eval import accuracy
from utils.general_utils import AverageMeter, ProgressMeter
from utils.model import (
    switch_to_bilevel,
    switch_to_prune,
    switch_to_finetune,
    get_epoch_data,
    calculate_IOU
)

import numpy as np


def train(
        model, device, train_loader, criterion, optimizer_list, epoch, args
):
    print("->->->->->->->->->-> One epoch with Natural training <-<-<-<-<-<-<-<-<-<-")
    train_loader, val_loader = train_loader

    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4f")
    top1 = AverageMeter("Acc_1", ":6.2f")
    top5 = AverageMeter("Acc_5", ":6.2f")
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch),
    )

    optimizer, mask_optimizer = optimizer_list

    model.train()
    end = time.time()

    duality_gaps = []
    losses_list = []

    if epoch == 0:
        #print the list of the named_modules as well as the number of the parameters
        for name, param in model.named_parameters():
            print(name, param.shape)

    for i, (train_data_batch, val_data_batch) in enumerate(zip(train_loader, val_loader)):
        train_images, train_targets = train_data_batch[0].to(device), train_data_batch[1].to(device)
        val_images, val_targets = val_data_batch[0].to(device), val_data_batch[1].to(device)
        if args.accelerate:
            switch_to_prune(model)
            output = model(train_images)
            loss = criterion(output, train_targets)
            
            # add a regularization term, defined as the (1-exp(-alpha*mask)) T vector full of ones
            loss += args.lambd * (1 - torch.exp(-args.alpha * model.mask)).sum()

            mask_optimizer.zero_grad()
            loss.backward()
            mask_optimizer.step()

            acc1, acc5 = accuracy(output, train_targets, topk=(1, 5))
            losses.update(loss.item(), train_images.size(0))
            top1.update(acc1[0], train_images.size(0))
            top5.update(acc5[0], train_images.size(0))

            switch_to_finetune(model)
            output = model(val_images)
            loss = criterion(output, val_targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc1, acc5 = accuracy(output, val_targets, topk=(1, 5))
            losses.update(loss.item(), val_images.size(0))
            top1.update(acc1[0], val_images.size(0))
            top5.update(acc5[0], val_images.size(0))

        else:

            def update_parameters(new_value):
                if not isinstance(new_value, torch.Tensor):
                    raise TypeError('expected torch.Tensor, but got: {}'
                                    .format(torch.typename(vec)))

                pointer = 0
                for param in model.parameters():
                    num_param = param.numel()

                    #update only if it is a popup score
                    #i.e. if param.requires_grad = True

                    if param.requires_grad:
                        param.data = new_value[pointer:pointer+num_param].view_as(param).data

                    pointer += num_param

            def mask_tensor(parameters):
                params = []
                for param in parameters:
                    if param.requires_grad:
                        params.append(param.view(-1).detach()) 
                    else:
                        params.append(torch.zeros_like(param.view(-1)).detach())
                return torch.cat(params)
            
            def mask_tensor_only(parameters):
                params = []
                for param in parameters:
                    if param.requires_grad:
                        params.append(param.view(-1).detach()) 
                return torch.cat(params)
            
            def mask_tensor_only_inverse(parameters):
                params = []
                for param in parameters:
                    if not param.requires_grad:
                        params.append(param.view(-1).detach()) 
                return torch.cat(params)


            print('-1-')
            mk_only_old = mask_tensor_only(model.parameters())
            print('Coefficient at 85 :', mk_only_old[85].item())
            print('-1-')
            
            
            #Lower level step
            #We do 1 step of SGD on the parameters of the model

            switch_to_finetune(model)
            output = model(val_images)
            loss = criterion(output, val_targets)

            loss*=args.lambd

            optimizer.zero_grad()
            loss.backward()
            
            print('-1.25-')
            mk_only = mask_tensor_only_inverse(model.parameters())
            print('Coefficient at 85 :', mk_only[85].item())
            print('Number of different coefficients :', (mk_only_old != mk_only).sum().item())
            print('-1.25-')

            optimizer.step()

            print('-1.5-')
            mk_only = mask_tensor_only_inverse(model.parameters())
            print('Coefficient at 85 :', mk_only[85].item())
            print('Number of different coefficients :', (mk_only_old != mk_only).sum().item())
            print('-1.5-')

            acc1, acc5 = accuracy(output, val_targets, topk=(1, 5))
            losses.update(loss.item(), val_images.size(0))
            top1.update(acc1[0], val_images.size(0))
            top5.update(acc5[0], val_images.size(0))

            
            #upper level step
            #we have to calculate first the implicit gradient
            
            switch_to_bilevel(model)
            optimizer.zero_grad()
            output = model(train_images)
            loss = criterion(output, train_targets)

            loss*=args.lambd

            # add a regularization term, defined as the (1-exp(-alpha*mask)) T vector full of ones
            for (name, vec) in model.named_modules():
                if hasattr(vec, "popup_scores"):
                    attr = getattr(vec, "popup_scores")
                    if attr is not None:
                        loss += (1 - args.lambd) * (1 - torch.exp(-args.alpha * attr)).sum()
            
            loss.backward()

            def grad2vec(parameters):
                grad_vec = []
                for param in parameters:
                    grad_vec.append(param.grad.view(-1).detach())
                return torch.cat(grad_vec)

            param_grad_vec = grad2vec(model.parameters())

            switch_to_prune(model)
            mask_optimizer.zero_grad()

            print('-2-')
            mk_only = mask_tensor_only(model.parameters())
            print('Coefficient at 85 :', mk_only[85].item())
            print('-2-')

            def calculate_loss_mask():
                loss_mask = criterion(model(train_images), train_targets)

                loss_mask *= args.lambd

                # add a regularization term, defined as the (1-exp(-alpha*mask)) T vector full of ones
                for (name, vec) in model.named_modules():
                    if hasattr(vec, "popup_scores"):
                        attr = getattr(vec, "popup_scores")
                        if attr is not None:
                            loss_mask += (1 - args.lambd) * (1 - torch.exp(-args.alpha * attr)).sum()
                
                return loss_mask
            
            loss_mask = calculate_loss_mask()
            
            loss_mask.backward()

            print('-3-')
            mk_only = mask_tensor_only(model.parameters())
            print('Coefficient at 85 :', mk_only[85].item())
            print('-3-')

            mask_grad_vec = grad2vec(model.parameters())
            implicit_gradient = -args.lr2 * mask_grad_vec * param_grad_vec            
            
            #then the outer gradient is simply:
            outer_gradient = mask_grad_vec + implicit_gradient

            #the linear minimization problem is very simple we don't need to use a solver
            #mstar is equal to 1 if c is negative, 0 otherwise

            m_star = torch.zeros_like(outer_gradient)
            m_star[outer_gradient < 0] = 1

            """ #we want to have a diminishing step size
            step_size = 2/(epoch * len(train_loader) + i + 2) """

            
            if i <= 10:
                mk_only = mask_tensor_only(model.parameters())

                """ #get the argmin of mk_only, and its value
                min_value, min_index = torch.min(mk_only, dim=0, keepdim=False)
                print("Minimum Value:", min_value.item())
                print("Index of Minimum Value:", min_index.item()) """

                #print the coefficient at 85 and the i
                print('-4-')
                print("Coefficient at 85:", mk_only[85].item())
                print('-4-')


            #here we are doing the armijo line search for the stepsize

            step_size = 1.0
            fk = loss_mask.item()
            mk = mask_tensor(model.parameters())

            dk = m_star - mk
            p = torch.dot(outer_gradient, dk).item()

            update_parameters(m_star)
            fk_new = calculate_loss_mask().item()

            counter = 0

            while fk_new > fk + args.gamma * step_size * p:
                
                step_size *= args.gamma
                new_value = mk + step_size * dk
                update_parameters(new_value)

                fk_new = calculate_loss_mask().item()

                counter += 1

            #we want to compute the duality gap as well
            #it is equal to d = - <outer_gradient, m_star - params>

            duality_gap = -p
            duality_gaps.append(duality_gap)

            output = model(train_images)
            acc1, acc5 = accuracy(output, train_targets, topk=(1, 5))  # log
            losses.update(loss.item(), train_images.size(0))
            top1.update(acc1[0], train_images.size(0))
            top5.update(acc5[0], train_images.size(0))

            losses_list.append(loss.item())

            if i <= 10:
                #print the number of steps in the armijo line search
                print("number of steps in the line search: ", counter)


        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
        
        if i <= 5:
            #print stats
            l0, l1 = 0, 0
            mini, maxi = 1000, -1000
            negs = 0
            zeros = 0
            for (name, vec) in model.named_modules():
                if hasattr(vec, "popup_scores"):
                    attr = getattr(vec, "popup_scores")
                    if attr is not None:
                        l0 += torch.sum(attr != 0).item()
                        l1 += (torch.sum(torch.abs(attr)).item())
                        mini = min(mini, torch.min(attr).item())
                        maxi = max(maxi, abs(torch.max(attr).item()))
                        negs += torch.sum(attr < 0).item()
                        zeros += torch.sum(attr == 0).item()
            
            print("l0 norm of mask: ", l0)
            print("l1 norm of mask: ", l1)
            print("min of mask: ", mini)
            print("max of mask: ", maxi)
            print("number of negative values: ", negs)
            print("number of zeros: ", zeros)
            
            print('-5-')
            mk_only = mask_tensor_only(model.parameters())
            print('Coefficient at 85 :', mk_only[85].item())
            print('-5-')

        
    #return data related to the mask of this epoch
    epoch_data = get_epoch_data(model)
    epoch_data.append(duality_gaps)
    epoch_data.append(losses_list)

    return epoch_data