import time

import torch

import torch.nn as nn

from utils.eval import accuracy
from utils.general_utils import AverageMeter, ProgressMeter
from utils.model import (
    switch_to_bilevel,
    switch_to_prune,
    switch_to_finetune,
    get_epoch_data
)

import numpy as np


def train(
        model, device, train_loader, criterion, optimizer_list, epoch, args, **kwargs
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

    dummy_model = kwargs["dummy_model"]
    optimizer, mask_optimizer = optimizer_list

    model.train()
    dummy_model.train()
    end = time.time()

    duality_gaps = []
    losses_list = []
    supports = []

    for i, (train_data_batch, val_data_batch) in enumerate(zip(train_loader, val_loader)):
        train_images, train_targets = train_data_batch[0].to(device), train_data_batch[1].to(device)
        val_images, val_targets = val_data_batch[0].to(device), val_data_batch[1].to(device)
        if args.accelerate:
            switch_to_prune(model)
            output = model(train_images)
            loss = criterion(output, train_targets)

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
            
            #Lower level step
            #We do 1 step of SGD on the parameters of the model

            switch_to_finetune(model)
            output = model(val_images)
            loss = criterion(output, val_targets)

            optimizer.zero_grad()
            loss.backward()

            #patch for the rounding bug
            #set to None all the gradients for the popup scores
            for param in model.parameters():
                if not param.requires_grad:
                    param.grad = None
            
            optimizer.step()

            acc1, acc5 = accuracy(output, val_targets, topk=(1, 5))
            losses.update(loss.item(), val_images.size(0))
            top1.update(acc1[0], val_images.size(0))
            top5.update(acc5[0], val_images.size(0))

            
            #upper level step 
            switch_to_prune(model)
            switch_to_finetune(dummy_model)

            #the parameters of the dummy model should be set to m * theta of the model
            score_list = []
            param_list = []
            bias_list = []

            #retrieve the parameters of the model
            with torch.no_grad():
                for (name, param) in model.named_parameters():
                    #retrieve the mask
                    if param.requires_grad:
                        score_list.append(param.data.detach())
                    #retrieve theta
                    if not param.requires_grad and not 'bias' in name:
                        param_list.append(param.data.detach())
                    #retrieve bias
                    if not param.requires_grad and 'bias' in name:
                        bias_list.append(param.data.detach())

            #set the parameters of the dummy model to m * theta
            with torch.no_grad():
                param_score_pointer = 0
                bias_pointer = 0
                for (name, param) in dummy_model.named_parameters():
                    if param.requires_grad and not 'bias' in name:
                        param.data.copy_(score_list[param_score_pointer] * param_list[param_score_pointer])
                        param_score_pointer += 1
                    if param.requires_grad and 'bias' in name:
                        param.data.copy_(bias_list[bias_pointer])
                        bias_pointer += 1
            

            with torch.no_grad():
                for param in dummy_model.parameters():
                    param.grad = torch.zeros_like(param.data)
            
            #compute grad_z l(z = m * theta)
            z_loss = criterion(dummy_model(train_images), train_targets)
            z_loss.backward()

            #retrieve grad_z l(z = m * theta)
            grad_z_list = []
            
            for (name, param) in dummy_model.named_parameters():
                if param.requires_grad and not 'bias' in name:
                    grad_z_list.append(param.grad.view(-1).detach())                       
            
            grad_z = torch.cat(grad_z_list)
            
            paramt = torch.cat([param.view(-1) for param in param_list])
            score = torch.cat([score.view(-1) for score in score_list])
            
            loss_grad_vec = (param - args.lr2 * score * grad_z) * grad_z
        
                
            def pen_grad2vec(parameters):
                penalization_grad = []
                for param in parameters:
                    if param.requires_grad:
                        penalization_grad.append(args.alpha * (torch.exp(-args.alpha * param.view(-1).detach())))
                return torch.cat(penalization_grad)
                
            pen_grad_vec = pen_grad2vec(model.parameters())

            #then the hypergradient is the convex combination of the baseline hypergradient and the penalization gradient
            hypergradient = args.lambd * (loss_grad_vec) + (1 - args.lambd) * pen_grad_vec

            #the linear minimization problem is very simple we don't need to use a solver
            #mstar is equal to 1 if c is negative, 0 otherwise

            m_star = torch.zeros_like(hypergradient)
            m_star[hypergradient < 0] = 1

            #we want to have a diminishing step size
            step_size = 2/(epoch * len(train_loader) + i + 2)


            def mask_tensor(parameters):
                params = []
                for param in parameters:
                    if param.requires_grad:
                        params.append(param.view(-1).detach())
                return torch.cat(params)

            m_k = mask_tensor(model.parameters())
            #then we update the parameters
            pointer = 0
            for param in model.parameters():
                num_param = param.numel()

                #update only if it is a popup score
                #i.e. if param.requires_grad = True

                if param.requires_grad:
                    param.data.copy_((1 - step_size) * param.data + step_size * m_star[pointer:pointer + num_param].view_as(param).data)

                    pointer += num_param
            

            #we want to compute the duality gap as well
            #it is equal to d = - <outer_gradient, m_star - params>

            duality_gap = -torch.dot(hypergradient, m_star - m_k).item()
            duality_gaps.append(duality_gap)

            #calculate the length of the support of mstar
            support = torch.sum(m_star).item()
            supports.append(support)

            output = model(train_images)
            acc1, acc5 = accuracy(output, train_targets, topk=(1, 5))  # log
            losses.update(loss.item(), train_images.size(0))
            top1.update(acc1[0], train_images.size(0))
            top5.update(acc5[0], train_images.size(0))

            losses_list.append(loss.item())



        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
        
        if i <= 3:
            #print stats
            l0, l1 = 0, 0
            mini, maxi = 1000, -1000
            for (name, vec) in model.named_modules():
                if hasattr(vec, "popup_scores"):
                    attr = getattr(vec, "popup_scores")
                    if attr is not None:
                        l0 += torch.sum(attr != 0).item()
                        l1 += (torch.sum(torch.abs(attr)).item())
                        mini = min(mini, torch.min(attr).item())
                        maxi = max(maxi, abs(torch.max(attr).item()))
            
            print("l0 norm of mask: ", l0)
            print("l1 norm of mask: ", l1)
            print("min of mask: ", mini)
            print("max of mask: ", maxi)

            #print the length of the support of mstar :
            print("support of mstar: ", torch.sum(m_star).item())

            #print the duality gap 
            print("duality gap: ", duality_gap)


    #return data related to the mask of this epoch
    epoch_data = get_epoch_data(model)
    epoch_data.append(duality_gaps)
    epoch_data.append(losses_list)
    epoch_data.append(supports)

    return epoch_data