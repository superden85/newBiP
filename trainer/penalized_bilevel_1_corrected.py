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
            #calculating the first part
            switch_to_prune(model)
            mask_optimizer.zero_grad()
            loss_mask = criterion(model(train_images), train_targets)

            loss_mask.backward()

            def grad2vec(parameters):
                grad_vec = []
                for param in parameters:
                    grad_vec.append(param.grad.view(-1).detach())
                return torch.cat(grad_vec)

            first_part = grad2vec(model.parameters())

            #calculating the second part with the dummy model
            switch_to_finetune(dummy_model)

            #the parameters of the dummy model should be set to m * theta of the model
            score_list = []
            param_list = []
            for (name, vec) in model.named_modules():
                #retrieve the mask
                if hasattr(vec, "popup_scores"):
                    attr = getattr(vec, "popup_scores")
                    if attr is not None:
                        score_list.append(attr.view(-1))
                #retrieve the parameters
                if not isinstance(vec, (nn.BatchNorm2d, nn.BatchNorm2d)):
                    if hasattr(vec, "weight"):
                        attr = getattr(vec, "weight")
                        if attr is not None:
                            param_list.append(attr.view(-1))

            score_list = torch.cat(score_list)
            param_list = torch.cat(param_list)

            pointer = 0
            for (name, vec) in dummy_model.named_modules():
                if i == 0:  
                    print(name, vec)
                if not isinstance(vec, (nn.BatchNorm2d, nn.BatchNorm2d)):
                    if hasattr(vec, "weight"):
                        attr = getattr(vec, "weight")
                        if attr is not None:
                            attr.data = param_list[pointer:pointer + attr.numel()].view_as(attr) * score_list[pointer:pointer + attr.numel()].view_as(attr)
                            pointer += attr.numel()
            
            with torch.no_grad():
                for param in dummy_model.parameters():
                    param.grad = torch.zeros_like(param)
            
            z_loss = criterion(dummy_model(train_images), train_targets)

            z_loss.backward()

            grad_z_list = []
            for (name, vec) in dummy_model.named_modules():
                if not isinstance(vec, (nn.BatchNorm2d, nn.BatchNorm2d)):
                    if hasattr(vec, "weight"):
                        attr = getattr(vec, "weight")
                        if attr is not None:
                            grad_z_list.append(attr.grad.view(-1))
            
            grad_z_list = torch.cat(grad_z_list)

            implicit_gradient = -args.lr2 * score_list * grad_z_list ** 2

            #we have to put the implicit gradient in the same shape as the mask gradient
            #we do that by putting zeros everywhere except where there is a mask
            second_part = torch.zeros_like(first_part)
            pointer = 0
            for param in dummy_model.parameters():
                if not param.requires_grad:
                    second_part[pointer:pointer + param.numel()] = implicit_gradient[pointer:pointer + param.numel()]
                    pointer += param.numel()

            #check that in the second part only the popup scores have a non zero gradient
            if i == 0:
                pointer = 0
                for (name, param) in dummy_model.named_parameters():
                    if not param.requires_grad:
                        print(name, param.grad.shape, torch.sum(param.grad != 0).item())
                        pointer += param.numel()
            
            def pen_grad2vec(parameters):
                penalization_grad = []
                for param in parameters:
                    if param.requires_grad:
                        penalization_grad.append(args.alpha * (torch.exp(-args.alpha * param.view(-1).detach())))
                    else:
                        penalization_grad.append(torch.zeros_like(param.view(-1).detach()))
                return torch.cat(penalization_grad)
            
            pen_grad_vec = pen_grad2vec(model.parameters())

            #then the hypergradient is the convex combination of the baseline hypergradient and the penalization gradient
            hypergradient = args.lambd * (first_part + second_part) + (1 - args.lambd) * pen_grad_vec

            #the linear minimization problem is very simple we don't need to use a solver
            #mstar is equal to 1 if c is negative, 0 otherwise

            """ if i<=3:
            #print the ten highest components and the linf norm and the 10 smallest ones
                print("Linf norm: ", torch.norm(hypergradient, p=float("inf")))
                print("Ten highest components: ", torch.topk(hypergradient, 10))
                print("Ten lowest components: ", torch.topk(-hypergradient, 10)) """

            m_star = torch.zeros_like(hypergradient)
            m_star[hypergradient < 0] = 1

            #we want to have a diminishing step size
            step_size = 2/(epoch * len(train_loader) + i + 2)

            #then we update the parameters

            if not isinstance(m_star, torch.Tensor):
                raise TypeError('expected torch.Tensor, but got: {}'
                                .format(torch.typename(vec)))

            pointer = 0
            for param in model.parameters():
                num_param = param.numel()

                #update only if it is a popup score
                #i.e. if param.requires_grad = True

                if param.requires_grad:
                    param.data = ((1 - step_size) * param.data + step_size * m_star[pointer:pointer + num_param].view_as(param).data)

                pointer += num_param

            #we want to compute the duality gap as well
            #it is equal to d = - <outer_gradient, m_star - params>

            def mask_tensor(parameters):
                params = []
                for param in parameters:
                    if param.requires_grad:
                        params.append(param.view(-1).detach())
                    else:
                        params.append(torch.zeros_like(param.view(-1)).detach())
                return torch.cat(params)

            params = mask_tensor(model.parameters())
            duality_gap = -torch.dot(hypergradient, m_star - params).item()
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