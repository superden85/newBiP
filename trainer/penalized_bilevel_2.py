import time

import torch

from utils.eval import accuracy
from utils.general_utils import AverageMeter, ProgressMeter
from utils.model import (
    switch_to_bilevel,
    switch_to_prune,
    switch_to_finetune,
)

from scipy.optimize import linprog
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


            #lower level : model retraining
            switch_to_finetune(model)
            output = model(val_images)
            loss = criterion(output, val_targets)

            """ #add a regularization term, defined as the (m * (1 - m)) T vector full of ones
            #we have to loop over all the modules and their popup_scores attribute
            for (name, vec) in model.named_modules():
                if hasattr(vec, 'popup_scores'):
                    loss += args.lambd * (vec.popup_scores * (1 - vec.popup_scores)).sum() """
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            acc1, acc5 = accuracy(output, val_targets, topk=(1, 5))
            losses.update(loss.item(), val_images.size(0))
            top1.update(acc1[0], val_images.size(0))
            top5.update(acc5[0], val_images.size(0))

            #upper level : model pruning
            switch_to_bilevel(model)
            optimizer.zero_grad()
            output = model(train_images)
            loss = criterion(output, train_targets)

            #add a regularization term, defined as the (m * (1 - m)) T vector full of ones
            #we have to loop over all the modules and their popup_scores attribute
            for (name, vec) in model.named_modules():
                if hasattr(vec, 'popup_scores'):
                    loss += args.lambd * (vec.popup_scores * (1 - vec.popup_scores)).sum()
            loss.backward()

            def grad2vec(parameters):
                grad_vec = []
                for param in parameters:
                    grad_vec.append(param.grad.view(-1).detach())
                return torch.cat(grad_vec)

            param_grad_vec = grad2vec(model.parameters())

            switch_to_prune(model)
            mask_optimizer.zero_grad()
            loss_mask = criterion(model(train_images), train_targets)

            #add a regularization term, defined as the (m * (1 - m)) T vector full of ones
            #we have to loop over all the modules and their popup_scores attribute
            for (name, vec) in model.named_modules():
                if hasattr(vec, 'popup_scores'):
                    loss += args.lambd * (vec.popup_scores * (1 - vec.popup_scores)).sum()

            loss_mask.backward()

            mask_grad_vec = grad2vec(model.parameters())
            implicit_gradient = -args.lr2 * mask_grad_vec * param_grad_vec
      

            def append_grad_to_vec(vec, parameters):

                if not isinstance(vec, torch.Tensor):
                    raise TypeError('expected torch.Tensor, but got: {}'
                                    .format(torch.typename(vec)))

                pointer = 0
                for param in parameters:
                    num_param = param.numel()

                    param.grad.copy_(param.grad + vec[pointer:pointer + num_param].view_as(param).data)

                    pointer += num_param

            append_grad_to_vec(implicit_gradient, model.parameters())
            mask_optimizer.step()

            output = model(train_images)
            acc1, acc5 = accuracy(output, train_targets, topk=(1, 5))  # log
            losses.update(loss.item(), train_images.size(0))
            top1.update(acc1[0], train_images.size(0))
            top5.update(acc5[0], train_images.size(0)) 
            
            """


            #Lower level step
            #We do 1 step of SGD on the parameters of the model

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

            
            #Upper level step
            #We want to do Frank-Wolfe on the upper level problem
            #We add also an smooth equivalent of the 0 norm of the mask

            switch_to_prune(model)
            mask_optimizer.zero_grad()
            output = model(train_images)
            loss = criterion(output, train_targets)

            #add a regularization term, defined as the (1-exp(-alpha*mask)) T vector full of ones
            #we have to loop over all the modules and their popup_scores attribute
            for (name, vec) in model.named_modules():
                if hasattr(vec, 'popup_scores'):
                    loss += args.lambd * (1 - torch.exp(-args.alpha * vec.popup_scores)).sum()
            loss.backward()

            #define g = grad(loss) w.r.t. to the mask only 

            def grad2vec(parameters):
                grad_vec = []
                for param in parameters:
                    grad_vec.append(param.grad.view(-1).detach())
                return torch.cat(grad_vec)
            
            mask_grad_vec = grad2vec(model.parameters())

            #the linear minimization problem is very simple we don't need to use a solver
            #mstar is equal to 1 if c is negative, 0 otherwise

            m_star = torch.zeros_like(mask_grad_vec)
            m_star[mask_grad_vec < 0] = 1

            #compute the step size
            #step_size = 2 / (i + 2)

            #update the mask with the step size

            def set_grad_to_vec(vec, parameters):

                if not isinstance(vec, torch.Tensor):
                    raise TypeError('expected torch.Tensor, but got: {}'
                                    .format(torch.typename(vec)))

                pointer = 0
                for param in parameters:
                    num_param = param.numel()

                    param.grad.copy_(vec[pointer:pointer + num_param].view_as(param).data)

                    pointer += num_param


            set_grad_to_vec(m_star, model.parameters())
            mask_optimizer.step()

            output = model(train_images)
            acc1, acc5 = accuracy(output, train_targets, topk=(1, 5))  # log
            losses.update(loss.item(), train_images.size(0))
            top1.update(acc1[0], train_images.size(0))
            top5.update(acc5[0], train_images.size(0))   """         


        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)