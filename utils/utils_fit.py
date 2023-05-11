
import os
import pdb
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils.utils import get_lr,compute_dice_score,f1_score_fun
import torch.nn as nn   
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
import torch.nn.functional as F 

Dice_loss = smp.losses.DiceLoss(mode='binary')
# out_dir = '/home/ktd/rpf_ws/yolov5-pytorch-main/logs'
# log = open(out_dir+'/log.fusion_train.txt',mode='a')#mode=a打开文件用于追加，不清空，继续写，没有则创建
# log.write('\n--- [START %s] %s\n\n' % ('Swin', '-' * 64))
# log.write('\n')
# log.write('** start Mask training here! **\n')
# # log.write('   batch_size = %d \n'%(batch_size))
# log.write('                     |-------------- VALID---------|---- TRAIN/BATCH ----------------\n')
# log.write('  epoch      | dice        loss        | dice         loss          | time          \n')
# log.write('-------------------------------------------------------------------------------------\n')
out_dir = '/home/ktd/rpf_ws/yolov5-pytorch-main/logs'


class FocalLossV1(nn.Module):

    def __init__(self,
                 alpha=0.25,
                 gamma=2,
                 reduction='mean',):
        super(FocalLossV1, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.crit = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, label):
        '''
        Usage is same as nn.BCEWithLogits:
            >>> criteria = FocalLossV1()
            >>> logits = torch.randn(8, 19, 384, 384)
            >>> lbs = torch.randint(0, 2, (8, 19, 384, 384)).float()
            >>> loss = criteria(logits, lbs)
        '''
        probs = torch.sigmoid(logits)
        coeff = torch.abs(label - probs).pow(self.gamma).neg()
        log_probs = torch.where(logits >= 0,
                F.softplus(logits, -1, 50),
                logits - F.softplus(logits, 1, 50))
        log_1_probs = torch.where(logits >= 0,
                -logits + F.softplus(logits, -1, 50),
                -F.softplus(logits, 1, 50))
        loss = label * self.alpha * log_probs + (1. - label) * (1. - self.alpha) * log_1_probs
        loss = loss * coeff

        if self.reduction == 'mean':
            loss = loss.mean()
        if self.reduction == 'sum':
            loss = loss.sum()
        return loss


def fit_one_epoch(model_train, model, ema, yolo_loss, loss_history, eval_callback, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, Epoch, cuda, fp16, scaler, save_period, save_dir, local_rank=0):
    def message(mode='print'):
        asterisk = ' '
        # if mode==('print'):
        #     loss = batch_loss
        if mode==('log'):
            loss = yolo_val_loss
            # if (iteration % iter_save == 0): asterisk = '*'

        text = \
            ('  %6.2f | '%( epoch)).replace('e-0','e-').replace('e+0','e+') + \
            '%4.3f  %4.3f  %4.3f  | '%(mask_total_score, loss_obj_mask,f1_score   ) + \
            '%4.3f    | '%(yolo_val_loss) 


        return text
    # pdb.set_trace()
    log = open(out_dir+'/log.fusion_train.txt',mode='a')#mode=a打开文件用于追加，不清空，继续写，没有则创建
    box_loss = 0.0
    mask_loss = 0.0

    loss        = 0
    val_loss    = 0
    yolo_train_loss = 0
    yolo_val_loss = 0

    if local_rank == 0:
        print('Start Fusion_Train')
        pbar = tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)
    model_train.train()
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step:#你是啥意思？
            break

        images, targets, y_trues,mask = batch[0], batch[1], batch[2],batch[3]#这里加入mask
        with torch.no_grad():
            if cuda:
                images  = images.cuda(local_rank)
                targets = [ann.cuda(local_rank) for ann in targets]
                y_trues = [ann.cuda(local_rank) for ann in y_trues]
                obj_masks = mask.cuda(local_rank)
                obj_masks = obj_masks.unsqueeze(1)
                # weights = 75*obj_masks + 1
                # weights = weights.to(device='cuda')
        #----------------------#
        #   清零梯度
        #----------------------#
        optimizer.zero_grad()
        if not fp16:
            #----------------------#
            #   前向传播
            #----------------------#
            outputs         = model_train(images)
            obj_mask_pred = outputs[3]
            obj_mask_pred = obj_mask_pred.to(device='cuda')
            # loss_obj_mask = nn.BCELoss()
            loss_obj_mask = F.binary_cross_entropy_with_logits(obj_mask_pred,obj_masks)
            loss_dice = Dice_loss(obj_mask_pred,obj_masks)

            loss_value_all  = 0
            #----------------------#
            #   计算损失
            #----------------------#
            for l in range(len(outputs)-1):
                loss_item = yolo_loss(l, outputs[l], targets, y_trues[l])
                loss_value_all  += loss_item
            loss_value = loss_value_all
            loss_value = loss_value_all+0.3*loss_obj_mask +0.3* loss_dice

            #----------------------#
            #   反向传播
            #----------------------#
            loss_value.backward()
            optimizer.step()
        else:
            from torch.cuda.amp import autocast
            with autocast():
                #----------------------#
                #   前向传播
                #----------------------#
                outputs         = model_train(images)


                loss_value_all  = 0

                #----------------------#
                #   计算损失===mask
                #----------------------#
                obj_mask_pred = outputs[3]
                obj_mask_pred = obj_mask_pred.to(device='cuda')
                loss_obj_mask = F.binary_cross_entropy_with_logits(obj_mask_pred,obj_masks)
                loss_dice = Dice_loss(obj_mask_pred,obj_masks)



                #----------------------#
                #   计算损失===box
                #----------------------#
                for l in range(len(outputs)-1):
                    loss_item = yolo_loss(l, outputs[l], targets, y_trues[l])
                    loss_value_all  += loss_item
                loss_value = loss_value_all
                loss_value = loss_value_all+0.5 *loss_obj_mask + 0.5 * loss_dice


            #----------------------#
            #   反向传播
            #----------------------#
            scaler.scale(loss_value).backward()
            scaler.step(optimizer)
            scaler.update()
        if ema:
            ema.update(model_train)
        obj_mask_pred = torch.nn.Sigmoid()(obj_mask_pred)

        yolo_train_loss += loss_value_all.item()
        loss += loss_value.item()
        print("Mask Loss:",loss_obj_mask.item())
        ce_score = 1-((abs(obj_mask_pred-obj_masks)).sum().item() )  / (obj_mask_pred.shape[0] *      obj_mask_pred.shape[1] * obj_mask_pred.shape[2]*obj_mask_pred.shape[3])
        avg_dice_score = compute_dice_score(obj_mask_pred,obj_masks)
        # pdb.set_trace()
        f1_score = f1_score_fun(obj_mask_pred,obj_masks)
        mask_total_score = 0.3 *ce_score+0.7* avg_dice_score
        print("Mask Acc:",mask_total_score)
        print("f1_score:",f1_score)
        if local_rank == 0:
            pbar.set_postfix(**{'loss'  : loss / (iteration + 1), 
                                'lr'    : get_lr(optimizer)})
            pbar.update(1)

#=====================================================================#
#
#
#===========================下面是进入验证环节==========================#
    # pdb.set_trace()
    if local_rank == 0:
        pbar.close()
        print('Finish Train')
        print('Start Validation')
        pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)

    if ema:
        model_train_eval = ema.ema
    else:
        model_train_eval = model_train.eval()
        
    for iteration, batch in enumerate(gen_val):
        if iteration >= epoch_step_val:
            break
        images, targets, y_trues,masks = batch[0], batch[1], batch[2],batch[3]
        with torch.no_grad():
            if cuda:
                images  = images.cuda(local_rank)
                obj_masks  = masks.cuda(local_rank)
                obj_masks = obj_masks.unsqueeze(1)
                targets = [ann.cuda(local_rank) for ann in targets]
                y_trues = [ann.cuda(local_rank) for ann in y_trues]
            #----------------------#
            #   清零梯度
            #----------------------#
            optimizer.zero_grad()
            #----------------------#
            #   前向传播
            #----------------------#
            outputs         = model_train_eval(images)
            obj_mask_pred = outputs[3]
            obj_mask_pred = obj_mask_pred.to(device='cuda')
            # loss_obj_mask = nn.BCELoss()
            # loss_obj_mask = loss_obj_mask(obj_mask_pred,obj_masks)
            loss_obj_mask = F.binary_cross_entropy_with_logits(obj_mask_pred,obj_masks)
            loss_dice = Dice_loss(obj_mask_pred,obj_masks)
            obj_mask_pred = torch.nn.Sigmoid()(obj_mask_pred)
            mask_visualization(images,obj_mask_pred,obj_masks,iteration)

            loss_value_all  = 0
            #----------------------#
            #   计算损失
            #----------------------#
            for l in range(len(outputs)-1):
                loss_item = yolo_loss(l, outputs[l], targets, y_trues[l])
                loss_value_all  += loss_item
            loss_value  =  loss_value_all+0.3 *loss_obj_mask+0.3 * loss_dice

        yolo_val_loss +=loss_value_all.item()
        val_loss += loss_value.item()
        # pdb.set_trace()
        print("Mask Loss:",loss_obj_mask.item())
        
        ce_score = 1-((abs(obj_mask_pred-obj_masks)).sum().item() )  / (obj_mask_pred.shape[0] *      obj_mask_pred.shape[1] * obj_mask_pred.shape[2]*obj_mask_pred.shape[3])
        avg_dice_score = compute_dice_score(obj_mask_pred,obj_masks)
        # pdb.set_trace()
        f1_score = f1_score_fun(obj_mask_pred,obj_masks)
        mask_total_score = 0.3 *ce_score+0.7* avg_dice_score
        print("Mask Acc:",mask_total_score)
        print("Mask Acc_f1_score:",f1_score)
        if local_rank == 0:
            pbar.set_postfix(**{'val_loss': val_loss / (iteration + 1)})
            pbar.update(1)
    # pdb.set_trace()
    # log.write(str(Epoch)+" | ",str(mask_total_score)+" | ",str(loss_obj_mask)+" |",str(yolo_val_loss)+ '\n')
    log.write(message(mode='log')+'\n')
    if local_rank == 0:
        pbar.close()
        print('Finish Validation')
        loss_history.append_loss(epoch + 1, yolo_train_loss / epoch_step, yolo_val_loss / epoch_step_val)
        eval_callback.on_epoch_end(epoch + 1, model_train_eval)
        print('Epoch:'+ str(epoch + 1) + '/' + str(Epoch))
        print('Total Loss: %.3f || Val Loss: %.3f ' % (loss / epoch_step, val_loss / epoch_step_val))
        
        #-----------------------------------------------#
        #   保存权值
        #-----------------------------------------------#
        if ema:
            save_state_dict = ema.ema.state_dict()
        else:
            save_state_dict = model.state_dict()

        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            torch.save(save_state_dict, os.path.join(save_dir, "ep%03d-loss%.3f-val_loss%.3f.pth" % (epoch + 1, loss / epoch_step, val_loss / epoch_step_val)))
            
        if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):
            print('Save best model to best_epoch_weights.pth')
            torch.save(save_state_dict, os.path.join(save_dir, "best_epoch_weights.pth"))
            
        torch.save(save_state_dict, os.path.join(save_dir, "last_epoch_weights.pth"))



def mask_eval(model,
        dataloader: DataLoader):
    
    """
    Args:
        Dataloader: A torch dataloader
    
    Note: This function evaluates the model on the dataloader and returns obj_mask_loss, vp_loss, obj_mask_acc, vp_acc

    """
    ex_path = '/home/ktd/rpf_ws/yolov5-pytorch-main/ckpt'
    model.eval()
    with torch.no_grad():
        total_mask_loss = 0
        obj_mask_acc = 0.0
        f1_score = 0.0

        num_batches = len(dataloader)
        for batch_number, (rgb_img,target,y_true,obj_mask) in enumerate(dataloader):
            if(batch_number%5==0):
                print("Eval Batch: " + str(batch_number) + " / " + str(num_batches))
            rgb_img = rgb_img.type(torch.FloatTensor)
            rgb_img = rgb_img.to(device='cuda')

            obj_mask = obj_mask.type(torch.FloatTensor)
            obj_mask = obj_mask.to(device='cuda')
            obj_mask = obj_mask.unsqueeze(1)

            # weights2 = 75*obj_mask + 1
            # weights2 = weights2.to(device = 'cuda')

            outputs = model(rgb_img)

            obj_mask_pred = outputs[3]
            obj_mask_pred = obj_mask_pred.to(device='cuda')

            # pdb.set_trace()
            # pdb.set_trace()
            # vp_visualization(vp_pred,vp,batch_number)
            # mask_visualization(rgb_img,obj_mask_pred,obj_mask,batch_number)


            # loss_obj_mask = nn.BCEWithLogitsLoss()
            loss_obj_mask = F.binary_cross_entropy_with_logits(obj_mask_pred,obj_mask)
            loss_dice = Dice_loss(obj_mask_pred,obj_mask)
            # loss_obj_mask = loss_obj_mask(obj_mask_pred,obj_mask)
            loss_obj_mask = loss_obj_mask+loss_dice
            total_mask_loss += loss_obj_mask.item()
            obj_mask_pred = torch.nn.Sigmoid()(obj_mask_pred)
            mask_visualization(rgb_img,obj_mask_pred,obj_mask,batch_number)
            


            #round_pred_obj = (obj_mask_pred > 0.5).float()
            #round_pred_vp = (vp_pred > 0.5).float()
            #round_pred_vp = torch.sigmoid(vp_pred)
            
            obj_mask_acc += compute_dice_score(obj_mask_pred,obj_mask)
            f1_score  += f1_score_fun(obj_mask_pred,obj_mask)

            # obj_mask_acc += (1-((abs(obj_mask_pred-obj_mask)).sum().item() )  / (obj_mask_pred.shape[0] * obj_mask_pred.shape[1] * obj_mask_pred.shape[2]*obj_mask_pred.shape[3]))

            # obj_mask_loss += loss_obj_mask

    total_mask_loss = total_mask_loss / num_batches
    # pdb.set_trace()

    obj_mask_acc = obj_mask_acc / num_batches
    f1_score = f1_score/num_batches


    torch.save(model.state_dict(),os.path.join(ex_path,f'{obj_mask_acc}mask_model.pth'))

    print('save model successfully!')

    return total_mask_loss,   obj_mask_acc,f1_score#为什么这里正确率要乘100


def mask_visualization(rgb_img,mask_pred,mask_true,batch_num):

    dir_path = '/home/ktd/rpf_ws/yolov5-pytorch-main/mask_vision'
    mask_pred.cpu()
    mask_true.cpu()
    # pdb.set_trace()
    bn = mask_pred.shape[0]
    assert bn==mask_true.shape[0]
    # for n in range(bn):
    n = 0
    mask_true_v = mask_true[n][0]
    mask_pred_v = mask_pred[n][0]
    rgb_img_v = rgb_img[0]

    plt.figure()
    plt.subplot(1,3,1)
    plt.imshow(mask_pred_v.cpu().numpy(),cmap='gray')
    # plt.imshow(mask_true.permute(1,2,0).cpu().numpy(),cmap='gray')#=========plt只支持(m,n)(m,n,3)(m,n,4)
    plt.title('mask_pred')
    plt.axis('off')
    plt.subplot(1,3,2)
    plt.imshow(mask_true_v.cpu().numpy(),cmap='gray')
    plt.title('mask_true')
    plt.axis('off')
    plt.subplot(1,3,3)
    plt.imshow(rgb_img_v.permute(1,2,0).cpu().numpy())
    plt.title('rgb_img')
    plt.axis('off')
    plt.savefig(os.path.join(dir_path,f'mask_val{batch_num}.png'))
    plt.close()

# def message(mode='print'):
#     asterisk = ' '
#     # if mode==('print'):
#     #     loss = batch_loss
#     if mode==('log'):
#         loss = train_loss
#         # if (iteration % iter_save == 0): asterisk = '*'

#     text = \
#         ('  %6.2f | '%( Epoch)).replace('e-0','e-').replace('e+0','e+') + \
#         '%4.3f  %4.3f    | '%(val_obj_mask_acc, val_obj_mask_loss   ) + \
#         '%4.3f  %4.3f   | '%(train_mask_acc,  train_loss,) + \
#         '%s' % (elapsed)

#     return text
# 
    # log.write(str(Epoch)+" | ",str(mask_total_score)+" | ",str(loss_obj_mask)+" |",str(yolo_val_loss)+ '\n')
