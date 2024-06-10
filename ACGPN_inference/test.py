import time
from collections import OrderedDict
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
import os
import numpy as np
import torch
from torch.autograd import Variable
from tensorboardX import SummaryWriter
import cv2
import argparse

writer = SummaryWriter('runs/G1G2')
SIZE = 320
NC = 14

def generate_label_plain(inputs):
    size = inputs.size()
    pred_batch = []
    for input in inputs:
        input = input.view(1, NC, 256, 192)
        pred = np.squeeze(input.data.max(1)[1].cpu().numpy(), axis=0)
        pred_batch.append(pred)

    pred_batch = np.array(pred_batch)
    pred_batch = torch.from_numpy(pred_batch)
    label_batch = pred_batch.view(size[0], 1, 256, 192)

    return label_batch

def generate_label_color(inputs):
    label_batch = []
    for i in range(len(inputs)):
        label_batch.append(util.tensor2label(inputs[i], opt.label_nc))
    label_batch = np.array(label_batch)
    label_batch = label_batch * 2 - 1
    input_label = torch.from_numpy(label_batch)

    return input_label

def complete_compose(img, mask, label):
    label = label.cpu().numpy()
    M_f = label > 0
    M_f = M_f.astype(np.int)
    M_f = torch.FloatTensor(M_f).cuda()
    masked_img = img * (1 - mask)
    M_c = (1 - mask.cuda()) * M_f
    M_c = M_c + torch.zeros(img.shape).cuda()  # broadcasting
    return masked_img, M_c, M_f

def compose(label, mask, color_mask, edge, color, noise):
    masked_label = label * (1 - mask)
    masked_edge = mask * edge
    masked_color_strokes = mask * (1 - color_mask) * color
    masked_noise = mask * noise
    return masked_label, masked_edge, masked_color_strokes, masked_noise

def changearm(old_label):
    label = old_label
    arm1 = torch.FloatTensor((data['label'].cpu().numpy() == 11).astype(np.int))
    arm2 = torch.FloatTensor((data['label'].cpu().numpy() == 13).astype(np.int))
    noise = torch.FloatTensor((data['label'].cpu().numpy() == 7).astype(np.int))
    label = label * (1 - arm1) + arm1 * 4
    label = label * (1 - arm2) + arm2 * 4
    label = label * (1 - noise) + noise * 4
    return label

# Argument parser
def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', type=str, required=True, help='path to dataset')
    parser.add_argument('--name', type=str, default='label2city', help='name of the experiment')
    parser.add_argument('--color_name', type=str, required=True, help='name of the color image file')
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
    parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
    parser.add_argument('--debug', action='store_true', help='debug mode')
    parser.add_argument('--display_freq', type=int, default=100, help='frequency of showing training results on screen')
    parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
    parser.add_argument('--save_latest_freq', type=int, default=5000, help='frequency of saving the latest results')
    parser.add_argument('--save_epoch_freq', type=int, default=5, help='frequency of saving checkpoints at the end of epochs')
    parser.add_argument('--niter', type=int, default=100, help='# of iter at starting learning rate')
    parser.add_argument('--niter_decay', type=int, default=100, help='# of iter to linearly decay learning rate to zero')
    parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
    parser.add_argument('--label_nc', type=int, default=14, help='# of input label channels')
    parser.add_argument('--loadSize', type=int, default=512, help='scale images to this size')
    parser.add_argument('--fineSize', type=int, default=512, help='then crop to this size')
    parser.add_argument('--resize_or_crop', type=str, default='resize_and_crop', help='scaling and cropping of images at load time')
    parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data augmentation')
    parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
    parser.add_argument('--nThreads', type=int, default=4, help='# threads for loading data')
    parser.add_argument('--tf_log', action='store_true', help='if specified, use tensorboard logging. Requires tensorflow installed')
    # Add other arguments as needed
    opt = parser.parse_args()
    return opt

if __name__ == '__main__':
    opt = get_opt()
    print(f"Selected color image: {opt.color_name}")

    os.makedirs('sample', exist_ok=True)
    iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')
    if opt.continue_train:
        try:
            start_epoch, epoch_iter = np.loadtxt(iter_path, delimiter=',', dtype=int)
        except:
            start_epoch, epoch_iter = 1, 0
        print('Resuming from epoch %d at iteration %d' % (start_epoch, epoch_iter))
    else:
        start_epoch, epoch_iter = 1, 0

    if opt.debug:
        opt.display_freq = 1
        opt.print_freq = 1
        opt.niter = 1
        opt.niter_decay = 0
        opt.max_dataset_size = 10

    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)
    print('# Inference images = %d' % dataset_size)

    model = create_model(opt)

    total_steps = (start_epoch - 1) * dataset_size + epoch_iter

    display_delta = total_steps % opt.display_freq
    print_delta = total_steps % opt.print_freq
    save_delta = total_steps % opt.save_latest_freq

    step = 0

    for epoch in range(start_epoch, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        if epoch != start_epoch:
            epoch_iter = epoch_iter % dataset_size
        for i, data in enumerate(dataset, start=epoch_iter):
            iter_start_time = time.time()
            total_steps += opt.batchSize
            epoch_iter += opt.batchSize

            save_fake = True

            ## Add gaussian noise channel
            t_mask = torch.FloatTensor((data['label'].cpu().numpy() == 7).astype(np.float))

            mask_clothes = torch.FloatTensor((data['label'].cpu().numpy() == 4).astype(np.int))
            mask_fore = torch.FloatTensor((data['label'].cpu().numpy() > 0).astype(np.int))
            img_fore = data['image'] * mask_fore
            img_fore_wc = img_fore * mask_fore
            all_clothes_label = changearm(data['label'])

            losses, fake_image, real_image, input_label, L1_loss, style_loss, clothes_mask, CE_loss, rgb, alpha = model(
                Variable(data['label'].cuda()), Variable(data['edge'].cuda()), Variable(img_fore.cuda()), Variable(mask_clothes.cuda()),
                Variable(data['color'].cuda()), Variable(all_clothes_label.cuda()), Variable(data['image'].cuda()), Variable(data['pose'].cuda()),
                Variable(data['image'].cuda()), Variable(mask_fore.cuda())
            )

            losses = [torch.mean(x) if not isinstance(x, int) else x for x in losses]
            loss_dict = dict(zip(model.module.loss_names, losses))

            loss_D = (loss_dict['D_fake'] + loss_dict['D_real']) * 0.5
            loss_G = loss_dict['G_GAN'] + torch.mean(CE_loss)

            writer.add_scalar('loss_d', loss_D, step)
            writer.add_scalar('loss_g', loss_G, step)
            writer.add_scalar('loss_CE', torch.mean(CE_loss), step)
            writer.add_scalar('loss_g_gan', loss_dict['G_GAN'], step)

            a = generate_label_color(generate_label_plain(input_label)).float().cuda()
            b = real_image.float().cuda()
            c = fake_image.float().cuda()
            d = torch.cat([clothes_mask, clothes_mask, clothes_mask], 1)
            combine = torch.cat([a[0], d[0], b[0], c[0], rgb
