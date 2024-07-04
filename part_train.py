import time
from options.train_options import TrainOptions
from data import PartDataLoader
from models import create_model
from util.writer import Writer
from test import run_test
from models.mesh_classifier import ClassifierModel
import torch
from tqdm import tqdm

def part_divide_dataset(dataset, opt):
    part_data = []
    part_part_data = []
    for i, data in enumerate(dataset):
        if i == 0:
            part_part_data.append(data)
        elif i % opt.part_size != opt.part_size-1:
            part_part_data.append(data)
        else:
            part_part_data.append(data)
            part_data.append(part_part_data)
            part_part_data = []
        if i == len(dataset)-1 and len(part_part_data) > 0:
            part_data.append(part_part_data)
    return part_data
    
if __name__ == '__main__':
    opt = TrainOptions().parse()
    dataset = PartDataLoader(opt)
    model = ClassifierModel(opt)
    dataset_size = len(dataset)
    part_data = part_divide_dataset(dataset, opt)
    print('part_data size: %d' % len(part_data))
    print('Dataset Size: %d' % dataset_size)
    writer = Writer(opt)
    if opt.continue_part_train:
        start_epoch = int(opt.which_super_epoch) + 1
    else:
        start_epoch = 0
    for super_epoch in range(start_epoch, opt.super_epoch):
        super_epoch_start_time = time.time()
        print('Super Epoch: %d' % super_epoch)
        for part_number, part_part_data in enumerate(part_data):
            print('Part Number: %d' % part_number)
            for miniepoch in tqdm(range(opt.superepoch_base)):
                # print('Mini Epoch: %d' % miniepoch)
                iter_start_time = time.time()
                for i, data in enumerate(part_part_data):
                    # print(super_epoch, miniepoch, data['label'], end=' ')
                    iter_start_time = time.time()
                    model.set_input(data)
                    model.optimize_parameters()
                    if i % opt.print_freq == 0 and i != 0:
                        loss = model.loss
                        t = (time.time() - iter_start_time) / opt.part_size
                        # writer.print_current_losses(miniepoch, i, loss, t)
                        # writer.plot_loss(loss, miniepoch, i, dataset_size)
                    if miniepoch % opt.save_epoch_freq == 0 and i == 0 and miniepoch != 0:
                        model.save_network('latest')
                        modelname = str(super_epoch) + '_' + str(part_number) + '_' + str(miniepoch)
                        model.save_network(modelname)
                    
            model.reset_fc2_layer(opt.init_gain)
        acc = run_test()
        model.save_network('super_epoch_' + str(super_epoch))
        print('Super Epoch Time: %d' % (time.time() - super_epoch_start_time))
        print('Super Epoch: %d Done' % super_epoch)
        model.update_learning_rate()
        acc = run_test()
        writer.plot_acc(acc, super_epoch)
    writer.close()
        