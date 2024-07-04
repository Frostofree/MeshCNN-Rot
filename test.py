from options.test_options import TestOptions
from data import DataLoader
from models import create_model
from util.writer import Writer
import time
import torch
from tqdm import tqdm
from models.mesh_classifier import ClassifierModel

def run_test(epoch=0):
    start_time = time.time()
    opt = TestOptions().parse()
    opt.serial_batches = True  # no shuffle
    dataset = DataLoader(opt)
    # model = create_model(opt)
    model = ClassifierModel(opt)
    writer = Writer(opt)
    # test
    writer.reset_counter()
    for i, data in tqdm(enumerate(dataset)):
        model.set_input(data)
        # print(data['mesh'][0].filename)
        # break
        ncorrect, nexamples , out , fc_1 = model.test()
        writer.update_counter(ncorrect, nexamples)

        out = out.cpu().detach()
        fc_1 = fc_1.cpu().detach()

        a = data['mesh'][0].filename
        # remove the .obj from the filename
        a = a[:-4]
        
        # Export out and fc_1 as numpy arrays for further analysis

        # torch.save(out, './features/out_p8_{}.pt'.format(i))
        torch.save(fc_1, './features/M40_heavy_part_train_30/test_features/{}.pt'.format(a))
    print('Time taken for testing: %f' % (time.time() - start_time))

        
    # writer.print_acc(epoch, writer.acc)
    return writer.acc


if __name__ == '__main__':
    run_test()
