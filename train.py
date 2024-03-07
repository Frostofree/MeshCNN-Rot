import time
from options.train_options import TrainOptions
from data import DataLoader
from models import create_model
from util.writer import Writer
from test import run_test



if __name__ == '__main__':
    opt = TrainOptions().parse()
    # print("Train Options: succesfully parsed")

    # For calculating the mean_std of the dataset, we take the entire dataset
    fraction = opt.fraction_of_data_per_class
    opt.fraction_of_data_per_class = 1.0
    dataset = DataLoader(opt)
    # print("DataLoader: succesfully loaded")
    dataset_size = len(dataset)
    # print('#training meshes = %d' % dataset_size)

    model = create_model(opt)
    # print("Model: succesfully created")
    opt.fraction_of_data_per_class = fraction
    writer = Writer(opt)
    # print("Writer: succesfully created")
    total_steps = 0

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0
        # i = 0
        # dataset_len = len(dataset)
        ## We want to Sample the dataset in a random order
        # dataset = dataset
        # dataset = dataset[torch.randperm(len(dataset))]

        # Create a new DataLoader for each epoch, so that the dataset keeps changing every epoch ( number of rotation augementations change)
        dataset = DataLoader(opt)
        dataset_size = len(dataset)
        print('#training meshes = %d' % dataset_size)

        for i, data in enumerate(dataset):
            iter_start_time = time.time()
            if total_steps % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            total_steps += opt.batch_size
            epoch_iter += opt.batch_size
            # print(data)
            # exit (0)
            model.set_input(data)
            model.optimize_parameters()

            if total_steps % opt.print_freq == 0:
                loss = model.loss
                t = (time.time() - iter_start_time) / opt.batch_size
                writer.print_current_losses(epoch, epoch_iter, loss, t, t_data)
                writer.plot_loss(loss, epoch, epoch_iter, dataset_size)

            if i % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_steps %d)' %
                      (epoch, total_steps))
                model.save_network('latest')

            iter_data_time = time.time()
        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' %
                  (epoch, total_steps))
            model.save_network('latest')
            model.save_network(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()
        if opt.verbose_plot:
            writer.plot_model_wts(model, epoch)

        if epoch % opt.run_test_freq == 0:
            acc = run_test(epoch)
            writer.plot_acc(acc, epoch)

    writer.close()
