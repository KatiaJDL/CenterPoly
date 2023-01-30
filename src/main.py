from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os

import torch
import random
import numpy as np
import torch.utils.data
from opts import opts
from models.model import create_model, load_model, save_model
from models.data_parallel import DataParallel
from logger import Logger
from datasets.dataset_factory import get_dataset
from trains.train_factory import train_factory

import wandb
import tracemalloc


def main(opt):

  wandb.init(project="CenterPoly", entity="kjdl", mode='disabled')

  random.seed(opt.seed)
  np.random.seed(opt.seed)
  torch.manual_seed(opt.seed)
  torch.cuda.manual_seed(opt.seed)
  torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.test
  Dataset = get_dataset(opt.dataset, opt.task)
  opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
  print(opt)

  logger = Logger(opt)

  os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
  opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')
  
  print('Creating model...')
  model = create_model(opt.arch, opt.heads, opt.head_conv)
  optimizer = torch.optim.Adam(model.parameters(), opt.lr)
  start_epoch = 0
  if opt.load_model != '':
    model, optimizer, start_epoch = load_model(
      model, opt.load_model, optimizer, opt.resume, opt.lr, opt.lr_step)

  wandb.config.update({
  "learning_rate": opt.lr,
  "epochs": opt.batch_size,
  "batch_size": 128,
  "lr_step": opt.lr_step,
  "repesentation": opt.rep,
  "poly_loss": opt.poly_loss,
  "poly_order": opt.poly_order,
  "task": opt.task,
  "nbr_points": opt.nbr_points,
  "dataset": opt.dataset,
  "backbone": opt.arch,
  "model": opt.load_model
  })

  #Size
  SIZE = True
  if SIZE :
    bits = 32
    print("Input size: {:.3f} MB".format(opt.batch_size * opt.input_h*opt.input_w*bits/ 1024**2))

    total_param = 0
    param_size = 0
    buffer_size = 0

    mods = list(model.modules())
    for m in mods:

      #total_param += m.num_parameters()

      for param in m.parameters():
        param_size += param.nelement() * param.element_size()
        total_param += param.nelement()

      for buffer in m.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
        total_param += buffer.nelement()


    print('Num of parameters:', total_param)

    size_all_gb = (param_size + buffer_size) / 1024 /1024 **2

    print('Size model parameters: {:.3f} GB'.format(size_all_gb))

    # input_ = torch.autograd.Variable(torch.FloatTensor(*(opt.batch_size, 3, opt.input_h, opt.input_w)))
    #
    # total_bits = 0
    # with torch.no_grad():
    #   for m in mods:
    #     try:
    #       #print(m)
    #       out = m(input_)
    #
    #       total_bits += np.prod(np.array(out).shape)*bits
    #       input_ = out
    #     except (ValueError,RuntimeError, TypeError, AttributeError):
    #       pass
    #
    #
    # size_all_gb = total_bits*2 / 1024 /1024 **2
    # print('Size intermediate variables: {:.3f} GB'.format(size_all_gb))



  Trainer = train_factory[opt.task]
  trainer = Trainer(opt, model, optimizer)
  trainer.set_device(opt.gpus, opt.chunk_sizes, opt.device)

  print('Setting up data...')
  val_loader = torch.utils.data.DataLoader(
      Dataset(opt, 'val'), 
      batch_size=1, 
      shuffle=False,
      num_workers=1,
      pin_memory=True
  )

  if opt.test:
    _, preds = trainer.val(0, val_loader)
    if opt.dataset == 'cityscapes':
        AP = val_loader.dataset.run_eval(preds, opt.save_dir)
        print('AP: ', AP)
    else:
        val_loader.dataset.run_eval(preds, opt.save_dir)
    return

  train_loader = torch.utils.data.DataLoader(
      Dataset(opt, 'train'), 
      batch_size=opt.batch_size, 
      shuffle=True,
      num_workers=opt.num_workers,
      pin_memory=True,
      drop_last=True
  )

  # logger.write_model(model)
  print('Starting training...')
  best = 1e10
  best_AP = 0
  AP = 0
  for epoch in range(start_epoch + 1, opt.num_epochs + 1):
    mark = epoch if opt.save_all else 'last'
    log_dict_train, _ = trainer.train(epoch, train_loader)
    logger.write('epoch: {} | '.format(epoch))
    for k, v in log_dict_train.items():
      logger.scalar_summary('train_{}'.format(k), v, epoch)
      logger.write('{} {:8f} | '.format(k, v))
    if opt.val_intervals > 0 and epoch % opt.val_intervals == 0:
      save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(mark)), 
                 epoch, model, optimizer)
      with torch.no_grad():
        log_dict_val, preds = trainer.val(epoch, val_loader)
        if True and opt.dataset == 'cityscapes' and (opt.task == 'polydet' or opt.task == 'diskdet'):
            AP = val_loader.dataset.run_eval(preds, opt.save_dir)
            print('AP: ', AP)
      logger.write('\n')
      logger.write('AP: {} | '.format(AP))
      for k, v in log_dict_val.items():
        logger.scalar_summary('val_{}'.format(k), v, epoch)
        logger.write('{} {:8f} | '.format(k, v))
      if True and opt.dataset == 'cityscapes' and (opt.task == 'polydet' or opt.task == 'diskdet'):
        logger.scalar_summary('AP', AP, epoch)
      if True and opt.dataset == 'cityscapes' and (opt.task == 'polydet' or opt.task == 'diskdet'):
          if AP > best_AP:
              best_AP = AP
              save_model(os.path.join(opt.save_dir, 'model_best.pth'),
                         epoch, model)
      else:
          if log_dict_val[opt.metric] < best:
            best = log_dict_val[opt.metric]
            save_model(os.path.join(opt.save_dir, 'model_best.pth'),
                       epoch, model)
    else:
      save_model(os.path.join(opt.save_dir, 'model_last.pth'),
                 epoch, model, optimizer)
    logger.write('\n')
    if epoch in opt.lr_step:
      save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)),
                 epoch, model, optimizer)
      lr = opt.lr * (0.1 ** (opt.lr_step.index(epoch) + 1))
      print('Drop LR to', lr)
      for param_group in optimizer.param_groups:
          param_group['lr'] = lr
  logger.close()

if __name__ == '__main__':
  opt = opts().parse()
  #tracemalloc.start()

  main(opt)

  #snapshot = tracemalloc.take_snapshot()
  #top_stats = snapshot.statistics('traceback')
  #stat = top_stats[0]
  #print("%s memory blocks: %.1f KiB" % (stat.count, stat.size/1024))
  #for line in stat.traceback.format():
  #  print(line)

  """
  tracemalloc.start()
  try:
    main(opt)
  except:
    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics('lineno')

    print('')
    print('[ Top 10 ]')
    for stat in top_stats[:10]:
      print(stat)
  """

