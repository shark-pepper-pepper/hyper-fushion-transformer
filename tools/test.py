import time
import copy
import datetime
import os
import sys
sys.path.append('../')
sys.path.insert(0,"the absolutely path of the file or the folder")

cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
sys.path.append(root_path)
# from .. import segmentron
import logging
import torch
import torch.nn as nn
import torch.utils.data as data

from torchvision import transforms
from segmentron.data.dataloader import get_segmentation_dataset
from segmentron.models.model_zoo import get_segmentation_model
from segmentron.solver.loss import get_segmentation_loss, get_hadamard_loss
from segmentron.solver.optimizer import get_optimizer
from segmentron.solver.lr_scheduler import get_scheduler
from segmentron.utils.distributed import *
from segmentron.utils.score import SegmentationMetric
from segmentron.utils.filesystem import save_checkpoint
from segmentron.utils.options import parse_args
from segmentron.utils.default_setup import default_setup
from segmentron.utils.visualize import show_flops_params, get_color_pallete
from segmentron.config import cfg

try:
    import apex
except:
    print('apex is not installed')

class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)
        self.use_fp16 = cfg.TRAIN.APEX

        # image transform
        input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(cfg.DATASET.MEAN, cfg.DATASET.STD),
        ])
        # dataset and dataloader
        data_kwargs = {'transform': input_transform,
                       'base_size': cfg.TRAIN.BASE_SIZE,
                       'crop_size': cfg.TRAIN.CROP_SIZE}

        data_kwargs_testval = {'transform': input_transform,
                       'base_size': cfg.TRAIN.BASE_SIZE,
                       'crop_size': cfg.TEST.CROP_SIZE}

        test_dataset = get_segmentation_dataset(cfg.DATASET.NAME, split='test', mode='testval', **data_kwargs_testval)

        self.classes = test_dataset.classes

        test_sampler = make_data_sampler(test_dataset, False, args.distributed)
        test_batch_sampler = make_batch_data_sampler(test_sampler, cfg.TEST.BATCH_SIZE, drop_last=False)

        self.test_loader = data.DataLoader(dataset=test_dataset,
                                          batch_sampler=test_batch_sampler,
                                          num_workers=cfg.DATASET.WORKERS,
                                          pin_memory=True)

        # create network
        self.model = get_segmentation_model().to(self.device)

        # print params and flops
        if get_rank() == 0:
            try:
                show_flops_params(copy.deepcopy(self.model), args.device)
            except Exception as e:
                logging.warning('get flops and params error: {}'.format(e))
        if cfg.MODEL.BN_TYPE not in ['BN']:
            logging.info('Batch norm type is {}, convert_sync_batchnorm is not effective'.format(cfg.MODEL.BN_TYPE))
        elif args.distributed and cfg.TRAIN.SYNC_BATCH_NORM:
            self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            logging.info('SyncBatchNorm is effective!')
        else:
            logging.info('Not use SyncBatchNorm!')
        self.criterion = get_segmentation_loss(cfg.MODEL.MODEL_NAME, use_ohem=cfg.SOLVER.OHEM,
                                               aux=cfg.SOLVER.AUX, aux_weight=cfg.SOLVER.AUX_WEIGHT,
                                               ignore_index=cfg.DATASET.IGNORE_INDEX).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), betas=(0.9, 0.999), eps=1e-08, lr=0.0001, weight_decay=0.0001)
        # apex
        if self.use_fp16:
            self.model, self.optimizer = apex.amp.initialize(self.model.cuda(), self.optimizer, opt_level="O1")
            logging.info('**** Initializing mixed precision done. ****')

        # resume checkpoint if needed
        self.start_epoch = 0
        if args.resume and os.path.isfile(args.resume):
            name, ext = os.path.splitext(args.resume)
            assert ext == '.pkl' or '.pth', 'Sorry only .pth and .pkl files supported.'
            logging.info('Resuming training, loading {}...'.format(args.resume))
            resume_sate = torch.load(args.resume)
            self.model.load_state_dict(resume_sate['state_dict'])
            self.start_epoch = resume_sate['epoch']
            logging.info('resume train from epoch: {}'.format(self.start_epoch))
            if resume_sate['optimizer'] is not None and resume_sate['lr_scheduler'] is not None:
                logging.info('resume optimizer and lr scheduler from resume state..')
                self.optimizer.load_state_dict(resume_sate['optimizer'])
                self.lr_scheduler.load_state_dict(resume_sate['lr_scheduler'])

        if args.distributed:
            self.model = nn.parallel.DistributedDataParallel(self.model,
                                                             device_ids=[args.local_rank],
                                                             output_device=args.local_rank,
                                                             find_unused_parameters=True)

    def test(self, vis=False):
        vis = True
        if self.args.distributed:
            model = self.model.module
        else:
            model = self.model
        torch.cuda.empty_cache()
        model.eval()
        for i, (image, target, filename) in enumerate(self.test_loader):
            image = image.to(self.device)
            target = target.to(self.device)

            with torch.no_grad():
                if cfg.DATASET.MODE == 'test' or cfg.TEST.CROP_SIZE is None:
                    output = model(image)[0][0]
                else:
                    size = image.size()[2:]
                    assert cfg.TEST.CROP_SIZE[0] == size[0]
                    assert cfg.TEST.CROP_SIZE[1] == size[1]
                    output = model(image)[0][0]

            if vis:
                vis_pred = output[0].permute(1,2,0).argmax(-1).data.cpu().numpy()
                vis_pred = get_color_pallete(vis_pred, dataset='trans10kv2')
                # save_path = os.path.join(cfg.TRAIN.MODEL_SAVE_DIR, 'vis')
                save_path = 'datasets/transparent/Trans10K_cls12/test/results'
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                # print(save_path + str(i) +'.png')
                vis_pred.save(os.path.join(save_path, filename[0][:-4]+'.png'))
                print("[VIS TEST] Sample: {:d}".format(i+1))
                continue

            self.metric.update(output, target)


if __name__ == '__main__':
    args = parse_args()
    # get config
    cfg.update_from_file(args.config_file)
    cfg.update_from_list(args.opts)
    cfg.PHASE = 'test'
    cfg.ROOT_PATH = root_path
    cfg.check_and_freeze()

    # setup python train environment, logger, seed..
    default_setup(args)

    # create a trainer and start train
    trainer = Trainer(args)
    trainer.test(args.vis)
