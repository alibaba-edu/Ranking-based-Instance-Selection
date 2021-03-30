
import datetime
import time

import numpy as np
import torch

from ret_benchmark.data.evaluations.eval import AccuracyCalculator
from ret_benchmark.utils.feat_extractor import feat_extractor
from ret_benchmark.utils.metric_logger import MetricLogger
from ret_benchmark.utils.log_info import log_info
from ret_benchmark.modeling.xbm import XBM


def flush_log(writer, iteration):
    for k, v in log_info.items():
        if isinstance(v, np.ndarray):
            writer.add_histogram(k, v, iteration)
        else:
            writer.add_scalar(k, v, iteration)
    for k in list(log_info.keys()):
        del log_info[k]


def do_train(
    cfg,
    model,
    train_loader,
    trainVal_loader,
    val_loader,
    optimizer,
    optimizer_center,
    scheduler,
    scheduler_center,
    criterion,
    criterion_xbm,
    checkpointer,
    writer,
    device,
    arguments,
    logger,
):
    logger.info("Start training")
    meters = MetricLogger(delimiter="  ")
    max_iter = cfg.SOLVER.MAX_ITERS

    best_iteration = -1
    best_mapr = 0

    start_training_time = time.time()
    end = time.time()

    USE_PRISM='PRISM' in cfg.LOSSES.NAME or 'PRISM' in cfg.LOSSES.NAME_XBM_LOSS
    CHECK_NOISE=cfg.NOISE.CHECK_NOISE

    if cfg.XBM.ENABLE:
        logger.info(">>> use XBM")
        xbm = XBM(cfg)
    

    iteration = arguments["iteration"]+1

    _train_loader = iter(train_loader)
    while iteration <= max_iter:
        try:
            images, targets, indices = _train_loader.next()
        except StopIteration:
            _train_loader = iter(train_loader)
            images, targets, indices = _train_loader.next()

        if (iteration % cfg.VALIDATION.VERBOSE == 0 or iteration == max_iter) and iteration > 0:
            model.eval()
            logger.info("Validation")

            if len(val_loader)==1:
                labels = val_loader[0].dataset.label_list
                labels = np.array([int(k) for k in labels])
                feats = feat_extractor(model, val_loader[0], logger=logger)
                feats_gallery=feats
                feats_query=feats
                labels_gallery=labels
                labels_query=labels
            else:
                labels = val_loader[0].dataset.label_list
                labels_gallery = np.array([int(k) for k in labels])
                labels = val_loader[1].dataset.label_list
                labels_query = np.array([int(k) for k in labels])
                feats_gallery = feat_extractor(model, val_loader[0], logger=logger)
                feats_query = feat_extractor(model, val_loader[1], logger=logger)

            ret_metric = AccuracyCalculator(include=("precision_at_1", "mean_average_precision_at_r", "r_precision",'mean_average_precision_at_100'), exclude=())
            ret_metric = ret_metric.get_accuracy(feats_query, feats_gallery, labels_query, labels_gallery, len(val_loader)==1)
            mapr_curr = ret_metric['precision_at_1']
            for k, v in ret_metric.items():
                log_info[f"e_{k}"] = v

            if cfg.SOLVER.LR_SCHEDULAR=='val':
                scheduler.step(log_info['e_precision_at_1'])
            log_info["lr"] = optimizer.param_groups[0]["lr"]
            if mapr_curr > best_mapr:
                best_mapr = mapr_curr
                best_iteration = iteration
                logger.info(f"Best iteration {iteration}: {ret_metric}")
                checkpointer.save("model_best")
            else:
                logger.info(f"Performance at iteration {iteration:06d}: {ret_metric}")
            flush_log(writer, iteration)

            #======training set precision====
            if iteration % cfg.VALIDATION.VERBOSE == 0 and iteration > 0:
                logger.info("Validation on Training set")

                labels = trainVal_loader.dataset.label_list
                labels = np.array([int(k) for k in labels])
                feats = feat_extractor(model, trainVal_loader, logger=logger)
                feats_gallery=feats
                feats_query=feats
                labels_gallery=labels
                labels_query=labels

                ret_metric = AccuracyCalculator(include=("precision_at_1", "mean_average_precision_at_r", "r_precision",'mean_average_precision_at_100'), exclude=())
                ret_metric = ret_metric.get_accuracy(feats_query, feats_gallery, labels_query, labels_gallery, len(trainVal_loader)==1)
                mapr_curr = ret_metric['precision_at_1']
                for k, v in ret_metric.items():
                    log_info[f"t_{k}"] = v
                logger.info(f"Performance on Training set at iteration {iteration:06d}: {ret_metric}")
                flush_log(writer, iteration)
            #========
        model.train()

        data_time = time.time() - end
        iteration = iteration + 1
        arguments["iteration"] = iteration

        images = images.to(device)
        targets = targets.to(device)

        feats = model(images)

        if CHECK_NOISE:
            is_noise=train_loader.dataset.is_noise[indices]
        else:
            is_noise=None
        
        if cfg.XBM.ENABLE and iteration > cfg.XBM.START_ITERATION and (not USE_PRISM):
            xbm.enqueue_dequeue(feats.detach(), targets.detach())
        if (not USE_PRISM) or (iteration <= cfg.XBM.START_ITERATION):
            loss = criterion(feats, targets, feats, targets)
            log_info["batch_loss"] = loss.item()

        if criterion_xbm is not None:
            criterion_=criterion_xbm
        else:
            criterion_=criterion

        if cfg.XBM.ENABLE and iteration > cfg.XBM.START_ITERATION:
            xbm_feats, xbm_targets = xbm.get()

            if USE_PRISM:
                xbm_loss,p_in = criterion_(feats, targets, xbm_feats, xbm_targets,is_noise=is_noise)
                if p_in is not None:
                    if p_in.dtype is torch.bool:
                        enqueue_feats=feats[p_in]
                        enqueue_targets=targets[p_in]
                else:
                    enqueue_feats=feats
                    enqueue_targets=targets
                if (p_in is not None) and (p_in.dtype is torch.bool):
                    selected_feats=feats[p_in]
                    selected_tar=targets[p_in]
                    loss = criterion(selected_feats, selected_tar, selected_feats, selected_tar)
                else:
                    loss = criterion(feats, targets, feats, targets)
                if not isinstance(loss,tuple):
                    log_info["batch_loss"] = loss.item()
                else:
                    loss = 0
            else:
                xbm_loss = criterion(feats, targets, xbm_feats, xbm_targets)
            loss = loss + cfg.XBM.WEIGHT * xbm_loss
        optimizer.zero_grad()
        if optimizer_center is not None:
            optimizer_center.zero_grad()
        if not isinstance(loss,(float,int)):
            loss.backward()
            optimizer.step()
            if optimizer_center is not None:
                optimizer_center.step()

        if cfg.XBM.ENABLE and iteration > cfg.XBM.START_ITERATION and USE_PRISM:
            xbm.enqueue_dequeue(enqueue_feats.detach(), enqueue_targets.detach())

        if cfg.SOLVER.LR_SCHEDULAR!='val':
            scheduler.step()
            if scheduler_center is not None:
                scheduler_center.step()

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time, loss=loss.item())
        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        if iteration % 20 == 0 or iteration == max_iter:
            logger.info(
                meters.delimiter.join(
                    [
                        "eta: {eta}",
                        "iter: {iter}",
                        "{meters}",
                        "lr: {lr:.6f}",
                        "max mem: {memory:.1f} GB",
                    ]
                ).format(
                    eta=eta_string,
                    iter=iteration,
                    meters=str(meters),
                    lr=optimizer.param_groups[0]["lr"],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0 / 1024.0,
                )
            )

            log_info["loss"] = loss.item()
            flush_log(writer, iteration)

        del feats
        del loss

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iter)
        )
    )

    logger.info(f"Best iteration: {best_iteration :06d} | best MAP@R {best_mapr} ")
    writer.close()
