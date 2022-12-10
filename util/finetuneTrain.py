import os
import argparse
import json
import jsonlines
import numpy as np
import paddle
import yaml
from paddle.io import DataLoader
from paddle.io import DistributedBatchSampler
from yacs.config import CfgNode

from paddlespeech.t2s.datasets.am_batch_fn import fastspeech2_multi_spk_batch_fn
from paddlespeech.t2s.datasets.data_table import DataTable
from paddlespeech.t2s.models.fastspeech2 import FastSpeech2
from visualdl import LogWriter
from paddlespeech.t2s.training.optimizer import build_optimizers
from paddlespeech.t2s.training.seeding import seed_everything
from paddlespeech.t2s.models.fastspeech2 import FastSpeech2Loss
from finetuneTTS import find_max_ckpt, freeze_layer, pretrained_model_dir, config_path, status_change


def save_model(output_checkpoints_dir, step, model, optimizer, only_one=True):
    # 只保存一个模型
    # 删除之前的模型
    if only_one:
        cmd = f"rm -rf {output_checkpoints_dir}/*.pdz"
        os.system(cmd)

    # 保存模型
    save_path = f"{output_checkpoints_dir}/snapshot_iter_{step}.pdz"
    archive = {
        "epoch": 100,
        "iteration": step,
        "main_params": model.state_dict(),
        "main_optimizer": optimizer.state_dict()
    }
    paddle.save(archive, save_path)

def finetune_train(dump_dir, output_dir, max_step, batch_size=None, learning_rate=None):
    exp_path = os.path.dirname(output_dir)
    train_status_json = os.path.join(output_dir, "train_status.json")
    ft_status_json = os.path.join(exp_path, "finetune_status.json")
    status_change(ft_status_json, True)

    # train_status_json 存在则删除
    if os.path.exists(train_status_json):
        cmd = f"rm {train_status_json}"
        os.system(cmd)
    
    # save_step：提高训练速度和空间，最多只保存10次模型
    max_save_cnt = 10
    save_step_index = int(max_step / max_save_cnt)
    if  save_step_index > 200:
        save_step_index = 200
    
    # 读取默认的yaml文件
    with open(config_path) as f:
        finetune_config = yaml.safe_load(f)
    # 多少个 step 保存一次
    # 1. 自动调整 batch ，通过 dump/train里面文件数量判断
    train_data_dir = os.path.join(dump_dir, "train/norm/data_speech")
    file_num = len([filename for filename in os.listdir(train_data_dir) if filename.endswith(".npy")])
    if not batch_size:    
        if file_num <= 32:
            batch_size = file_num
            finetune_config['batch_size'] = batch_size
        else:
            finetune_config['batch_size'] = 32
    else:
        if file_num <= batch_size:
            batch_size = file_num
        finetune_config['batch_size'] = batch_size

    # 2. 支持调整 learning_rate
    if learning_rate:
        finetune_config['learning_rate'] = learning_rate

    # 重新生成这次试验需要的yaml文件
    new_config_path = os.path.join(dump_dir, "finetune.yaml")
    with open(new_config_path, "w", encoding="utf8") as f:
        yaml.dump(finetune_config, f)


    train_metadata = f"{dump_dir}/train/norm/metadata.jsonl"
    # dev_metadata = f"{dump_dir}/dev/norm/metadata.jsonl"
    speaker_dict = f"{dump_dir}/speaker_id_map.txt"
    phones_dict = f"{dump_dir}/phone_id_map.txt"
    num_workers = 2

    default_config_file = f"{pretrained_model_dir}/default.yaml"
    with open(default_config_file) as f:
        config = CfgNode(yaml.safe_load(f))

    # 冻结神经层
    with open(new_config_path) as f2:
            finetune_config = CfgNode(yaml.safe_load(f2))
    config.batch_size = batch_size = finetune_config.batch_size if finetune_config.batch_size > 0 else config.batch_size
    config.optimizer.learning_rate = finetune_config.learning_rate if finetune_config.learning_rate > 0 else config.optimizer.learning_rate
    config.num_snapshots = finetune_config.num_snapshots if finetune_config.num_snapshots > 0 else config.num_snapshots
    frozen_layers = finetune_config.frozen_layers

    fields = [
            "text", "text_lengths", "speech", "speech_lengths", "durations",
            "pitch", "energy"
        ]
    converters = {"speech": np.load, "pitch": np.load, "energy": np.load}
    collate_fn = fastspeech2_multi_spk_batch_fn
    with open(speaker_dict, 'rt') as f:
        spk_id = [line.strip().split() for line in f.readlines()]
    spk_num = len(spk_id)
    fields += ["spk_id"]

    with jsonlines.open(train_metadata, 'r') as reader:
        train_metadata = list(reader)
    train_dataset = DataTable(
        data=train_metadata,
        fields=fields,
        converters=converters, )
    # with jsonlines.open(dev_metadata, 'r') as reader:
    #     dev_metadata = list(reader)
    # dev_dataset = DataTable(
    #     data=dev_metadata,
    #     fields=fields,
    #     converters=converters, )
    train_batch_size = min(len(train_metadata), batch_size)
    train_sampler = DistributedBatchSampler(
            train_dataset,
            batch_size=train_batch_size,
            shuffle=True,
            drop_last=True)
    train_dataloader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        collate_fn=collate_fn,
        num_workers=num_workers)

    # dev_dataloader = DataLoader(
    #     dev_dataset,
    #     shuffle=False,
    #     drop_last=False,
    #     batch_size=batch_size,   # 输入 batch size 大小
    #     collate_fn=collate_fn,
    #     num_workers=num_workers)
    print("dataloaders done!")

    with open(phones_dict, "r") as f:
        phn_id = [line.strip().split() for line in f.readlines()]
    vocab_size = len(phn_id)
    print("vocab_size:", vocab_size)
    odim = config.n_mels
    
    seed_everything(config.seed)

    # 初始化模型，优化器，损失函数
    model = FastSpeech2(
        idim=vocab_size, odim=odim, spk_num=spk_num, **config["model"])
    optimizer = build_optimizers(model, **config["optimizer"])
    use_masking=config["updater"]['use_masking']
    use_weighted_masking=False
    criterion = FastSpeech2Loss(use_masking=use_masking, use_weighted_masking=use_weighted_masking)
    
    # 检查之前是否有模型是否存在
    output_checkpoints_dir = os.path.join(output_dir, "checkpoints")
    if os.path.exists(output_checkpoints_dir):
        ckpt = find_max_ckpt(output_checkpoints_dir)
        if ckpt != 99200 and ckpt != 0:
            use_pretrain_model = os.path.join(output_checkpoints_dir, f"snapshot_iter_{ckpt}.pdz")
            start_step = ckpt
        else:
            # 是默认的预训练模型
            cmd = f"rm -rf {output_checkpoints_dir}/*.pdz"
            os.system(cmd)
            use_pretrain_model = os.path.join(pretrained_model_dir, "snapshot_iter_99200.pdz")
            start_step = 0
    else:
        os.makedirs(output_checkpoints_dir, exist_ok=True)
        use_pretrain_model = os.path.join(pretrained_model_dir, "snapshot_iter_99200.pdz")
        start_step = 0
    
    # 加载预训练模型
    archive = paddle.load(use_pretrain_model)
    model.set_state_dict(archive['main_params'])
    optimizer.set_state_dict(archive['main_optimizer'])
    

    # 冻结层
    if frozen_layers != []:
        freeze_layer(model, frozen_layers)
    
    # 开始训练
    if start_step >= max_step:
        print(f"当前模型步数： {start_step} 大于最大训练步数 {max_step}")
        status_change(ft_status_json, False)
        if os.path.exists(train_status_json):
            os.remove(train_status_json)
        return
        
    step = start_step

    # 进入训练流程
    writer = LogWriter(logdir=f"{output_dir}/log") 
    while True:
        for batch_id, batch in enumerate(train_dataloader()):
            #前向计算的过程
            losses_dict = {}
            # spk_id!=None in multiple spk fastspeech2 
            spk_id = batch["spk_id"] if "spk_id" in batch else None
            spk_emb = batch["spk_emb"] if "spk_emb" in batch else None
            # No explicit speaker identifier labels are used during voice cloning training.
            if spk_emb is not None:
                spk_id = None

            # result = model(
            #     text=batch["text"],
            #     text_lengths=batch["text_lengths"],
            #     speech=batch["speech"],
            #     speech_lengths=batch["speech_lengths"],
            #     durations=batch["durations"],
            #     pitch=batch["pitch"],
            #     energy=batch["energy"],
            #     spk_id=spk_id,
            #     spk_emb=spk_emb)
            # print(f"报错result: {len(result)}")
            # print(result)

            before_outs, after_outs, d_outs, p_outs, e_outs, ys, olens, _ = model(
                text=batch["text"],
                text_lengths=batch["text_lengths"],
                speech=batch["speech"],
                speech_lengths=batch["speech_lengths"],
                durations=batch["durations"],
                pitch=batch["pitch"],
                energy=batch["energy"],
                spk_id=spk_id,
                spk_emb=spk_emb)
            
            l1_loss, duration_loss, pitch_loss, energy_loss, _ = criterion(
                after_outs=after_outs,
                before_outs=before_outs,
                d_outs=d_outs,
                p_outs=p_outs,
                e_outs=e_outs,
                ys=ys,
                ds=batch["durations"],
                ps=batch["pitch"],
                es=batch["energy"],
                ilens=batch["text_lengths"],
                olens=olens)

            loss = l1_loss + duration_loss + pitch_loss + energy_loss

            # optimizer = self.optimizer
            optimizer.clear_grad()
            loss.backward()
            optimizer.step()
            
            step += 1
            
            # 控制台可视化
            losses_dict["l1_loss"] = float(l1_loss)
            losses_dict["duration_loss"] = float(duration_loss)
            losses_dict["pitch_loss"] = float(pitch_loss)
            losses_dict["energy_loss"] = float(energy_loss)
            losses_dict["loss"] = float(loss)
            msg = f"Step: {step}, Max_step: {max_step}, " + ', '.join('{}: {:>.6f}'.format(k, v)
                                for k, v in losses_dict.items())
            print(msg)
            
            # vdl 可视化
            writer.add_scalar(tag="train/loss", step=step, value=float(loss))
            writer.add_scalar(tag="train/l1_loss", step=step, value=float(l1_loss))
            writer.add_scalar(tag="train/duration_loss", step=step, value=float(duration_loss))
            writer.add_scalar(tag="train/pitch_loss", step=step, value=float(pitch_loss))
            writer.add_scalar(tag="train/energy_loss", step=step, value=float(energy_loss))

            # streamlit 通过这个文件查询训练状态
            with open(train_status_json, "w", encoding="utf8") as f:
                status_dict = {
                    "step": step,
                    "max_step": max_step,
                    "loss": round(float(loss), 6)
                }
                json.dump(status_dict, f, indent=3)
            
            # 训练中
            status_change(ft_status_json, True)
            
            if step % save_step_index == 0:
                # 保存模型
                save_model(output_checkpoints_dir, step, model, optimizer)
            
            # 模型训练结束
            if step >= max_step:
                # 保存模型
                save_model(output_checkpoints_dir, step, model, optimizer)
                # 训练结束删除 json
                if os.path.exists(train_status_json):
                    os.remove(train_status_json)
                # 恢复训练状态
                status_change(ft_status_json, False)
                return


if __name__ == '__main__':
    # parse config and args
    parser = argparse.ArgumentParser(
        description="Preprocess audio and then extract features.")

    parser.add_argument(
        "--dump_dir",
        type=str,
        default="./dump",
        help="directory to save feature files and metadata.")

    parser.add_argument(
        "--output_dir",
        type=str,
        default="./exp/default/",
        help="directory to save finetune model.")

    parser.add_argument(
        "--max_step", type=int, default=10, help="train max step")
    parser.add_argument(
        "--batch_size", type=int, default=None, help="batch size to train")
    parser.add_argument(
        "--learning_rate", type=float, default=None, help="learning rate for train")
    
    parser.add_argument(
        "--finetune_config",
        type=str,
        default="./finetune.yaml",
        help="Path to finetune config file")

    args = parser.parse_args()

    exp_path = os.path.dirname(args.output_dir)
    ft_status_json = os.path.join(exp_path, "finetune_status.json")
    status_change(ft_status_json, True)
    # try:
    finetune_train(args.dump_dir, args.output_dir, args.max_step, batch_size=args.batch_size, learning_rate=args.learning_rate)
    # except:
        # status_change(ft_status_json, False)
    




