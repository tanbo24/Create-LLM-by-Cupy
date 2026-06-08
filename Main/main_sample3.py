# NCCL communicator を作る前、CuPy/NCCL初期化前に必ず実行
import os

os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ.pop("NCCL_P2P_LEVEL", None)

os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["NCCL_NET"] = "Socket"

# 単一ノード内 bootstrap 用
os.environ["NCCL_SOCKET_IFNAME"] = "lo"

# P2P無効時の単一ノード内通信で重要
os.environ["NCCL_SHM_DISABLE"] = "0"


import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from model.ZeRO3 import TrainMultiGPU


"""
CPU Offloadは固い
ZeRO3 も堅いかな~
いったんはfobaの高速化
現在180秒これをできれば90秒で終わらせたい
batchの最大化を行うのが一番丸い
"""

if __name__ == "__main__":
    padding_id=2

    #model
    vocab_size = 8_0000
    hidden_size=3072
    time_size=1024#これ1000越えはいけるのでは？
    dropout_rate=0#ゼロであればdropoutインスタンスは生成されない
    layer_num=37
    q_head=16
    kv_head=2
    half_float=True

    #optimizer
    lr_max=0.0003
    lr_mini=0.00003
    weight_decay=0.1
    warm_up=1000

    #LossScale
    GradScale=10
    DownRate=0.7
    UpRate=2.0

    #GPU
    use_devices=[0,1,2,3,4,5,6,7,8,9]
    main_device=0

    #etc
    batch_size=12
    max_epoch=1
    accum_step=8
    data_chunk_num=100
    max_grads=1
    save_step=2000
    data_size=10000000

    wand=False
    pa=True
    save=True#保存するか否か
    vali=False
    vali_step=600

    model_dic={
        'vocab_size': vocab_size,
        'hidden_size': hidden_size,
        'dropout_rate': dropout_rate,
        'layer_num': layer_num,
        'time_size': time_size,
        'q_head': q_head,
        'kv_head': kv_head,
        'padding_id': padding_id,
        'half_float': half_float,

        'keep_layer_num': 1,
        'keep_embedding_until_backward': False,

        'ce_chunk_size': 1000,

        'gqa_cache_attn': False,
        'gqa_cache_out_value': False,
        'swiglu_cache_gate': False,

        'flash_backend': 'cupy',
        'flash_block_m': 128,
        'flash_block_n': 128,
        'flash_bwd_block_m': 128,
        'flash_bwd_block_n': 128,
        'flash_bwd_block_d': 128,
    }
    
    GradScale_dic = {
        "StartScale":GradScale,
        "UpRate":UpRate,
        "DownRate":DownRate,
        "Scale_Up_Step":GradScale
    }

    print(f'vocab size{vocab_size}')
    trainer=TrainMultiGPU(use_devices,batch_size
                ,max_epoch,lr_max
                ,wand
                ,max_grads
                ,accum_step,lr_mini
                ,padding_id
                ,save,weight_decay
                ,warm_up,vali
                ,GradScale_dic,model_dic
                ,main_device,half_float,
                save_step,data_chunk_num,
                vali_step)
    
    trainer.fit(data_size)