import os

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from model.ZeRO2_5 import TrainMultiGPU

os.environ.setdefault("NCCL_P2P_DISABLE", "1")
os.environ.setdefault("NCCL_IB_DISABLE", "1")
os.environ.setdefault("NCCL_SOCKET_IFNAME", "lo") # 同一ノード内通信を想定


"""
150日まで短縮できている
CPU Offloadは固い
ZeRO3 も堅いかな~
いったんはfobaの高速化
backwardの改善を行う
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
    batch_size=7
    max_epoch=1
    accum_step=14
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
        'vocab_size':vocab_size,
        'hidden_size':hidden_size,
        'dropout_rate':dropout_rate,
        'layer_num':layer_num,
        'time_size':time_size,
        'q_head':q_head,
        'kv_head':kv_head,
        'padding_id':padding_id,
        'half_float':half_float,

        # forward/backward高速化: ZeRO3とモデルサイズは維持
        # 2にするとblock parameter gatherの同期回数がほぼ半減。OOMするなら1へ戻す。
        'keep_layer_num': 1,
        # forwardでgatherした共有embeddingをbackwardまで保持し、巨大embeddingの再broadcastを削る。
        'keep_embedding_until_backward': False,
        # GQA backwardで out_value=attn@V を再計算しない。activationメモリ+約1GB程度。
        'gqa_cache_out_value': False,

        # ZeRO2.5: この割合ぶんのパラメータを各GPUに常駐レプリカとして保持する。
        # 0.0 => 従来ZeRO3 / 0.15〜0.30あたりから試すのがおすすめ。
        'zero25_replica_ratio': 0,
        # largest_first: 通信削減効果が大きい大きなtensorから常駐化。
        # embedding_firstにすると共有Embeddingを優先して常駐化。
        'zero25_replica_policy': 'largest_first',
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
                vali_step,
                optimizer_device='cpu',  # GPUに乗るなら 'gpu' に変更
                zero25_replica_ratio=model_dic['zero25_replica_ratio'],
                zero25_replica_policy=model_dic['zero25_replica_policy'])
    
    trainer.fit(data_size)