import pickle,gzip,gc,os,sys,multiprocessing,wandb,subprocess,json,time
from model.model import Cupa_ver4
from model.functions import ScaleGrad, print_memory_status
from model.optimizer import ZeRO_Adam_Offload, ZeRO_Adam_ShardedGPU
import cupy as cp
import numpy as np
from cupy.cuda import nccl
from datetime import datetime
from pathlib import Path
from typing import Iterator, List, Optional

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

model_f16_drive_path = "gdrive:model_save/flat_params_f16.pkl.gz"
optimizer_drive_path = "gdrive:model_save/flat_params_f16.pkl.gz"
REMOTE_NAME = "gdrive"
REMOTE_DIR = f"{REMOTE_NAME}:sample_dataset1024tokens80kvocab"


def data_shuffle(Xs,Ts):
    # CPU上のPython listをシャッフルするだけなので、CuPyを使わない。
    # cp.random.permutation + Python int変換はGPU同期が大量に起きて遅い。
    idx = np.random.permutation(len(Xs))
    xdata = [Xs[int(i)] for i in idx]
    tdata = [Ts[int(i)] for i in idx]
    return xdata,tdata



def get_batch(x, t, batch_size, time_size,time_idx):
    data_size = len(x)
    jump = data_size // batch_size
    offsets = [i * jump for i in range(batch_size)]  # バッチの各サンプルの読み込み開始位置

    for time in range(time_size):
        batch_x=[]
        batch_t=[]
        batch_x_pa=[]
        batch_t_pa=[]
        for i, offset in enumerate(offsets):
            batch_x.append(x[(offset + time_idx) % data_size][0])
            batch_t.append(t[(offset + time_idx) % data_size][0])
            batch_x_pa.append(x[(offset + time_idx) % data_size][1])
            batch_t_pa.append(t[(offset + time_idx) % data_size][1])
        time_idx += 1

    return batch_x, batch_t,batch_x_pa, batch_t_pa,time_idx


def padding_proceccing(xdata, tdata, xpadding, tpadding, padding_ID, batch_size):
    xdata = cp.asarray(xdata, dtype=cp.int32)
    tdata = cp.asarray(tdata, dtype=cp.int32)
    xpadding = cp.asarray(xpadding, dtype=cp.int32) - 1

    rows = cp.arange(batch_size)
    xdata[rows, xpadding] = padding_ID

    tdata = cp.concatenate(
        (tdata[:, 1:], cp.full((batch_size, 1), padding_ID, dtype=cp.int32)),
        axis=1,
    )

    return xdata, tdata, xpadding

def padding_proceccing_vali(xdata, tdata, xpadding, tpadding, padding_ID, batch_size):
    
    xpadding = [x - 1 for x in xpadding]

    new_xdata = []
    for i in range(batch_size):
        row = list(xdata[i]) 
        for j in range(len(row) - 1, -1, -1):
            if row[j] != padding_ID:
                row[j] = padding_ID
                break
        new_xdata.append(row)
    
    xdata = new_xdata

    new_tdata = []
    for i in range(batch_size):
        row = tdata[i]
        new_row = row[1:] + [padding_ID]
        new_tdata.append(new_row)
    
    tdata = new_tdata

    return xdata, tdata, xpadding




#--------------------------------------------------------------------------------------
# --- メインワーカープロセス ---------------------------------------------------------
def training_worker_nccl_stream(rank, world_size, comm_id, all_args, batch_size, padding_ID):
    """
    高速化版 training worker (pinned/staging + async H2D + stream overlap)
    - data_chunk: 各プロセスに渡された分割データ (X, T, X_pa, T_pa)
    """
    device_id = all_args['use_devices'][rank]
    is_main_device = (rank == 0)
    padding_id=all_args['model_dic']['padding_id']

    # デバイス設定とNCCLコミュニケータの初期化
    with cp.cuda.Device(device_id):
        print(f"[Rank {rank}/{world_size}] Initializing on device {device_id}...")

        # モデルを各GPUで初期化
        if all_args['train_continue_path'] is not None:
            model = Cupa_ver4(all_args['model_dic'],all_args['train_continue_path'])
        else:
            model = Cupa_ver4(all_args['model_dic'])
        #ここでモデルサイズの計算
        model_size=0
        for param in model.params:
            model_size+=param.size
        
        if is_main_device:
            if all_args['wand']:
                if all_args['train_continue_path'] is not None:
                    wandb.init(project="zero-scrach-1B LLM", config={
                    # モデルのハイパーパラメーター
                    "Max Learning Rate": all_args['lr_max'],
                    "Min learning Rate":all_args['lr_mini'],
                    "Max Epoch": all_args['max_epoch'],
                    "Batch Size":batch_size,
                    "Check Point":all_args['save_step'],
                    "Save": all_args['save'],
                    'Init GradScale':all_args['ScaleGrad_dic']['StartScale'],
                    'GradScale UpRate':all_args['ScaleGrad_dic']['UpRate'],
                    'GradScale DownRate':all_args['ScaleGrad_dic']['DownRate'],
                    'Warm Up':all_args['warm_up'],
                    'Weight Decay':all_args['weight_decay'],
                    'Clip Max Grads':all_args['max_grads'],
                    'Accumulate Batch Size':batch_size*all_args['accum_step']*world_size,
                    'Model Size':model_size
                    })
                else:
                    wandb.init(project="zero-scrach-1B LLM", config={
                    # モデルのハイパーパラメーター
                    "Embedding ID": all_args['model_dic']['padding_id'],
                    "Vocab Size": all_args['model_dic']['vocab_size'],
                    "Hidden Size": all_args['model_dic']['hidden_size'],
                    "Time Size":all_args['model_dic']['time_size'],
                    "Dropout Rate": all_args['model_dic']['dropout_rate'],
                    "Layer Num": all_args['model_dic']['layer_num'],
                    "Q Head Num": all_args['model_dic']['q_head'],
                    "KV Head Num": all_args['model_dic']['kv_head'],
                    "Max Learning Rate": all_args['lr_max'],
                    "Min learning Rate":all_args['lr_mini'],
                    "Max Epoch": all_args['max_epoch'],
                    "Batch Size":batch_size,
                    "Check Point":all_args['save_step'],
                    "Save": all_args['save'],
                    'Init GradScale':all_args['ScaleGrad_dic']['StartScale'],
                    'GradScale UpRate':all_args['ScaleGrad_dic']['UpRate'],
                    'GradScale DownRate':all_args['ScaleGrad_dic']['DownRate'],
                    'Warm Up':all_args['warm_up'],
                    'Weight Decay':all_args['weight_decay'],
                    'Clip Max Grads':all_args['max_grads'],
                    'Accumulate Batch Size':batch_size*all_args['accum_step']*world_size,
                    'Model Size':model_size
                    })
            print(f'モデルのパラメーターサイズ : {model_size}')
        comm = nccl.NcclCommunicator(world_size, comm_id, rank)
        h2d_stream = cp.cuda.Stream(non_blocking=True)
        #================================================================================
        if is_main_device:
            print_memory_status("model終了後")
        #======================================================================================

        Scaling=ScaleGrad(all_args['ScaleGrad_dic'])
        #ここが問題
        #================================================================================
        if is_main_device:
            print_memory_status("mana終了後")
        #======================================================================================
        # manager 作成後
        manager = MemoryZeRO2_5Manager(
            model, world_size, comm, rank, h2d_stream,
            replica_ratio=all_args.get('zero25_replica_ratio', all_args['model_dic'].get('zero25_replica_ratio', 0.0)),
            replica_policy=all_args.get('zero25_replica_policy', all_args['model_dic'].get('zero25_replica_policy', 'largest_first')),
        )

        # これは不要。model.params は manager 内で None にされている
        # model.link_params(model.params)

        my_rank_shard_f16 = manager.get_param_shard_view(rank)
        # replica param自体は全rankがfullで持つが、optimizer state/updateはrankごとのsliceだけ持つ。
        my_replica_opt_shard_f16 = manager.get_replica_optimizer_shard_view()

        optimizer_cls = (
            ZeRO_Adam_ShardedGPU
            if all_args.get('optimizer_device', 'gpu') == 'gpu'
            else ZeRO_Adam_Offload
        )

        optimizer = None
        if my_rank_shard_f16.size > 0:
            optimizer = optimizer_cls(
                my_rank_shard_f16,
                all_args['lr_max'],
                all_args['max_epoch'] * all_args['max_iters'],
                all_args['lr_mini'],
                all_args['weight_decay'],
                all_args['warm_up'],
                all_args['wand'],
            )

        replica_optimizer = None
        if my_replica_opt_shard_f16.size > 0:
            # ZeRO2.5改善版:
            #   replica_param_f16 は全rankがfullで保持してforward/backwardのbroadcastを削る。
            #   ただしAdam state/updateはfullではなく、このrank担当sliceだけにする。
            replica_optimizer = optimizer_cls(
                my_replica_opt_shard_f16,
                all_args['lr_max'],
                all_args['max_epoch'] * all_args['max_iters'],
                all_args['lr_mini'],
                all_args['weight_decay'],
                all_args['warm_up'],
                False,
            )

        if is_main_device:
            print(f"optimizer_device={all_args.get('optimizer_device', 'gpu')}")
            print(f"zero25_replica_ratio={manager.replica_ratio:.3f}, policy={manager.replica_policy}")

        # manager.flat_params_f16 の初期 broadcast ブロックは削除
        # 初期同期は MemoryZeRO3Manager.__init__ 内の flat_tmp broadcast で既にやっている
        #================================================================================
        if is_main_device:
            print_memory_status("opti終了後")
        #======================================================================================


        # --- メイン学習ループ ---
        max_iters = all_args['max_iters']
        GradScale=all_args['ScaleGrad_dic']['StartScale']

        count=0
        chunk_num=0
        
        # 通信の完了を待機
        h2d_stream.synchronize()



        X=[]
        #================================================================================
        if is_main_device:
            print_memory_status('学習前')
            sum_by = manager.param_shard_f16.nbytes
            rep_by = manager.replica_param_f16.nbytes
            rep_opt_by = manager.replica_param_opt_shard_f16.nbytes
            print(
                f"Shard Size: {sum_by / 1024**3:.4f} GB / "
                f"Replica Size: {rep_by / 1024**3:.4f} GB / "
                f"ReplicaOptShard Size: {rep_opt_by / 1024**3:.4f} GB"
            )
        #======================================================================================
        for epoch in range(all_args['max_epoch']):
            iters=0
            chunk_num=0
            count=0
            while True:
                manager.zero_grad()
                iters+=1
                if iters==(max_iters+1):
                    break
                
                if iters==1 or  count>=(len(X)//batch_size):
                    count=0
                    if iters!=1 and count>=(len(X)//batch_size):
                        del X,T
                        gc.collect()
                        cp.get_default_memory_pool().free_all_blocks()

                    data = load_input_ids_from_drive(max_samples=100000)
                
                    X=[]
                    T=[]
                    for i in data:
                        if padding_id in i:
                            X.append((i,i.index(padding_id)))
                        else:
                            X.append((i,len(i)))

                    T=X
                    X,T=data_shuffle(X,T)


                    time_idx = 0  # 新しいデータチャンクのためにtime_idxをリセット

                #ここでvalidation用のデータを用意これは一貫して同じデータにする。
                if all_args['vali'] and epoch == 0 and chunk_num == 1:
                    vali_idx = 0
                    validation_box = []
                    x_vali = X[:batch_size * all_args['accum_step']]
                    y_vali = T[:batch_size * all_args['accum_step']]
                    X = X[batch_size * all_args['accum_step']:]
                    T = T[batch_size * all_args['accum_step']:]

                    for i in range(all_args['accum_step']):

                        x_vali_min, y_vali_min, x_vali_pa, y_vali_pa, vali_idx = get_batch(x_vali, y_vali, batch_size, 1, vali_idx)
                
                        x_v, y_v, p_v = padding_proceccing_vali(x_vali_min, y_vali_min, x_vali_pa, y_vali_pa, padding_ID, batch_size)
                        
                        validation_box.append((
                            np.array(x_v, dtype=np.int32), 
                            np.array(y_v, dtype=np.int32), 
                            np.array(p_v, dtype=np.int32)
                        ))

                #param_norm = manager.calculate_global_param_norm(comm, h2d_stream, rank)

                loss_accum_gpu = cp.array(0.0, dtype=cp.float32)

                for _ in range(all_args['accum_step']):
                    count += 1
                    batch_x, batch_t, batch_x_pa, batch_t_pa, time_idx = get_batch(
                        X, T, batch_size, 1, time_idx
                    )
                    bx, bt, bpa = padding_proceccing(
                        batch_x, batch_t, batch_x_pa, batch_t_pa, padding_id, batch_size
                    )

                    with h2d_stream:

                        start = time.perf_counter()
                        loss_i = model.forward(bx, bt, bpa, manager)
                        end = time.perf_counter()
                        if is_main_device:
                            print(f"fo処理時間: {end - start:.6f} 秒")

                        start = time.perf_counter()
                        model.backward(GradScale, manager)
                        end = time.perf_counter()
                        if is_main_device:
                            print(f"ba処理時間: {end - start:.6f} 秒")

                        loss_accum_gpu += loss_i.astype(cp.float32)

                        manager.flush_buckets()
                        manager.synchronize()

                    h2d_stream.synchronize()

                avg_loss = float(loss_accum_gpu.get()) / float(all_args['accum_step'])



                with h2d_stream:
                    manager.flush_buckets()

                    manager.synchronize()
                    # ZeRO2.5改善版:
                    # replica paramはfull常駐のまま、gradはallReduce後にrank担当sliceだけを
                    # replica_grad_opt_shard_f16へ切り出してoptimizerに渡す。
                    manager.prepare_replicated_grads_for_sharded_optimizer()

                    final_base_scale = 1.0 / (world_size * all_args['accum_step'] * GradScale)
                    
                    if manager.check_nan_inf(manager.grad_shard_f16, comm, h2d_stream):
                        GradScale = Scaling.UpdateScale(True) # NaN検知時はスケールダウンしてスキップ
                        if is_main_device:
                            print(f"Iter {iters}: Update SKIPPED due to NaN/Inf. New Scale: {GradScale}")
                        continue
                    
                    # グローバルノルム計算とクリッピング
                    # replicated gradは全rankで同じなので、world_size回数えないようmanager内で補正する。
                    g_norm = manager.calculate_global_grad_norm(manager.grad_shard_f16, comm, h2d_stream)
                    actual_norm = g_norm * final_base_scale
                    clip_coef = min(1.0, all_args['max_grads'] / (actual_norm + 1e-6))
                    
                    step_scale = final_base_scale * clip_coef

                    start = time.perf_counter()

                    if optimizer is not None:
                        updated_shard_f16 = optimizer.update(
                            manager.grad_shard_f16,
                            step_scale,
                            is_main_device,
                            out_shard_f16=manager.param_shard_f16,
                            stream=h2d_stream,
                        )
                        if updated_shard_f16 is not manager.param_shard_f16:
                            manager.set_param_shard(updated_shard_f16)

                    if replica_optimizer is not None:
                        updated_replica_shard_f16 = replica_optimizer.update(
                            manager.replica_grad_opt_shard_f16,
                            step_scale,
                            False,
                            out_shard_f16=manager.replica_param_opt_shard_f16,
                            stream=h2d_stream,
                        )
                        if updated_replica_shard_f16 is not manager.replica_param_opt_shard_f16:
                            manager.set_replica_optimizer_shard(updated_replica_shard_f16)

                        # このrankが更新したreplica sliceをallGatherして、
                        # 次stepのforward/backward用full replica_param_f16を全rankで復元する。
                        manager.allgather_replicated_params_from_optimizer_shards()

                    # GPU optimizer is asynchronous, so synchronize before timing.
                    h2d_stream.synchronize()
                    end = time.perf_counter()
                    if is_main_device:
                        print(f"opti処理時間: {end - start:.6f} 秒")
                h2d_stream.synchronize()
                    
                # 同期終了。この時点で model.params[i] はすべて最新状態に自動更新されている
                h2d_stream.synchronize()
                GradScale=Scaling.UpdateScale(False)

                if all_args['vali']:
                    if (iters - 1) % all_args['vali_step'] == 0:
                        vali_loss_arr = []
                        
                        # モデルを評価モードにする（もしあれば）
                        # model.eval() 
                        
                        for x_cpu, y_cpu, p_cpu in validation_box:
                            # 【転送】計算する直前に CPU -> GPU へ転送
                            x_gpu = cp.array(x_cpu, dtype=cp.int32)
                            y_gpu = cp.array(y_cpu, dtype=cp.int32)
                            p_gpu = cp.array(p_cpu, dtype=cp.int32)

                            # 順伝播（勾配計算を無効化する設定があれば追加）
                            loss_vali = model.predict(x_gpu, y_gpu, p_gpu)
                            
                            # 結果をCPUに戻して保存
                            vali_loss_arr.append(float(loss_vali))
                            
                            # 【解放】GPUメモリ上のバッチデータを即座に削除
                            del x_gpu, y_gpu, p_gpu
                            
                        # 通信処理 (allReduce) は既存のロジックを維持
                        loss_arr = cp.array(vali_loss_arr).astype(cp.float16)
                            
                        with h2d_stream:
                            # NCCL 型を決定
                            nccl_dtype = nccl.NCCL_FLOAT16

                            loss_arr = cp.ascontiguousarray(loss_arr)

                            # ポインタやサイズを取得
                            send_ptr   = int(loss_arr.data.ptr)
                            recv_ptr   = int(loss_arr.data.ptr)  # in-place
                            elem_count = int(loss_arr.size)
                            stream_ptr = int(h2d_stream.ptr)

                            # allReduce で和を取る
                            comm.allReduce(
                                send_ptr,
                                recv_ptr,
                                elem_count,
                                nccl_dtype,
                                nccl.NCCL_SUM,
                                stream_ptr
                            )

                            # 同期
                        h2d_stream.synchronize()
                        

                        # 必要なら平均化（world_size = GPU 数）
                        if is_main_device:
                            loss_sum=cp.sum(loss_arr)
                            vali_loss_mean = (loss_sum.item() / (world_size*all_args['accum_step']))
                            print(f'validation-loss:{vali_loss_mean}')
                            if all_args['wand']:
                                wandb.log({
                                        "validation": vali_loss_mean
                                        })
                        del loss_arr,vali_loss_arr
                        gc.collect()

                #================================================================================
                if is_main_device:
                    print_memory_status("foba終了後")
                #======================================================================================

                if is_main_device:
                    now=datetime.now()
                    time_str=now.strftime("%H:%M:%S")
                    print('time:'+time_str+f' : [Chunk {chunk_num}/{all_args["data_chunk_num"]}] [Epoch {epoch+1}/{all_args["max_epoch"]}] Iter {iters}/{max_iters} AvgLoss: {avg_loss:.4f}')

                    if all_args['wand']:
                        wandb.log({
                            "GradScale":GradScale,
                            "Grad Norm": actual_norm,
                            "loss": avg_loss,
                            "Perplexity": float(cp.exp(avg_loss))
                                   })

                if iters % all_args['save_step'] == 0:
                    if all_args['save']:
                        model_state = {
                            "param_shard_f16": manager.param_shard_f16,
                            "rank": rank,
                            "world_size": world_size,
                            "shard_size": manager.shard_size,
                            "padded_total": manager.padded_total,
                            "param_offsets": manager.param_offsets,
                            "param_shapes": manager.param_shapes,
                            "param_sizes": manager.param_sizes,
                            "replica_param_f16": manager.replica_param_f16,
                            "replica_offsets": manager.replica_offsets,
                            "replica_total": manager.replica_total,
                            "replica_padded_total": manager.replica_padded_total,
                            "replica_opt_shard_size": manager.replica_opt_shard_size,
                            "replicated_param_indices": sorted(manager.replicated_param_indices),
                            "zero25_replica_ratio": manager.replica_ratio,
                        }

                        save_pickle_gzip_to_drive(
                            model_state,
                            f"gdrive:model_save/model_shard_rank{rank}.pkl.gz"
                        )

                        opt_state = {
                            "m": None if optimizer is None else optimizer.m,
                            "v": None if optimizer is None else optimizer.v,
                            "w_fp32": None if optimizer is None else optimizer.master_weight_shard,
                            "t": None if optimizer is None else optimizer.t,
                            "replica_m": None if replica_optimizer is None else replica_optimizer.m,
                            "replica_v": None if replica_optimizer is None else replica_optimizer.v,
                            "replica_w_fp32": None if replica_optimizer is None else replica_optimizer.master_weight_shard,
                            "replica_t": None if replica_optimizer is None else replica_optimizer.t,
                        }

                        save_pickle_gzip_to_drive(
                            opt_state,
                            f"gdrive:model_save/optimizer_rank{rank}.pkl.gz"
                        )
                        
        if is_main_device:
            if all_args.get('wand'):
                wandb.finish()

def look_list(x,t):
    path='word_id.pkl.gz'
    with gzip.open(path, 'rb') as file:
        word_id=pickle.load(file)
    id_word={v:k for k,v in word_id.items()}
    k=[]
    l=[]
    for i in range(len(x[0])):
        k.append(id_word[int(x[0][i])])
        l.append(id_word[int(t[0][i])])
    print(k)
    print(l)
    return None




def save_pickle_gzip_to_drive(obj, drive_path):

    p = subprocess.Popen(
        ["rclone", "rcat", drive_path],
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    try:
        with gzip.GzipFile(fileobj=p.stdin, mode="wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

        # stdinを閉じてrcloneへ書き込み完了を伝える
        if p.stdin:
            p.stdin.close()

        stderr = p.stderr.read().decode("utf-8", errors="replace")
        return_code = p.wait()

        if return_code != 0:
            raise RuntimeError(f"rclone rcat failed:\n{stderr}")

    except Exception:
        p.kill()
        raise





def list_remote_jsonl_gz_files(remote_dir: str = REMOTE_DIR) -> list[str]:
    """
    Google Drive上のremote_dirにある .jsonl.gz ファイル一覧を取得する。
    例:
        gdrive:sample_dataset1024/sample_dataset_00000.jsonl.gz
    """
    result = subprocess.run(
        [
            "rclone",
            "lsf",
            remote_dir,
            "--files-only",
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    files = []
    for line in result.stdout.splitlines():
        name = line.strip()
        if name.endswith(".jsonl.gz"):
            files.append(f"{remote_dir}/{name}")

    files.sort()
    return files


def iter_input_ids_from_drive(
    remote_dir: str = REMOTE_DIR,
) -> Iterator[list[int]]:
    """
    Google Drive上の .jsonl.gz を直接読み込み、
    1サンプルずつ input_ids を返す generator。

    返るもの:
        input_ids: list[int]
    """
    remote_files = list_remote_jsonl_gz_files(remote_dir)

    for remote_file in remote_files:
        print(f"Reading from Drive: {remote_file}")

        proc = subprocess.Popen(
            [
                "rclone",
                "cat",
                remote_file,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        try:
            with gzip.GzipFile(fileobj=proc.stdout, mode="rb") as gz:
                for raw_line in gz:
                    line = raw_line.decode("utf-8").strip()

                    if not line:
                        continue

                    record = json.loads(line)

                    yield record["input_ids"]

        finally:
            if proc.stdout is not None:
                proc.stdout.close()

            return_code = proc.wait()

            if return_code != 0:
                stderr = proc.stderr.read().decode("utf-8", errors="replace")
                raise RuntimeError(
                    f"rclone cat failed for {remote_file}\n{stderr}"
                )


def load_input_ids_from_drive(
    remote_dir: str = REMOTE_DIR,
    max_samples: Optional[int] = None,
) -> list[list[int]]:
    """
    Google Drive上の .jsonl.gz を読み込み、
    list[list[int]] として返す。

    max_samples=None の場合は全件読み込む。
    ただし90万件すべて読むとメモリをかなり使うので注意。
    """
    dataset = []

    for i, input_ids in enumerate(iter_input_ids_from_drive(remote_dir)):
        dataset.append(input_ids)

        if max_samples is not None and len(dataset) >= max_samples:
            break

    return dataset





class TrainMultiGPU:
    def __init__(self, use_devices, batch_size, max_epoch, lr_max,
                 wand, max_grads, accum_step, lr_mini, padding_ID,
                 save, weight_decay, warm_up, vali, ScaleGrad_dic, model_dic,
                 main_device,half_float,save_step,data_chunk_num,vali_step,train_continue_path=None,
                 optimizer_device="cpu", zero25_replica_ratio=0.0,
                 zero25_replica_policy="largest_first"):
        # optimizer_device: "gpu" => fastest sharded GPU Adam, "cpu" => CPU offload
        # 引数をインスタンス変数として保存
        self.use_devices = use_devices
        self.batch_size = batch_size
        self.max_epoch = max_epoch
        self.lr_max = lr_max
        self.wand = wand
        self.max_grads = max_grads
        self.accum_step = accum_step
        self.lr_mini = lr_mini
        self.padding_ID = padding_ID
        self.save = save
        self.weight_decay = weight_decay
        self.warm_up = warm_up
        self.vali = vali
        self.model_dic = model_dic
        self.main_device = main_device
        self.time_idx = 0
        self.half_float=half_float
        self.save_step=save_step
        self.data_chunk_num=data_chunk_num
        self.vali=vali
        self.vali_step=vali_step
        self.ScaleGrad_dic=ScaleGrad_dic
        self.train_continue_path=train_continue_path
        self.optimizer_device=optimizer_device
        self.zero25_replica_ratio=zero25_replica_ratio
        self.zero25_replica_policy=zero25_replica_policy

    def fit(self,data_size):

        if self.vali:
            # TODO: バリデーションデータの準備 (メインデバイスで実行)
            pass
        
        world_size = len(self.use_devices)
        max_iters = data_size // (self.batch_size * self.accum_step)

        # データセットをワーカー数だけ分割
        comm_id = nccl.get_unique_id()
        ctx = multiprocessing.get_context('spawn')
        processes = []
        
        # 全ワーカーで共有する引数辞書
        shared_args = {
            'use_devices': self.use_devices, 'model_dic': self.model_dic, 'lr_max': self.lr_max,
            'max_epoch': self.max_epoch, 'max_iters': max_iters, 'accum_step': self.accum_step,
            'lr_mini': self.lr_mini, 'wand': self.wand, 'weight_decay': self.weight_decay,
            'warm_up': self.warm_up, 'ScaleGrad_dic': self.ScaleGrad_dic,
            'max_grads': self.max_grads, 'main_device': self.main_device, 'save': self.save,
            'half_float':self.half_float,'save_step':self.save_step,
            'data_chunk_num':self.data_chunk_num,'vali_step':self.vali_step,'vali':self.vali,
            'train_continue_path':self.train_continue_path,
            'optimizer_device':self.optimizer_device,
            'zero25_replica_ratio':self.zero25_replica_ratio,
            'zero25_replica_policy':self.zero25_replica_policy
        }
        

        # 各GPUに対応するワーカープロセスを起動
        for rank in range(world_size):
            p = ctx.Process(target=training_worker_nccl_stream, 
                            args=(rank, world_size, comm_id, shared_args,self.batch_size, self.padding_ID))
            p.start()
            processes.append(p)
        
        # 全てのプロセスが終了するのを待つ
        for p in processes:
            p.join()

        print("Training finished.")





# L2 Norm計算用カーネル (変更なし)
l2_sq_sum_kernel = cp.ReductionKernel(
    'T x',
    'float32 y',
    '((float)x) * ((float)x)',
    'a + b',
    'y = a',
    '0',
    'l2_sq_sum_kernel'
)

class MemoryZeRO2_5Manager:
    """
    ZeRO2.5 style parameter manager.

    - replicated_param_indices: 全rankがfp16 full paramを常駐保持する部分
      -> forward/backwardのparam broadcastなし
      -> gradはmicrobatch中はローカルfull gradへ加算
      -> optimizer直前にfull gradをallReduceし、その後rank担当sliceだけAdam更新
      -> 更新済みsliceをallGatherしてfull replica paramを復元
    - non replicated params: 従来通りZeRO3 shard保持
      -> 必要時だけownerからbroadcast
      -> gradはowner rankのgrad_shardへreduce
    """

    def __init__(self, model, world_size, comm, rank, stream,
                 alignment=256, replica_ratio=0.0, replica_policy="largest_first"):
        self.model = model
        self.world_size = world_size
        self.comm = comm
        self.rank = rank
        self.stream = stream
        self.comm_stream = cp.cuda.Stream(non_blocking=True)
        self.kept_buffers = []
        self.live_param_to_idx = {}

        self.replica_ratio = float(max(0.0, min(1.0, replica_ratio)))
        self.replica_policy = str(replica_policy)

        original_params = list(model.params)
        self.param_shapes = [p.shape for p in original_params]
        self.param_sizes = [int(p.size) for p in original_params]
        self.param_dtypes = [p.dtype for p in original_params]
        self.param_index_by_initial_id = {id(p): i for i, p in enumerate(original_params)}

        self._tag_model_param_indices(model)
        self.replicated_param_indices = self._select_replicated_params(original_params)

        self._build_replica_layout(original_params, alignment)
        self._build_replica_optimizer_layout(alignment)
        self._build_sharded_layout(original_params, alignment)

        self.param_shard_f16 = cp.zeros(self.shard_size, dtype=cp.float16)
        self.grad_shard_f16 = cp.zeros(self.shard_size, dtype=cp.float16)

        # replica_param_f16 はforward/backward用のfull replica storage。
        # allGatherを直接受けられるように world_size * replica_opt_shard_size までpaddingする。
        self.replica_param_f16 = cp.zeros(self.replica_padded_total, dtype=cp.float16)
        self.replica_grad_f16 = cp.zeros(self.replica_padded_total, dtype=cp.float16)

        # optimizer state/update用はrank担当sliceだけ。
        self.replica_param_opt_shard_f16 = cp.zeros(self.replica_opt_shard_size, dtype=cp.float16)
        self.replica_grad_opt_shard_f16 = cp.zeros(self.replica_opt_shard_size, dtype=cp.float16)

        self._initialize_params_from_rank0(original_params)
        self._refresh_replica_optimizer_shard_from_full()
        self._build_replica_views()

        none_params = [None] * len(original_params)
        model.link_params(none_params)

        del original_params, none_params
        gc.collect()
        cp.cuda.Device().synchronize()

        if rank == 0:
            rep_numel = int(self.replica_total)
            shard_logical = int(sum(self.param_sizes[i] for i in self.sharded_param_indices))
            print(
                f"[ZeRO2.5 Init] replica_ratio={self.replica_ratio:.3f}, "
                f"replicated_numel={rep_numel}, sharded_numel={shard_logical}, "
                f"shard_size={self.shard_size}, padded_total={self.padded_total}, "
                f"replica_opt_shard_size={self.replica_opt_shard_size}, "
                f"replica_padded_total={self.replica_padded_total}"
            )

    def _select_replicated_params(self, params):
        total = int(sum(p.size for p in params))
        if self.replica_ratio <= 0.0 or total == 0:
            return set()
        if self.replica_ratio >= 1.0:
            return set(range(len(params)))

        target = int(total * self.replica_ratio)
        sizes = [int(p.size) for p in params]

        if self.replica_policy == "first":
            order = list(range(len(params)))
        elif self.replica_policy == "last":
            order = list(range(len(params) - 1, -1, -1))
        elif self.replica_policy == "embedding_first":
            emb = getattr(self.model, "_zero3_embedding_idx", None)
            rest = [i for i in range(len(params)) if i != emb]
            rest.sort(key=lambda i: sizes[i], reverse=True)
            order = ([emb] if emb is not None else []) + rest
        else:
            order = sorted(range(len(params)), key=lambda i: sizes[i], reverse=True)

        chosen = []
        acc = 0
        for i in order:
            if acc >= target:
                break
            chosen.append(i)
            acc += sizes[i]
        return set(chosen)

    def _build_replica_layout(self, params, alignment):
        self.replica_offsets = [-1] * len(params)
        off = 0
        for i, p in enumerate(params):
            if i not in self.replicated_param_indices:
                continue
            if alignment > 1:
                off = -(-off // alignment) * alignment
            self.replica_offsets[i] = int(off)
            off += int(p.size)
        self.replica_total = int((-(-off // alignment) * alignment) if off > 0 else 0)

    def _build_replica_optimizer_layout(self, alignment):
        """
        replica paramはfullで常駐させるが、optimizer state/updateはrankごとに分割する。

        allGatherは各rank同じ要素数を送る必要があるため、
        replica_opt_shard_size * world_size までpaddingしたstorageを使う。
        """
        if self.replica_total == 0:
            self.replica_opt_shard_size = 0
            self.replica_padded_total = 0
            self.replica_opt_start = 0
            self.replica_opt_end = 0
            return

        shard = -(-int(self.replica_total) // int(self.world_size))
        if alignment > 1:
            shard = -(-shard // alignment) * alignment

        self.replica_opt_shard_size = int(shard)
        self.replica_padded_total = int(self.replica_opt_shard_size * self.world_size)
        self.replica_opt_start = int(self.rank * self.replica_opt_shard_size)
        self.replica_opt_end = int(self.replica_opt_start + self.replica_opt_shard_size)

    def _build_sharded_layout(self, params, alignment):
        self.sharded_param_indices = [i for i in range(len(params)) if i not in self.replicated_param_indices]
        self.param_offsets = [-1] * len(params)
        self.param_owners = [-1] * len(params)

        total_raw_size = sum(int(params[i].size) for i in self.sharded_param_indices)
        if total_raw_size == 0:
            self.shard_size = 0
            self.padded_total = 0
            return

        temp_shard_size = -(-total_raw_size // self.world_size)
        temp_shard_size = -(-temp_shard_size // alignment) * alignment

        while True:
            current_offset = 0
            max_offset = 0
            offsets = {}
            for i in self.sharded_param_indices:
                p_size = int(params[i].size)
                start_shard = current_offset // temp_shard_size
                end_shard = (current_offset + p_size - 1) // temp_shard_size
                if start_shard != end_shard:
                    current_offset = (start_shard + 1) * temp_shard_size
                offsets[i] = current_offset
                current_offset += p_size
                max_offset = current_offset

            total_capacity = temp_shard_size * self.world_size
            if max_offset <= total_capacity:
                self.shard_size = int(temp_shard_size)
                self.padded_total = int(total_capacity)
                for i, off in offsets.items():
                    self.param_offsets[i] = int(off)
                    self.param_owners[i] = int(off // self.shard_size)
                return

            overflow = max_offset - total_capacity
            add_per_rank = -(-overflow // self.world_size)
            add_per_rank = -(-add_per_rank // alignment) * alignment
            temp_shard_size += max(add_per_rank, alignment)

    def _tag_model_param_indices(self, model):
        for layer in model.layers:
            layer._zero3_param_indices = [self.param_index_by_initial_id[id(p)] for p in layer.params]

        model._zero3_embedding_idx = self.param_index_by_initial_id[id(model.Af_Em_para[0])]
        model.norm_._zero3_param_indices = [self.param_index_by_initial_id[id(p)] for p in model.norm_.params]

    def _build_replica_views(self):
        self.replica_views = {}
        for i in self.replicated_param_indices:
            off = self.replica_offsets[i]
            size = self.param_sizes[i]
            self.replica_views[i] = self.replica_param_f16[off:off + size].reshape(self.param_shapes[i])

    def _is_replicated(self, param_idx):
        return param_idx in self.replicated_param_indices

    def _get_replicated_param(self, param_idx):
        arr = self.replica_views[param_idx]
        self.live_param_to_idx[id(arr)] = param_idx
        return arr

    def _broadcast_param(self, param_idx):
        if self._is_replicated(param_idx):
            return self._get_replicated_param(param_idx)

        size = self.param_sizes[param_idx]
        shape = self.param_shapes[param_idx]
        owner = self.param_owners[param_idx]

        recv = cp.empty(size, dtype=cp.float16)

        if self.rank == owner:
            local_off = self.param_offsets[param_idx] - owner * self.shard_size
            src = self.param_shard_f16[local_off: local_off + size]
            send_ptr = src.data.ptr
        else:
            send_ptr = recv.data.ptr

        self.comm.broadcast(
            send_ptr,
            recv.data.ptr,
            int(size),
            nccl.NCCL_FLOAT16,
            int(owner),
            self.stream.ptr,
        )

        arr = recv.reshape(shape)
        self.live_param_to_idx[id(arr)] = param_idx
        return arr

    def gather_module(self, module):
        params = [self._broadcast_param(i) for i in module._zero3_param_indices]
        self.stream.synchronize()
        module.params = params
        return params

    def gather_modules(self, modules):
        for module in modules:
            module.params = [self._broadcast_param(i) for i in module._zero3_param_indices]
        self.stream.synchronize()

    def release_module(self, module):
        if getattr(module, "params", None) is not None:
            for p in module.params:
                if p is not None:
                    self.live_param_to_idx.pop(id(p), None)
            module.params = [None] * len(module.params)

    def release_modules(self, modules):
        for module in modules:
            self.release_module(module)

    def get_live_embedding(self):
        W = self.model.Af_Em_para[0]
        if W is None:
            return None
        return W if id(W) in self.live_param_to_idx else None

    def gather_embedding(self):
        W = self.get_live_embedding()
        if W is not None:
            return W

        W = self._broadcast_param(self.model._zero3_embedding_idx)
        self.stream.synchronize()
        self.model.Af_Em_para = [W]
        return W

    def release_embedding(self):
        W = self.model.Af_Em_para[0]
        if W is not None:
            self.live_param_to_idx.pop(id(W), None)
        self.model.Af_Em_para = [None]

    def zero_grad(self):
        if self.grad_shard_f16.size > 0:
            self.grad_shard_f16.fill(0)
        if self.replica_grad_f16.size > 0:
            self.replica_grad_f16.fill(0)
        if self.replica_grad_opt_shard_f16.size > 0:
            self.replica_grad_opt_shard_f16.fill(0)
        self.kept_buffers = []

    def flush_buckets(self):
        pass

    def synchronize(self):
        self.comm_stream.synchronize()
        self.stream.synchronize()
        self.kept_buffers = []

    def get_param_shard_view(self, rank=None):
        return self.param_shard_f16

    def get_replica_param_view(self):
        # forward/backward用のfull replica storage。optimizerには使わない。
        return self.replica_param_f16

    def get_replica_optimizer_shard_view(self):
        # Adam master/m/v を作る対象は、このrank担当sliceだけ。
        return self.replica_param_opt_shard_f16

    def set_param_shard(self, updated_shard_f16):
        if self.param_shard_f16.size > 0:
            self.param_shard_f16[...] = updated_shard_f16

    def set_replica_param(self, updated_replica_f16):
        # 互換用。通常のtrainではallgather_replicated_params_from_optimizer_shardsを使う。
        if self.replica_param_f16.size > 0:
            n = min(int(updated_replica_f16.size), int(self.replica_param_f16.size))
            self.replica_param_f16[:n] = updated_replica_f16.reshape(-1)[:n]
            self._refresh_replica_optimizer_shard_from_full()

    def set_replica_optimizer_shard(self, updated_replica_shard_f16):
        if self.replica_param_opt_shard_f16.size > 0:
            self.replica_param_opt_shard_f16[...] = updated_replica_shard_f16

    def _refresh_replica_optimizer_shard_from_full(self):
        if self.replica_opt_shard_size == 0:
            return
        s = self.replica_opt_start
        e = self.replica_opt_end
        self.replica_param_opt_shard_f16[...] = self.replica_param_f16[s:e]

    def _reduce_flat_to_global(self, global_start, grad_flat):
        if grad_flat.dtype != cp.float16:
            grad_flat = grad_flat.astype(cp.float16)
        grad_flat = cp.ascontiguousarray(grad_flat)

        global_end = global_start + grad_flat.size
        cur = int(global_start)
        while cur < global_end:
            owner = cur // self.shard_size
            shard_start = owner * self.shard_size
            shard_end = shard_start + self.shard_size
            part_end = min(global_end, shard_end)
            part_count = int(part_end - cur)
            grad_off = int(cur - global_start)
            send_part = cp.ascontiguousarray(grad_flat[grad_off: grad_off + part_count])

            recv_ptr = 0
            recv_tmp = None
            if self.rank == owner:
                dst_off = int(cur - shard_start)
                recv_tmp = cp.empty(part_count, dtype=cp.float16)
                recv_ptr = recv_tmp.data.ptr

            compute_stream = cp.cuda.get_current_stream()
            event = cp.cuda.Event()
            event.record(compute_stream)
            self.comm_stream.wait_event(event)

            with self.comm_stream:
                self.comm.reduce(
                    send_part.data.ptr,
                    recv_ptr,
                    part_count,
                    nccl.NCCL_FLOAT16,
                    nccl.NCCL_SUM,
                    int(owner),
                    self.comm_stream.ptr,
                )
                if self.rank == owner:
                    self.grad_shard_f16[dst_off: dst_off + part_count] += recv_tmp

            self.kept_buffers.append(send_part)
            if recv_tmp is not None:
                self.kept_buffers.append(recv_tmp)
            cur = int(part_end)

    def _accumulate_replicated_flat(self, param_idx, flat_start, grad_flat):
        if grad_flat.dtype != cp.float16:
            grad_flat = grad_flat.astype(cp.float16)
        grad_flat = cp.ascontiguousarray(grad_flat)
        off = self.replica_offsets[param_idx] + int(flat_start)
        self.replica_grad_f16[off: off + grad_flat.size] += grad_flat.reshape(-1)

    def accumulate_grads(self, params, grads):
        if grads is None:
            return
        if isinstance(params, (list, tuple)):
            for p, g in zip(params, grads):
                self.accumulate_grads(p, g)
            return

        param_idx = self.live_param_to_idx[id(params)]
        if self._is_replicated(param_idx):
            self._accumulate_replicated_flat(param_idx, 0, grads.ravel())
        else:
            global_start = self.param_offsets[param_idx]
            self._reduce_flat_to_global(global_start, grads.ravel())

    def accumulate_grad_slice(self, param, row_start, row_end, grad_chunk):
        if grad_chunk is None:
            return
        param_idx = self.live_param_to_idx[id(param)]
        H = self.param_shapes[param_idx][1]
        flat_start = int(row_start) * H
        if self._is_replicated(param_idx):
            self._accumulate_replicated_flat(param_idx, flat_start, grad_chunk.reshape(-1))
        else:
            global_start = self.param_offsets[param_idx] + flat_start
            self._reduce_flat_to_global(global_start, grad_chunk.reshape(-1))

    def accumulate_embedding_grads_chunked(self, param, idx, dout, padding_id=None, vocab_chunk_size=8192):
        if idx is None:
            return
        param_idx = self.live_param_to_idx[id(param)]
        V, H = self.param_shapes[param_idx]

        idx_flat = idx.reshape(-1).astype(cp.int32)
        dout_flat = dout.reshape(-1, H)
        if padding_id is not None:
            mask = idx_flat != padding_id
            idx_flat = idx_flat[mask]
            dout_flat = dout_flat[mask]
        if idx_flat.size == 0:
            return

        unique_ids, inverse = cp.unique(idx_flat, return_inverse=True)
        grad_rows = cp.zeros((unique_ids.size, H), dtype=dout_flat.dtype)
        cp.add.at(grad_rows, inverse, dout_flat)

        num_chunks = (V + vocab_chunk_size - 1) // vocab_chunk_size
        local_has = cp.zeros((num_chunks,), dtype=cp.int32)
        local_has[unique_ids // vocab_chunk_size] = 1
        global_has = local_has.copy()

        with self.comm_stream:
            self.comm.allReduce(
                global_has.data.ptr,
                global_has.data.ptr,
                num_chunks,
                nccl.NCCL_INT32,
                nccl.NCCL_MAX,
                self.comm_stream.ptr,
            )
        self.comm_stream.synchronize()

        for c in cp.asnumpy(cp.where(global_has != 0)[0]):
            s = int(c) * vocab_chunk_size
            e = min(s + vocab_chunk_size, V)
            rows = cp.where((unique_ids >= s) & (unique_ids < e))[0]
            chunk_grad = cp.zeros((e - s, H), dtype=grad_rows.dtype)
            chunk_grad[unique_ids[rows] - s] = grad_rows[rows]
            self.accumulate_grad_slice(param, s, e, chunk_grad)

    def prepare_replicated_grads_for_sharded_optimizer(self):
        """
        backward中に各rankで溜めたfull replica gradを全rankで合算し、
        このrankのoptimizer担当sliceだけを replica_grad_opt_shard_f16 に切り出す。

        まずは互換性重視で allReduce + slice にしている。
        NCCL reduceScatter が安定して使える環境なら、ここをreduceScatterに置き換えると
        通信量と一時メモリをさらに削れる。
        """
        if self.replica_grad_f16.size == 0:
            return

        with self.comm_stream:
            self.comm.allReduce(
                self.replica_grad_f16.data.ptr,
                self.replica_grad_f16.data.ptr,
                int(self.replica_grad_f16.size),
                nccl.NCCL_FLOAT16,
                nccl.NCCL_SUM,
                self.comm_stream.ptr,
            )
        self.comm_stream.synchronize()

        s = self.replica_opt_start
        e = self.replica_opt_end
        self.replica_grad_opt_shard_f16[...] = self.replica_grad_f16[s:e]

    def sync_replicated_grads(self):
        # 旧API互換: 以前はfull gradをallReduceするだけだったが、
        # 今はsharded optimizer用slice作成まで行う。
        self.prepare_replicated_grads_for_sharded_optimizer()

    def allgather_replicated_params_from_optimizer_shards(self):
        """
        各rankが更新した replica_param_opt_shard_f16 を集めて、
        次stepのforward/backwardで使うfull replica_param_f16を全rankで復元する。
        """
        if self.replica_param_opt_shard_f16.size == 0:
            return

        # CPU offload optimizerのH2D setやGPU optimizer kernelが終わってからallGatherする。
        self.stream.synchronize()

        with self.comm_stream:
            self.comm.allGather(
                self.replica_param_opt_shard_f16.data.ptr,
                self.replica_param_f16.data.ptr,
                int(self.replica_opt_shard_size),
                nccl.NCCL_FLOAT16,
                self.comm_stream.ptr,
            )
        self.comm_stream.synchronize()

    def check_nan_inf(self, grad_shard, comm, stream):
        local_bad = False
        if self.grad_shard_f16.size > 0:
            local_bad = local_bad or bool(cp.asnumpy(cp.logical_not(cp.isfinite(self.grad_shard_f16)).any()))
        if self.replica_grad_f16.size > 0:
            local_bad = local_bad or bool(cp.asnumpy(cp.logical_not(cp.isfinite(self.replica_grad_f16)).any()))
        bad_tensor = cp.array([int(local_bad)], dtype=cp.int32)
        with stream:
            comm.allReduce(bad_tensor.data.ptr, bad_tensor.data.ptr, 1, nccl.NCCL_INT32, nccl.NCCL_MAX, stream.ptr)
        stream.synchronize()
        return int(bad_tensor[0]) > 0

    def calculate_global_grad_norm(self, grad_shard, comm, stream):
        local_shard_sq = cp.array([0.0], dtype=cp.float32)
        if self.grad_shard_f16.size > 0:
            local_shard_sq[0] = cp.sum(self.grad_shard_f16.astype(cp.float32) ** 2)
        with stream:
            comm.allReduce(local_shard_sq.data.ptr, local_shard_sq.data.ptr, 1, nccl.NCCL_FLOAT32, nccl.NCCL_SUM, stream.ptr)
        stream.synchronize()

        replica_sq = cp.array(0.0, dtype=cp.float32)
        if self.replica_grad_f16.size > 0:
            replica_sq = cp.sum(self.replica_grad_f16.astype(cp.float32) ** 2)
        return float(cp.sqrt(local_shard_sq[0] + replica_sq))

    def calculate_global_param_norm(self, comm, stream, rank=None):
        local_shard_sq = cp.array([0.0], dtype=cp.float32)
        if self.param_shard_f16.size > 0:
            local_shard_sq[0] = cp.sum(self.param_shard_f16.astype(cp.float32) ** 2)
        with stream:
            comm.allReduce(local_shard_sq.data.ptr, local_shard_sq.data.ptr, 1, nccl.NCCL_FLOAT32, nccl.NCCL_SUM, stream.ptr)
        stream.synchronize()

        replica_sq = cp.array(0.0, dtype=cp.float32)
        if self.replica_param_f16.size > 0:
            replica_sq = cp.sum(self.replica_param_f16.astype(cp.float32) ** 2)
        return float(cp.sqrt(local_shard_sq[0] + replica_sq))

    def _initialize_params_from_rank0(self, params):
        for i, p in enumerate(params):
            size = self.param_sizes[i]

            if self.rank == 0:
                buf = p.ravel()
                if buf.dtype != cp.float16:
                    buf = buf.astype(cp.float16)
                buf = cp.ascontiguousarray(buf)
            else:
                buf = cp.empty(size, dtype=cp.float16)

            with self.stream:
                self.comm.broadcast(
                    buf.data.ptr,
                    buf.data.ptr,
                    int(size),
                    nccl.NCCL_FLOAT16,
                    0,
                    self.stream.ptr,
                )
            self.stream.synchronize()

            if self._is_replicated(i):
                off = self.replica_offsets[i]
                self.replica_param_f16[off: off + size] = buf
            else:
                owner = self.param_owners[i]
                if self.rank == owner:
                    local_off = self.param_offsets[i] - owner * self.shard_size
                    self.param_shard_f16[local_off: local_off + size] = buf
            self.stream.synchronize()
            del buf