import cupy as cp
import gc
import random


def sigmoid(x):
    return 1/(1+cp.exp(-x))


def softmax(x):

    x = x - cp.max(x, axis=1, keepdims=True)  # 数値安定性のため最大値を引く
    exp_x = cp.exp(x)
    del x
    cp.get_default_memory_pool().free_all_blocks()
    cp.cuda.Stream.null.synchronize()
    return exp_x / cp.sum(exp_x, axis=1, keepdims=True)

def Ltotal_clip_grads(grads,max_grads):
    total_norm=0
    for grad in grads:
        total_norm = cp.sqrt(cp.sum(grad**2))
        if total_norm > max_grads:
            rate = max_grads / (total_norm + 1e-5)
            return grad * rate
        else:
            return grad


def place_enco(T, d_model,half_float):
    # 位置エンコーディング行列を初期化
    if half_float:
        PE = cp.zeros((T, d_model), dtype=cp.float16)
        # 位置インデックスと次元インデックスを計算
        position = cp.arange(T)[:, cp.newaxis]  # (T, 1)
        div_term = cp.exp(cp.arange(0, d_model, 2) * -(cp.log(10000.0) / d_model)).astype(cp.float16)

        # 偶数と奇数のインデックスでsinとcosを適用
        PE[:, 0::2] = cp.sin(position * div_term).astype(cp.float16)   
        PE[:, 1::2] = cp.cos(position * div_term).astype(cp.float16)  
    else:
        PE = cp.zeros((T, d_model), dtype=cp.float32)

        # 位置インデックスと次元インデックスを計算
        position = cp.arange(T)[:, cp.newaxis]  # (T, 1)
        div_term = cp.exp(cp.arange(0, d_model, 2) * -(cp.log(10000.0) / d_model)).astype(cp.float32)  

        # 偶数と奇数のインデックスでsinとcosを適用
        PE[:, 0::2] = cp.sin(position * div_term).astype(cp.float32)     
        PE[:, 1::2] = cp.cos(position * div_term).astype(cp.float32)    
    return PE



def xavier_initialization(shape):
    fan_in = shape[0]
    fan_out = shape[1]
    stddev = cp.sqrt(2. / (fan_in + fan_out))
    return cp.random.normal(loc=0.0, scale=stddev, size=shape).astype(cp.float32)


def init_weight(shape,stddev):
    return cp.random.normal(loc=0.0, scale=stddev, size=shape).astype(cp.float16)


def total_clip_grads(grad, min_grad, max_grad):
    dtype=cp.float32
    grad=grad.astype(cp.float32)
    min_grad = dtype(min_grad)
    max_grad = dtype(max_grad)

    # 勾配の絶対値の最大値でスケーリングしてL2ノルムを計算
    max_val = cp.max(cp.abs(grad))
    max_val_safe = cp.maximum(max_val, dtype(1e-5))  # 0割防止

    # スケーリングした勾配の二乗和を計算
    scaled_grad = grad / max_val_safe
    norm = cp.sqrt(cp.sum(scaled_grad**2)) * max_val_safe

    # スケーリング係数計算（min_grad/max_gradでクリップ）
    scale = cp.clip(
        cp.maximum(min_grad, cp.minimum(max_grad, norm)) / (norm + dtype(1e-5)),
        min_grad / (norm + dtype(1e-5)),
        max_grad / (norm + dtype(1e-5)),
    )

    # 勾配にスケールをかけて返す
    clipped_grad = grad * scale

    # スケーリング後の範囲を確認し、最終的にクリップ
    clipped_grad = cp.clip(clipped_grad, -max_grad, max_grad)
    return clipped_grad


def Ltotal_clip_grads(grads, min_grads, max_grads):
    
    #勾配クリッピングを逐次的に実行する
    
    # データ型とクリップの上限・下限を設定
    dtype = cp.float32
    min_g, max_g = dtype(min_grads), dtype(max_grads)
    eps = dtype(1e-5) # ゼロ除算を避けるための微小な値

    clipped_grads = [] # 結果を格納するための空のリスト

    # 各勾配を順番に処理する
    for grad in grads:
        # 勾配をfloat32に変換
        grad_f32 = grad.astype(dtype)
        
        # 1. L2ノルム（勾配の大きさ）を計算
        norm = cp.linalg.norm(grad_f32)
        
        # 2. ノルムをクリッピングし、スケーリング率を計算
        #    勾配の大きさがmin_gより小さい場合は、min_gにクリップされる
        rate = cp.clip(norm, min_g, max_g) / (norm + eps)
        
        # 3. 勾配をスケーリング
        scaled_grad = grad_f32 * rate
        
        # 4. 最後に、各要素が[-max_g, max_g]の範囲に収まるようにクリッピング
        final_clipped_grad = cp.clip(scaled_grad, -max_g, max_g)
        
        clipped_grads.append(final_clipped_grad)
        
    return clipped_grads



def ZeRO_clip_grads_(grads, max_norm):
    """
    全ての勾配に対して標準的なGlobal Norm Clippingを行う関数
    - 範囲指定をなくし、渡された勾配リスト全体でノルムを計算・適用します
    """
    dtype = cp.float32
    max_norm = dtype(max_norm)
    eps = dtype(1e-6) # ゼロ除算防止

    # ---------------------------------------------------------
    # 1. Global Norm（全体のL2ノルム）の計算
    # ---------------------------------------------------------
    total_norm_sq = dtype(0.0)
    
    # リスト内の全ての勾配を走査して二乗和を蓄積
    for grad in grads:
        # 計算精度確保のためfloat32へキャストして計算
        grad_f32 = grad.astype(dtype)
        total_norm_sq += cp.sum(grad_f32 ** 2)
    
    total_norm = cp.sqrt(total_norm_sq)

    # ---------------------------------------------------------
    # 2. スケーリング係数の計算
    # ---------------------------------------------------------
    # max_norm を超えている場合のみ 1未満 の値（縮小率）になる
    clip_coef = max_norm / (total_norm + eps)
    
    # ---------------------------------------------------------
    # 3. 勾配のスケーリング適用 (上限を超えている場合のみ)
    # ---------------------------------------------------------
    if clip_coef < 1.0:
        # 全ての勾配に対して一律の係数を掛ける
        for i in range(len(grads)):
            grads[i] *= clip_coef
            
    return grads


def ZeRO_clip_grads(grads, max_norm,scale):
    """
    全ての勾配に対して標準的なGlobal Norm Clippingを行う関数
    - 範囲指定をなくし、渡された勾配リスト全体でノルムを計算・適用します
    """
    dtype = cp.float32
    max_norm = dtype(max_norm)
    eps = dtype(1e-6) # ゼロ除算防止

    # ---------------------------------------------------------
    # 1. Global Norm（全体のL2ノルム）の計算
    # ---------------------------------------------------------
    total_norm_sq = dtype(0.0)
    
    # リスト内の全ての勾配を走査して二乗和を蓄積
    for grad in grads:
        # 計算精度確保のためfloat32へキャストして計算
        grad_f32 = grad.astype(dtype)
        total_norm_sq += cp.sum(grad_f32 ** 2)
    
    total_norm = cp.sqrt(total_norm_sq)*scale

    if total_norm > max_norm:
        clip_coef = max_norm / (total_norm + eps)
        return clip_coef
    else:     
        return 1.0
    


def print_memory_status(key_word):
    mempool = cp.get_default_memory_pool()
    
    # 実際に配列データとして使用されているバイト数
    used = mempool.used_bytes() 
    # OSから確保済みだが、今は使われていない（次の確保用にプールされている）分も含む合計
    total = mempool.total_bytes() 
    
    print(f"{key_word}:Used: {used / 1024**3:.2f} GB / Total Alloc: {total / 1024**3:.2f} GB")



class ScaleGrad:
    def __init__(self,ScaleGrad_dic):
        self.StartScale=ScaleGrad_dic['StartScale']
        self.DownRate=ScaleGrad_dic['DownRate']
        self.UpRate=ScaleGrad_dic['UpRate']
        self.Count=0
        self.ScaleNow=ScaleGrad_dic['StartScale']
        self.con_count=0
        self.Scale_Up_Step=ScaleGrad_dic['Scale_Up_Step']

    def UpdateScale(self,nan_sign):
        if nan_sign:
            self.ScaleNow=int(self.ScaleNow*self.DownRate)
            self.con_count+=1
            if self.ScaleNow<=0.01:
                value = random.randint(1000,2000)
                self.ScaleNow=self.StartScale+value
                self.con_count=0
        else:
            self.con_count=0
            if self.Count%self.Scale_Up_Step==0 and self.Count!=0:
                self.ScaleNow=self.ScaleNow*self.UpRate
        self.Count+=1

        return self.ScaleNow
            