import os,sys,pickle,gzip
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from model.fast_layers import TimeDropout,SwiGLU,RMSNorm,TimeEmbedding_shard,GroupedQueryAttention,ChunkedCrossEntropy_Share
import cupy as cp
from model.functions import init_weight



# Embeddingパラメーター共有で学習
# 
# Decoderとモデルが合わさったバージョン
class Cupa_ver4:
    def __init__(self,model_dic,param_path=None):
        if param_path == None:
            self.layernum=model_dic['layer_num']
            self.time_size=model_dic['time_size']
            self.hidden_size=model_dic['hidden_size']
            self.dropout_rate=model_dic['dropout_rate']
            self.padding_id=model_dic['padding_id']
            self.half_float=model_dic['half_float']
            self.vocab_size=model_dic['vocab_size']
            self.kv_head=model_dic['kv_head']
            self.q_head=model_dic['q_head']
            chunk_size = int(model_dic.get('ce_chunk_size', 10000))
            
            self.Af_Em_para = [init_weight([self.vocab_size,self.hidden_size],0.02)]
            #勾配のメモリをあらかじめ確保
            self.AF_Cross_loss = ChunkedCrossEntropy_Share(self.padding_id,chunk_size,self.half_float)
            self.embed=TimeEmbedding_shard(self.padding_id,self.half_float)
            self.params=[]
            self._initialize_layers(self.hidden_size)
            self.params+=self.Af_Em_para
            self.params+=self.norm_.params
            self.one_cycle_num=len(self.layers)//self.layernum


            #GQAの部分
            self.head_dim = self.hidden_size // self.q_head
            self.cos, self.sin = precompute_freqs_cis(
            self.head_dim, self.time_size, theta=10000.0, dtype=cp.float16
            )
            self.ba_sin = -self.sin
            mask_bool = cp.triu(cp.ones((self.time_size, self.time_size), dtype=bool), k=1)
            self.attn_bias = cp.where(mask_bool, -float("inf"), 0.0).astype(cp.float16)
            self.attn_bias = self.attn_bias[None, None, None, :, :]
            self.gqa_pack = (self.cos,self.sin,self.ba_sin,self.attn_bias)
            self.keep_layer_num = 1
        
        else:
            self.layernum=model_dic['layer_num']
            self.time_size=model_dic['time_size']
            self.hidden_size=model_dic['hidden_size']
            self.dropout_rate=model_dic['dropout_rate']
            self.padding_id=model_dic['padding_id']
            self.half_float=model_dic['half_float']
            self.vocab_size=model_dic['vocab_size']
            self.kv_head=model_dic['kv_head']
            self.q_head=model_dic['q_head']
            chunk_size = int(model_dic.get('ce_chunk_size', 10000))
            self.Af_Em_para = [init_weight([self.vocab_size,self.hidden_size],0.02)]
            #勾配のメモリをあらかじめ確保
            self.AF_Cross_loss = ChunkedCrossEntropy_Share(self.padding_id,chunk_size,self.half_float)
            #self.Af_Em_grad = cp.zeros_like(self.Af_Em_para, dtype=self.Af_Em_para)
            self.embed=TimeEmbedding_shard(self.padding_id,self.half_float)
            self.params=[]
            self._initialize_layers(self.hidden_size)
            self.params+=self.Af_Em_para
            self.params+=self.norm_.params
            self.one_cycle_num=len(self.layers)//self.layernum
            self.one_cycle_num = len(self.layers) // self.layernum


            self.load_weights_from_flat_file(param_path)
            self.self_link_params()

        self._apply_runtime_options(model_dic)



    def _initialize_layers(self,hidden_size):
        self.layers=[]
        if self.dropout_rate != 0:
            self.dropout_=TimeDropout(self.dropout_rate,self.half_float)
        self.norm_=RMSNorm(self.hidden_size,self.half_float)
        for _ in range(self.layernum):
            GQA=GroupedQueryAttention(hidden_size,self.time_size,self.q_head,self.kv_head,self.layernum,self.dropout_rate,self.half_float)
            feed=SwiGLU(hidden_size,self.layernum,self.half_float,self.dropout_rate)
            norm1=RMSNorm(self.hidden_size,self.half_float)
            norm2=RMSNorm(self.hidden_size,self.half_float)
            self.layers.append(GQA)
            self.params+=GQA.params
            self.layers.append(feed)
            self.params+=feed.params
            self.layers.append(norm1)
            self.params+=norm1.params
            self.layers.append(norm2)
            self.params+=norm2.params


    def _apply_runtime_options(self, model_dic):
        self.keep_layer_num = int(model_dic.get('keep_layer_num', getattr(self, 'keep_layer_num', 1)))
        self.keep_embedding_until_backward = bool(model_dic.get('keep_embedding_until_backward', True))

        self.gqa_cache_out_value = bool(model_dic.get('gqa_cache_out_value', False))
        self.gqa_cache_attn = bool(model_dic.get('gqa_cache_attn', False))
        self.swiglu_cache_gate = bool(model_dic.get('swiglu_cache_gate', False))

        flash_backend = str(model_dic.get("flash_backend", "none"))
        flash_block_m = int(model_dic.get("flash_block_m", 64))
        flash_block_n = int(model_dic.get("flash_block_n", 64))
        flash_bwd_block_m = int(model_dic.get("flash_bwd_block_m", 32))
        flash_bwd_block_n = int(model_dic.get("flash_bwd_block_n", 64))

        for layer in getattr(self, 'layers', []):
            if isinstance(layer, GroupedQueryAttention):
                layer.cache_out_value = self.gqa_cache_out_value
                layer.cache_attn = self.gqa_cache_attn

                layer.flash_backend = str(model_dic.get("flash_backend", "none"))
                layer.flash_block_m = int(model_dic.get("flash_block_m", 64))
                layer.flash_block_n = int(model_dic.get("flash_block_n", 64))
                layer.flash_bwd_block_m = int(model_dic.get("flash_bwd_block_m", 32))
                layer.flash_bwd_block_n = int(model_dic.get("flash_bwd_block_n", 64))
                layer.flash_bwd_block_d = int(model_dic.get("flash_bwd_block_d", 64))

            elif isinstance(layer, SwiGLU):
                layer.cache_gate = self.swiglu_cache_gate


    def forward(self, x, t, x_pa, manager):
        self.grads = []

        # ---------------------------------------------------------
        # Embedding gather
        # ---------------------------------------------------------
        W = manager.gather_embedding()
        if self.padding_id is not None:
            W[self.padding_id, :] = 0

        x = self.embed.forward(x, self.Af_Em_para)

        if self.dropout_rate != 0:
            x = self.dropout_.forward(x)

        pad_mask = self.make_mask(x_pa)

        # ---------------------------------------------------------
        # Transformer blocks
        #
        # self.layers の並び:
        # [GQA0, feed0, norm1_0, norm2_0,
        #  GQA1, feed1, norm1_1, norm2_1,
        #  ...]
        #
        # one_cycle_num = 4
        # keep_layer_num = 2 なら、2 block 分 = 8 module をまとめて gather
        # ---------------------------------------------------------
        block_size = self.one_cycle_num
        blocks_per_gather = getattr(self, "keep_layer_num", 1)
        group_size = block_size * blocks_per_gather

        for group_start in range(0, len(self.layers), group_size):
            group_end = min(group_start + group_size, len(self.layers))
            group_modules = self.layers[group_start:group_end]

            # 複数block分のparamsをまとめてgather
            manager.gather_modules(group_modules)

            # group内をforward順に処理
            for block_start in range(0, len(group_modules), block_size):
                GQA, feed, norm1, norm2 = group_modules[
                    block_start:block_start + block_size
                ]

                # PreNorm Attention block
                out = x
                x = norm1.forward(x)
                x = GQA.forward(x, pad_mask, self.gqa_pack)
                x += out

                # PreNorm FFN block
                out = x
                x = norm2.forward(x)
                x = feed.forward(x)
                x += out

            # group分のfull paramsをrelease
            manager.release_modules(group_modules)

        self.x_pa = x_pa

        # ---------------------------------------------------------
        # Final RMSNorm
        # ---------------------------------------------------------
        manager.gather_module(self.norm_)
        x = self.norm_.forward(x)
        manager.release_module(self.norm_)

        # ---------------------------------------------------------
        # Shared embedding cross entropy
        # ---------------------------------------------------------
        loss = self.AF_Cross_loss.forward(x, t, W)

        # backwardで同じembedding Wを使うので、通常はここでreleaseしない。
        # これで80k x 3072のembedding broadcastをmicrobatchごとに1回削減できる。
        if not self.keep_embedding_until_backward:
            manager.release_embedding()

        return loss
    




    def backward(self, GradScale, manager):
        # ---------------------------------------------------------
        # Embedding gather
        # ---------------------------------------------------------
        W = manager.get_live_embedding()
        if W is None:
            W = manager.gather_embedding()
        if self.padding_id is not None:
            W[self.padding_id, :] = 0

        def grad_writer(s, e, dW_chunk):
            manager.accumulate_grad_slice(W, s, e, dW_chunk)

        # ---------------------------------------------------------
        # CrossEntropy backward
        # dWはchunkごとにgrad_writerでZeRO3 shardへreduce
        # ---------------------------------------------------------
        norm_y = self.AF_Cross_loss.x
        dout, _ = self.AF_Cross_loss.backward(
            W,
            GradScale,
            grad_writer=grad_writer,
        )

        # ---------------------------------------------------------
        # Final RMSNorm backward
        # ---------------------------------------------------------
        manager.gather_module(self.norm_)

        dout = self.norm_.backward(dout, norm_y)
        manager.accumulate_grads(self.norm_.params, self.norm_.grads)
        self.norm_.grads = None

        manager.release_module(self.norm_)

        # ---------------------------------------------------------
        # Transformer blocks backward
        #
        # forwardでは group を前から処理したので、
        # backwardでは group を後ろから処理する。
        #
        # group内でも block を後ろから処理する。
        # 各blockのbackward順:
        # feed -> norm2 -> GQA -> norm1
        # ---------------------------------------------------------
        block_size = self.one_cycle_num
        blocks_per_gather = getattr(self, "keep_layer_num", 1)
        group_size = block_size * blocks_per_gather

        for group_end in range(len(self.layers), 0, -group_size):
            group_start = max(0, group_end - group_size)
            group_modules = self.layers[group_start:group_end]

            # 複数block分のparamsをまとめてgather
            manager.gather_modules(group_modules)

            # group内のblockを逆順に処理
            for block_start in range(len(group_modules) - block_size, -1, -block_size):
                GQA, feed, norm1, norm2 = group_modules[
                    block_start:block_start + block_size
                ]

                # forward時のcacheを取り出す
                # GQA.cache[0] は norm1.forward後の入力相当
                # feed.x は norm2.forward後の入力相当
                norm1_y = GQA.cache[0]
                norm2_y = feed.x

                # -------------------------
                # FFN branch backward
                # x = feed(norm2(x)) + residual
                # -------------------------
                out = dout

                dout = feed.backward(dout)
                manager.accumulate_grads(feed.params, feed.grads)
                feed.grads = None

                dout = norm2.backward(dout, norm2_y)
                manager.accumulate_grads(norm2.params, norm2.grads)
                norm2.grads = None

                dout += out

                # -------------------------
                # Attention branch backward
                # x = GQA(norm1(x)) + residual
                # -------------------------
                out = dout

                dout = GQA.backward(dout, self.gqa_pack)
                manager.accumulate_grads(GQA.params, GQA.grads)
                GQA.grads = None

                dout = norm1.backward(dout, norm1_y)
                manager.accumulate_grads(norm1.params, norm1.grads)
                norm1.grads = None

                dout += out

            # group分のfull paramsをrelease
            manager.release_modules(group_modules)

        # ---------------------------------------------------------
        # Embedding backward
        # ---------------------------------------------------------
        if self.dropout_rate != 0:
            dout = self.dropout_.backward(dout)

        manager.accumulate_embedding_grads_chunked(
            W,
            self.embed.idx,
            dout,
            padding_id=self.padding_id,
            vocab_chunk_size=self.AF_Cross_loss.chunk_size,
        )

        self.embed.idx = None
        manager.release_embedding()

        return None
    

    def predict(self,x,x_pa):
        self.grads=[]
        x=self.embed.forward(x,False)
        pad_mask=self.make_mask(x_pa)
        for i in range(0, len(self.layers),self.one_cycle_num):  # feedforward層を除く
            Multi_head,feed,norm1,norm2= self.layers[i:i+self.one_cycle_num]
            out=x
            x=norm1.forward(x,False)
            x=Multi_head.forward(x,pad_mask,False)
            x+=out
            out=x
            x=norm2.forward(x,False)
            x=feed.forward(x,False)
            x+=out
        self.x_pa=x_pa
        x=self.norm_.forward(x,False)
        x=self.AF.forward(x,False)
        return x


    
    def generate(self,out10,start_id,end_id,padding_id,temp=0.7,top=0.9,penalty=1.2):
        out10=[start_id]+out10
        count=len(out10)
        for i in range(self.time_size-len(out10)):
            out10.append(padding_id)
        out10=cp.array([out10])
        while True:
            penalty_array=cp.ones(self.vocab_size)
            x_pa = cp.array([count])
            pad_mask=self.make_mask(x_pa)
            x = self.embed.forward(out10,False)
            for i in range(0, len(self.layers),self.one_cycle_num): 
                GQA,feed,norm1,norm2= self.layers[i:i+self.one_cycle_num]
                out=x
                x=norm1.forward(x,False)
                x=GQA.forward(x,pad_mask,False)
                x+=out
                out=x
                x=norm2.forward(x,False)
                x=feed.forward(x,False)
                x+=out
            self.x_pa=x_pa
            x=self.norm_.forward(x,False)
            x=self.AF.forward(x,False)
            Score = x[x.shape[0]-1,count-1,:]
            Score[padding_id]=cp.float16('-inf')
            current_tokens = out10[0, :count]
            penalty_array[current_tokens]=penalty
            Score = cp.where(Score < 0, Score * penalty_array, Score / penalty_array)
            Score=_softmax(Score,temp)
            b=top_p_sampling(Score,top)
            if b==end_id:
                out10[0][count]=b
                count+=1
                out10=out10[0][:count]
                break
            elif count==self.time_size-1:
                break
            else:
                out10[0][count]=b
                count+=1
        return out10
    

    def chat_generate(self,out10,user_id,end_id,bot_id,padding_id,temp=0.7,top=0.9,penalty=1.2):
        out10=[user_id]+out10+[end_id]+[bot_id]#プロンプトを二回行うと性能が向上する。
        count=len(out10)
        for i in range(self.time_size-len(out10)):
            out10.append(padding_id)
        out10=cp.array([out10])
        while True:
            penalty_array=cp.ones(self.vocab_size)
            x_pa=[count]
            pad_mask=self.make_mask(x_pa)
            x = self.embed.forward(out10,False)
            for i in range(0, len(self.layers),self.one_cycle_num): 
                GQA,feed,norm1,norm2= self.layers[i:i+self.one_cycle_num]
                out=x
                x=norm1.forward(x,False)
                x=GQA.forward(x,pad_mask,False)
                x+=out
                out=x
                x=norm2.forward(x,False)
                x=feed.forward(x,False)
                x+=out
            self.x_pa=x_pa
            x=self.norm_.forward(x,False)
            x=self.AF.forward(x,False)
            Score = x[x.shape[0]-1,count-1,:]
            Score[padding_id]=cp.float16('-inf')
            current_tokens = out10[0, :count]
            penalty_array[current_tokens]=penalty
            Score = cp.where(Score < 0, Score * penalty_array, Score / penalty_array)
            Score=_softmax(Score,temp)
            b=top_p_sampling(Score,top)
            if b==end_id:
                out10[0][count]=b
                count+=1
                out10=out10[0][:count]
                break
            elif count==self.time_size-1:
                break
            else:
                out10[0][count]=b
                count+=1
        return out10
    
    
    def add_words(self, add_num):
        self.embed.add_words(add_num)
        self.AF.add_words(add_num)
        self.params = []
        for layer in self.layers:
            self.params += layer.params
        self.params += self.AF.params
        self.params += self.embed.params
        self.params += self.norm_.params
        return None


    def make_mask(self,padding_box):
        seq_ids = cp.arange(self.time_size)[None, :]
        valid_lengths = padding_box[:, None]
        pad_mask = seq_ids >= valid_lengths
        dtype = cp.float16 if self.half_float else cp.float32
        inf_val = dtype('-inf')
        zero_val = dtype(0.0)
        pad_mask = cp.where(pad_mask[:, None, None, :], inf_val, zero_val)

        return pad_mask


    def load_weights_from_flat_file(self,file_path):
        print(f"Loading weights from {file_path} ...")
        
        with gzip.open(file_path, 'rb') as f:
            flat_params = pickle.load(f)
        
        offset = 0
        for i, param in enumerate(self.params):
            size = param.size
            shape = param.shape
            
            # 平坦化配列から、このパラメータの分だけ切り出す
            # flat_params[開始位置 : 開始位置 + サイズ]
            flat_slice = flat_params[offset : offset + size]
            
            self.params[i] = self.params[i].astype(cp.float16)
            self.params[i][:] = flat_slice.reshape(shape)

            offset += size

        print("Weights loaded successfully.")


    def self_link_params(self):
        current_idx = 0
        # 1. 各レイヤーのリンク
        for layer in self.layers:
            for i in range(len(layer.params)):
                layer.params[i] = self.params[current_idx]
                current_idx += 1

        self.Af_Em_para = [self.params[current_idx]]
        
        current_idx += 1

        # 3. Norm層
        for i in range(len(self.norm_.params)):
            self.norm_.params[i] = self.params[current_idx]
            current_idx += 1

    def link_params(self, new_params_list):
        current_idx = 0
    
        for layer in self.layers:
            for i in range(len(layer.params)):
                layer.params[i] = new_params_list[current_idx]
                current_idx += 1
        
        self.Af_Em_para = [new_params_list[current_idx]]
        
        current_idx += 1

        for i in range(len(self.norm_.params)):
            self.norm_.params[i] = new_params_list[current_idx]
            current_idx += 1

        self.params = new_params_list



def _softmax(x,temperature):
    x/=temperature
    max_=cp.max(x, axis=-1, keepdims=True)
    x-=max_
    exp_x=cp.exp(x)  # expを一度だけ計算
    B=cp.sum(exp_x,axis=-1,keepdims=True)
    soft=exp_x/B  # softmaxを即座に計算
    return soft

def top_p_sampling(probs, p):
    probs = cp.asarray(probs, dtype=cp.float16)
    #target_index=53944
    #rank = sorted(probs, reverse=True).index(probs[target_index]) + 1
    #print(rank)

    sorted_idx = cp.argsort(-probs)
    sorted_probs = probs[sorted_idx]
    cum_probs = cp.cumsum(sorted_probs)

    # 🔧 修正点：p を ndarray に変換して渡す
    cutoff = int(cp.searchsorted(cum_probs, cp.array([p])))

    candidates = sorted_idx[:cutoff + 1]
    print(len(candidates))
    choice = cp.random.choice(candidates,size=1)

    return int(choice.get())


def precompute_freqs_cis(dim, end, theta=10000.0, dtype=cp.float32):
    freqs = 1.0 / (theta ** (cp.arange(0, dim, 2)[: (dim // 2)].astype(cp.float32) / dim))
    t = cp.arange(end).astype(cp.float32)
    emb = cp.outer(t, freqs)
    # cos, sinの生成
    return cp.cos(emb).astype(dtype), cp.sin(emb).astype(dtype)