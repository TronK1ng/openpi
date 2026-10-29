# -*- coding: utf-8 -*-
"""π0 / π0.5 核心模型 —— 逐行注释学习版（learn 分支）。

基底：Physical-Intelligence/openpi @ 215abfb 的 src/openpi/models/pi0.py
说明：本文件即上游原文件，仅新增了本说明与 "#" 注释；代码逻辑、变量名、语句顺序均未改动。
      目前只改了 pi0.py，main 分支保持与上游一致，便于 `git diff main..learn` 查看笔记。

为什么说 pi0.py 是 models/ 里最重要的文件？
  - models/model.py    定义接口与数据契约（Observation / BaseModel），是"骨架"；
  - models/gemma.py    是 Transformer 主干（双专家注意力），是"肌肉"；
  - models/pi0.py      定义 π0 / π0.5 的训练目标与推理算法（流匹配 + Euler 去噪），是"大脑"。
  读懂本文件，等于读懂了 openpi 的核心算法；其余文件都是为它服务的。

============================ 形状速查（默认配置） ============================
  B    batch size
  H    action_horizon = 50（pi0 默认）/ 15 或 10（pi05 各配置）
  A    action_dim     = 32（所有平台都 pad 到 32）
  L    max_token_len  = 48（pi0）/ 200（pi05）
  Wp   PaliGemma 宽度 = 2048
  Wa   action expert 宽度 = 1024
  P    图像 token 数 = 3 相机 × (224/14)² = 3 × 256 = 768

  图像   [B,224,224,3] --SigLIP--> [B,256,Wp]     （每张图 256 个 patch）
  语言   [B,L]         --Embed --> [B,L,Wp]
  prefix = [B, 768+L, Wp]                         双向注意力（ar_mask=0）
  state  [B,32]        --Linear-> [B,1,Wa]        （仅 π0；π0.5 把 state 写进 prompt）
  动作   [B,H,32]      --Linear-> [B,H,Wa]
  suffix = [B,1+H,Wa]（π0）或 [B,H,Wa]（π0.5）
  输出   [B,H,32]                                 预测速度场 v_t

============================ 两个专家到底是什么 ============================
  不是两条独立网络，而是"同一序列里，不同 token 用不同权重做 q/k/v"：
  prefix token 用 PaliGemma 的权重，suffix token 用 action expert 的权重，
  拼成一条序列后做联合 self-attention（见 gemma.py 的 Attention）。
  这样动作 token 能看到图像/语言，图像/语言却看不到动作（由 attn_mask 保证）。

============================ 建议阅读顺序 ============================
  make_attn_mask → posemb_sincos → Pi0.__init__ → embed_prefix → embed_suffix
  → compute_loss（训练）→ sample_actions（推理）
"""

# =============================================================================
# 0. 依赖导入
# =============================================================================
import logging                     # 标准库日志；本文件里 logger 实际未使用，保留给扩展

import einops                      # 张量重排："b -> b s" 这类写法让维度意图自解释
import flax.nnx as nnx             # Flax NNX：可变模块 + split/merge 的函数式边界
import flax.nnx.bridge as nnx_bridge  # 适配器：把 Linen 写的 gemma/siglip 包成 NNX 模块
import jax                         # jit / random / lax.while_loop / tree
import jax.numpy as jnp            # JAX 版 NumPy
from typing_extensions import override  # 标记"重写了基类抽象方法"，便于静态检查

from openpi.models import model as _model      # Observation / Actions / BaseModel / preprocess_observation
from openpi.models import pi0_config           # Pi0Config：action_dim / horizon / variant 等超参
import openpi.models.gemma as _gemma           # 双专家 Transformer 主干
import openpi.models.siglip as _siglip         # 视觉编码器（So400m/14）
from openpi.shared import array_typing as at   # beartype + jaxtyping：运行时校验形状与 dtype

logger = logging.getLogger("openpi")  # 与训练脚本共用一个 logger 命名空间（本文件未实际打印）


# =============================================================================
# 1. 注意力掩码：把 "块因果（block-causal）" 变成 2D 布尔矩阵
# =============================================================================
def make_attn_mask(input_mask, mask_ar):
    """Adapted from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` bool[?B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: bool[?B, N] mask that's true where previous tokens cannot depend on
        it and false where it shares the same attention mask as the previous token.
    """
    # mask_ar 可能只有 [N]（所有样本相同），先广播到和 input_mask 一样的 [B, N]
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    # 累积求和把 "块标记" 变成 "块编号"：同一块内编号相同，进入下一块时 +1
    cumsum = jnp.cumsum(mask_ar, axis=1)
    # 第 i 个 query 能看第 j 个 key，当且仅当 key 的块号 <= query 的块号
    # 注意广播方向：cumsum[:, None, :] 是 key（列），cumsum[:, :, None] 是 query（行）
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    # padding 位置既不能当 query，也不能被当成 key：两个方向都要有效
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    # 两个条件同时满足才允许注意；返回 [B, N_query, N_key]
    return jnp.logical_and(attn_mask, valid_mask)


# =============================================================================
# 2. 标量位置的正弦编码：给 flow matching 的时间步 t 用
# =============================================================================
@at.typecheck  # 运行时检查：pos 形状必须是 [b]，返回 [b, embedding_dim]
def posemb_sincos(
    pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "b {embedding_dim}"]:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    # 一半维度给 sin，一半给 cos，所以总维度必须是偶数
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    # 0..1 均匀取 embedding_dim/2 个点，用来生成对数间隔的周期
    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    # 周期从 min_period 几何级数增长到 max_period（对 t 越敏感的高频分量周期越短）
    period = min_period * (max_period / min_period) ** fraction
    # 外积：每个 batch 的标量 pos 乘以每个频率的角速度（2π/period）
    # precision=HIGHEST 是为了让 t 很小时仍有足够数值精度
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    # 前半 sin、后半 cos，拼成 [b, embedding_dim]
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


# =============================================================================
# 3. 主模型：一个类同时实现 π0 与 π0.5（用 self.pi05 分支）
# =============================================================================
class Pi0(_model.BaseModel):
    def __init__(self, config: pi0_config.Pi0Config, rngs: nnx.Rngs):
        # 基类 BaseModel 只保存三个公共维度，供 loss/inference 与数据管道使用
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        # 一个布尔开关贯穿全文件：π0.5 的 state 进 prompt、时间用 adaRMS
        self.pi05 = config.pi05
        # PaliGemma 2B：width=2048, depth=18, heads=8, head_dim=256
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        # action expert（Gemma 300M）：width=1024, depth=18；与 2B 共享 attention 的头配置
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        # TODO: rewrite gemma in NNX. For now, use bridge.
        # 把 Linen 的 _gemma.Module 包成 NNX Module，才能享受 nnx.split/merge 与 freeze filter
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],  # 两个"专家"的宽度/深度
                embed_dtype=config.dtype,                         # 例如 bfloat16
                adarms=config.pi05,                               # π0.5 才启用 adaRMS
            )
        )
        # lazy_init：先用假输入跑一遍 init，把参数形状确定下来
        # use_adarms=[False, True]：专家 0 用普通 RMSNorm，专家 1（动作）用 adaRMS
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True] if config.pi05 else [False, False])
        # 视觉编码器：So400m/14，输出宽度对齐到 PaliGemma 的 width
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,  # 用 Dense head 把 patch 投到 2048 维
                variant="So400m/14",
                pool_type="none",                    # 不池化，保留 16×16=256 个 patch token
                scan=True,                           # 27 层用 scan 只编译一份
                dtype_mm=config.dtype,
            )
        )
        # 用 fake_obs 里的图像张量做一次 lazy_init，确定视觉塔参数形状
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        # 用 nnx.Dict 组织，使参数路径形如 PaliGemma/llm/...、PaliGemma/img/...，与预训练权重对齐
        self.PaliGemma = nnx.Dict(llm=llm, img=img)

        # 动作 32 维 → action expert 宽度 1024
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)

        if config.pi05:
            # π0.5：时间步单独过一个小 MLP，输出作为 adaRMS 的调制条件（不拼进 token）
            self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        else:
            # π0：state 单独投影成一个 token
            self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
            # 时间编码与动作特征 concat 后过 MLP（1024+1024 → 1024）
            self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)

        # 动作专家输出 1024 维 → 动作 32 维（输出的是"速度场 v_t"，不是动作本身）
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)

        # This attribute gets automatically set by model.train() and model.eval().
        # 控制 dropout 行为；训练脚本会调用 model.train() 把它置 False
        self.deterministic = True

    # -------------------------------------------------------------------------
    # 3.1 prefix：图像 + 语言，双向注意力
    # -------------------------------------------------------------------------
    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        input_mask = []  # 每个 token 是否有效（非 padding），最后拼成 [b, 768+L]
        ar_mask = []     # 每个 token 的块边界标记，最后拼成 [768+L]
        tokens = []      # 每个 token 的 embedding，最后拼成 [b, 768+L, 2048]

        # embed images
        # 三路相机按 dict 顺序遍历（base / left_wrist / right_wrist）
        for name in obs.images:
            # SigLIP 返回 (投影后的 patch 特征, 中间结果 dict)，这里只要前者
            # 形状：[b, 224, 224, 3] → [b, 256, 2048]
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)

            tokens.append(image_tokens)  # 收集图像 token
            # 一张图对应 256 个 token，但 image_mask 只有 [b]，所以把标量掩码复制 256 份
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_tokens.shape[1],
                )
            )
            # image tokens attend to each other
            # ar_mask=False 表示"与上一个 token 同块"→ 图像内部双向可见
            ar_mask += [False] * image_tokens.shape[1]

        # add language (aka tokenized inputs)
        # 语言 prompt 的 token 已经由 transforms 里的 PaligemmaTokenizer 生成并 pad 到 L
        if obs.tokenized_prompt is not None:
            # 只做 embedding 查表（method="embed"），不跑 Transformer
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")  # [b, L, 2048]
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)  # prompt 的 padding 位置为 False
            # full attention between image and language inputs
            # 继续用 False：图像与语言属于同一个双向注意力的 prefix 块
            ar_mask += [False] * tokenized_inputs.shape[1]

        # 沿序列维拼接：图像 token 在前，语言 token 在后
        tokens = jnp.concatenate(tokens, axis=1)          # [b, 768+L, 2048]
        input_mask = jnp.concatenate(input_mask, axis=1)  # [b, 768+L]
        ar_mask = jnp.array(ar_mask)                      # [768+L]，前面全是 False（块号 0）
        return tokens, input_mask, ar_mask

    # -------------------------------------------------------------------------
    # 3.2 suffix：state + 带噪动作 + 时间步，块内因果
    # -------------------------------------------------------------------------
    @at.typecheck
    def embed_suffix(
        self, obs: _model.Observation, noisy_actions: _model.Actions, timestep: at.Float[at.Array, " b"]
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b emb"] | None,
    ]:
        input_mask = []  # suffix 内部的有效位（这里恒为 True，只有长度信息）
        ar_mask = []     # suffix 的块边界标记
        tokens = []      # suffix token 的 embedding

        if not self.pi05:
            # add a single state token
            # π0：state 是一个连续向量，投影成 1 个 token 放在动作前面
            state_token = self.state_proj(obs.state)[:, None, :]  # [b,32] → [b,1,1024]
            tokens.append(state_token)
            # state token 永远有效
            input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
            # image/language inputs do not attend to state or actions
            # True 表示"开启新块"：state 只能被后面的动作看到，prefix 看不到它
            ar_mask += [True]

        # 带噪动作 [b,H,32] 投影到专家宽度 [b,H,1024]
        action_tokens = self.action_in_proj(noisy_actions)
        # embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        # 时间 t 是标量，编码成 1024 维向量；min/max period 决定对 t 的分辨率
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)

        if self.pi05:
            # time MLP (for adaRMS)
            # π0.5：时间只用来调制归一化，不进入 token 序列
            time_emb = self.time_mlp_in(time_emb)
            time_emb = nnx.swish(time_emb)            # swish = SiLU
            time_emb = self.time_mlp_out(time_emb)
            time_emb = nnx.swish(time_emb)
            action_expert_tokens = action_tokens      # 动作 token 原样保留
            adarms_cond = time_emb                    # 作为 adaRMS 条件传给每一层
        else:
            # mix timestep + action information using an MLP (no adaRMS)
            # π0：把时间广播到每个动作 token，再 concat 后过 MLP 融合
            time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
            action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)  # [b,H,2048]
            action_time_tokens = self.action_time_mlp_in(action_time_tokens)             # [b,H,1024]
            action_time_tokens = nnx.swish(action_time_tokens)
            action_time_tokens = self.action_time_mlp_out(action_time_tokens)            # [b,H,1024]
            action_expert_tokens = action_time_tokens
            adarms_cond = None                                                           # 不用 adaRMS

        tokens.append(action_expert_tokens)
        # 动作 token 的有效位：形状 [b,H]，全 True
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))
        # image/language/state inputs do not attend to action tokens
        # 第一个动作 token 开新块（True），其余 H-1 个与它同块（False）
        # → 所有动作 token 互相可见，但 prefix/state 看不到动作
        ar_mask += [True] + ([False] * (self.action_horizon - 1))

        # [b, 1+H, 1024]（π0）或 [b, H, 1024]（π0.5）
        tokens = jnp.concatenate(tokens, axis=1)
        # [b, 1+H] 或 [b, H]
        input_mask = jnp.concatenate(input_mask, axis=1)
        # [1+H] 或 [H]
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask, adarms_cond

    # -------------------------------------------------------------------------
    # 3.3 训练：流匹配（flow matching）回归速度场
    # -------------------------------------------------------------------------
    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        # 一个随机 key 拆成三个独立子 key：数据增强、噪声、时间采样
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        # 训练时做 resize/裁剪/旋转/颜色抖动；推理时 train=False 只 resize
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        # 动作形状 [b,H,A]；前缀批维允许任意（*b）
        batch_shape = actions.shape[:-2]
        # 从标准正态采噪声 ε，形状与动作相同
        noise = jax.random.normal(noise_rng, actions.shape)
        # 采样时间 t ~ Beta(1.5,1) 再压到 (0.001, 0.999)，偏向接近 1（噪声端）以稳定训练
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        # [b] → [b,1,1]，方便与 [b,H,A] 广播
        time_expanded = time[..., None, None]
        # 线性插值：t=1 时是纯噪声，t=0 时是真动作（与扩散文献一致，和 π0 论文相反）
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        # 速度场目标：从噪声指向动作的向量（d x_t / d t，t 从 1 到 0）
        u_t = noise - actions

        # one big forward pass of prefix + suffix at once
        # prefix：图像 + 语言，[b, 768+L, 2048]
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        # suffix：state + 带噪动作 + 时间，[b, 1+H, 1024]
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
        # 有效位沿序列拼接：[b, 768+L+1+H]
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        # 块标记拼接：[768+L+1+H]
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        # 生成 [b, total, total] 的布尔注意力掩码
        attn_mask = make_attn_mask(input_mask, ar_mask)
        # 位置编号：只在有效 token 上递增（padding 不占位置）
        positions = jnp.cumsum(input_mask, axis=1) - 1
        # 两个专家一次前向：专家 0 处理 prefix，专家 1 处理 suffix，在 attention 内联合
        # 返回 (list[prefix_out, suffix_out], kv_cache)；训练不需要 cache，用 _ 丢弃
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
        )
        # 只取 suffix 的最后 H 个 token（动作 token），投影回 32 维得到预测速度
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

        # 对动作维与时间步取均方误差，得到每个样本一个标量 loss → [b]
        # 外层 train_step 再做 jnp.mean 得到总 loss
        return jnp.mean(jnp.square(v_t - u_t), axis=-1)

    # -------------------------------------------------------------------------
    # 3.4 推理：KV cache + Euler 积分，从纯噪声迭代出动作
    # -------------------------------------------------------------------------
    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,  # 去噪步数；默认 10
        noise: at.Float[at.Array, "b ah ad"] | None = None,  # 可注入固定噪声（复现实验）
    ) -> _model.Actions:
        # 推理不做数据增强，所以 rng 传 None
        observation = _model.preprocess_observation(None, observation, train=False)
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        # 负步长：t 从 1.0 递减到 0.0
        dt = -1.0 / num_steps
        # 批大小来自 state（[b,32] 的第 0 维）
        batch_size = observation.state.shape[0]
        if noise is None:
            # 从纯噪声出发，形状 [b,H,32]
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        # first fill KV cache with a forward pass of the prefix
        # prefix 只算一次：图像+语言的 k/v 缓存起来，后续每步复用
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        # prefix 内部的注意力掩码 [b, P, P]
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        # prefix 的位置编号 [b, P]
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        # 专家 1 传 None → 只跑专家 0；第二个返回值是 KV cache
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        # 单步 Euler：输入当前 x_t 与 time，输出 x_{t+dt}
        def step(carry):
            x_t, time = carry
            # 每一步都要重新算 suffix：因为动作 token 变了，时间也变了
            # 但 state 投影、时间编码很便宜；昂贵的 prefix 已在 cache 里
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each
            # other
            # suffix 内部掩码：[b, S, S]，动作块内互相可见
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the
            # prefix tokens
            # 每个 suffix token 都能看所有有效 prefix token（padding 除外）
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            # `combined_mask` is shape (b, suffix_len, prefix_len + suffix_len) indicating how the suffix tokens (which
            # generate the queries) can attend to the full prefix + suffix sequence (which generates the keys and values)
            # 拼成 [b, S, P+S]：左侧是 cache 里的 prefix，右侧是当前 suffix
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            # 防御式断言：形状必须与 (P+S) 的 key 长度一致
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )
            # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
            # suffix 的位置从 prefix 的有效长度开始继续编号
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            # 专家 0 传 None → 只跑专家 1；kv_cache 让 suffix 的 query 看到 prefix 的 k/v
            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            # 只跑专家 1 时，prefix 输出必须是 None
            assert prefix_out is None
            # 取动作 token 的输出，投影成速度场 [b,H,32]
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            # Euler 更新：沿速度场走一步；time 同步前进
            return x_t + dt * v_t, time + dt

        # 循环条件：time 还没到 0（用 -dt/2 做浮点容差，保证正好 num_steps 步）
        def cond(carry):
            x_t, time = carry
            # robust to floating-point error
            return time >= -dt / 2

        # 整个去噪循环编译成一个 while_loop，避免 Python 逐步调度
        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        # 返回去噪后的动作 [b,H,32]（此时仍是"归一化空间"，由外层 Unnormalize 还原）
        return x_0


# =============================================================================
# 附录：读代码时最容易卡住的四个问题
# =============================================================================
#
# Q1. 为什么 H 个动作 token 在同一个注意力块里（而不是因果逐个生成）？
#     π0 用流匹配"一次性"预测整段动作块，H 个动作是并行去噪的，
#     因此它们必须互相可见，否则每个动作只能看到前面的动作，无法联合一致。
#     π0-FAST 才是自回归的，所以它的 ar_mask 是逐 token 为 1（见 pi0_fast.py）。
#
# Q2. 为什么 π0 的 state 在 suffix，而 π0.5 的 state 在 prompt？
#     π0：state 是连续向量，作为一个 token 和动作放一起，方便动作专家读取。
#     π0.5：把 state 离散化成 256 个 bin 写进语言 prompt（"Task: ..., State: ...; Action: "），
#           让"状态"也成为语言上下文的一部分，提升开放世界泛化；
#           代价是 suffix 只剩动作 token，时间信息只能靠 adaRMS 注入。
#     注意 pi05_libero 配置里 discrete_state_input=False，是向后兼容的例外。
#
# Q3. 为什么推理时 prefix 可以只算一次？
#     因为 prefix 只依赖图像与语言，不随去噪步变化；
#     self-attention 中它的 k/v 不变，缓存起来即可（KV cache 形状 [层, b, P, kv_head, head_dim]）。
#     suffix 每步都要重算，但只有 H 个 token，成本远小于 prefix 的 768+ 个 token。
#
# Q4. 为什么 loss 是对 "速度场" 的 MSE，而不是直接对动作？
#     这是 flow matching 的核心：网络学的是从噪声指向数据的向量场 v_θ(x_t, t)，
#     训练目标 u_t = noise - actions 是解析可得的"真速度"。
#     推理时沿该速度场做数值积分（Euler），就能从噪声走到动作分布。
#
# 术语表
#   prefix    图像 + 语言 token 序列，双向注意力
#   suffix    state（仅 π0）+ 带噪动作 + 时间，块内因果
#   adaRMS    adaptive RMSNorm：用时间 embedding 生成 scale/shift/gate 来调制每层
#   KV cache  缓存 prefix 每层的 key/value，避免每步重算
#   Euler     一阶数值积分：x_{t+dt} = x_t + dt · v_t
#   norm stats 训练集上统计的 mean/std/q01/q99，推理时用于反归一化
# =============================================================================
