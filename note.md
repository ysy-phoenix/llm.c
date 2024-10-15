## 预备知识
矩阵求导的本质：矩阵 A 中的每一个元素对矩阵 B 中的每一个元素求导。
Karpathy 原视频：[Andrej Karpathy Building makemore Part 4: Becoming a Backprop Ninja](https://www.youtube.com/watch?v=q8SA3rM6ckI)

### cross entropy loss backprop

```python
loss = F.corss_entropy(logits, Yb)
logit_maxes = logits.max(1, keepdim=True).values
norm_logits = logits - logit_maxes
counts = norm_logits.exp()
counts_sum = counts.sum(1, keepdims=True)
counts_sum_inv = counts_sum**-1
probs = counts * counts_sum_inv
logprobs = probs.log()
loss = -logprobs[range(n), Yb].mean()
```

> 这里求导时将 `logits_maxes` 视为常数

更简洁的写法

```python
probs = F.softmax(logits, dim=1) # shape [B, T]
loss = -probs.log()[range(n), Yb].mean()

dlogits = F.softmax(logits, 1)
dlogits[range(n), Yb] -= 1
dlogits /= n
```

易知：

$$
\frac{\partial \text{loss}}{\partial \text{logits}_{ij}} = \sum_k \frac{\partial \text{loss}}{\partial \text{probs}_{ik}} \frac{\partial \text{probs}_{ik}}{\partial \text{logits}_{ij}}
$$

其中

$$
\frac{\partial \text{loss}}{\partial \text{probs}_{ik}} = -\frac{1}{\text{probs}_{ik}} \times \frac{1}{n} \times \mathcal{I}[\text{Yb}[i] == k]
$$

> 以下均用 $n$ 表示 `B`（batch size）。

在推导 $\frac{\partial \text{probs}_{ik}}{\partial \text{logits}_{ij}}$ 前，首先推导 `softmax` 函数求导。

$$
p_i = \text{softmax}(z_i) = \frac{e^{z_i}}{\sum_{j} e^{z_j}}
$$

$$
\frac{\partial p_i}{\partial z_i}
= \frac{(\sum_{j} e^{z_j}) \cdot e^{z_i} - e^{z_i} \cdot e^{z_i}}{(\sum_{j} e^{z_j})^2}
= \frac{e^{z_i} \left( \sum_{j} e^{z_j} - e^{z_i} \right)}{(\sum_{j} e^{z_j})^2}
= p_i (1 - p_i)
\quad (i = k)
$$

$$
\frac{\partial p_i}{\partial z_k}
= \frac{-e^{z_i} e^{z_k}}{(\sum_{j} e^{z_j})^2}
= -p_i p_k
\quad (i \neq k)
$$

从而
$$
\frac{\partial \text{probs}_{ik}}{\partial \text{logits}_{ij}} = \begin{cases}
p_k (1 - p_k) & \text{if } k = j \\
-p_k p_j & \text{if } k \neq j
\end{cases}
$$

> 方便起见，记 $\text{probs}_{ik} = p_k$。

故

$$
\frac{\partial \text{loss}}{\partial \text{logits}_{ij}}
= \sum_{k} \frac{\partial \text{loss}}{\partial \text{probs}_{ik}} \frac{\partial \text{probs}_{ik}}{\partial \text{logits}_{ij}}
= \begin{cases}
\frac{1}{\text{probs}_{ik}} \times \frac{1}{n} \times p_k (p_k - 1)\\
\frac{1}{\text{probs}_{ik}} \times \frac{1}{n} \times p_k p_j
\end{cases}
= \begin{cases}
\frac{1}{n} (\text{probs}_{ik} - 1) & \text{if } k = \text{Yb}[i] = j \\
\frac{1}{n} \text{probs}_{ij} & \text{if } k = \text{Yb}[i] \neq j
\end{cases}
$$

### batch norm backprop

#### forward

$$
\begin{aligned}
\text{BN}(x) &= \gamma \cdot \frac{x - \mu_B}{\sigma_B} + \beta \\
\mu_B &= \frac{1}{n} \sum_{i=1}^n x_i \\
\sigma_B^2 &= \frac{1}{n - 1} \sum_{i=1}^n (x_i - \mu_B)^2 + \epsilon \\
\hat{x}_i &= \frac{x_i - \mu_B}{\sigma_B} \\
y_i &= \gamma\hat{x}_i + \beta
\end{aligned}
$$

> 这里考虑无偏估计版本。

#### backward

$$
\begin{aligned}
\frac{\partial \text{loss}}{\partial x_i}
&= \sum_{k}^n \frac{\partial \text{loss}}{\partial y_k} \cdot \frac{\partial y_k}{\partial x_i} \\
&= \sum_{k}^n \frac{\partial \text{loss}}{\partial y_k} \cdot \frac{\partial (\gamma \hat{x}_k + \beta)}{\partial x_i} \\
&= \sum_{k}^n \frac{\partial \text{loss}}{\partial y_k} \cdot \gamma \cdot \frac{\partial \hat{x}_k}{\partial x_i} \\
&= \sum_{k}^n \frac{\partial \text{loss}}{\partial y_k} \cdot \gamma \cdot \sum_j^n \frac{\partial \hat{x}_k}{\partial (x_j - \mu_B)} \cdot \frac{\partial (x_j - \mu_B)}{\partial x_i} \\
\end{aligned}
$$

以上可以拆分成五部分

$$
\begin{aligned}
\frac{\partial \text{loss}}{\partial x_i}
&= \gamma \cdot \sum_{k}^n \frac{\partial \text{loss}}{\partial y_k} \cdot \sum_j^n \frac{\partial \hat{x}_k}{\partial (x_j - \mu_B)} \cdot \frac{\partial (x_j - \mu_B)}{\partial x_i} \\
&= \gamma \cdot \sum_{k\neq i} \frac{\partial \text{loss}}{\partial y_k} \cdot \sum_{j\neq i, j \neq k} \frac{\partial \hat{x}_k}{\partial (x_j - \mu_B)} \cdot \frac{\partial (x_j - \mu_B)}{\partial x_i} \\
&+ \gamma \cdot \sum_{k\neq i} \frac{\partial \text{loss}}{\partial y_k} \cdot \frac{\partial \hat{x}_k}{\partial (x_k - \mu_B)} \cdot \frac{\partial (x_k - \mu_B)}{\partial x_i} \quad (j = k) \\
&+ \gamma \cdot \sum_{k\neq i} \frac{\partial \text{loss}}{\partial y_k} \cdot \frac{\partial \hat{x}_k}{\partial (x_i - \mu_B)} \cdot \frac{\partial (x_i - \mu_B)}{\partial x_i} \quad (j = i) \\
&+ \gamma \cdot \frac{\partial \text{loss}}{\partial y_i} \cdot \sum_{j\neq i} \frac{\partial \hat{x}_i}{\partial (x_j - \mu_B)} \cdot \frac{\partial (x_j - \mu_B)}{\partial x_i} \\
&+ \gamma \cdot \frac{\partial \text{loss}}{\partial y_i} \cdot \frac{\partial \hat{x}_i}{\partial (x_i - \mu_B)} \cdot \frac{\partial (x_i - \mu_B)}{\partial x_i} \\
\end{aligned}
$$

为什么要拆成这五部分呢，因为

$$
\begin{aligned}
\frac{\partial \hat{x}_k}{\partial (x_j - \mu_B)}
&= \begin{cases}
\frac{1}{\sigma_B}(1 - \frac{\hat{x}_k^2}{n - 1}) & \text{if } j = k \\
-\frac{1}{\sigma_B} \cdot \frac{1}{n - 1} \cdot \hat{x}_k \cdot \hat{x}_j & \text{if } j \neq k
\end{cases} \\
\frac{\partial (x_j - \mu_B)}{\partial x_i} &= \begin{cases}
1 - \frac{1}{n} & \text{if } i = j \\
-\frac{1}{n} & \text{if } i \neq j
\end{cases}
\end{aligned}
$$

从而

$$
\begin{aligned}
\frac{\partial \text{loss}}{\partial x_i}
&= \gamma \cdot \sum_{k\neq i} \frac{\partial \text{loss}}{\partial y_k} \cdot
\sum_{j\neq i, j \neq k} -\frac{1}{\sigma_B} \cdot \frac{1}{n - 1} \cdot \hat{x}_k \cdot \hat{x}_j \cdot -\frac{1}{n} \\
&+ \gamma \cdot \sum_{k\neq i} \frac{\partial \text{loss}}{\partial y_k} \cdot
\frac{1}{\sigma_B}(1 - \frac{\hat{x}_k^2}{n - 1}) \cdot -\frac{1}{n} \quad (j = k) \\
&+ \gamma \cdot \sum_{k\neq i} \frac{\partial \text{loss}}{\partial y_k} \cdot
-\frac{1}{\sigma_B} \cdot \frac{1}{n - 1} \cdot \hat{x}_k \cdot \hat{x}_i \cdot (1 - \frac{1}{n}) \quad (j = i) \\
&+ \gamma \cdot \frac{\partial \text{loss}}{\partial y_i} \cdot
\sum_{j\neq i} -\frac{1}{\sigma_B} \cdot \frac{1}{n - 1} \cdot \hat{x}_i \cdot \hat{x}_j \cdot -\frac{1}{n} \\
&+ \gamma \cdot \frac{\partial \text{loss}}{\partial y_i} \cdot
\frac{1}{\sigma_B} \cdot (1 - \frac{1}{n}) \cdot (1 - \frac{\hat{x}_i^2}{n - 1}) \\
\end{aligned}
$$

整理，得

$$
\begin{aligned}
\frac{\partial \text{loss}}{\partial x_i}
&= \frac{\gamma}{\sigma_B} \cdot \sum_k \frac{\partial \text{loss}}{\partial y_k} \cdot \frac{1}{n(n-1)} \cdot \hat{x}_k \cdot \sum_j\hat{x}_j \quad (\sum_j \hat{x}_j = 0) \\
&+ \frac{\gamma}{\sigma_B} \cdot \frac{\partial \text{loss}}{\partial y_i}\\
&- \frac{\gamma}{\sigma_B} \cdot \sum_k\frac{\partial \text{loss}}{\partial y_k} \cdot \frac{1}{n} \\
&- \frac{\gamma}{\sigma_B} \cdot \hat{x}_i \cdot \sum_k \frac{\partial \text{loss}}{\partial y_k} \cdot \frac{1}{n - 1} \cdot \hat{x}_k \\
&= \frac{\gamma}{\sigma_B} \cdot \left(\frac{\partial \text{loss}}{\partial y_i} - \frac{1}{n} \sum_k \frac{\partial \text{loss}}{\partial y_k} - \frac{1}{n - 1} \hat{x}_i \sum_k \frac{\partial \text{loss}}{\partial y_k} \cdot \hat{x}_k \right)
\end{aligned}
$$

> Batch Norm 和 Layer Norm 的 backward 推导是类似的，区别在于求和的维度。

## llm.c Forward

### encoder

#### forward

```cpp
// wte = nn.Embedding(config.vocab_size, config.n_embd),
// wpe = nn.Embedding(config.block_size, config.n_embd)
// b, t = idx.size() # t is the sequence length
// assert t <= self.config.block_size  # max sequence length
// pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)
// tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
// pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
// x = tok_emb + pos_emb

/*
out <-> x
inp <-> idx
out_bt <-> x[b, t, :]
ix <-> inp[b, t] \in [0, vocab_size)
wte <-> self.transformer.wte.weight
wpe <-> self.transformer.wpe.weight
B <-> b
T <-> t
C <-> n_embd
*/

void encoder_forward(float* out,
                   int* inp, float* wte, float* wpe,
                   int B, int T, int C) {
    // out is (B,T,C). At each position (b,t), a C-dimensional vector summarizing token & position
    // inp is (B,T) of integers, holding the token ids at each (b,t) position
    // wte is (V,C) of token embeddings, short for "weight token embeddings"
    // wpe is (maxT,C) of position embeddings, short for "weight positional embedding"
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            // seek to the output position in out[b,t,:]
            float* out_bt = out + b * T * C + t * C;
            // get the index of the token at inp[b, t]
            int ix = inp[b * T + t];
            // seek to the position in wte corresponding to the token
            float* wte_ix = wte + ix * C;
            // seek to the position in wpe corresponding to the position
            float* wpe_t = wpe + t * C;
            // add the two vectors and store the result in out[b,t,:]
            for (int i = 0; i < C; i++) {
                out_bt[i] = wte_ix[i] + wpe_t[i];
            }
        }
    }
}
```

#### backward
$$
\text{d}wte = \frac{\partial L}{\partial wte} = \sum_{b,t} \text{d}out_{b,t} \frac{\partial out_{b,t}}{\partial wte} = \sum_{b,t} \text{d}out_{b,t} \times 1
\\
\text{d}wpe = \frac{\partial L}{\partial wpe} = \sum_{b,t} \text{d}out_{b,t} \frac{\partial out_{b,t}}{\partial wpe} = \sum_{b,t} \text{d}out_{b,t} \times 1
\\
\text{d}out_{b,t} = \frac{\partial L}{\partial out_{b,t}}
$$

```cpp
// dout is (B,T,C)
// inp is (B,T)
// dwte is (V,C)
// dwpe is (maxT,C)
void encoder_backward(float* dwte, float* dwpe,
                      float* dout, int* inp,
                      int B, int T, int C) {
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            float* dout_bt = dout + b * T * C + t * C;
            int ix = inp[b * T + t];
            float* dwte_ix = dwte + ix * C;
            float* dwpe_t = dwpe + t * C;
            for (int i = 0; i < C; i++) {
                float d = dout_bt[i];
                dwte_ix[i] += d;
                dwpe_t[i] += d;
            }
        }
    }
}
```
### layernorm

#### forward

$$
y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta
\\
\hat{x} = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}}
\\
$$

```cpp
// self.ln_1 = nn.LayerNorm(config.n_embd)
// y = self.ln_1(x) # (B,T,C)
// out <-> y
// inp <-> x
// bias <-> beta
// weight <-> gamma
// m <-> mean <-> E[x]
// v <-> rstd <-> Var[x]
// s <-> \sqrt{\mathrm{Var}[x] + \epsilon}
// rstd <-> \frac{1}{s}
// weight <-> \gamma
// bias <-> \beta

void layernorm_forward(float* out, float* mean, float* rstd,
                       float* inp, float* weight, float* bias,
                       int B, int T, int C) {
    // reference: https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
    // both inp and out are (B,T,C) of the activations
    // mean and rstd are (B,T) buffers, to be used later in backward pass
    // at each position (b,t) of the input, the C-dimensional vector
    // of activations gets normalized, then scaled and shifted
    float eps = 1e-5f;
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            // seek to the input position inp[b,t,:]
            float* x = inp + b * T * C + t * C;
            // calculate the mean
            float m = 0.0f;
            for (int i = 0; i < C; i++) {
                m += x[i];
            }
            m = m/C;
            // calculate the variance (without any bias correction)
            float v = 0.0f;
            for (int i = 0; i < C; i++) {
                float xshift = x[i] - m;
                v += xshift * xshift;
            }
            v = v/C;
            // calculate the rstd (reciprocal standard deviation)
            float s = 1.0f / sqrtf(v + eps);
            // seek to the output position in out[b,t,:]
            float* out_bt = out + b * T * C + t * C;
            for (int i = 0; i < C; i++) {
                float n = (s * (x[i] - m)); // normalize
                float o = n * weight[i] + bias[i]; // scale and shift
                out_bt[i] = o; // write
            }
            // cache the mean and rstd for the backward pass later
            mean[b * T + t] = m;
            rstd[b * T + t] = s;
        }
    }
}
```

#### backward
回顾 Batch Norm 的 backward
$$
\frac{\partial \text{loss}}{\partial x_i} = \frac{\gamma}{\sigma_B} \cdot \left(\frac{\partial \text{loss}}{\partial y_i} - \frac{1}{n} \sum_k \frac{\partial \text{loss}}{\partial y_k} - \frac{1}{n - 1} \hat{x}_i \sum_k \frac{\partial \text{loss}}{\partial y_k} \cdot \hat{x}_k \right)
$$
Layer Norm 区别就是 $\gamma$ 移进去变成 $\gamma_k$。


```cpp
void layernorm_backward(float* dinp, float* dweight, float* dbias,
                        float* dout, float* inp, float* weight, float* mean, float* rstd,
                        int B, int T, int C) {
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            float* dout_bt = dout + b * T * C + t * C;
            float* inp_bt = inp + b * T * C + t * C;
            float* dinp_bt = dinp + b * T * C + t * C;
            float mean_bt = mean[b * T + t];
            float rstd_bt = rstd[b * T + t];

            // first: two reduce operations
            float dnorm_mean = 0.0f;
            float dnorm_norm_mean = 0.0f;
            for (int i = 0; i < C; i++) {
                float norm_bti = (inp_bt[i] - mean_bt) * rstd_bt;
                float dnorm_i = weight[i] * dout_bt[i];
                dnorm_mean += dnorm_i;
                dnorm_norm_mean += dnorm_i * norm_bti;
            }
            dnorm_mean = dnorm_mean / C;
            dnorm_norm_mean = dnorm_norm_mean / C;

            // now iterate again and accumulate all the gradients
            for (int i = 0; i < C; i++) {
                float norm_bti = (inp_bt[i] - mean_bt) * rstd_bt;
                float dnorm_i = weight[i] * dout_bt[i];
                // gradient contribution to bias
                dbias[i] += dout_bt[i];
                // gradient contribution to weight
                dweight[i] += norm_bti * dout_bt[i];
                // gradient contribution to input
                float dval = 0.0f;
                dval += dnorm_i; // term 1
                dval -= dnorm_mean; // term 2
                dval -= norm_bti * dnorm_norm_mean; // term 3
                dval *= rstd_bt; // final scale
                dinp_bt[i] += dval;
            }
        }
    }
}
```
