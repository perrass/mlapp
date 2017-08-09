# DL-NLP-RNN

### RNN

#### init docs

parse word and punctuation to int

-> get two dicts, int to word, and word to int

#### Subsampling

只有计算词向量的时候用subsampling

除去过多重复的词
$$
P(w_i) = (\sqrt{z(w_i)\over 0.001} + 1)\cdot{0.001\over z(w_i)}
$$
$z(w_i)$千分之7，一半

#### Get_batches

```python
[
  # First Batch
  [
    # Batch of Input
    [[ 1  2  3], [ 7  8  9]],
    # Batch of targets
    [[ 2  3  4], [ 8  9 10]]
  ],
 
  # Second Batch
  [
    # Batch of Input
    [[ 4  5  6], [10 11 12]],
    # Batch of targets
    [[ 5  6  7], [11 12 13]]
  ]
]
```
batch num and sequence length, sequence length perfers the amout of words in each line

#### `init_cell`

2-layer LSTM/RNN/GRU

#### `build_nn`

Input data (int docs)

-> embedding (int docs + vocab_size + embed_dim -> embeded vector)

-> build_rnn (init_cell + embeded vector -> rnn output (**there is no activate function in this step**))

-> a fully connected layer with a **linear activation** (logit - output before softmax)

#### `activiation function`

softmax (logit -> prob)

#### `sequence_loss`

sparse_softmax_cross_entropy_with_logits

#### gradient clipping

there are two steps in `tf.train.optimizer.minimize()`, and for gradient clipping, we add one more step

1. `compute_gradients`: return a list of tuples, which consist of `(gradient, variables)`
2. If gradient > max, set to max, if gradient < min, set to min
3. `apply_gradients`

为什么会有gradient explosion

为什么要使用gradient clipping

因为每一次输入都要计算一次gradient, $s_t = f(Ux_t + Ws_{t-1})$, 因此会出现w^n的情况

为什么LSTM解决了gradient vanishing

用sigmoid存储消息，而不是sigmoid的一阶导，且tanh提供更多的信息 

Weight sharing in CNN/RNN

### LSTM

Compared with RNN, LSTM add cell state, which carefully regulated by gates to remove or add information

#### Step1: Forget gate

$$
f_t = \sigma(W_f\cdot[h_{t-1}, x_t] + b_f)
$$

有多少输入信息需要被使用

#### Step2: What new info shall we store

Input gate, decides which value we should add
$$
i_t = \sigma(W_i\cdot[h_{t-1}, x_t] + b_i)
$$
How much we add to the odd state
$$
\hat C_t = tanh(W_C\cdot[h_{t-1}, x_t] + b_C)
$$
Update the new state
$$
C_t = f_t * C_{t-1} + i_t * \hat C_t
$$

#### Step3: Output

$$
o_t = \sigma(W_o[h_{t-1}, x_t] + b_o)
$$

$$
h_t = o_t * tanh(C_t)
$$

## Word2vec

For one input (1 * 10000) is multiplies by a 10000 * 300 matrx, return a 1 * 300 vector, assuming there are 300 features