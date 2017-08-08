# DL-NLP-RNN

### RNN

#### init docs

parse word and punctuation to int

-> get two dicts, int to word, and word to int

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
