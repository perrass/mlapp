# DL-RL-DQN

## 确定基本环境

1. 动作： 前，左，右，None
2. 环境：红绿灯，左右/正面车流
3. 不同环境下不同动作的reward
   1. 不同交通问题对应不同的violation, 如闯红灯，绿灯没有走，与对面车流相撞，与左右车流相撞
   2. 如果没有violation, 得到reward, 行动正常，没有行动但是红灯，其他情况
   3. reward受deadline限制, $2-penalty$, $penalty = \frac {{gradient}^{fnc}-1} {gradient-1}$, where $gradient=10, fnc=\frac {trial} {trial+state['deadline']} $

## 确定状态空间

```python
{
  'state-1': {
    'action-1': Qvalue-1,
    ...
  },
  'state-2': {
    'action-1': Qvalue-1,
    ...
  }
}
```

每一个state对应action及其Qvalue, state是各个状态变量的组合

## 确定学习策略

**$\epsilon$衰减策略**: 

1. $\epsilon$的概率自由选择，$1-\epsilon$的概率选择效用大的
2. $\epsilon$逐层衰减，直到一阈值，衰减函数: $\epsilon = a^t, \epsilon=\frac 1 {t^2}$
3. 初始值和阈值的设定，至少要包含len(self.Q)次随机，以保证学习到足够的情况和选择在进行优化

`self.Q[state][action] = (1-self.alpha)*self.Q[state][action]+self.alpha*(reward+gamma*max(Q(s', a)))`

1. `max(Q(s', a))`是未来奖励或者记忆中的奖励，记忆中下一个状态s'的动作中效用的最大值，我们倾向于下一个动作会吃到甜头的动作
2. 这个方式本质是指数系数衰减的加权平均，$\frac 1 2 ^t$
3. 这个项目中并未使用未来奖励，因为未来奖励是随机的，和一组数中随机选取一个数没有区别