Note:
reward_func 最好支持多种调用方式，evaluate已经不好改了
1. 3+1+kwargs
2. 1+kwargs

ans = self.reward_func(eval_bpp, eval_msssim, eval_psnr, eval_sums, **self.reward_kwargs)
reward = self.reward_func(env)

目前
automl == ppo
lfs == reinforce

lfs.a
is sampled hyper-param

