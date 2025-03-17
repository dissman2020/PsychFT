### SFT

To run SFT:

```shell
conda activate psy

CUDA_VISIBLE_DEVICES=0,1 nohup \
  sh scripts/run_sft.sh \
  > "logs/nohup_console_logs/trian_$(date +%Y-%m-%d-%H-%M).log" \2>&1 &
```

To run test:
```shell
conda activate psy

CUDA_VISIBLE_DEVICES=0,1 nohup \
  sh scripts/run_test.sh \
  > "logs/nohup_console_logs/test_$(date +%Y-%m-%d-%H-%M).log" \2>&1 &
```