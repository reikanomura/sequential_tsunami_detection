# sequential_tsunami_detection

## 


```
┌── run2.py             メインコードです
├── ttsplit.py          テストケース（2）と学習ケース（664）を分けるサブルーチンです
├── POD.py              モード分解(POD)を行っているサブルーチンです
├── psudo_inv.py        テストケースの係数αを擬似的に求めるサブルーチンです。
├── particle_filter.py  粒子フィルタによる推定・更新のサブルーチンコードです
├── beautyfun.py        諸々の関数をまとめた煩雑なサブルーチンです
└── graphing.py         グラフ描画をコード実行時に並行して行うサブルーチンです。

```
## Required environment

We confirmed the code can run under the following environment.

- CentOS(Linux) Ver.7

- python 3.8.7 (via pyenv)
  - numpy==1.21.2
  - matplotlib==3.4.3
  - scipy==1.7.1
  - dask==2021.8.1
  - seaborn==0.11.2

