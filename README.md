# Cuda::NMF
[![Gem Version](https://badge.fury.io/rb/cuda-nmf.svg)](https://badge.fury.io/rb/cuda-nmf)

非負値行列因子分解 (NMF) を NVIDIA 製 GPU で行う Gem です。データは numo-narray で扱います。

注意: まだ開発段階のため正しく動作しない場合があります。

Note: it is still in the development stage and may not work properly.

## インストール
```bash
gem install cuda-nmf
```

## ビルド
### コンパイル
```bash
rake compile
```

### Gem の生成
```bash
rake build
```

### インストール先について
```bash
gem contents cuda-nmf
```

### Gem のインストール
```bash
rake install
```
