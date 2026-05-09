# LoRA論文メモ

## 核心の数式
h = W0x + BAx * (α/r)

## 初期化
- A: Gaussian初期化
- B: ゼロ初期化
- 学習開始時にΔW=BAがゼロになるようにする

## 適用箇所
- TransformerのWqとWvのみ

## ハイパーパラメータ
- rank r: 論文ではr=4が多くのタスクで十分
- alpha: 最初のrと同じ値に設定