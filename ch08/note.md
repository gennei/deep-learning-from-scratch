# 8章 ディープラーニング

## 8.1 ネットワークをより深く

## 8.2 ディープラーニングの小歴史
- 2012年のコンペでディープラーニングを使ったものが1位になった
  - 2012 AlexNet
  - 2013 Clarifi
  - 2014 VGG
  - 2014 GoogLeNet
  - 2015 ResNet

## 8.3 ディープラーニングの高速化
ボトルネックは畳み込み層で行われている演算
畳み込み層では積和演算が行われている

### GPU

### 分散学習
結論:TensorFlowのようなライブラリを使え

### 演算ビットの削減
ビットが多いほどバス帯域の通信料が増える

### 参考
「Google I/O 2017」で機械学習チップTPUのセカンドバージョン
https://news.nifty.com/article/technology/techall/12158-1628816/

## 8.4 ディープラーニングの実用例
### 物体検出
1. 画像に写っているオブジェクトの領域を検出
1. 領域に対して、CNNを適用しクラス分類

### セグメンテーション
ピクセル単位でクラス分類を行う問題

### 画像キャプション生成
ヒカキン動画字幕生成botのことではない

手法:NIC
CNNとRNNの組合せ

## 8.5 ディープラーニングの未来

### 画像スタイル変換
「コンテンツ画像」と「スタイル画像」をインプットとし、アウトプットとして、スタイル風の「コンテンツ画像」を出力する。


#### 参考
機械学習したAIがレンブラントの"新作"を出力。絵具の隆起も3D再現した「The Next Rembrandt」公開
http://www.huffingtonpost.jp/engadget-japan/the-next-rembrandt_b_9631702.html

### 画像生成
#### DCGAN
- 画像を生成する人と、識別する人がいる

TensorFlowによるDCGANでアイドルの顔画像生成
http://memo.sugyan.com/entry/20160516/1463359395

### 自動運転

### 強化学習(DQN)
環境とエージェントの相互作用。
報酬を最大化する問題
リアルタイム性のゲームは難しい

Google Deepmind、人工知能『DQN』を開発。レトロゲームを自力で学習、人間に勝利
http://japanese.engadget.com/2015/02/26/google-deepmind-dqn/