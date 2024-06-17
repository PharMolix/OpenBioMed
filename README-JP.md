<div align="center"><h1>OpenBioMed</h1></div>
<h4 align="center">
    <p>
        <b>日本語</b> |
        <a href="./README.md">英語</a> |
        <a href="./README-CN.md">中国語</a>
    <p>
</h4>

## 更新情報 🎉

- [2024/05/16] 🔥 **LangCell**の実装をリリースしました (📃[論文](https://arxiv.org/abs/2405.06708), 💻[コード](https://github.com/PharMolix/LangCell), 🤖[モデル](https://drive.google.com/drive/folders/1cuhVG9v0YoAnjW-t_WMpQQguajumCBTp?usp=sharing), 📎[引用](#LangCellを引用する))。

    > LangCellは、PharMolixとAI産業研究院（AIR）が共同開発した最初の「言語-細胞」マルチモーダル事前学習モデルです。このモデルは、細胞のアイデンティティ情報を豊富に含む知識豊かなテキストを学習することで、シングルセルトランスクリプトミクスの理解を効果的に向上させ、データ不足のシナリオでの細胞アイデンティティ理解タスクを解決します。LangCellは、ゼロショット細胞アイデンティティ理解を効果的に行うことができる唯一のシングルセルモデルであり、少数ショットおよびファインチューニングのシナリオでもSOTAを達成しています。LangCellは、OpenBioMedに近日中に統合される予定です。

- [2023/08/14] 🔥 **BioMedGPT-10B** (📃[論文](https://arxiv.org/abs/2308.09442v2), 🤖[モデル](https://pan.baidu.com/s/1iAMBkuoZnNAylhopP5OgEg?pwd=7a6b#list/path=%2F), 📎[引用](#BioMedGPTを引用する))、**BioMedGPT-LM-7B** (🤗[HuggingFaceモデル](https://huggingface.co/PharMolix/BioMedGPT-LM-7B))、**DrugFM** (🤖[モデル](https://pan.baidu.com/s/1iAMBkuoZnNAylhopP5OgEg?pwd=7a6b#list/path=%2F))の実装をリリースしました。

    > BioMedGPT-10Bは、PharMolixとAI産業研究院（AIR）が共同でリリースした、最初の商用マルチモーダル生物医学基盤モデルです。このモデルは、分子構造やタンパク質配列などの生命の言語と、人間の自然言語を組み合わせ、生物医学QAベンチマークで人間の専門家と同等のパフォーマンスを発揮し、分子やタンパク質のクロスモーダル質問応答タスクで強力なパフォーマンスを示します。BioMedGPT-LM-7Bは、Llama-2に基づいた最初の商用生物医学専用生成基盤モデルです。

    > DrugFMは、AI産業研究院（AIR）と北京人工知能研究院（BAAI）が共同で開発したマルチモーダル小分子基盤モデルです。このモデルは、小分子薬物の組織規則と表現学習に対してより細かい設計を行い、小分子薬物の事前学習モデルUniMapを形成し、マルチモーダル小分子基盤モデルMolFMと有機的に組み合わせています。このモデルは、クロスモーダル抽出タスクでSOTAを達成しています。

- [2023/06/12] **MolFM** (📃[論文](https://arxiv.org/abs/2307.09484), 🤖[モデル](https://pan.baidu.com/s/1iAMBkuoZnNAylhopP5OgEg?pwd=7a6b#list/path=%2F), 📎[引用](#MolFMを引用する)) と **CellLM** (📃[論文](https://arxiv.org/abs/2306.04371), 🤖[モデル](https://pan.baidu.com/s/1iAMBkuoZnNAylhopP5OgEg), 📎[引用](#CellLMを引用する)) の実装をリリースしました。

    > MolFMは、分子構造、生物医学文書、知識グラフを統合的に理解するマルチモーダル分子基盤モデルです。ゼロショットとファインチューニングの設定で、MolFMは既存のモデルをそれぞれ12.03%、5.04%上回ります。MolFMは、分子キャプション、テキストベースの分子生成、分子特性予測でも顕著な結果を達成しています。

    > CellLMは、正常細胞とがん細胞の両方で分割征服型コントラスト学習戦略を用いて同時に訓練された、最初の大規模細胞表現学習モデルです。CellLMは、細胞型注釈（71.8 vs 68.8）、少数ショットシナリオでのシングルセル薬物感受性予測（88.9 vs 80.6）、シングルオミクス細胞系薬物感受性予測でScBERTを上回ります（93.4 vs 87.2）。

- [2023/04/23] **BioMedGPT-1.6B** (🤖[モデル](https://pan.baidu.com/s/1iAMBkuoZnNAylhopP5OgEg)) と **OpenBioMed** の実装をリリースしました。

## 目次

- [紹介](#紹介)
- [インストール](#インストール)
- [クイックスタート](#クイックスタート)
- [ドキュメンテーション](./docs)
- [チュートリアル](#チュートリアル)
- [制限事項](#制限事項)
- [引用](#引用)

## 紹介

OpenBioMedは、AI強化生物医学のためのPython深層学習ツールキットです。OpenBioMedは、分子、タンパク質、単一細胞の分子構造、トランスクリプトーム、知識グラフ、生物医学テキストなど、マルチモーダル生物医学データへの簡単なアクセスを提供します。OpenBioMedは、従来のAI薬物発見タスクから新たに登場したマルチモーダルの課題まで、幅広い下流アプリケーションをサポートしています。

OpenBioMedは、研究者に以下の簡単なAPIを提供します。

- **分子、タンパク質、単一細胞の3つの異なるモダリティ**: 分子構造またはトランスクリプトーム、生物医学テキスト、知識グラフ。OpenBioMedは、これらのモダリティにアクセスし、処理し、融合するための統一されたパイプラインを研究者に提供します。
- **10の下流タスク**、薬物-標的結合親和性予測や分子特性予測などのAI薬物発見（AIDD）タスク、分子キャプションやテキストベースの分子生成などのマルチモーダルタスクに分類されます。
- **20以上の深層学習モデル**、[BioMedGPT-10B](https://pan.baidu.com/s/1iAMBkuoZnNAylhopP5OgEg?pwd=7a6b#list/path=%2F)、[MolFM](https://arxiv.org/abs/2307.09484)、[CellLM](https://arxiv.org/abs/2306.04371)などが含まれます。研究者は、異なるコンポーネントを柔軟に組み合わせて自分のモデルを作成できます。
- **20以上のデータセット**、AI駆動型生物医学研究で最も人気のあるデータセットが含まれます。豊富なモデルの組み合わせと包括的な評価による再現可能なベンチマークが提供されます。

OpenBioMedの主な特徴は以下の通りです。

- **統一されたデータ処理パイプライン**: 異なる生物医学エンティティとモダリティからの異種データを簡単にロードし、変換し、統一された形式にすることができます。
- **即時利用可能な推論**: 公開されている事前学習済みモデルと推論デモを提供し、独自のデータやタスクに簡単に転送できます。
- **再現可能なモデル動物園**: 既存および新しいアプリケーションで最先端のモデルを柔軟に複製および拡張することができます。

以下の表は、OpenBioMedでサポートされているタスク、データセット、モデルを示しています。これは継続的な取り組みであり、リストをさらに拡大する予定です。

|                  タスク                   |         サポートされているデータセット         |              サポートされているモデル               |
| :-------------------------------------: | :--------------------------------: | :-----------------------------------------: |
|          クロスモーダル検索          |               PCdes                |   KV-PLM, SciBERT, MoMu, GraphMVP, MolFM    |
|           分子キャプション           |              ChEBI-20              |   MolT5, MoMu, GraphMVP, MolFM, BioMedGPT   |
|     テキストベースの分子生成      |              ChEBI-20              |         MolT5, SciBERT, MoMu, MolFM         |
|       分子質問応答       |             ChEMBL-QA              |           MolT5, MolFM, BioMedGPT           |
|       タンパク質質問応答        |             UniProtQA              |                  BioMedGPT                  |
|        細胞型分類         |          Zheng68k, Baron           |               scBERT, CellLM                |
|  シングルセル薬物反応予測   |                GDSC                |            DeepCDR, TGSA, CellLM            |
|      分子特性予測       |            MoleculeNet             | MolCLR, GraphMVP, MolFM, DeepEIK, BioMedGPT |
| 薬物-標的結合親和性予測 | Yamanishi08, BMKG-DTI, DAVIS, KIBA |         DeepDTA, MGraphDTA, DeepEIK         |
| タンパク質-タンパク質相互作用予測  |      SHS27k, SHS148k, STRING       |         PIPR, GNN-PPI, OntoProtein          |


## インストール

1. (オプション) conda環境の作成:

```bash
conda create -n OpenBioMed python=3.9
conda activate OpenBioMed
```

2. 必要なパッケージのインストール:

```
pip install -r requirements.txt
```

3. Pygの依存関係のインストール:

```bash
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-(your_torch_version)+(your_cuda_version).html
pip install torch-geometric
# 上記のPyTorch関連パッケージのインストールに問題がある場合は、https://pytorch.org/get-started/locally/ および https://github.com/pyg-team/pytorch_geometric の指示が役立つ場合があります。PyTorch Geometricおよびその拡張機能をhttps://data.pyg.org/whl/ から直接インストールすることも便利です。
```

**注**: 一部の下流タスクでは、追加のパッケージが必要になる場合があります。

## クイックスタート

[Jupytor notebooks](./examples) と [ドキュメンテーション](./docs) をチェックして、すぐに始めましょう！

| 名前                                                         | 説明                                                  |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [BioMedGPT-10Bの推論](./examples/biomedgpt_inference.ipynb) | 分子やタンパク質に関する質問にBioMedGPT-10Bを使用する例。 |
| [MolFMによるクロスモーダル検索](./examples/cross_modal_retrieval) | 分子に最も関連するテキスト記述をMolFMで検索する例。 |
| [MolT5によるテキストベースの分子生成](./examples/molecule_generation.ipynb) | テキスト記述に基づいてMolT5で分子のSMILES文字列を生成する例。 |
| [CellLMによる細胞型分類](./examples/cell_type_classification.ipynb) | 微調整されたCellLMを使用して細胞型を分類する例。   |
| [分子特性予測](./docs/dp.md)                 | 分子特性予測タスクのトレーニングとテストのパイプライン |
| [薬物反応予測](./docs/drp.md)                    | 薬物反応予測タスクのトレーニングとテストのパイプライン |
| [薬物-標的結合親和性予測](./docs/dti.md)     | 薬物-標的結合親和性予測タスクのトレーニングとテストのパイプライン |
| [分子キャプション](./docs/molcap.md)                      | 分子キャプションタスクのトレーニングとテストのパイプライン  |

## チュートリアル
水木分子チームには、さまざまな分野からの仲間がいます。あなたも私たちと同じように、大規模モデル技術とその垂直分野での応用に興味があるかもしれません。そのため、興味のある方々に、エンジニアがモデルを実行するところから始めて、大規模モデルを体験し、使用し、自分の大規模モデルを開発するまでのプロセスを詳細に説明する一連の記事を書くことは意義深い作業です。皆さんが異なるバックグラウンドを持っていることを考慮し、非常に詳細に各ステップを提示する一連の記事を書くことにしました。具体的なサンプルコードは[チュートリアル](./examples/course/)に移動してください。    
**記事リスト：**    
[01. 自分のコンピュータで大規模モデルを実行する方法は？（Windows/Mac）](https://mp.weixin.qq.com/s/8--ddNKE0QC1mONycDRiSw) | [サンプルコード](./examples/course/01_run_biomedgpt_lm_on_your_computer.ipynb)

## 制限事項

このリポジトリにはBioMedGPT-LM-7BとBioMedGPT-10Bが含まれており、これらのモデルの責任ある使用を強調します。BioMedGPTは、一般公開のサービス提供には**使用しないでください**。BioMedGPTを使用して、直接的または間接的に、国家の憲法、法律、行政規則に反する内容を生成したり、国家の政権を転覆させたり、国家の安全や利益を損なったり、テロリズム、過激主義、人種的憎悪や差別、暴力、ポルノ、虚偽の有害情報などを広めたりすることは厳禁です。BioMedGPTは、ユーザーが提供または公開するいかなるコンテンツ、データ、情報に起因するいかなる結果に対しても責任を負いません。

## ライセンス

このリポジトリは[MITライセンス](./LICENSE)の下でライセンスされています。BioMedGPT-LM-7BおよびBioMedGPT-10Bモデルの使用は、[使用許諾契約](./USE_POLICY.md)に従う必要があります。

## お問い合わせ

フレームワークの改善に役立つフィードバックをお待ちしております。技術的な質問や提案がある場合は、遠慮なくGitHubのissueに投稿してください。商業サポートや協力に関心がある場合は、[opensource@pharmolix.com](mailto:opensource@pharmolix.com)までご連絡ください。


## 引用

私たちのオープンソースコードやモデルがあなたの研究に役立つと思われる場合は、このリポジトリに🌟スターを付けて📎引用してください。ご支援ありがとうございます！

##### OpenBioMedを引用する場合:

```
@misc{OpenBioMed_code,
      author={Luo, Yizhen and Yang, Kai and Hong, Massimo and Liu, Xing Yi and Zhao, Suyuan and Zhang, Jiahuan and Wu, Yushuai and Nie, Zaiqing},
      title={Code of OpenBioMed},
      year={2023},
      howpublished={\url{https://github.com/BioFM/OpenBioMed.git}}
}
```

##### BioMedGPTを引用する場合:

```
@misc{luo2023biomedgpt,
      title={BioMedGPT: Open Multimodal Generative Pre-trained Transformer for BioMedicine}, 
      author={Yizhen Luo and Jiahuan Zhang and Siqi Fan and Kai Yang and Yushuai Wu and Mu Qiao and Zaiqing Nie},
      year={2023},
      eprint={2308.09442},
      archivePrefix={arXiv},
      primaryClass={cs.CE}
}
```

##### DeepEIKを引用する場合:

```
@misc{luo2023empowering,
      title={Empowering AI drug discovery with explicit and implicit knowledge}, 
      author={Yizhen Luo and Kui Huang and Massimo Hong and Kai Yang and Jiahuan Zhang and Yushuai Wu and Zaiqing Nie},
      year={2023},
      eprint={2305.01523},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
##### MolFMを引用する場合:
```
@misc{luo2023molfm,
      title={MolFM: A Multimodal Molecular Foundation Model}, 
      author={Yizhen Luo and Kai Yang and Massimo Hong and Xing Yi Liu and Zaiqing Nie},
      year={2023},
      eprint={2307.09484},
      archivePrefix={arXiv},
      primaryClass={q-bio.BM}
}
```

##### CellLMを引用する場合:
```
@misc{zhao2023largescale,
      title={Large-Scale Cell Representation Learning via Divide-and-Conquer Contrastive Learning}, 
      author={Suyuan Zhao and Jiahuan Zhang and Zaiqing Nie},
      year={2023},
      eprint={2306.04371},
      archivePrefix={arXiv},
      primaryClass={cs.CE}
}
```

##### LangCellを引用する場合:
```
@misc{zhao2024langcell,
      title={LangCell: Language-Cell Pre-training for Cell Identity Understanding}, 
      author={Suyuan Zhao and Jiahuan Zhang and Yizhen Luo and Yushuai Wu and Zaiqing Nie},
      year={2024},
      eprint={2405.06708},
      archivePrefix={arXiv},
      primaryClass={q-bio.GN}
}
```
