# SURDS: Self-Supervised Attention-guided Reconstruction and Dual Triplet Loss for Writer Independent Offline Signature Verification, ICPR 2022.

**Soumitri Chattopadhyay**, Siladittya Manna, Saumik Bhattacharya and Umapada Pal, [**"SURDS: Self-Supervised Attention-guided Reconstruction and Dual Triplet Loss for Writer Independent Offline Signature Verification"**](https://arxiv.org/abs/2201.10138), _26th International Conference on Pattern Recognition_ **(ICPR)**, 2022.

## Abstract
Offline Signature Verification (OSV) is a fundamental biometric task across various forensic, commercial and legal applications. The underlying task at hand is to carefully model fine-grained features of the signatures to distinguish between genuine and forged ones, which differ only in minute deformities. This makes OSV more challenging compared to other verification problems. In this work, we propose a two-stage deep learning framework that leverages self-supervised representation learning as well as metric learning for writer-independent OSV. First, we train an image reconstruction network using an encoder-decoder architecture that is augmented by a 2D spatial attention mechanism using signature image patches. Next, the trained encoder backbone is fine-tuned with a projector head using a supervised metric learning framework, whose objective is to optimize a novel dual triplet loss by sampling negative samples from both within the same writer class as well as from other writers. The intuition behind this is to ensure that a signature sample lies closer to its positive counterpart compared to negative samples from both intra-writer and cross-writer sets. This results in robust discriminative learning of the embedding space. To the best of our knowledge, this is the first work of using self-supervised learning frameworks for OSV. The proposed two-stage framework has been evaluated on two publicly available offline signature datasets and compared with various state-of-the-art methods. It is noted that the proposed method provided promising results outperforming several existing pieces of work.

## Citation
If you find this article useful in your research, consider citing us:
```
@inproceedings{chattopadhyay2022surds,
    author = {Soumitri Chattopadhyay and Siladittya Manna and Saumik Bhattacharya and Umapada Pal},
    title = {SURDS: Self-Supervised Attention-guided Reconstruction and Dual Triplet Loss for Writer Independent Offline Signature Verification},
    booktitle = {International Conference on Pattern Recognition (ICPR)},
    year = {2022}
}
```

**Code coming soon !**
