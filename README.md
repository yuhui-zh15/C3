# Connect, Collapse, Corrupt: Learning Cross-Modal Tasks with Uni-Modal Data

This repo provides the PyTorch source code of our paper: 
[Connect, Collapse, Corrupt: Learning Cross-Modal Tasks with Uni-Modal Data](https://openreview.net/forum?id=ttXg3SKAg5)

## Abstract

*Building cross-modal applications is challenging due to limited paired multi-modal data. Recent works have shown that leveraging a pre-trained multi-modal contrastive representation space enables cross-modal tasks to be learned from uni-modal data. This is based on the assumption that contrastive optimization makes embeddings from different modalities interchangeable. However, this assumption is under-explored due to the poorly understood geometry of the multi-modal contrastive space, where a modality gap exists. In our study, we provide a theoretical explanation of this space's geometry and introduce a three-step method, $C^3$ (Connect, Collapse, Corrupt), to bridge the modality gap, enhancing the interchangeability of embeddings. Our $C^3$ method significantly improves cross-modal learning from uni-modal data, achieving state-of-the-art results on zero-shot image / audio / video captioning and text-to-image generation.*

## Approach

![](.figures/figure1.png) ![](.figures/figure2.png)
**Figure: Overview of the motivation behind our approach, $C^3$, that enhances the interchangeable use of embeddings from different modalities.** Our work describes and provides a theoretical explanation of the unique geometry that arises from multi-modal contrastive learning where a modality gap and alignment noise exists in the learned representation space Building upon this observation, we present a straightforward technique, $C^3$, which enhances the interchangeability of embeddings between modalities, enabling the creation of cross-modal applications using only uni-modal data. 