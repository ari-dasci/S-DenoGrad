# S-noise-gradient

Noise reduction method via gradient optimisation

Transformer library is from the paper [Local Attention: Enhancing the Transformer Architecture for Efficient Time Series Forecasting](https://ieeexplore.ieee.org/abstract/document/10650762) and its code can be seen [here](https://github.com/ari-dasci/S-LAM).  


# Noise reduction algorithms

## autoencoders:

@inproceedings{10.1145/3394486.3403140,
author = {Cui, Ganqu and Zhou, Jie and Yang, Cheng and Liu, Zhiyuan},
title = {Adaptive Graph Encoder for Attributed Graph Embedding},
year = {2020},
isbn = {9781450379984},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3394486.3403140},
doi = {10.1145/3394486.3403140},
abstract = {Attributed graph embedding, which learns vector representations from graph topology and node features, is a challenging task for graph analysis. Recently, methods based on graph convolutional networks (GCNs) have made great progress on this task. However,existing GCN-based methods have three major drawbacks. Firstly,our experiments indicate that the entanglement of graph convolutional filters and weight matrices will harm both the performance and robustness. Secondly, we show that graph convolutional filters in these methods reveal to be special cases of generalized Laplacian smoothing filters, but they do not preserve optimal low-pass characteristics. Finally, the training objectives of existing algorithms are usually recovering the adjacency matrix or feature matrix, which are not always consistent with real-world applications. To address these issues, we propose Adaptive Graph Encoder (AGE), a novel attributed graph embedding framework. AGE consists of two modules: (1) To better alleviate the high-frequency noises in the node features, AGE first applies a carefully-designed Laplacian smoothing filter. (2) AGE employs an adaptive encoder that iteratively strengthens the filtered features for better node embeddings. We conduct experiments using four public benchmark datasets to validate AGE on node clustering and link prediction tasks. Experimental results show that AGE consistently outperforms state-of-the-artgraph embedding methods considerably on these tasks.},
booktitle = {Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery \& Data Mining},
pages = {976–985},
numpages = {10},
keywords = {adaptive learning, attributed graph embedding, graph convolutional networks, laplacian smoothing},
location = {Virtual Event, CA, USA},
series = {KDD '20}
}


https://www.researchgate.net/publication/340793215_Deep_Denoising_Autoencoder_for_Seismic_Random_Noise_Attenuation


P. Singh and A. Sharma, "Attention-Based Convolutional Denoising Autoencoder for Two-Lead ECG Denoising and Arrhythmia Classification," in IEEE Transactions on Instrumentation and Measurement, vol. 71, pp. 1-10, 2022, Art no. 4007710, doi: 10.1109/TIM.2022.3197757.
keywords: {Electrocardiography;Noise reduction;Databases;Feature extraction;Signal to noise ratio;Recording;Convolution;Arrhythmias;atrial fibrillation (AF);convolutional neural network (CNN);denoising autoencoder (DAE);electrocardiogram (ECG)},


W. -H. Lee, M. Ozger, U. Challita and K. W. Sung, "Noise Learning-Based Denoising Autoencoder," in IEEE Communications Letters, vol. 25, no. 9, pp. 2983-2987, Sept. 2021, doi: 10.1109/LCOMM.2021.3091800.
keywords: {Noise reduction;Training;Noise measurement;Random variables;Encoding;Decoding;Internet of Things;Machine learning;noise learning based denoising autoencoder;signal restoration;symbol demodulation;precise localization},

## SSA reconstruiction:

https://www.worldscientific.com/doi/abs/10.1142/S0219477510000289

https://arxiv.org/abs/2311.16198

