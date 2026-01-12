# GeometryCrafter: Consistent Geometry Estimation for Open-world Videos with Diffusion Priors

**Tian-Xing Xu¹** $\quad$ **Xiangjun Gao³** $\quad$ **Wenbo Hu²$\dagger$** $\quad$ **Xiaoyu Li²** $\quad$ **Song-Hai Zhang¹$\dagger$** $\quad$ **Ying Shan²**
¹Tsinghua University $\quad$ ²ARC Lab, Tencent PCG $\quad$ ³HKUST

**Project Page:** [https://geometrycrafter.github.io](https://geometrycrafter.github.io)

---

![Figure 1](https://replicate.delivery/xpbkg/H8fGqfX6Y8vBVSf5V3o55w63q684f8X4V68f6Xf8X64fX8f6X/file.jpg)
**Figure 1. We present GeometryCrafter, a novel approach that estimates temporally consistent, high-quality point maps from open-world videos, facilitating downstream applications such as 3D/4D reconstruction and depth-based video editing or generation.**

---

### Abstract
Despite remarkable advancements in video depth estimation, existing methods exhibit inherent limitations in achieving geometric fidelity through the affine-invariant predictions, limiting their applicability in reconstruction and other metrically grounded downstream tasks. We propose **GeometryCrafter**, a novel framework that recovers high-fidelity point map sequences with temporal coherence from open-world videos, enabling accurate 3D/4D reconstruction, camera parameter estimation, and other depth-based applications. At the core of our approach lies a point map Variational Autoencoder (VAE) that learns a latent space agnostic to video latent distributions for effective point map encoding and decoding. Leveraging the VAE, we train a video diffusion model to model the distribution of point map sequences conditioned on the input videos. Extensive evaluations on diverse datasets demonstrate that GeometryCrafter achieves state-of-the-art 3D accuracy, temporal consistency, and generalization capability.

---

### 1. Introduction
Inferring 3D geometry from 2D observations remains a long-standing challenge in computer vision, serving as a fundamental pillar for numerous applications, ranging from autonomous navigation and virtual reality to 3D/4D reconstruction and generation. However, its inherently ill-posed nature poses persistent difficulties in achieving reliable and consistent geometry estimation from diverse open-world videos.

Pioneered by Marigold, recent methods harness diffusion models to generate affine-invariant depth maps or sequences, which is achieved by recasting depth cues as pseudo-RGB frames that are suitable for Variational Autoencoder (VAE) processing. Although these methods exhibit remarkable spatial and temporal fidelity, the compression of unbounded depth values into the fixed input range of the VAE inevitably leads to a non-trivial information loss, especially for distant scene elements, as shown in Fig. 2. Moreover, the absence of camera intrinsics and the presence of unknown shift values impede accurate 3D reconstruction, thereby limiting their utility in downstream applications. Another line of research uses pretrained image foundation models to directly estimate metric depth or point maps. However, neglecting temporal context often induces flickering artifacts when applying these methods to videos.

In this paper, we propose a novel approach, named GeometryCrafter, to estimate high-fidelity and temporally coherent point maps from open-world videos. These point maps facilitate 3D/4D point cloud reconstruction, camera pose estimation, and the derivation of temporally consistent depth maps and camera intrinsics. Our method exhibits robust zero-shot generalization capabilities by exploiting the inherent video diffusion priors of natural videos. Central to our approach is a novel point map VAE, tailored to effectively encode and decode unbounded 3D coordinates without compressing depth values into a bounded range. It contains a dual-encoder architecture: an encoder inherited from the original video VAE to capture the primary point map information, and a newly designed residual encoder to embed the remaining information in a latent offset. Leveraging this design, we can preserve the latent space analogous to the video VAE by regulating the adjusted latent code with the original video decoder. This analogous latent distribution enables the utilization of pre-trained diffusion weights for robust zero-shot generalizations.

For VAE training, we disentangle the point map into log-space depth and diagonal field of view, rather than directly encoding 3D coordinates in the camera coordinate system or adopting a cuboid-based representation as in prior works. This disentangled representation demonstrates enhanced suitability for the VAE to capture the intrinsic structure of the point map, largely attributed to its location invariance and resolution independence. For supervision, we augment the standard reconstruction objective with a normal loss, a multi-scale depth loss to enhance local geometric fidelity, and a regularization term penalizing deviations from the original latent distribution. Furthermore, our GeometryCrafter integrates a diffusion U-net that generates point map latents from video latents, forming a robust framework for producing high-fidelity and temporally coherent point maps from open-world videos.

We comprehensively evaluate GeometryCrafter on diverse datasets, ranging from static to dynamic scenes, indoor to outdoor environments, and realistic to cartoonish styles. Our method significantly outperforms existing methods by a large margin, both qualitatively and quantitatively. Extensive ablation studies validate the effectiveness of our proposed components, and demonstrate the applicability of our method to 3D/4D point cloud reconstruction and camera pose estimation. Our contributions are summarized as follows:
* We present GeometryCrafter, a novel approach for estimating high-fidelity and temporally coherent geometry from diverse open-world videos.
* We propose a point map VAE for effective encoding and decoding of point maps, which employs a dual-encoder architecture to maintain the latent space analogous to the inherited video VAE for generalization ability.
* We introduce the disentangled point map representation and multi-scale depth loss to train the VAE, significantly improving the robustness and fidelity of our method.

---

![Figure 2](https://replicate.delivery/xpbkg/p6fGqfX6Y8vBVSf5V3o55w63q684f8X4V68f6Xf8X64fX8f6X/file.jpg)
**Figure 2. Diffusion-based depth estimation methods, e.g., DepthCrafter and DAV, suffer from significant metric errors in distant regions due to the compression of unbounded depth values into the bounded input range of VAEs.**

---

### 2. Related Works
**Monocular depth estimation (MDE).** MDE methods predict depth maps from single images or videos. To achieve zero-shot generalization, MiDaS introduces affine-invariant supervision and trains on mixing datasets. Depth Anything and its V2 extend this framework to transformer-based architectures and semi-supervised learning, using large-scale unlabeled images for improved generalization. Pioneered by Marigold, recent works adapt pretrained image diffusion models to MDE by converting depth maps to pseudo-RGB representations and exclusively finetuning the U-net on depth latent codes, achieving superior quality and robustness. To generalize to videos, previous methods employ test-time optimization, memory mechanisms, or stabilization networks for temporal coherence, whereas recent studies finetune video diffusion models to yield high-quality, temporally consistent depth sequences. However, these methods ignore the camera intrinsic estimation and provide only affine-invariant depth, which is scale and shift ambiguous, hindering 3D accuracy and downstream applications that require projection into 3D space.

**Monocular geometry estimation (MGE).** To overcome these limitations, MGE methods jointly infer camera parameters and metric or up-to-scale depth maps. LeRes utilizes 3D point cloud encoders to recover missing shift and focal parameters during depth estimation. UniDepth decouples camera parameter prediction from depth estimation via a pseudo-spherical 3D representation and a camera self-prompting mechanism. DepthPro introduces a ViT-based design for high-resolution depth estimation, coupled with a dedicated image encoder for focal length prediction. DUSt3R projects two-view images or identical pairs to scale-invariant point maps in camera space, facilitating the derivation of camera intrinsic and depth maps. MoGe employs affine-invariant point maps to mitigate focal-distance ambiguity, achieving state-of-the-art performance. Besides, Metric3D and its V2 rely on user-provided camera parameters to estimate metrically accurate depth maps. However, these approaches are restricted to static images and incur flickering artifacts when directly applied to video sequences.

**Steganography and information hiding.** Steganography visually hides secret information within existing features. Previous works have demonstrated the capacity of neural networks to embed visual data within images or videos, such as invertible down-scaling, grayscaling, and mono-nizing. The most relevant work to ours is LayerDiffuse, which conceals image transparency information within a small perturbation in the latent space of Stable Diffusion. Adhering to the same insight, we encode point maps into the latent space of diffusion models while preserving the underlying distribution, facilitating the utilization of pretrained diffusion models for geometry estimation.

---

### 3. Method
Given an input RGB video $\mathbf{v} \in \mathbb{R}^{T \times H \times W \times 3}$, we aim to predict a temporally consistent point map sequence $\mathbf{p} \in \mathbb{R}^{T \times H \times W \times 3}$ alongside a valid mask $\mathbf{m} \in [0, 1]^{T \times H \times W}$ to exclude undefined regions (e.g., sky). Each point map contains the 3D coordinates $\mathbf{p} = (x_p, y_p, z_p)^T$ in the camera coordinate system for every pixel. To this end, we propose GeometryCrafter, a novel approach that leverages video diffusion models (VDMs) for robust point map estimation from open-world videos. We model the joint distribution $\mathcal{P}(\mathbf{p}, \mathbf{m} | \mathbf{v})$ in the latent space. While VDMs' native VAE effectively encodes video frames and masks, accurate point map representation necessitates a dedicated VAE tailored for geometric encoding and decoding.

![Figure 3](https://replicate.delivery/xpbkg/B8fGqfX6Y8vBVSf5V3o55w63q684f8X4V68f6Xf8X64fX8f6X/file.jpg)
**Figure 3. Architecture of our point map VAE.** The point map VAE encodes and decodes point maps with unbounded values, alleviating the inaccurate prediction in distant regions. We adopt a dual-encoder design: the native encoder $\mathcal{E}_{SVD}$ inherited from SVD captures normalized disparity maps, while a residual encoder $\mathcal{E}_\epsilon$ embeds remaining information as an offset. It preserves the original latent space by regulating the latents via the original decoder $\mathcal{D}_{SVD}$, enabling the utilization of pretrained diffusion priors. A point map decoder $\mathcal{D}_{pmap}$ recovers the final point maps from the latent codes.

#### 3.1. Architecture of Point Map VAE
Existing diffusion-based depth estimation methods simply employ the native VAE to encode and decode only partial information from the point maps, i.e. the normalized disparity maps $\tilde{\mathbf{x}}_{disp}$:

$$ \tilde{\mathbf{x}}_{disp} = 2 \times \frac{\mathbf{x}_{disp} - \min(\mathbf{x}_{disp})}{\max(\mathbf{x}_{disp}) - \min(\mathbf{x}_{disp})} - 1, $$
$$ \mathbf{x}_{disp} = b \cdot f / z_p, \quad (1) $$

where $b$ is the baseline, $f$ is the focal length, and $z_p$ is the $z$-coordinate of the point map $\mathbf{p}$. However, such normalization often misestimates depths in distant regions (Fig. 2), resulting in geometric distortions due to compressing unbounded depths into the VAE’s fixed input range.

To this end, we propose a point map VAE that directly handles point maps over the unbounded range $[0, +\infty]$. Crucially, its latent distribution should be tightly aligned with that of the native VAE to fully exploit pre-trained VDMs. Inspired by LayerDiffuse, we propose a dual-encoder architecture: the inherited native encoder $\mathcal{E}_{SVD}$ captures the primary point map features, while a newly designed residual encoder $\mathcal{E}_\epsilon$ encodes remaining information as an offset (see Fig. 3). Given that the normalized disparity maps $\tilde{\mathbf{x}}_{disp}$ encapsulate significant relative depth cues, we employ $\mathcal{E}_{SVD}$ on $\tilde{\mathbf{x}}_{disp}$ and harness $\mathcal{E}_\epsilon$ to embed the residual information into the offset. The final point map latent is obtained by their summation:

$$ \mathbf{z}_{pmap} = \mathcal{E}_{SVD}(\tilde{\mathbf{x}}_{disp}) + \mathcal{E}_\epsilon(\mathbf{p}, \mathbf{m}, \tilde{\mathbf{x}}_{disp}). \quad (2) $$

This dual-encoder architecture allows us to explicitly regularize the latent space of $\mathbf{z}_{pmap}$ to avoid disrupting the original latent distribution. Considering most VAEs in VDM are diagonal Gaussian models (i.e. mean and variance), we apply the offset solely to the mean, retaining the original variance for simplicity. For decoding, we design a dedicated decoder $\mathcal{D}_{pmap}$ to reconstruct both the point map $\hat{\mathbf{p}}$ and the valid mask $\hat{\mathbf{m}}$:

$$ \hat{\mathbf{p}}, \hat{\mathbf{m}} = \mathcal{D}_{pmap}(\mathbf{z}_{pmap}). \quad (3) $$

To ensure temporal consistency, we employ temporal layers in the decoder to capture the temporal dependencies across frames.

#### 3.2. Training of Point Map VAE
**Point map representation.** The points $\mathbf{p} = (x_p, y_p, z_p)^T$ are scattered non-uniformly across the view frustum, resulting in a complex spatial distribution that poses a challenge to deep networks in capturing their inherent structure. To mitigate this, existing point map estimation methods assume a centered camera principal point and remap depth values into log-space, thereby projecting the points into a cuboid domain:

$$ \mathbf{p}_{cuboid} = [x_p/z_p, y_p/z_p, \log z_p]. \quad (4) $$

However, this representation is suboptimal for our point map VAE. In particular, the first two channels of $\mathbf{p}_{cuboid}$ encode ray directions from the camera center to each pixel, conveying location-specific information that diverges from the translation-invariant nature of RGB features. To address this discrepancy, we propose decoupling the point map into:

$$ \mathbf{p}_{dec} = [\theta_{diag}, \log z_p], \quad (5) $$

where $\theta_{diag} = \sqrt{W^2 + H^2} / 2f$ denotes the diagonal field of view, a constant map for all points in a frame. Since $\mathbf{p}_{dec}$ is independent of spatial location, it is more suitable for our VAE to learn an effective latent distribution. Moreover, this formulation enables us to train our network only on fixed-resolution videos, while generalizing to varying resolutions and aspect ratios, owing to the invariance of $\theta_{diag}$. The original point map $\mathbf{p}$ can be effortlessly recovered from $\mathbf{p}_{dec}$ via the inverse perspective transformation.

![Figure 4](https://replicate.delivery/xpbkg/Q8fGqfX6Y8vBVSf5V3o55w63q684f8X4V68f6Xf8X64fX8f6X/file.jpg)
**Figure 4. Diffusion UNet.** We jointly condition the diffusion model on video latents and per-frame geometry priors from an image MGE model $\mathcal{M}_{img}$. The geometry is encoded into latent space via our point map VAE, while the video latents are obtained from the native VAE.

**Loss functions.** To train the point map VAE, we define reconstruction loss $\mathcal{L}_{recon}$ as the $L_1$ norm between the decoded depth and diagonal field of view and their ground truth counterparts. Besides, we also impose a mask loss $\mathcal{L}_{mask}$ to exclude undefined regions, e.g. sky, as the $L_2$ norm between the predicted and ground truth valid masks. To promote surface quality, we introduce a normal loss $\mathcal{L}_n$ that supervises the normal maps derived from the reconstructed point maps and the ground truth, as well as a multi-scale depth loss $\mathcal{L}_{ms}$ that measures the alignment between reconstructed and ground truth depth maps within local regions, inspired by MoGe. Importantly, to regularize our latent space agnostic to the original SVD’s latent distribution, we employ a loss term $\mathcal{L}_{identity}$ to penalize the latent deviation:

$$ \mathcal{L}_{identity} = ||\tilde{\mathbf{x}}_{disp} - \mathcal{D}_{SVD}(\mathbf{z}_{pmap})||_2^2. \quad (6) $$

The final training objective $\mathcal{L}_{VAE}$ is defined as:

$$ \mathcal{L}_{VAE} = \underbrace{\mathcal{L}_{recon} + \mathcal{L}_{ms} + \lambda_n \mathcal{L}_n}_{\mathcal{L}_{pmap}} + \mathcal{L}_{identity} + \lambda_{mask} \mathcal{L}_{mask}. \quad (7) $$

Please refer to the supplementary material for more details on the loss functions.

#### 3.3. Diffusion UNet
Since our point map VAE is meticulously designed and regularized to align closely with the original SVD’s latent distribution, we can train a diffusion UNet to estimate point maps from videos with only synthetic data, using the pretrained generative prior of video diffusion models. Although it significantly alleviates the issue of lacking high-quality point map annotations in real-world videos, the synthetic data still suffers from limited diversity in camera intrinsics, which may degrade the generalization ability of diagonal field-of-view prediction on real-world scenarios. To mitigate this, alongside video latents, we propose the integration of per-frame geometry priors as conditioning inputs within the diffusion UNet, as shown in Fig. 4. We employ our point map VAE to encode the per-frame point maps predicted by MoGe into the latent space to act as geometry priors that provide strong camera-intrinsic clues, although they may suffer from inaccuracies and flickering.

Following DepthCrafter, we train the diffusion UNet with the EDM pre-conditioning and noise schedule, and adopt a multi-stage training strategy to capture long temporal context under GPU memory constraints. After training, the UNet can process videos with varying lengths (e.g. 1 to 110 frames) at a time, and we adopt the stitching inference strategy to handle videos with arbitrary lengths. Besides, inspired by recent advancements in reformulating the diffusion process into a deterministic single-step framework for depth estimation, we also train a deterministic variant by removing the noisy latent from input.

---

### 4. Experiments
#### 4.1. Implementation Details
We build GeometryCrafter upon the SVD framework. The residual encoder and point decoder in the point map VAE adopt the same architecture as in SVD’s VAE, supplemented by zero convolution in the output layers. We collected 14 synthetic RGBD datasets, comprising 1.85M frames, for training. Among these, 11 datasets can form 12K video clips with up to 150 frames each. For training stability, we normalize point clouds with a shared scale factor across frames, yielding up-to-scale point clouds akin to structure-from-motion. We first train the point map VAE from scratch on RGBD images with an AdamW optimizer at a learning rate of $10^{-4}$ for 40K iterations, then finetune on video data for an additional 20K iterations. The diffusion UNet is finetuned with a learning rate of $10^{-5}$ for 40K and 30K iterations in two stages. All experiments are conducted on 8 GPUs and take about 3 days. Further details are in the supplementary material.

#### 4.2. Quantitative and Qualitative Evaluation
**Evaluation protocol.** For evaluation, we employ seven datasets unseen during training: **GMU Kitchens** and **ScanNet** are captured with Kinect for indoor scenes; **DDAD** and **KITTI** are collected via lidar sensors for outdoor driving; **Monkaa** and **Sintel** are synthetic datasets with precise depth annotations and challenging dynamics; and **DIODE** is a high-resolution image dataset with far-range depth maps. Besides, we also qualitatively evaluate on **DAVIS**, **DL3DV**, **Sora**-generated, and open-world videos. To assess up-to-scale point map quality, we use the relative point error $\text{Rel}^p$ and percentage of inliers $\delta^p$ (threshold 0.25), following MoGe. We align predicted point maps with ground truth by optimizing a *shared scale factor across the entire video* for all methods. We also evaluate derived depth sequences using the absolute relative error $\text{Rel}^d$ and the inlier percentage $\delta^d$ (threshold 1.25), following.

**Table 1. Evaluation on point map estimation.** Results are aligned with the ground truth by optimizing a shared scale factor across the entire video. $\text{Rel}^p$ and $\delta^p$ are in percentage. The best and second-best results are highlighted in **bold** and <u>underline</u>, respectively. “G” denotes the diffusion version of our model and “D” denotes the deterministic variant.

| Method | GMU Kitchen [21] | Monkaa [48] | Sintel [7] | ScanNet [12] | DDAD [24] | KITTI [20] | DIODE [65] | Avg. Rank |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| | $\text{Rel}^p \downarrow$ $\delta^p \uparrow$ | $\text{Rel}^p \downarrow$ $\delta^p \uparrow$ | $\text{Rel}^p \downarrow$ $\delta^p \uparrow$ | $\text{Rel}^p \downarrow$ $\delta^p \uparrow$ | $\text{Rel}^p \downarrow$ $\delta^p \uparrow$ | $\text{Rel}^p \downarrow$ $\delta^p \uparrow$ | $\text{Rel}^p \downarrow$ $\delta^p \uparrow$ | ↓ |
| DUSt3R [70]† | 22.2 68.2 | 37.0 45.1 | 43.9 35.6 | 13.3* 87.7* | 37.8 37.3 | 17.2 87.8 | 20.0 85.3 | 6.3 |
| MonST3R [89]† | 24.1 64.4 | 40.2 32.6 | 40.0 34.1 | 13.6* 87.2* | 40.7 25.9 | 24.0 58.1 | 22.2 79.4 | 7.4 |
| MonST3R [89]‡ | 11.4 91.5 | 36.2 42.6 | 38.6 35.9 | 6.13* 97.6* | 40.0 29.3 | 25.0 58.7 | 22.2 79.4 | 5.4 |
| UniDepth [56] | 9.96 94.1 | 23.0 65.6 | 30.9 50.6 | 6.54* 98.4* | 23.0 64.9 | **4.24** **99.3** | 16.1 88.0 | 2.9 |
| DepthPro [5] | 14.0 86.5 | 29.5 50.4 | 45.0 36.3 | 10.5* 93.6* | 39.8 43.1 | 12.8 93.6 | 18.6 87.1 | 5.6 |
| MoGe [69] | 21.3 69.1 | 28.0 58.1 | 31.2 52.0 | 13.5 88.0 | 16.0 85.5 | 8.51 95.7 | 13.5 **93.5** | 4.4 |
| Ours(D) | <u>8.88</u> **94.3** | <u>18.8</u> **79.7** | 25.9 62.8 | **8.92** **96.4** | 15.6 89.0 | 6.73 98.4 | **13.0** 92.8 | 2.0 |
| Ours(G) | **8.52** **94.3** | 20.5 <u>75.5</u> | **25.6** **64.9** | <u>9.16</u> <u>96.0</u> | **15.0** **90.6** | <u>6.34</u> <u>98.7</u> | <u>13.1</u> 92.7 | **1.9** |
| | $\text{Rel}^d \downarrow$ $\delta^d \uparrow$ | $\text{Rel}^d \downarrow$ $\delta^d \uparrow$ | $\text{Rel}^d \downarrow$ $\delta^d \uparrow$ | $\text{Rel}^d \downarrow$ $\delta^d \uparrow$ | $\text{Rel}^d \downarrow$ $\delta^d \uparrow$ | $\text{Rel}^d \downarrow$ $\delta^d \uparrow$ | $\text{Rel}^d \downarrow$ $\delta^d \uparrow$ | ↓ |
| DUSt3R [70]† | 21.6 64.0 | 35.8 41.9 | 41.3 36.0 | 13.1* 84.5* | 32.3 46.5 | 10.9 87.9 | 15.9 85.4 | 6.7 |
| MonST3R [89]† | 22.6 61.9 | 38.8 31.4 | 37.5 33.4 | 13.3* 84.4* | 30.7 46.4 | 9.06 91.8 | 17.4 80.8 | 6.6 |
| MonST3R [89]‡ | 8.91 90.7 | 33.9 41.3 | 35.9 36.1 | 5.18* 97.1* | 31.9 46.1 | 13.4 80.4 | 17.4 80.8 | 5.3 |
| UniDepth [56] | **8.11** **93.7** | 20.8 60.0 | 28.4 48.5 | <u>5.32</u>* **98.0*** | 22.9 63.4 | **3.45** **99.1** | 11.5 89.9 | 2.6 |
| DepthPro [5] | 14.0 83.1 | 28.4 45.2 | 43.2 34.3 | 10.0* 90.7* | 38.3 40.6 | 9.47 91.5 | 12.6 87.4 | 6 |
| MoGe [69] | 20.6 64.7 | 25.7 54.8 | 29.2 49.0 | 13.3 84.9 | 14.6 85.2 | 7.69 94.1 | **8.13** **93.5** | 4.3 |
| Ours(D) | 8.51 93.4 | **16.1** **77.1** | **22.2** **65.7** | **7.88** **95.5** | <u>12.3</u> <u>88.4</u> | 5.88 97.6 | 10.2 92.1 | 2.3 |
| Ours(G) | <u>8.30</u> <u>93.6</u> | <u>18.3</u> <u>71.5</u> | <u>22.6</u> <u>63.7</u> | 8.39 95.0 | **12.0** **90.4** | <u>5.44</u> <u>98.2</u> | <u>10.0</u> 92.4 | **2.1** |

\*: Not strictly zero-shot (trained on ScanNet or ScanNet++); †: Inference with duplicated frames; ‡: Post-optimization with external data.

---

**Evaluation on point maps.** We compare our method with representative point map estimation approaches, e.g., DUSt3R, MonST3R, UniDepth, DepthPro, and MoGe. Among them, DUSt3R and MonST3R are designed for two-view scenarios, addressing static and dynamic scenes, respectively, and are evaluated by inputting two identical frames. For MonST3R, we also evaluate with its post-processing, which requires external optical flows to refine global point clouds and poses. As shown in Tab. 1, our method outperforms others on most benchmarks, with substantial gains on the challenging Monkaa and Sintel datasets. Although UniDepth shows better performance on KITTI (likely due to training on DrivingStereo with a shared LiDAR sensor), our approach attains a superior average rank. For the image benchmark DIODE, our method still achieves competitive performance compared to methods specialized for static images. Notably, some methods are trained on ScanNet or ScanNet++, violating the zero-shot evaluation, yet our method sustains comparable accuracy on ScanNet. Moreover, visual comparisons in Fig. 5 indicate that only our method can produce temporally consistent point maps with fine-grained details, while others (UniDepth and MoGe) exhibit issues like flickering or blurred details.

![Figure 5](https://replicate.delivery/xpbkg/H8fGqfX6Y8vBVSf5V3o55w63q684f8X4V68f6Xf8X64fX8f6X/file.jpg)
**Figure 5. Qualitative comparison of point map estimation.** Disparity maps are derived from estimated point maps via Eq. (1). The green boxes highlight temporal profiles of the disparity maps, sliced along the time axis at the green lines. Zoom in for better visualization.

**Evaluation on depth maps.** To compare our method with cutting-edge monocular depth estimation methods, e.g., ChronoDepth, DepthCrafter, DAV, and DepthAnything (DA) V1 and V2, we follow the evaluation protocol in [31], except for a center crop to meet the aspect ratio requirement (0.5 to 2). As shown in Tab. 2, our method achieves the best performance on almost all video datasets and remains competitive even on the image dataset DIODE. The qualitative comparison in Fig. 6 demonstrates that our method generates superior depth maps and point clouds, e.g., the potato chip bucket and the plant in the first two examples. In driving scenarios, such as the third example, DepthCrafter and DAV predict infinity values for distant buildings, resulting in missing structures, whereas our method consistently produces regular structures and plausible depth values, even when the ground truth exceeds the LiDAR sensor’s range.

**Table 2. Evaluation on depth map estimation.** Results are aligned with the ground truth by optimizing a shared scale factor and shift across the entire video. $\text{Rel}^d$ and $\delta^d$ are in percentage. The best and second-best results are highlighted in **bold** and <u>underline</u>, respectively. “G” denotes the diffusion version of our model and “D” denotes the deterministic variant.

| Method | GMU Kitchen [21] | Monkaa [48] | Sintel [7] | ScanNet [12] | DDAD [24] | KITTI [20] | DIODE [65] | Avg. Rank |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| | $\text{Rel}^d \downarrow$ $\delta^d \uparrow$ | $\text{Rel}^d \downarrow$ $\delta^d \uparrow$ | $\text{Rel}^d \downarrow$ $\delta^d \uparrow$ | $\text{Rel}^d \downarrow$ $\delta^d \uparrow$ | $\text{Rel}^d \downarrow$ $\delta^d \uparrow$ | $\text{Rel}^d \downarrow$ $\delta^d \uparrow$ | $\text{Rel}^d \downarrow$ $\delta^d \uparrow$ | ↓ |
| DA [81] | 18.0 68.2 | 24.1 62.1 | 37.2 59.8 | 11.3 87.8 | 13.4 85.1 | 8.70 92.4 | **6.94** 94.6 | 4.1 |
| DA V2 [82] | 19.3 65.4 | 24.0 61.5 | 40.6 55.0 | 12.3 85.1 | 13.9 84.7 | 11.1 87.0 | **6.94** 95.0 | 5.1 |
| ChronoDepth [62] | 20.1 65.2 | 35.1 52.6 | 45.1 57.9 | 14.3 81.4 | 34.6 45.8 | 15.0 79.8 | 12.2 90.6 | 6.9 |
| DepthCrafter [31] | 13.8 80.6 | 23.4 73.8 | 30.5 67.0 | 11.3 87.3 | 15.6 80.7 | 9.96 89.6 | 12.6 86.2 | 4.9 |
| DAV [80] | 10.8 89.4 | 19.0 72.3 | 35.7 67.3 | 8.83 92.7 | **12.3** 85.4 | 7.13 95.3 | 7.47 93.7 | 3.1 |
| Ours(D) | <u>8.28</u> <u>93.2</u> | **12.0** **83.5** | **16.3** **74.3** | **7.27** **96.1** | 13.4 86.2 | <u>5.60</u> <u>97.7</u> | 7.00 **96.2** | <u>1.9</u> |
| Ours(G) | **8.03** **94.0** | <u>13.0</u> <u>80.5</u> | <u>16.9</u> <u>73.2</u> | <u>7.57</u> <u>95.9</u> | <u>12.7</u> **87.5** | **5.25** **98.3** | 7.03 <u>96.1</u> | **2.0** |

---

#### 4.3. Ablation Study
**Effectiveness of point map VAE.** We conduct ablation studies by removing the point map VAE and using SVD’s VAE to encode normalized disparity maps while keeping other components unchanged and retraining the model. As shown in Tab. 3, the estimated depth maps exhibit a significant performance drop across various datasets, except for KITTI, where the ground truth is constrained by the LiDAR sensor’s range. The visual comparison in Fig. 7 reveals that the performance decline is due to information loss from compressing unbounded depth values into a bounded range, leading to the neglect of distant objects.

**Components in point map VAE.** We perform ablation studies to examine the effectiveness of the point map representation, multi-scale loss $\mathcal{L}_{ms}$, temporal layers in the decoder, and latent alignment. As shown in Tab. 4, the decoupled point map representation Eq. (5) markedly enhances reconstruction fidelity. Results in the second to fourth rows also highlight the importance of multi-scale supervision in the spatial domain and contextual information in the temporal domain. Eliminating the latent alignment component not only increases point map errors (see the last two rows of Tab. 4), but also hinders effectively leveraging video diffusion priors. As shown in Tab. 5 and Fig. 8, latent alignment substantially improves the quality and robustness of point map predictions.

![Figure 6](https://replicate.delivery/xpbkg/D6fGqfX6Y8vBVSf5V3o55w63q684f8X4V68f6Xf8X64fX8f6X/file.jpg)
**Figure 6. Qualitative comparison of depth map estimation.** We transform point maps and disparity maps into metric depth maps for better visualization of distant regions. Zoom in for better visualization.

![Figure 7](https://replicate.delivery/xpbkg/K8fGqfX6Y8vBVSf5V3o55w63q684f8X4V68f6Xf8X64fX8f6X/file.jpg)
**Figure 7. Comparison on disparity (top) and depth (bottom) quality between our full model and the w/o point map VAE variant.**

**Table 3. Ablation study on the effectiveness of point map VAE.**

| | GMU Kitchen | Monkaa | Sintel | ScanNet | KITTI |
| :--- | :---: | :---: | :---: | :---: | :---: |
| | $\text{Rel}^d \downarrow$ $\delta^d \uparrow$ | $\text{Rel}^d \downarrow$ $\delta^d \uparrow$ | $\text{Rel}^d \downarrow$ $\delta^d \uparrow$ | $\text{Rel}^d \downarrow$ $\delta^d \uparrow$ | $\text{Rel}^d \downarrow$ $\delta^d \uparrow$ |
| w/o | 9.50 90.8 | 16.1 79.2 | 25.2 72.8 | 8.11 95.0 | **5.00** 97.9 |
| w/ | **8.03** **94.0** | **13.0** **80.5** | **16.9** **73.2** | **7.57** **95.9** | 5.25 **98.3** |

---

**Table 4. VAE reconstruction performance with different components.** Light gray background highlights our final VAE configuration.

| | | | | ScanNet | Sintel | Monkaa |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Representation | $\mathcal{L}_{ms}$ | Temporal layers | Latent alignment | $\text{Rel}^p \downarrow$ $\delta^p \uparrow$ $\text{Rel}^d \downarrow$ $\delta^d \uparrow$ | $\text{Rel}^p \downarrow$ $\delta^p \uparrow$ $\text{Rel}^d \downarrow$ $\delta^d \uparrow$ | $\text{Rel}^p \downarrow$ $\delta^p \uparrow$ $\text{Rel}^d \downarrow$ $\delta^d \uparrow$ |
| Eq. (4) | ✓ | | ✓ | 7.67 99.8 2.03 99.8 | 8.25 94.4 6.46 94.8 | 6.05 99.5 3.93 99.4 |
| Eq. (5) | ✓ | | ✓ | **1.63** 99.8 <u>1.51</u> 99.7 | <u>4.24</u> 97.8 4.02 97.9 | 2.19 **99.6** 2.07 **99.5** |
| Eq. (5) | | | ✓ | 2.95 99.8 2.31 99.6 | 5.32 97.3 4.72 97.5 | 2.83 99.5 2.68 99.4 |
| <span style="background-color: #f0f0f0">Eq. (5)</span> | <span style="background-color: #f0f0f0">✓</span> | <span style="background-color: #f0f0f0">✓</span> | <span style="background-color: #f0f0f0">✓</span> | <span style="background-color: #f0f0f0"><u>1.65</u> **99.9** **1.47** **99.9**</span> | <span style="background-color: #f0f0f0">**3.45** **98.1** **3.06** **98.1**</span> | <span style="background-color: #f0f0f0">**2.01** 99.5 **1.84** <u>99.4</u></span> |
| Eq. (5) | ✓ | ✓ | | 1.95 **99.9** 1.77 **99.9** | 4.48 98.0 3.78 97.9 | 2.94 99.1 2.64 98.9 |

---

**Table 5. UNet prediction performance with different components.** Light gray background highlights our final UNet configuration.

| | | GMU Kitchen | Sintel | DDAD |
| :--- | :---: | :---: | :---: | :---: |
| Latent alignment | Per-frame geometry prior | $\text{Rel}^p \downarrow$ $\delta^p \uparrow$ $\text{Rel}^d \downarrow$ $\delta^d \uparrow$ | $\text{Rel}^p \downarrow$ $\delta^p \uparrow$ $\text{Rel}^d \downarrow$ $\delta^d \uparrow$ | $\text{Rel}^p \downarrow$ $\delta^p \uparrow$ $\text{Rel}^d \downarrow$ $\delta^d \uparrow$ |
| | ✓ | 9.51 93.6 9.26 92.6 | **25.4** **65.5** 23.0 61.7 | **14.3** 90.1 12.5 89.6 |
| ✓ | | 12.9 88.7 12.0 84.7 | 34.4 38.8 24.7 57.9 | 25.5 58.7 15.7 78.3 |
| <span style="background-color: #f0f0f0">✓</span> | <span style="background-color: #f0f0f0">✓</span> | <span style="background-color: #f0f0f0">**8.52** **94.3** **8.30** **93.6**</span> | <span style="background-color: #f0f0f0">25.6 64.9 **22.6** **63.7**</span> | <span style="background-color: #f0f0f0">15.0 **90.6** **12.0** **90.4**</span> |
| DUSt3R | | 22.2 68.2 21.6 64.0 | 43.9 35.6 41.3 36.0 | 37.8 37.3 32.3 46.5 |
| Ours(G) + DUSt3R | | **12.2** **90.4** **11.5** **88.1** | **34.4** **39.9** **26.2** **55.8** | **28.7** **48.8** **15.9** **80.9** |

---

**UNet design.** We investigate the impact and robustness of per-frame geometry priors derived from MoGe by excluding them from the UNet input and replacing MoGe with DUSt3R. As shown in Tab. 5, per-frame priors benefit the model across diverse scenarios by compensating for limited camera intrinsics in the training data. Moreover, replacing MoGe with DUSt3R also consistently improves performance, confirming the robustness of our method to different priors. Besides, we present two variants of the UNet: one with the diffusion framework (noted as Ours(G)) and the other with a deterministic scheme (noted as Ours(D)). As shown in Tab. 1 and Tab. 2, the deterministic approach exhibits slightly lower accuracy but achieves a 1.1× acceleration in inference speed, e.g. 4.1 v.s. 3.7 FPS at a $448 \times 768$ resolution on our experimental setup. Users may choose one of the two variants based on their requirements for speed or accuracy.

![Figure 8](https://replicate.delivery/xpbkg/A8fGqfX6Y8vBVSf5V3o55w63q684f8X4V68f6Xf8X64fX8f6X/file.jpg)
**Figure 8. Effectiveness of latent alignment.**

#### 4.4. Applications
**3D/4D reconstruction.** With our temporally consistent, high-quality point maps, we enable 3D/4D reconstruction, whose cornerstone is the camera pose estimation. To this end, if dynamic objects exist, we first obtain their masks using SegmentAnything and XMem. Then, we detect interest points in the static regions with SuperPoint and track them via SpaTracker. Finally, we optimize the camera poses with the established correspondences by 3D geometric constraints, taking only a few minutes to converge. Examples of 3D/4D reconstruction are shown in Fig. 1 and supplementary materials.

**Depth-conditioned video generation.** Depth sequences are pivotal to controllable video generation, capturing the inherent 3D structures of videos. Our consistent depth maps serve directly as conditioning inputs in existing depth-driven methods (e.g., Control-A-Video), enabling creative outputs as shown in Fig. 9.

![Figure 9](https://replicate.delivery/xpbkg/R8fGqfX6Y8vBVSf5V3o55w63q684f8X4V68f6Xf8X64fX8f6X/file.jpg)
**Figure 9. Application of depth-conditioned video generation. The prompt is “a car is drifting on roads, snowy day, artstation”.**

---

### 5. Conclusion
We present GeometryCrafter, a novel method that estimates temporally consistent, high-quality point maps from open-world videos, facilitating downstream applications such as 3D/4D reconstruction and depth-based video editing or generation. Our core design is a point map VAE that learns a latent space agnostic to original video latent distribution, enabling effective encoding and decoding of unbounded point map values. We also introduce a decoupled point map representation to eliminate the location-dependent characteristics of point maps, enhancing the robustness to resolutions and aspect ratios. Furthermore, we integrate a per-frame geometry prior conditioned diffusion model to model the distribution of point sequences conditioned on the input videos. Comprehensive evaluations confirm that our method outperforms prior methods in performance and generalization. Its main limitation is relatively high computational and memory overhead due to the large model size.

### Acknowledgement
Tian-Xing Xu completed this work during his internship at Tencent ARC Lab. The project was supported by the Tsinghua-Tencent Joint Laboratory for Internet Innovation Technology.

---

### References
(Note: References [1]-[95] are listed in the original document across pages 9-12.)

---

# GeometryCrafter: Consistent Geometry Estimation for Open-world Videos with Diffusion Priors
## Supplementary Material

### 1. Datasets
#### 1.1. Training Datasets
We collect 14 open-source synthetic RGBD datasets to facilitate the training of GeometryCrafter, among which 11 can be composited into video sequences. To construct the training video dataset, we extract non-overlapping segments with a sequence length not exceeding 150 frames. An overview of the training datasets is provided in Tab. 1.

**Table 1. An overview of the training datasets.**
| Dataset | Domain | #Frames | #Videos |
| :--- | :--- | :---: | :---: |
| 3DKenBurns [50] | In-the-wild | 76K | 526 |
| DynamicReplica [34] | Indoor/Outdoor | 145K | 1126 |
| GTA-SfM [66] | Outdoor/In-the-wild | 19K | 234 |
| Hypersim [58] | Indoor | 75K | × |
| IRS [67] | Indoor | 103K | 722 |
| MatrixCity [42] | Outdoor/Driving | 452K | 3029 |
| MidAir [16] | Outdoor/In-the-wild | 357K | 2433 |
| MVS-Synth [32] | Outdoor/Driving | 12K | 120 |
| Spring [49] | In-the-wild | 5K | 49 |
| Structured3D [94] | Indoor | 71K | × |
| Synthia [60] | Outdoor/Driving | 178K | 1276 |
| TartanAir [71] | In-the-wild | 306K | 2245 |
| UrbanSyn [22] | Outdoor/Driving | 7K | × |
| VirtualKitti2 [8] | Driving | 43K | 320 |
| **Total** | - | **1.85M** | **12K** |

#### 1.2. Evaluation Datasets
We exhaustively evaluate GeometryCrafter and previous state-of-the-art methods using seven datasets with ground truth labels that remain entirely unseen during the training phase.
* **GMU Kitchens [21]:** All scenarios are employed for evaluation. Video resolution $960 \times 540$.
* **ScanNet [12]:** 100 scenes from test split. Resolution $624 \times 464$.
* **DDAD [24]:** 50 sequences from validation split. Resolution $640 \times 384$.
* **KITTI [20]:** Initial 110 frames resulting in 13 videos. Resolution $736 \times 368$.
* **Monkaa [48]:** 9 scenes. Resolution $960 \times 540$.
* **Sintel [7]:** All sequences in training split. Resolution $872 \times 436$.
* **DIODE [65]:** 771 images from validation split.

### 2. Loss Functions of VAE and UNet
(Details the reconstruction loss $\mathcal{L}_{recon}$, normal loss $\mathcal{L}_n$, multi-scale depth loss $\mathcal{L}_{ms}$, identity loss $\mathcal{L}_{identity}$, mask loss $\mathcal{L}_{mask}$, and the diffusion objective $\mathcal{L}_{UNet}$.)

### 3. More Implementation Details
(Describes reusing SVD's VAE architecture, zero convolution initialization, training stages, batch sizes, learning rates ($1e-4$ for VAE, $1e-5$ for UNet), and compute resources (8 GPUs, ~3 days training).)

### 4. Camera Pose Estimation
(Mathematical formulation for optimizing camera poses $W_1 \dots W_T$ by minimizing reprojection error between frames using tracked points and scale-invariant depth.)

**Table 2. Inference time of different components on $448 \times 768$ videos with 110 frames.**
| Method | Per-frame Prior | Encoder | UNet | Decoder | Total |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Ours(G) | 0.1 | 0.04 | 0.04 | 0.08 | 0.27s/frame |
| Ours(D) | 0.1 | 0.04 | 0.01 | 0.08 | 0.24s/frame |

### 5. Limitations
The major limitation is expensive computation and memory cost due to large model sizes. The decoder is the bottleneck during inference.

### 6. More results
(Reference to Figures 1-4 in the supplementary showing results on Sora videos, DL3DV, and DAVIS datasets.)

---

![Supplemental Figure 1](https://replicate.delivery/xpbkg/N8fGqfX6Y8vBVSf5V3o55w63q684f8X4V68f6Xf8X64fX8f6X/file.jpg)
**Figure 1. Visual results on Sora-generated videos. The rows from left to right are the input videos, the disparity maps and the point cloud of the first frame.**

![Supplemental Figure 2](https://replicate.delivery/xpbkg/S8fGqfX6Y8vBVSf5V3o55w63q684f8X4V68f6Xf8X64fX8f6X/file.jpg)
**Figure 2. Visual comparison with monocular geometry estimation methods. All point maps are converted to disparity maps for better visualization the sharpness of depth prediction.**

![Supplemental Figure 3](https://replicate.delivery/xpbkg/U8fGqfX6Y8vBVSf5V3o55w63q684f8X4V68f6Xf8X64fX8f6X/file.jpg)
**Figure 3. Visual results on DL3DV [45] with camera poses estimated from the output point maps. We concatenate 8 aligned point maps from the original point map sequence for visualization.**

![Supplemental Figure 4](https://replicate.delivery/xpbkg/W8fGqfX6Y8vBVSf5V3o55w63q684f8X4V68f6Xf8X64fX8f6X/file.jpg)
**Figure 4. Visual results on DAVIS [54] with camera poses estimated from the output point maps. We concatenate 8 aligned point maps from the original point map sequence for visualization.**