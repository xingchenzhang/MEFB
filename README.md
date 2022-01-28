# MEFB: Benchmarking and Comparing Multi-exposure Image Fusion Algorithms
This is the official webpage of MEFB, which is a multi-exposure image fusion benchmark.

**MEFB is the first benchmark in the field of multi-exposure image fusion (MEF)**, aiming to provide a platform to perform fair and comprehensive performance comparision of MEF methods. Currently, **100 image pairs, 21 fusion algorithms and 20 evaluation metrics** are integrated in MEFB, which can be utilized to compare performances conveniently. All the fusion results are also available that can be used by users directly. In addition, more test images, fusion algorithms (in Matlab), evaluation metrics and fused images can be easily added using the provided toolkit.

For more details, please refer to the following paper:

**Benchmarking and Comparing Multi-exposure Image Fusion Algorithms**  
Xingchen Zhang  
Information Fusion, Vol. 74, pp. 111-131, 2021.  
From Imperial College London  
Contact: xingchen.zhang@imperial.ac.uk  
[[Download paper](https://www.sciencedirect.com/science/article/pii/S1566253521000233)]

**If you find this work useful, please cite**:

	@article{zhang2021benchmarking,
	  title={Benchmarking and comparing multi-exposure image fusion algorithms},
	  author={Zhang, Xingchen},
	  journal={Information Fusion},
	  year={2021},
	  volome = {74},
	  pages = {111-131},
	  publisher={Elsevier}
	}

## Abstract
Multi-exposure image fusion (MEF) is an important area in computer vision and has attracted increasing
interests in recent years. Apart from conventional algorithms, deep learning techniques have also been
applied to MEF. However, although many efforts have been made on developing MEF algorithms, the lack
of benchmarking studies makes it difficult to perform fair and comprehensive performance comparison among
MEF algorithms, thus hindering the development of this field significantly. In this paper, we fill this gap by
proposing a benchmark of multi-exposure image fusion (MEFB), which consists of a test set of 100 image pairs,
a code library of 21 algorithms, 20 evaluation metrics, 2100 fused images, and a software toolkit. To the best
of our knowledge, this is the first benchmarking study in the field of MEF. This paper also gives a literature
review on MEF methods with a focus on deep learning-based algorithms. Extensive experiments have been
conducted using MEFB for comprehensive performance evaluation and for identifying effective algorithms.
We expect that MEFB will serve as an effective platform for researchers to compare the performance of MEF
algorithms.

## What are contained in MEFB

### Dataset
The dataset in MEFB is a test set. A part of the dataset is created by the author. A part of the dataset is collected by the authors from the [Internet](https://www.ino.ca/en/solutions/video-analytics-dataset/) and from existing datasets (details will be provided later).  We appreciate the authors of these datasets very much for making these images publicly available for research. **Please also cite these papers if you use MEFB**. Thanks! 

![](https://github.com/xingchenzhang/MEFB/blob/main/dataset.jpg)

### Methods integrated
Currently, we have integrated 21 MEF algorithms in MEFB. Many thanks for the authors of these algorithms for making their codes available to the community. **Please cite these papers as well if you use MEFB**. Thanks!

1. DeepFuse [1] [[Download](https://val.serc.iisc.ernet.in/DeepFuseICCV17/)]
2. DEM [2] [[Download](https://github.com/QTWANGBUAA/exposure-fusion)]
3. DSIFT_EF [3] [[Download](https://github.com/yuliu316316/DSIFT-EF)]
4. FMMEF [4] [[Download](https://github.com/xiaohuiben/fmmef-TIP-2020)]
5. FusionDN [5] [[Download](https://github.com/hanna-xu/FusionDN)] 
6. GD [6] [[Download](https://uk.mathworks.com/matlabcentral/fileexchange/48782-multi-exposure-and-multi-focus-image-fusion-in-gradient-domain)] 
7. GFF [7] [[Download](http://xudongkang.weebly.com/)]
8. IFCNN [8] [[Download](https://github.com/uzeful/IFCNN)] 
9. MEFAW [9] [[Download](https://github.com/tkd1088/multi-exposure-image-fusion)]
10. MEFCNN [10] [[Download](https://github.com/xiaohuiben/MEF-CNN-feature)] 
11. MEFDSIFT [11] [[Download](https://github.com/ImranNust/Source-Code)] 
12. MEF-GAN [12] [[Download](https://github.com/jiayi-ma/MEF-GAN)] 
13. MEFNet [13] [[Download](https://github.com/makedede/MEFNet)]
14. MEFOpt [14] [[Download](https://kedema.org/Publications.html)] 
15. MGFF [15] [[Download](https://www.mathworks.com/matlabcentral/fileexchange/72451-multi-scale-guided-image-and-video-fusion?s_tid=prof_contriblnk)] 
16. MTI [16] [[Download](https://github.com/emmmyiyang/MEF-Two-Images)] 
17. PMEF [17] [[Download](https://github.com/hangxiaotian/Perceptual-Multi-exposure-Image-Fusion)] 
18. PMGI [18] [[Download](https://github.com/jiayi-ma/PMGI_AAAI2020)] 
19. PWA [19] [[Download](https://kedema.org/Publications.html)] 
20. SPD-MEF [20] [[Download](https://kedema.org/Publications.html)] 
21. U2Fusion [21] [[Download](https://github.com/hanna-xu/U2Fusion)] 

The download links of each algorithm can also be found on [this Chinese website](https://zhuanlan.zhihu.com/p/340781608). For each algorithm, we use original settings reported by corresponding authors in their papers. For deep learning-based methods, the pretrained model provided by corresponding authors are used. We did not retrain these algorithms.

#### Algorithms written in Matlab
Please download the codes in Matlab (DEM, DSIFT_EF, FMMEF, GD, GFF, MEFAW, MEFCNN, MEFDSIFT, MEFOpt, MGFF, MTI, PMEF, PWA, SPD_MEF) using the links provided above, and then put these algorithms in \methods. You will need to change the interface of these algorithms to use.  

#### Algorithms written in Python or other languages
For algorithms written in Python or other languages, we ran them and changed the name of the fused images and put them in the \output\fused_images folder. If your algorithm is in Python or other languages, please generate the fused images first and change their names. After that, put the fused imgaes into \output\fused_images. 

### Evaluation metrics integrated
We have integrated 20 evaluation metrics in MEFB. The codes were collected from the Internet, forum, etc. and checked by the author.

Many thanks to the authors of these evaluation metric codes for sharing their codes with the community. This is very helpful for the research in this field. Many thanks!

1. Cross entropy (CE) [22]
2. Entropy (EN) [23]
3. Feature mutual information (FMI) [24,25]
4. Nomalized mutual information (NMI) [26]
5. Peak signal-to-noise ratio (PSNR) [27]
6. Nonliner correlation information entropy (QNCIE) [28,29]
7. Tsallis entropy (TE) [30]
8. Average gradient (AG) [31]
9. Edge intensity (EI) [32]
10. Gradient-based similarity measurement (QABF) [33]
11. Phase congruency (QP) [34]
12. Standard division (SD) [35]
13. Spatial frequency (SF) [36]
14.  Cvejie's metric (QC) [37]
15.  Peilla's metric (QW) [38]
16.  Yang's metric (QY) [39]
17.  MEF structural similarity index measure (MEF-SSIM) [40]
18.  Human visual perception (QCB) [41]
19.  QCV [42]
20.  VIF [43]

### Fused images using 21 MEF alrogithms
Please download the fused images from [Google Drive](https://drive.google.com/file/d/1qB3UwFDWWe1Uq5L7aJPj2Hzkdat94Mu0/view?usp=sharing) or [Baidu Netdisk](https://pan.baidu.com/s/1FBj4hGvuFEdcIDFKKtcw8w) (code: mefb), and put the images into \output\fused_images

### Examples of fused images
![](https://github.com/xingchenzhang/MEFB/blob/main/flower.png)


## How to use
### How to run algorithms
1. Add algorithms into \methods
2. Please set the algorithms you want to run in util\configMethods.m
3. Please set the images you want to fuse in util\configImgs, and change the path of these images
4. main_running.m is used to run the fusion algorithms. Please change the output path in main_running.m.
5. Enjoy!


### How to compute evaluation metrics
1. Please set the metrics you want to compute in util\configMetrics.m
2. compute_metrics.m is used to compute evaluation metrics. Please change the output path in compute_metrics.m
3. Enjoy!

### How to add algorithms (or fused images)
1. For methods written in MATLAB, please put them in the folder \methods. For example, for method "DEM", put the codes inside a folder called "DEM", and put the folder "DEM" inside \methods. Then change the main file of DEM to run_DEM.m. In run_DEM.m, please change the interface as according to the provided examples.
2. For algorithms written in Python or other languages, we suggest the users change the name of the fused images according to examples we provided and put them in the \output\fused_images folder. Then add the methods in util\configMethods.m. Then, the evaluation metrics can be computed.

## Acknowledgement
The overall framework of MEFB is created based on OTB [44] and VIFB [45]. We thank the authors of OTB very much for making OTB publicly available. We also thank all authors of the integrated images, MEF methods and evaluation metrics (especially Dr. Zheng Liu [46], https://github.com/zhengliu6699/imageFusionMetrics)  for sharing their work to the community! 

### References
[1] K.R. Prabhakar, V.S. Srikar, R.V. Babu, Deepfuse: A deep unsupervised approach for exposure fusion with extreme exposure image pairs, in: 2017 IEEE International Conference on Computer Vision (ICCV), IEEE, 2017, pp. 4724–4732.  
[2] Q. Wang, W. Chen, X. Wu, Z. Li, Detail-enhanced multi-scale exposure fusion in yuv color space, IEEE Trans. Circuits Syst. Video Technol. 26 (3) (2019) 1243–1252.  
[3] Y. Liu, Z. Wang, Dense sift for ghost-free multi-exposure fusion, J. Vis. Commun. Image Represent. 31 (2015) 208–224.  
[4] H. Li, K. Ma, H. Yong, L. Zhang, Fast multi-scale structural patch decomposition for multi-exposure image fusion, IEEE Trans. Image Process. 29 (2020) 5805–5816.  
[5] H. Xu, J. Ma, Z. Le, J. Jiang, X. Guo, Fusiondn: a unified densely connected network for image fusion, in: Proceedings of the AAAI Conference on Artificial Intelligence, 2020, pp. 12484–12491.  
[6] S. Paul, I.S. Sevcenco, P. Agathoklis, Multi-exposure and multi-focus image fusion in gradient domain, J. Circuits Syst. Comput. 25 (10) (2016) 1650123.  
[7] S. Li, X. Kang, J. Hu, Image fusion with guided filtering, IEEE Trans. Image Process. 22 (7) (2013) 2864–2875.  
[8] Y. Zhang, Y. Liu, P. Sun, H. Yan, X. Zhao, L. Zhang, IFCNN: A general image fusion framework based on convolutional neural network, Inf. Fusion 54 (2020) 99–118.  
[9] S.-h. Lee, J.S. Park, N.I. Cho, A multi-exposure image fusion based on the adaptive weights reflecting the relative pixel intensity and global gradient, in: 2018 25th IEEE International Conference on Image Processing, IEEE, 2018, pp. 1737–1741.  
[10] H. Li, L. Zhang, Multi-exposure Fusion with CNN Features, in: 2018 25th IEEE International Conference on Image Processing, 2018, pp. 1723–1727.  
[11] N. Hayat, M. Imran, Ghost-free multi exposure image fusion technique using dense sift descriptor and guided filter, J. Vis. Commun. Image Represent. 62 (2019) 295–308.  
[12] H. Xu, J. Ma, X.-P. Zhang, MEF-GAN: multi-exposure image fusion via generative adversarial networks, IEEE Trans. Image Process. 29 (2020) 7203–7216.  
[13] K. Ma, Z. Duanmu, H. Zhu, Y. Fang, Z. Wang, Deep guided learning for fast multi-exposure image fusion, IEEE Trans. Image Process. 29 (2020) 2808–2819.  
[14] K. Ma, Z. Duanmu, H. Yeganeh, Z. Wang, Multi-exposure image fusion by optimizing a structural similarity index, IEEE Trans. Comput. Imaging 4 (1) (2018) 60–72.  
[15] D.P. Bavirisetti, G. Xiao, J. Zhao, R. Dhuli, G. Liu, Multi-scale guided image and video fusion: A fast and efficient approach, Circuits Systems Signal Process. 38 (12) (2019) 5576–5605.  
[16] Y. Yang, W. Cao, S. Wu, Z. Li, Multi-scale fusion of two large-exposure-ratio images, IEEE Signal Process. Lett. 25 (12) (2018) 1885–1889.  
[17] Perceptual multi-exposure image fusion, IEEE Trans. Multimedia (submitted)   
[18] H. Zhang, H. Xu, Y. Xiao, X. Guo, J. Ma, Rethinking the image fusion: a fast unified image fusion network based on proportional maintenance of gradient and intensity, in: Proceedings of the AAAI Conference on Artificial Intelligence, 2020, pp. 12797–12804.  
[19] K. Ma, Z. Wang, Multi-exposure image fusion: A patch-wise approach, in: 2015 IEEE International Conference on Image Processing, IEEE, 2015, pp. 1717–1721.    
[20] K. Ma, H. Li, H. Yong, Z. Wang, D. Meng, L. Zhang, Robust multi-exposure image fusion: A structural patch decomposition approach, IEEE Trans. Image Process. 26 (5) (2017) 2519–2532.  
[21] H. Xu, J. Ma, J. Jiang, X. Guo, H. Ling, U2fusion: a unified unsupervised image fusion network, IEEE Trans. Pattern Anal. Mach. Intell. (2020).  
[22] D.M. Bulanon, T. Burks, V. Alchanatis, Image fusion of visible and thermal images for fruit detection, Biosyst. Eng. 103 (1) (2009) 12–22.  
[23] V. Aardt, Jan, Assessment of image fusion procedures using entropy, image quality, and multispectral classification, J. Appl. Remote Sens. 2 (1) (2008) 023522.  
[24] M.B.A. Haghighat, A. Aghagolzadeh, H. Seyedarabi, A non-reference image fusion metric based on mutual information of image features, Comput. Electr. Eng. 37 (5) (2011) 744–756.  
[25] G. Qu, D. Zhang, P. Yan, Information measure for performance of image fusion, Electron. Lett. 38 (7) (2002) 313–315.  
[26] M. Hossny, S. Nahavandi, D. Creighton, Comments on’information measure for performance of image fusion’, Electron. Lett. 44 (18) (2008) 1066–1067.  
[27] P. Jagalingam, A.V. Hegde, A review of quality metrics for fused image, Aquat. Procedia 4 (2015) 133–142.  
[28] Q. Wang, Y. Shen, J.Q. Zhang, A nonlinear correlation measure for multivariable data set, Physica D 200 (3–4) (2005) 287–295.  
[29] Q. Wang, Y. Shen, J. Jin, Performance evaluation of image fusion techniques, Image Fusion Algorithms Appl. 19 (2008) 469–492.  
[30] N. Cvejic, C. Canagarajah, D. Bull, Image fusion metric based on mutual information and tsallis entropy, Electron. Lett. 42 (11) (2006) 626–627.  
[31] G. Cui, H. Feng, Z. Xu, Q. Li, Y. Chen, Detail preserved fusion of visible and infrared images using regional saliency extraction and multi-scale image decomposition, Opt. Commun. 341 (2015) 199–209.  
[32] B. Rajalingam, R. Priya, Hybrid multimodality medical image fusion technique for feature enhancement in medical diagnosis, Int. J. Eng. Sci. Invent. 2 (Special issue) (2018) 52–60.  
[33] C.S. Xydeas, P.V. V, Objective image fusion performance measure, Mil. Tech. Cour. 36 (4) (2000) 308–309.  
[34] J. Zhao, R. Laganiere, Z. Liu, Performance assessment of combinative pixellevel image fusion based on an absolute feature measurement, Int. J. Innovative Comput. Inf. Control 3 (6) (2007) 1433–1447.  
[35] Y.-J. Rao, In-fibre bragg grating sensors, Meas. Sci. Technol. 8 (4) (1997) 355.  
[36] A.M. Eskicioglu, P.S. Fisher, Image quality measures and their performance, IEEE Trans. Commun. 43 (12) (1995) 2959–2965.  
[37] N. Cvejic, A. Loza, D. Bull, N. Canagarajah, A similarity metric for assessment of image fusion algorithms, Int. J. Signal Process. 2 (3) (2005) 178–182.  
[38] G. Piella, H. Heijmans, A new quality metric for image fusion, in: Proceedings of International Conference on Image Processing, Vol. 3, IEEE, 2003, pp. III–173 – III–176.  
[39] C. Yang, J.-Q. Zhang, X.-R. Wang, X. Liu, A novel similarity based quality metric for image fusion, Inf. Fusion 9 (2) (2008) 156–160.  
[40] K. Ma, K. Zeng, Z. Wang, Perceptual quality assessment for multi-exposure image fusion, IEEE Trans. Image Process. 24 (11) (2015) 3345–3356.  
[41] Y. Chen, R.S. Blum, A new automated quality assessment algorithm for image fusion, Image Vis. Comput. 27 (10) (2009) 1421–1432.  
[42] H. Chen, P.K. Varshney, A human perception inspired quality metric for image fusion based on regional information, Inf. Fusion 8 (2) (2007) 193–207.  
[43] Y. Han, Y. Cai, Y. Cao, X. Xu, A new image fusion performance metric based on visual information fidelity, Inf. Fusion 14 (2) (2013) 127–135.  
[44] Y. Wu,, J. Lim, & M. H. Yang, Online object tracking: A benchmark. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 2411-2418), 2013.  
[45] X. Zhang, P. Ye, and G. Xiao. "VIFB: a visible and infrared image fusion benchmark." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops. 2020.  
[46] Liu, Z., Blasch, E., Xue, Z., Zhao, J., Lagani¨¦re, R., and Wu, W., ``Objective assessment of multiresolution image fusion algorithms for context enhancement in Night vision: A comparative study", IEEE Transactions on Pattern Analysis and Machine Intelligence, Volume 34, Issue 1, 2012, Article number 5770270, Pages 94-109. 