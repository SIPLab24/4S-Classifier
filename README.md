# 4SC
Empowering Conservation through Semi-Supervised Learning for Endangered Species
To address the issues of sparse samples and high anno-tation costs for rare species.
![Examples of sample counts for endangered versus common species](https://github.com/user-attachments/assets/161151f9-180a-4e58-b0d8-852eabf0d9bc)

These models are less accurate when it comes to identifying rare species, such as the Black Grouse in England, the Spix's Macaw, and the Yangtze River dolphin. For these endangered or declared-extinct animals, labeled data is scarce, and the high cost of annotation makes it difficult for models to effectively capture their unique features during training.

![Imbalanced species labels](https://github.com/user-attachments/assets/358c9f58-679b-4c9e-b9cc-46b575c98e94)The distribution of labeled and unlabeled data across different species categories. This figure illustrates the imbalanced representation of species labels in our dataset, highlighting the challenges faced in training models effectively due to the scarcity of labeled samples for certain endangered species.

# RSEmbed
![RSEmbed](https://github.com/user-attachments/assets/e2bd702f-41e4-4b63-a085-bf3dc71779ff)
Attention-based Rare Species Embedding.The At-tention Map from the pre-trained SE-Net,and Gates focuson specific dimensions, and their weights are used to fusethe embedding vectors of Rare Species samples with thoseof other samples.The above pathway represents the genera-tion of the feature map, while the below pathway represents the generation of the gate weight matrix.

The code can be found in RSM.
# RSBank
RSBank is a module we add into the baseline.
![4SC](https://github.com/user-attachments/assets/9fad1e50-a920-40d8-a786-b41fcf47329f)
Overview of proposed 4S-Classifier.The architecture of 4S-Classifier is an improvement over FullMatch, retaining only the ANL (Adaptive Negative Learning) and incorporating two methods we proposed: RSEmbed and RSBank. Here, we only present RSBank, and the process of RSEmbed will be detailed in subsequent sections with images.

The code can be found in RSM.
