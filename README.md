# Bean-Leaf-Disease-Classification-Using-Deep-Learning
A deep learning project that classifies bean leaf images into three disease categories: **Angular Leaf Spot (ALS)**, **Bean Rust**, and **Healthy**, using the **EfficientNetB3** architecture.

##  Project Overview

*  **Task**: Multi-class image classification (3 classes)
*  **Input**: Bean leaf images captured under real-world field conditions

  **Model** Fine-tuned `EfficientNetB3` (pretrained on ImageNet)
*  **Best Validation Accuracy**: ~70.5%

---

##  Dataset

* Source: Makerere AI Lab  [Harvard Dataverse DOI](https://doi.org/10.7910/DVN/TCKVEW)
* [T](https://doi.org/10.7910/DVN/TCKVEW)otal Samples: 15,000+
* Classes:

  * `ALS`
  * `Bean Rust`
  * `Healthy`
  * 
* Images were resized to 224×224 and normalized to [0–1] pixel scale.
* Augmentation: Rotation, horizontal flip, width/height shift

---

##  Model Architecture

The model is based on `EfficientNetB3`, with the following custom classification head:

EfficientNetB3 (pretrained)
└── GlobalAveragePooling2D
└── Dropout (0.5)
└── Dense (128, ReLU)
└── Dropout (0.3)
└── Dense (3, Softmax)

* **Loss Function**: Categorical Crossentropy
* **Optimizer**: Adam (learning rate: 1e-5)
* **Epochs Trained**: 10
* **Batch Size**: 32

---


###  Launch locally

```bash
pip install gradio tensorflow pillow
python app.py
```

###  Or launch in Google Colab

```python
!pip install gradio
interface.launch(share=True)
```

---

##  Results

| Metric              | Value |
| ------------------- | ----- |
| Training Accuracy   | 93.5% |
| Validation Accuracy | 70.5% |
| Validation Loss     | ~1.16 |
| Optimal Epoch       | 6     |

* Performance visualizations included: training/validation accuracy and loss plots.
* Early overfitting was mitigated through dropout and fine-tuning.

---

##  Findings

* EfficientNetB3 yielded high accuracy while maintaining computational efficiency.
* Fine-tuning the pretrained model improved validation accuracy significantly.
* The model was deployed successfully as a user-facing web app with real-time image input.

---

##  Challenges

* Overfitting began after epoch 6 despite increasing training accuracy.
* Initial training with a frozen base model produced near-random results (~33%).
* Training time per epoch was long due to model depth and dataset size.

---

##  Proposed Solutions

* Introduced dropout layers and used a reduced learning rate to stabilize fine-tuning.
* Used data augmentation to improve generalization.
* Future improvement: Add EarlyStopping, try class rebalancing, or augment with synthetic data.

  \

---

##  Folder Structure

```
bean_disease_model.h5       -> Saved model
app.py                      -> Gradio app file
training_notebook.ipynb     -> Google Colab notebook for training
dataset/                    -> Folder with ALS, Bean_Rust, Healthy
README.md                   -> Project documentation
```

---

