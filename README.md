# Noise Labeling Detection

## Teacher - Student Model
Using the teacher-student model, we can train a student model using the teacher model as a guide. The teacher model is trained on an auxiliary dataset (WHU Building), while the student model is trained on the noisy dataset (Kaggle challenge).

The teacher model provides a teaching "path" for the student model to follow. The student model is trained to minimize the combined MSE loss between the last teacher and student feature maps of encoder layer and the cross-entropy loss between the student prediction and the nosiy labels.

![Teacher-Student Model](https://pytorch.org/tutorials/_static/img/knowledge_distillation/fitnets_knowledge_distill.png)

# References
- [Documento Latex](https://www.overleaf.com/4294187349mvhdfqzgtrgn#bee3cb)
- [Kaggle challenge](https://www.kaggle.com/competitions/data-centric-land-cover-classification-challenge/overview)