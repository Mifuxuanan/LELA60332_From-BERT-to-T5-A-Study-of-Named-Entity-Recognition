# NER Downstream Task

**Coursework for Computational Linguistics 2**: Named Entity Recognition Experiments

---

## Project Overview

This repository contains code and resources for Named Entity Recognition (NER) experiments, which fine-tune the BERT and T5 models on both full (7-tag) and simplified (3-tag) BIO tagsets (Data preprocessing in every main experiment) by span-level F1 score as early stopping checkpoints. I evaluate their performance on in-domain and out-of-domain test sets, reporting a bunch of metrics and detailed error analysis (extract the error sentences from every dataset in BERT_NER.ipynb).

---

## Repository Structure

```
root/
│
├── Data preprocessing/                # Dataset
│   ├── PrepareNERData.ipynb           # └─ Notebook: Preprocessing full-tag dataset
│   └── ner_data_dict.json             # └─ Preprocessed full-tag dataset (tokens + 7-class labels)
│
├── Data visualization/                # Data distribution visualizations
│   ├── fulltag_dist.png               # └─ Full‐tag (7 classes) distribution heatmap
│   └── tag_dist.png                   # └─ Simplified (3 classes) distribution heatmap
│
├── Main experiments/                  # Fine-tuning pretrained models (Jupyter notebooks)
│   ├── BERT_NER.ipynb                 # └─ Encoder-only (BERT) on NER
│   └── T5_NER.ipynb                   # └─ Seq2Seq (T5) on NER
│
└── README.md                          # Project overview

```
## Running the code

1. Directly open the two scripts of the main experiments
2. Run the entire notebook
3. Load the related checkpoints for evaluation stages, which are available via Hugging Face:(The document size is too large to upload in the github)
  * **Download link**: https://huggingface.co/Mifuxuanan/CL2_Checkpoints/resolve/main/Checkpoints.rar
4. If you don't want to load the checkpoints, you can directly adjust the parameter positions of "encoder" and "clf_head"(BERT) or "model"(T5) when calling the validation and evaluation (Test & OOD) function.
 * e.g. For BERT in 'BERT_NER.ipynb'
```
# Evaluation with Checkpoint
BERT_Full_enc, BERT_Full_clf = load_checkpoint("BERT_Full_Weightedfn0.8_epoch18_lr5.pt", device, encoder, clf_head)
## Labelled span-level F1 and P/R are running in the function 'Validation_epoch' with the first parameter as 'Test' or 'OOD' in the section of '3-3. Test & OOD'
Test_Full_Loss, Test_Full_Accuracy, Test_Full_Precision, Test_Full_Recall, Test_Full_F1 = validate_epoch(Test, tokeniser, BERT_Full_enc, BERT_Full_clf,
                                                                                                        device, tag_to_i, i_to_tag, batch_size=32, labelled=True)
## Other metrics are running in the function 'evaluation'
Test_result = evaluation(Test, process_batch_sent,
                         tokeniser, BERT_Full_enc, BERT_Full_clf, device,
                         tag_to_i, i_to_tag, batch_size=32)
```
```
# Evaluation with the current model
Test_Full_Loss, Test_Full_Accuracy, Test_Full_Precision, Test_Full_Recall, Test_Full_F1 = validate_epoch(Test, tokeniser, encoder, clf_head,
                                                                                                        device, tag_to_i, i_to_tag, batch_size=32, labelled=True)
Test_result = evaluation(Test, process_batch_sent,
                         tokeniser, encoder, clf_head, device,
                         tag_to_i, i_to_tag, batch_size=32)
```
* e.g. For T5 in 'T5_NER.ipynb'
```
# Evaluation with Checkpoint
T5_Full_mod = load_checkpoint_T5("T5_Full_epoch07_4.pt", device, model)
batch_size=32
## Labelled span-level F1 and P/R are running in the function 'Validation_epoch_T5' with the first parameter as 'Test' or 'OOD' in the section of '3-2. Test & OOD'
Test_Full_Precision, Test_Full_Recall, Test_Full_F1 = validate_epoch_T5(Test, batch_size,
                                               tokeniser, T5_Full_mod, device, max_len=256)

## Other metrics are running in the function 'evaluation_T5'
Test_result_T5 = evaluation_T5(Test, tokeniser, T5_Full_mod, 
                               device, tag_to_i, batch_size=32)
```
```
# Evaluation with the current model
batch_size=32
Test_Full_Precision, Test_Full_Recall, Test_Full_F1 = validate_epoch_T5(Test, batch_size,
                                               tokeniser, model, device, max_len=256)

Test_result_T5 = evaluation_T5(Test, tokeniser, model, 
                               device, tag_to_i, batch_size=32)
```
---

## Data Preparation

1. Load the json document of the full-tag dataset, which has already been done in the scripts of the main experiments
2. Run the First part "Setup" in the main experiemnts

   ```bash
   -- ner_data_dict.json\
   -- BERT_NER.ipynb\
   -- T5_NER.ipynb\
   ```
---

## Training and Validation

### BERT (Encoder-Only)

```bash
python BERT_NER.ipynb \
  --model_type bert \
  --weighted cross-entropy for 0.8 \
  --batch_size 32 \
  --learning_rate 1e-5 \
  --num_epochs 20 \
  --early stopping 5 \
  --seed 42\
  --different hyperparameters can be explored\
```

### T5 (Seq2Seq)

```bash
python T5_NER.ipynb\
  --model_type t5 \
  --few-shot with two examples \
  --batch_size 32 \
  --learning_rate 1e-4 \
  --num_epochs 20 \
  --early stopping 5 \
  --seed 42\
  --different hyperparameters can be explored\
```

---

## Evaluation and Error Analysis

Run evaluation on both in-domain and OOD splits with the checkpoints:

* Computes **span-level** labelled/unlabelled Precision, Recall, F1, macro F1
* Computes **token-level** accuracy
* Computes **span-level** labelled Precision, Recall, F1 for the tag types: "LOC", "PER", "ORG"
* The related operations for evaluating with and without checkpoints have shown above
  
* Generates dataframes for error samples across domains (Only for BERT)
-- e.g. '3-4. Error Analysis in Test/OOD'
```
# Error Analysis with Checkpoint
clf_head = ClassificationHead(n_classes=7).to(device)
## Extract the spans and tags for both gold and predicted ones, which are running in the function 'prediction_for_EA' with the first parameter as 'Test' or 'OOD'
## Followed by the code about Error Dataframes in the original document
BERT_Full_enc, BERT_Full_clf = load_checkpoint("BERT_Full_Weightedfn0.8_epoch18_lr5.pt", device, encoder, clf_head)
Full_word_ls2, Full_all_golds2, Full_all_preds2, Full_gold_spans2, Full_pred_spans2 = prediction_for_EA(Test, 
                                      tokeniser, BERT_Full_enc, BERT_Full_clf,
                                      device, tag_to_i, i_to_tag, batch_size=32, labelled=True)
```
```
# Error Analysis with the current model
clf_head = ClassificationHead(n_classes=7).to(device)
Full_word_ls2, Full_all_golds2, Full_all_preds2, Full_gold_spans2, Full_pred_spans2 = prediction_for_EA(Test, 
                                      tokeniser, encoder, clf_head,
                                      device, tag_to_i, i_to_tag, batch_size=32, labelled=True)
```
---

## Citation & References

* Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. In NAACL-HLT.
* Raffel, C., Shazeer, N., Roberts, A., Lee, K., Narang, S., Matena, M., ... & Liu, P. J. (2020). Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer. Journal of Machine Learning Research, 21(140), 1–67.
* Jurafsky, D., & Martin, J. H. (2021). Speech and Language Processing (3rd ed.) Chapter 17: Named Entity Recognition.
* Li, J., Sun, A., Han, J., & Li, C. (2020). A Survey on Deep Learning for Named Entity Recognition. IEEE Transactions on Knowledge and Data Engineering, 34(1), 50–70.
* ACL LaTeX template: [https://github.com/acl-org/acl-style-files](https://github.com/acl-org/acl-style-files)
