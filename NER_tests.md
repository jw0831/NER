
# Klue Electra
![img1](<./image_file/run_ner_add_tensorboard.png>)

## 119 data new (koelectra) (step = 2000)
![img2](<./image_file/2000step_ner_data_new.png>)

```
  f1 = 0.7458688778858199
  loss = 0.37479466593099964
  precision = 0.7049106387776963
  recall = 0.791880423079359

              precision    recall  f1-score   support

         DAN       0.00      0.00      0.00         2
         DIR       0.70      0.74      0.72      3446
         LOC       0.80      0.82      0.81      1297
         MET       0.53      0.60      0.56        15
         NUM       0.66      0.69      0.68       837
         ORG       0.75      0.86      0.80      2694
         PER       0.78      0.90      0.84      3697
         QTY       0.53      0.54      0.53       250
         SIT       0.39      0.49      0.44      1516
         TIM       0.75      0.87      0.80      2035

   micro avg       0.70      0.79      0.75     15789
   macro avg       0.59      0.65      0.62     15789
weighted avg       0.71      0.79      0.75     15789

```

## 119 data_new fixed + augmented (koelectra) (step = 2000)
![img3](<./image_file/2000step_ner_data_new_fixed_augmented.png>)
```
  f1 = 0.7544657452816168
  loss = 0.35140063241124153
  precision = 0.7115945279210585
  recall = 0.8028338288316782

              precision    recall  f1-score   support

         DAN       0.00      0.00      0.00         9
         DIR       0.71      0.77      0.74      3579
         LOC       0.82      0.86      0.84      1307
         MET       0.73      0.92      0.81        48
         NUM       0.67      0.68      0.67       859
         ORG       0.78      0.84      0.81      2565
         PER       0.78      0.90      0.84      3600
         QTY       0.51      0.65      0.57       254
         SIT       0.38      0.50      0.43      1470
         TIM       0.76      0.88      0.82      2118

   micro avg       0.71      0.80      0.75     15809
   macro avg       0.61      0.70      0.65     15809
weighted avg       0.72      0.80      0.76     15809

```

## 119 data_new fixed + augmented (kcelectra) (step = 2000)
![img4](<./image_file/kcelectra2000.png>)
```
  f1 = 0.7518783854621702
  loss = 0.37201588021384346
  precision = 0.7110046265697291
  recall = 0.7977382276603634

              precision    recall  f1-score   support

         DAN       1.00      0.83      0.91         6
         DIR       0.70      0.76      0.73      3650
         LOC       0.79      0.85      0.82      1368
         MET       0.82      0.87      0.84        52
         NUM       0.66      0.66      0.66       884
         ORG       0.77      0.84      0.81      2690
         PER       0.77      0.90      0.83      3709
         QTY       0.56      0.74      0.64       250
         SIT       0.42      0.50      0.46      1532
         TIM       0.74      0.88      0.81      2041

   micro avg       0.71      0.80      0.75     16182
   macro avg       0.72      0.78      0.75     16182
weighted avg       0.71      0.80      0.75     16182


```

# Klue RoBERTa
![img](<./image_file/roberta_seoulsi_ner.png>)

- checkpoint
    model ??????
    - ckpt 1200
    - ckpt 2200 

```
  # ckpt 1200

  f1 = 0.9991445680068435
  loss = 0.002619927439946457
  precision = 0.9988159452703592
  recall = 0.9994734070563455

              precision    recall  f1-score   support

         AGE       0.97      0.99      0.98       231
         LOC       1.00      1.00      1.00      5932
         SEX       1.00      1.00      1.00      1433

   micro avg       1.00      1.00      1.00      7596
   macro avg       0.99      1.00      0.99      7596
weighted avg       1.00      1.00      1.00      7596

```

```
  # ckpt 2200

  f1 = 0.9998683517640864
  loss = 0.0029501954728520133
  precision = 0.9998683517640864
  recall = 0.9998683517640864

              precision    recall  f1-score   support

         AGE       1.00      1.00      1.00       231
         LOC       1.00      1.00      1.00      5932
         SEX       1.00      1.00      1.00      1433

   micro avg       1.00      1.00      1.00      7596
   macro avg       1.00      1.00      1.00      7596
weighted avg       1.00      1.00      1.00      7596

```
## result
- input : `??????????????? ????????? ???????????? ?????? ?????? 30?????? ????????? ??? ??? ?????????`
- output : `<NE tag='LOC'>??????????????? ????????? ?????????</NE>??? ?????? <NE tag='SEX'>??????</NE> <NE tag='AGE'>30??????</NE> ????????? ??? ??? ?????????`


# ?????? ?????? ??????

`./dev_119new/ner_func.py`
```
# 2022-12-20
## added : ?????? : fix_spaced_word -> '???????????????'?????? '??? ?????????'??? ????????? ?????? ????????? : convert_tagging_for_front??? ?????????
# 2022-12-21
## added : AGE, SEX tag
## fix : tokenizer.tokenize('2~40??? ??????') 
### tokenized result = ['2', '~', '40', '##???', '??????'] ##~??? ???????????? ??????????????? ?????????
- '??????????????? ????????? ???????????? ?????? ?????? 30?????? ????????? ??? ??? ?????????' ??? ????????????
AGE : 30
O : ???
AGE : ???
?????? ????????? ?????? ???????????? (AGE, 30??????) ?????? ????????????

```
