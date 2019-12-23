# Vietnamese Punctuation Prediction Using Deep Neural Networks
(link paper)

In this paper, we have studied the punctuation prediction problem for the Vietnamese language. We collect two large-scale datasets and conduct extensive experiments with both traditional method (using CRF models) and a deep learning approach. We address the class imbalance problem in this task and show promising results using the focal loss on the Vietnamese Newspapers data.

# Requirements
This code has been tested with  
python 3.6.8
tensorflow 1.13.1
fasttext (https://fasttext.cc/docs/en/crawl-vectors.html)

# Dataset

In this work, we collect over 40,000 articles from the Vietnamese news (https://baomoi.com/) and 86 novels (https://gacsach.com/tac-gia/nguyen-nhat-anh.html) to build two datasets with a total of over 900000 sentences.

### Data Preprocessing
We label each word by its immediately following punctuation, where label O denotes a space. Example:
```
Biển tạo ra 1/2 lượng oxy con người hít thở, giúp lưu chuyển nhiệt quanh Trái Đất và hấp thụ một lượng lớn CO2.
(The ocean produces a half of the amount of oxygen that humans can breathe, and help to circulate heat around the Earth and absorb large amounts of CO2.)
```
This paragraph can be labeled as follows:
```
biển/O tạo/O ra/O 1/2/O lượng/O oxy/O con/O người/O hít/O thở/Comma giúp/O lưu/O chuyển/O nhiệt/O quanh/O trái đất/O và/O hấp/O thụ/O một/O lượng/O lớn/O co2/Period
```


### For reproducing results on Cifar10 with 4000 labeled data with CNN13 architechture
```
python main.py  --dataset cifar10  --num_labeled 400 --num_valid_samples 500 --root_dir experiments/ --data_dir data/cifar10/ --batch_size 100  --arch cnn13 --dropout 0.0 --mixup_consistency 100.0 --pseudo_label mean_teacher  --consistency_rampup_starts 0 --consistency_rampup_ends 100 --epochs 400  --lr_rampdown_epochs 450 --print_freq 200 --momentum 0.9 --lr 0.1 --ema_decay 0.999  --mixup_sup_alpha 1.0 --mixup_usup_alpha 1.0
```

### For reproducing results on SVHN with 1000 labeled data with CNN13 architechture
```
python main.py  --dataset svhn  --num_labeled 100 --num_valid_samples 100 --root_dir experiments/ --data_dir data/svhn/ --batch_size 100  --arch cnn13 --dropout 0.0 --mixup_consistency 100.0 --pseudo_label mean_teacher  --consistency_rampup_starts 0 --consistency_rampup_ends 100 --epochs 400  --lr_rampdown_epochs 450 --print_freq 200 --momentum 0.9 --lr 0.1 --ema_decay 0.999  --mixup_sup_alpha 0.1 --mixup_usup_alpha 0.1
```

Running above commands will create a experiment directory with an appropriate name in the directory experiments. For example:experiments/SSL_cifar10_labels_400_valids_500_archcnn13_do0.0_optsgd_lr_0.1_init_lr_0.0_ramp_up_0_ramp_dn_450_ema_d_0.999_m_consis_10.0_type_mse_ramp_0_100_l2_0.0001_eph_400_bs_100_m_sup_a1.0_m_usup_a1.0_pl_mean_teacher_job_id_ 

### Argument description
All the results of the paper can be reproduced by using the appropriate args in the above commands. Following are the args that should be varied to reproduce different experiments of the paper:

--dataset : cifar10 or svhn

--data_dir : data/cifar10 or data/svhn

--num_labeled : number of labeled samples *per class* ( 100/200/400 for cifar10, 25/50/100 for svhn)

--num_valid_samples : number of validation samples *per class* ( 500 for cifar10,  100 for svhn)

--arch : cnn13 or WRN28_2

--mixup_consistency : Max value of consistency coefficient ( check the best values for different experiments in the paper: Section "Experiments")

--consistency_rampup_ends : number of epochs at which the consistency coefficient reaches it maximum value. In all our experiments, this was set to one-fourth of total number of epochs.

--epochs : number of epochs 

--lr_rampdown_epochs: we use cosine ramp-down of learning rate. --lr_rampdown_epochs is the number of epochs at which the learning rate reaches a value of zero. This should be set approximantely 10% more than --epochs

--mixup_sup_alpha, --mixup_usup_alpha : alpha hyperparameter for mixing in supervised loss and consistency loss, respectively. We used same values for both to do minimal hyperparameter seach. Please see the best values for different experiments in the paper.





Disclaimer: Instead of providing a highly optimized code, the purpose of this repo is to provide a user with an easy to use code for reproducing the results in our paper and to use this work in their own research. This repo uses the code from Mean-Teacher repo : (https://github.com/CuriousAI/mean-teacher)
