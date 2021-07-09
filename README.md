## A pipeline of pre-training BERT on Google TPU

### Pre-request
1. Pre-training Data
2. Github repo
3. Google Cloud Account

### Step 1: Create a project on Google Cloud
After creating the project, there will a unique id for it (Project ID, which will be used in the next step). We also need to create bucket (for data storage). Then, we need to upload the pre-trained BERT model and pre-training data to this bucket. In the example here, the project name is "subbert" and the bucket name is "subbert_file".

### Step 2: Create VM for TPU
Click the button on the right top to activate the Google Cloud shell:
```
ctpu up --project= project_id --zone=us-central1-b --tf-version=1.15 --name=subbert --tpu-size=v3-8 --preemptible
```
During the creation of the VM, you will be asked to set a password about ssh, just set it and remember it.

After the creation, just ssh into the VM and you can run the following command to check the status of your VM and TPU:
```
ctpu status
```

### Step 3: Git Clone the code to the VM for TPU
```
git clone https://github.com/any_repo
```

### Step 4: Pre-training BERT model using TPU
```
STORAGE_BUCKET=gs://subbert_file

BERT_BASE_DIR=$STORAGE_BUCKET/bert

output_dir=$STORAGE_BUCKET/bert_new
pretraining_file=$STORAGE_BUCKET/data/pre_training_data

python3 run_pretraining.py  --input_file=${pretraining_file}.tfrecord  --output_dir=$output_dir  --do_train=True --do_eval=True --bert_config_file=$BERT_BASE_DIR/bert_config.json --init_checkpoint=${output_dir}/model.ckpt --train_batch_size=192  --max_seq_length=128 --max_predictions_per_seq=20 --num_train_steps=100000 --num_warmup_steps=10  --learning_rate=2e-5 --use_tpu=True --tpu_name=subbert
```
### Step 5: Cleanup
Make sure you delete all resources: both the VM and the TPU instance. Otherwise, you will be charged for the resource until someone recall it.
