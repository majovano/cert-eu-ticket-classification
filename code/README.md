## Structure of the code

Cert-EU/
│
├── data/
│   ├── test_dataset.jsonl           
│   └── train_dataset.jsonl         
│
├── llm_test_generation/
│   └── queue_compare.py            # Script: Compare model vs LLM predictions
│               
├── m_requirements.txt              
├── data_processor.py
├── main.py                         # Main script for model training and validation
├── model_trainer.py                # Model architecture 
├── predict.py                      # Generate the assigned_queue on test_dataset.jsonl
│
└── README.md                       

### Note
this is the structure without running anything 

After running training and evaluation, additional files will appear in your directory structure:

data/ will contain new prediction files (e.g., test_predictions.jsonl, test_predictions_low_conf.jsonl).

models/ will contain saved model weights, processor files, and logs (e.g., hybrid_roberta_model/, data_processor.pkl, etc.).

Output: 

In models, you can find the results with additional information with a cv_analysis_report.txt 

(Optional)

By running queue_compare.py

data/ will contain new prediction file (test_llm_predictions.jsonl)

llm_test_generation/ will include output files such as category_metrics.csv, mismatch.csv, mismatch.txt, and other comparison results.

## Requirements
For all dependencies please run the following command:

pip install -r requirements.txt

alternatively:

uv pip install -r requirements.txt (if you want to follow the exact same setup as me)

Note: you can find in section Environment / System Info all the specifications regarding the python version, 


## The main usage 

### Regular Training

python main.py --data_path data/train_dataset.jsonl --epochs 5 --batch_size 16 

It will ask the following:
Suggested learning rate: 3.16e-04
Original learning rate: 2.00e-05
Use suggested LR? (y/n): 
For best results, please press n 


### Cross-validation

python main.py --data_path data/train_dataset.jsonl --cross_validate --cv_folds 5 --epochs 5


## Generate the prediction

python predict.py --test_data data/test_dataset.jsonl --model_dir ./models --output data/test_predictions.jsonl

Output is in : data/test_predictions.jsonl

For really uncertain tickets another json will be created, the goal is to be able to do a qualitative analysis to improve the model on the next iteration.

Output is in: data/test_predictions_low_conf.jsonl

## (Optional) Run LLM predictions

if you want to run this option, please do install: pip install openai 

### Mode 1: Only Compare Model and LLM Outputs

python queue_compare.py --llm_preds ../data/test_llm_predictions.jsonl --model_preds ../data/test_predictions.jsonl


python queue_compare.py --run_llm \
  --infile data/test_dataset.jsonl \
  --llm_preds test_llm_predictions.jsonl \
  --azure_endpoint <your_azure_endpoint> \
  --azure_key <your_azure_key>

### Mode 2: Generate LLM Predictions and Then Compare with Model Predictions

python queue_compare.py --run_llm --infile ../data/test_dataset.jsonl --llm_preds ../data/test_llm_predictions.jsonl --model_preds ../data/test_predictions.jsonl --azure_endpoint <YOUR_ENDPOINT> --azure_key <YOUR_KEY>


## Note
The training step creates a model and processor that are required for the prediction step. Make sure you train the model first, then run predictions.

If you would like you can also call the optional run to compare the model's prediction against LLM prediction (Azure OpenAI GPT in this case) 

It generates csv and txt that you can investigate and do a qualitative assessment too. 
For example in mismatch.txt:

[TKT-BCD0AD8C], [SMS], [Trash] # [ticket-number], [model-prediction] [llm-prediction]

[Ticket-Title],
[Ticket-Content]


## Environment / System Info


This solution was developed and tested on:

    CPU:  Core i9-14900K (24x 3.2/6.0GHz)

    RAM: 96 GB

    GPU: Nvidia RTX-4090 (24GB VRAM)

    Python: 3.11

    PyTorch: 2.7, Transformers: 4.54