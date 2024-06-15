# NMT_Gender_Bias


This project implements a neural machine translation (NMT) system using the OpenNMT-tf library. The system is designed to translate from Hindi to English, leveraging debiased GloVe embeddings to mitigate gender bias in translations. The evaluation is conducted using the WinoBias and WinoMT datasets to measure translation accuracy and gender bias metrics.

Features
Transformer Model: Utilizes a 6-layer Transformer architecture with 512-dimensional hidden representations and 8 attention heads in both the encoder and decoder.
Debiased Embeddings: Incorporates debiased GloVe embeddings to reduce gender bias in the translation output.
Evaluation on Gender Bias: Evaluates the model using the WinoBias and WinoMT datasets, focusing on accuracy, Delta G (gender bias), and Delta S (stereotype bias).
Setup
Install Dependencies:

sh

pip install OpenNMT-tf tensorflow numpy
Download Datasets:

IIT Bombay Bias Balanced Corpus for training.
WinoBias and WinoMT datasets for evaluation.
Pre-trained debiased GloVe embeddings.
Prepare Directories:

Place training data in the appropriate directory (e.g., path/to/iiit_bias_balanced_corpus).
Place evaluation data in separate directories (e.g., path/to/winobias and path/to/winomt).
Ensure the debiased GloVe embeddings file is available at path/to/debiased_glove.txt.
Training
Configure the Model:

Define model parameters, including embedding dimensions, number of layers, units, and attention heads.
Load Data:

Use the provided functions to load training and evaluation datasets.
Train the Model:


runner.train(
    num_steps=100000,
    report_every=100,
    save_checkpoint_steps=1000
)
Evaluation
Load Evaluation Datasets:

Use the load_eval_dataset function to load the WinoBias and WinoMT datasets.
Evaluate the Model:

Use the evaluate_model function to run the model on the evaluation datasets and calculate metrics.



# Training the model
runner.train(num_steps=100000, report_every=100, save_checkpoint_steps=1000)

# Evaluating the model
wino_bias_results = evaluate_model("path/to/model_dir", wino_bias_data)
wino_mt_results = evaluate_model("path/to/model_dir", wino_mt_data)

# Print evaluation results
print("WinoBias Evaluation Results:")
print("Accuracy:", wino_bias_results["accuracy"])
print("Delta G:", wino_bias_results["delta_g"])
print("Delta S:", wino_bias_results["delta_s"])

print("WinoMT Evaluation Results:")
print("Accuracy:", wino_mt_results["accuracy"])
print("Delta G:", wino_mt_results["delta_g"])
print("Delta S:", wino_mt_results["delta_s"])
This project demonstrates how to implement an NMT system with OpenNMT-tf, focusing on reducing gender bias in translations and evaluating the effectiveness of the debiased embeddings using specialized datasets.
