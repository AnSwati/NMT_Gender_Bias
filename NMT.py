import opennmt as onmt
import tensorflow as tf
import numpy as np
import os

# Load the debiased GloVe embeddings
debiased_embeddings = {}
embedding_dim = 300
with open('path/to/debiased_glove.txt', 'r', encoding='utf-8') as f:
    for line in f:
        tokens = line.rstrip().split(' ')
        word = tokens[0]
        vec = np.array([float(x) for x in tokens[1:]])
        debiased_embeddings[word] = vec

# Define the model configuration
config = {
    "model_type": "Transformer",
    "encoder": {
        "embeddings": {
            "embedding_dim": embedding_dim,
            "scale_embeddings": True,
            "embeddings_source": {
                "path": "path/to/pretrained/embeddings.txt",
                "type": "preset_pm"
            }
        },
        "num_layers": 6,
        "num_units": 512,
        "num_heads": 8
    },
    "decoder": {
        "embeddings": {
            "embedding_dim": embedding_dim,
            "scale_embeddings": True,
            "embeddings_source": debiased_embeddings
        },
        "num_layers": 6,
        "num_units": 512,
        "num_heads": 8
    }
}

# Define the dataset loader function
def load_dataset(data_dir, source_language, target_language, shuffle=True):
    return onmt.data.Dataset(
        source_files=[os.path.join(data_dir, f"train.{source_language}")],
        target_files=[os.path.join(data_dir, f"train.{target_language}")],
        shuffle=shuffle
    )

# Load the training dataset
train_data = load_dataset(
    data_dir="path/to/iiit_bias_balanced_corpus",
    source_language="hi",
    target_language="en",
    shuffle=True
)

# Create the custom embedding matrix for the target inputter
vocab_size = len(debiased_embeddings)
pretrained_embedding_matrix = np.zeros((vocab_size, embedding_dim))
word_to_index = {word: idx for idx, word in enumerate(debiased_embeddings.keys())}
for word, idx in word_to_index.items():
    pretrained_embedding_matrix[idx] = debiased_embeddings[word]

# Define the model
model = onmt.models.Transformer(
    source_inputter=onmt.inputters.WordEmbedder(
        embedding_size=config["encoder"]["embeddings"]["embedding_dim"],
        vocabulary_file_key="source_words_vocabulary",
        embedding_file_with_header=None
    ),
    target_inputter=onmt.inputters.WordEmbedder(
        embedding_size=config["decoder"]["embeddings"]["embedding_dim"],
        vocabulary_file_key="target_words_vocabulary",
        embedding_file_with_header=None,
        embedding=pretrained_embedding_matrix
    ),
    num_layers=config["encoder"]["num_layers"],
    num_units=config["encoder"]["num_units"],
    num_heads=config["encoder"]["num_heads"],
    ffn_inner_dim=2048,
    dropout=0.1,
    attention_dropout=0.1
)

# Define the optimizer
optimizer = tf.keras.optimizers.Adam(
    learning_rate=2.0,
    beta_1=0.9,
    beta_2=0.998
)

# Define the training configuration
training_config = onmt.TrainingConfig(
    model_dir="path/to/model_dir",
    batch_size=64,
    save_checkpoints_steps=1000,
    keep_checkpoint_max=5
)

# Define the runner
runner = onmt.Runner(
    model=model,
    config=training_config,
    optimizer=optimizer,
    train_dataset=train_data
)

# Train the model
runner.train(
    num_steps=100000,
    report_every=100,
    save_checkpoint_steps=1000
)

# Load the evaluation datasets
def load_eval_dataset(data_dir, source_language, target_language):
    return onmt.data.Dataset(
        source_files=[os.path.join(data_dir, f"{source_language}.txt")],
        target_files=[os.path.join(data_dir, f"{target_language}.txt")],
        shuffle=False
    )

wino_bias_data = load_eval_dataset(
    data_dir="path/to/winobias",
    source_language="hi",
    target_language="en"
)

wino_mt_data = load_eval_dataset(
    data_dir="path/to/winomt",
    source_language="hi",
    target_language="en"
)

# Function to evaluate the model
def evaluate_model(model_dir, eval_data):
    translator = onmt.translate.Translator(model_dir=model_dir)
    predictions = translator.translate(eval_data)

    # Calculate evaluation metrics
    accuracy = calculate_accuracy(predictions)
    delta_g = calculate_delta_g(predictions)
    delta_s = calculate_delta_s(predictions)
    
    return {
        "accuracy": accuracy,
        "delta_g": delta_g,
        "delta_s": delta_s
    }

# Implementing the metric calculations
def calculate_accuracy(predictions):
    correct = 0
    total = len(predictions)
    
    for prediction in predictions:
        source, target, predicted = prediction
        if target == predicted:
            correct += 1
    
    return correct / total if total > 0 else 0

def calculate_delta_g(predictions):
    masculine_correct = 0
    feminine_correct = 0
    masculine_total = 0
    feminine_total = 0
    
    for prediction in predictions:
        source, target, predicted = prediction
        if "he" in target or "him" in target or "his" in target:
            masculine_total += 1
            if target == predicted:
                masculine_correct += 1
        elif "she" in target or "her" in target:
            feminine_total += 1
            if target == predicted:
                feminine_correct += 1
    
    masculine_accuracy = masculine_correct / masculine_total if masculine_total > 0 else 0
    feminine_accuracy = feminine_correct / feminine_total if feminine_total > 0 else 0
    
    return masculine_accuracy - feminine_accuracy

def calculate_delta_s(predictions):
    pro_stereotypical_correct = 0
    anti_stereotypical_correct = 0
    pro_stereotypical_total = 0
    anti_stereotypical_total = 0
    
    for prediction in predictions:
        source, target, predicted = prediction
        if is_pro_stereotypical(source):
            pro_stereotypical_total += 1
            if target == predicted:
                pro_stereotypical_correct += 1
        elif is_anti_stereotypical(source):
            anti_stereotypical_total += 1
            if target == predicted:
                anti_stereotypical_correct += 1
    
    pro_stereotypical_accuracy = pro_stereotypical_correct / pro_stereotypical_total if pro_stereotypical_total > 0 else 0
    anti_stereotypical_accuracy = anti_stereotypical_correct / anti_stereotypical_total if anti_stereotypical_total > 0 else 0
    
    return pro_stereotypical_accuracy - anti_stereotypical_accuracy

def is_pro_stereotypical(sentence):
    # Implement logic to determine if a sentence is pro-stereotypical
    return "doctor" in sentence or "engineer" in sentence

def is_anti_stereotypical(sentence):
    # Implement logic to determine if a sentence is anti-stereotypical
    return "nurse" in sentence or "teacher" in sentence

# Evaluate the model on WinoBias and WinoMT datasets
wino_bias_results = evaluate_model("path/to/model_dir", wino_bias_data)
wino_mt_results = evaluate_model("path/to/model_dir", wino_mt_data)

# Print the evaluation results
print("WinoBias Evaluation Results:")
print("Accuracy:", wino_bias_results["accuracy"])
print("Delta G:", wino_bias_results["delta_g"])
print("Delta S:", wino_bias_results["delta_s"])

print("WinoMT Evaluation Results:")
print("Accuracy:", wino_mt_results["accuracy"])
print("Delta G:", wino_mt_results["delta_g"])
print("Delta S:", wino_mt_results["delta_s"])
