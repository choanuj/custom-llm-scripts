# Tutorial: custom-llm-scripts

This project provides a complete workflow for building your *very own* **custom Language Model (LLM)**.
It guides you through preparing your raw text data, training the LLM to understand and generate human-like text,
and then optimizing this trained model into a more efficient format for easy deployment and text generation.
Ultimately, it allows you to create and use a personalized digital storyteller.


## Visual Overview

```mermaid
flowchart TD
    A0["LLM Model Architecture (GPT-like)
"]
    A1["LLM Training Process
"]
    A2["Text Data Preparation
"]
    A3["Model Optimization (GGUF Conversion)
"]
    A4["Text Generation (Inference)
"]
    A5["User Configuration
"]
    A6["Transformer Building Blocks
"]
    A5 -- "Configures" --> A0
    A5 -- "Configures" --> A1
    A2 -- "Feeds Data to" --> A1
    A1 -- "Trains" --> A0
    A6 -- "Composes" --> A0
    A0 -- "Performs Inference" --> A4
    A3 -- "Prepares for" --> A4
```
# Chapter 1: User Configuration

Welcome to the exciting world of building your own Large Language Model (LLM)! Think of an LLM as a very smart digital assistant that can understand and generate human-like text.

Before you start building anything, whether it's a LEGO castle or a sophisticated computer program, you need a plan. For our LLM, this plan involves making some important decisions about *how* it will be built and *how* it will learn. This is what we call **User Configuration**.

### What is User Configuration?

Imagine you're setting up a brand-new smartphone. You get to choose things like:
*   How much storage it has.
*   How fast its processor is.
*   Which features are turned on or off.

User configuration for our LLM is very similar! It's where you define the basic "brain size" and "learning strategy" for your model. This flexibility is key because it allows you to experiment and fine-tune your LLM to achieve different goals, whether you want a tiny, fast model or a large, powerful one.

Our project `custom-llm-scripts` lets you control two main types of settings:

1.  **Model Configuration**: These settings define the *structure* of your LLM. Think of them as the "blueprint" for your model's digital brain. They determine how big and complex it will be.
2.  **Training Configuration**: These settings define *how* your LLM learns. They are like a "study plan" that tells the model how to practice and improve.

Let's look at how you can set these up!

### Setting Up Your LLM's Blueprint (Model Configuration)

The `llm_model.py` file contains the instructions for building your LLM's core structure. Before you even start training, you'll run this file once to decide on its basic "design."

When you run `llm_model.py`, it will ask you a series of questions. Let's see an example:

```bash
python llm_model.py
```

After running this command, you will see prompts like these:

```
Configure your LLM model:
Enter vocabulary size (e.g., 10000 for small models): 5000
Enter embedding dimension (e.g., 256): 128
Enter number of attention heads (e.g., 8): 4
Enter number of transformer layers (e.g., 6): 3
Enter max sequence length (e.g., 128): 64
Enter dropout rate (e.g., 0.1): 0.05
```

Let's quickly understand what each of these means:

| Setting           | Analogy / What it means                                                              |
| :---------------- | :----------------------------------------------------------------------------------- |
| `vocab_size`      | How many unique "words" (or characters) your LLM can understand and use.             |
| `n_embd`          | The "width" of the model's internal processing. Larger means more detail.            |
| `n_head`          | How many different "lenses" the model uses to look at text at the same time.        |
| `n_layer`         | How many "layers" or processing steps your model has. More layers, deeper thinking.  |
| `block_size`      | The maximum number of "words" or characters the model can look at simultaneously.    |
| `dropout`         | A technique to prevent the model from "over-memorizing" its training data.           |

After you enter these values, the script will save them into a file called `model_config.pth`. This file is like a blueprint document that `train_llm.py` will read later.

### Setting Up Your LLM's Study Plan (Training Configuration)

Once your model's blueprint is ready, you'll move to `train_llm.py` to define *how* it will learn. This file also has a section where it asks for your input:

```bash
python train_llm.py
```

You'll see prompts for training parameters:

```
Configure training parameters:
Enter batch size (e.g., 32): 16
Enter learning rate (e.g., 0.0003): 0.001
Enter number of epochs (e.g., 5): 10
Enter path to text dataset (e.g., 'input.txt'): my_data.txt
```

Here's what these settings mean for your LLM's learning process:

| Setting           | Analogy / What it means                                                              |
| :---------------- | :----------------------------------------------------------------------------------- |
| `batch_size`      | How many "sentences" or text chunks the model learns from at once before updating its knowledge. |
| `learning_rate`   | How big of a "step" the model takes when adjusting its knowledge. Small is slow but precise; large is fast but can miss details. |
| `max_epochs`      | How many times the model will read through its *entire* training material (your text data). |
| `dataset_path`    | The location of the text file your LLM will learn from.                               |

### How It All Connects (Under the Hood)

Let's visualize how your inputs are used by the scripts.

```mermaid
sequenceDiagram
    participant You as User
    participant llm_model_script as llm_model.py
    participant train_llm_script as train_llm.py

    You->>llm_model_script: 1. Run to configure model
    llm_model_script->>You: Ask for model settings (vocab size, layers etc.)
    You->>llm_model_script: Provide model choices
    llm_model_script->>llm_model_script: Saves choices to "model_config.pth" file
    Note over llm_model_script: This file acts as your model's blueprint.

    You->>train_llm_script: 2. Run to train the LLM
    train_llm_script->>train_llm_script: Reads "model_config.pth"
    train_llm_script->>You: Ask for training settings (batch size, epochs etc.)
    You->>train_llm_script: Provide training choices
    Note over train_llm_script: Uses both sets of choices to build & train the LLM.
```

As you can see, you first tell `llm_model.py` about the structure, and it saves this. Then, `train_llm.py` loads that structure and asks you for the training plan.

Let's peek at the code snippets that make this happen.

In `llm_model.py`, the `get_user_config()` function is responsible for asking you about the model's structure:

```python
# From llm_model.py
def get_user_config():
    print("Configure your LLM model:")
    # Prompts for model architecture settings
    vocab_size = int(input("Enter vocabulary size (e.g., 10000 for small models): ") or 10000)
    n_embd = int(input("Enter embedding dimension (e.g., 256): ") or 256)
    # ... more settings collected similarly ...
    return {
        'vocab_size': vocab_size,
        'n_embd': n_embd,
        'n_head': n_head,
        'n_layer': n_layer,
        'block_size': block_size,
        'dropout': dropout
    }
```
This function gathers your choices and packs them into a Python dictionary. A dictionary is like a simple list of key-value pairs (e.g., `vocab_size: 5000`). This dictionary is then saved to a file (`model_config.pth`) for later use.

Similarly, in `train_llm.py`, the `get_training_config()` function collects your preferences for how the model will learn:

```python
# From train_llm.py
def get_training_config():
    print("Configure training parameters:")
    # Prompts for training process settings
    batch_size = int(input("Enter batch size (e.g., 32): ") or 32)
    learning_rate = float(input("Enter learning rate (e.g., 0.0003): ") or 0.0003)
    # ... more settings collected similarly ...
    dataset_path = input("Enter path to text dataset (e.g., 'input.txt'): ") or 'input.txt'
    return {
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'max_epochs': max_epochs,
        'dataset_path': dataset_path
    }
```
This function also creates a dictionary, but this one holds all the details about the training process.

Finally, in the main part of `train_llm.py`, both of these configurations are put to use:

```python
# From train_llm.py (main execution block)
if __name__ == "__main__":
    # Load model configuration saved earlier by llm_model.py
    model_config = torch.load('model_config.pth')
    
    # Get training configuration directly from user
    train_config = get_training_config()
    
    # ... then these configurations are used:
    # 1. To create the LLM model based on model_config
    # 2. To start the training process using train_config
    # ...
```
This snippet shows that `train_llm.py` first loads the model's blueprint from the `model_config.pth` file. Then, it asks you for the training plan. With both sets of instructions, it's ready to bring your LLM to life!

### Conclusion

In this chapter, you've learned that configuring your LLM involves making choices about its structure (model configuration) and its learning process (training configuration). You've seen how to provide these inputs and how the scripts use them to prepare for building and training your custom LLM.

Now that we know *how* to set up our LLM, the next step is to prepare the "study material" for it. This means getting our text data ready.

Let's move on to [Text Data Preparation]

# Chapter 2: Text Data Preparation

Welcome back! In [Chapter 1: User Configuration](01_user_configuration_.md), you learned how to set up the blueprint and study plan for your custom Large Language Model (LLM). You decided on things like its brain size (`vocab_size`, `n_layer`) and how it will learn (`batch_size`, `learning_rate`).

Now, imagine your LLM is a very hungry student, eager to learn. But before it can "read" anything, we need to prepare its "study material" – your raw text data. This is where **Text Data Preparation** comes in!

### Why Prepare Text Data?

Think of it like a chef preparing ingredients for a recipe. You can't just throw a whole raw potato, an entire onion, and a live chicken into a pot and expect a delicious soup! You need to:
1.  **Wash and peel** the ingredients.
2.  **Chop** them into smaller, usable pieces.
3.  **Measure** the right amount of each.

Similarly, an LLM can't directly understand a long, raw text file. It needs the text to be:
*   **Broken down** into tiny, understandable pieces (like chopping ingredients).
*   **Converted into numbers**, because computers only understand numbers (like a chef weighing ingredients).
*   **Organized** into small, manageable servings (like portioning the meal).

The goal of text data preparation is to transform a messy, long string of characters into a structured sequence of numbers that our LLM can easily learn from.

### Our Use Case: Preparing `my_story.txt`

Let's say you have a simple text file named `my_story.txt` that looks like this:

```
Hello world!
This is a small story.
```

Our mission in this chapter is to understand how `custom-llm-scripts` takes this raw text and makes it ready for the LLM's learning process.

### Key Concepts in Text Preparation

Let's break down the "chopping" and "measuring" steps:

#### 1. Tokenization: Breaking Text into "Tokens" and Assigning Numbers

The first step is to break down the text into the smallest meaningful units our LLM will understand. These units are called **tokens**. For our simple LLM, we'll treat **each individual character** (like 'H', 'e', 'l', 'l', 'o', ' ', 'w', etc.) as a token.

Then, just like a chef assigns a code to each ingredient (e.g., potato = 01, onion = 02), we need to give each unique character a unique number. This process is called **encoding**.

Let's see an example with the word "hello":

| Character | Unique ID (Number) |
| :-------- | :----------------- |
| 'h'       | 0                  |
| 'e'       | 1                  |
| 'l'       | 2                  |
| 'o'       | 3                  |

So, the word "hello" would become the sequence of numbers `[0, 1, 2, 2, 3]`. This collection of all unique characters and their corresponding numbers is called the **vocabulary** (`vocab_size` from Chapter 1).

#### 2. Chunking: Cutting Text into Smaller Overlapping Pieces

LLMs can't process an entire book at once. They need to learn from smaller "snippets" or "chunks" of text. In [Chapter 1: User Configuration](01_user_configuration_.md), you set a `block_size` (or `max_sequence_length`). This `block_size` defines how long each snippet of text should be.

Imagine our chef cutting a long ribbon of pasta into smaller, equal-sized pieces.

For example, if our encoded text is `[0, 1, 2, 2, 3, 4, 5, 6, 7, 8]` (representing "hello world") and our `block_size` is 5:

*   **Chunk 1**: `[0, 1, 2, 2, 3]` (representing "hello")
*   **Chunk 2**: `[1, 2, 2, 3, 4]` (representing "ello ") - *Notice the overlap! This helps the LLM learn connections.*
*   **Chunk 3**: `[2, 2, 3, 4, 5]` (representing "llo w")
*   ...and so on.

This overlapping is important because it gives the LLM more opportunities to learn how characters follow each other in different contexts.

#### 3. Input and Target Pairs: Learning to Predict the Next Token

For training, our LLM doesn't just read a chunk; it tries to *predict* the very next character in the sequence. This is like a quiz where the model sees part of a word and has to guess the next letter.

Each chunk is split into two parts:
*   **Input (X)**: The part the LLM sees.
*   **Target (Y)**: The very next character that the LLM needs to predict.

Let's take our "hello" chunk `[0, 1, 2, 2, 3]` with a `block_size` of 5.

| Position | Input (X) | Target (Y) |
| :------- | :-------- | :--------- |
| 1        | `[0]`     | `[1]`      |
| 2        | `[0, 1]`  | `[1, 2]`   |
| 3        | `[0, 1, 2]` | `[1, 2, 2]`|
| 4        | `[0, 1, 2, 2]` | `[1, 2, 2, 3]` |

The LLM will learn that if it sees `[0, 1, 2, 2]` (h e l l), it should predict `3` (o). This is the core idea of how LLMs learn to generate text: by predicting the next token.

### How `custom-llm-scripts` Handles Text Preparation

You don't need to write any code for these preparation steps! The `train_llm.py` script automatically takes care of everything when you provide your text file path.

Recall from Chapter 1 that when you run `train_llm.py`, it asks for a `dataset_path`:

```bash
python train_llm.py
```
```
Configure training parameters:
Enter batch size (e.g., 32): 16
Enter learning rate (e.g., 0.0003): 0.001
Enter number of epochs (e.g., 5): 10
Enter path to text dataset (e.g., 'input.txt'): my_story.txt  <-- You provide your file here!
```

When you enter `my_story.txt` (or any other path), `train_llm.py` will read this file, perform tokenization, chunking, and create the input/target pairs, all behind the scenes. The result is numerical data that the LLM can immediately start learning from.

### Under the Hood: TextDataset

Let's look at the key steps inside `train_llm.py` that make this happen.

#### The Overall Flow

Here's a simple diagram showing the steps taken by the script:

```mermaid
sequenceDiagram
    participant You as User
    participant train_llm_script as train_llm.py
    participant TextDataset as TextDataset class
    participant DataLoader as DataLoader utility

    You->>train_llm_script: 1. Run script and provide 'my_story.txt'
    train_llm_script->>train_llm_script: 2. Reads raw text from 'my_story.txt'
    train_llm_script->>TextDataset: 3. Passes text and 'block_size' to initialize
    TextDataset->>TextDataset: 4. Finds unique characters (vocab)
    TextDataset->>TextDataset: 5. Assigns numbers to each char (stoi, itos)
    TextDataset->>TextDataset: 6. Converts entire text to numbers
    train_llm_script->>DataLoader: 7. Uses TextDataset to create batches of numerical chunks
    DataLoader->>train_llm_script: 8. Provides 'input (X)' and 'target (Y)' for training
    Note over train_llm_script: Now the LLM can start learning!
```

#### Code Details

All the magic for text data preparation happens within a special class called `TextDataset` inside `train_llm.py`. This class is designed specifically to handle our character-level text.

First, the script reads your chosen text file:

```python
# From train_llm.py (main execution block)
if __name__ == "__main__":
    # ...
    train_config = get_training_config() # This gets your dataset_path
    
    # Load and prepare dataset
    with open(train_config['dataset_path'], 'r', encoding='utf-8') as f:
        text = f.read() # Reads your entire text file into 'text' variable
    # ...
```
This simple piece of code opens your `my_story.txt` file and reads all its content into a Python variable called `text`.

Next, the `TextDataset` class gets to work. When it's initialized, it first builds the vocabulary and converts the text into numbers:

```python
# From train_llm.py (inside TextDataset class)
class TextDataset(Dataset):
    def __init__(self, text, block_size):
        self.block_size = block_size
        chars = sorted(list(set(text))) # Find all unique characters and sort them
        self.vocab_size = len(chars)     # Count how many unique characters there are
        
        # Create mapping: character to integer (stoi = string to integer)
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        # Create mapping: integer to character (itos = integer to string)
        self.itos = {i: ch for i, ch in enumerate(chars)}
        
        # Convert the entire text into a sequence of numbers
        self.data = torch.tensor([self.stoi[c] for c in text], dtype=torch.long)
```
In this `__init__` part of the `TextDataset` class:
*   It finds all unique characters (`chars`) in your `text`.
*   It creates `stoi` (string-to-integer) and `itos` (integer-to-string) "dictionaries" (like a phonebook mapping names to numbers and vice-versa).
*   Finally, it uses `stoi` to convert your entire `text` into a long list of numbers, stored in `self.data`.

After the `TextDataset` is initialized, `train_llm.py` updates the `vocab_size` in the `model_config` to match the actual number of unique characters found in your data:

```python
# From train_llm.py (main execution block)
    # ...
    # Initialize dataset and dataloader
    dataset = TextDataset(text, model_config['block_size']) # Create an instance of our dataset
    
    # Update vocab size from dataset, as it's determined by the actual text used
    model_config['vocab_size'] = dataset.vocab_size  
    # ...
```
This is important because your model needs to know exactly how many unique tokens it might encounter.

Finally, the `TextDataset` prepares the chunks and splits them into input (X) and target (Y) pairs. This happens whenever the model asks for a new "item" (a chunk) from the dataset:

```python
# From train_llm.py (inside TextDataset class)
    def __len__(self):
        # Tells how many possible chunks can be made from the data
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        # Get a chunk of data of size block_size + 1
        chunk = self.data[idx:idx + self.block_size + 1]
        
        # The input (xb) is the chunk without the last token
        xb = chunk[:-1] 
        # The target (yb) is the chunk starting from the second token
        yb = chunk[1:]   
        return xb, yb # Return the input and target pair
```
When `train_llm.py` asks `TextDataset` for a new chunk of data (usually through something called a `DataLoader`, which organizes batches for training), the `__getitem__` method is called. It takes a starting position (`idx`), grabs `block_size + 1` tokens, and then cleverly splits them into `xb` (input) and `yb` (target) ready for the LLM to learn from.

### Conclusion

In this chapter, you've taken your raw text data and understood how it's transformed into a numerical, digestible format for your LLM. You've seen the crucial steps of tokenization (breaking text into characters and numbering them) and chunking (creating small, overlapping sequences for learning). You now know that the `train_llm.py` script, especially through its `TextDataset` class, handles all this preparation for you based on the `dataset_path` and `block_size` you configured earlier.

With your "study material" now perfectly prepared, it's time to understand the "student" itself – the LLM model's architecture.

Let's move on to [Chapter 3: LLM Model Architecture (GPT-like)]

# Chapter 3: LLM Model Architecture (GPT-like)

Welcome back to building your own LLM! In [Chapter 2: Text Data Preparation](02_text_data_preparation_.md), we learned how to prepare our raw text data, transforming it into numerical "study material" that our LLM can understand. We took our `my_story.txt` and turned it into numbers, ready for learning.

Now, it's time to meet the "student" itself – the Large Language Model. This chapter will dive into the **LLM Model Architecture (GPT-like)**, which is the actual "brain" or "engine" that will learn from our prepared data and generate new text.

### What is LLM Model Architecture?

Imagine you're designing a very sophisticated machine whose job is to write stories or answer questions. This machine isn't just a simple calculator; it needs to understand context, grammar, and even subtle meanings. The "architecture" is like the detailed blueprint of this machine, showing all its internal parts and how they work together.

For our custom LLM, this architecture is inspired by **GPT-like models** (like OpenAI's GPT-3 or GPT-4, though ours will be much simpler!). These models are particularly good at understanding sequences of words and predicting the *next* word, which is the core of generating human-like text.

The problem this architecture solves is: **How do we build a digital brain that can learn from text, understand patterns, and then use that understanding to create new, coherent sentences?**

### Key Concepts of a GPT-like Brain

Our GPT-like brain is built from many identical "building blocks" stacked on top of each other. Think of it like a multi-story building, where each floor (or "layer") is a powerful processing unit.

1.  **"Understanding" Layers (Transformer Blocks)**: The core of a GPT-like model is made of many "Transformer Blocks." Each block is like a mini-brain that processes the text it sees. It helps the model understand:
    *   What each word means in context.
    *   How words relate to each other in a sentence (e.g., "apple" relates to "eat" more than "sleep").
    *   The "flow" of the sentence.
    *   We'll explore these fascinating "Transformer Building Blocks" in more detail in [Chapter 5: Transformer Building Blocks](05_transformer_building_blocks_.md).

2.  **Predicting the Next Word**: The ultimate goal of our LLM during training is to get very good at predicting the next word (or character in our case). If it sees "Hello worl", it needs to predict "d!". This ability to predict the next character is what allows it to *generate* entire sentences, paragraphs, or even stories, one character at a time.

### How to Build Your LLM's Brain

In [Chapter 1: User Configuration](01_user_configuration_.md), you already started building this brain by making choices in `llm_model.py`. When you ran that script, you were essentially deciding on the "size" and "complexity" of your LLM's architecture.

Recall the settings you provided for the model:

| Setting         | What it means for the Architecture                           |
| :-------------- | :----------------------------------------------------------- |
| `vocab_size`    | How many unique "output slots" the model needs for predictions (one for each character it knows). |
| `n_embd`        | The "width" or "depth" of the model's internal thinking. Larger means more complex concepts. |
| `n_head`        | How many different ways the model looks at text simultaneously within each layer. |
| `n_layer`       | How many "layers" or "floors" your model's building has. More layers, deeper thinking. |
| `block_size`    | The maximum "window" of text the model can consider at once. |
| `dropout`       | A technique to prevent the model from "over-memorizing".    |

When you run `llm_model.py`, it uses these choices to construct the `GPTLanguageModel` class, which is the actual digital representation of your LLM's brain.

```bash
python llm_model.py
```

After asking for inputs, the script creates this model and saves its *configuration* (your choices) into `model_config.pth`. This configuration is then loaded by `train_llm.py` to build the actual model structure before training.

### Under the Hood: Building the GPTLanguageModel

Let's look at how the `GPTLanguageModel` class in `llm_model.py` is put together. Think of it as an assembly line constructing our text-generating machine.

#### The Assembly Line Process

```mermaid
sequenceDiagram
    participant You as User
    participant llm_model_script as llm_model.py
    participant GPTLanguageModel as GPTLanguageModel class

    You->>llm_model_script: 1. Run to configure and create model blueprint
    llm_model_script->>GPTLanguageModel: 2. Initialize GPTLanguageModel with your config
    GPTLanguageModel->>GPTLanguageModel: 3. Create "token embeddings" (understand word meanings)
    GPTLanguageModel->>GPTLanguageModel: 4. Create "position embeddings" (understand word order)
    GPTLanguageModel->>GPTLanguageModel: 5. Stack "Transformer Blocks" (the main processing layers)
    GPTLanguageModel->>GPTLanguageModel: 6. Create final "prediction head"
    llm_model_script->>llm_model_script: 7. Saves model configuration to 'model_config.pth'
    Note over llm_model_script: The model is now ready to be trained!
```

#### Code Details: Inside `GPTLanguageModel`

The `GPTLanguageModel` class in `llm_model.py` is where all these parts come to life. Let's break down its key components.

**1. The Constructor (`__init__`)**

This is where the model's internal parts are created based on your configuration:

```python
# From llm_model.py (inside GPTLanguageModel class)
class GPTLanguageModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 1. Token Embedding: Turns character IDs into rich numerical "meanings"
        self.token_embedding = nn.Embedding(config['vocab_size'], config['n_embd'])
        
        # 2. Position Embedding: Helps the model understand the order of characters
        self.position_embedding = nn.Embedding(config['block_size'], config['n_embd'])
```
*   `token_embedding`: Imagine you have a dictionary. This part takes the character's unique number (e.g., 'h' is 0, 'e' is 1) and turns it into a more complex number-list (a "vector") that captures its "meaning." This is like giving each character a richer internal representation than just a single ID.
*   `position_embedding`: Language isn't just about words; it's also about their order. This part adds information about *where* a character is in the sequence (e.g., is it the first character, second, etc.?). This is crucial for understanding context.

Next, we add the main "thinking" layers:

```python
# From llm_model.py (inside GPTLanguageModel class, continuing __init__)
        # 3. Transformer Blocks: The core "brain" layers for processing text
        self.blocks = nn.ModuleList([
            # Each block performs complex calculations (attention, feed-forward networks)
            # We'll learn more about TransformerBlock in Chapter 5!
            TransformerBlock(config['n_embd'], config['n_head'], config['block_size'], config['dropout'])
            for _ in range(config['n_layer']) # Create as many layers as you configured
        ])
```
*   `self.blocks`: This is a list of our `TransformerBlock` units. Each block is a sophisticated processing unit that takes the combined "meaning" and "position" information and refines it, allowing the model to "think" deeper about the input text. The more `n_layer` you set, the more of these blocks are stacked, making the model capable of deeper reasoning.

Finally, the output part:

```python
# From llm_model.py (inside GPTLanguageModel class, continuing __init__)
        # 4. Final LayerNorm: Helps stabilize the learning process
        self.ln_f = nn.LayerNorm(config['n_embd']) 
        
        # 5. Prediction Head: Turns the processed "thinking" into predictions for the next character
        self.head = nn.Linear(config['n_embd'], config['vocab_size'])
        self.block_size = config['block_size']
        
        # Apply special starting weights (technical detail for better learning)
        self.apply(self._init_weights)
```
*   `self.ln_f`: This is a "Layer Normalization" step, a technical detail that helps the model learn more stably.
*   `self.head`: This is the "final decision maker." After all the processing by the Transformer Blocks, this `head` takes the refined information and uses it to make a prediction for what the *next* character should be. It outputs a list of scores, one for each character in your `vocab_size`, indicating how likely each character is to be the next one.

**2. The Forward Pass (`forward`)**

The `forward` method describes how data flows *through* our LLM brain when it's trying to understand input and make predictions.

```python
# From llm_model.py (inside GPTLanguageModel class)
    def forward(self, idx, targets=None):
        B, T = idx.size() # B=Batch size, T=Sequence length (block_size)
        
        # 1. Get initial "meaning" from token and position embeddings
        tok_emb = self.token_embedding(idx) # shape (B, T, n_embd)
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # position indices
        pos_emb = self.position_embedding(pos) # shape (T, n_embd)
        x = tok_emb + pos_emb # Add them together
        
        # 2. Pass through the "thinking" (Transformer Blocks)
        for block in self.blocks:
            x = block(x) # x goes through each TransformerBlock
        
        # 3. Final normalization and prediction
        x = self.ln_f(x)
        logits = self.head(x) # Raw prediction scores for each character
        
        loss = None
        # 4. If training, calculate how "wrong" our predictions were
        if targets is not None:
            # Reshape for loss calculation (technical detail)
            logits_flat = logits.view(B*T, self.config['vocab_size'])
            targets_flat = targets.view(B*T)
            # Calculate the loss: how far are our predictions from the true next characters?
            loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)
            
        return logits, loss # Return the predictions and (if training) the loss
```
When you give your LLM an input sequence (`idx`), it goes through these steps:
1.  **Embeddings**: Each character (numerical ID) is converted into a rich `n_embd`-sized vector, combined with its positional information.
2.  **Transformer Blocks**: This combined information then flows through each `TransformerBlock`. At each block, the model refines its understanding of the sequence, considering how each character relates to others.
3.  **Prediction Head**: Finally, the processed information goes to the `head`, which outputs a `vocab_size`-sized list of probabilities for the next character. The character with the highest probability is what the model "predicts."
4.  **Loss Calculation (during training)**: If we are training (i.e., `targets` are provided), the model compares its predictions (`logits`) with the actual `targets` (the true next characters) and calculates a `loss`. This `loss` tells the model how "wrong" its predictions were. A smaller loss means better predictions!

### Conclusion

In this chapter, you've gained a fundamental understanding of what constitutes the "brain" of our custom LLM. You've seen that it's a sophisticated machine made of "embedding layers" to understand meaning and position, multiple "Transformer Blocks" for deep thinking, and a "prediction head" for generating the next character. You also saw how `llm_model.py` uses your configuration choices to construct this architecture.

Now that we have prepared our data and understood the architecture of our LLM, the next crucial step is to teach this brain how to learn from the data.

Let's move on to [Chapter 4: LLM Training Process]

# Chapter 4: LLM Training Process

Welcome back, future LLM expert! In [Chapter 3: LLM Model Architecture (GPT-like)](03_llm_model_architecture__gpt_like__.md), you learned about the "brain" of our LLM, understanding its internal structure, from embeddings to Transformer Blocks, and how it's designed to make predictions. We also previously prepared our "study material" in [Chapter 2: Text Data Preparation](02_text_data_preparation_.md).

Now, imagine you have a brilliant student (our LLM model) and a stack of textbooks (our prepared text data). The student has a brain designed to learn, but it hasn't learned anything yet! This is where the **LLM Training Process** comes in.

### What is LLM Training?

This abstraction describes how our LLM "learns" from text data. It's like a student studying a textbook: the `TextDataset` prepares the lessons (text chunks), the `train_model` function guides the student through repeated practice (epochs) and feedback (loss calculation), and an `optimizer` helps the student adjust its understanding (model weights). The goal is for the model to get better at predicting the next word, becoming more fluent and coherent over time.

The problem this process solves is: **How do we make our LLM smart enough to generate human-like text?** The answer is by showing it a lot of text examples and helping it learn the patterns of language.

### Key Concepts of Training

Let's break down the core ideas behind how our LLM learns:

1.  **Epochs: Repeated Practice**
    *   Imagine a student reading an entire textbook from cover to cover. That's one "epoch."
    *   In LLM training, an **epoch** means the model has seen and processed *all* of its training data once.
    *   Just like a student might re-read a textbook multiple times to truly grasp the material, our LLM will go through many epochs (you configured `max_epochs` in [Chapter 1: User Configuration](01_user_configuration_.md)) to learn effectively.

2.  **Loss Calculation: Getting Feedback**
    *   After the model makes a prediction (e.g., it predicts "w" but the correct next character was "d"), we need to know how "wrong" it was.
    *   **Loss** is a number that tells us exactly this: how far off the model's prediction was from the actual correct answer. A higher loss means the model was very wrong; a lower loss means it was closer to correct.
    *   This is like a teacher giving feedback on a quiz: "You got 8 out of 10 wrong." The model uses this feedback to improve.

3.  **Optimizer: Adjusting Understanding**
    *   Once the model knows *how* wrong it was (the loss), it needs a way to *adjust* its internal "knowledge" (its "weights" or "parameters").
    *   The **optimizer** is like a wise tutor who tells the student *how* to change their study habits to get better results. It uses the `loss` to figure out which internal connections in the LLM's brain need to be strengthened or weakened.
    *   The `learning_rate` you set in [Chapter 1: User Configuration](01_user_configuration_.md) determines how big of a "step" the optimizer takes in adjusting the model's knowledge.

4.  **Batches: Learning in Chunks**
    *   Our LLM doesn't learn from one tiny character at a time, nor does it try to digest an entire book at once.
    *   Instead, it processes text in small groups called **batches**. A batch is a collection of `batch_size` (which you configured in [Chapter 1: User Configuration](01_user_configuration_.md)) text chunks.
    *   This is like a student doing a set of practice problems before checking the answers and making corrections, rather than checking after every single problem.

### How to Train Your LLM

You've already set the stage for training your LLM! In [Chapter 1: User Configuration](01_user_configuration_.md), you provided the `batch_size`, `learning_rate`, `max_epochs`, and `dataset_path`. All of these settings are used by the `train_llm.py` script to manage the training process.

To start the training, simply run:

```bash
python train_llm.py
```

After you enter your training configuration settings (like `max_epochs` and `dataset_path`), the script will load your model's blueprint from `model_config.pth`, prepare your text data, and then begin the training loop.

You will see output similar to this as training progresses:

```
Configure training parameters:
Enter batch size (e.g., 32): 16
Enter learning rate (e.g., 0.0003): 0.001
Enter number of epochs (e.g., 5): 10
Enter path to text dataset (e.g., 'input.txt'): my_story.txt
Training on cuda with 0.15M parameters # (or 'cpu' if no GPU)
Epoch 1/10, Loss: 1.8234
Epoch 2/10, Loss: 1.5510
Epoch 3/10, Loss: 1.4021
...
Epoch 10/10, Loss: 1.1250
```

Notice how the `Loss` number generally gets smaller with each epoch. This indicates that your LLM is learning and its predictions are becoming more accurate!

### Under the Hood: The `train_model` Function

All the magic of training happens within the `train_model` function in `train_llm.py`. Let's explore its step-by-step process.

#### The Training Loop Flow

Here's a simplified diagram of how the training process works:

```mermaid
sequenceDiagram
    participant You as User
    participant train_llm_script as train_llm.py
    participant DataLoader as DataLoader
    participant GPTLanguageModel as GPTLanguageModel
    participant Optimizer as Optimizer

    You->>train_llm_script: 1. Run script
    train_llm_script->>train_llm_script: 2. Load configs, prepare data
    train_llm_script->>GPTLanguageModel: 3. Initialize model
    train_llm_script->>Optimizer: 4. Initialize optimizer
    train_llm_script->>train_llm_script: 5. Start Epoch Loop (max_epochs times)
    train_llm_script->>DataLoader: 6. Request a Batch (xb, yb)
    DataLoader->>train_llm_script: 7. Provide input (xb) and target (yb) batch
    train_llm_script->>GPTLanguageModel: 8. Pass xb, yb to model (forward pass)
    GPTLanguageModel->>GPTLanguageModel: 9. Predict & calculate loss
    GPTLanguageModel-->>train_llm_script: 10. Return predictions (logits) and loss
    train_llm_script->>Optimizer: 11. Clear previous adjustments
    train_llm_script->>GPTLanguageModel: 12. Tell model to find how to adjust (backward pass)
    GPTLanguageModel-->>Optimizer: 13. Provides adjustment instructions
    Optimizer->>Optimizer: 14. Apply adjustments to model's brain
    Optimizer-->>train_llm_script: 15. Adjustments done
    train_llm_script->>train_llm_script: 16. Repeat for next batch
    train_llm_script->>train_llm_script: 17. Repeat for next epoch
    train_llm_script->>You: 18. Report final trained model
```

#### Code Details: Inside `train_model`

The `train_model` function brings together the model, the data, and the optimizer to perform the actual learning.

Here's a breakdown of the `train_model` function in `train_llm.py`:

```python
# From train_llm.py
def train_model(model, train_loader, config, device):
    # 1. Choose the "tutor" (optimizer) to help model adjust
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])
    
    # 2. Set the model to "training mode"
    model.train()
    
    # 3. Start the Epoch Loop (reading textbook multiple times)
    for epoch in range(config['max_epochs']):
        total_loss = 0
        # 4. Loop through each "batch" of study material
        for xb, yb in train_loader:
            # Move data to the right "desk" (CPU/GPU)
            xb, yb = xb.to(device), yb.to(device)
            
            # 5. Model makes predictions and calculates how "wrong" it was
            logits, loss = model(xb, yb) 
            
            # 6. Clear old adjustment plans
            optimizer.zero_grad(set_to_none=True)
            
            # 7. Model figures out *how* to adjust its internal knowledge
            loss.backward()
            
            # 8. The "tutor" (optimizer) applies the adjustments
            optimizer.step()
            
            # Keep track of the total "wrongness" for this epoch
            total_loss += loss.item()
            
        # 9. Report average "wrongness" (loss) for the epoch
        print(f"Epoch {epoch+1}/{config['max_epochs']}, Loss: {total_loss/len(train_loader):.4f}")
    return model
```

Let's understand each critical line:

```python
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])
```
*   `optimizer = torch.optim.AdamW(...)`: This line creates our "tutor." `AdamW` is a popular type of optimizer, chosen because it's good at finding efficient ways to adjust the model's internal numbers. It needs to know `model.parameters()` (all the adjustable parts of the LLM's brain) and the `learning_rate` you configured.

```python
    model.train()
```
*   `model.train()`: This line puts the model in "training mode." This is a technical step that tells certain layers (like `dropout` from [Chapter 1: User Configuration](01_user_configuration_.md)) to behave differently during training than they would during text generation (inference).

```python
    for epoch in range(config['max_epochs']):
        # ...
        for xb, yb in train_loader:
            # ...
```
*   These are the two main loops. The outer loop iterates for each `epoch` (each time the model reads all the data). The inner loop iterates through each `batch` of data provided by the `train_loader` (which gets its data from our `TextDataset` prepared in [Chapter 2: Text Data Preparation](02_text_data_preparation_.md)). `xb` is the input batch, and `yb` is the target batch.

```python
            xb, yb = xb.to(device), yb.to(device)
```
*   `xb, yb = xb.to(device), yb.to(device)`: This moves the `xb` (input) and `yb` (target) data to the appropriate processing unit, either your computer's CPU or, if available and configured, a faster GPU (`cuda`). This is important for performance!

```python
            logits, loss = model(xb, yb)
```
*   `logits, loss = model(xb, yb)`: This is the core "thinking" step. We pass the input `xb` and the correct `yb` (targets) to our `model`. As explained in [Chapter 3: LLM Model Architecture (GPT-like)](03_llm_model_architecture__gpt_like__.md), the `model` (specifically its `forward` method) will then:
    *   Process `xb` through its embedding layers and Transformer Blocks.
    *   Generate `logits` (raw prediction scores for the next character).
    *   Compare `logits` with `yb` (the true next characters) to calculate the `loss` (how "wrong" its predictions were). Both `logits` and `loss` are returned.

```python
            optimizer.zero_grad(set_to_none=True)
```
*   `optimizer.zero_grad()`: Before calculating new adjustments, we must "clear" any old adjustment instructions. Think of it like clearing a whiteboard before writing new notes.

```python
            loss.backward()
```
*   `loss.backward()`: This is a powerful step called "backpropagation." The model uses the `loss` value to figure out *how much* each of its internal numbers (parameters) contributed to that "wrongness." It then calculates precise "adjustment instructions" for every single part of the model.

```python
            optimizer.step()
```
*   `optimizer.step()`: Finally, the `optimizer` takes all those "adjustment instructions" calculated by `loss.backward()` and *applies* them to the `model`'s internal numbers. This is where the model actually "learns" and updates its understanding to make better predictions next time.

```python
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{config['max_epochs']}, Loss: {total_loss/len(train_loader):.4f}")
```
*   These lines simply keep track of the total `loss` for the current epoch and then print the average loss after all batches in that epoch have been processed. Seeing the loss decrease over epochs is a good sign that your LLM is learning!

After all epochs are complete, the `train_model` function returns the now "smarter" model. The `train_llm.py` script then saves this trained model to `trained_model.pth`.

### Conclusion

In this chapter, you've grasped the fundamental process by which your LLM learns. You've seen how the `train_model` function orchestrates repeated practice (epochs), calculates feedback (loss), and adjusts the model's understanding (optimizer) using batches of prepared text data. The goal is to continuously reduce the "loss," making your LLM better at predicting the next character and, ultimately, at generating coherent text.

Now that our LLM has a brain and has been taught how to learn, let's peek inside the "Transformer Building Blocks" that power its sophisticated thinking.

Let's move on to [Chapter 5: Transformer Building Blocks]

# Chapter 5: Transformer Building Blocks

Welcome back, future LLM expert! In [Chapter 4: LLM Training Process](04_llm_training_process_.md), you learned how to teach your LLM to "read" and "understand" text by repeatedly showing it examples and adjusting its "brain" based on how "wrong" its predictions were.

Now, let's open up that "brain" and peek inside! Imagine your LLM is a powerful, multi-story building. Each floor, or "layer," is designed to process information in a very clever way. These floors are built from special components, the most important of which are the **Transformer Building Blocks**.

### What are Transformer Building Blocks?

These blocks are the fundamental "gears" and "levers" that allow your LLM to "think" and understand language deeply. They solve the crucial problem of: **How does an LLM figure out which words in a sentence are important to each other, even if they are far apart, and then use that understanding to make smart predictions?**

Let's use an example: "The quick brown **fox** jumps over the lazy **dog**."
A human quickly understands that "fox" is doing the "jumping" and "dog" is "lazy." But for a computer, these are just separate words. Transformer Building Blocks help the LLM connect these dots, understanding relationships and context.

Our LLM, like many modern LLMs (including GPT models), relies heavily on two main components within each "Transformer Block":

1.  **Multi-Head Self-Attention**: This is like the model's ability to "focus" on different important parts of a sentence simultaneously. It helps the model understand how words relate to each other regardless of their position.
2.  **Feed-Forward Network**: This is a simpler neural network that takes the refined information from the attention mechanism and processes it further, allowing for deeper "thought" or transformations.

A `TransformerBlock` is a combination of these two, acting as a single "processing unit" that can be stacked many times to form the complete deep learning model.

### Key Concepts

Let's break down these essential components.

#### 1. Multi-Head Self-Attention: The "Focus" Mechanism

Imagine you're reading a sentence, and for each word, you ask: "Which other words in this sentence are important for understanding *this* word?"

**Self-Attention** is the core idea: when the LLM processes a word, it doesn't just look at that word in isolation. It looks at *all other words* in the input sequence to figure out how much "attention" it should pay to each of them. This helps it understand the word's meaning within its specific context.

*   **Analogy**: Think of it like a detective investigating a crime scene. For each piece of evidence (a word), the detective considers *all other evidence* (other words) to find connections and build a complete picture.

**Multi-Head** means the model doesn't just do this "focusing" once. It does it multiple times (as many as `n_head` you configured in [Chapter 1: User Configuration](01_user_configuration_.md)), each time looking for slightly different kinds of relationships or patterns.

*   **Analogy**: Instead of one detective, you have several detectives working in parallel, each focusing on a different aspect of the crime (e.g., one on motives, one on alibis, one on physical evidence). Then they combine their insights.

How does it "focus"? It uses three special "versions" of each word's numerical representation:
*   **Query (Q)**: "What am I looking for?" (the word currently being processed)
*   **Key (K)**: "What do I have?" (all other words in the sentence)
*   **Value (V)**: "What information should I extract if I decide to pay attention?" (the actual content of all other words)

The LLM compares a `Query` with all `Keys` to calculate "attention scores." High scores mean strong relationships, telling the model to pay more "attention" to those words' `Values`.

#### 2. TransformerBlock: The "Processing Unit"

A `TransformerBlock` is where the `MultiHeadSelfAttention` and a `Feed-Forward Network` come together. It's a full cycle of processing.

*   **Analogy**: It's like a single "floor" in our LLM's multi-story building. Each floor takes the text, applies sophisticated thinking (attention), refines that thinking (feed-forward network), and then passes the improved understanding up to the next floor.

Many `TransformerBlock`s are stacked on top of each other. The more layers (`n_layer` from [Chapter 1: User Configuration](01_user_configuration_.md)) you have, the deeper and more complex the LLM's understanding can become.

| Component                 | Purpose                                                                                                                                                                                                            |
| :------------------------ | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **MultiHeadSelfAttention**| Allows the model to weigh the importance of different words in the input sequence when processing a specific word, capturing long-range dependencies and contextual meaning.                                     |
| **Feed-Forward Network**  | A standard neural network that applies further transformations to the output of the attention mechanism. It helps the model learn more complex patterns and features from the attended information.                   |
| **Layer Normalization**   | (Like `ln1`, `ln2`) A technical trick that helps stabilize the training process, ensuring that the numbers flowing through the network remain in a healthy range.                                                   |
| **Skip Connections**      | (Adding `x + ...`) Allows the original input `x` to be added back to the output of the attention and feed-forward layers. This helps prevent information loss and makes it easier for very deep models to learn. |

### How It's Used in Your LLM

You don't directly "run" `MultiHeadSelfAttention` or `TransformerBlock` yourself. They are internal, specialized layers that are automatically part of the `GPTLanguageModel` when it's built.

Recall from [Chapter 3: LLM Model Architecture (GPT-like)](03_llm_model_architecture__gpt_like__.md) that your `GPTLanguageModel` is constructed by stacking `n_layer` number of `TransformerBlock`s. Each `TransformerBlock` then uses `n_head` instances of `MultiHeadSelfAttention` internally.

When you configure your model in `llm_model.py` and specify values for `n_head` and `n_layer`, you are directly influencing the number and complexity of these crucial building blocks within your LLM's brain!

### Under the Hood: Building Blocks in `llm_model.py`

Let's look at how these building blocks are assembled and how they process information within the `llm_model.py` file.

#### The Flow of Information

```mermaid
sequenceDiagram
    participant GPTLanguageModel as GPTModel
    participant TransformerBlock as TransformerBlock
    participant MultiHeadSelfAttention as MultiHeadSelfAttention
    participant FeedForwardNetwork as FFN

    GPTModel->>TransformerBlock: 1. Input text (processed token + position embeddings)
    Note over TransformerBlock: This is one of many stacked blocks.
    TransformerBlock->>MultiHeadSelfAttention: 2. Process input with attention
    MultiHeadSelfAttention-->>TransformerBlock: 3. Return contextually aware text
    TransformerBlock->>FFN: 4. Process attention output with FFN
    FFN-->>TransformerBlock: 5. Return deeply refined text
    TransformerBlock-->>GPTModel: 6. Pass refined text to next block (or final head)
    Note over GPTModel: Repeats for each TransformerBlock layer
```

#### Code Details: Inside `llm_model.py`

All the definitions for these blocks are found in `llm_model.py`.

**1. The `MultiHeadSelfAttention` Class**

This class defines how the "focusing" mechanism works.

```python
# From llm_model.py
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        self.n_head = n_head
        self.head_size = n_embd // n_head # Size of each attention head
        self.key = nn.Linear(n_embd, n_embd)   # Layer to generate 'Key' vectors
        self.query = nn.Linear(n_embd, n_embd) # Layer to generate 'Query' vectors
        self.value = nn.Linear(n_embd, n_embd) # Layer to generate 'Value' vectors
        self.proj = nn.Linear(n_embd, n_embd)  # Final linear projection layer
        self.dropout = nn.Dropout(dropout)
        # tril is a lower triangular matrix, used for masking future tokens (GPT is causal)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.size() # Batch, Sequence Length, Embedding Dimension
        # 1. Transform input 'x' into Query, Key, Value for all heads
        q = self.query(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)
        k = self.key(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)
        
        # 2. Calculate attention scores: Query @ Key (scaled)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_size))
        # 3. Apply masking: Prevent tokens from looking at future information
        att = att.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        att = torch.softmax(att, dim=-1) # 4. Normalize scores to probabilities
        att = self.dropout(att) # Apply dropout
        
        # 5. Multiply attention probabilities with Value vectors
        y = att @ v # Result of attention
        # 6. Recombine heads and apply final linear projection
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.dropout(self.proj(y))
```
*   The `__init__` sets up the different `nn.Linear` layers that generate the Query, Key, and Value representations.
*   The `forward` method is where the attention calculation happens. It first computes how similar each word's `Query` is to other words' `Keys` (the `q @ k.transpose` part), which gives us raw "attention scores." Then, it uses `softmax` to turn these scores into probabilities (how much "attention" to pay). Finally, it multiplies these probabilities by the `Value` vectors (`att @ v`), effectively creating a new representation for each word that is a weighted sum of all other words' content, based on their relevance. The `tril` part ensures the model only looks at words that came *before* it, mimicking how language flows.

**2. The `TransformerBlock` Class**

This class wraps the attention mechanism and a Feed-Forward Network into one reusable unit.

```python
# From llm_model.py
class TransformerBlock(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        # 1. Multi-Head Self-Attention layer
        self.attn = MultiHeadSelfAttention(n_embd, n_head, block_size, dropout)
        self.ln1 = nn.LayerNorm(n_embd) # LayerNorm before attention
        
        # 2. Feed-Forward Network (a simple neural network)
        self.ffn = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), # Expand dimension
            nn.ReLU(),                     # Apply non-linearity
            nn.Linear(4 * n_embd, n_embd), # Project back to original dimension
            nn.Dropout(dropout)
        )
        self.ln2 = nn.LayerNorm(n_embd) # LayerNorm before FFN

    def forward(self, x):
        # Apply attention with 'skip connection' and LayerNorm
        x = x + self.attn(self.ln1(x))
        # Apply Feed-Forward Network with another 'skip connection' and LayerNorm
        x = x + self.ffn(self.ln2(x))
        return x
```
*   The `__init__` defines the `self.attn` (our attention component) and `self.ffn` (a small, standard neural network with `nn.Linear` and `nn.ReLU`). It also includes `LayerNorm` layers to help stabilize learning, and `Dropout` for regularization.
*   The `forward` method shows the flow: the input `x` first goes through Layer Normalization, then `self.attn`. The result is added back to the original `x` (`x = x + ...`), which is a crucial "skip connection" for deep networks. This process is then repeated for the `self.ffn`.

These `TransformerBlock`s are the core of your LLM's "thinking" process, enabling it to understand complex language patterns and relationships crucial for generating coherent and contextually relevant text.

### Conclusion

In this chapter, you've looked inside the "brain" of your LLM and explored its core "Transformer Building Blocks." You've learned how `MultiHeadSelfAttention` allows the model to "focus" on important relationships within text, and how `TransformerBlock`s combine this attention with a Feed-Forward Network to form powerful processing units that are stacked to create deep understanding. These components, controlled by your `n_head` and `n_layer` configurations, are what give modern LLMs their impressive capabilities.

Now that we understand how the model works and how it learns, the next step is to make it super efficient for practical use.

Let's move on to [Chapter 6: Model Optimization (GGUF Conversion)]

# Chapter 6: Model Optimization (GGUF Conversion)

Welcome back, future LLM expert! In [Chapter 5: Transformer Building Blocks](05_transformer_building_blocks_.md), you dived deep into the internal "gears and levers" of your LLM's brain, understanding how Multi-Head Self-Attention and Feed-Forward Networks enable sophisticated language processing. You now have a trained LLM that understands and can generate text!

However, your trained LLM is like a powerful, high-performance racing car – it's designed for specialized tracks (powerful GPUs during training) and might not be practical for everyday driving on regular roads (your personal computer's CPU). This is where **Model Optimization (GGUF Conversion)** comes in.

### What is Model Optimization (GGUF Conversion)?

This process is like **packaging a large, complex machine into a smaller, more efficient version for distribution**. Imagine you've just built a massive, super-detailed sculpture. It's beautiful, but hard to move. You want to make a smaller, lighter version that still looks great and can be easily shared or displayed anywhere.

The problem this abstraction solves is: **How do we make our large, trained LLM smaller, faster, and more usable on common computers (like your laptop's CPU), without losing too much of its intelligence?**

Our goal is to convert the PyTorch model (`trained_model.pth`) into the **GGUF format**. GGUF is specifically designed for faster and more memory-efficient text generation (inference), especially on consumer hardware like CPUs.

### Key Concepts for Efficient Models

Let's break down the ideas behind making our LLM more "portable" and efficient:

1.  **The GGUF Format: The Efficient Package**
    *   **What it is**: GGUF is a special file format developed by the `llama.cpp` project. Think of it as a highly optimized "zip file" specifically for LLMs. It packs the model's information in a way that is very quick to load and run on many different types of hardware, especially CPUs.
    *   **Why it's needed**: Your `trained_model.pth` is a standard PyTorch file, which is great for training, but not always the most efficient for running on basic hardware. GGUF solves this by laying out the model's "brain" in a way that computers can process incredibly fast.

2.  **Quantization: Making Numbers Smaller**
    *   **What it is**: Inside your LLM, all the "knowledge" is stored as numbers (its "weights" or "parameters"). Originally, these numbers are usually very precise (like `3.1415926535`). Quantization is the process of reducing this precision, for example, making `3.1415926535` into just `3.14` or even just `3`.
    *   **Analogy**: It's like taking a high-resolution photograph (many colors, lots of detail) and converting it to a lower-resolution one. It looks *almost* the same, but the file size is much, much smaller.
    *   **Benefit**: When numbers are less precise, they take up less space in memory and can be processed much faster. This makes the model file smaller and quicker to run.
    *   **Trade-off**: While the model remains largely accurate, there might be a very slight reduction in performance or quality, similar to how a lower-res photo might lose tiny details. We try to find a good balance. Common quantization types are `Q4_0` (4-bit quantization, very small) or `Q8_0` (8-bit quantization, slightly larger but often more accurate).

3.  **Hugging Face Format: The Bridge**
    *   Before we can convert our model to GGUF, we need an intermediate step. The tools that convert to GGUF usually expect models to be in a common format, specifically the **Hugging Face `transformers` library format**.
    *   **Why it's needed**: Our `trained_model.pth` is a custom PyTorch model. We need to temporarily "wrap" our model so it looks like a standard Hugging Face model. This "bridge" allows the GGUF conversion tools to understand its structure and weights.

4.  **`llama.cpp`: The Conversion Toolset**
    *   **What it is**: `llama.cpp` is an amazing open-source project that specializes in running LLMs very efficiently on CPUs. It provides the tools necessary to convert models into the GGUF format and then run them.
    *   **Prerequisite**: To use `llama.cpp`'s conversion tools, you first need to download and "build" (compile) the `llama.cpp` project on your computer. This creates the special programs (`convert_hf_to_gguf.py`, `llama-quantize`) that we'll use.

### How to Convert Your LLM to GGUF

The `custom-llm-scripts` project provides a script called `convert_to_gguf.py` to handle all these steps for you!

Before you run it, make sure you have `llama.cpp` set up.
1.  **Clone `llama.cpp`**: `git clone https://github.com/ggerganov/llama.cpp`
2.  **Build `llama.cpp`**: `cd llama.cpp; make` (This command will compile the necessary tools for your system. It might take a few minutes.)

Once `llama.cpp` is ready, simply run the conversion script:

```bash
python convert_to_gguf.py
```

By default, this command will:
1.  Load your `trained_model.pth` and `model_config.pth`.
2.  Export them temporarily into a Hugging Face-compatible format.
3.  Use the `llama.cpp` tools to convert that into a GGUF file.
4.  Quantize the GGUF file using the `Q4_0` (4-bit) quantization type.

You can also specify a different quantization type using the `--quant_type` argument:

```bash
python convert_to_gguf.py --quant_type Q8_0
```

The script will produce output indicating its progress, and if successful, you will find a new GGUF file in your project directory, typically named something like `model.gguf.Q4_0.gguf` (or `model.gguf.Q8_0.gguf` if you chose `Q8_0`).

This new `.gguf` file is your optimized, ready-to-use LLM!

### Under the Hood: The Conversion Process

Let's look at how the `convert_to_gguf.py` script orchestrates this optimization.

#### The Conversion Flow

```mermaid
sequenceDiagram
    participant You as User
    participant convert_script as convert_to_gguf.py
    participant HF_Export as export_to_hf()
    participant LlamaCPP_Convert as convert_to_gguf() (internal)
    participant LlamaCPP_Quantize as llama-quantize

    You->>convert_script: 1. Run conversion script
    convert_script->>HF_Export: 2. Export our custom model to Hugging Face format
    Note over HF_Export: Wraps model, saves config.json and weights.
    HF_Export-->>convert_script: 3. Returns path to temporary HF model
    convert_script->>LlamaCPP_Convert: 4. Call llama.cpp's convert_hf_to_gguf.py
    Note over LlamaCPP_Convert: Reads HF model, creates initial F16 GGUF.
    LlamaCPP_Convert-->>convert_script: 5. F16 GGUF file created
    convert_script->>LlamaCPP_Quantize: 6. Call llama.cpp's llama-quantize tool
    Note over LlamaCPP_Quantize: Reads F16 GGUF, makes smaller quantized GGUF.
    LlamaCPP_Quantize-->>convert_script: 7. Quantized GGUF file created
    convert_script->>You: 8. Conversion complete!
```

#### Code Details: Inside `convert_to_gguf.py`

The `convert_to_gguf.py` script contains two main functions that perform the steps described above.

**1. `export_to_hf` function: Creating the Hugging Face Bridge**

This function takes your trained PyTorch model and its configuration and saves them in a format that the Hugging Face `transformers` library understands. This involves creating special "wrapper" classes (`GPTConfig`, `GPTPreTrainedModel`) that help our custom model look like a standard `transformers` model.

```python
# From convert_to_gguf.py
class GPTConfig(PretrainedConfig):
    # This class tells Hugging Face how our model's settings map to its own
    model_type = "gpt2" # We pretend it's a GPT-2 for compatibility
    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size, dropout, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = n_embd # Our n_embd is HF's hidden_size
        # ... and so on for other settings ...

class GPTPreTrainedModel(PreTrainedModel):
    # This class wraps our custom GPTLanguageModel
    config_class = GPTConfig
    def __init__(self, config):
        super().__init__(config)
        # Here we connect our actual GPTLanguageModel
        self.model = GPTLanguageModel(config.__dict__)
    # We also need a simple forward method to make it work with HF's save_pretrained

def export_to_hf(model_path='trained_model.pth', config_path='model_config.pth', output_dir='hf_model'):
    # Load our config and model
    config_dict = torch.load(config_path)
    config = GPTConfig(**config_dict) # Use our wrapper config
    model = GPTPreTrainedModel(config) # Use our wrapper model
    model.load_state_dict(torch.load(model_path)) # Load the trained weights
    
    # Save it in Hugging Face format
    os.makedirs(output_dir, exist_ok=True) # Create folder for saving
    model.save_pretrained(output_dir) # This is the magic HF saving function
    # Also save a simple config.json for compatibility
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config.to_dict(), f)
    logger.info(f"Exported to HF format at {output_dir}")
    return output_dir
```
This function first loads your `model_config.pth` and `trained_model.pth`. It then creates instances of `GPTConfig` and `GPTPreTrainedModel` (which are special classes designed to make our custom model look like a standard Hugging Face model). Finally, it uses `model.save_pretrained()` to save the model and its configuration into a new directory (`hf_model`), ready for the next step.

**2. `convert_to_gguf` function: Running `llama.cpp` tools**

This function uses Python's `subprocess` module to run the command-line tools provided by `llama.cpp`.

```python
# From convert_to_gguf.py
def convert_to_gguf(hf_dir, gguf_output='model.gguf', quant_type='Q4_0', llama_cpp_path='llama.cpp'):
    # Path to llama.cpp's conversion script
    convert_script = os.path.join(llama_cpp_path, 'convert_hf_to_gguf.py')
    
    # First, convert HF model to GGUF (initially as F16, full precision)
    subprocess.run(['python', convert_script, hf_dir, '--outfile', gguf_output, '--outtype', 'f16'], check=True)
    
    # Second, quantize the GGUF model to a smaller size
    quant_exe = os.path.join(llama_cpp_path, 'llama-quantize')
    subprocess.run([quant_exe, gguf_output, f"{gguf_output}.{quant_type}.gguf", quant_type], check=True)
    logger.info(f"Converted to GGUF: {gguf_output}.{quant_type}.gguf")
```
This function performs two main steps using `llama.cpp`'s tools:
*   It first runs `convert_hf_to_gguf.py` to convert the Hugging Face format model into an initial GGUF file, typically with full precision (F16).
*   Then, it runs `llama-quantize` on that F16 GGUF file to create the final, smaller, quantized GGUF file (e.g., `Q4_0`). The `check=True` argument ensures that if any of these commands fail, Python will raise an error, letting you know.

After these steps, you'll have a highly optimized `.gguf` file ready to be used for efficient text generation!

### Conclusion

In this chapter, you've learned the critical final step in preparing your LLM for practical use: Model Optimization through GGUF conversion. You now understand that this involves packaging your model into an efficient format and "quantizing" its numbers to make it smaller and faster, especially for running on common consumer hardware like CPUs. You've also seen how the `convert_to_gguf.py` script, leveraging the power of `llama.cpp` and Hugging Face's tools, handles this complex process for you.

With your LLM now optimized and ready, it's time to bring it to life and make it generate some text!

Let's move on to [Chapter 7: Text Generation (Inference)]

# Chapter 7: Text Generation (Inference)

Welcome to the grand finale of our journey into building a custom LLM! In [Chapter 6: Model Optimization (GGUF Conversion)](06_model_optimization__gguf_conversion__.md), you learned how to take your trained LLM and optimize it into a super-efficient `.gguf` file, making it ready for practical use on your own computer.

Now, it's time to bring your LLM to life and watch it perform its most exciting trick: **Text Generation**, also known as **Inference**.

### What is Text Generation (Inference)?

This is the "creative writing" phase of the LLM. After all the training and optimization, this is how we ask the model to write new text. It's like having a digital storyteller that can continue a narrative based on your prompt, or even answer simple questions.

The problem this abstraction solves is: **How do we make our LLM produce new, coherent, and human-like text based on a starting idea?**

Imagine you want your LLM to finish a story you started. You give it a few sentences (this is your "query" or "prompt"), and the LLM then predicts the most likely next word, then the next, and so on, building a complete response.

**Our Use Case**: Asking your trained LLM to generate a continuation of a given phrase, using the optimized `.gguf` file.

### Key Concepts for Generating Text

When an LLM generates text, it's not just randomly picking words. It's making careful predictions based on everything it learned during training.

1.  **Inference**:
    *   This term simply means *using* a trained model to make predictions or generate output. It's different from "training," where the model is learning.
    *   Think of it like this: Training is a student studying for a test. Inference is the student taking the test (applying what they learned).

2.  **Query (or Prompt)**:
    *   This is the starting phrase or question you give to the LLM. It's the spark that ignites the model's creativity.
    *   Example: "Once upon a time, in a land far away,"

3.  **Token-by-Token Prediction**:
    *   LLMs generate text one "token" (in our case, one character) at a time.
    *   It predicts the most likely next character based on the query, then adds that character to the query, and predicts the *next* one, and so on. This continues until it reaches a desired length or generates a special "end of text" character.

4.  **Sampling Parameters (Controlling Creativity)**:
    *   If the LLM *always* picked the single most likely next character, its text might be boring and repetitive. Sampling parameters allow you to control the randomness and creativity of the output.

    | Parameter      | What it does                                                                                                                                                                                                                             | Analogy                                                                                |
    | :------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------- |
    | **Temperature**| A higher temperature (e.g., 0.8-1.0+) makes the model's choices "softer," allowing it to pick less probable but more creative words. A lower temperature (e.g., 0.1-0.5) makes it more deterministic and predictable, sticking to common choices. | Like setting the creativity dial: High temperature for wild ideas, low for strict rules. |
    | **Top-K**      | Limits the model's choices to only the `K` most probable next tokens. If `K=10`, the model will only pick from the top 10 most likely characters, ignoring all others. This prevents truly outlandish predictions.                           | Like a multiple-choice test: Only choose from the top 'K' options you think are right.  |

### How to Generate Text with Your LLM

You will use the `inference_gguf.py` script to generate text with your optimized `.gguf` model. This script leverages the `llama-cpp-python` library, which allows us to load and run the GGUF models efficiently.

To use it, you'll need to specify a few things:
*   The path to your `.gguf` file (which you created in Chapter 6).
*   Your `query` (the starting text).
*   Optionally, `max_tokens` (how many new characters to generate) and `use_gpu` (if you want to try running on your GPU for speed, if supported).

To run the inference, open your terminal and navigate to your project directory. Then, execute the script like this:

```bash
python inference_gguf.py --gguf_path model.gguf.Q4_0.gguf --query "The quick brown fox" --max_tokens 50
```

**Example Output:**

```
The quick brown fox jumped over the lazy dog. He then ran into the forest, looking for food. The sun was setting, casting long shadows across the trees. It was a peaceful evening.
```
*(Your actual output will vary because LLMs are creative!)*

If you want to try using your GPU (if `llama-cpp-python` and your system support it, e.g., Metal for Mac, or CUDA for Nvidia GPUs), you can add the `--use_gpu` flag:

```bash
python inference_gguf.py --gguf_path model.gguf.Q4_0.gguf --query "Once upon a time" --max_tokens 100 --use_gpu
```

### Under the Hood: `inference_gguf.py`

Let's see how the `inference_gguf.py` script uses your `.gguf` model to generate text.

#### The Inference Flow

```mermaid
sequenceDiagram
    participant You as User
    participant InferenceScript as inference_gguf.py
    participant LlamaCPP as llama_cpp.Llama

    You->>InferenceScript: 1. Run script with GGUF path and query
    InferenceScript->>LlamaCPP: 2. Initialize Llama model with gguf_path
    Note over LlamaCPP: Loads the optimized model from the .gguf file.
    InferenceScript->>LlamaCPP: 3. Send query and generation parameters (max_tokens, etc.)
    Note over LlamaCPP: Model predicts character by character.
    LlamaCPP-->>InferenceScript: 4. Return generated text
    InferenceScript->>You: 5. Print the generated text
```

#### Code Details: Inside `inference_gguf.py`

The `inference_gguf.py` script is quite concise, thanks to the `llama-cpp-python` library which handles the complex parts of loading and running the GGUF model.

Here's the core of the `run_inference` function:

```python
# From inference_gguf.py
from llama_cpp import Llama # Import the Llama class

def run_inference(gguf_path, query, max_tokens=100, use_gpu=False):
    # Determine if GPU layers should be offloaded
    n_gpu_layers = -1 if use_gpu else 0 # -1 means offload all layers to GPU
    
    # 1. Initialize the Llama model
    # This loads your optimized .gguf file into memory
    llm = Llama(gguf_path, n_gpu_layers=n_gpu_layers, verbose=False)
    
    # 2. Perform the text generation
    # The 'query' is your prompt, 'max_tokens' is the length
    output = llm(query, max_tokens=max_tokens, echo=True)
    
    # 3. Extract the generated text from the output
    response = output['choices'][0]['text']
    
    return response
```

Let's break down the key lines:

```python
    from llama_cpp import Llama
```
*   `from llama_cpp import Llama`: This line imports the `Llama` class from the `llama-cpp-python` library. This `Llama` class is a Python "wrapper" around the highly optimized `llama.cpp` code, allowing us to easily interact with our GGUF model.

```python
    n_gpu_layers = -1 if use_gpu else 0
```
*   `n_gpu_layers`: This parameter tells `llama.cpp` how many of the model's layers should be loaded onto the GPU. If `use_gpu` is `True`, `-1` means "load all layers onto the GPU" (if possible). If `False`, `0` means "load no layers onto the GPU" (run entirely on CPU).

```python
    llm = Llama(gguf_path, n_gpu_layers=n_gpu_layers, verbose=False)
```
*   `llm = Llama(...)`: This is the crucial step where your GGUF model is loaded. You pass the `gguf_path` to the `Llama` constructor. It reads the `.gguf` file, sets up the model's internal structure in memory, and prepares it for generation. This is also where the `n_gpu_layers` setting takes effect.

```python
    output = llm(query, max_tokens=max_tokens, echo=True)
```
*   `output = llm(...)`: This line *calls* the `llm` object like a function. This is the actual command to start text generation!
    *   `query`: Your input prompt (e.g., "The quick brown fox").
    *   `max_tokens`: The maximum number of new characters the model should generate.
    *   `echo=True`: This tells the model to include your original `query` in the output, so you see the complete generated text.

The `llama-cpp-python` library handles the intricate process of tokenizing your query, passing it through the model layer by layer, predicting the next token, applying sampling parameters (like `temperature` and `top_k`, although not explicitly shown in this snippet, `llama_cpp.Llama` supports them in its `__call__` method for more advanced control), and repeating this process until `max_tokens` is reached.

```python
    response = output['choices'][0]['text']
```
*   `response = ...`: The output from `llm()` is a dictionary containing the generated text. This line simply extracts the actual text string from that dictionary.

The `inference_gguf.py` script also includes basic resource monitoring (CPU, RAM, GPU usage), which is helpful to see how much power your LLM is using while generating text.

### Conclusion

Congratulations! You've reached the end of your journey with `custom-llm-scripts`. In this final chapter, you've learned about Text Generation (Inference), the exciting process of making your trained and optimized LLM write new text. You now understand how to provide a starting query, how the model predicts character by character, and how sampling parameters like `temperature` and `top_k` influence its creativity. Most importantly, you know how to run the `inference_gguf.py` script to interact with your very own custom-built LLM.

From configuring your model's "brain size" to preparing its "study material," teaching it to learn, understanding its internal "thinking" blocks, optimizing it for efficiency, and finally, making it generate text, you've gained a comprehensive, beginner-friendly understanding of the entire LLM pipeline.

Go forth and generate amazing text with your custom LLM!

---