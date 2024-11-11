# App Review Sentiment Analysis for Microsoft Office

The **App Review Sentiment Analysis** project aims to analyze and classify user sentiments from reviews of the Microsoft Office application. By utilizing comprehensive text preprocessing techniques and robust deep learning models, this project seeks to provide insights into user satisfaction, highlight areas for improvement, and enhance overall user experience.

## Table of Contents

- [Features](#features)
- [Dataset](#dataset)
  - [Dataset Source](#dataset-source)
  - [Dataset Structure](#dataset-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Data Preprocessing](#data-preprocessing)
  - [Cleaning Text](#cleaning-text)
  - [Handling Slang Words](#handling-slang-words)
- [Modeling](#modeling)
  - [ResNet Model](#resnet-model)
  - [Bidirectional LSTM Model](#bidirectional-lstm-model)
  - [Convolutional Neural Network (CNN) Model](#convolutional-neural-network-cnn-model)
- [Evaluation](#evaluation)
- [Results](#results)
- [Saving Models](#saving-models)
- [Inference](#inference)

## Features

- **Advanced Text Preprocessing**  
Cleans and preprocesses review texts by removing URLs, numbers, punctuation, and handling slang words.
- **Slang Words Dictionary**  
Translates informal slang words into their formal counterparts to improve model accuracy.
- **Sentiment Labeling**  
Uses positive and negative lexicons to assign sentiment scores and labels to each review.
- **Deep Learning Models**  
  - Implements a Residual Network architecture for sentiment classification.
  - Utilizes a Bidirectional Long Short-Term Memory network for capturing contextual information.
  - Applies CNN for feature extraction and sentiment classification.
- **Model Evaluation**  
Assesses model performance using accuracy and loss metrics.
- **Visualization**  
Generates word clouds, histograms, and bar charts to visualize sentiment distributions and common words.
- **Inference Function**  
Provides a function to predict sentiments for new, unseen reviews.

## Dataset

### Dataset Source

The dataset used in this project consists of app reviews for Microsoft Office in Play Store. The data is gathered by scraping reviews from the Play Store using the `google-play-scraper` library.
The reviews includes user-generated content that reflects their experiences and sentiments towards the application.

### Dataset Structure

#### `app_reviews.csv`

| Column           | Description                                        | Data Type |
|------------------|----------------------------------------------------|-----------|
| `content`        | The text content of the app review                 | String    |
| `sentiment_score`| Numerical score representing sentiment strength    | Integer   |
| `sentiment`      | Categorical sentiment label (Positive, Neutral, Negative) | String    |
| `text_processed` | The preprocessed text after cleaning and stemming  | String    |
| `sentiment_encoded` | Numerical encoding of sentiment labels         | Integer   |

## Installation

Follow these steps to set up the project locally:

1. **Clone the Repository**

    ```bash
    git clone https://github.com/alfikiafan/app-review-sentiment-analysis.git
    cd app-review-sentiment-analysis
    ```

2. **Create and Activate a Virtual Environment**

    ```bash
    python -m venv env
    source env/bin/activate  # For Linux/Mac
    env\Scripts\activate     # For Windows
    ```

3. **Install Dependencies**

    ```bash
    pip install -r requirements.txt
    ```

    **Note**: Ensure you have `pip` installed. If not, install it from [here](https://pip.pypa.io/en/stable/installation/).

4. **Download NLTK Data**

    The project uses NLTK for text processing. Download the necessary NLTK data by running:

    ```python
    import nltk
    nltk.download('punkt')
    ```

## Usage

1. **Load the Dataset**

    Ensure that `app_reviews.csv` is placed in the root directory of the project.

2. **Run the Jupyter Notebook**

    Open the Jupyter notebook and execute each cell sequentially to perform data preprocessing, modeling, evaluation, and visualization.

    ```bash
    jupyter notebook model_training.ipynb
    ```

3. **View Results**

    After running the notebook, review the generated visualizations and evaluation metrics to understand model performance.

## Data Preprocessing

### Cleaning Text

The preprocessing pipeline involves several steps to clean and standardize the review texts:

- **Removing URLs**: Eliminates any URLs present in the text.
- **Removing Numbers**: Strips out numerical values.
- **Removing Punctuation**: Removes punctuation marks.
- **Handling Newlines and Spaces**: Replaces newlines with spaces and trims leading/trailing spaces.

### Handling Slang Words

Given that app reviews often contain informal language and slang, a comprehensive dictionary of slang words is used to translate them into formal terms. This step enhances the quality of the text data, making it more suitable for sentiment analysis.

## Modeling

Three different deep learning models are implemented to classify sentiments:

### ResNet Model

- **Architecture**  
Residual Network with convolutional layers and residual blocks.
- **Feature Extraction**  
Uses tokenization and padding to prepare input sequences.
- **Compilation**  
Optimized using Adam optimizer with sparse categorical crossentropy loss.
- **Training**  
Utilizes early stopping to prevent overfitting.

### Bidirectional LSTM Model

- **Architecture**  
Bidirectional LSTM network that captures dependencies in both directions.
- **Feature Extraction**  
Similar tokenization and padding as ResNet.
- **Compilation**  
Adam optimizer with sparse categorical crossentropy loss.
- **Training**  
Early stopping based on validation loss.

### Convolutional Neural Network (CNN) Model

- **Architecture**  
CNN with convolutional and pooling layers for feature extraction.
- **Feature Extraction**  
Tokenization and padding of input sequences.
- **Compilation**  
Adam optimizer with sparse categorical crossentropy loss.
- **Training**  
Early stopping based on validation loss.

## Evaluation

Each model is evaluated based on:

- **Accuracy**: Measures the proportion of correctly classified instances.
- **Loss**: Represents the model's error during training and validation.

### Evaluation Results

| Model      | Loss  | Validation Accuracy |
|------------|-------|---------------------|
| ResNet     | 0.130 | 95.15%              |
| LSTM       | 0.218 | 92.75%              |
| CNN        | 0.161 | 93.40%              |

**Interpretation**:
- The CNN model outperforms the ResNet and LSTM models in terms of both loss and validation accuracy.
- All models demonstrate reasonable performance, with room for further optimization.

## Results

- **Word Clouds**: Visual representations of the most frequent words in positive, neutral, and negative sentiments.
- **Histograms**: Distribution of sentiment scores to understand the spread and central tendencies.
- **Bar Charts**: Top 20 most frequent words across all reviews.

**Word Cloud for Positive Sentiments**  

![Word Cloud Positive](/img/positive-sentiment-word-cloud.png)

**Word Cloud for Neutral Sentiments**  

![Word Cloud Neutral](/img/neutral-sentiment-word-cloud.png)

**Word Cloud for Negative Sentiments**  

![Word Cloud Negative](/img/negative-sentiment-word-cloud.png)

## Saving Models

The trained models are saved for future use in inference tasks:

- **LSTM Model**: `model_lstm.keras`
- **ResNet Model**: `model_resnet.keras`
- **CNN Model**: `model_cnn.keras`

## Inference

An inference function is provided to predict sentiments of new app reviews. The function preprocesses the input text and uses the trained models to classify the sentiment as Positive, Neutral, or Negative.

**Example Usage**:

```python
# Sample Texts with Manual Labels
sample_texts = [
    ("Mantap mantap sekali. Luar biasa. Ok.", "Positive"),
    ("Bisa untuk membuka file dari email, cukup dibutuhkan oleh semua orang", "Neutral"),
    ("Aplikasi jelek, banyak bug. Bikin kecewa", "Negative")
]

# Perform Inference
results = inference_function(sample_texts)

# Display Results
print(results)