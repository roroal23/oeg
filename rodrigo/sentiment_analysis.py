import pandas as pd
import numpy as np
from torch.optim import AdamW
from transformers import BertTokenizer, BertForSequenceClassification
from fine_tuning_pipeline import FineTuningPipeline

def preprocess_dataset(path):
    """ Remove unnecessary characters and encode the sentiment labels.

    The type of preprocessing required changes based on the dataset. For the
    IMDb dataset, the review texts contains HTML break tags (<br/>) leftover
    from the scraping process, and some unnecessary whitespace, which are
    removed. Finally, encode the sentiment labels as 0 for "negative" and 1 for
    "positive". This method assumes the dataset file contains the headers
    "review" and "sentiment".

    Parameters:
        path (str): A path to a dataset file containing the sentiment analysis
            dataset. The structure of the file should be as follows: one column
            called "review" containing the review text, and one column called
            "sentiment" containing the ground truth label. The label options
            should be "negative" and "positive".

    Returns:
        df_dataset (pd.DataFrame): A DataFrame containing the raw data
            loaded from the self.dataset path. In addition to the expected
            "review" and "sentiment" columns, are:

            > review_cleaned - a copy of the "review" column with the HTML
                break tags and unnecessary whitespace removed

            > sentiment_encoded - a copy of the "sentiment" column with the
                "negative" values mapped to 0 and "positive" values mapped
                to 1
    """
    df_dataset = pd.read_csv(path)

    df_dataset['review_cleaned'] = df_dataset['review'].\
        apply(lambda x: x.replace('<br />', ''))

    df_dataset['review_cleaned'] = df_dataset['review_cleaned'].\
        replace(r'\s+', ' ', regex=True)

    df_dataset['sentiment_encoded'] = df_dataset['sentiment'].\
        apply(lambda x: 0 if x == 'negative' else 1)

    return df_dataset

if __name__ == "__main__":

    #Parameters
    dataset = preprocess_dataset('IMDB Dataset.csv')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased', num_labels = 2)
    optimizer = AdamW(model.parameters())

    #Fine-tune model
    fine_tuned_model = FineTuningPipeline(
        dataset = dataset,
        tokenizer = tokenizer,
        model = model,
        optimizer = optimizer,
        val_size = 0.1,
        epochs = 2,
        seed = 31,
        seq_max_length = 128,
        batch_size = 8,
    )

    probabilities = fine_tuned_model.predict(fine_tuned_model.val_dataloader)
    predicitons = np.argmax(probabilities, axis = 1)

    print("Sample predictions: (0: negative, 1: positive)")
    print(predicitons[:5])