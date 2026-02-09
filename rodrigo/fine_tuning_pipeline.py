from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import TensorDataset, DataLoader
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    get_linear_schedule_with_warmup)


class FineTuningPipeline:

    def __init__(
            self,
            dataset,
            tokenizer,
            model,
            optimizer,
            loss_function=nn.CrossEntropyLoss(),
            val_size=0.1,
            epochs=4,
            seed=42,
            seq_max_length = 512,
            batch_size=16):

        self.df_dataset = dataset
        self.tokenizer = tokenizer
        self.model = model
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.val_size = val_size
        self.epochs = epochs
        self.seed = seed

        self.seq_max_length = seq_max_length
        self.batch_size = batch_size

        # Check if GPU is available for faster training time
        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')

        # Perform fine-tuning
        self.model.to(self.device)
        self.set_seeds()
        self.token_ids, self.attention_masks = self.tokenize_dataset()
        self.train_dataloader, self.val_dataloader = self.create_dataloaders()
        self.scheduler = self.create_scheduler()
        self.fine_tune()

    def tokenize(self, text):
        """ Tokenize input text and return the token IDs and attention mask.

        Tokenize an input string, setting a maximum length of 512 tokens.
        Sequences with more than 512 tokens will be truncated to this limit,
        and sequences with less than 512 tokens will be supplemented with [PAD]
        tokens to bring them up to this limit. The datatype of the returned
        tensors will be the PyTorch tensor format. These return values are
        tensors of size 1 x max_length where max_length is the maximum number
        of tokens per input sequence (512 for BERT).

        Parameters:
            text (str): The text to be tokenized.

        Returns:
            token_ids (torch.Tensor): A tensor of token IDs for each token in
                the input sequence.

            attention_mask (torch.Tensor): A tensor of 1s and 0s where a 1
                indicates a token can be attended to during the attention
                process, and a 0 indicates a token should be ignored. This is
                used to prevent BERT from attending to [PAD] tokens during its
                training/inference.
        """
        batch_encoder = self.tokenizer(
            text,
            max_length=self.seq_max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt')

        token_ids = batch_encoder['input_ids']
        attention_mask = batch_encoder['attention_mask']

        return token_ids, attention_mask

    def tokenize_dataset(self):
        """ Apply the self.tokenize method to the fine-tuning dataset.

        Tokenize and return the input sequence for each row in the fine-tuning
        dataset given by self.dataset. The return values are tensors of size
        len_dataset x max_length where len_dataset is the number of rows in the
        fine-tuning dataset and max_length is the maximum number of tokens per
        input sequence (512 for BERT).

        Parameters:
            None.

        Returns:
            token_ids (torch.Tensor): A tensor of tensors containing token IDs
            for each token in the input sequence.

            attention_masks (torch.Tensor): A tensor of tensors containing the
                attention masks for each sequence in the fine-tuning dataset.
        """
        token_ids = []
        attention_masks = []

        for review in self.df_dataset['review_cleaned']:
            tokens, masks = self.tokenize(review)
            token_ids.append(tokens)
            attention_masks.append(masks)

        token_ids = torch.cat(token_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)

        return token_ids, attention_masks

    def create_dataloaders(self):
        """ Create dataloaders for the train and validation set.

        Split the tokenized dataset into train and validation sets according to
        the self.val_size value. For example, if self.val_size is set to 0.1,
        90% of the data will be used to form the train set, and 10% for the
        validation set. Convert the "sentiment_encoded" column (labels for each
        row) to PyTorch tensors to be used in the dataloaders.

        Parameters:
            None.

        Returns:
            train_dataloader (torch.utils.data.dataloader.DataLoader): A
                dataloader of the train data, including the token IDs,
                attention masks, and sentiment labels.

            val_dataloader (torch.utils.data.dataloader.DataLoader): A
                dataloader of the validation data, including the token IDs,
                attention masks, and sentiment labels.

        """
        train_ids, val_ids = train_test_split(
            self.token_ids,
            test_size=self.val_size,
            shuffle=False)

        train_masks, val_masks = train_test_split(
            self.attention_masks,
            test_size=self.val_size,
            shuffle=False)

        labels = torch.tensor(self.df_dataset['sentiment_encoded'].values)
        train_labels, val_labels = train_test_split(
            labels,
            test_size=self.val_size,
            shuffle=False)

        train_data = TensorDataset(train_ids, train_masks, train_labels)
        train_dataloader = DataLoader(train_data, shuffle=True, batch_size= self.batch_size)
        val_data = TensorDataset(val_ids, val_masks, val_labels)
        val_dataloader = DataLoader(val_data, batch_size= self.batch_size)

        return train_dataloader, val_dataloader

    def create_scheduler(self):
        """ Create a linear scheduler for the learning rate.

        Create a scheduler with a learning rate that increases linearly from 0
        to a maximum value (called the warmup period), then decreases linearly
        to 0 again. num_warmup_steps is set to 0 here based on an example from
        Hugging Face:

        https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2
        d008813037968a9e58/examples/run_glue.py#L308

        Read more about schedulers here:

        https://huggingface.co/docs/transformers/main_classes/optimizer_
        schedules#transformers.get_linear_schedule_with_warmup
        """
        num_training_steps = self.epochs * len(self.train_dataloader)
        scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=0,
            num_training_steps=num_training_steps)

        return scheduler

    def set_seeds(self):
        """ Set the random seeds so that results are reproduceable.

        Parameters:
            None.

        Returns:
            None.
        """
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

    def fine_tune(self):
        """Train the classification head on the BERT model.

        Fine-tune the model by training the classification head (linear layer)
        sitting on top of the BERT model. The model trained on the data in the
        self.train_dataloader, and validated at the end of each epoch on the
        data in the self.val_dataloader. The series of steps are described
        below:

        Training:

        > Create a dictionary to store the average training loss and average
          validation loss for each epoch.
        > Store the time at the start of training, this is used to calculate
          the time taken for the entire training process.
        > Begin a loop to train the model for each epoch in self.epochs.

        For each epoch:

        > Switch the model to train mode. This will cause the model to behave
          differently than when in evaluation mode (e.g. the batchnorm and
          dropout layers are activated in train mode, but disabled in
          evaluation mode).
        > Set the training loss to 0 for the start of the epoch. This is used
          to track the loss of the model on the training data over subsequent
          epochs. The loss should decrease with each epoch if training is
          successful.
        > Store the time at the start of the epoch, this is used to calculate
          the time taken for the epoch to be completed.
        > As per the BERT authors' recommendations, the training data for each
          epoch is split into batches. Loop through the training process for
          each batch.

        For each batch:

        > Move the token IDs, attention masks, and labels to the GPU if
          available for faster processing, otherwise these will be kept on the
          CPU.
        > Invoke the zero_grad method to reset the calculated gradients from
          the previous iteration of this loop.
        > Pass the batch to the model to calculate the logits (predictions
          based on the current classifier weights and biases) as well as the
          loss.
        > Increment the total loss for the epoch. The loss is returned from the
          model as a PyTorch tensor so extract the float value using the item
          method.
        > Perform a backward pass of the model and propagate the loss through
          the classifier head. This will allow the model to determine what
          adjustments to make to the weights and biases to improve its
          performance on the batch.
        > Clip the gradients to be no larger than 1.0 so the model does not
          suffer from the exploding gradients problem.
        > Call the optimizer to take a step in the direction of the error
          surface as determined by the backward pass.

        After training on each batch:

        > Calculate the average loss and time taken for training on the epoch.

        Validation step for the epoch:

        > Switch the model to evaluation mode.
        > Set the validation loss to 0. This is used to track the loss of the
          model on the validation data over subsequent epochs. The loss should
          decrease with each epoch if training was successful.
        > Store the time at the start of the validation, this is used to
          calculate the time taken for the validation for this epoch to be
          completed.
        > Split the validation data into batches.

        For each batch:

        > Move the token IDs, attention masks, and labels to the GPU if
          available for faster processing, otherwise these will be kept on the
          CPU.
        > Invoke the no_grad method to instruct the model not to calculate the
          gradients since we wil not be performing any optimization steps here,
          only inference.
        > Pass the batch to the model to calculate the logits (predictions
          based on the current classifier weights and biases) as well as the
          loss.
        > Extract the logits and labels from the model and move them to the CPU
          (if they are not already there).
        > Increment the loss and calculate the accuracy based on the true
          labels in the validation dataloader.
        > Calculate the average loss and accuracy, and add these to the loss
          dictionary.
        """

        loss_dict = {
            'epoch': [i + 1 for i in range(self.epochs)],
            'average training loss': [],
            'average validation loss': []
        }

        t0_train = datetime.now()

        for epoch in range(0, self.epochs):

            # Train step
            self.model.train()
            training_loss = 0
            t0_epoch = datetime.now()

            print(f'{"-" * 20} Epoch {epoch + 1} {"-" * 20}')
            print('\nTraining:\n---------')
            print(f'Start Time:       {t0_epoch}')

            for batch in self.train_dataloader:
                batch_token_ids = batch[0].to(self.device)
                batch_attention_mask = batch[1].to(self.device)
                batch_labels = batch[2].to(self.device)

                self.model.zero_grad()

                loss, logits = self.model(
                    batch_token_ids,
                    token_type_ids=None,
                    attention_mask=batch_attention_mask,
                    labels=batch_labels,
                    return_dict=False)

                training_loss += loss.item()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.scheduler.step()

            average_train_loss = training_loss / len(self.train_dataloader)
            time_epoch = datetime.now() - t0_epoch

            print(f'Average Loss:     {average_train_loss}')
            print(f'Time Taken:       {time_epoch}')

            # Validation step
            self.model.eval()
            val_loss = 0
            val_accuracy = 0
            t0_val = datetime.now()

            print('\nValidation:\n---------')
            print(f'Start Time:       {t0_val}')

            for batch in self.val_dataloader:
                batch_token_ids = batch[0].to(self.device)
                batch_attention_mask = batch[1].to(self.device)
                batch_labels = batch[2].to(self.device)

                with torch.no_grad():
                    (loss, logits) = self.model(
                        batch_token_ids,
                        attention_mask=batch_attention_mask,
                        labels=batch_labels,
                        token_type_ids=None,
                        return_dict=False)

                logits = logits.detach().cpu().numpy()
                label_ids = batch_labels.to('cpu').numpy()
                val_loss += loss.item()
                val_accuracy += self.calculate_accuracy(logits, label_ids)

            average_val_accuracy = val_accuracy / len(self.val_dataloader)
            average_val_loss = val_loss / len(self.val_dataloader)
            time_val = datetime.now() - t0_val

            print(f'Average Loss:     {average_val_loss}')
            print(f'Average Accuracy: {average_val_accuracy}')
            print(f'Time Taken:       {time_val}\n')

            loss_dict['average training loss'].append(average_train_loss)
            loss_dict['average validation loss'].append(average_val_loss)

        print(f'Total training time: {datetime.now() - t0_train}')

    def calculate_accuracy(self, preds, labels):
        """ Calculate the accuracy of model predictions against true labels.

        Parameters:
            preds (np.array): The predicted label from the model
            labels (np.array): The true label

        Returns:
            accuracy (float): The accuracy as a percentage of the correct
                predictions.
        """
        pred_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        accuracy = np.sum(pred_flat == labels_flat) / len(labels_flat)

        return accuracy

    def predict(self, dataloader):
        """Return the predicted probabilities of each class for input text.

        Parameters:
            dataloader (torch.utils.data.DataLoader): A DataLoader containing
                the token IDs and attention masks for the text to perform
                inference on.

        Returns:
            probs (PyTorch.Tensor): A tensor containing the probability values
                for each class as predicted by the model.

        """

        self.model.eval()
        all_logits = []

        for batch in dataloader:
            batch_token_ids, batch_attention_mask = tuple(t.to(self.device) \
                                                          for t in batch)[:2]

            with torch.no_grad():
                outputs = self.model(batch_token_ids, batch_attention_mask)
                logits = outputs.logits.detach().cpu()

            all_logits.append(logits)

        all_logits = torch.cat(all_logits, dim=0)

        probs = F.softmax(all_logits, dim=1).cpu().numpy()
        return probs