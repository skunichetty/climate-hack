import torch
from torch.utils.data import Dataset, DataLoader
import infra
from collections.abc import Callable
from typing import List
from tqdm import tqdm


class Metrics:
    """
    Stores metric computation functions
    """

    def __init__(self, *metrics: Callable):
        """ """
        self.metrics = metrics

    def evaluate(self, y_true, y_pred, y_score):
        if self.metrics is not None:
            scores = map(lambda x: x(y_true, y_pred, y_score), self.metrics)
            return [score for score in scores]
        return []

    def __call__(self, y_true, y_pred, y_score):
        return self.evaluate(y_true, y_pred, y_score)


class Evaluator:
    """
    A wrapper around training tools to easily train and validate model performance
    """

    def __init__(
        self,
        model: torch.nn.Module,
        loss_fn: Callable,
        optimizer: torch.optim.Optimizer,
        predictor: Callable,
        metrics: Metrics,
    ):
        """
        `Evaluator` constructor. Note that all objects passed must already be initialized and configured
        as per user specification. The `Evaluator` class is only responsible for organizing the behavior
        of these objects.

        Args:
            model (torch.nn.Module): Model to train. Any subclass of `torch.nn.Module` is accepted.
            loss_fn (Callable): Loss function to evaluate model performance.
            optimizer (torch.optim.Optimizer): Optimization algorithm to use. This instance must already be initialized with the model's parameters and all configurations.
            predictor (Callable): Transformation from raw logits to labels. Defaults to None (as used in regression tasks).
            metrics (Metrics): Set of metrics to evaluate model with.
        """
        self.model = model
        self.loss_fn = loss_fn
        self.metrics = metrics
        self.optimizer = optimizer
        self.predictor = predictor
        self.train_loss, self.val_loss = [], []
        self.train_metrics, self.val_metrics = [], []

    def _evaluate(self, X: torch.tensor, y: torch.tensor):
        """
        Evaluates the model on inputs and stores statistics comoputed

        Args:
            X (torch.tensor): Input tensor
            y (torch.tensor): Output tensor - labels for classification or regression values

        Returns:
            tuple: Tuple of (prediction, class score (for classification), loss value)
        """
        output = self.model(X)
        predicted = self.predictor(output)
        score = torch.nn.functional.softmax(output.data, dim=1)
        l = self.loss_fn(output, y)
        return predicted, score, l

    def evaluate(
        self,
        X: torch.tensor,
        y: torch.tensor,
        train: bool = False,
    ):
        """
        Evaluates the model on inputs and stores statistics comoputed

        Args:
            X (torch.tensor): Input tensor
            y (torch.tensor): Output tensor - labels for classification or regression values
            train (bool): If false, disables computational graph generation. Defaults to False.

        Returns:
            tuple: Tuple of (prediction, class score (for classification), loss value)
        """
        if predictor is None:
            predictor = lambda x: x
        if not train:
            self.model.eval()
            with torch.no_grad():
                predicted, score, l = self._evaluate(X, y, self.predictor)
            self.model.train()
        else:
            predicted, score, l = self._evaluate(X, y, self.predictor)
        return predicted, score, l

    def _fbkpass(
        self,
        data: DataLoader,
        loss_history: List,
        metric_history: List,
        pass_function: Callable,
        verbose=True,
    ):
        """
        Performs one forward-backward pass combo on input data

        Args:
            data (DataLoader): Data to propagate through the network.
            loss_history (List): Storage for recorded loss vlaues
            metric_history (List): Storage for recorded metrics
            pass_function (Callable): Function that performs the forward/backwards pass logic
            verbose (bool, optional): Prints more training information. Defaults to True.

        Returns:
            _type_: _description_
        """
        if verbose:
            data = tqdm(data)
        for X, y in data:
            predicted, score, l = pass_function(X, y)
            loss_history.append(l.item())
            metric_history.append(self.metrics(predicted, y, score))

        return loss_history[-1], metric_history[-1]

    def train_epoch(self, train_set: DataLoader, verbose=True):
        """
        Trains the model on one epoch of the training set.

        Args:
            train_set (DataLoader): Training dataset.
            verbose (bool, optional): Prints more training information. Defaults to True.
        """

        def train_pass(X, y):
            pred, score, l = self._evaluate(X, y)

            self.optimizer.zero_grad()
            l.backward()
            self.optimizer.step()

            return pred, score, l

        return self._fbkpass(
            train_set, self.train_loss, self.train_metrics, train_pass, verbose
        )

    def validate(self, validation_set: DataLoader, verbose=False):
        """
        Validates the model on the validation set.

        Args:
            validation_set (DataLoader): Validation dataset.
            verbose (bool, optional): Prints more validation information. Defaults to True.
        """
        self.model.eval()
        with torch.no_grad():
            val_metrics, val_loss = self._fbkpass(
                validation_set, self.val_loss, self.val_metrics, self._evaluate, verbose
            )

        self.model.train()
        return val_metrics, val_loss


def train(
    model: infra.model_base.ModelBase,
    loss_fn,
    optimizer: torch.optim.Optimizer,
    train: DataLoader,
    validation: DataLoader,
    metrics: Metrics,
    num_epochs,
    verbose=True,
):
    predictor = lambda x: torch.argmax(x, dim=1)
    evaluator = Evaluator(model, loss_fn, optimizer, predictor, metrics)
    for epoch in range(num_epochs):
        if verbose:
            print("═" * 10 + f" Epoch {epoch+1} " + "═" * 10)
            print("Training pass: ", end="")
            train_loss, _ = evaluator.train_epoch(train, verbose)
            print("Validation pass: ", end="")
            val_loss, _ = evaluator.validate(validation, verbose)
        if verbose:
            print(f"Train Loss: {train_loss}, Val Loss: {val_loss}")
    return (
        evaluator.train_loss,
        evaluator.train_metrics,
        evaluator.val_loss,
        evaluator.val_metrics,
    )
