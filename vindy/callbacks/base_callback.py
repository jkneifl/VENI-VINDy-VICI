class Callback:
    """Base class for training callbacks."""

    def set_model(self, model):
        self.model = model

    def on_epoch_begin(self, epoch, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        pass

    def on_train_begin(self, logs=None):
        pass

    def on_train_end(self, logs=None):
        pass


class CallbackList:
    """Container for managing a list of callbacks."""

    def __init__(self, callbacks=None):
        self.callbacks = callbacks or []

    def set_model(self, model):
        for cb in self.callbacks:
            cb.set_model(model)

    def on_epoch_begin(self, epoch, logs=None):
        for cb in self.callbacks:
            cb.on_epoch_begin(epoch, logs)

    def on_epoch_end(self, epoch, logs=None):
        for cb in self.callbacks:
            cb.on_epoch_end(epoch, logs)

    def on_train_begin(self, logs=None):
        for cb in self.callbacks:
            cb.on_train_begin(logs)

    def on_train_end(self, logs=None):
        for cb in self.callbacks:
            cb.on_train_end(logs)
