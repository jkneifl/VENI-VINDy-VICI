import tensorflow as tf
import webbrowser


class SpotifyCallback(tf.keras.callbacks.Callback):

    def __init__(
        self,
        song_link="https://open.spotify.com/track/3cHyrEgdyYRjgJKSOiOtcS?si=3281ab97ddde4286",
        n_epochs=20,
    ):

        self.song_link = song_link
        self.loss = 0
        self.n_epochs = n_epochs
        self.epoch_counter = 0
        self.finished = False
        # super init
        super(SpotifyCallback, self).__init__()

    def on_epoch_end(self, epoch, logs=None):

        if not self.finished:
            current_loss = logs.get("loss")

            if current_loss < self.loss:
                self.loss = current_loss
                self.epoch_counter += 1
            else:
                self.loss = current_loss
                self.epoch_counter = 0

            if self.epoch_counter >= self.n_epochs:
                webbrowser.open(self.song_link, new=2)
                self.finished = True
