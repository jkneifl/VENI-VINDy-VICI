import abc


class BaseLibrary(abc.ABC):
    """
    Abstract class for feature libraries
    """

    def __init__(self):
        pass

    @abc.abstractmethod
    def __call__(self, x):
        """
        Construct features for the input x.

        Parameters
        ----------
        x : array-like
            Input data.

        Returns
        -------
        array-like
            Feature representation.
        """
        pass

    @abc.abstractmethod
    def get_names(self, x):
        """
        Construct the names of the features for the input x.

        Parameters
        ----------
        x : array-like
            Input data.

        Returns
        -------
        list of str
            Feature names.
        """
        pass
