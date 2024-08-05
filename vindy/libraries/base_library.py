import abc


class BaseLibrary(abc.ABC):
    """
    Abstract class for feature libraries.
    """

    def __init__(self):
        pass

    @abc.abstractmethod
    def __call__(self, x):
        """
        Construct features for the input x.

        Parameters
        ----------
        x : any
            Input data.

        Returns
        -------
        any
            Constructed features.
        """
        pass

    @abc.abstractmethod
    def get_names(self, x):
        """
        Construct the names of the features for the input x.

        Parameters
        ----------
        x : any
            Input data.

        Returns
        -------
        list of str
            Names of the features.
        """
        pass