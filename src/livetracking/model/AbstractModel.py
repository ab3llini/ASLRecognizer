import abc


class AbstractModel:
    """
    AbstractModel is a class that simply defines the method that models that will be used by the livetracker should
    implement. Such model classes should serve as wrappers to keep the livetracker code clean and independent
    from the model it uses.
    """
    def __init__(self, model):
        self.model = model

    @abc.abstractmethod
    def predict(self, x):
        pass
