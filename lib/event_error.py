class EventError(Exception):
    """Exception raised for erros during event operation.

    Attributes:
        message -- explanation of the error
    """

    def __init__(self, message: str) -> None:
        self.message = (message,)
        super().__init__(self.message)


class EventGoalFailed(EventError):
    pass
