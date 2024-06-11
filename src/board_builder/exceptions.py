class BoardBuilderException(Exception):
    message: str

    def __init__(self, message: str, *args, **kwargs):
        self.message = message
