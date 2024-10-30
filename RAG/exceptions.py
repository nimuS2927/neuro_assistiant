class InvalidFileFormatError(Exception):
    def __init__(
        self, message="Invalid file format", file_path=None, expected_formats=None
    ):
        self.file_path = file_path
        self.expected_formats = expected_formats
        super().__init__(
            f"{message}. File: {file_path}, Expected formats: {expected_formats}"
        )
