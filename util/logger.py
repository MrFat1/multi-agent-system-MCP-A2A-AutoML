import logging

class Logger:
    def __init__(self, project_name, logs_pathname):
        '''
        Initializes the Logger with project name and log file path.

        Args:
            project_name (str): The name of the project that will be displayed
            logs_pathname (str): Where the log files will be stored
        '''

        self.project_name = project_name
        self.logs_pathname = logs_pathname
        self.logger = None

    def launch_logging(self):
        '''
        Initializes and launches the logger with the given name
        This class configures how console and file logs will be printed / saved (wich format)

        Returns:
            logging.Logger: Configured logger instance
        '''

        # Define the logger's name and logging level.
        self.logger = logging.getLogger(self.project_name)
        self.logger.setLevel(logging.DEBUG)

        # Console handler, for debug messages inside code
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)

        # File handler, for logging file creation and storage
        file_handler = logging.FileHandler(self.logs_pathname)
        file_handler.setLevel(logging.INFO)

        # Create formatters and add them to the handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)

        # Add the handlers to the logger (avoiding duplicate handlers)
        if not self.logger.handlers:  # Prevent adding handlers multiple times
            self.logger.addHandler(console_handler)
            self.logger.addHandler(file_handler)

        return self.logger