import logging
import os

AGENT_LOGGER_NAME = "agents"

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
    
_loggers: dict = {}  # cache para evitar duplicados

def get_logger(name: str, log_file: str = "outputs/logs/ml_pipeline.log") -> logging.Logger:
    """
    Devuelve un logger configurado. Si ya existe uno con ese nombre, lo reutiliza.
    Crea el directorio de logs si no existe.
    """
    if name in _loggers:
        return _loggers[name]
    
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger = Logger(name, log_file).launch_logging()
    _loggers[name] = logger
    return logger

def get_agent_logger(log_file: str = "outputs/logs/agents_v3.log") -> logging.Logger:
    """
    Logger dedicado a las respuestas y razonamientos de los agentes.
    Escribe en agents.log (además de consola) para separar la traza
    del LLM de los logs de infraestructura.
    """
    if AGENT_LOGGER_NAME in _loggers:
        return _loggers[AGENT_LOGGER_NAME]

    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    agent_logger = logging.getLogger(AGENT_LOGGER_NAME)
    agent_logger.setLevel(logging.DEBUG)

    # Solo fichero — las respuestas de los agentes no necesitan salir por consola
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s\n%(message)s\n' + '-' * 80
    )
    file_handler.setFormatter(formatter)

    if not agent_logger.handlers:
        agent_logger.addHandler(file_handler)

    # Evitar que suba al root logger (que lo mandaría también a consola)
    agent_logger.propagate = False

    _loggers[AGENT_LOGGER_NAME] = agent_logger
    return agent_logger