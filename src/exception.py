import sys
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def error_message_detail(error, error_detail):
    """
    Returns a detailed error message including file name, line number, and error message.
    """
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = (
        f"Error occurred in Python script named [{file_name}] "
        f"line number [{exc_tb.tb_lineno}] error message [{error}]"
    )
    return error_message

class CustomException(Exception):
    def __init__(self, error, error_detail):
        super().__init__(error_message_detail(error, error_detail))
        self.error_message = error_message_detail(error, error_detail)
    
    def __str__(self):
        return self.error_message