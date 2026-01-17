import logging
import json
from datetime import datetime

logging.basicConfig(level=logging.INFO)

def logger(event_type: str, payload: dict, level: str = "INFO"):
    log_entry = {
        "TIMESTAMP": datetime.utcnow().isoformat(),
        "Event": event_type,
        "Payload": payload
    }
    message=json.dumps(log_entry)
    
    
    level = level.upper()
    if level == "DEBUG":
        logging.debug(message)
    elif level == "WARNING":
        logging.warning(message)
    elif level == "ERROR":
        logging.error(message)
    else:
        logging.info(message)