from __future__ import absolute_import
import logging
from .sim_food_detector import SimFoodDetector

try:
    from .spnet_detector import SPNetDetector
except ImportError as e:
    logging.warn("Failed to import SPNetDetector. This is okay for simulation code. Error: %s" % e)

try:
    from .retinanet_detector import RetinaNetDetector
except ImportError as e:
    logging.warn("Failed to import RetinaNetDetector. This is okay for simulation code. Error: %s" % e)

try:
    from .spanet_detector import SPANetDetector
except ImportError as e:
    logging.warn("Failed to import SPANetDetector. This is okay for simulation code. Error: %s" % e)
