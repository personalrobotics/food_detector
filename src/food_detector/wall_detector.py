from enum import Enum

class WallClass(Enum):
	# Ethan give these proper names  
	ISOLATED = 1
	NEXT_TO_WALL = 2
	ON_TOP_OF_LEAF = 3


# Pseudocode for API
class WallDetector(object):

	# API
	def detect_wall(self, uvpoint, camera_tf):
		return WallClass.ISOLATED


