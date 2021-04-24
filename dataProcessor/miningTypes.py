from enum import Enum

class MiningTypes(Enum):
	HARD_NEGATIVE = 0
	HARD_POSITIVE = 1
	RANDOM_HARD_NEGATIVE = 2
	RANDOM_HARD_POSITIVE = 3

	def getType(mining_type):
		if mining_type == 'HARD_NEGATIVE':
			return MiningTypes.HARD_NEGATIVE
		if mining_type == 'HARD_POSITIVE':
			return MiningTypes.HARD_POSITIVE
		if mining_type == 'RANDOM_HARD_NEGATIVE':
			return MiningTypes.RANDOM_HARD_NEGATIVE
		if mining_type == 'RANDOM_HARD_POSITIVE':
			return MiningTypes.RANDOM_HARD_POSITIVE