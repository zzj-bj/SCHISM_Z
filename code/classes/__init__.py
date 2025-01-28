def str2bool(str_val):
	if str_val.lower() == "false":
		return False
	elif str_val.lower() == "true":
		return True
	else:
		raise ValueError("Cannot convert string to bool, please check type mispelling in .ini file for bool parameters.")