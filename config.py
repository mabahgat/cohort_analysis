from typing import Iterable

import yaml
import pathlib


def load_config(config_path=None):
	if config_path is None:
		root_path = pathlib.Path(__file__).parent.absolute()
		config_path = "{}/config.yaml".format(root_path)
	with open(config_path, 'r') as yaml_file:
		config_dict = yaml.safe_load(yaml_file)
		return Config(config_dict)


class Config:

	def __init__(self, config_dict):
		self._config_dict = config_dict

	def __getattribute__(self, key):
		if key == '_config_dict' or key not in self._config_dict:
			return super().__getattribute__(key)
		value = self._config_dict[key]
		if isinstance(value, dict):
			return Config(value)
		elif isinstance(value, list):
			return [Config(item) if isinstance(item, dict) else item for item in value]
		else:
			return value

	def __getitem__(self, key):
		return self.__getattribute__(key)

	def __setattr__(self, key, value):
		if key == '_config_dict':
			return super().__setattr__(key, value)
		self._config_dict[key] = value

	def __setitem__(self, key, value):
		return self.__setattr__(key, value)

	def __contains__(self, item):
		return item in self._config_dict

	def to_dict(self):
		return self._config_dict.copy()

	def to_primitives_dict(self):
		def to_primitive(value):
			if type(value) in (int, float, str, bool, None):
				return value
			elif type(value) is dict:
				return {k: to_primitive(v) for k, v in value.items()}
			elif isinstance(value, Iterable):
				return type(value)([to_primitive(i) for i in value])
			else:
				return str(value)
		p_dict = self.to_dict()
		return to_primitive(p_dict)


global_config = load_config()
