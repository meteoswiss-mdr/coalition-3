## Add paths to dependencies in PYTHONPATH:
import sys
import os
import configparser

## Read paths from config file:
file_path = os.path.realpath(__file__)
cfg_path = os.path.abspath(
                os.path.join(file_path,
                             u"../../config/dependency_paths.cfg"))
config = configparser.RawConfigParser()
config.read(cfg_path)

## Adding pySTEPS path:
sys.path.append(config["dependency_paths"]["pysteps_path"])

## Adding metranet path:
sys.path.append(config["dependency_paths"]["metranet_path"])

## Adding mpop.satin path:
sys.path.append(config["dependency_paths"]["mpop_satin_path"])
