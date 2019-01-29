## Add paths to dependencies in PYTHONPATH:
import sys
import os
import configparser

## Read paths from config file:
config = configparser.RawConfigParser()
config.read(os.path.abspath(u"../config/dependency_paths.cfg")

## Adding pySTEPS path:
sys.path.append(config["dependency_paths"]["pysteps_path"])

## Adding metranet path:
sys.path.append(config["dependency_paths"]["metranet_path"])

## Adding mpop.satin path:
sys.path.append(config["dependency_paths"]["mpop_satin_path"])
