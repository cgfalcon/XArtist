from yaml import load
from yaml import Loader
import os

# This is the environment key name in System properties
env_key_property = 'project_env'

# All the configs in yaml file will be loaded into this object
# Example of using ProjectConfig
#  - ProjectConfig['db_config']['username']
ProjectConfig = {}

def _reload_configs():
    # Project Environment Parameter. This Parameter should be provided from the system environment
    env = os.getenv(env_key_property, 'dev')

    # Project config file pattern
    project_config_filename = f'{env}_env.yml'

    # Resolve current directory
    current_directory = os.path.dirname(os.path.abspath(__file__))

    # Find the config file's path
    configfile_full_path = os.path.join(current_directory, '..', 'configs', project_config_filename)
    print(f'Ready to load config file: {configfile_full_path}')

    # Load config
    global ProjectConfig
    with open(configfile_full_path, 'r') as config_file:
        ProjectConfig = load(config_file, Loader)

    return ProjectConfig


# Load config
_reload_configs()
print(f'Project Configuration Inited As =>: {ProjectConfig}')


