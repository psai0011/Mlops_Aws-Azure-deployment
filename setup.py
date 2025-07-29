from setuptools import setup, find_packages
from typing import List


def get_requirements(file_path: str) -> List[str]:
    """
    This function will return the list of requirements
    mentioned in the requirements.txt file.
    """
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace('\n', '') for req in requirements]
        if '-e .' in requirements:
            requirements.remove('-e .')
    return requirements

setup(
    name='Ml_Project_AWS_AZURE_Deployment',
    version='0.0.1',
    author='Saikiran',
    author_email="panthaganisai123@gmail.com",
    packages=find_packages(),
    install_requires = get_requirements('requirements.txt'),
    
    description='Machine Learning Project for AWS and Azure Deployment',

) 