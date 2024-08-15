from setuptools import setup, find_packages


def get_requirements():
    with open('requirements.txt') as f:
        return f.read().splitlines()


setup(
    name='gym-llm',
    version='1.0.0',
    author="Rodrigo Sánchez Molina",
    author_email="rsanchezm98@gmail.com",
    maintainer="Rodrigo Sánchez Molina",
    maintainer_email="rsanchezm98@gmail.com",
    packages=find_packages(),
    python_requires='>=3.8,<3.11',
    install_requires=get_requirements(),
)