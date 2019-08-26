from setuptools import setup

setup(
    name="DataFlame",
    version="1.0.0-beta",
    description="DataFlame: Machine-Learning Oriented Dataframe Manipulation Interface for Pandas Dataframes",
    url="https://github.com/shayanfazeli/dataflame",
    author="Shayan Fazeli",
    author_email="shayan@cs.ucla.edu",
    license="Apache",
    classifiers=[
          'Intended Audience :: Science/Research',
          'Development Status :: 1 - Beta',
          'License :: Apache Software License',
          'Programming Language :: Python :: 3.6',
          'Topic :: Scientific/Engineering :: Artificial Intelligence'
      ],
    keywords="dataflame machine learning dataframe label interpolation",
    packages=["dataflame"],
    python_requires='>3.6.0',
    install_requires=[
        'numpy',
        'pandas',
        'sklearn'
    ],
    zip_safe=False
)