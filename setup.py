import setuptools


with open('README.md') as f:
    README = f.read()

setuptools.setup(
    author='Fedor Indukaev',
    author_email='indukaev@mail.ru',
    name='supervenn',
    license='MIT',
    description='supervenn is a tool for visualization of relations of many sets using matplotlib',
    version='0.1.0',
    long_description=README,
    url='https://github.com/gecko984/supervenn',
    packages=setuptools.find_packages(),
    install_requires=['numpy', 'matplotlib>=3.0.3'],
    classifiers = [
        'Development Status :: 3 — Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Topic :: Scientific/Engineering :: Visualization',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers'
    ]
)
