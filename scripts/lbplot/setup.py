from distutils.core import setup

setup(
    name='lbplot',
    version='1.0',
    description='LBANN output plotting utilities.',
    author='Luke Jaffe',
    author_email='lukejaffe@users.noreply.github.com',
    packages=['lbplot'],
    package_dir={'lbplot': 'src'},
    scripts=['src/script/lbplot']
 )
