import json
import sys
import os
import zipfile
import shutil
import string
import random
try:
    import urllib.request as urlrequest
except ImportError:
    import urllib as urlrequest


def main():
    import argparse

    arg_parser = argparse.ArgumentParser(description=__doc__)
    arg_parser.add_argument('output_dir', metavar='OUTPUT_DIR', type=str, nargs='?', default='./')

    results = arg_parser.parse_args()

    if os.path.exists(results.output_dir + '/z3'):
        exit('z3 folder exists in ' + os.path.abspath(results.output_dir))

    # detect os name
    system = sys.platform
    library_files = ['.txt', '.py']
    if 'linux' in system:
        system = 'ubuntu-14.04'
        library_files.append('libz3.so')
    elif 'darwin' in system:
        system = 'osx'
        library_files.append('libz3.dylib')
    elif 'windows' in system:
        system = 'win'
        library_files.append('Microsoft.Z3.dll')
        library_files.append('Microsoft.Z3.xml')

    # read from GitHub apis
    releases = json.loads(urlrequest.urlopen('https://api.github.com/repos/Z3Prover/z3/releases')
                          .read().decode('utf-8'))[0]

    for asset in releases['assets']:
        if system in asset['name'] and 'x64' in asset['name']:
            # find a random non-conflict tmp folder name
            tmp_folder = './tmp'
            while os.path.exists(tmp_folder):
                tmp_folder = './' + ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(5))

            # download the release zip file
            os.mkdir(tmp_folder)
            os.mkdir(tmp_folder + '/z3')

            def progress(count, block_size, total_size):
                percent = (count * block_size) / total_size
                progress_count = int(percent * 30)
                dot_count = 30 - progress_count - 1
                sys.stdout.write('Downloading %s ' % releases['name'])
                sys.stdout.write('[' + progress_count * '=' + '>' +
                                 dot_count * '.' + '] ' + '{: >3d}'.format(int(percent * 100)) + '%\r')
                sys.stdout.flush()

            urlrequest.urlretrieve(asset['browser_download_url'], tmp_folder + '/z3.zip', progress)
            print('Downloading %s...' % releases['name'] + '[' + 29 * '=' + '>] 100%')

            print('Download finished. Extracting files...')
            zf = zipfile.ZipFile(tmp_folder + '/z3.zip')
            zf.extractall(tmp_folder)

            for file in zf.namelist():
                # move the license file, python bindings and corresponding library
                for library_file in library_files:
                    if file.endswith(library_file):
                        shutil.move(tmp_folder + '/' + file, tmp_folder + '/z3/' + file[file.rfind('/'):])

            print('Extract finished, Removing temporary files...')
            shutil.move(tmp_folder + '/z3', results.output_dir)
            shutil.rmtree(tmp_folder)
            print('Done.')
            break


if __name__ == '__main__':
    main()
