import subprocess
import shlex

# Use Fluidsynth software to convert MIDI in .wav using Soundfont

def midi_2_wav(infile, ofile=None, verbose=True):
    """
    Converts a MIDI file to .wav format using Fluidsynth
    """
    if ofile is None:
        ofile = infile + '.wav'
    if ofile[-4:] != '.wav':
        ofile = ofile + '.wav'

    # sf_path_rel = 'Steinway Grand-SF4U-v1.sf2'
    sf_path_rel = 'font.sf2'

    command = 'fluidsynth -F ' + shlex.quote(ofile) + ' -T wav -g 1 -R 1 ' + shlex.quote(sf_path_rel) + ' ' + shlex.quote(infile)

    p = subprocess.Popen(command, universal_newlines=True, shell=True,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    while p.poll() is None:
        l = p.stdout.readline()  # This blocks until it receives a newline.
        if verbose:
            print(l.rstrip('\n'))

    # When the subprocess terminates there might be unconsumed output
    # that still needs to be processed.
    if verbose:
        print(p.stdout.read())
