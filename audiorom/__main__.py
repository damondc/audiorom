"""audiorom <system> <input audio> <output rom> [options]"""
import importlib, sys

SYSTEMS = {
    'sms':     ('audiorom.sms',     'Sega Master System (.sms)'),
    'genesis': ('audiorom.genesis', 'Sega Genesis / Mega Drive (.md)'),
    'nes':     ('audiorom.nes',     'Nintendo NES (.nes)'),
    'snes':    (None,               'Super Nintendo (.sfc) - not implemented yet'),
}


def usage():
    sys.stderr.write('usage: audiorom <system> <input audio> <output rom> [options]\n\nsystems:\n')
    for k, (_, d) in SYSTEMS.items():
        sys.stderr.write('  %-8s %s\n' % (k, d))
    sys.stderr.write('\nrun "audiorom <system> -h" for that system\'s options\n')
    sys.exit(2)


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ('-h', '--help') or sys.argv[1] not in SYSTEMS:
        usage()
    name = sys.argv[1]
    mod, desc = SYSTEMS[name]
    if mod is None:
        sys.exit('audiorom: %s' % desc)
    sys.argv = ['audiorom %s' % name] + sys.argv[2:]
    importlib.import_module(mod).main()


if __name__ == '__main__':
    main()
