# audiorom

Turn an audio file into a ROM for a retro console that plays it back, with an
on-screen audio-player UI. No hardware DAC required: each system streams PCM
through whatever the chip offers.

| system    | output | status | how it makes sound |
|-----------|--------|--------|--------------------|
| `sms`     | `.sms` | working | SN76489 PSG volume registers used as a DAC (noise-shaped, cycle-counted Z80) |
| `genesis` | `.md`  | working | Z80 streams 8-bit PCM to the YM2612 DAC; 68000 runs the UI |
| `nes`     | `.nes` | working | CPU writes the DMC level register ($4011) every N cycles from a cycle-counted 6502 loop; UxROM-style banking, up to 4 MB |
| `snes`    | `.sfc` | working | BRR (4-bit ADPCM) stream played by the S-DSP; the 65816 feeds an SPC700 ring buffer over the I/O ports; HiROM, up to 4 MB |

## Usage

Requires Python 3 and `ffmpeg` on `PATH`. No other dependencies.

    bin/audiorom sms     song.mp3 song.sms     [--rate 8000] [--title NAME]
    bin/audiorom genesis song.mp3 song.md      [--rate 22050] [--stereo] [--autoplay]
    bin/audiorom nes     song.mp3 song.nes     [--rate 16000] [--autoplay] [--no-vis]
    bin/audiorom snes    song.mp3 song.sfc     [--rate 32000] [--autoplay]

`--rate 0` (default) picks the highest sample rate that fits the cartridge.
`--gain` sets loudness boost in dB before a peak limiter. Run
`bin/audiorom <system> -h` for all options.

### Player controls

| action | Genesis | NES / SNES |
|--------|---------|------------|
| play / pause | START | START |
| stop and rewind | A | B |
| toggle loop | B | A |
| toggle LED-bar visualizer | C | SELECT |

## Notes

* The assemblers (Z80, 68000, 6502, 65816, SPC700) and the BRR encoder are built in; there is no external toolchain.
* NES: 7-bit output, sample rate capped at ~16 kHz by the per-sample cycle budget; output is
  linearised against the APU mixer curve.
* SNES: 32 kHz mono BRR at best (about 18 KB/s); `--rate` picks 128000/N Hz. Songs up to
  about 3.5 minutes fit a 4 MB cartridge at 32 kHz; longer songs automatically use a lower rate.
* SMS: `--att-min N` limits the loudest PSG level used (higher is quieter).
* Don't commit audio you don't have the rights to; generated ROMs contain it.

## License

MIT, see `LICENSE`. The copyright and license notice must be kept in copies and
substantial portions of the code.

### Credit

If you use audiorom in a project, video or write-up, a mention of audiorom or
its author, in whatever form suits you, is appreciated. Link to
https://github.com/damondc/audiorom if you can. This is a request, not a
license condition.

ROMs you generate contain the audio you feed in, so their rights are those of
that audio, not of this tool.
