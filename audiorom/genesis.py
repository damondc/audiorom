#!/usr/bin/env python3
"""genesis - turn an MP3 into a Sega Genesis / Mega Drive audio-player ROM.

The Z80 streams 8-bit unsigned PCM from banked cartridge ROM into the YM2612's
DAC (channel 6) at a cycle-counted rate; the 68000 only sets things up and then
runs the on-screen UI from the vblank interrupt.  Needs ffmpeg; the Z80 and
68000 assemblers are built in.

    audiorom genesis song.mp3 song.md [--rate 22050] [--title NAME]

Controls: START play/pause, A stop+rewind, B loop on/off, C visualizer,
LEFT/RIGHT seek 5 s.
"""
import argparse, array, math, os, re, struct, subprocess, sys, wave

from audiorom import sms                 # Z80 assembler, font, tilemap helpers

Z80_HZ = 3579545                          # NTSC
ROM_MAX = 0x400000                        # 4 MB without a mapper
DATA_OFF = 0x10000                        # sample data starts here (32KB aligned)
BANK = 0x8000                             # Z80 bank window size
BUS_WAIT = 3.3                            # avg Z80 wait states per ROM-bus read
FIXED = 88                                # mono loop cycles excluding pad, see build_z80
YM_SLOT = 1008 / 15.0                     # one YM2612 output sample = 67.2 Z80 cycles
RAM = 0xFF0000
FRAMECNT, RELOAD, TBLPTR, STATE, PREV = RAM, RAM + 2, RAM + 4, RAM + 8, RAM + 9
LOOP, VIS, VISDIV, VISPTR = RAM + 10, RAM + 11, RAM + 12, RAM + 14
PREVLR = RAM + 18                         # left/right edge detection
CMD = 0xA01FFD                            # command byte in Z80 RAM (0 play, 1 pause, 2 stop)
LOOPF = 0xA01FFC                          # loop flag in Z80 RAM
VIS_BARS, VIS_ROWS, VIS_FPS = 8, 8, 30     # 8 bars on every console
VIS_ROW0, VIS_COL0 = 5, 8                 # top tile row / first column of the LED bars
VDP_CTRL, VDP_DATA, PSG = 0xC00004, 0xC00000, 0xC00011

# ---------------------------------------------------------------- 68000 asm

class M68K:
    def __init__(self, org):
        self.org, self.b, self.lab, self.fix = org, bytearray(), {}, []

    @property
    def pc(self):
        return self.org + len(self.b)

    def label(self, n):
        self.lab[n] = self.pc

    def word(self, v):
        self.b += struct.pack('>H', v & 0xFFFF)

    def long(self, v):
        if isinstance(v, str):
            self.fix.append((len(self.b), v, 'l'))
            v = 0
        self.b += struct.pack('>I', v & 0xFFFFFFFF)

    def ea(self, spec, size):
        """-> (mode<<3|reg, [extension items])"""
        m = re.fullmatch(r'd([0-7])', spec)
        if m: return int(m[1]), []
        m = re.fullmatch(r'a([0-7])', spec)
        if m: return 8 | int(m[1]), []
        m = re.fullmatch(r'\(a([0-7])\)', spec)
        if m: return 0x10 | int(m[1]), []
        m = re.fullmatch(r'\(a([0-7])\)\+', spec)
        if m: return 0x18 | int(m[1]), []
        if spec == 'sr': return None, []
        if spec.startswith('#'):
            v = spec[1:]
            v = int(v.replace('$', '0x'), 0) if re.match(r'\$?[0-9A-Fa-fx]+$', v) \
                and not v.isidentifier() else v
            return 0x3C, [('l' if size == 'l' else 'w', v)]
        v = int(spec.replace('$', '0x'), 0) if re.match(r'\$[0-9A-Fa-f]+$|\d+$', spec) else spec
        return 0x39, [('l', v)]

    def emit(self, items):
        for k, v in items:
            self.word(v) if k == 'w' else self.long(v)

    def move(self, size, src, dst):
        sz = {'b': 1, 'w': 3, 'l': 2}[size]
        smr, sext = self.ea(src, size)
        dmr, dext = self.ea(dst, size)
        self.word((sz << 12) | ((dmr & 7) << 9) | ((dmr >> 3) << 6) | smr)
        self.emit(sext); self.emit(dext)

    def lea(self, addr, an):
        _, ext = self.ea(addr, 'l')
        self.word(0x41C0 | (an << 9) | 0x39); self.emit(ext)

    def moveq(self, v, dn):
        self.word(0x7000 | (dn << 9) | (v & 0xFF))

    def op_ea(self, base, spec, size, extra=None):
        mr, ext = self.ea(spec, size)
        self.word(base | mr)
        if extra is not None: self.word(extra)
        self.emit(ext)

    def bcc(self, cc, label):                  # 16-bit displacement
        self.word(0x6000 | (cc << 8))
        self.fix.append((len(self.b), label, 'rel'))
        self.word(0)

    def bra(self, l): self.bcc(0x0, l)
    def bne(self, l): self.bcc(0x6, l)
    def beq(self, l): self.bcc(0x7, l)

    def dbra(self, dn, label):
        self.word(0x51C8 | dn)
        self.fix.append((len(self.b), label, 'rel'))
        self.word(0)

    def finish(self):
        for off, name, kind in self.fix:
            t = self.lab[name]
            if kind == 'l':
                self.b[off:off + 4] = struct.pack('>I', t)
            else:
                self.b[off:off + 2] = struct.pack('>h', t - (self.org + off))
        return bytes(self.b)

# ------------------------------------------------------------------ Z80 side

def build_z80(start_bank, last_bank, pad):
    """Sample bytes are 1..254; a 0x00 byte marks the end of the song, so the
    loop restart (or stop) happens exactly at the last sample."""
    a = sms.Asm(0)
    a.db(0xF3)                                 # di
    a.db(0x31, 0xF0, 0x1F)                     # ld sp,$1FF0
    a.db(0x21, 0x00, 0x40)                     # ld hl,$4000   YM2612 part 1
    a.db(0x36, 0x2B, 0x23, 0x36, 0x80)         # reg $2B = $80 (DAC on)
    a.db(0x23, 0x36, 0xB6, 0x23, 0x36, 0xC0)   # part 2 reg $B6 = $C0 (L+R)
    a.db(0x21, 0x00, 0x40, 0x36, 0x2A)         # ld hl,$4000 ; reg $2A latched
    a.db(0x01, 0x00, 0x60)                     # ld bc,$6000   bank register
    a.db(0x3E, start_bank, 0x32, 0xFE, 0x1F)   # bank = start
    a.abs(0xCD, 'setbank')
    a.db(0x21, 0x00, 0x80)                     # ld hl,$8000
    a.abs(0xC3, 'ctl')                         # then follow the command byte
    a.label('loop')
    a.db(0x7E, 0xB7)                           # ld a,(hl) ; or a
    a.jr(0x28, 'endsong')                      # jr z,endsong
    a.db(0x32, 0x01, 0x40, 0x23)               # ld ($4001),a ; inc hl  (DAC)
    a.db(0xCB, 0x7C)                           # bit 7,h
    a.jr(0x28, 'nextbank')                     # jr z,nextbank
    a.db(0x3A, 0xFD, 0x1F, 0xB7)               # ld a,($1FFD) ; or a
    a.jr(0x20, 'ctl')                          # jr nz,ctl (pause / stop)
    emit_pad(a, 'pad', pad[0], pad[1], pad[2])
    a.jr(0x18, 'loop')
    a.label('ctl')                             # paused or stopped: silence
    a.db(0x3E, 0x80, 0x32, 0x01, 0x40)         # DAC = $80
    a.label('wait')
    a.db(0x3A, 0xFD, 0x1F, 0xB7)               # ld a,($1FFD) ; or a
    a.jr(0x28, 'loop')                         # jr z,loop   (play / resume)
    a.db(0xFE, 0x03)                           # cp 3
    a.jr(0x28, 'seek')                         # seek request
    a.db(0xFE, 0x02)                           # cp 2
    a.jr(0x20, 'wait')                         # paused: keep waiting
    a.db(0x3E, start_bank, 0x32, 0xFE, 0x1F)   # stopped: rewind to the start
    a.abs(0xCD, 'setbank')
    a.db(0x21, 0x00, 0x80)                     # ld hl,$8000
    a.jr(0x18, 'wait')
    a.label('seek')                            # bank in $1FFB, command to resume in $1FFA
    a.db(0x3A, 0xFB, 0x1F, 0x32, 0xFE, 0x1F)  # ld a,($1FFB) ; ld ($1FFE),a
    a.abs(0xCD, 'setbank')
    a.db(0x21, 0x00, 0x80)                     # ld hl,$8000
    a.db(0x3A, 0xFA, 0x1F, 0x32, 0xFD, 0x1F)  # ld a,($1FFA) ; ld ($1FFD),a
    a.jr(0x18, 'wait')
    a.label('endsong')                         # end marker reached
    a.db(0x3A, 0xFC, 0x1F, 0xB7)               # ld a,($1FFC) ; or a   loop flag
    a.jr(0x28, 'stopit')
    a.db(0x3E, start_bank, 0x32, 0xFE, 0x1F)   # loop: back to the start
    a.abs(0xCD, 'setbank')
    a.db(0x21, 0x00, 0x80)
    a.abs(0xC3, 'loop')
    a.label('stopit')
    a.db(0x3E, 0x02, 0x32, 0xFD, 0x1F)         # no loop: command = stop
    a.abs(0xC3, 'ctl')
    a.label('nextbank')
    a.db(0x3A, 0xFE, 0x1F, 0x3C)               # ld a,($1FFE) ; inc a
    a.db(0x32, 0xFE, 0x1F)                     # ld ($1FFE),a
    a.abs(0xCD, 'setbank')
    a.db(0x21, 0x00, 0x80)                     # ld hl,$8000
    a.jr(0x18, 'loop')
    a.label('setbank')                         # A = bank number (A15..A22)
    for _ in range(8):
        a.db(0x02, 0x0F)                       # ld (bc),a ; rrca
    a.db(0xAF, 0x02, 0xC9)                     # xor a ; ld (bc),a ; ret (A23=0)
    return a.finish()


def pad_for(x):
    """Cheapest pad of at least x cycles: (n, nops, cp0s, cycles).
    n>0: `ld a,n; dec a; jr nz` loop (16n+2); nops are 4, `cp 0` is 7."""
    x = max(int(math.ceil(x)), 0)
    best = None
    for n in range(0, 40):
        base = 16 * n + 2 if n else 0
        for b in range(0, 4):
            r = x - base - 7 * b
            a_ = max(0, int(math.ceil(r / 4.0)))
            tot = base + 7 * b + 4 * a_
            cand = (tot, n + a_ + b, n, a_, b)
            if best is None or cand < best:
                best = cand
    tot, _, n, a_, b = best
    return n, a_, b, tot


def emit_pad(a, label, n, k, b=0):
    if n:
        a.db(0x3E, n)
        a.label(label)
        a.db(0x3D)
        a.jr(0x20, label)
    a.db(*([0x00] * k))
    for _ in range(b):
        a.db(0xFE, 0x00)                       # cp 0 (7 cycles)


SAMPLES_PER_BANK = 32766 // 3              # 3-byte samples; last 2 bytes of a bank unused


def build_z80_hybrid(start_bank, last_bank, pad, psg_init):
    """Stereo from two different outputs: the YM2612 DAC (L-R, panned hard
    left) plus the PSG (R, which plays in both ears).  Left ear = (L-R)+R = L,
    right ear = R.  Per sample the ROM holds 3 bytes: DAC value, PSG cmd, cmd."""
    a = sms.Asm(0)
    a.db(0xF3)                                 # di
    a.db(0x31, 0xF0, 0x1F)                     # ld sp,$1FF0
    a.db(0x3E, 0x2B, 0x32, 0x00, 0x40)         # YM reg $2B = $80 (DAC on)
    a.db(0x3E, 0x80, 0x32, 0x01, 0x40)
    a.db(0x3E, 0xB6, 0x32, 0x02, 0x40)         # part 2 reg $B6 = $80 (left only)
    a.db(0x3E, 0x80, 0x32, 0x03, 0x40)
    a.db(0x3E, 0x2A, 0x32, 0x00, 0x40)         # latch DAC data register
    for v in (0x80, 0x00, 0xA0, 0x00, 0xC0, 0x00, 0xFF) + tuple(psg_init):
        a.db(0x3E, v, 0x32, 0x11, 0x7F)        # PSG: tone period 0 (DC), noise off, volumes
    a.db(0x3E, start_bank, 0x32, 0xFE, 0x1F)
    a.abs(0xCD, 'setbank')
    a.db(0x21, 0x00, 0x80)                     # ld hl,$8000
    a.abs(0xC3, 'ctl')
    a.label('loop')
    a.db(0x7E, 0x32, 0x01, 0x40, 0x23)         # DAC: ld a,(hl);ld ($4001),a;inc hl
    a.db(0x7E, 0x32, 0x11, 0x7F, 0x23)         # PSG cmd 1
    a.db(0x7E, 0x32, 0x11, 0x7F, 0x23)         # PSG cmd 2
    a.db(0x7C, 0xFE, 0xFF)                     # ld a,h ; cp $FF
    a.jr(0x20, 'notend')
    a.db(0x7D, 0xFE, 0xFE)                     # ld a,l ; cp $FE
    a.jr(0x30, 'nextbank')                     # jr nc
    a.label('notend')
    a.db(0x3A, 0xFD, 0x1F, 0xB7)               # ld a,($1FFD) ; or a
    a.jr(0x20, 'ctl')
    emit_pad(a, 'pad', pad[0], pad[1], pad[2])
    a.abs(0xC3, 'loop')
    a.label('ctl')                             # paused/stopped: DAC to centre
    a.db(0x3E, 0x80, 0x32, 0x01, 0x40)
    a.label('wait')
    a.db(0x3A, 0xFD, 0x1F, 0xB7)
    a.abs(0xCA, 'loop')                        # jp z,loop
    a.db(0xFE, 0x03)
    a.jr(0x28, 'seek')                         # seek request
    a.db(0xFE, 0x02)
    a.jr(0x20, 'wait')
    a.db(0x3E, start_bank, 0x32, 0xFE, 0x1F)   # stopped: rewind
    a.abs(0xCD, 'setbank')
    a.db(0x21, 0x00, 0x80)
    a.jr(0x18, 'wait')
    a.label('seek')
    a.db(0x3A, 0xFB, 0x1F, 0x32, 0xFE, 0x1F)
    a.abs(0xCD, 'setbank')
    a.db(0x21, 0x00, 0x80)
    a.db(0x3A, 0xFA, 0x1F, 0x32, 0xFD, 0x1F)
    a.jr(0x18, 'wait')
    a.label('nextbank')
    a.db(0x3A, 0xFE, 0x1F, 0x3C)
    a.db(0xFE, last_bank + 1)
    a.jr(0x28, 'ended')
    a.db(0x32, 0xFE, 0x1F)
    a.abs(0xCD, 'setbank')
    a.db(0x21, 0x00, 0x80)
    a.abs(0xC3, 'loop')
    a.label('ended')
    a.db(0x3E, 0x02, 0x32, 0xFD, 0x1F)
    a.abs(0xC3, 'ctl')
    a.label('setbank')
    for _ in range(8):
        a.db(0x32, 0x00, 0x60, 0x0F)
    a.db(0xAF, 0x32, 0x00, 0x60, 0xC9)
    return a.finish()


# ------------------------------------------------------------------ UI data

def md_tiles():
    font = sms.load_font()
    t = bytearray()
    def row_bytes(bits, color):
        out = bytearray()
        for i in range(4):
            hi = color if bits & (0x80 >> (2 * i)) else 0
            lo = color if bits & (0x40 >> (2 * i)) else 0
            out.append(hi << 4 | lo)
        return out
    for c in range(32, 96):
        for r in font(c):
            t += row_bytes(r, 1)
    for r in range(8):                         # tile 64: bar full
        t += row_bytes(0xFF, 2) if 1 <= r <= 6 else bytes(4)
    for r in range(8):                         # tile 65: bar empty
        t += row_bytes(0xFF, 3) if r in (3, 4) else bytes(4)
    for color in (2, 4, 5, 6):                 # tiles 66..69: LED green/yellow/red/off
        for r in range(8):
            t += row_bytes(0x7E, color) if 1 <= r <= 6 else bytes(4)
    return bytes(t)


def md_tilemap(title, total, info):
    w = [0] * (32 * 32)

    def put(row, text, col=None):
        c = (32 - len(text)) // 2 if col is None else col
        for i, ch in enumerate(text):
            w[row * 32 + c + i] = sms.ch_tile(ch)
    put(0, 'SEGA GENESIS AUDIO PLAYER')
    put(2, title[:30])
    for i in range(sms.BAR_LEN):
        w[sms.BAR_ROW * 32 + sms.BAR_COL0 + i] = 65
    for i, t in enumerate(sms.mmss_tiles(0)):
        w[sms.TIME_ROW * 32 + sms.TIME_COL + i] = t
    put(sms.TIME_ROW, ' / ', sms.TIME_COL + 5)
    for i, t in enumerate(sms.mmss_tiles(total)):
        w[sms.TIME_ROW * 32 + sms.TIME_COL + 8 + i] = t
    put(3, info)
    put(24, 'START=PLAY/PAUSE')
    put(25, 'LEFT/RIGHT=SEEK 5 SEC')
    put(26, 'A=STOP  B=LOOP  C=VISUAL')
    return struct.pack('>%dH' % len(w), *w)


def _words(text):
    return struct.pack('>8H', *[sms.ch_tile(c) for c in text.ljust(8)[:8]])


def md_labels():
    return b''.join(_words(t) for t in ('PLAYING', 'PAUSED', 'STOPPED')) + bytes(16 * 5)


def md_flag_labels():
    """loop (off, on) then visualizer (off, on), 16 bytes each"""
    return b''.join(_words(t) for t in ('LOOP OFF', 'LOOP ON', 'VIS OFF', 'VIS ON'))


def md_vis_tiles():
    """for each height 0..8: 8 tile words, top row first"""
    out = bytearray()
    for h in range(VIS_ROWS + 2):                      # h=9: blank (visualizer off)
        for row in range(VIS_ROWS):
            if h > VIS_ROWS:
                out += struct.pack('>H', 0)
                continue
            level = VIS_ROWS - row                     # top row = highest level
            if level <= h:
                tile = 66 if level <= 4 else (67 if level <= 6 else 68)
            else:
                tile = 69
            out += struct.pack('>H', tile)
    return bytes(out)


def compute_vis(pcm, rate, secs):
    """VIS_BARS log-spaced band levels (0..8) per 1/30 s, for the LED bars."""
    import operator
    mul = operator.mul
    n = VIS_BARS
    fl, fh = 90.0, min(7000.0, rate * 0.45)
    freqs = [fl * (fh / fl) ** (i / (n - 1.0)) for i in range(n)]
    step = max(1, int(round(rate / 4000.0)))           # decimate to ~4-8 kHz
    r2 = rate / float(step)
    flen = int(r2 / VIS_FPS)
    win = [0.5 - 0.5 * math.cos(2 * math.pi * i / flen) for i in range(flen)]
    cs, ss = [], []
    for f in freqs:
        w = 2 * math.pi * f / r2
        cs.append([win[i] * math.cos(w * i) for i in range(flen)])
        ss.append([win[i] * math.sin(w * i) for i in range(flen)])
    dec = [sum(pcm[i:i + step]) / step for i in range(0, len(pcm) - step + 1, step)]
    nfr = secs * VIS_FPS
    mags = []
    for k in range(nfr):
        seg = dec[k * flen:(k + 1) * flen]
        if len(seg) < flen:
            seg = seg + [0.0] * (flen - len(seg))
        row = []
        for b in range(n):
            re = sum(map(mul, seg, cs[b])); im = sum(map(mul, seg, ss[b]))
            row.append(math.hypot(re, im))
        mags.append(row)
    # per-band reference: 90th percentile, so every bar uses its full range
    refs = []
    for b in range(n):
        col = sorted(r[b] for r in mags)
        refs.append(max(col[int(len(col) * 0.90)], 1e-9))
    out = bytearray()
    prev = [0] * n
    for r in mags:
        for b in range(n):
            db = 20 * math.log10(max(r[b], 1e-9) / refs[b])        # 0 dB ~ 8 LEDs
            h = int(round(VIS_ROWS + db / 3.5))                    # 3.5 dB per LED
            h = max(0, min(VIS_ROWS, h))
            h = max(h, prev[b] - 1)                                # slow fall
            prev[b] = h
            out.append(h)
    return bytes(out)


def md_ui_table(total):
    out = bytearray()
    for s in range(total + 1):
        fill = s * sms.BAR_LEN // max(total, 1)
        col = max(fill - 1, 0)
        tile = 64 if fill >= 1 else 65
        addr = 0xC000 + (sms.BAR_ROW * 32 + sms.BAR_COL0 + col) * 2
        ctrl = 0x40000000 | ((addr & 0x3FFF) << 16) | (addr >> 14)
        out += struct.pack('>5HIH', *sms.mmss_tiles(s), ctrl, tile)
    out += b'\xFF' * 16
    return bytes(out)

# ----------------------------------------------------------------- 68000 code

def build_68k(org, zlen, ui_addr, tiles_words, autoplay, vis_base):
    m = M68K(org)
    ctrl = lambda addr: 0x40000000 | ((addr & 0x3FFF) << 16) | (addr >> 14)
    V = lambda n: '$%X' % n
    CALL = lambda l: (m.word(0x4EB9), m.long(l))
    m.label('start')
    m.word(0x46FC); m.word(0x2700)                       # move.w #$2700,sr
    m.move('b', '$A10001', 'd0')
    m.word(0x0200); m.word(0x000F)                       # andi.b #$0F,d0
    m.beq('notmss')
    m.move('l', '#$53454741', '$A14000')                 # 'SEGA' -> TMSS
    m.label('notmss')
    for v in (0x9F, 0xBF, 0xDF, 0xFF):                   # silence the PSG
        m.move('b', '#$%X' % v, V(PSG))
    m.move('b', '#$40', '$A10009')                       # pad 1: TH is an output
    m.move('b', '#$40', '$A10003')
    # load the Z80 program, set its command bytes, start it
    m.move('w', '#$100', '$A11100')                      # bus request
    m.move('w', '#$100', '$A11200')                      # reset released
    m.label('zw')
    m.word(0x0839); m.word(0); m.long(0xA11100)          # btst #0,$A11100
    m.bne('zw')
    m.lea('zprog', 0)
    m.lea('$A00000', 1)
    m.move('w', '#%d' % (zlen - 1), 'd0')
    m.label('zc')
    m.move('b', '(a0)+', '(a1)+')
    m.dbra(0, 'zc')
    m.move('b', '#%d' % (0 if autoplay else 2), V(CMD))
    m.move('b', '#0', V(LOOPF))                          # loop off
    m.move('w', '#0', '$A11200')                         # Z80 reset
    m.move('w', '#$FF', 'd0')
    m.label('zd')
    m.dbra(0, 'zd')
    m.move('w', '#0', '$A11100')                         # release bus
    m.move('w', '#$100', '$A11200')                      # run
    # VDP
    m.lea(V(VDP_CTRL), 1)
    m.lea(V(VDP_DATA), 2)
    m.lea('vdpregs', 0)
    m.moveq(18, 0)
    m.label('vr')
    m.move('w', '(a0)+', '(a1)')
    m.dbra(0, 'vr')
    m.move('l', '#$40000000', '(a1)')                    # clear all VRAM
    m.moveq(0, 1)
    m.move('w', '#$7FFF', 'd0')
    m.label('vc')
    m.move('w', 'd1', '(a2)')
    m.dbra(0, 'vc')
    m.move('l', '#$C0000000', '(a1)')                    # palette
    m.lea('pal', 0)
    m.moveq(15, 0)
    m.label('pl')
    m.move('w', '(a0)+', '(a2)')
    m.dbra(0, 'pl')
    m.move('l', '#$40000000', '(a1)')                    # tiles at VRAM 0
    m.lea('tiles', 0)
    m.move('w', '#%d' % (tiles_words - 1), 'd0')
    m.label('tl')
    m.move('w', '(a0)+', '(a2)')
    m.dbra(0, 'tl')
    m.move('l', '#$40000003', '(a1)')                    # plane A at $C000
    m.lea('tmap', 0)
    m.move('w', '#1023', 'd0')
    m.label('ml')
    m.move('w', '(a0)+', '(a2)')
    m.dbra(0, 'ml')
    m.move('w', '#$8164', '(a1)')                        # display on + vint
    # variables
    m.move('w', '#60', V(RELOAD))                        # 60 (NTSC) or 50 (PAL)
    m.word(0x0839); m.word(6); m.long(0xA10001)          # btst #6,$A10001
    m.beq('ntsc')
    m.move('w', '#50', V(RELOAD))
    m.label('ntsc')
    m.move('w', V(RELOAD), V(FRAMECNT))
    m.move('l', '#%d' % (ui_addr + 16), V(TBLPTR))
    m.move('b', '#%d' % (0 if autoplay else 2), V(STATE))
    m.move('b', '#$0F', V(PREV))                         # ignore buttons held at boot
    m.move('b', '#$0C', V(PREVLR))
    m.move('b', '#0', V(LOOP))
    m.move('b', '#1', V(VIS))
    m.move('b', '#0', V(VISDIV))
    m.move('l', '#%d' % vis_base, V(VISPTR))
    CALL('draw_label'); CALL('draw_loop'); CALL('draw_visflag')
    m.word(0x46FC); m.word(0x2000)                       # enable interrupts
    m.label('idle')
    m.word(0x4E72); m.word(0x2000)                       # stop #$2000
    m.bra('idle')

    # ---- vblank: pad -> state machine -> clock -> LED bars
    m.label('vint')
    m.move('w', V(VDP_CTRL), 'd0')                       # ack
    m.move('b', '#$40', '$A10003')
    m.word(0x4E71); m.word(0x4E71)
    m.move('b', '$A10003', 'd2')                         # Up,Down,Left,Right,B,C
    m.word(0x1C02)                                       # move.b d2,d6
    m.move('b', '#$00', '$A10003')
    m.word(0x4E71); m.word(0x4E71)
    m.move('b', '$A10003', 'd3')                         # A,Start
    m.word(0xE80A)                                       # lsr.b #4,d2
    m.word(0x0202); m.word(0x0003)                       # andi.b #3,d2
    m.word(0xE40B)                                       # lsr.b #2,d3
    m.word(0x0203); m.word(0x000C)                       # andi.b #$0C,d3
    m.word(0x8403)                                       # or.b d3,d2
    m.word(0x4602)                                       # not.b d2
    m.word(0x0202); m.word(0x000F)                       # andi.b #$0F,d2
    m.move('b', V(PREV), 'd4')
    m.move('b', 'd2', V(PREV))
    m.word(0x4604)                                       # not.b d4
    m.word(0xC404)                                       # and.b d4,d2  (new presses)
    m.word(0x1A02)                                       # move.b d2,d5
    m.word(0x4606)                                       # not.b d6
    m.word(0x0206); m.word(0x000C)                       # andi.b #$0C,d6  (Left=4, Right=8)
    m.move('b', V(PREVLR), 'd4')
    m.move('b', 'd6', V(PREVLR))
    m.word(0x4604)                                       # not.b d4
    m.word(0xCC04)                                       # and.b d4,d6  (new presses)
    # bit0=B bit1=C bit2=A bit3=Start
    m.move('b', V(STATE), 'd0')
    m.word(0x0805); m.word(3)                            # btst #3,d2   Start
    m.beq('p1')
    m.word(0x4A00)                                       # tst.b d0
    m.bne('p0')
    m.moveq(1, 0); m.bra('p1')                           # playing -> pause
    m.label('p0')
    m.moveq(0, 0)                                        # paused/stopped -> play
    m.label('p1')
    m.word(0x0805); m.word(2)                            # btst #2,d2   A = stop
    m.beq('p2'); m.moveq(2, 0)
    m.label('p2')
    m.move('b', V(STATE), 'd1')
    m.word(0xB200)                                       # cmp.b d0,d1
    m.beq('p3')
    m.move('b', 'd0', V(STATE))
    m.move('b', 'd0', 'd1')
    CALL('sendcmd')
    CALL('draw_label')
    m.move('b', V(STATE), 'd0')
    m.word(0x0C00); m.word(2)                            # cmpi.b #2,d0
    m.bne('p3')
    CALL('rewind_ui')                                    # stopped: back to 00:00
    m.label('p3')
    m.word(0x0805); m.word(0)                            # btst #0,d2   B = loop
    m.beq('p4')
    m.word(0x0A39); m.word(1); m.long(LOOP)              # eori.b #1,LOOP
    m.move('b', V(LOOP), 'd1')
    CALL('sendloop')
    CALL('draw_loop')
    m.label('p4')
    m.word(0x0805); m.word(1)                            # btst #1,d2   C = visualizer
    m.beq('seekchk')
    m.word(0x0A39); m.word(1); m.long(VIS)               # eori.b #1,VIS
    CALL('draw_visflag')
    CALL('clear_vis')
    # ---- LEFT / RIGHT: seek 5 seconds (not while stopped)
    m.label('seekchk')
    m.word(0x0C39); m.word(2); m.long(STATE)             # cmpi.b #2,STATE
    m.beq('tick')
    m.word(0x0806); m.word(2)                            # btst #2,d6   Left
    m.bne('sk_left')
    m.word(0x0806); m.word(3)                            # btst #3,d6   Right
    m.beq('tick')
    m.moveq(16, 7)
    m.bra('sk_go')
    m.label('sk_left')
    m.moveq(0, 7)
    m.label('sk_go')
    m.move('l', V(TBLPTR), 'd0')
    m.word(0x0480); m.long(ui_addr + 16)                 # subi.l #ui+16,d0
    m.word(0xE888)                                       # lsr.l #4,d0   -> current second
    m.word(0xEB88)                                       # lsl.l #5,d0   -> 32 bytes per second
    m.lea('seektab', 1)
    m.word(0xD3C0)                                       # adda.l d0,a1
    m.word(0xD3C7)                                       # adda.l d7,a1  (left 0 / right 16)
    m.move('w', '(a1)+', 'd1')                           # bank, $FFFF = nothing to do
    m.word(0x0C41); m.word(0xFFFF)                       # cmpi.w #$FFFF,d1
    m.beq('tick')
    m.move('w', '(a1)+', 'd3')                           # target second
    m.move('l', '(a1)+', 'd4')                           # TBLPTR for it
    m.move('l', '(a1)+', 'd5')                           # VISPTR for it
    m.move('w', '#$100', '$A11100')
    m.label('skw')
    m.word(0x0839); m.word(0); m.long(0xA11100)
    m.bne('skw')
    m.move('b', 'd1', V(0xA01FFB))                       # bank to jump to
    m.move('b', V(STATE), V(0xA01FFA))                   # command to resume with
    m.move('b', '#3', V(CMD))                            # 3 = seek
    m.move('w', '#0', '$A11100')
    CALL('rewind_ui')                                    # clears bar + LEDs
    m.move('w', 'd3', 'd0')
    m.beq('sk_nobar')
    m.word(0x5340)                                       # subq.w #1,d0
    m.lea(V(ui_addr + 16), 2)
    m.label('sk_bl')
    m.word(0x23EA); m.word(10); m.long(VDP_CTRL)         # move.l 10(a2),VDP_CTRL
    m.word(0x33EA); m.word(14); m.long(VDP_DATA)         # move.w 14(a2),VDP_DATA
    m.word(0xD4FC); m.word(16)                           # adda.w #16,a2
    m.dbra(0, 'sk_bl')
    m.label('sk_nobar')
    m.word(0x2044)                                       # movea.l d4,a0
    m.word(0x90FC); m.word(16)                           # suba.w #16,a0
    CALL('draw_entry')
    m.move('l', 'd4', V(TBLPTR))
    m.move('l', 'd5', V(VISPTR))
    m.label('tick')
    m.word(0x4A39); m.long(STATE)                        # tst.b STATE
    m.bne('vdone')                                       # only while playing
    m.word(0x5379); m.long(FRAMECNT)                     # subq.w #1,FRAMECNT
    m.bne('visupd')
    m.move('w', V(RELOAD), V(FRAMECNT))
    m.move('l', V(TBLPTR), 'a0')                         # movea.l TBLPTR,a0
    m.word(0x0C50); m.word(0xFFFF)                       # cmpi.w #$FFFF,(a0)
    m.bne('dodraw')
    m.word(0x4A39); m.long(LOOP)                         # tst.b LOOP
    m.bne('looped')
    m.move('b', '#2', V(STATE))                          # end of song: stop
    m.moveq(2, 1)
    CALL('sendcmd')
    CALL('draw_label')
    CALL('rewind_ui')
    m.bra('vdone')
    m.label('looped')
    CALL('rewind_ui')                                    # looping: start the clock over
    m.bra('visupd')
    m.label('dodraw')
    CALL('draw_entry')
    m.move('l', 'a0', V(TBLPTR))
    m.label('visupd')
    m.word(0x4A39); m.long(VIS)                          # tst.b VIS
    m.beq('vdone')
    m.word(0x0A39); m.word(1); m.long(VISDIV)            # eori.b #1,VISDIV
    m.word(0x4A39); m.long(VISDIV)
    m.beq('vdone')                                       # every other frame -> 30 fps
    m.move('l', V(VISPTR), 'a0')
    CALL('draw_vis')
    m.move('l', 'a0', V(VISPTR))
    m.label('vdone')
    m.word(0x4E73)                                       # rte

    # ---- d1.b = command -> Z80 RAM
    for name, addr in (('sendcmd', CMD), ('sendloop', LOOPF)):
        m.label(name)
        m.move('w', '#$100', '$A11100')
        m.label(name + 'w')
        m.word(0x0839); m.word(0); m.long(0xA11100)
        m.bne(name + 'w')
        m.move('b', 'd1', V(addr))
        m.move('w', '#0', '$A11100')
        m.word(0x4E75)                                   # rts

    # ---- a0 -> UI entry: draw time + bar tile, a0 advances 16 bytes
    m.label('draw_entry')
    ta = 0xC000 + (sms.TIME_ROW * 32 + sms.TIME_COL) * 2
    m.move('l', '#$%X' % ctrl(ta), V(VDP_CTRL))
    for _ in range(5):
        m.move('w', '(a0)+', V(VDP_DATA))
    m.move('l', '(a0)+', V(VDP_CTRL))
    m.move('w', '(a0)+', V(VDP_DATA))
    m.word(0x4E75)

    # ---- back to 00:00: empty progress bar and LED bars, rewind pointers
    m.label('rewind_ui')
    bar = 0xC000 + (sms.BAR_ROW * 32 + sms.BAR_COL0) * 2
    m.move('l', '#$%X' % ctrl(bar), V(VDP_CTRL))
    m.moveq(sms.BAR_LEN - 1, 0)
    m.label('rb')
    m.move('w', '#65', V(VDP_DATA))
    m.dbra(0, 'rb')
    m.move('l', '#%d' % ui_addr, 'a0')
    CALL('draw_entry')
    m.move('l', 'a0', V(TBLPTR))
    m.move('w', V(RELOAD), V(FRAMECNT))
    m.move('l', '#%d' % vis_base, V(VISPTR))
    m.move('b', '#0', V(VISDIV))
    CALL('clear_vis')
    m.word(0x4E75)

    # ---- LED bars: a0 -> 16 height bytes (0..8); a0 advances 16
    m.label('draw_vis')
    m.move('w', '#$8F40', V(VDP_CTRL))                   # auto-increment 64 = one tile row
    m.move('l', '#$%X' % ctrl(0xC000 + (VIS_ROW0 * 32 + VIS_COL0) * 2), 'd2')
    m.moveq(VIS_BARS - 1, 1)
    m.label('dv')
    m.moveq(0, 0)
    m.move('b', '(a0)+', 'd0')
    m.word(0xE948)                                       # lsl.w #4,d0   (16 bytes/height)
    m.lea('vistiles', 1)
    m.word(0xD2C0)                                       # adda.w d0,a1
    m.move('l', 'd2', V(VDP_CTRL))
    for _ in range(VIS_ROWS):
        m.move('w', '(a1)+', V(VDP_DATA))
    m.word(0x0682); m.long(0x00040000)                   # addi.l #$00040000,d2 (next bar, 2 tiles right)
    m.dbra(1, 'dv')
    m.move('w', '#$8F02', V(VDP_CTRL))                   # auto-increment back to 2
    m.word(0x4E75)

    m.label('clear_vis')                                 # unlit grid, or blank when off
    m.lea('zeros', 0)
    m.word(0x4A39); m.long(VIS)                          # tst.b VIS
    m.bne('cv')
    m.lea('blanks', 0)
    m.label('cv')
    CALL('draw_vis')
    m.word(0x4E75)

    # ---- PLAYING / PAUSED / STOPPED, LOOP OFF/ON, VIS OFF/ON
    m.label('draw_label')
    m.move('b', V(STATE), 'd0')
    m.word(0x0240); m.word(3)                            # andi.w #3,d0
    m.word(0xE948)                                       # lsl.w #4,d0
    m.lea('labels', 0)
    m.word(0xD0C0)                                       # adda.w d0,a0
    la = 0xC000 + (18 * 32 + 12) * 2
    m.move('l', '#$%X' % ctrl(la), V(VDP_CTRL))
    for _ in range(8):
        m.move('w', '(a0)+', V(VDP_DATA))
    m.word(0x4E75)

    for name, var, lbl, row, col in (('draw_loop', LOOP, 'lflags', 20, 6),
                                     ('draw_visflag', VIS, 'vflags', 20, 18)):
        m.label(name)
        m.move('b', V(var), 'd0')
        m.word(0x0240); m.word(1)                        # andi.w #1,d0
        m.word(0xE948)                                   # lsl.w #4,d0
        m.lea(lbl, 0)
        m.word(0xD0C0)                                   # adda.w d0,a0
        pos = 0xC000 + (row * 32 + col) * 2
        m.move('l', '#$%X' % ctrl(pos), V(VDP_CTRL))
        for _ in range(8):
            m.move('w', '(a0)+', V(VDP_DATA))
        m.word(0x4E75)

    m.label('dummy')
    m.word(0x4E73)
    return m


def md_seek_table(total, seek, ui_addr, vis_base):
    """Per current second k: a 16-byte entry for LEFT and one for RIGHT:
    bank word ($FFFF = nothing), target second word, new TBLPTR, new VISPTR, pad."""
    out = bytearray()
    for k in range(total + 1):
        for delta in (-5, 5):
            t = k + delta
            if seek is None or t < 0 and k == 0 or t >= total:
                out += struct.pack('>HHIII', 0xFFFF, 0, 0, 0, 0)
                continue
            t = max(t, 0)
            spb = seek['secs_per_bank']
            n = max(0, min(seek['nbanks'] - 1, int(round(t / spb))))
            sec = min(total - 1, int(round(n * spb)))
            out += struct.pack('>HHIII', seek['start_bank'] + n, sec,
                               ui_addr + 16 * (sec + 1),
                               vis_base + sec * VIS_FPS * VIS_BARS, 0)
    return bytes(out)


def build_rom(samples, title, total_secs, info, zprog, vis, autoplay=False, seek=None):
    # --- non-code blobs, all below DATA_OFF
    tiles = md_tiles()
    pal = struct.pack('>16H', 0x0000, 0x0EEE, 0x00E0, 0x0666, 0x00EE, 0x000E,
                      0x0222, *([0x0000] * 9))
    regs = struct.pack('>19H', 0x8004, 0x8104, 0x8230, 0x8334, 0x8407, 0x8578,
                       0x8600, 0x8700, 0x8800, 0x8900, 0x8A00, 0x8B00, 0x8C00,
                       0x8D3F, 0x8E00, 0x8F02, 0x9000, 0x9100, 0x9200)
    tmap = md_tilemap(title, total_secs, info)
    ui = md_ui_table(total_secs)
    labels = md_labels()
    blobs = (('zprog', zprog), ('tiles', tiles), ('pal', pal),
             ('vdpregs', regs), ('tmap', tmap), ('labels', labels),
             ('flags', md_flag_labels()), ('vistiles', md_vis_tiles()),
             ('zeros', bytes(VIS_BARS)), ('blanks', bytes([9] * VIS_BARS)),
             ('ui', ui), ('seektab', bytes((total_secs + 1) * 32)))
    vis_base = DATA_OFF + len(samples)

    code_org = 0x200
    probe = build_68k(code_org, len(zprog), 0, len(tiles) // 2, autoplay, vis_base)
    for n_, _ in blobs:
        probe.lab[n_] = 0
    probe.lab['lflags'] = probe.lab['vflags'] = 0
    codelen = len(probe.finish())
    p = (code_org + codelen + 1) & ~1
    addr = {}
    for name, blob in blobs:
        addr[name] = p
        p += (len(blob) + 1) & ~1
    assert p < DATA_OFF, 'UI data too big for first 64KB (%d)' % p
    m = build_68k(code_org, len(zprog), addr['ui'], len(tiles) // 2, autoplay,
                  vis_base)
    m.lab.update({k: v for k, v in addr.items() if k != 'ui'})
    blobs = [(n_, md_seek_table(total_secs, seek, addr['ui'], vis_base) if n_ == 'seektab' else b_)
             for n_, b_ in blobs]
    m.lab['lflags'] = addr['flags']
    m.lab['vflags'] = addr['flags'] + 32
    code = m.finish()
    assert len(code) == codelen

    rom = bytearray(DATA_OFF)
    # vector table
    vec = [0x00FFFE00, m.lab['start']] + [m.lab['dummy']] * 62
    vec[30] = m.lab['vint']                              # level 6 = vblank
    rom[0:256] = struct.pack('>64I', *vec)
    # header
    def txt(s, n): return s.encode('ascii', 'replace').ljust(n)[:n]
    name = title[:48]
    hdr = (txt('SEGA MEGA DRIVE', 16) + txt('(C)SMSMP3 2026.OCT', 16) +
           txt(name, 48) + txt(name, 48) + txt('GM 00000000-00', 14) +
           b'\0\0' + txt('J', 16) + struct.pack('>IIII', 0, 0, 0xFF0000, 0xFFFFFF) +
           txt('', 12) + txt('', 12) + txt('', 40) + txt('JUE', 16))
    assert len(hdr) == 0x100, len(hdr)
    rom[0x100:0x200] = hdr
    rom[0x200:0x200 + len(code)] = code
    for name_, blob in blobs:
        rom[addr[name_]:addr[name_] + len(blob)] = blob
    rom += samples
    assert len(rom) == vis_base
    rom += vis
    rom += bytes(len(rom) & 1)
    size = len(rom)
    assert size <= ROM_MAX, 'ROM too big'
    rom[0x1A4:0x1A8] = struct.pack('>I', size - 1)       # ROM end
    cs = 0
    for i in range(0x200, size - 1, 2):
        cs = (cs + (rom[i] << 8 | rom[i + 1])) & 0xFFFF
    rom[0x18E:0x190] = struct.pack('>H', cs)
    return bytes(rom)

# ---------------------------------------------------------------- audio side

def to_u8(pcm, fb, peak=None):
    """s16 -> unsigned 8-bit with first-order error feedback."""
    peak = peak or max(1, max(max(pcm), -min(pcm)))
    k = 127.0 * 0.97 / peak
    out = bytearray(len(pcm))
    e = 0.0
    for i, s in enumerate(pcm):
        v = s * k + 128.0 + e
        q = int(round(v))
        q = 1 if q < 1 else 254 if q > 254 else q          # 0 is the end marker
        out[i] = q
        e = max(-1.0, min(1.0, (v - q) * fb))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('mp3'); ap.add_argument('rom')
    ap.add_argument('--rate', type=int, default=0,
                    help='sample rate in Hz (total, L+R together; default: highest that fits, max 26000)')
    ap.add_argument('--title'); ap.add_argument('--start', type=float, default=0)
    ap.add_argument('--duration', type=float, default=0)
    ap.add_argument('--gain', type=int, default=0,
                    help='dB of boost before a slow peak limiter (default 0)')
    ap.add_argument('--wait', type=float, default=None,
                    help='assumed Z80 wait-state cycles per ROM read')
    ap.add_argument('--shape', type=float, default=-0.5,
                    help='noise-shaping feedback (negative pushes noise to bass)')
    ap.add_argument('--slots', type=int, default=0,
                    help='sample period in YM2612 output slots (default: smallest that fits, min 2)')
    ap.add_argument('--stereo', action='store_true',
                    help='stereo: DAC carries L-R (left only), PSG carries R (both ears)')
    ap.add_argument('--dac-gain', type=float, default=0.383,
                    help='scale of the DAC (L-R) path against the PSG path')
    ap.add_argument('--autoplay', action='store_true',
                    help='start playing at power-on instead of waiting for Start')
    ap.add_argument('--wav')
    args = ap.parse_args()

    if args.wait is None:
        # three back-to-back ROM-bus reads per sample cost more wait states
        args.wait = 5.03 if args.stereo else BUS_WAIT
    title = args.title or os.path.splitext(os.path.basename(args.mp3))[0]
    pr = subprocess.run(['ffprobe', '-v', 'error', '-show_entries',
                         'format=duration', '-of', 'csv=p=0', args.mp3],
                        check=True, capture_output=True, text=True)
    dur = float(pr.stdout) - args.start
    if args.duration:
        dur = min(dur, args.duration)
    room = ROM_MAX - DATA_OFF - int(VIS_BARS * VIS_FPS * dur) - 2048   # LED-bar table
    bps = 3 if args.stereo else 1             # ROM bytes per sample
    fit = int(room / dur / bps) - 16           # samples/s that fit
    if args.rate:
        rate = min(args.rate, fit)
        T = Z80_HZ / rate
        slots = None
    else:
        # lock the sample period to a whole number of YM2612 output slots:
        # the chip only updates its output every 67.2 Z80 cycles
        slots = args.slots or 2
        while Z80_HZ / (YM_SLOT * slots) > fit:
            slots += 1
        T = YM_SLOT * slots
    if args.stereo:
        # 3 ROM bytes per sample: fixed loop ~ 3*(26+W) + 23 + 24 + 10
        pad = pad_for(T - (3 * 26 + 57 + 3 * args.wait))
        cyc = 3 * 26 + 57 + 3 * args.wait + pad[3]
    else:
        pad = pad_for(T - FIXED - args.wait)
        cyc = FIXED + args.wait + pad[3]
    real = Z80_HZ / cyc
    per_ch = real
    print('%.1f cycles/sample%s -> %.1f Hz%s, %.1fs' %
          (cyc, ' (%d YM slots)' % slots if slots else '', real,
           ' stereo' if args.stereo else '', dur))

    af = 'lowpass=f=%d' % int(per_ch * 0.45)
    if args.gain:
        af += (',volume=%ddB,alimiter=limit=0.9:attack=5:release=200'
               ':level=disabled' % args.gain)
    cmd = ['ffmpeg', '-v', 'error', '-i', args.mp3, '-vn',
           '-ac', '2' if args.stereo else '1']
    if args.start: cmd += ['-ss', str(args.start)]
    if args.duration: cmd += ['-t', str(args.duration)]
    cmd += ['-af', af, '-ar', str(int(round(per_ch))), '-f', 's16le', '-']
    raw = subprocess.run(cmd, check=True, stdout=subprocess.PIPE).stdout
    pcm = array.array('h'); pcm.frombytes(raw[:len(raw) // 2 * 2])
    if args.stereo:
        left, right = pcm[0::2], pcm[1::2]
        n_ = min(len(left), len(right))
        peak = max(1, max(max(pcm), -min(pcm)))
        # PSG carries R: two volume writes per sample, noise-shaped
        sms.GLITCH_F = 29.3 / cyc
        secs_ = max(1, int(math.ceil(n_ / real)))
        rr = array.array('h', right[:n_])
        psg = sms.quantize(rr, secs_, int(round(real)), peak=peak)
        # DAC carries L-R, scaled to match the PSG's output level
        dac = bytearray(n_)
        e = 0.0
        k = 127.0 * 0.97 * args.dac_gain / peak
        for i in range(n_):
            v = (left[i] - right[i]) * k + 128.0 + e
            q = int(round(v)); q = 0 if q < 0 else 255 if q > 255 else q
            dac[i] = q
            e = max(-1.0, min(1.0, (v - q) * args.shape))
        nsamp = n_
        per_bank = SAMPLES_PER_BANK
        banks = []
        for s0 in range(0, nsamp, per_bank):
            blk = bytearray()
            for i in range(s0, min(s0 + per_bank, nsamp)):
                blk += bytes((dac[i], psg[2 * i], psg[2 * i + 1]))
            blk += bytes([0x80, 0x9F, 0x9F]) * ((per_bank * 3 - len(blk)) // 3)
            blk += bytes(BANK - len(blk))
            banks.append(blk)
        data = bytearray().join(banks)
    else:
        data = to_u8(pcm, args.shape)
        nsamp = len(data)
    total = max(1, int(math.ceil(nsamp / per_ch)))
    if args.stereo:
        mono = array.array('h', ((pcm[2 * i] + pcm[2 * i + 1]) // 2 for i in range(nsamp)))
    else:
        mono = pcm
        data += b'\x00'                                   # end marker (+ zero pad)
    data += (b'\x80' if args.stereo else b'\x00') * (-len(data) % BANK)
    print('computing LED-bar levels ...')
    vis = compute_vis(mono, per_ch, total)
    nb = len(data) // BANK
    start_bank = DATA_OFF // BANK
    last_bank = start_bank + nb - 1
    assert last_bank < 255
    if args.stereo:
        info = '%d HZ  STEREO' % int(round(per_ch))
        zprog = build_z80_hybrid(start_bank, last_bank, pad,
                                 [0x90 | v for v in sms.start_state()[:1]] +
                                 [0xB0 | sms.start_state()[1],
                                  0xD0 | sms.start_state()[2]])
    else:
        info = '%d HZ  8BIT DAC' % int(round(real))
        zprog = build_z80(start_bank, last_bank, pad)
    spb = (SAMPLES_PER_BANK if args.stereo else BANK) / float(per_ch)
    seek = dict(start_bank=start_bank, nbanks=nb, secs_per_bank=spb)
    rom = build_rom(bytes(data), title.upper(), total, info, zprog, vis,
                    args.autoplay, seek)
    open(args.rom, 'wb').write(rom)
    print('wrote %s (%d KB, %d s)' % (args.rom, len(rom) // 1024, total))
    if args.wav:
        w = wave.open(args.wav, 'wb')
        w.setnchannels(1); w.setsampwidth(1); w.setframerate(int(round(real)))
        w.writeframes(bytes(data)); w.close()


if __name__ == '__main__':
    main()
