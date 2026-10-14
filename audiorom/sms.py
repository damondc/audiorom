#!/usr/bin/env python3
"""sms - turn an MP3 into a Sega Master System audio-player ROM.

The SMS has no DAC, so the ROM streams PCM through the SN76489 PSG: all three
tone channels are parked at period 0 (a DC output) and their 4-bit volume
registers are rewritten each sample (two writes per sample, chosen by this tool
so the intermediate glitch is accounted for).  The sample loop is cycle-counted Z80
assembly (assembled here by a tiny built-in assembler), so no external
toolchain is needed beyond ffmpeg.

    audiorom sms song.mp3 song.sms [--rate 11025] [--title NAME]
"""
import argparse, array, bisect, gzip, math, os, struct, subprocess, sys, wave

Z80_HZ = 3579545          # NTSC
MAX_BANKS = 256           # Sega mapper: 256 x 16KB
MAX_AUTO_RATE = 28000
DATA_BANK0 = 2            # bank 0 = code, bank 1 = UI data + header
BANK_SIZE = 0x4000
HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------- audio side

def decode(path, rate, start, dur, gain=0):
    cmd = ['ffmpeg', '-v', 'error', '-i', path, '-vn', '-ac', '1']
    if start:
        cmd += ['-ss', str(start)]
    if dur:
        cmd += ['-t', str(dur)]
    af = 'lowpass=f=%d' % int(rate * 0.45)
    if gain:
        # boost then peak-limit with a slow release: the PSG's absolute output
        # is capped, so loudness has to come from a higher average level
        af += ',volume=%ddB,alimiter=limit=0.9:attack=5:release=200:level=disabled' % gain
    cmd += ['-af', af, '-ar', str(rate),
            '-f', 's16le', '-']
    raw = subprocess.run(cmd, check=True, stdout=subprocess.PIPE).stdout
    a = array.array('h')
    a.frombytes(raw[:len(raw) // 2 * 2])
    return a


ATT_AMP = [10 ** (-a / 10.0) for a in range(15)] + [0.0]    # amplitude per attenuation
# The PSG follows nominal 2 dB steps for attenuations 0..15.  --att-min raises
# the loudest level used (quieter output, less summed-channel clipping).
ATT_MIN = 0
TOP = 3 * ATT_AMP[ATT_MIN]            # loudest summed level
MID = TOP / 2                         # silence


def set_att_min(n):
    global ATT_MIN, TOP, MID
    ATT_MIN, TOP = n, 3 * ATT_AMP[n]
    MID = TOP / 2


def start_state():
    """Attenuations (a0,a1,a2) whose summed amplitude is closest to mid-scale."""
    r = range(ATT_MIN, 16)
    return min(((a0, a1, a2) for a0 in r for a1 in r for a2 in r),
               key=lambda t: abs(sum(ATT_AMP[a] for a in t) - MID))


GLITCH_F = 24 / 322.0     # fraction of a sample period spent between the two writes
PAIRS = ((0, 1), (1, 0), (0, 2), (2, 0), (1, 2), (2, 1))


def quantize(pcm, secs, per_sec, thr=0.003, fb=-0.6, peak=None):
    """Two PSG volume writes per sample (two bytes: channel + attenuation).

    The tool picks the one or two channel changes whose *time-averaged* level
    over the sample period (including the brief intermediate state between the
    two writes) lands nearest the target, with first-order error feedback so
    glitch energy is compensated rather than left as scratchiness."""
    f = GLITCH_F
    asc = sorted((ATT_AMP[a], a) for a in range(ATT_MIN, 16))
    amps = [x[0] for x in asc]
    att_of = [x[1] for x in asc]
    peak = peak or max(1, max(max(pcm), -min(pcm)))
    scale = 0.97 * MID / peak
    n = secs * per_sec
    amp = [ATT_AMP[a] for a in start_state()]
    out = bytearray(2 * n)
    e = 0.0
    bl = bisect.bisect_left
    NA = len(amps)

    def nearest(need):
        k = bl(amps, need)
        if k >= NA:
            return NA - 1
        if k > 0 and need - amps[k - 1] < amps[k] - need:
            return k - 1
        return k

    for i in range(n):
        v = MID + (pcm[i] * scale if i < len(pcm) else 0.0) + e
        tot = amp[0] + amp[1] + amp[2]
        best = 9.0
        sel = None
        for ch in (0, 1, 2):
            k = nearest(v - (tot - amp[ch]))
            c = tot - amp[ch] + amps[k]
            d = abs(c - v)
            if d < best:
                best, sel = d, (ch, k, -1, 0, c)
        if best > thr:
            for a, b in PAIRS:
                base = tot - amp[a] - amp[b]
                ob = amp[b]
                for ka in range(NA):
                    mid = base + amps[ka] + ob
                    kb = nearest((v - f * mid) / (1 - f) - base - amps[ka])
                    c = (1 - f) * (base + amps[ka] + amps[kb]) + f * mid
                    d = abs(c - v)
                    if d < best:
                        best, sel = d, (a, ka, b, kb, c)
        ch, k, b, kb, cell = sel
        amp[ch] = amps[k]
        c1 = 0x90 | (ch << 5) | att_of[k]
        if b >= 0:
            amp[b] = amps[kb]
            c2 = 0x90 | (b << 5) | att_of[kb]
        else:
            c2 = c1
        out[2 * i] = c1
        out[2 * i + 1] = c2
        e = max(-0.4, min(0.4, (v - cell) * fb))
    return out

# --------------------------------------------------------------- Z80 assembler

class Asm:
    def __init__(self, org=0):
        self.org, self.b, self.lab, self.fix = org, bytearray(), {}, []

    @property
    def pc(self):
        return self.org + len(self.b)

    def label(self, n):
        self.lab[n] = self.pc

    def db(self, *v):
        self.b += bytes(v)

    def w(self, v):
        self.b += struct.pack('<H', v)

    def jr(self, op, n):           # op: 0x18 jr, 0x20 jr nz, 0x28 jr z, 0x10 djnz
        self.db(op, 0)
        self.fix.append((len(self.b) - 1, n, True))

    def abs(self, op, n):          # 3-byte absolute (jp/call/ld hl,...) w/ label
        self.db(op, 0, 0)
        self.fix.append((len(self.b) - 2, n, False))

    def finish(self):
        for off, n, rel in self.fix:
            t = self.lab[n] if isinstance(n, str) else n
            if rel:
                d = t - (self.org + off + 1)
                assert -128 <= d < 128, (n, d)
                self.b[off] = d & 0xFF
            else:
                self.b[off:off + 2] = struct.pack('<H', t)
        return bytes(self.b)

# ------------------------------------------------------------------- UI data

def load_font():
    d = gzip.open(os.path.join(HERE, 'font8.psf.gz')).read()
    assert d[:2] == b'\x36\x04' and d[3] == 8
    return lambda c: d[4 + c * 8:4 + c * 8 + 8]


def tiles_4bpp(font):
    t = bytearray()
    for c in range(32, 96):                    # tiles 0..63: ASCII 32..95
        for r in font(c):
            t += bytes((r, 0, 0, 0))           # colour 1
    for r in range(8):                         # tile 64: bar full (colour 2)
        t += bytes((0, 0xFF if 1 <= r <= 6 else 0, 0, 0))
    for r in range(8):                         # tile 65: bar empty (colour 3)
        t += bytes((0xFF, 0xFF, 0, 0) if r in (3, 4) else (0, 0, 0, 0))
    return bytes(t)

BAR_COL0, BAR_LEN, BAR_ROW, TIME_ROW, TIME_COL = 2, 28, 14, 16, 9

def ch_tile(c):
    o = ord(c.upper())
    return o - 32 if 32 <= o <= 95 else ord('-') - 32


def mmss_tiles(sec):
    m, s = divmod(sec, 60)
    m = min(m, 99)
    return [16 + m // 10, 16 + m % 10, ch_tile(':'), 16 + s // 10, 16 + s % 10]


def tilemap(title, total, info, head=('SEGA MASTER SYSTEM', 'AUDIO PLAYER')):
    m = [[0] * 32 for _ in range(28)]

    def put(row, text, col=None):
        text = text[:32]
        c = (32 - len(text)) // 2 if col is None else col
        for i, ch in enumerate(text):
            m[row][c + i] = ch_tile(ch)
    put(3, head[0])
    put(5, head[1])
    put(9, 'NOW PLAYING')
    put(11, title[:28])
    for i in range(BAR_LEN):
        m[BAR_ROW][BAR_COL0 + i] = 65
    for i, t in enumerate(mmss_tiles(0)):
        m[TIME_ROW][TIME_COL + i] = t
    put(TIME_ROW, ' / ', TIME_COL + 5)
    for i, t in enumerate(mmss_tiles(total)):
        m[TIME_ROW][TIME_COL + 8 + i] = t
    put(21, info)
    b = bytearray()
    for row in m:
        for t in row:
            b += struct.pack('<H', t)
    return bytes(b)


def ui_table(total):
    rows = bytearray()
    for s in range(total + 1):
        fill = s * BAR_LEN // max(total, 1)
        col = max(fill - 1, 0)
        tile = 64 if fill >= 1 else 65
        rows += bytes(mmss_tiles(s)) + bytes((col, tile, 0))
    rows += bytes([0xFF]) * 8
    return bytes(rows)

# ------------------------------------------------------------------ the ROM

FIXED = 98                                 # cycles of the loop minus the pad
BANKVAR, UIPOS = 0xC000, 0xC001
CODE_MAX = 0x400
PAL = bytes((0x10, 0x3F, 0x0C, 0x15, 0x0F, 0x3C, 0x03, 0x30) + (0x10,) * 8) * 2
PAL = PAL[:16] + bytes([0x10]) + PAL[17:32]  # border colour = background


def build_code(vols, rate, per_sec, cyc_pad_n, nops, ui_addr, tiles_addr, tiles_len,
               pal_addr, map_addr, ntiles_bytes):
    a = Asm(0)
    # reset / interrupt vectors
    a.db(0xF3, 0xED, 0x56)                    # di ; im 1
    a.abs(0xC3, 'init')                       # jp init
    while a.pc < 0x38: a.db(0)
    a.db(0xED, 0x4D)                          # reti
    while a.pc < 0x66: a.db(0)
    a.db(0xED, 0x45)                          # retn (pause button)
    while a.pc < 0x100: a.db(0)

    a.label('init')
    a.db(0x31, 0xF0, 0xDF)                    # ld sp,$DFF0
    for reg, val in ((0xFFFC, 0), (0xFFFD, 0), (0xFFFE, 1), (0xFFFF, DATA_BANK0)):
        a.db(0x3E, val, 0x32); a.w(reg)       # ld a,val ; ld (reg),a
    a.db(0x3E, DATA_BANK0, 0x32); a.w(BANKVAR)
    # VDP registers (display off for now)
    a.abs(0x21, 'vdpregs')                    # ld hl,vdpregs
    a.db(0x06, 22, 0x0E, 0xBF, 0xED, 0xB3)   # ld b,22 ; ld c,$BF ; otir
    # clear VRAM
    a.db(0xAF, 0xD3, 0xBF, 0x3E, 0x40, 0xD3, 0xBF)
    a.db(0x01); a.w(0x4000)                   # ld bc,$4000
    a.label('clr')
    a.db(0xAF, 0xD3, 0xBE, 0x0B, 0x78, 0xB1)  # xor a;out;dec bc;ld a,b;or c
    a.jr(0x20, 'clr')
    # palette
    a.db(0xAF, 0xD3, 0xBF, 0x3E, 0xC0, 0xD3, 0xBF)
    a.db(0x21); a.w(pal_addr)
    a.db(0x06, 32, 0x0E, 0xBE, 0xED, 0xB3)
    # tiles -> VRAM $0000
    a.db(0xAF, 0xD3, 0xBF, 0x3E, 0x40, 0xD3, 0xBF)
    a.db(0x21); a.w(tiles_addr)
    a.db(0x01); a.w(tiles_len)
    a.abs(0xCD, 'copy')
    # tilemap -> VRAM $3800
    a.db(0xAF, 0xD3, 0xBF, 0x3E, 0x78, 0xD3, 0xBF)
    a.db(0x21); a.w(map_addr)
    a.db(0x01); a.w(0x700)
    a.abs(0xCD, 'copy')
    # sprite table: Y=$D0 ends the list
    a.db(0xAF, 0xD3, 0xBF, 0x3E, 0x7F, 0xD3, 0xBF, 0x3E, 0xD0, 0xD3, 0xBE)
    # PSG: tone period 0 on all channels, noise off, volumes off
    for b in (0x80, 0x00, 0xA0, 0x00, 0xC0, 0x00, 0xFF,
              0x90 | vols[0], 0xB0 | vols[1], 0xD0 | vols[2]):
        a.db(0x3E, b, 0xD3, 0x7F)
    # display on
    a.db(0x3E, 0x40, 0xD3, 0xBF, 0x3E, 0x81, 0xD3, 0xBF)
    a.db(0x21); a.w(ui_addr + 8)              # ld hl,uitable+8
    a.db(0x22); a.w(UIPOS)                    # ld (UIPOS),hl
    a.db(0x21); a.w(0x8000)                   # ld hl,$8000
    a.db(0x01); a.w(per_sec)                  # ld bc,per_sec

    # ---- the sample loop: every path through is cycle-counted ----
    a.label('s')
    a.db(0x7E, 0xD3, 0x7F, 0x23)              # ld a,(hl) ; out ($7f),a ; inc hl
    a.db(0x7E, 0xD3, 0x7F, 0x23)              # second volume write
    a.db(0xCB, 0x74)                          # bit 6,h
    a.jr(0x20, 'bank')                        # jr nz,bank
    a.label('back')
    a.db(0x0B, 0x78, 0xB1)                    # dec bc ; ld a,b ; or c
    a.jr(0x28, 'tick')                        # jr z,tick
    a.db(0x3E, cyc_pad_n)                     # ld a,N
    a.label('pad')
    a.db(0x3D)                                # dec a
    a.jr(0x20, 'pad')
    a.db(*([0x00] * nops))
    a.jr(0x18, 's')

    a.label('bank')                           # crossed a 16KB page
    a.db(0x3A); a.w(BANKVAR)                  # ld a,(BANKVAR)
    a.db(0x3C, 0x32); a.w(BANKVAR)            # inc a ; ld (BANKVAR),a
    a.db(0x32); a.w(0xFFFF)                   # ld ($FFFF),a
    a.db(0x26, 0x80)                          # ld h,$80
    a.jr(0x18, 'back')

    a.label('tick')                           # once per second: update UI
    a.db(0xE5)                                # push hl
    a.db(0x2A); a.w(UIPOS)                    # ld hl,(UIPOS)
    a.db(0x7E, 0xFE, 0xFF)                    # ld a,(hl) ; cp $FF
    a.jr(0x20, 'wr')
    # end of song: rewind
    a.db(0x3E, DATA_BANK0, 0x32); a.w(BANKVAR)
    a.db(0x32); a.w(0xFFFF)
    a.db(0x21); a.w(0x8000)                   # ld hl,$8000
    a.db(0xE3)                                # ex (sp),hl
    a.db(0x3E, (0x3800 + (BAR_ROW * 32 + BAR_COL0) * 2) & 0xFF, 0xD3, 0xBF)
    a.db(0x3E, 0x40 | (0x3800 + (BAR_ROW * 32 + BAR_COL0) * 2) >> 8, 0xD3, 0xBF)
    a.db(0x06, BAR_LEN)
    a.label('cl')
    a.db(0x3E, 65, 0xD3, 0xBE, 0xAF, 0xD3, 0xBE)
    a.jr(0x10, 'cl')
    a.db(0x21); a.w(ui_addr)                  # ld hl,uitable (entry 0)
    a.label('wr')
    ta = 0x3800 + (TIME_ROW * 32 + TIME_COL) * 2
    a.db(0x3E, ta & 0xFF, 0xD3, 0xBF, 0x3E, 0x40 | ta >> 8, 0xD3, 0xBF)
    a.db(0x06, 5)
    a.label('dl')
    a.db(0x7E, 0xD3, 0xBE, 0x23, 0xAF, 0xD3, 0xBE)
    a.jr(0x10, 'dl')
    ba = 0x3800 + (BAR_ROW * 32 + BAR_COL0) * 2
    a.db(0x7E, 0x23, 0x87, 0xC6, ba & 0xFF, 0xD3, 0xBF)   # col*2+lo
    a.db(0x3E, 0x40 | ba >> 8, 0xD3, 0xBF)
    a.db(0x7E, 0x23, 0xD3, 0xBE, 0xAF, 0xD3, 0xBE, 0x23)
    a.db(0x22); a.w(UIPOS)                    # ld (UIPOS),hl
    a.db(0xE1)                                # pop hl
    a.db(0x01); a.w(per_sec)                  # ld bc,per_sec
    a.abs(0xC3, 's')                          # jp s

    a.label('copy')                           # hl=src bc=len -> VDP data
    a.db(0x7E, 0xD3, 0xBE, 0x23, 0x0B, 0x78, 0xB1)
    a.jr(0x20, 'copy')
    a.db(0xC9)

    a.label('vdpregs')
    regs = [0x04, 0x00, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0xFF]
    for i, v in enumerate(regs):
        a.db(v, 0x80 | i)
    code = a.finish()
    assert len(code) < CODE_MAX, len(code)
    return code

# ---------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('mp3'); ap.add_argument('rom')
    ap.add_argument('--rate', type=int, default=0,
                    help='sample rate in Hz (default: highest that fits in 4MB, max %d)' % MAX_AUTO_RATE)
    ap.add_argument('--title'); ap.add_argument('--start', type=float, default=0)
    ap.add_argument('--duration', type=float, default=0)
    ap.add_argument('--gain', type=int, default=16,
                    help='dB of boost before a slow peak limiter; louder but '
                         'pumpier (default 16, 0 = original dynamics)')
    ap.add_argument('--att-min', type=int, default=0, choices=range(0, 8),
                    help='loudest PSG attenuation to use (0 = full scale; '
                         'higher is quieter)')
    ap.add_argument('--wav', help='also write a preview WAV of what the PSG outputs')
    args = ap.parse_args()
    set_att_min(args.att_min)

    title = args.title or os.path.splitext(os.path.basename(args.mp3))[0]
    probe = subprocess.run(['ffprobe', '-v', 'error', '-show_entries',
                            'format=duration', '-of', 'csv=p=0', args.mp3],
                           check=True, capture_output=True, text=True)
    dur = float(probe.stdout) - args.start
    if args.duration:
        dur = min(dur, args.duration)
    room = (MAX_BANKS - DATA_BANK0) * BANK_SIZE
    fit = int(room / dur / 2) - 8          # 2 bytes per sample
    want = args.rate if args.rate else MAX_AUTO_RATE
    if want > fit:
        print('song too long for 4MB at %d Hz; using %d Hz' % (want, fit))
    rate = min(want, fit, MAX_AUTO_RATE)
    # cycle budget per sample -> exact rate
    cyc = -(-Z80_HZ // rate)                       # round up: never exceed rate
    n = (cyc - FIXED) // 16
    nops = -(-(cyc - FIXED - 16 * n) // 4)
    cyc = FIXED + 16 * n + 4 * nops
    real_rate = Z80_HZ / cyc
    per_sec = round(real_rate)
    globals()['GLITCH_F'] = 24.0 / cyc   # time between the two writes, as a fraction
    assert dur * per_sec * 2 <= room, 'cannot fit; use --duration'
    print('%d cycles/sample -> %.1f Hz, %.1fs' % (cyc, real_rate, dur))

    pcm = decode(args.mp3, per_sec, args.start, args.duration, args.gain)
    secs = max(1, (len(pcm) + per_sec - 1) // per_sec)
    data = quantize(pcm, secs, per_sec)
    print('%d samples (%d s)' % (len(data) // 2, secs))

    # bank 1: tiles, palette, tilemap, UI table
    info = '%d HZ  3CH PSG PCM' % per_sec
    tiles = tiles_4bpp(load_font())
    b1 = bytearray()
    tiles_addr = 0x4000 + len(b1); b1 += tiles
    pal_addr = 0x4000 + len(b1); b1 += PAL
    map_addr = 0x4000 + len(b1); b1 += tilemap(title, secs, info)
    ui_addr = 0x4000 + len(b1); b1 += ui_table(secs)
    assert len(b1) < 0x3FF0, 'song too long for UI table'

    code = build_code(start_state(), real_rate, per_sec, n, nops, ui_addr, tiles_addr,
                      len(tiles), pal_addr, map_addr, len(tiles))
    bank0 = bytearray(BANK_SIZE)
    bank0[:len(code)] = code

    bank1 = bytearray(BANK_SIZE)
    bank1[:len(b1)] = b1
    hdr = 0x3FF0
    bank1[hdr:hdr + 8] = b'TMR SEGA'
    bank1[hdr + 15] = 0x4C                     # SMS export, 32KB size code
    rom = bytearray(bank0 + bank1)
    cs = sum(rom[:0x7FF0]) & 0xFFFF
    rom[0x7FFA:0x7FFC] = struct.pack('<H', cs)
    rom += data
    rom += bytes(-len(rom) % BANK_SIZE)
    assert len(rom) <= MAX_BANKS * BANK_SIZE
    open(args.rom, 'wb').write(rom)
    print('wrote %s (%d KB)' % (args.rom, len(rom) // 1024))

    if args.wav:
        amp = [ATT_AMP[a] for a in start_state()]
        out = array.array('h')
        f = GLITCH_F
        for i in range(0, len(data), 2):
            amp[(data[i] >> 5) & 3] = ATT_AMP[data[i] & 15]
            mid = sum(amp)
            amp[(data[i + 1] >> 5) & 3] = ATT_AMP[data[i + 1] & 15]
            lvl = f * mid + (1 - f) * sum(amp)
            out.append(int((lvl - MID) / MID * 30000))
        w = wave.open(args.wav, 'wb')
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(per_sec)
        w.writeframes(out.tobytes()); w.close()


if __name__ == '__main__':
    main()
