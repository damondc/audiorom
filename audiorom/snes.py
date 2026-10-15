#!/usr/bin/env python3
"""snes - turn an audio file into a Super Nintendo audio-player ROM.

The S-DSP plays 4-bit BRR samples, so the converter encodes the audio as a
BRR stream (32 kHz at best, 9 bytes per 16 samples) and stores it in a HiROM
cartridge.  A small SPC700 program (assembled here and uploaded through the
IPL boot protocol) keeps a ring buffer of BRR blocks filled from the 65816
over the four communication ports, plays it with one DSP voice and reports the
playback position back for the on-screen UI.  Needs ffmpeg; the 65816 and
SPC700 assemblers are built in.

    audiorom snes song.mp3 song.sfc [--rate 32000] [--title NAME] [--autoplay]

Controls: START play/pause, B stop+rewind, A loop on/off, SELECT visualizer,
LEFT/RIGHT seek 5 s.
"""
import argparse, array, math, os, re, struct, sys

from audiorom import sms

DSP_RATE = 32000            # pitch $1000
T0_HZ = 8000                # SPC700 timer 0 base rate
VIS_FPS = 30
VIS_BARS = 8
VIS_ROWS = 8
RING_BLOCKS = 1024          # 9 KB BRR ring in ARAM
RING = 0x0600
SPC_LOAD = 0x0200
BANK_VIS = 0xC1
BANK_STREAM = 0xC2
MAX_ROM = 4 * 1024 * 1024
VIS_FRAMES = 8192


# ---------------------------------------------------------------------------
# generic two-pass assembler core
# ---------------------------------------------------------------------------
class _Asm:
    def __init__(self, org, syms):
        self.org = org
        self.syms = dict(syms)
        self.items = []
        self.labels = {}
        self.uid_n = 0

    def uid(self, p='L'):
        self.uid_n += 1
        return '%s_%d' % (p, self.uid_n)

    def lbl(self, name):
        self.items.append(('lbl', name))

    def db(self, data):
        self.items.append(('db', bytes(data)))

    def align(self, n):
        self.items.append(('align', n))

    def _eval(self, expr, labels):
        if isinstance(expr, int):
            return expr
        e = expr.strip().replace('$', '0x')
        env = dict(self.syms)
        env.update(labels)
        return int(eval(e, {'__builtins__': {}}, env))

    def _known(self, expr):
        try:
            return self._eval(expr, {})
        except Exception:
            return None

    def _layout(self):
        pc = self.org
        for it in self.items:
            k = it[0]
            if k == 'lbl':
                self.labels[it[1]] = pc
            elif k == 'ins':
                pc += it[-1]                    # last field is the size
            elif k == 'db':
                pc += len(it[1])
            elif k == 'align':
                pc = (pc + it[1] - 1) // it[1] * it[1]
        return pc


# ---------------------------------------------------------------------------
# 65816
# ---------------------------------------------------------------------------
_G1 = {'dix': 0x01, 'dp': 0x05, 'dil': 0x07, 'imm': 0x09, 'abs': 0x0D, 'lng': 0x0F,
       'diy': 0x11, 'di': 0x12, 'dpx': 0x15, 'dily': 0x17, 'aby': 0x19, 'abx': 0x1D,
       'lngx': 0x1F}
OPS16 = {}
for _mn, _base in (('ora', 0x00), ('and', 0x20), ('eor', 0x40), ('adc', 0x60),
                   ('sta', 0x80), ('lda', 0xA0), ('cmp', 0xC0), ('sbc', 0xE0)):
    OPS16[_mn] = {m: _base + o for m, o in _G1.items() if not (_mn == 'sta' and m == 'imm')}
OPS16.update({
    'asl': dict(acc=0x0A, dp=0x06, abs=0x0E, dpx=0x16, abx=0x1E),
    'rol': dict(acc=0x2A, dp=0x26, abs=0x2E, dpx=0x36, abx=0x3E),
    'lsr': dict(acc=0x4A, dp=0x46, abs=0x4E, dpx=0x56, abx=0x5E),
    'ror': dict(acc=0x6A, dp=0x66, abs=0x6E, dpx=0x76, abx=0x7E),
    'inc': dict(acc=0x1A, dp=0xE6, abs=0xEE, dpx=0xF6, abx=0xFE),
    'dec': dict(acc=0x3A, dp=0xC6, abs=0xCE, dpx=0xD6, abx=0xDE),
    'ldx': dict(imm=0xA2, dp=0xA6, abs=0xAE, dpy=0xB6, aby=0xBE),
    'ldy': dict(imm=0xA0, dp=0xA4, abs=0xAC, dpx=0xB4, abx=0xBC),
    'stx': dict(dp=0x86, abs=0x8E, dpy=0x96),
    'sty': dict(dp=0x84, abs=0x8C, dpx=0x94),
    'stz': dict(dp=0x64, dpx=0x74, abs=0x9C, abx=0x9E),
    'cpx': dict(imm=0xE0, dp=0xE4, abs=0xEC),
    'cpy': dict(imm=0xC0, dp=0xC4, abs=0xCC),
    'bit': dict(imm=0x89, dp=0x24, abs=0x2C, dpx=0x34, abx=0x3C),
    'jmp': dict(abs=0x4C, lng=0x5C, ind=0x6C),
    'jsr': dict(abs=0x20), 'jsl': dict(lng=0x22),
    'bpl': dict(rel=0x10), 'bmi': dict(rel=0x30), 'bvc': dict(rel=0x50), 'bvs': dict(rel=0x70),
    'bra': dict(rel=0x80), 'bcc': dict(rel=0x90), 'bcs': dict(rel=0xB0), 'bne': dict(rel=0xD0),
    'beq': dict(rel=0xF0), 'brl': dict(rell=0x82),
    'rep': dict(imm8=0xC2), 'sep': dict(imm8=0xE2),
    'mvn': dict(mv=0x54), 'mvp': dict(mv=0x44),
})
for _mn, _op in dict(clc=0x18, sec=0x38, cli=0x58, sei=0x78, cld=0xD8, sed=0xF8, clv=0xB8,
                     inx=0xE8, iny=0xC8, dex=0xCA, dey=0x88, tax=0xAA, tay=0xA8, txa=0x8A,
                     tya=0x98, txs=0x9A, tsx=0xBA, txy=0x9B, tyx=0xBB, tcs=0x1B, tcd=0x5B,
                     tsc=0x3B, tdc=0x7B, pha=0x48, pla=0x68, phx=0xDA, plx=0xFA, phy=0x5A,
                     ply=0x7A, php=0x08, plp=0x28, phb=0x8B, plb=0xAB, phd=0x0B, pld=0x2B,
                     phk=0x4B, xce=0xFB, xba=0xEB, nop=0xEA, wai=0xCB, rts=0x60, rtl=0x6B,
                     rti=0x40).items():
    OPS16[_mn] = dict(imp=_op)
_SZ16 = dict(imp=1, acc=1, imm=2, dp=2, dpx=2, dpy=2, dix=2, di=2, diy=2, dil=2, dily=2, abs=3,
             abx=3, aby=3, ind=3, lng=4, lngx=4, rel=2, rell=3, mv=3, imm8=2)
_M_OPS = {'lda', 'adc', 'sbc', 'cmp', 'and', 'ora', 'eor', 'bit'}
_X_OPS = {'ldx', 'ldy', 'cpx', 'cpy'}


class A816(_Asm):
    def __init__(self, org, syms):
        super().__init__(org, syms)
        self.m = 1                          # 1 = 8-bit accumulator
        self.x = 1                          # 1 = 8-bit index registers

    def assume(self, m, x):
        self.m, self.x = m, x

    def ins(self, mn, op=None):
        modes = OPS16[mn]
        if mn in ('mvn', 'mvp'):
            self.items.append(('ins', mn, 'mv', op, modes['mv'], 3))
            return
        mode, expr = None, op
        if op is None:
            mode = 'imp' if 'imp' in modes else 'acc'
        elif isinstance(op, str) and op.lower() == 'a':
            mode = 'acc'
        elif isinstance(op, str) and op.startswith('#'):
            expr = op[1:]
            if mn in ('rep', 'sep'):
                mode = 'imm8'
                v = self._eval(expr, {})
                if mn == 'rep':
                    if v & 0x20: self.m = 0
                    if v & 0x10: self.x = 0
                else:
                    if v & 0x20: self.m = 1
                    if v & 0x10: self.x = 1
            else:
                mode = 'imm'
        elif isinstance(op, str) and op.startswith('@'):
            mode = 'lngx' if re.search(r',\s*x$', op) else 'lng'
            expr = re.sub(r',\s*x$', '', op[1:])
        elif isinstance(op, str) and op.startswith('['):
            m_ = re.fullmatch(r'\[(.+)\],\s*y', op)
            if m_:
                mode, expr = 'dily', m_.group(1)
            else:
                mode, expr = 'dil', op[1:-1]
        elif isinstance(op, str) and op.startswith('('):
            m_ = re.fullmatch(r'\((.+)\),\s*y', op)
            if m_:
                mode, expr = 'diy', m_.group(1)
            else:
                m_ = re.fullmatch(r'\((.+),\s*x\)', op)
                if m_:
                    mode, expr = 'dix', m_.group(1)
                else:
                    mode, expr = ('ind' if mn == 'jmp' else 'di'), op[1:-1]
        elif 'rel' in modes or 'rell' in modes:
            mode = 'rel' if 'rel' in modes else 'rell'
        else:
            idx = None
            if isinstance(op, str) and re.search(r',\s*[xy]$', op):
                expr, idx = re.split(r',\s*', op)
                idx = idx.lower()
            known = self._known(expr)
            zp = known is not None and 0 <= known < 256
            if idx is None:
                mode = 'dp' if zp and 'dp' in modes else 'abs'
            elif idx == 'x':
                mode = 'dpx' if zp and 'dpx' in modes else 'abx'
            else:
                mode = 'dpy' if zp and 'dpy' in modes else 'aby'
        if mode not in modes:
            raise ValueError('%s: bad mode %s for %r' % (mn, mode, op))
        size = _SZ16[mode]
        if mode == 'imm':
            wide = (self.m == 0) if mn in _M_OPS else (self.x == 0)
            size = 3 if wide else 2
        self.items.append(('ins', mn, mode, expr, modes[mode], size))

    def __getattr__(self, name):
        mn = name.rstrip('_')
        if mn in OPS16:
            return lambda op=None: self.ins(mn, op)
        raise AttributeError(name)

    def assemble(self):
        self._layout()
        out = bytearray()
        pc = self.org
        for it in self.items:
            k = it[0]
            if k == 'ins':
                _, mn, mode, expr, opc, size = it
                out.append(opc)
                if mode in ('imp', 'acc'):
                    pass
                elif mode == 'mv':
                    out += bytes([self._eval(expr[1], self.labels) & 0xFF,
                                  self._eval(expr[0], self.labels) & 0xFF])
                elif mode == 'rel':
                    d = self._eval(expr, self.labels) - (pc + 2)
                    assert -128 <= d <= 127, 'branch to %s out of range (%d)' % (expr, d)
                    out.append(d & 0xFF)
                elif mode == 'rell':
                    d = self._eval(expr, self.labels) - (pc + 3)
                    out += struct.pack('<h', d)
                elif mode == 'imm':
                    e = expr.strip()
                    if e.startswith('<'):
                        v = self._eval(e[1:], self.labels) & 0xFF
                    elif e.startswith('>'):
                        v = (self._eval(e[1:], self.labels) >> 8) & 0xFF
                    else:
                        v = self._eval(e, self.labels)
                    if size == 2:
                        assert -128 <= v <= 255, (mn, expr, v)
                        out.append(v & 0xFF)
                    else:
                        assert -32768 <= v <= 65535, (mn, expr, v)
                        out += struct.pack('<H', v & 0xFFFF)
                elif mode == 'imm8':
                    out.append(self._eval(expr, self.labels) & 0xFF)
                else:
                    v = self._eval(expr, self.labels)
                    if size == 2:
                        assert 0 <= v < 256, (mn, expr, v)
                        out.append(v)
                    elif size == 3:
                        out += struct.pack('<H', v & 0xFFFF)
                    else:
                        out += struct.pack('<I', v & 0xFFFFFF)[:3]
                pc += size
            elif k == 'db':
                out += it[1]; pc += len(it[1])
            elif k == 'align':
                n = (pc + it[1] - 1) // it[1] * it[1]
                out += bytes(n - pc); pc = n
        return bytes(out)


# ---------------------------------------------------------------------------
# SPC700
# ---------------------------------------------------------------------------
# (mnemonic, dst kind, src kind) -> opcode ; operand byte order by kinds
_SPC = {}


def _spc(mn, rows):
    for opc, d, s in rows:
        _SPC[(mn, d, s)] = opc


_spc('mov', [
    (0xE8, 'a', 'imm'), (0xCD, 'x', 'imm'), (0x8D, 'y', 'imm'),
    (0xE4, 'a', 'dp'), (0xF4, 'a', 'dpx'), (0xE5, 'a', 'abs'), (0xF5, 'a', 'absx'),
    (0xF6, 'a', 'absy'), (0xF7, 'a', 'ily'), (0xE7, 'a', 'ilx'), (0xE6, 'a', '(x)'),
    (0xBF, 'a', '(x)+'),
    (0xC4, 'dp', 'a'), (0xD4, 'dpx', 'a'), (0xC5, 'abs', 'a'), (0xD5, 'absx', 'a'),
    (0xD6, 'absy', 'a'), (0xD7, 'ily', 'a'), (0xC7, 'ilx', 'a'), (0xC6, '(x)', 'a'),
    (0xAF, '(x)+', 'a'),
    (0xF8, 'x', 'dp'), (0xE9, 'x', 'abs'), (0xD8, 'dp', 'x'), (0xC9, 'abs', 'x'),
    (0xEB, 'y', 'dp'), (0xEC, 'y', 'abs'), (0xCB, 'dp', 'y'), (0xCC, 'abs', 'y'),
    (0x7D, 'a', 'x'), (0xDD, 'a', 'y'), (0x5D, 'x', 'a'), (0xFD, 'y', 'a'),
    (0x9D, 'x', 'sp'), (0xBD, 'sp', 'x'),
    (0x8F, 'dp', 'imm'), (0xFA, 'dp', 'dp')])
_spc('cmp', [(0x68, 'a', 'imm'), (0x64, 'a', 'dp'), (0x65, 'a', 'abs'), (0x75, 'a', 'absx'),
             (0x76, 'a', 'absy'), (0x66, 'a', '(x)'), (0xC8, 'x', 'imm'), (0x3E, 'x', 'dp'),
             (0xAD, 'y', 'imm'), (0x7E, 'y', 'dp'), (0x78, 'dp', 'imm'), (0x69, 'dp', 'dp')])
_spc('adc', [(0x88, 'a', 'imm'), (0x84, 'a', 'dp'), (0x85, 'a', 'abs'), (0x98, 'dp', 'imm'),
             (0x89, 'dp', 'dp')])
_spc('sbc', [(0xA8, 'a', 'imm'), (0xA4, 'a', 'dp'), (0xA5, 'a', 'abs'), (0xB8, 'dp', 'imm'),
             (0xA9, 'dp', 'dp')])
_spc('and', [(0x28, 'a', 'imm'), (0x24, 'a', 'dp'), (0x25, 'a', 'abs'), (0x38, 'dp', 'imm')])
_spc('or', [(0x08, 'a', 'imm'), (0x04, 'a', 'dp'), (0x05, 'a', 'abs'), (0x18, 'dp', 'imm')])
_spc('eor', [(0x48, 'a', 'imm'), (0x44, 'a', 'dp'), (0x45, 'a', 'abs')])
_spc('movw', [(0xBA, 'ya', 'dp'), (0xDA, 'dp', 'ya')])
_spc('addw', [(0x7A, 'ya', 'dp')])
_spc('subw', [(0x9A, 'ya', 'dp')])
_spc('cmpw', [(0x5A, 'ya', 'dp')])
_SPC1 = {('inc', 'a'): 0xBC, ('inc', 'dp'): 0xAB, ('inc', 'x'): 0x3D, ('inc', 'y'): 0xFC,
         ('dec', 'a'): 0x9C, ('dec', 'dp'): 0x8B, ('dec', 'x'): 0x1D, ('dec', 'y'): 0xDC,
         ('asl', 'a'): 0x1C, ('lsr', 'a'): 0x5C, ('rol', 'a'): 0x3C, ('ror', 'a'): 0x7C,
         ('incw', 'dp'): 0x3A, ('decw', 'dp'): 0x1A,
         ('jmp', 'abs'): 0x5F, ('call', 'abs'): 0x3F,
         ('push', 'a'): 0x2D, ('pop', 'a'): 0xAE, ('push', 'x'): 0x4D, ('pop', 'x'): 0xCE,
         ('push', 'y'): 0x6D, ('pop', 'y'): 0xEE}
_SPC0 = dict(ret=0x6F, reti=0x7F, nop=0x00, clrc=0x60, setc=0x80, mul=0xCF, div=0x9E)
_SPCB = dict(bra=0x2F, beq=0xF0, bne=0xD0, bcs=0xB0, bcc=0x90, bmi=0x30, bpl=0x10,
             bvs=0x70, bvc=0x50)
_SPCSZ = dict(a=0, x=0, y=0, ya=0, sp=0, c=0, **{'(x)': 0, '(x)+': 0})


class ASpc(_Asm):
    @staticmethod
    def _kind(op):
        o = op.strip()
        lo = o.lower()
        if lo in ('a', 'x', 'y', 'ya', 'sp', 'c'):
            return lo, None
        if lo in ('(x)', '(x)+'):
            return lo, None
        if o.startswith('#'):
            return 'imm', o[1:]
        if o.startswith('!'):
            r = o[1:]
            m = re.fullmatch(r'(.+)\+([xy])', r)
            if m:
                return 'abs' + m.group(2), m.group(1)
            return 'abs', r
        m = re.fullmatch(r'\[(.+)\]\+y', o)
        if m:
            return 'ily', m.group(1)
        m = re.fullmatch(r'\[(.+)\+x\]', o)
        if m:
            return 'ilx', m.group(1)
        m = re.fullmatch(r'(.+)\+x', o)
        if m:
            return 'dpx', m.group(1)
        return 'dp', o

    def ins(self, mn, *ops):
        if mn in _SPC0:
            self.items.append(('ins', mn, 'imp', [], _SPC0[mn], 1))
            return
        if mn in _SPCB:
            self.items.append(('ins', mn, 'rel', [ops[0]], _SPCB[mn], 2))
            return
        kinds = [self._kind(o) for o in ops]
        if len(kinds) == 1:
            k, v = kinds[0]
            if mn in ('jmp', 'call'):
                k = 'abs'
                v = ops[0].lstrip('!')
            opc = _SPC1[(mn, k)]
            vals = [] if v is None else [v]
            size = 1 + (2 if k.startswith('abs') else (1 if vals else 0))
            self.items.append(('ins', mn, k, vals, opc, size))
            return
        (dk, dv), (sk, sv) = kinds
        if mn in ('set1', 'clr1'):
            raise ValueError(mn)
        key = (mn, dk, sk)
        if key not in _SPC:
            raise ValueError('bad spc700 form %s %s,%s' % (mn, ops[0], ops[1]))
        opc = _SPC[key]
        vals = [v for v in (dv, sv) if v is not None]
        if (dk, sk) == ('dp', 'imm') or (mn, dk, sk) == ('cmp', 'dp', 'imm') or \
                (dk == 'dp' and sk == 'imm'):
            vals = [sv, dv]                 # immediate first, then dp
        elif (dk, sk) == ('dp', 'dp'):
            vals = [sv, dv]                 # source first
        size = 1 + sum(2 if (dk.startswith('abs') and v is dv) or
                       (sk.startswith('abs') and v is sv) else 1 for v in vals)
        self.items.append(('ins', mn, (dk, sk), vals, opc, size))

    def __getattr__(self, name):
        mn = name.rstrip('_')
        if mn in _SPC0 or mn in _SPCB or mn in {k[0] for k in _SPC} or \
                mn in {k[0] for k in _SPC1}:
            return lambda *ops: self.ins(mn, *ops)
        raise AttributeError(name)

    def assemble(self):
        self._layout()
        out = bytearray()
        pc = self.org
        for it in self.items:
            k = it[0]
            if k == 'ins':
                _, mn, kind, vals, opc, size = it
                out.append(opc)
                if kind == 'rel':
                    d = self._eval(vals[0], self.labels) - (pc + 2)
                    assert -128 <= d <= 127, 'spc branch out of range %s %d' % (vals[0], d)
                    out.append(d & 0xFF)
                else:
                    kinds = kind if isinstance(kind, tuple) else (kind,)
                    wide = any(x.startswith('abs') for x in kinds) or mn in ('jmp', 'call')
                    for v in vals:
                        val = self._eval(v, self.labels)
                        if wide:
                            out += struct.pack('<H', val & 0xFFFF)
                        else:
                            assert -128 <= val <= 255, (mn, vals, val)
                            out.append(val & 0xFF)
                pc += size
            elif k == 'db':
                out += it[1]; pc += len(it[1])
            elif k == 'align':
                n = (pc + it[1] - 1) // it[1] * it[1]
                out += bytes(n - pc); pc = n
        return bytes(out)


# ---------------------------------------------------------------------------
# BRR encoder
# ---------------------------------------------------------------------------
def _pred(f, p1, p2):
    if f == 0:
        return 0
    if f == 1:
        return p1 + ((-p1) >> 4)
    if f == 2:
        return (p1 << 1) + ((-((p1 << 1) + p1)) >> 5) - p2 + (p2 >> 4)
    return (p1 << 1) + ((-(p1 + (p1 << 2) + (p1 << 3))) >> 6) - p2 + (((p2 << 1) + p2) >> 4)


def brr_encode(pcm):
    """pcm: ints in the S-DSP's 15-bit domain (-16384..16383) -> BRR bytes."""
    n = len(pcm)
    out = bytearray()
    p1 = p2 = 0
    for b in range(0, n, 16):
        blk = pcm[b:b + 16]
        if len(blk) < 16:
            blk = blk + [0] * (16 - len(blk))
        # open-loop choice of filter on the ideal signal
        best = None
        for f in (0, 1, 2, 3):
            q1, q2 = p1, p2
            cost = 0
            mx = 0
            for t in blk:
                d = t - _pred(f, q1, q2)
                if d < 0:
                    d = -d
                cost += d
                if d > mx:
                    mx = d
                q2, q1 = q1, t
            key = (mx, cost)
            if best is None or key < best[0]:
                best = (key, f)
        f = best[1]
        mx = best[0][0]
        r = 0
        while r < 12 and mx > (7 << r >> 1) + (1 << r >> 2 if r else 0):
            r += 1
        for _ in range(2):
            q1, q2 = p1, p2
            nib = []
            clipped = False
            for t in blk:
                pr = _pred(f, q1, q2)
                d = t - pr
                if r:
                    nn = int(math.floor(d / float(1 << (r - 1)) + 0.5))
                else:
                    nn = int(math.floor(d * 2.0 + 0.5))
                if nn > 7:
                    nn = 7
                    if d > (7 << r >> 1) + (1 << r):
                        clipped = True
                elif nn < -8:
                    nn = -8
                    if d < (-8 << r >> 1) - (1 << r):
                        clipped = True
                s = (nn << r) >> 1
                v = pr + s
                if v > 32767:
                    v = 32767
                elif v < -32768:
                    v = -32768
                v = ((v + 16384) & 0x7FFF) - 16384          # 15-bit wrap
                nib.append(nn & 15)
                q2, q1 = q1, v
            if clipped and r < 12:
                r += 1
                continue
            break
        out.append((r << 4) | (f << 2))
        for i in range(0, 16, 2):
            out.append((nib[i] << 4) | nib[i + 1])
        p1, p2 = q1, q2
    return bytes(out)


def brr_decode(data):
    """Reference decoder (15-bit domain), used by --wav and tests."""
    out = []
    p1 = p2 = 0
    for b in range(0, len(data) - 8, 9):
        h = data[b]
        r, f = h >> 4, (h >> 2) & 3
        for i in range(8):
            for nn in (data[b + 1 + i] >> 4, data[b + 1 + i] & 15):
                if nn > 7:
                    nn -= 16
                s = (nn << r) >> 1
                v = s + _pred(f, p1, p2)
                v = max(-32768, min(32767, v))
                v = ((v + 16384) & 0x7FFF) - 16384
                out.append(v)
                p2, p1 = p1, v
    return out


# ---------------------------------------------------------------------------
# SPC700 driver
# ---------------------------------------------------------------------------
DSP_REGS = dict(MVOLL=0x0C, MVOLR=0x1C, EVOLL=0x2C, EVOLR=0x3C, KON=0x4C, KOFF=0x5C,
                FLG=0x6C, EFB=0x0D, PMON=0x2D, NON=0x3D, EON=0x4D, DIR=0x5D, ESA=0x6D,
                EDL=0x7D)


def build_spc(t0_div, pitch, vis_fps=VIS_FPS):
    ring_end = RING + 9 * RING_BLOCKS
    last_blk = RING + 9 * (RING_BLOCKS - 1)
    limit = RING_BLOCKS - 4
    visinc = vis_fps * t0_div
    a = ASpc(SPC_LOAD + 4, {})

    def dsp(reg, val):
        a.mov('$f2', '#%d' % reg)
        a.mov('$f3', '#%d' % val)

    def dspv(reg, expr):
        a.mov('$f2', '#%d' % reg)
        a.mov('$f3', '#%s' % expr)

    a.lbl('start')
    a.mov('x', '#$40'); a.mov('a', '#0')
    a.lbl('clr')
    a.dec('x'); a.mov('(x)', 'a'); a.bne('clr')
    for off, val in ((0x12, limit), (0x14, ring_end), (0x16, last_blk), (0x18, 8000),
                     (0x1A, visinc), (0x00, RING), (0x20, 3), (0x22, RING), (0x24, 9)):
        a.mov('$%02x' % off, '#%d' % (val & 0xFF))
        a.mov('$%02x' % (off + 1), '#%d' % (val >> 8))
    dsp(0x6C, 0x20)
    for reg in ('MVOLL', 'MVOLR'):
        dsp(DSP_REGS[reg], 0x70)
    for reg in ('EVOLL', 'EVOLR', 'EFB', 'PMON', 'NON', 'EON', 'KON', 'EDL', 'ESA'):
        dsp(DSP_REGS[reg], 0)
    dsp(DSP_REGS['KOFF'], 1)
    dsp(DSP_REGS['DIR'], SPC_LOAD >> 8)
    dsp(0x00, 0x7F); dsp(0x01, 0x7F)                       # voice 0 volume
    dsp(0x04, 0); dsp(0x05, 0); dsp(0x06, 0); dsp(0x07, 0x7F)
    dsp(0x02, pitch & 0xFF); dsp(0x03, pitch >> 8)
    a.mov('$fa', '#%d' % (t0_div & 0xFF))
    a.mov('$f1', '#0')
    a.mov('a', '$f4'); a.mov('$07', 'a')       # ignore whatever the boot ROM left in port 0
    a.mov('$f4', '#0'); a.mov('$f5', '#0'); a.mov('$f6', '#0')
    a.call('!updstat')

    # ---- main loop: transfers first (acknowledged as early as possible)
    a.lbl('main')
    a.mov('a', '$f4')
    a.cmp('a', '$07')
    a.bne('xfer')
    a.mov('a', '$fd')
    a.beq('main')
    a.mov('x', 'a')
    a.call('!tick')
    a.bra('main')
    a.lbl('xfer')
    a.mov('$1f', 'a')
    a.and_('a', '#$80')
    a.bne('docmd')
    a.mov('y', '$02')                                  # byte offset in the block: 0, 3, 6
    a.mov('a', '$f5'); a.mov('[$00]+y', 'a'); a.inc('y')
    a.mov('a', '$f6'); a.mov('[$00]+y', 'a'); a.inc('y')
    a.mov('a', '$f7'); a.mov('[$00]+y', 'a')
    a.mov('a', '$1f'); a.mov('$f4', 'a'); a.mov('$07', 'a')      # ack: ports are free again
    a.mov('a', '$02'); a.clrc(); a.adc('a', '#3'); a.mov('$02', 'a')
    a.cmp('a', '#9'); a.beq('blk')
    a.bra('main')
    a.lbl('blk')
    a.mov('$02', '#0')
    a.movw('ya', '$00'); a.cmpw('ya', '$16'); a.bne('nl')
    a.mov('y', '#0'); a.mov('a', '[$00]+y'); a.or_('a', '#3'); a.mov('[$00]+y', 'a')
    a.movw('ya', '$00')
    a.lbl('nl')
    a.addw('ya', '$24'); a.cmpw('ya', '$14'); a.bne('w_ok')
    a.movw('ya', '$22')
    a.lbl('w_ok')
    a.movw('$00', 'ya'); a.incw('$03')
    a.mov('a', '$fd'); a.beq('b_nt')
    a.mov('x', 'a'); a.call('!tick')
    a.lbl('b_nt')
    a.call('!updstat')
    a.bra('main')

    # ---- commands
    a.lbl('docmd')
    a.mov('a', '$f5')
    for cid, name in enumerate(('c_reset', 'c_play', 'c_pause', 'c_resume', 'c_end', 'c_wrap', 'c_seek')):
        a.cmp('a', '#%d' % cid)
        a.beq(name + '_j')
    a.jmp('!c_done')
    for name in ('c_reset', 'c_play', 'c_pause', 'c_resume', 'c_end', 'c_wrap', 'c_seek'):
        a.lbl(name + '_j')
        a.jmp('!' + name)
    a.lbl('c_reset')
    a.call('!reset_body')
    a.jmp('!c_done')
    a.lbl('c_seek')                                    # ports 2,3 = new frame counter
    a.mov('$30', '$f6'); a.mov('$31', '$f7')
    a.call('!reset_body')
    a.mov('$0a', '$30'); a.mov('$0b', '$31')
    a.mov('$f5', '$30'); a.mov('$f6', '$31')
    a.jmp('!c_done')
    a.lbl('c_play')
    dsp(DSP_REGS['KOFF'], 0); dsp(DSP_REGS['KON'], 1)
    a.mov('$0c', '#1'); a.mov('$f1', '#1')
    a.jmp('!c_done')
    a.lbl('c_pause')
    a.mov('$0c', '#0'); a.mov('$f1', '#0')
    dsp(0x02, 0); dsp(0x03, 0)
    a.jmp('!c_done')
    a.lbl('c_resume')
    dsp(0x02, pitch & 0xFF); dsp(0x03, pitch >> 8)
    a.mov('$0c', '#1'); a.mov('$f1', '#1')
    a.jmp('!c_done')
    a.lbl('c_end')
    a.mov('$0d', '#1'); a.movw('ya', '$03'); a.movw('$0e', 'ya')
    a.jmp('!c_done')
    a.lbl('c_wrap')
    a.movw('ya', '$03'); a.movw('$1c', 'ya'); a.mov('$1e', '#1')
    a.lbl('c_done')
    a.call('!updstat')
    a.mov('a', '$1f'); a.mov('$f4', 'a'); a.mov('$07', 'a')
    a.jmp('!main')

    # ---- stop the voice and clear all playback state
    a.lbl('reset_body')
    dsp(DSP_REGS['KOFF'], 1)
    a.mov('$f1', '#0')
    a.mov('x', '#$12'); a.mov('a', '#0')
    a.lbl('cr_l')
    a.dec('x'); a.mov('(x)', 'a'); a.cmp('x', '#2'); a.bne('cr_l')
    a.mov('$1e', '#0')
    a.mov('$00', '#%d' % (RING & 0xFF)); a.mov('$01', '#%d' % (RING >> 8))
    dsp(0x02, pitch & 0xFF); dsp(0x03, pitch >> 8)
    a.mov('$f5', '#0'); a.mov('$f6', '#0')
    a.ret()

    # ---- timer ticks (X = count): one tick = one BRR block played
    a.lbl('tick')
    a.lbl('t_l')
    a.mov('a', '$0c'); a.beq('t_n')
    a.incw('$05')
    a.mov('a', '$1e'); a.beq('t_nw')
    a.movw('ya', '$05'); a.cmpw('ya', '$1c'); a.bne('t_nw')
    a.mov('a', '#0'); a.mov('$0a', 'a'); a.mov('$0b', 'a')
    a.mov('$08', 'a'); a.mov('$09', 'a'); a.mov('$1e', 'a')
    a.lbl('t_nw')
    a.movw('ya', '$08'); a.addw('ya', '$1a'); a.cmpw('ya', '$18'); a.bcc('t_vs')
    a.subw('ya', '$18'); a.incw('$0a')
    a.lbl('t_vs')
    a.movw('$08', 'ya')
    a.lbl('t_n')
    a.dec('x'); a.bne('t_l')
    a.movw('ya', '$0a'); a.movw('$f5', 'ya')
    a.mov('a', '$0d'); a.beq('t_done')
    a.movw('ya', '$0e'); a.subw('ya', '$05')         # blocks left, as a signed 16-bit count
    a.beq('t_end'); a.bmi('t_end'); a.bra('t_done')
    a.lbl('t_end')
    a.mov('$10', '#1'); a.mov('$0d', '#0'); a.mov('$0c', '#0'); a.mov('$f1', '#0')
    dsp(DSP_REGS['KOFF'], 1)
    a.lbl('t_done')
    a.call('!updstat')
    a.ret()

    # ---- status port: bit7 ready, bit1 ring has room, bit0 ended
    a.lbl('updstat')
    a.movw('ya', '$03'); a.subw('ya', '$05'); a.bmi('us_room')       # negative = underrun
    a.cmpw('ya', '$12'); a.bcs('us_full')
    a.lbl('us_room')
    a.mov('a', '#$82'); a.or_('a', '$10'); a.mov('$f7', 'a'); a.ret()
    a.lbl('us_full')
    a.mov('a', '$10'); a.or_('a', '#$80'); a.mov('$f7', 'a'); a.ret()

    code = a.assemble()
    a.labels  # noqa
    entry = SPC_LOAD + 4
    dirent = struct.pack('<HH', RING, RING)
    return dirent + code, entry


# ---------------------------------------------------------------------------
# 65816 program
# ---------------------------------------------------------------------------
ZP16 = dict(PTR=0x00, SEQ=0x03, LASTSEQ=0x04, STATE=0x05, LOOPF=0x06, VISF=0x07,
            STREAM=0x08, NMIF=0x09, PENDING=0x0A, JOY=0x0C, PREV=0x0E, PRESS=0x10,
            FRAME=0x12, SEC=0x14, LASTSEC=0x16, PX=0x18, ACC=0x1A, MM=0x1C, SS=0x1D,
            DMT=0x1E, DMO=0x1F, DST=0x20, DSO=0x21, TMP=0x22, CCOL=0x24, FRAC=0x25,
            VMASK=0x26, TMPB=0x27, CNT=0x28, PEND=0x29, FRT=0x2A, TMP2=0x2C)
BUFB = 0x0800

ROW_HEAD1, ROW_HEAD2, ROW_TITLE, ROW_INFO = 2, 3, 6, 7
ROW_VIS, COL_VIS = 8, 8
ROW_BAR, COL_BAR, BAR_LEN = 17, 2, 28
ROW_TIME, COL_TIME = 19, 9
ROW_STATE, COL_STATE = 21, 12
ROW_FLAGS, COL_LOOP, COL_VISF = 23, 5, 19
T_PART0, T_FULL, T_LED, T_SPACE = 1, 9, 16, 32


def cell(row, col):
    return BUFB + (row * 32 + col) * 2


def build_code(P):
    """Assemble the 65816 program and its data tables (loaded at $00:8000)."""
    syms = dict(ZP16)
    syms.update(BUFB=BUFB, VISBASE=BANK_VIS << 16, STARTLO=0x0000, STARTBANK=BANK_STREAM,
                ENDLO=P['end'] & 0xFFFF, ENDBANK=BANK_STREAM + (P['end'] >> 16),
                TOTAL=P['total_sec'], VFPS=P['vis_fps'], BPSB=P['bytes_per_sec'])
    a = A816(0x8000, syms)
    nop8 = lambda: [a.nop() for _ in range(8)]

    def dma(mode, dest, src, length):
        a.lda('#%d' % mode); a.sta('$4300')
        a.lda('#%d' % dest); a.sta('$4301')
        a.ldx('#%s' % src); a.stx('$4302')
        a.lda('#0'); a.sta('$4304')
        a.ldx('#%d' % length); a.stx('$4305')
        a.lda('#1'); a.sta('$420B')

    # ---- reset
    a.lbl('reset')
    a.sei(); a.clc(); a.xce()
    a.rep('#$38')
    a.ldx('#$1FFF'); a.txs()
    a.lda('#$0000'); a.tcd()
    a.sep('#$20')
    a.lda('#$00'); a.pha(); a.plb()
    a.lda('#$8F'); a.sta('$2100')
    for r in range(0x2101, 0x2134):
        a.stz('$%04X' % r)
        if 0x210D <= r <= 0x2114 or 0x211B <= r <= 0x2120:
            a.stz('$%04X' % r)
    for r in range(0x4200, 0x420E):
        if r != 0x4201:
            a.stz('$%04X' % r)
    a.lda('#$01'); a.sta('$420D')                  # FastROM: run from bank $80
    a.jmp('@fast+$800000')
    a.lbl('fast')
    a.lda('#$80'); a.pha(); a.plb()
    a.lda('#$80'); a.sta('$2115')
    a.stz('$2105')
    a.lda('#$00'); a.sta('$2107')
    a.lda('#$01'); a.sta('$210B')
    a.lda('#$01'); a.sta('$212C')
    a.ldx('#$1000'); a.stx('$2116')
    dma(1, 0x18, 'CHR', 2048)
    a.stz('$2121')
    dma(0, 0x22, 'PAL', 32)
    a.ldx('#0'); a.stx('$2116')
    dma(1, 0x18, 'TILEMAP', 2048)
    a.ldx('#0')
    a.lbl('cp')
    a.lda('TILEMAP,x'); a.sta('$0800,x'); a.inx(); a.cpx('#2048'); a.bne('cp')

    # ---- upload the SPC700 program through the IPL
    a.lbl('w1'); a.lda('$2140'); a.cmp('#$AA'); a.bne('w1')
    a.lbl('w1b'); a.lda('$2141'); a.cmp('#$BB'); a.bne('w1b')
    a.ldx('#%d' % SPC_LOAD); a.stx('$2142')
    a.lda('#1'); a.sta('$2141')
    a.lda('#$CC'); a.sta('$2140')
    a.lbl('w2'); a.lda('$2140'); a.cmp('#$CC'); a.bne('w2')
    a.ldx('#0')
    a.lbl('up')
    a.lda('SPCIMG,x'); a.sta('$2141'); a.txa(); a.sta('$2140')
    a.lbl('w3'); a.cmp('$2140'); a.bne('w3')
    a.inx(); a.cpx('#%d' % P['spc_len']); a.bne('up')
    a.ldx('#%d' % P['spc_entry']); a.stx('$2142')
    a.stz('$2141')
    a.lda('#%d' % ((P['spc_len'] + 1) & 0xFF)); a.sta('$2140')
    a.lbl('rdy'); a.lda('$2143'); a.bpl('rdy')

    # ---- state
    a.rep('#$20'); a.lda('#STARTLO'); a.sta('PTR'); a.sep('#$20')
    a.lda('#STARTBANK'); a.sta('PTR+2')
    a.lda('#1'); a.sta('STREAM'); a.sta('VISF')
    a.stz('SEQ'); a.stz('LASTSEQ'); a.stz('STATE'); a.stz('LOOPF'); a.stz('NMIF')
    a.stz('PENDING')
    a.rep('#$20'); a.stz('PREV'); a.stz('LASTSEC'); a.stz('PX'); a.stz('ACC'); a.sep('#$20')
    a.lda('#$81'); a.sta('$4200')
    a.lda('#$0F'); a.sta('$2100')
    a.stz('PEND')
    if P['autoplay']:
        a.lda('#1'); a.sta('PEND')
    a.cli()

    a.lbl('main')
    a.jsr('feed')
    a.lda('PEND'); a.beq('m_nop')                    # start once the ring is full
    a.lda('$2143'); a.and_('#$02'); a.bne('m_nop')
    a.stz('PEND')
    a.lda('#1'); a.jsr('send_cmd'); a.lda('#1'); a.sta('STATE')
    a.lbl('m_nop')
    a.lda('NMIF'); a.beq('main')
    a.stz('NMIF')
    a.jsr('ui')
    a.bra('main')

    # ---- NMI: copy the tile map to VRAM, flag the main loop
    a.lbl('nmi')
    a.rep('#$30'); a.pha(); a.phx(); a.phy()
    a.sep('#$20')
    a.lda('$4210')
    a.lda('PENDING'); a.beq('nm_skip')
    a.ldx('#%d' % (ROW_VIS * 32)); a.stx('$2116')
    a.lda('#1'); a.sta('$4300')
    a.lda('#$18'); a.sta('$4301')
    a.ldx('#%d' % (BUFB + ROW_VIS * 64)); a.stx('$4302')
    a.stz('$4304')
    a.ldx('#%d' % ((ROW_FLAGS - ROW_VIS + 1) * 64)); a.stx('$4305')
    a.lda('#1'); a.sta('$420B')
    a.stz('PENDING')
    a.lbl('nm_skip')
    a.lda('#1'); a.sta('NMIF')
    a.rep('#$30'); a.ply(); a.plx(); a.pla()
    a.rti()
    a.lbl('irq')
    a.rti()

    # ---- send_cmd: A = command id
    a.lbl('send_cmd')
    a.assume(1, 0)
    a.pha()
    a.lbl('sc_w'); a.lda('$2140'); a.cmp('LASTSEQ'); a.bne('sc_w')
    a.pla(); a.sta('$2141')
    a.lda('SEQ'); a.inc('a'); a.and_('#$7F'); a.sta('SEQ')
    a.ora('#$80'); a.sta('$2140'); a.sta('LASTSEQ')
    a.lda('SEQ')
    a.rts()

    # ---- feed: move one 3-byte group of BRR data to the SPC700 when it is ready
    a.lbl('feed')
    a.assume(1, 0)
    a.lda('STREAM'); a.beq('f_ret')
    a.lda('$2140'); a.cmp('LASTSEQ'); a.bne('f_ret')
    a.lda('STREAM'); a.cmp('#2'); a.beq('f_sendend')
    a.lda('$2143'); a.and_('#$02'); a.beq('f_ret')
    a.lda('[PTR]'); a.sta('$2141')
    a.ldy('#1'); a.lda('[PTR],y'); a.sta('$2142')
    a.iny(); a.lda('[PTR],y'); a.sta('$2143')
    a.lda('SEQ'); a.inc('a'); a.and_('#$7F'); a.sta('SEQ')
    a.sta('$2140'); a.sta('LASTSEQ')
    a.lda('PTR'); a.clc(); a.adc('#3'); a.sta('PTR'); a.bcc('f_chk')
    a.inc('PTR+1'); a.bne('f_chk'); a.inc('PTR+2')
    a.lbl('f_chk')
    a.lda('PTR+2'); a.cmp('#ENDBANK'); a.bcc('f_ret'); a.bne('f_end')
    a.rep('#$20'); a.lda('PTR'); a.cmp('#ENDLO'); a.sep('#$20'); a.bcc('f_ret')
    a.lbl('f_end')
    a.lda('LOOPF'); a.beq('f_setend')
    a.rep('#$20'); a.lda('#STARTLO'); a.sta('PTR'); a.sep('#$20')
    a.lda('#STARTBANK'); a.sta('PTR+2')
    a.lda('#5'); a.jsr('send_cmd')
    a.rts()
    a.lbl('f_setend')
    a.lda('#2'); a.sta('STREAM')
    a.lbl('f_ret')
    a.rts()
    a.lbl('f_sendend')
    a.lda('#4'); a.jsr('send_cmd')
    a.stz('STREAM')
    a.rts()

    # ---- stop and rewind
    a.lbl('do_stop')
    a.assume(1, 0)
    a.lda('#0'); a.jsr('send_cmd')
    a.stz('STATE'); a.stz('PEND')
    a.rep('#$20'); a.lda('#STARTLO'); a.sta('PTR'); a.sep('#$20')
    a.lda('#STARTBANK'); a.sta('PTR+2')
    a.lda('#1'); a.sta('STREAM')
    a.rts()

    # ---- do_seek: TMP = target second (16-bit)
    a.lbl('do_seek')
    a.assume(1, 0)
    a.rep('#$20')
    a.stz('TMP2')
    a.lda('#STARTLO'); a.sta('PTR')
    a.sep('#$20'); a.lda('#STARTBANK'); a.sta('PTR+2'); a.rep('#$20')
    a.ldx('TMP'); a.beq('sk_done')
    a.lbl('sk_l')
    a.lda('TMP2'); a.clc(); a.adc('#VFPS'); a.sta('TMP2')
    a.lda('PTR'); a.clc(); a.adc('#BPSB'); a.sta('PTR')
    a.bcc('sk_nc')
    a.sep('#$20'); a.inc('PTR+2'); a.rep('#$20')
    a.lbl('sk_nc')
    a.dex(); a.bne('sk_l')
    a.lbl('sk_done')
    a.sep('#$20')
    a.lbl('sk_w'); a.lda('$2140'); a.cmp('LASTSEQ'); a.bne('sk_w')
    a.rep('#$20'); a.lda('TMP2'); a.sta('$2142'); a.sep('#$20')
    a.lda('#6'); a.jsr('send_cmd')
    a.lda('#1'); a.sta('STREAM'); a.sta('PEND')
    a.rts()

    # ---- draw_col: A = height 0..8, Y = byte offset of the column's top cell
    a.lbl('draw_col')
    a.assume(1, 0)
    a.rep('#$20'); a.and_('#$00FF'); a.asl('a'); a.asl('a'); a.asl('a'); a.tax()
    a.sep('#$20')
    for r in range(VIS_ROWS):
        a.lda('COLTAB+%d,x' % r); a.sta('BUFB+%d,y' % (64 * r))
    a.rts()

    # ---- per-frame UI
    a.lbl('ui')
    a.assume(1, 0)
    a.lbl('uw'); a.lda('$4212'); a.and_('#1'); a.bne('uw')
    a.rep('#$20')
    a.lda('$4218'); a.sta('JOY')
    a.lda('PREV'); a.eor('#$FFFF'); a.and_('JOY'); a.sta('PRESS')
    a.lda('JOY'); a.sta('PREV')
    a.sep('#$20')
    # START: play / pause / resume
    a.lda('PRESS+1'); a.and_('#$10'); a.beq('nostart')
    a.lda('PEND'); a.bne('nostart')
    a.lda('STATE'); a.bne('s1')
    a.lda('#1'); a.sta('PEND'); a.bra('nostart')
    a.lbl('s1'); a.cmp('#1'); a.bne('s2')
    a.lda('#2'); a.jsr('send_cmd'); a.lda('#2'); a.sta('STATE'); a.bra('nostart')
    a.lbl('s2'); a.lda('#3'); a.jsr('send_cmd'); a.lda('#1'); a.sta('STATE')
    a.lbl('nostart')
    a.lda('PRESS+1'); a.and_('#$80'); a.beq('nob')           # B = stop
    a.jsr('do_stop')
    a.lbl('nob')
    a.lda('PRESS'); a.and_('#$80'); a.beq('noa')             # A = loop
    a.lda('LOOPF'); a.eor('#1'); a.sta('LOOPF')
    a.lbl('noa')
    a.lda('PRESS+1'); a.and_('#$20'); a.beq('nosel')         # SELECT = visualizer
    a.lda('VISF'); a.eor('#1'); a.sta('VISF')
    a.lbl('nosel')
    a.lda('STATE'); a.beq('noseek')                          # LEFT / RIGHT = seek 5 s
    a.lda('PRESS+1'); a.and_('#$02'); a.beq('nol')
    a.rep('#$20'); a.lda('SEC'); a.sec(); a.sbc('#5'); a.bcs('l_ok')
    a.lda('#0')
    a.lbl('l_ok')
    a.sta('TMP'); a.sep('#$20'); a.jsr('do_seek'); a.bra('noseek')
    a.lbl('nol')
    a.lda('PRESS+1'); a.and_('#$01'); a.beq('noseek')
    a.rep('#$20'); a.lda('SEC'); a.clc(); a.adc('#5'); a.sta('TMP')
    a.inc('a'); a.cmp('#TOTAL'); a.sep('#$20'); a.bcs('noseek')
    a.jsr('do_seek')
    a.lbl('noseek')
    a.lda('$2143'); a.and_('#1'); a.beq('noend')             # playback finished
    a.lda('STATE'); a.beq('noend')
    a.jsr('do_stop')
    a.lbl('noend')

    # elapsed time
    a.rep('#$20')
    a.lda('$2141'); a.sta('FRAME'); a.sta('$4204')
    a.sep('#$20')
    a.lda('#VFPS'); a.sta('$4206')
    nop8()
    a.rep('#$20'); a.lda('$4214'); a.sta('SEC'); a.sta('$4204'); a.sep('#$20')
    a.lda('#60'); a.sta('$4206')
    nop8()
    a.lda('$4214'); a.sta('MM'); a.lda('$4216'); a.sta('SS')
    for src, hi, lo in (('MM', 'DMT', 'DMO'), ('SS', 'DST', 'DSO')):
        a.lda(src); a.sta('$4204'); a.stz('$4205'); a.lda('#10'); a.sta('$4206')
        nop8()
        a.lda('$4214'); a.ora('#$30'); a.sta(hi)
        a.lda('$4216'); a.ora('#$30'); a.sta(lo)
    t = cell(ROW_TIME, COL_TIME)
    for dst, off in (('DMT', 0), ('DMO', 2), ('DST', 6), ('DSO', 8)):
        a.lda(dst); a.sta('%d' % (t + off))

    # state / loop / visualizer text
    def text8(var, tab, row, col):
        a.rep('#$20'); a.lda(var); a.and_('#$00FF'); a.asl('a'); a.asl('a'); a.asl('a')
        a.tax(); a.sep('#$20')
        base = cell(row, col)
        for i in range(8):
            a.lda('%s+%d,x' % (tab, i)); a.sta('%d' % (base + 2 * i))
    text8('STATE', 'LBLTAB', ROW_STATE, COL_STATE)
    text8('LOOPF', 'LFTAB', ROW_FLAGS, COL_LOOP)
    text8('VISF', 'VFTAB', ROW_FLAGS, COL_VISF)

    # progress bar
    a.rep('#$20')
    a.lda('SEC'); a.cmp('LASTSEC'); a.beq('bar_same'); a.bcs('bar_adv')
    a.stz('LASTSEC'); a.stz('PX'); a.stz('ACC')
    a.lbl('bar_adv')
    a.lda('LASTSEC'); a.cmp('SEC'); a.beq('bar_same')
    a.inc('LASTSEC')
    a.lda('ACC'); a.clc(); a.adc('#%d' % (BAR_LEN * 8)); a.sta('ACC')
    a.lbl('bar_c')
    a.lda('ACC'); a.cmp('#TOTAL'); a.bcc('bar_adv')
    a.sbc('#TOTAL'); a.sta('ACC'); a.inc('PX'); a.bra('bar_c')
    a.lbl('bar_same')
    a.lda('PX'); a.cmp('#%d' % (BAR_LEN * 8)); a.bcc('bar_ok')
    a.lda('#%d' % (BAR_LEN * 8 - 1)); a.sta('PX')
    a.lbl('bar_ok')
    a.sep('#$20')
    a.lda('PX'); a.and_('#7'); a.sta('FRAC')
    a.lda('PX'); a.lsr('a'); a.lsr('a'); a.lsr('a'); a.sta('CCOL')
    a.ldx('#0'); a.ldy('#0')
    a.lbl('bl')
    a.txa(); a.cmp('CCOL'); a.bcc('b_full'); a.beq('b_part')
    a.lda('#%d' % T_PART0); a.bra('b_put')
    a.lbl('b_full'); a.lda('#%d' % T_FULL); a.bra('b_put')
    a.lbl('b_part'); a.lda('#%d' % T_PART0); a.clc(); a.adc('FRAC')
    a.lbl('b_put')
    a.sta('%d,y' % cell(ROW_BAR, COL_BAR))
    a.iny(); a.iny(); a.inx(); a.cpx('#%d' % BAR_LEN); a.bne('bl')

    # LED bars (redrawn at 30 fps)
    a.lda('FRT'); a.eor('#1'); a.sta('FRT'); a.bne('vis_go')
    a.jmp('vis_end')
    a.lbl('vis_go')
    a.lda('VISF'); a.beq('vm0')
    a.lda('STATE'); a.beq('vm0')
    a.lda('#$FF'); a.bra('vmd')
    a.lbl('vm0'); a.lda('#0')
    a.lbl('vmd'); a.sta('VMASK')
    a.rep('#$20'); a.lda('FRAME'); a.and_('#$1FFF'); a.asl('a'); a.asl('a')
    a.tax(); a.sep('#$20')
    for k in range(VIS_BARS // 2):
        a.lda('@VISBASE+%d,x' % k); a.sta('TMPB')
        a.phx()
        a.lsr('a'); a.lsr('a'); a.lsr('a'); a.lsr('a'); a.and_('VMASK')
        a.ldy('#%d' % (cell(ROW_VIS, COL_VIS) - BUFB + 4 * (2 * k))); a.jsr('draw_col')
        a.lda('TMPB'); a.and_('#$0F'); a.and_('VMASK')
        a.ldy('#%d' % (cell(ROW_VIS, COL_VIS) - BUFB + 4 * (2 * k + 1))); a.jsr('draw_col')
        a.plx()
    a.lbl('vis_end')
    a.lda('#1'); a.sta('PENDING')
    a.rts()

    # ---- data tables
    a.lbl('LBLTAB'); a.db(b'STOPPED PLAYING PAUSED  ')
    a.lbl('LFTAB'); a.db(b'LOOP OFFLOOP ON ')
    a.lbl('VFTAB'); a.db(b'VIS OFF VIS ON  ')
    a.lbl('COLTAB')
    a.db(bytes(T_LED if h > 7 - r else T_SPACE for h in range(9) for r in range(8)))
    a.lbl('PAL'); a.db(P['pal'])
    a.lbl('CHR'); a.db(P['chr'])
    a.lbl('TILEMAP'); a.db(P['tilemap'])
    a.lbl('SPCIMG'); a.db(P['spc'])
    a.align(1)
    code = a.assemble()
    assert 0x8000 + len(code) <= 0xFFC0, 'code too large: %d' % len(code)
    return code, a.labels


# ---------------------------------------------------------------------------
# graphics
# ---------------------------------------------------------------------------
def make_chr():
    font = sms.load_font()
    tiles = [bytes(16)] * 128

    def t2(p0, p1):
        return bytes(b for r in range(8) for b in (p0[r], p1[r]))
    tl = list(tiles)
    for c in range(32, 128):
        tl[c] = t2(font(c), bytes(8))
    for n in range(8):
        m = (0xFF << (8 - n)) & 0xFF
        p0 = bytes(m if 2 <= r < 6 else 0 for r in range(8))
        p1 = bytes((~m & 0xFF) if 2 <= r < 6 else 0 for r in range(8))
        tl[T_PART0 + n] = t2(p0, p1)
    tl[T_FULL] = t2(bytes(0xFF if 2 <= r < 6 else 0 for r in range(8)), bytes(8))
    tl[T_LED] = t2(bytes(0x7E if 1 <= r < 7 else 0 for r in range(8)), bytes(8))
    return b''.join(tl)


def rgb(r, g, b):
    return r | (g << 5) | (b << 10)


def make_palette():
    cols = [0, rgb(31, 31, 31), rgb(13, 13, 13), 0,       # 0 text / bar
            0, rgb(31, 2, 2), 0, 0,                       # 1 red LEDs
            0, rgb(31, 26, 0), 0, 0,                      # 2 yellow
            0, rgb(4, 28, 4), 0, 0]                       # 3 green
    cols += [0] * (16 - len(cols))
    return b''.join(struct.pack('<H', c) for c in cols + [0] * 16)


def mmss(sec):
    m, s = divmod(int(sec), 60)
    return '%02d:%02d' % (min(m, 99), s)


def make_tilemap(title, total_secs, info, head):
    cells = [T_SPACE] * (32 * 32)

    def put(row, col, s, pal=0):
        for i, ch in enumerate(s):
            cells[row * 32 + col + i] = ord(ch) | (pal << 10)

    def centre(row, s):
        put(row, max(0, (32 - len(s)) // 2), s)
    centre(ROW_HEAD1, head[0])
    centre(ROW_HEAD2, head[1])
    centre(ROW_TITLE, title[:30])
    centre(ROW_INFO, info[:30])
    put(ROW_TIME, COL_TIME, '00:00 / ' + mmss(total_secs))
    put(ROW_STATE, COL_STATE, 'STOPPED ')
    put(ROW_FLAGS, COL_LOOP, 'LOOP OFF')
    put(ROW_FLAGS, COL_VISF, 'VIS ON  ')
    for c in range(BAR_LEN):
        cells[ROW_BAR * 32 + COL_BAR + c] = T_PART0
    centre(25, 'START=PLAY/PAUSE')
    centre(26, 'LEFT/RIGHT=SEEK 5 SEC')
    centre(27, 'B=STOP A=LOOP SELECT=VISUAL')
    for r in range(VIS_ROWS):
        pal = 1 if r < 2 else (2 if r < 4 else 3)
        for c in range(2 * VIS_BARS):
            cells[(ROW_VIS + r) * 32 + COL_VIS + c] = T_SPACE | (pal << 10)
    return b''.join(struct.pack('<H', c) for c in cells)


# ---------------------------------------------------------------------------
# ROM
# ---------------------------------------------------------------------------
def build_rom(stream, vis_rows, title, total_secs, info, t0_div, pitch, vis_fps,
              autoplay=False, head=('SNES AUDIO PLAYER', '')):
    assert len(stream) % 9 == 0
    spc, entry = build_spc(t0_div, pitch, vis_fps)
    spc = bytearray(spc)
    while not 2 <= (len(spc) + 1) & 0xFF < 0x80:    # port-0 value left by the boot ROM
        spc.append(0)
    P = dict(end=len(stream), total_sec=max(1, int(math.ceil(total_secs))), vis_fps=vis_fps,
             bytes_per_sec=9 * int(round(T0_HZ / float(t0_div))),
             spc=bytes(spc), spc_len=len(spc), spc_entry=entry, autoplay=autoplay,
             chr=make_chr(), pal=make_palette(),
             tilemap=make_tilemap(title, total_secs, info, head))
    code, labels = build_code(P)
    size = 0x20000 + len(stream)
    full = 0x80000
    while full < size:
        full *= 2
    assert full <= MAX_ROM, 'ROM too large (%d bytes)' % size
    rom = bytearray(full)
    rom[0x8000:0x8000 + len(code)] = code
    # vis table: 8 bytes per frame, 4-bit heights
    vt = bytearray(VIS_FRAMES * 4)
    for f, row in enumerate(vis_rows[:VIS_FRAMES]):
        for k in range(VIS_BARS // 2):
            vt[f * 4 + k] = (row[2 * k] << 4) | row[2 * k + 1]
    rom[0x10000:0x10000 + len(vt)] = vt
    rom[0x20000:0x20000 + len(stream)] = stream
    h = 0xFFC0
    rom[h:h + 21] = title.upper().encode('ascii', 'replace')[:21].ljust(21)
    rom[h + 0x15] = 0x31                   # HiROM, fast
    rom[h + 0x16] = 0x00
    rom[h + 0x17] = (full // 1024).bit_length() - 1
    rom[h + 0x18] = 0
    rom[h + 0x19] = 0x01
    rom[h + 0x1A] = 0x33
    nmi, irq, rst = labels['nmi'], labels['irq'], labels['reset']
    for off, val in ((0xFFE4, irq), (0xFFE6, irq), (0xFFE8, irq), (0xFFEA, nmi),
                     (0xFFEC, rst), (0xFFEE, irq), (0xFFF4, irq), (0xFFF8, irq),
                     (0xFFFA, nmi), (0xFFFC, rst), (0xFFFE, irq)):
        rom[off:off + 2] = struct.pack('<H', val)
    rom[h + 0x1C:h + 0x20] = b'\x00\x00\x00\x00'
    chk = sum(rom) & 0xFFFF
    rom[h + 0x1C:h + 0x20] = struct.pack('<HH', chk ^ 0xFFFF, chk)
    return bytes(rom), dict(size=full, code=len(code), labels=labels)


def compute_stream(pcm, rate, gain_norm=0.95):
    peak = max(max(pcm), -min(pcm), 1)
    k = min(gain_norm * 32767.0 / peak, 10.0)
    t = [int(round(s * k / 2.0)) for s in pcm]
    t = [max(-16384, min(16383, v)) for v in t]
    return brr_encode(t), t


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('mp3'); ap.add_argument('rom')
    ap.add_argument('--rate', type=int, default=0,
                    help='sample rate in Hz (default: highest that fits; 128000/N for '
                         'integer N, max %d)' % DSP_RATE)
    ap.add_argument('--title'); ap.add_argument('--start', type=float, default=0)
    ap.add_argument('--duration', type=float, default=0)
    ap.add_argument('--gain', type=int, default=0, help='dB of boost before a peak limiter')
    ap.add_argument('--autoplay', action='store_true', help='start playing at power-on')
    ap.add_argument('--wav', help='also write a preview WAV of the encoded audio')
    args = ap.parse_args()

    title = (args.title or os.path.splitext(os.path.basename(args.mp3))[0]).upper()
    title = ''.join(c if 32 <= ord(c) < 127 else ' ' for c in title)

    probe = sms.decode(args.mp3, 8000, args.start, args.duration)
    secs = len(probe) / 8000.0
    if args.rate:
        n = max(4, int(round(128000.0 / args.rate)))
    else:
        n = 4
        while secs * (128000.0 / n / 16) * 9 > MAX_ROM - 0x20000 and n < 255:
            n += 1
    rate = int(round(128000.0 / n))
    pitch = int(round(4096 * (128000.0 / n) / 32000))
    vis_fps = VIS_FPS if secs * VIS_FPS <= VIS_FRAMES else VIS_FPS // 2
    print('rate %d Hz (timer divisor %d, pitch $%04X, %d vis fps)' % (rate, n, pitch, vis_fps))
    pcm = sms.decode(args.mp3, rate, args.start, args.duration, args.gain)
    secs = len(pcm) / float(rate)
    print('%d samples (%.0f s)' % (len(pcm), secs))
    print('encoding BRR ...')
    stream, t15 = compute_stream(list(pcm), rate)
    from audiorom.genesis import compute_vis
    print('computing LED-bar levels ...')
    vis = compute_vis(list(pcm), rate, int(math.ceil(secs)) + 2)
    step = VIS_FPS // vis_fps
    rows = [vis[i:i + VIS_BARS] for i in range(0, len(vis), VIS_BARS)][::step]
    info = '%d HZ  BRR  MONO' % rate
    rom, fi = build_rom(stream, rows, title, secs, info, n, pitch, vis_fps, args.autoplay)
    with open(args.rom, 'wb') as f:
        f.write(rom)
    print('wrote %s (%d KB, %d KB of audio, code %d bytes)' % (
        args.rom, len(rom) // 1024, len(stream) // 1024, fi['code']))
    if args.wav:
        import wave
        dec = brr_decode(stream)
        out = array.array('h', (max(-32768, min(32767, v * 2)) for v in dec))
        w = wave.open(args.wav, 'wb')
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate)
        w.writeframes(out.tobytes()); w.close()


if __name__ == '__main__':
    main()
