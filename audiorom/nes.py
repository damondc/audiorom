#!/usr/bin/env python3
"""nes - turn an audio file into a Nintendo NES audio-player ROM.

The 2A03's DMC channel has a 7-bit output level register ($4011) that the CPU
can write directly, so the ROM streams PCM by writing one sample every N CPU
cycles from a cycle-counted 6502 loop.  Samples live in a UxROM-style (mapper
2) banked cartridge; the player UI is drawn by the same loop, which is kept at
a constant period by slotting a few cycles of UI work between samples during
vblank.  Needs ffmpeg; the 6502 assembler is built in.

    audiorom nes song.mp3 song.nes [--rate 16000] [--title NAME] [--autoplay]

Controls: START play/pause, B stop+rewind, A loop on/off, SELECT visualizer,
LEFT/RIGHT seek 5 s (A and B are swapped relative to the Genesis player, as
on Nintendo pads).
"""
import argparse, array, bisect, math, os, re, struct, sys

from audiorom import sms

CPU_HZ = 1789773            # NTSC 2A03
BANK = 0x4000
MAX_BANKS = 256             # UxROM, 4 MB (NES 2.0 header)
MAX_RATE = 16000            # cycle budget for the UI slots caps the rate
C_MIN = 112                 # CPU cycles per sample at the highest rate
VIS_FPS = 30
VIS_BARS = 8                # bars; each is one tile wide with a blank tile between
VIS_SPAN = 2 * VIS_BARS
VIS_ROWS = 8


# ---------------------------------------------------------------------------
# 6502 assembler with cycle counting
# ---------------------------------------------------------------------------
_R = 'imm zp zpx abs abx aby izy imp acc rel ind izx'.split()
OPS = {
    'lda': dict(imm=(0xA9, 2), zp=(0xA5, 3), zpx=(0xB5, 4), abs=(0xAD, 4), abx=(0xBD, 4), aby=(0xB9, 4), izy=(0xB1, 5), izx=(0xA1, 6)),
    'ldx': dict(imm=(0xA2, 2), zp=(0xA6, 3), abs=(0xAE, 4), aby=(0xBE, 4)),
    'ldy': dict(imm=(0xA0, 2), zp=(0xA4, 3), abs=(0xAC, 4), abx=(0xBC, 4)),
    'sta': dict(zp=(0x85, 3), zpx=(0x95, 4), abs=(0x8D, 4), abx=(0x9D, 5), aby=(0x99, 5), izy=(0x91, 6)),
    'stx': dict(zp=(0x86, 3), abs=(0x8E, 4)),
    'sty': dict(zp=(0x84, 3), abs=(0x8C, 4)),
    'adc': dict(imm=(0x69, 2), zp=(0x65, 3), abs=(0x6D, 4), abx=(0x7D, 4), aby=(0x79, 4)),
    'sbc': dict(imm=(0xE9, 2), zp=(0xE5, 3), abs=(0xED, 4), abx=(0xFD, 4)),
    'and': dict(imm=(0x29, 2), zp=(0x25, 3), abs=(0x2D, 4), abx=(0x3D, 4)),
    'ora': dict(imm=(0x09, 2), zp=(0x05, 3), abs=(0x0D, 4), abx=(0x1D, 4)),
    'eor': dict(imm=(0x49, 2), zp=(0x45, 3), abs=(0x4D, 4), abx=(0x5D, 4)),
    'cmp': dict(imm=(0xC9, 2), zp=(0xC5, 3), abs=(0xCD, 4), abx=(0xDD, 4)),
    'cpx': dict(imm=(0xE0, 2), zp=(0xE4, 3), abs=(0xEC, 4)),
    'cpy': dict(imm=(0xC0, 2), zp=(0xC4, 3), abs=(0xCC, 4)),
    'inc': dict(zp=(0xE6, 5), abs=(0xEE, 6)),
    'dec': dict(zp=(0xC6, 5), abs=(0xCE, 6)),
    'asl': dict(acc=(0x0A, 2), zp=(0x06, 5)),
    'lsr': dict(acc=(0x4A, 2), zp=(0x46, 5)),
    'rol': dict(acc=(0x2A, 2), zp=(0x26, 5)),
    'ror': dict(acc=(0x6A, 2), zp=(0x66, 5)),
    'bit': dict(zp=(0x24, 3), abs=(0x2C, 4)),
    'jmp': dict(abs=(0x4C, 3), ind=(0x6C, 5)),
    'jsr': dict(abs=(0x20, 6)),
    'rts': dict(imp=(0x60, 6)), 'rti': dict(imp=(0x40, 6)),
    'bpl': dict(rel=(0x10, 2)), 'bmi': dict(rel=(0x30, 2)), 'bvc': dict(rel=(0x50, 2)),
    'bvs': dict(rel=(0x70, 2)), 'bcc': dict(rel=(0x90, 2)), 'bcs': dict(rel=(0xB0, 2)),
    'bne': dict(rel=(0xD0, 2)), 'beq': dict(rel=(0xF0, 2)),
    'inx': dict(imp=(0xE8, 2)), 'iny': dict(imp=(0xC8, 2)), 'dex': dict(imp=(0xCA, 2)),
    'dey': dict(imp=(0x88, 2)), 'tax': dict(imp=(0xAA, 2)), 'tay': dict(imp=(0xA8, 2)),
    'txa': dict(imp=(0x8A, 2)), 'tya': dict(imp=(0x98, 2)), 'txs': dict(imp=(0x9A, 2)),
    'tsx': dict(imp=(0xBA, 2)), 'pha': dict(imp=(0x48, 3)), 'pla': dict(imp=(0x68, 4)),
    'clc': dict(imp=(0x18, 2)), 'sec': dict(imp=(0x38, 2)), 'cli': dict(imp=(0x58, 2)),
    'sei': dict(imp=(0x78, 2)), 'cld': dict(imp=(0xD8, 2)), 'nop': dict(imp=(0xEA, 2)),
}
SIZE = dict(imp=1, acc=1, imm=2, zp=2, zpx=2, izx=2, izy=2, rel=2, abs=3, abx=3, aby=3, ind=3)


class A65:
    def __init__(self, org, syms):
        self.org = org
        self.syms = dict(syms)
        self.items = []
        self.uid_n = 0
        self.labels = {}

    # -- building ---------------------------------------------------------
    def uid(self, p='L'):
        self.uid_n += 1
        return '%s_%d' % (p, self.uid_n)

    def lbl(self, name):
        self.items.append(('lbl', name))

    def db(self, data):
        self.items.append(('db', bytes(data)))

    def align(self, n):
        self.items.append(('align', n))

    def _num(self, s):
        return int(s.replace('$', '0x'), 0)

    def _known(self, expr):
        """Value of expr if resolvable now (constants / zero-page names)."""
        try:
            return self._eval(expr, {})
        except Exception:
            return None

    def _eval(self, expr, labels):
        if isinstance(expr, int):
            return expr
        e = expr.strip().replace('$', '0x')
        env = dict(self.syms)
        env.update(labels)
        return int(eval(e, {'__builtins__': {}}, env))

    def ins(self, mn, op=None):
        modes = OPS[mn]
        mode, expr = None, op
        if op is None:
            mode = 'imp' if 'imp' in modes else 'acc'
        elif isinstance(op, str) and op.lower() == 'a':
            mode = 'acc'
        elif isinstance(op, str) and op.startswith('#'):
            mode, expr = 'imm', op[1:]
        elif isinstance(op, str) and op.startswith('('):
            m = re.fullmatch(r'\((.+)\),\s*y', op)
            if m:
                mode, expr = 'izy', m.group(1)
            else:
                m = re.fullmatch(r'\((.+),\s*x\)', op)
                if m:
                    mode, expr = 'izx', m.group(1)
                else:
                    mode, expr = 'ind', op[1:-1]
        elif mn in ('bpl', 'bmi', 'bvc', 'bvs', 'bcc', 'bcs', 'bne', 'beq'):
            mode = 'rel'
        else:
            idx = None
            if isinstance(op, str) and re.search(r',\s*[xy]$', op):
                expr, idx = re.split(r',\s*', op)
                idx = idx.lower()
            known = self._known(expr)
            zp = known is not None and 0 <= known < 256
            if idx is None:
                mode = 'zp' if zp and 'zp' in modes else 'abs'
            elif idx == 'x':
                mode = 'zpx' if zp and 'zpx' in modes else 'abx'
            else:
                mode = 'aby'
        if mode not in modes:
            raise ValueError('%s: bad mode %s for %r' % (mn, mode, op))
        opc, cyc = modes[mode]
        self.items.append(('ins', mn, mode, expr, opc, cyc))

    def __getattr__(self, name):
        mn = name.rstrip('_')
        if mn in OPS:
            return lambda op=None: self.ins(mn, op)
        raise AttributeError(name)

    # -- timing helpers -----------------------------------------------------
    @staticmethod
    def cyc(items):
        return sum(it[5] if it[0] == 'ins' else (it[1] if it[0] == 'adj' else 0)
                   for it in items)

    def capture(self, fn):
        saved, self.items = self.items, []
        fn()
        got, self.items = self.items, saved
        return got

    def pad(self, n):
        if n == 0:
            return
        assert n >= 2, 'cannot pad %d cycle(s)' % n
        if n % 2:
            self.bit_('TMP')
            n -= 3
        for _ in range(n // 2):
            self.nop()

    def if_balanced(self, cond, then_fn, else_fn):
        """Branch on cond to then_fn, else else_fn; both paths take equal cycles."""
        ti, ei = self.capture(then_fn), self.capture(else_fn)
        T, E = self.cyc(ti), self.cyc(ei)
        X = max(3 + T, 2 + E + 3)
        while X - (3 + T) == 1 or X - (2 + E + 3) == 1:
            X += 1
        lt, le = self.uid('t'), self.uid('e')
        start = len(self.items)
        self.ins(cond, lt)
        self.items += ei
        self.pad(X - (2 + E + 3))
        self.jmp_(le)
        self.lbl(lt)
        self.items += ti
        self.pad(X - (3 + T))
        self.lbl(le)
        # only one arm runs: report the path cost, not the sum of both arms
        self.items.append(('adj', X - self.cyc(self.items[start:])))

    # -- assembly -------------------------------------------------------------
    def assemble(self):
        pc = self.org
        for it in self.items:
            if it[0] == 'lbl':
                self.labels[it[1]] = pc
            elif it[0] == 'ins':
                pc += SIZE[it[2]]
            elif it[0] == 'db':
                pc += len(it[1])
            elif it[0] == 'align':
                pc = (pc + it[1] - 1) // it[1] * it[1]
        out = bytearray()
        pc = self.org
        for it in self.items:
            if it[0] == 'ins':
                _, mn, mode, expr, opc, cyc = it
                out.append(opc)
                if mode in ('imp', 'acc'):
                    pass
                elif mode == 'rel':
                    d = self._eval(expr, self.labels) - (pc + 2)
                    assert -128 <= d <= 127, 'branch to %s out of range (%d)' % (expr, d)
                    out.append(d & 0xFF)
                elif mode == 'imm':
                    e = expr.strip()
                    if e.startswith('<'):
                        v = self._eval(e[1:], self.labels) & 0xFF
                    elif e.startswith('>'):
                        v = (self._eval(e[1:], self.labels) >> 8) & 0xFF
                    else:
                        v = self._eval(e, self.labels)
                    assert -128 <= v <= 255, (mn, expr, v)
                    out.append(v & 0xFF)
                else:
                    v = self._eval(expr, self.labels)
                    if SIZE[mode] == 2:
                        assert 0 <= v < 256, (mn, expr, v)
                        out.append(v)
                    else:
                        out += struct.pack('<H', v & 0xFFFF)
                pc += SIZE[mode]
            elif it[0] == 'db':
                out += it[1]; pc += len(it[1])
            elif it[0] == 'align':
                n = (pc + it[1] - 1) // it[1] * it[1]
                out += bytes(n - pc); pc = n
        return bytes(out)


# ---------------------------------------------------------------------------
# memory map
# ---------------------------------------------------------------------------
ZP = {n: i for i, n in enumerate((
    'PTR', 'PTRH', 'STATE', 'NEEDREW', 'RESET', 'LOOP', 'VIS', 'BANK', 'BTN',
    'PREV', 'PRESS', 'D0', 'D1', 'D2', 'D3', 'ACCL', 'ACCH', 'CARRY', 'TMP',
    'TMP2', 'FC', 'MASK', 'BK', 'VPL', 'VPH', 'YS', 'VM', 'ZB',
    'BTN2', 'PREV2', 'PRESS2', 'CAND0', 'CAND1', 'CAND2', 'CAND3'))}
ZP['PTR1'] = ZP['PTRH']

PPUCTRL, PPUMASK, PPUSTATUS, PPUSCROLL, PPUADDR, PPUDATA = (
    0x2000, 0x2001, 0x2002, 0x2005, 0x2006, 0x2007)
DMC_LEVEL = 0x4011

# nametable layout (row, col)
ROW_HEAD1, ROW_HEAD2, ROW_TITLE, ROW_INFO = 2, 3, 6, 7
ROW_VIS, COL_VIS = 8, 8
ROW_BAR, COL_BAR, BAR_LEN = 17, 2, 28
ROW_TIME, COL_TIME = 19, 9
ROW_STATE, COL_STATE = 21, 12
ROW_FLAGS, COL_LOOP, COL_VISF = 23, 5, 19

T_PART0, T_FULL, T_LED, T_SPACE = 1, 9, 16, 32


def nt_addr(row, col):
    return 0x2000 + row * 32 + col


# ---------------------------------------------------------------------------
# graphics
# ---------------------------------------------------------------------------
def make_chr():
    font = sms.load_font()
    pt = bytearray(256 * 16)
    for c in range(32, 128):
        g = font(c)
        pt[c * 16:c * 16 + 8] = g                 # plane 0 only -> colour 1
    for n in range(8):                            # partial progress tiles
        m = (0xFF << (8 - n)) & 0xFF
        t = bytearray(16)
        for r in range(2, 6):
            t[r] = m
            t[8 + r] = ~m & 0xFF                  # colour 2 for the unfilled part
        pt[(T_PART0 + n) * 16:(T_PART0 + n) * 16 + 16] = t
    t = bytearray(16)
    for r in range(2, 6):
        t[r] = 0xFF
    pt[T_FULL * 16:T_FULL * 16 + 16] = t
    t = bytearray(16)
    for r in range(1, 7):
        t[r] = 0x7E
    pt[T_LED * 16:T_LED * 16 + 16] = t
    return bytes(pt) + bytes(0x1000)              # pattern table 1 unused


PALETTE = bytes([0x0F, 0x30, 0x10, 0x00,          # text / progress bar
                 0x0F, 0x16, 0x06, 0x00,          # red LEDs
                 0x0F, 0x28, 0x18, 0x00,          # yellow
                 0x0F, 0x2A, 0x1A, 0x00]) * 2     # green


def mmss(sec):
    m, s = divmod(int(sec), 60)
    return '%02d:%02d' % (min(m, 99), s)


def make_nametable(title, total_secs, info, head):
    nt = bytearray([T_SPACE]) * 960
    attr = bytearray(64)

    def put(row, col, s):
        for i, ch in enumerate(s):
            nt[row * 32 + col + i] = ord(ch)

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
        nt[ROW_BAR * 32 + COL_BAR + c] = T_PART0
    centre(25, 'START=PLAY/PAUSE')
    centre(26, 'LEFT/RIGHT=SEEK 5 SEC')
    centre(27, 'B=STOP A=LOOP SELECT=VISUAL')
    # LED colour zones: attribute quadrants are 2x2 tiles
    for ar in (ROW_VIS // 4, ROW_VIS // 4 + 1):
        for ac in range(COL_VIS // 4, COL_VIS // 4 + VIS_SPAN // 4):
            if ar == ROW_VIS // 4:
                q = (1 << 0 | 1 << 2) | (2 << 4 | 2 << 6)       # rows 0-1 red, 2-3 yellow
            else:
                q = 3 | 3 << 2 | 3 << 4 | 3 << 6                # rows 4-7 green
            attr[ar * 8 + ac] = q
    return bytes(nt) + bytes(attr)


# ---------------------------------------------------------------------------
# audio
# ---------------------------------------------------------------------------
def dmc_levels():
    """Mixer output (0..1) for DMC levels 0..127 with the other channels at 0."""
    def f(d):
        return 0.0 if d == 0 else 159.79 / (1.0 / (d / 22638.0) + 100.0)
    top = f(127)
    return [f(d) / top for d in range(128)]


def quantize(pcm, shape=0.5, norm=0.92):
    """int16 samples -> DMC levels 1..127 (0 is the end marker), error-feedback
    quantised in the mixer's output domain so the DAC curve is linearised."""
    lv = dmc_levels()
    lo, hi = lv[1], lv[127]
    tab = lv[1:]
    peak = max(max(pcm), -min(pcm), 1)
    k = min(norm * 32767.0 / peak, 10.0) / 32767.0
    out = bytearray(len(pcm))
    e = 0.0
    span = hi - lo
    for i, s in enumerate(pcm):
        t = lo + (s * k * 0.5 + 0.5) * span
        v = t - shape * e
        j = bisect.bisect_left(tab, v)
        if j <= 0:
            j = 0
        elif j >= len(tab):
            j = len(tab) - 1
        elif v - tab[j - 1] < tab[j] - v:
            j -= 1
        e = tab[j] - v
        out[i] = j + 1
    return bytes(out)


# ---------------------------------------------------------------------------
# program
# ---------------------------------------------------------------------------
def build_fixed(cyc, sp, vp, nbanks, autoplay, nt, bar_px, vis_on, extra, phases=2):
    """Assemble the fixed 16 KB bank.  cyc = CPU cycles per sample."""
    limit = 0x80 + sp                       # first page past the sample area
    vph0 = limit                            # vis data starts right after samples
    P = cyc - 35                            # spare cycles per slot
    a = A65(0xC000, dict(ZP, LIMIT=limit, VPH0=vph0))
    S = a

    # ---- sample routine (page aligned so taken branches stay in-page)
    a.align(256)
    a.lbl('sample')
    a.lda('STATE'); a.cmp('#1'); a.bne('s_idle')
    a.lda('(PTR),y'); a.beq('s_end')
    a.sta(DMC_LEVEL); a.iny(); a.bne('s_ret')
    a.inc('PTRH')
    a.lda('PTRH'); a.cmp('#LIMIT'); a.bne('s_ret')
    a.inc('BANK'); a.ldx('BANK'); a.lda('BANKTAB,x'); a.sta('BANKTAB,x')
    a.lda('#$80'); a.sta('PTRH')
    a.lda('#0'); a.sta('VPL'); a.lda('#VPH0'); a.sta('VPH')
    a.lbl('s_ret')
    a.rts()
    a.lbl('s_idle')
    a.lda('NEEDREW'); a.bne('s_rew')
    a.pad(10)
    a.rts()
    a.lbl('s_rew')
    a.ldy('#0'); a.sty('PTR'); a.sty('BANK'); a.sty('NEEDREW'); a.sty('VPL')
    a.lda('#$80'); a.sta('PTRH'); a.lda('#VPH0'); a.sta('VPH')
    a.lda('BANKTAB'); a.sta('BANKTAB')
    a.lda('#$40'); a.sta(DMC_LEVEL)
    a.rts()
    a.lbl('s_end')
    a.lda('LOOP'); a.beq('s_stop')
    a.lda('#1'); a.sta('RESET')
    a.jmp('s_rew')
    a.lbl('s_stop')
    a.lda('#0'); a.sta('STATE')
    a.lda('#1'); a.sta('RESET'); a.sta('NEEDREW')
    a.rts()

    # ---- main loop: one sample per `cyc` cycles, poll for vblank
    a.lbl('main')
    a.jsr('sample')
    a.bit_(PPUSTATUS)
    a.bmi('vbl')
    a.pad(cyc - 35 - 9)
    a.jmp('main')
    a.lbl('vbl')
    a.pad(cyc - 35 - 7)

    # ---- UI work units (each runs between two samples)
    def lbl_unit(base_col, tab, var, row):
        hi, lo = nt_addr(row, base_col) >> 8, nt_addr(row, base_col) & 0xFF

        def f():
            a.lda('#%d' % hi); a.sta(PPUADDR)
            a.lda('FC'); a.and_('#7'); a.sta('TMP'); a.clc(); a.adc('#%d' % lo); a.sta(PPUADDR)
            if var == 'STATE':
                a.lda('STATE')
            else:
                a.lda(var)
            a.asl('a'); a.asl('a'); a.asl('a'); a.ora('TMP'); a.tax()
            a.lda('%s,x' % tab); a.sta(PPUDATA)
        return f

    def u_time():
        ad = nt_addr(ROW_TIME, COL_TIME)
        a.lda('#%d' % (ad >> 8)); a.sta(PPUADDR); a.lda('#%d' % (ad & 255)); a.sta(PPUADDR)
        a.lda('D3'); a.sta(PPUDATA); a.lda('D2'); a.sta(PPUDATA)
        a.lda('#%d' % ord(':')); a.sta(PPUDATA)
        a.lda('D1'); a.sta(PPUDATA); a.lda('D0'); a.sta(PPUDATA)

    def u_bar_calc():
        a.ldx('BANK'); a.lda('BARPX,x'); a.sta('TMP')
        a.lsr('a'); a.lsr('a'); a.lsr('a'); a.sta('TMP2')
        a.lda('BK'); a.sec(); a.sbc('TMP2'); a.clc(); a.adc('#32'); a.tax()
        a.lda('TMP'); a.and_('#7'); a.and_('FRACMASK,x'); a.clc(); a.adc('BARTILE,x'); a.sta('TMP')

    def u_bar_put():
        ad = nt_addr(ROW_BAR, COL_BAR)
        a.lda('#%d' % (ad >> 8)); a.sta(PPUADDR)
        a.lda('BK'); a.clc(); a.adc('#%d' % (ad & 255)); a.sta(PPUADDR)
        a.lda('TMP'); a.sta(PPUDATA)

    u_state = lbl_unit(COL_STATE, 'LBLTAB', 'STATE', ROW_STATE)
    u_loop = lbl_unit(COL_LOOP, 'LFTAB', 'LOOP', ROW_FLAGS)
    u_visf = lbl_unit(COL_VISF, 'VFTAB', 'VIS', ROW_FLAGS)

    ppu_units = [u_time, u_bar_calc, u_bar_put, u_state, u_loop, u_visf]

    if vis_on:
        def u_vsetup():
            a.lda('#4'); a.sta(PPUCTRL)

        ppu_units.append(u_vsetup)
        vad = nt_addr(ROW_VIS, COL_VIS)

        ncol = VIS_BARS // phases
        sh_lo = {2: 1, 4: 2, 8: 3}[ncol]      # log2(bars per frame)
        sh_b = sh_lo - 1                      # byte offset per phase = ncol/2

        for j in range(ncol):
            def col_a(j=j):
                a.lda('#%d' % (vad >> 8)); a.sta(PPUADDR)
                a.lda('FC'); a.and_('#%d' % (phases - 1))
                for _ in range(sh_lo + 1):          # bars sit two columns apart
                    a.asl('a')
                a.clc(); a.adc('#%d' % ((vad & 255) + 2 * j)); a.sta(PPUADDR)
                a.lda('FC'); a.and_('#%d' % (phases - 1))
                for _ in range(sh_b):
                    a.asl('a')
                a.clc(); a.adc('#%d' % (j >> 1)); a.sty('YS'); a.tay()
                a.lda('(VPL),y')
                if j % 2 == 0:
                    a.lsr('a'); a.lsr('a'); a.lsr('a'); a.lsr('a')
                else:
                    a.and_('#$0F')
                a.and_('VM'); a.sta('TMP'); a.ldy('YS')

            def col_b(j=j, rows=range(0, 4)):
                a.ldx('TMP')
                for r in rows:
                    a.lda('VCOL+%d,x' % (r * 16)); a.sta(PPUDATA)

            ppu_units += [col_a,
                          lambda j=j: col_b(j, range(0, 4)),
                          lambda j=j: col_b(j, range(4, 8))]

    def u_vend():
        a.lda('#0'); a.sta(PPUSCROLL); a.sta(PPUSCROLL); a.sta(PPUCTRL)

    ppu_units.append(u_vend)

    # logic units (run after the PPU writes, still inside the vblank routine)
    def u_strobe():
        a.lda('#1'); a.sta(0x4016); a.lda('#0'); a.sta(0x4016); a.sta('BTN'); a.sta('BTN2')

    def u_read2():
        for _ in range(2):
            a.lda(0x4016); a.lsr('a'); a.rol('BTN')

    def u_read2b():                     # D-pad: up, down, left, right
        for _ in range(2):
            a.lda(0x4016); a.lsr('a'); a.rol('BTN2')

    def u_press():
        a.lda('PREV'); a.eor('#$FF'); a.and_('BTN'); a.sta('PRESS')
        a.lda('BTN'); a.sta('PREV')
        a.lda('PREV2'); a.eor('#$FF'); a.and_('BTN2'); a.sta('PRESS2')
        a.lda('BTN2'); a.sta('PREV2')

    def u_state_tr():
        a.lda('PRESS'); a.and_('#5'); a.sta('TMP')
        a.lda('STATE'); a.asl('a'); a.asl('a'); a.asl('a'); a.asl('a')
        a.ora('TMP'); a.tax(); a.lda('TRANS,x'); a.sta('STATE')

    def u_flag_a():                     # B = stop and rewind
        a.lda('PRESS'); a.lsr('a'); a.lsr('a'); a.and_('#1'); a.sta('TMP')
        a.ora('NEEDREW'); a.sta('NEEDREW')
        a.lda('TMP'); a.ora('RESET'); a.sta('RESET')

    def u_flag_b():                     # A = loop, SELECT = visualizer
        a.lda('PRESS'); a.lsr('a'); a.lsr('a'); a.lsr('a'); a.and_('#1'); a.eor('LOOP'); a.sta('LOOP')
        a.lda('PRESS'); a.lsr('a'); a.and_('#1'); a.eor('VIS'); a.sta('VIS')

    def u_time():                       # elapsed time follows the current bank
        a.ldx('BANK')
        for d in ('D0', 'D1', 'D2', 'D3'):
            a.lda('T%s,x' % d); a.sta(d)

    def u_cand():                       # candidate banks for a seek
        a.lda('BANK'); a.sta('CAND0'); a.sta('CAND3')
        a.tax(); a.lda('FWD,x'); a.sta('CAND1')
        a.lda('BACK,x'); a.sta('CAND2')

    def u_seekpick():                   # RIGHT = forward, LEFT = back (D-pad bits 0 / 1)
        a.lda('PRESS2'); a.and_('#3'); a.ldx('STATE'); a.and_('SEEKMASK,x'); a.tax()
        a.lda('CAND0,x'); a.sta('TMP')

    def u_seek():
        a.lda('TMP'); a.cmp('BANK')

        def apply():
            a.sta('BANK')
            a.lda('#$80'); a.sta('PTRH'); a.ldy('#0')
            a.ldx('BANK'); a.lda('BANKTAB,x'); a.sta('BANKTAB,x')
            a.lda('#0'); a.sta('VPL'); a.lda('#VPH0'); a.sta('VPH')
        a.if_balanced('beq', lambda: None, apply)

    def u_vm():
        a.lda('VIS'); a.asl('a'); a.asl('a'); a.clc(); a.adc('STATE'); a.tax()
        a.lda('VMT,x'); a.sta('VM')

    def u_vadv():
        a.lda('FC'); a.and_('#1'); a.asl('a'); a.asl('a'); a.clc(); a.adc('STATE'); a.tax()
        a.lda('VADV,x'); a.clc(); a.adc('VPL'); a.sta('VPL')
        a.lda('VPH'); a.adc('#0'); a.sta('VPH')

    def u_fc():
        a.inc('FC'); a.ldx('BK'); a.lda('NEXTK,x'); a.sta('BK')

    logic_units = [u_strobe, u_read2, u_read2, u_read2b, u_read2b, u_press, u_state_tr,
                   u_flag_a, u_flag_b, u_cand, u_seekpick, u_seek, u_time, u_vm, u_vadv, u_fc]

    # ---- pack units into slots of P cycles (last slot also holds the jmp)
    units = ppu_units + logic_units
    costs = [a.cyc(a.capture(f)) for f in units]
    cap = P - 3
    assert max(costs) <= cap, 'unit too long for %d-cycle slots (%d)' % (P, max(costs))
    slots, cur, cs = [], [], 0
    for f, c in zip(units, costs):
        if cs + c > cap:
            slots.append(cur); cur, cs = [], 0
        cur.append((f, c)); cs += c
    slots.append(cur)
    # a remainder of exactly one cycle cannot be padded: spill a unit forward
    for i in range(len(slots)):
        tot = sum(c for _, c in slots[i])
        rem = P - tot - (3 if i == len(slots) - 1 else 0)
        if rem == 1:
            if i == len(slots) - 1:
                slots.append([slots[i].pop()])
            else:
                slots[i + 1].insert(0, slots[i].pop())
    # when does the last PPU write happen?
    npu = len(ppu_units)
    done, n = 0, 0
    for i, s in enumerate(slots):
        n += len(s)
        if n >= npu:
            done = i + 1
            break
    ppu_end = (done + 1) * cyc           # worst case: vblank noticed one sample late
    assert ppu_end < 2273 - 60, 'PPU work does not fit in vblank (%d cycles)' % ppu_end

    for i, s in enumerate(slots):
        a.jsr('sample')
        tot = 0
        for f, c in s:
            f()
            tot += c
        last = i == len(slots) - 1
        a.pad(P - tot - (3 if last else 0))
        if last:
            a.jmp('main')

    # ---- reset / init
    a.lbl('reset')
    a.sei(); a.cld(); a.ldx('#$FF'); a.txs(); a.inx()
    a.stx(PPUCTRL); a.stx(PPUMASK); a.stx(0x4010); a.stx(0x4015)
    a.lda('#$40'); a.sta(0x4017)
    a.bit_(PPUSTATUS)
    a.lbl('w1'); a.bit_(PPUSTATUS); a.bpl('w1')
    a.lda('#0'); a.tax()
    a.lbl('clr')
    for pg in range(0, 8):
        a.sta('$%04X,x' % (pg * 256))
    a.inx(); a.bne('clr')
    a.lbl('w2'); a.bit_(PPUSTATUS); a.bpl('w2')
    a.lda('#$3F'); a.sta(PPUADDR); a.lda('#0'); a.sta(PPUADDR)
    a.ldx('#0')
    a.lbl('pal'); a.lda('PALTAB,x'); a.sta(PPUDATA); a.inx(); a.cpx('#32'); a.bne('pal')
    a.lda('#$20'); a.sta(PPUADDR); a.lda('#0'); a.sta(PPUADDR)
    a.lda('#<NTDATA'); a.sta('TMP'); a.lda('#>NTDATA'); a.sta('TMP2')
    a.ldy('#0'); a.ldx('#4')
    a.lbl('ntl'); a.lda('(TMP),y'); a.sta(PPUDATA); a.iny(); a.bne('ntl')
    a.inc('TMP2'); a.dex(); a.bne('ntl')
    a.lda('#0'); a.sta(PPUSCROLL); a.sta(PPUSCROLL); a.sta(PPUCTRL)
    a.lda('#$0A'); a.sta(PPUMASK)
    # player state
    a.lda('#%d' % (1 if autoplay else 0)); a.sta('STATE')
    a.lda('#1'); a.sta('VIS')
    a.lda('#$30'); a.sta('D0'); a.sta('D1'); a.sta('D2'); a.sta('D3')
    a.lda('#$80'); a.sta('PTRH')
    a.lda('#VPH0'); a.sta('VPH')
    a.lda('BANKTAB'); a.sta('BANKTAB')
    a.lda('#$40'); a.sta(DMC_LEVEL)
    a.ldy('#0')
    a.jmp('main')
    a.lbl('irq'); a.rti()

    # ---- tables (page aligned)
    def tab(name, data, al=256):
        a.align(al)
        a.lbl(name)
        a.db(data)

    tab('BANKTAB', bytes(range(256)))
    for i, d in enumerate(('D0', 'D1', 'D2', 'D3')):
        tab('T' + d, bytes(x['digits'][i] for x in extra['time']) + bytes(256 - len(extra['time'])))
    tab('FWD', bytes(extra['fwd']) + bytes(256 - len(extra['fwd'])))
    tab('BACK', bytes(extra['back']) + bytes(256 - len(extra['back'])))
    tab('SEEKMASK', bytes([0, 3, 3, 0]))
    tab('BARPX', bytes(bar_px) + bytes(256 - len(bar_px)))
    bt = bytes([T_FULL] * 32 + [T_PART0] * 32)
    tab('BARTILE', bt)
    tab('FRACMASK', bytes([0] * 32 + [7] + [0] * 31))
    tr = bytearray(64)
    for s, nxt in enumerate(((0, 1, 0, 0), (1, 2, 0, 0), (2, 1, 0, 0))):
        for ev, v in zip((0, 1, 4, 5), nxt):
            tr[s * 16 + ev] = v
    tab('TRANS', tr)
    tab('NEXTK', bytes([(k + 1) % BAR_LEN for k in range(BAR_LEN)] + [0] * 4))
    tab('VMT', bytes([0, 0, 0, 0, 0, 0xFF, 0xFF, 0]))
    tab('VADV', bytes([0, 0, 0, 0, 0, 4, 0, 0]))       # 30 vis frames/s
    tab('LBLTAB', b'STOPPED PLAYING PAUSED  ')
    tab('LFTAB', b'LOOP OFFLOOP ON ')
    tab('VFTAB', b'VIS OFF VIS ON  ')
    vcol = bytearray(128)
    for r in range(8):
        for h in range(16):
            vcol[r * 16 + h] = T_LED if h >= 8 - r else T_SPACE
    tab('VCOL', vcol)
    tab('PALTAB', PALETTE)
    tab('NTDATA', nt)

    a.align(1)
    code = a.assemble()
    base = a.org
    assert len(code) <= 0x4000 - 6, 'fixed bank overflow: %d' % len(code)
    bank = bytearray(0x4000)
    bank[:len(code)] = code
    vec = struct.pack('<HHH', a.labels['irq'], a.labels['reset'], a.labels['irq'])
    bank[0x4000 - 6:] = vec
    info = dict(slots=len(slots), ppu_end=ppu_end, code=len(code), labels=a.labels)
    return bytes(bank), info


def build_rom(samples, rate_cyc, title, total_secs, info_line, vis, autoplay=False,
              head=('NES AUDIO PLAYER', '')):
    real_rate = CPU_HZ / float(rate_cyc)
    n = len(samples)
    # vis pages per bank: enough 30 fps frames (4 bytes each) for one bank's playtime
    if vis is not None:
        per_bank_nominal = BANK                      # upper bound, refined below
        vp = 0
        for vp in range(1, 8):
            sp = 64 - vp
            secs = sp * 256 / real_rate
            frames = int(math.ceil(secs * VIS_FPS)) + 3
            if frames * 4 <= vp * 256:
                break
        else:
            raise SystemExit('rate too low for visualizer data')
    else:
        vp = 0
    sp = 64 - vp
    per = sp * 256
    nb = (n + 1 + per - 1) // per
    assert nb + 1 <= MAX_BANKS, 'song too long: %d banks' % nb
    tot = 2
    while tot < nb + 1:
        tot *= 2
    prg = bytearray(tot * BANK)
    for b in range(nb):
        chunk = samples[b * per:(b + 1) * per]
        prg[b * BANK:b * BANK + len(chunk)] = chunk
        if vis is not None:
            g0 = int(round(b * per * VIS_FPS / real_rate))
            for f in range(vp * 64):
                g = g0 + f
                row = vis[g * VIS_BARS:(g + 1) * VIS_BARS] if (g + 1) * VIS_BARS <= len(vis) else bytes(VIS_BARS)
                off = b * BANK + per + f * 4
                for k in range(VIS_BARS // 2):
                    prg[off + k] = (row[2 * k] << 4) | row[2 * k + 1]
    bar_px = [min(BAR_LEN * 8 - 1, b * BAR_LEN * 8 // max(nb, 1)) for b in range(nb)]
    nt = make_nametable(title, total_secs, info_line, head)
    step = max(1, int(round(5 * real_rate / per)))           # banks per 5-second seek
    time = []
    for b in range(nb):
        m, s = divmod(int(b * per / real_rate), 60)
        m = min(m, 99)
        time.append(dict(digits=[0x30 + s % 10, 0x30 + s // 10, 0x30 + m % 10, 0x30 + m // 10]))
    extra = dict(time=time,
                 fwd=[b + step if b + step < nb else b for b in range(nb)],
                 back=[max(0, b - step) for b in range(nb)])
    for phases in (1, 2):
        try:
            fixed, finfo = build_fixed(rate_cyc, sp, vp, nb, autoplay, nt, bar_px,
                                       vis is not None, extra, phases)
            break
        except AssertionError as e:
            if phases == 2 or vis is None or 'vblank' not in str(e):
                raise
    finfo['phases'] = phases
    prg[(tot - 1) * BANK:] = fixed
    hdr = bytearray(b'NES\x1A') + bytes([tot & 0xFF, 1, 0x20, 0x00, 0, tot >> 8, 0, 0, 0, 0, 0, 0])
    if tot > 255:
        hdr[7] = 0x08                      # NES 2.0
    rom = bytes(hdr) + bytes(prg) + make_chr()
    finfo.update(banks=nb, total_banks=tot, vis_pages=vp, rate=real_rate)
    return rom, finfo


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('mp3'); ap.add_argument('rom')
    ap.add_argument('--rate', type=int, default=0,
                    help='sample rate in Hz (default: highest that fits in 4 MB, max %d)' % MAX_RATE)
    ap.add_argument('--title'); ap.add_argument('--start', type=float, default=0)
    ap.add_argument('--duration', type=float, default=0)
    ap.add_argument('--gain', type=int, default=0, help='dB of boost before a peak limiter')
    ap.add_argument('--shape', type=float, default=0.5,
                    help='noise-shaping feedback 0..1 (pushes quantisation noise upward)')
    ap.add_argument('--no-vis', action='store_true', help='omit the LED visualizer data')
    ap.add_argument('--autoplay', action='store_true', help='start playing at power-on')
    ap.add_argument('--wav', help='also write a preview WAV of the quantised audio')
    args = ap.parse_args()

    title = (args.title or os.path.splitext(os.path.basename(args.mp3))[0]).upper()
    title = ''.join(c if 32 <= ord(c) < 127 else ' ' for c in title)

    def fit(cyc):
        return CPU_HZ / float(cyc)
    if args.rate:
        cyc = max(C_MIN, int(round(CPU_HZ / float(args.rate))))
    else:
        cyc = C_MIN
        probe = sms.decode(args.mp3, 8000, args.start, args.duration)
        secs = len(probe) / 8000.0
        while True:
            vpg = 0 if args.no_vis else 1
            per = (64 - vpg) * 256
            if secs * fit(cyc) + 1 <= (MAX_BANKS - 1) * per or cyc > 400:
                break
            cyc += 1
    rate = int(round(fit(cyc)))
    print('rate %d Hz (%d cycles/sample)' % (rate, cyc))
    pcm = sms.decode(args.mp3, rate, args.start, args.duration, args.gain)
    print('%d samples (%.0f s)' % (len(pcm), len(pcm) / float(rate)))
    q = quantize(list(pcm), args.shape)
    secs = len(q) / float(rate)
    vis = None
    if not args.no_vis:
        from audiorom.genesis import compute_vis
        print('computing LED-bar levels ...')
        vis = compute_vis(list(pcm), rate, int(math.ceil(secs)) + 2)
    info = '%d HZ  7-BIT  MONO' % rate
    rom, fi = build_rom(q, cyc, title, secs, info, vis, args.autoplay)
    with open(args.rom, 'wb') as f:
        f.write(rom)
    print('wrote %s (%d KB, %d sample banks, %d slots, PPU done at %d cyc, %d-phase bars)' % (
        args.rom, len(rom) // 1024, fi['banks'], fi['slots'], fi['ppu_end'], fi['phases']))
    if args.wav:
        import wave
        lv = dmc_levels()
        out = array.array('h', (int((lv[b] - 0.5) * 2 * 30000) for b in q))
        w = wave.open(args.wav, 'wb')
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate)
        w.writeframes(out.tobytes()); w.close()


if __name__ == '__main__':
    main()
