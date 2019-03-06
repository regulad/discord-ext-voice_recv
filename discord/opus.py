# -*- coding: utf-8 -*-

"""
The MIT License (MIT)

Copyright (c) 2015-2019 Rapptz

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
"""

import array
import ctypes
import ctypes.util
import logging
import sys
import time
import struct
import os.path
import sys
import bisect
import threading
import traceback

from math import log10

from . import utils
from .rtp import RTPPacket, RTCPPacket, SilencePacket, FECPacket
from .errors import DiscordException


log = logging.getLogger(__name__)

c_int_ptr   = ctypes.POINTER(ctypes.c_int)
c_int16_ptr = ctypes.POINTER(ctypes.c_int16)
c_float_ptr = ctypes.POINTER(ctypes.c_float)

_lib = None

class EncoderStruct(ctypes.Structure):
    pass

class DecoderStruct(ctypes.Structure):
    pass

EncoderStructPtr = ctypes.POINTER(EncoderStruct)
DecoderStructPtr = ctypes.POINTER(DecoderStruct)

## Some constants from opus_defines.h
# Error codes
OK      = 0
BAD_ARG = -1

# Encoder CTLs
APPLICATION_AUDIO    = 2049
APPLICATION_VOIP     = 2048
APPLICATION_LOWDELAY = 2051

CTL_SET_BITRATE      = 4002
CTL_SET_BANDWIDTH    = 4008
CTL_SET_FEC          = 4012
CTL_SET_PLP          = 4014
CTL_SET_SIGNAL       = 4024

# Decoder CTLs
CTL_SET_GAIN             = 4034
CTL_LAST_PACKET_DURATION = 4039

band_ctl = {
    'narrow': 1101,
    'medium': 1102,
    'wide': 1103,
    'superwide': 1104,
    'full': 1105,
}

signal_ctl = {
    'auto': -1000,
    'voice': 3001,
    'music': 3002,
}

def _err_lt(result, func, args):
    if result < OK:
        log.info('error has happened in %s', func.__name__)
        raise OpusError(result)
    return result

def _err_ne(result, func, args):
    ret = args[-1]._obj
    if ret.value != OK:
        log.info('error has happened in %s', func.__name__)
        raise OpusError(ret.value)
    return result

# A list of exported functions.
# The first argument is obviously the name.
# The second one are the types of arguments it takes.
# The third is the result type.
# The fourth is the error handler.
exported_functions = [
    ('opus_strerror',
        [ctypes.c_int], ctypes.c_char_p, None),
    ('opus_packet_get_bandwidth',
        [ctypes.c_char_p], ctypes.c_int, _err_lt),
    ('opus_packet_get_nb_channels',
        [ctypes.c_char_p], ctypes.c_int, _err_lt),
    ('opus_packet_get_nb_frames',
        [ctypes.c_char_p, ctypes.c_int], ctypes.c_int, _err_lt),
    ('opus_packet_get_samples_per_frame',
        [ctypes.c_char_p, ctypes.c_int], ctypes.c_int, _err_lt),

    ('opus_encoder_get_size',
        [ctypes.c_int], ctypes.c_int, None),
    ('opus_encoder_create',
        [ctypes.c_int, ctypes.c_int, ctypes.c_int, c_int_ptr], EncoderStructPtr, _err_ne),
    ('opus_encode',
        [EncoderStructPtr, c_int16_ptr, ctypes.c_int, ctypes.c_char_p, ctypes.c_int32], ctypes.c_int32, _err_lt),
    ('opus_encoder_ctl',
        None, ctypes.c_int32, _err_lt),
    ('opus_encoder_destroy',
        [EncoderStructPtr], None, None),

    ('opus_decoder_get_size',
        [ctypes.c_int], ctypes.c_int, None),
    ('opus_decoder_create',
        [ctypes.c_int, ctypes.c_int, c_int_ptr], DecoderStructPtr, _err_ne),
    ('opus_decoder_get_nb_samples',
        [DecoderStructPtr, ctypes.c_char_p, ctypes.c_int32], ctypes.c_int, _err_lt),
    ('opus_decode',
        [DecoderStructPtr, ctypes.c_char_p, ctypes.c_int32, c_int16_ptr, ctypes.c_int, ctypes.c_int],
        ctypes.c_int, _err_lt),
    ('opus_decoder_ctl',
        None, ctypes.c_int32, _err_lt),
    ('opus_decoder_destroy',
        [DecoderStructPtr], None, None)
]

def libopus_loader(name):
    # create the library...
    lib = ctypes.cdll.LoadLibrary(name)

    # register the functions...
    for item in exported_functions:
        func = getattr(lib, item[0])

        try:
            if item[1]:
                func.argtypes = item[1]

            func.restype = item[2]
        except KeyError:
            pass

        try:
            if item[3]:
                func.errcheck = item[3]
        except KeyError:
            log.info("Error assigning check function to %s", item[0])

    return lib

def _load_default():
    global _lib
    try:
        if sys.platform == 'win32':
            _basedir = os.path.dirname(os.path.abspath(__file__))
            _bitness = 'x64' if sys.maxsize > 2**32 else 'x86'
            _filename = os.path.join(_basedir, 'bin', 'libopus-0.{}.dll'.format(_bitness))
            _lib = libopus_loader(_filename)
        else:
            _lib = libopus_loader(ctypes.util.find_library('opus'))
    except Exception as e:
        _lib = None
        log.warning("Unable to load opus lib, %s", e)

    return _lib is not None

def load_opus(name):
    """Loads the libopus shared library for use with voice.

    If this function is not called then the library uses the function
    :func:`ctypes.util.find_library` and then loads that one if available.

    Not loading a library and attempting to use PCM based AudioSources will
    lead to voice not working.

    This function propagates the exceptions thrown.

    .. warning::

        The bitness of the library must match the bitness of your python
        interpreter. If the library is 64-bit then your python interpreter
        must be 64-bit as well. Usually if there's a mismatch in bitness then
        the load will throw an exception.

    .. note::

        On Windows, this function should not need to be called as the binaries
        are automatically loaded.

    .. note::

        On Windows, the .dll extension is not necessary. However, on Linux
        the full extension is required to load the library, e.g. ``libopus.so.1``.
        On Linux however, :func:`ctypes.util.find_library` will usually find the library automatically
        without you having to call this.

    Parameters
    ----------
    name: :class:`str`
        The filename of the shared library.
    """
    global _lib
    _lib = libopus_loader(name)

def is_loaded():
    """Function to check if opus lib is successfully loaded either
    via the :func:`ctypes.util.find_library` call of :func:`load_opus`.

    This must return ``True`` for voice to work.

    Returns
    -------
    :class:`bool`
        Indicates if the opus library has been loaded.
    """
    global _lib
    return _lib is not None

class OpusError(DiscordException):
    """An exception that is thrown for libopus related errors.

    Attributes
    ----------
    code: :class:`int`
        The error code returned.
    """

    def __init__(self, code):
        self.code = code
        msg = _lib.opus_strerror(self.code).decode('utf-8')
        log.info('"%s" has happened', msg)
        super().__init__(msg)

class OpusNotLoaded(DiscordException):
    """An exception that is thrown for when libopus is not loaded."""
    pass

class _OpusStruct:
    SAMPLING_RATE = 48000
    CHANNELS = 2
    FRAME_LENGTH = 20 # in ms
    SAMPLE_SIZE = 4 # (bit_rate / 8) * CHANNELS (bit_rate == 16)
    SAMPLES_PER_FRAME = int(SAMPLING_RATE / 1000 * FRAME_LENGTH)

    FRAME_SIZE = SAMPLES_PER_FRAME * SAMPLE_SIZE

class Encoder(_OpusStruct):
    def __init__(self, application=APPLICATION_AUDIO):
        if not is_loaded():
            if not _load_default():
                raise OpusNotLoaded()

        self.application = application
        self._state = self._create_state()
        self.set_bitrate(128)
        self.set_fec(True)
        self.set_expected_packet_loss_percent(0.15)
        self.set_bandwidth('full')
        self.set_signal_type('auto')

    def __del__(self):
        if hasattr(self, '_state'):
            _lib.opus_encoder_destroy(self._state)
            self._state = None

    def _create_state(self):
        ret = ctypes.c_int()
        return _lib.opus_encoder_create(self.SAMPLING_RATE, self.CHANNELS, self.application, ctypes.byref(ret))

    def set_bitrate(self, kbps):
        kbps = min(512, max(16, int(kbps)))

        _lib.opus_encoder_ctl(self._state, CTL_SET_BITRATE, kbps * 1024)
        return kbps

    def set_bandwidth(self, req):
        if req not in band_ctl:
            raise KeyError('%r is not a valid bandwidth setting. Try one of: %s' % (req, ','.join(band_ctl)))

        k = band_ctl[req]
        _lib.opus_encoder_ctl(self._state, CTL_SET_BANDWIDTH, k)

    def set_signal_type(self, req):
        if req not in signal_ctl:
            raise KeyError('%r is not a valid signal setting. Try one of: %s' % (req, ','.join(signal_ctl)))

        k = signal_ctl[req]
        _lib.opus_encoder_ctl(self._state, CTL_SET_SIGNAL, k)

    def set_fec(self, enabled=True):
        _lib.opus_encoder_ctl(self._state, CTL_SET_FEC, 1 if enabled else 0)

    def set_expected_packet_loss_percent(self, percentage):
        _lib.opus_encoder_ctl(self._state, CTL_SET_PLP, min(100, max(0, int(percentage * 100))))

    def encode(self, pcm, frame_size):
        max_data_bytes = len(pcm)
        pcm = ctypes.cast(pcm, c_int16_ptr)
        data = (ctypes.c_char * max_data_bytes)()

        ret = _lib.opus_encode(self._state, pcm, frame_size, data, max_data_bytes)

        return array.array('b', data[:ret]).tobytes()

class Decoder(_OpusStruct):
    def __init__(self):
        if not is_loaded():
            raise OpusNotLoaded()

        self._state = self._create_state()

    def __del__(self):
        if hasattr(self, '_state'):
            _lib.opus_decoder_destroy(self._state)
            self._state = None

    def _create_state(self):
        ret = ctypes.c_int()
        return _lib.opus_decoder_create(self.SAMPLING_RATE, self.CHANNELS, ctypes.byref(ret))

    @staticmethod
    def packet_get_nb_frames(data):
        """Gets the number of frames in an Opus packet"""
        return _lib.opus_packet_get_nb_frames(data, len(data))

    @staticmethod
    def packet_get_nb_channels(data):
        """Gets the number of channels in an Opus packet"""
        return _lib.opus_packet_get_nb_channels(data)

    @classmethod
    def packet_get_samples_per_frame(cls, data):
        """Gets the number of samples per frame from an Opus packet"""
        return _lib.opus_packet_get_samples_per_frame(data, cls.SAMPLING_RATE)

    def _set_gain(self, adjustment):
        """Configures decoder gain adjustment.
        Scales the decoded output by a factor specified in Q8 dB units.
        This has a maximum range of -32768 to 32767 inclusive, and returns
        OPUS_BAD_ARG (-1) otherwise. The default is zero indicating no adjustment.
        This setting survives decoder reset (irrelevant for now).

        gain = 10**x/(20.0*256)

        (from opus_defines.h)
        """
        return _lib.opus_decoder_ctl(self._state, CTL_SET_GAIN, adjustment)

    def set_gain(self, dB):
        """Sets the decoder gain in dB, from -128 to 128."""

        dB_Q8 = max(-32768, min(32767, round(dB*256))) # dB * 2^n where n is 8 (Q8)
        return self._set_gain(dB_Q8)

    def set_volume(self, mult):
        """Sets the output volume as a float percent, i.e. 0.5 for 50%, 1.75 for 175%, etc."""
        return self.set_gain(20*log10(mult)) # amplitude ratio

    def _get_last_packet_duration(self):
        """Gets the duration (in samples) of the last packet successfully decoded or concealed."""

        ret = ctypes.c_int32()
        _lib.opus_decoder_ctl(self._state, CTL_LAST_PACKET_DURATION, ctypes.byref(ret))
        return ret.value

    def decode(self, data, *, fec=False):
        if data is None and fec:
            raise OpusError("Invalid arguments: FEC cannot be used with null data")

        if data is None:
            frame_size = self._get_last_packet_duration() or self.SAMPLES_PER_FRAME
        else:
            frames = self.packet_get_nb_frames(data)
            samples_per_frame = self.packet_get_samples_per_frame(data)
            frame_size = frames * samples_per_frame

        pcm = (ctypes.c_int16 * (frame_size * self.CHANNELS))()
        pcm_ptr = ctypes.cast(pcm, ctypes.POINTER(ctypes.c_int16))

        result = _lib.opus_decode(self._state, data, len(data) if data else 0, pcm_ptr, frame_size, fec)
        return array.array('h', pcm).tobytes()

class BufferedDecoder(threading.Thread):
    DELAY = Decoder.FRAME_LENGTH / 1000.0

    def __init__(self, ssrc, output_func, *, buffer=200):
        super().__init__(daemon=True, name='ssrc-%s' % ssrc)

        self.output_func = output_func
        self.ssrc = ssrc

        self._decoder = Decoder()
        self._buffer = []
        self._last_seq = 0
        self._last_ts = 0

        # Optional diagnostic state stuff
        self._overflow_mult = self._overflow_base = 2.0
        self._overflow_incr = 0.5

        # minimum (lower bound) size of the jitter buffer (n * 20ms per packet)
        self.buffer_size = buffer // self._decoder.FRAME_LENGTH

        self._end = threading.Event()
        self._primed = threading.Event()
        self._lock = threading.RLock()

        # TODO: Add RTCP queue


        self.start()

    def stop(self, *, flush=False):
        # Since this function can (usually is?) called from the websocket read loop,
        # it might not be a bad idea to return a future and set it when flushing is done

        if flush:
            ... # write out the rest of buffer (set delay to 0?)
        self._end.set()

    def _push(self, item):
        if not self._primed.is_set():
            self._primed.set()

        if not isinstance(item, (RTPPacket, SilencePacket)):
            raise TypeError(f"item should be an RTPPacket, not {item.__class__.__name__}")

        # Fake packet loss
        # import random
        # if random.randint(1, 100) <= 10 and isinstance(item, RTPPacket):
        #     return

        with self._lock:
            existing_packet = utils.get(self._buffer, timestamp=item.timestamp)
            if isinstance(existing_packet, SilencePacket):
                # Replace silence packets with rtp packets
                self._buffer[self._buffer.index(existing_packet)] = item
                return
            elif isinstance(existing_packet, RTPPacket):
                return # duplicate packet

            bisect.insort(self._buffer, item)

        # Optional diagnostics, will probably remove later
            bufsize = len(self._buffer) # indent intentional
        if bufsize >= self.buffer_size * self._overflow_mult:
            print(f"[router:push] Warning: rtp heap size has grown to {bufsize}")
            self._overflow_mult += self._overflow_incr

        elif bufsize <= self.buffer_size * (self._overflow_mult - self._overflow_incr) \
            and self._overflow_mult > self._overflow_base:

            print(f"[router:push] Info: rtp heap size has shrunk to {bufsize}")
            self._overflow_mult = max(self._overflow_base, self._overflow_mult - self._overflow_incr)

    def _pop(self):
        # print(f"[router:pop] removing packet from heap ({len(self._buffer)})")
        with self._lock:
            if self._buffer:
                self._buffer.append(SilencePacket(self.ssrc, self._buffer[-1].timestamp + Decoder.SAMPLES_PER_FRAME))
                return self._buffer.pop(0), self._buffer[0] if self._buffer else None
            else:
                raise RuntimeError("rtp buffer is empty WHY HAS THIS HAPPENED?")

    def _initial_fill(self):
        """Artisanal hand-crafted function for buffering packets and clearing discord's stupid fucking rtp buffer."""

        # Very small sleep to check if there's buffered packets
        time.sleep(0.001)
        if len(self._buffer) > 3:
            # looks like there's some old packets in the buffer
            # we need to figure out where the old packets stop and where the fresh ones begin
            # for that we need to see when we return to the normal packet accumulation rate

            with self._lock:
                last_size = len(self._buffer)

            # wait until we have the correct rate of packet ingress
            while len(self._buffer) - last_size > 1:
                with self._lock:
                    last_size = len(self._buffer)
                time.sleep(0.001)

            # collect some fresh packets
            time.sleep(0.06)

            # generate list of differences between packet sequences
            with self._lock:
                diffs = [self._buffer[i+1].sequence-self._buffer[i].sequence for i in range(len(self._buffer)-1)]
            sdiffs = sorted(diffs, reverse=True)

            # decide if there's a jump
            jump1, jump2 = sdiffs[:2]
            if jump1 > jump2*3:
                # remove the stale packets and keep the fresh ones
                self.truncate(size=len(self._buffer[diffs.index(jump1)+1:]))
            else:
                # otherwise they're all stale, dump 'em
                with self._lock:
                    self._buffer.clear()

        # fill buffer to at least half full
        while len(self._buffer) < self.buffer_size // 2:
            time.sleep(0.001)

        # fill the buffer with silence aligned with the first packet
        # if an rtp packet already exists for the given silence packet ts, the silence packet is ignored
        start_ts = self._buffer[0].timestamp
        for x in range(1, 1 + self.buffer_size - len(self._buffer)):
            with self._lock:
                self._push(SilencePacket(self.ssrc, start_ts + x * Decoder.SAMPLES_PER_FRAME))

        # now fill the rest
        while len(self._buffer) < self.buffer_size:
            time.sleep(0.001)

    def feed_rtp(self, packet):
        if self._last_ts < packet.timestamp:
            self._push(packet)

    def reset(self):
        # resetting the decoder does not pause the decoder... what to do...
        # lock?
        self._decoder = Decoder() # TODO: Add a reset function to Decoder itself
        self._last_seq = self._last_ts = 0
        self._buffer.clear()

    def truncate(self, *, size=None):
        """Discards old data to shrink buffer back down to buffer_size."""

        size = self.buffer_size if size is None else size
        with self._lock:
            self._buffer = self._buffer[-size:]

    def _packet_gen(self):
        while True:
            packet, nextpacket = self._pop()
            self._last_ts = getattr(packet, 'timestamp', self._last_ts + Decoder.SAMPLES_PER_FRAME)
            self._last_seq += 1 # self._last_seq = packet.sequence?

            if isinstance(packet, RTPPacket):
                pcm = self._decoder.decode(packet.decrypted_data)

            elif isinstance(nextpacket, RTPPacket):
                pcm = self._decoder.decode(packet.decrypted_data, fec=True)
                fec_packet = FECPacket(self.ssrc, nextpacket.sequence - 1, nextpacket.timestamp - Decoder.SAMPLES_PER_FRAME)
                yield fec_packet, pcm

                packet, _ = self._pop()
                self._last_ts += Decoder.SAMPLES_PER_FRAME
                self._last_seq += 1

                pcm = self._decoder.decode(packet.decrypted_data)
            else:
                pcm = self._decoder.decode(None)

            yield packet, pcm

    # TODO: Add some way to reset to the start of this loop (another event for run() loop?)
    def _do_run(self):
        self._primed.wait()
        self._initial_fill()

        start_time = time.perf_counter()
        packet_gen = self._packet_gen()
        loops = 0
        try:
            while not self._end.is_set():
                packet, pcm = next(packet_gen)
                self.output_func(pcm, packet.decrypted_data, packet)

                # TODO: if we fall below a certain threshold on buffer_size
                #       somehow reduce delay for a short time until we recover

                next_time = start_time + self.DELAY * loops
                loops += 1

                t = time.perf_counter()

                # print(f"delay: {self.DELAY}, loop:{loops: 4}, start: {start_time:.9f}, time: {t:.9f}, elapsed: {t-start_time:.9f}, next: {next_time:.9f} sleep_for: {max(0, self.DELAY + (next_time - t)):.9f}")
                time.sleep(max(0, self.DELAY + (next_time - time.perf_counter())))
        finally:
            packet_gen.close()

    def run(self):
        try:
            self._do_run()
        except Exception as e:
            traceback.print_exc()
