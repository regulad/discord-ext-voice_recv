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
import heapq
import struct
import os.path
import sys

from collections import namedtuple, defaultdict

from .rtp import SilencePacket
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

def _err_lt(result, func, args):
    if result < 0:
        log.info('error has happened in %s', func.__name__)
        raise OpusError(result)
    return result

def _err_ne(result, func, args):
    ret = args[-1]._obj
    if ret.value != 0:
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


# Some constants...
OK = 0
APPLICATION_AUDIO    = 2049
APPLICATION_VOIP     = 2048
APPLICATION_LOWDELAY = 2051
CTL_SET_BITRATE      = 4002
CTL_SET_BANDWIDTH    = 4008
CTL_SET_FEC          = 4012
CTL_SET_PLP          = 4014
CTL_SET_SIGNAL       = 4024

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
    def __init__(self, application=APPLICATION_VOIP):
        if not is_loaded():
            raise OpusNotLoaded()

        self.application = application
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

    def decode(self, data, *, fec=False):
        if data is None and fec:
            raise OpusError("Invalid arguments: FEC cannot be used with null data")

        if data is None:
            # You're supposed to use the data from the previous frame but w/e
            frame_size = self.SAMPLES_PER_FRAME
        else:
            frames = self.packet_get_nb_frames(data)
            samples_per_frame = self.packet_get_samples_per_frame(data)
            # channels = self.packet_get_nb_channels(data)
            frame_size = frames * samples_per_frame

        pcm_size = frame_size * self.CHANNELS
        pcm = (ctypes.c_int16 * pcm_size)()
        pcm_ptr = ctypes.cast(pcm, ctypes.POINTER(ctypes.c_int16))

        result = _lib.opus_decode(self._state, data, len(data) if data else 0, pcm_ptr, frame_size, fec)
        return array.array('h', pcm).tobytes()

class OpusRouter:
    """
        timestamp delta should be decoder.SAMPLES_PER_FRAME

        seq delta should be 1 for normal packets,
        5 between client added silence from speaking state change
        * I think, something about that doesn't exactly make sense *
    """

    def __init__(self, output_func, *, buffer=200):
        self.output_func = output_func
        self.buffer_size = buffer // 20
        self.rtpheap = []

        # This is how long the router waits to pump silence (in 20ms chunks)
        # higher values will wait longer and pump in larger chunks
        self.silence_threshold = 10

        self.last_packet_seq = 0
        self.last_packet_recv = 0
        self.last_decode_seq = 0

        self._decoder = Decoder()
        self.ssrc = None

    def feed(self, packet):
        self.ssrc = packet.ssrc # dumb hack

        if not self.rtpheap or packet.sequence > self.rtpheap[0].sequence:
            self.last_packet_seq = packet.sequence
            self.last_packet_recv = time.time()

            heapq.heappush(self.rtpheap, packet)
        else:
            print("Rejected packet %s < %s" % (
                packet.sequence, self.rtpheap[0].sequence))

    def flush_packets(self):
        self._flush(len(self.rtpheap))

    def flush_half(self):
        self._flush(self.buffer_size//2)

    def _flush(self, count):
        for x in range(min(len(self.rtpheap), count)):
            packet = heapq.heappop(self.rtpheap)
            self._add_silence(packet)
            self.decode(packet)

    def _add_silence(self, packet):
        gap = packet.sequence - self.last_decode_seq
        if gap >= 5 and self.last_decode_seq:
            # print(f"Adding in {gap} silence frames before decode")
            for x in range(gap):
                self.decode(None)

    def decode(self, packet):
        opus = None
        if packet:
            opus = packet.decrypted_data
            self.last_decode_seq = packet.sequence
            # print("Decoding packet %s" % packet.sequence)
        else:
            packet = SilencePacket(self.ssrc)
            # TODO: Test incrementing seq in filler silence packets

        pcm = self._decoder.decode(opus)
        opus = opus or b'\xF8\xFF\xFE'
        self.output_func(pcm, opus, packet)

    def reset(self, *, flush=True):
        # TODO optimization: check if the decoder /needs/ to be reset
        if flush:
            self.flush_packets()

        for x in range(10):
            self._decoder.decode(None)

        self.last_packet_seq = 0
        self.last_packet_recv = 0
        self.last_decode_seq = 0

    def notify(self):
        if not self.last_packet_recv:
            # print("Warning: no packet received yet")
            return

        self.flush_half()

        # - 1 so there's leftover time to reduce rounding related issues
        gap = time.time() - self.last_packet_recv - 1
        missing = int(round(gap / 0.02, 0))

        if missing <= self.silence_threshold:
            return
        elif missing > 10000:
            print(f"Missing a LOT of frames for {self.ssrc}: {missing} (last: {self.last_packet_recv})")
            # something weird can happen where you get a fuckton of
            # missing frames from the calculation, not sure exactly
            # how to handle this yet
            return

        # print("Missing %s, flushing %s" % (missing, len(self.rtpheap)))
        self.flush_packets()

        for x in range(missing):
            self.decode(None)

        # Inflate stats to account for silence
        self.last_packet_seq += missing
        self.last_packet_recv += gap
        self.last_decode_seq += missing
