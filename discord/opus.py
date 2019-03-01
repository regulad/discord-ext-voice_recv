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

# class OpusRouter:
#     """
#         timestamp delta should be decoder.SAMPLES_PER_FRAME

#         seq delta should be 1 for normal packets,
#         5 between client added silence from speaking state change
#         * I think, something about that doesn't exactly make sense *
#     """

#     def __init__(self, output_func, *, buffer=200):
#         self.output_func = output_func
#         self.buffer_size = buffer // 20
#         self.rtpheap = []

#         # This is how many packets the router waits to pump silence (in 20ms chunks)
#         # higher values will wait longer and pump in larger chunks
#         self.silence_threshold = 10

#         self.last_packet_seq = 0
#         self.last_packet_recv = 0
#         self.last_decode_seq = 0
#         self.last_packet_was_silence = False

#         self._decoder = Decoder()
#         self.ssrc = None

#     def feed(self, packet):
#         self.ssrc = packet.ssrc # dumb hack

#         if not self.rtpheap or packet.sequence > self.rtpheap[0].sequence:
#             self.last_packet_seq = packet.sequence
#             self.last_packet_recv = time.time()

#             heapq.heappush(self.rtpheap, packet)
#         else:
#             print("Rejected packet %s < %s" % (
#                 packet.sequence, self.rtpheap[0].sequence))

#     def flush_packets(self):
#         self._flush(len(self.rtpheap))

#     def flush_half(self):
#         self._flush(self.buffer_size//2)

#     def _flush(self, count):
#         for x in range(min(len(self.rtpheap), count)):
#             packet = heapq.heappop(self.rtpheap)
#             self._add_silence(packet)
#             self.decode(packet)

#     def _add_silence(self, packet):
#         gap = packet.sequence - self.last_decode_seq
#         if gap >= 5 and self.last_decode_seq:
#             # print(f"Adding in {gap} silence frames before decode")
#             for x in range(gap):
#                 self.decode(None)

#     def decode(self, packet):
#         opus = None
#         if packet:
#             # print("Decoding packet %s" % packet.sequence)
#             opus = packet.decrypted_data
#             if self.last_packet_was_silence:
#                 ...
#         else:
#             packet = SilencePacket(self.ssrc, self.last_decode_seq)
#             self.last_packet_was_silence = True
#             # TODO: Test incrementing seq in filler silence packets

#         pcm = self._decoder.decode(opus)
#         self.last_decode_seq += 1

#         opus = packet.decrypted_data
#         self.output_func(pcm, opus, packet)

#     def reset(self, *, flush=True):
#         # TODO optimization: check if the decoder /needs/ to be reset
#         if flush:
#             self.flush_packets()

#         self.rtpheap.clear()

#         for x in range(10):
#             self._decoder.decode(None)

#         self.last_packet_seq = 0
#         self.last_packet_recv = 0
#         self.last_decode_seq = 0

#     def notify(self):
#         if not self.last_packet_recv:
#             # print("Warning: no packet received yet")
#             return

#         self.flush_half()

#         # - 1 so there's leftover time to reduce rounding related issues
#         gap = time.time() - self.last_packet_recv - 1
#         missing = int(round(gap / 0.02, 0))

#         if missing <= self.silence_threshold:
#             return
#         elif missing > 10000:
#             print(f"Missing a LOT of frames for {self.ssrc}: {missing} (last: {self.last_packet_recv})")
#             # something weird can happen where you get a fuckton of
#             # missing frames from the calculation, not sure exactly
#             # how to handle this yet
#             return

#         # print("Missing %s, flushing %s" % (missing, len(self.rtpheap)))
#         self.flush_packets()

#         for x in range(missing):
#             self.decode(None)

#         # Inflate stats to account for silence
#         self.last_packet_seq += missing
#         self.last_packet_recv += gap
#         self.last_decode_seq += missing


class OpusRouter(threading.Thread):
    DELAY = Decoder.FRAME_LENGTH / 1000.0

    def __init__(self, output_func, *, buffer=200):
        super().__init__(daemon=True)

        self.output_func = output_func
        self.ssrc = 0

        self._decoder = Decoder()
        self.cycle_time = 20 # ms
        self.last_seq = 0
        self.last_ts = 0

        # Optional diagnostic state stuff
        self._overflow_mult = self._overflow_base = 2.0
        self._overflow_incr = 0.5

        # minimum (lower bound) size of the jitter buffer (n * 20ms per packet)
        self.buffer_size = buffer // self._decoder.FRAME_LENGTH

        self._end = threading.Event()
        self._primed = threading.Event()
        self._lock = threading.RLock()
        self._buffer = []

        # TODO: Add RTCP queue

        self.start() # see feed() comment

    # see feed() comment
    @property
    def _name(self):
        return 'ssrc-{}'.format(self.ssrc or '?')

    @_name.setter
    def _name(self, _):
        pass

    def stop(self, *, flush=False):
        # Since this function can (usually is?) called from the websocket read loop,
        # it might not be a bad idea to return a future and set it when flushing is done

        if flush:
            ... # write out the rest of buffer (set delay to 0?)
        self._end.set()

    def _push(self, item):
        if not self._primed.is_set():
            self._primed.set()

        if not isinstance(item, RTPPacket):
            raise TypeError(f"item should be an RTPPacket, not {item.__class__.__name__}")

        # Fake packet loss
        # import random
        # if random.randint(1, 100) <= 10:
        #     return

        with self._lock:
            # Replace silence packets with rtp packets
            sp = utils.get(self._buffer, timestamp=item.timestamp)
            if isinstance(sp, SilencePacket):
                self._buffer[self._buffer.index(sp)] = item
                # print([f'<{hi.__class__.__name__[:3]} seq={hi.sequence}>' for hi in self._buffer])
                return
            # else:
                # ... # compare data sometime to see if its a dupe packet

            bisect.insort(self._buffer, item)
            # print([f'<{hi.__class__.__name__[0]} seq={hi.sequence}>' for hi in self._buffer])

        # Optional diagnostics
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

    # Do not worry about this function looking horrifyingly slow, its quite fast for small buffer sizes
    # Just kidding its worthless since silence packets DONT HAVE SEQUENCES
    # def _fill_silence(self, perc=1.0):
    #     return
    #     with self._lock:
    #         fillcount = min(self.buffer_size, self.buffer_size - self.fill_threshold - len(self._buffer))
    #         if fillcount <= self.fill_threshold:
    #             return
    #
    #         fillcount = int(fillcount/perc)
    #         filler = []
    #         last_seq = self._buffer[0].sequence
    #
    #         self._buffer.sort() # luckily timsort takes advantage of the partially sorted nature of heaps
    #         for index, packet in enumerate(self._buffer):
    #             if packet.sequence > last_seq + 1:
    #                 for x in range(packet.sequence - last_seq - 1):
    #                     filler.append(SilencePacket(self.ssrc, last_seq + x + 1))
    #                     fillcount -= 1
    #
    #                 last_seq = packet.sequence
    #                 if fillcount <= 0:
    #                     break
    #
    #             last_seq = packet.sequence
    #
    #         if fillcount:
    #             highest_seq = max(
    #                 self._peekat(-1).sequence if self._buffer else 0,
    #                 filler[-1].sequence if filler else 0
    #             )
    #             for x in range(fillcount):
    #                 filler.append(SilencePacket(self.ssrc, highest_seq + x + 1))
    #
    #         if filler:
    #             self._buffer = list(heapq.merge(self._buffer, filler))

    def _fill_silence(self):
        """I don't know if I need this function anymore but i'll have to check
        how the buffer fills up initially to maybe do at least one pass at the start.
        """
        pass

    def feed(self, packet):
        # dumb hack, alternative is a defaultdict subclass that passes
        # key from __missing__(self, key) to the factory function
        self.ssrc = packet.ssrc

        if self.last_ts < packet.timestamp:
            self._push(packet)

    def reset(self):
        # resetting the decoder does not pause the decoder... what to do...
        # lock?
        self._decoder = Decoder() # TODO: Add a reset function to Decoder itself
        self.last_seq = self.last_ts = 0
        self._buffer.clear()

    def _packet_gen(self):
        while True:
            packet, nextpacket = self._pop()
            self.last_ts = getattr(packet, 'timestamp', self.last_ts + Decoder.SAMPLES_PER_FRAME)
            self.last_seq += 1 # self.last_seq = packet.sequence?

            if isinstance(packet, RTPPacket):
                pcm = self._decoder.decode(packet.decrypted_data)

            elif isinstance(nextpacket, RTPPacket):
                pcm = self._decoder.decode(packet.decrypted_data, fec=True)
                fec_packet = FECPacket(self.ssrc, nextpacket.sequence - 1, nextpacket.timestamp - Decoder.SAMPLES_PER_FRAME)
                yield fec_packet, pcm

                packet, _ = self._pop()
                self.last_ts += Decoder.SAMPLES_PER_FRAME
                self.last_seq += 1

                pcm = self._decoder.decode(packet.decrypted_data)
            else:
                pcm = self._decoder.decode(None)

            yield packet, pcm

    # TODO: Add some way to reset to the start of this loop (another event for run() loop?)
    def _do_run(self):
        self._primed.wait()
        while len(self._buffer) < self.buffer_size:
            time.sleep(0.02)

        self._fill_silence()

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
