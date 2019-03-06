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

import time
import wave
import select
import socket
import audioop
import logging
import threading
import traceback

from . import rtp
from .utils import Defaultdict
from .rtp import SilencePacket
from .opus import Decoder, BufferedDecoder

try:
    import nacl.secret
    from nacl.exceptions import CryptoError
except ImportError:
    pass

log = logging.getLogger(__name__)

__all__ = [
    'AudioSink',
    'WaveSink',
    'PCMVolumeTransformerFilter',
    'ConditionalFilter',
    'TimedFilter',
    'UserFilter',
]

class AudioSink:
    def __del__(self):
        self.cleanup()

    def write(self, data):
        raise NotImplementedError

    def wants_opus(self):
        return False

    def cleanup(self):
        pass

class WaveSink(AudioSink):
    def __init__(self, destination):
        self._file = wave.open(destination, 'wb')
        self._file.setnchannels(Decoder.CHANNELS)
        self._file.setsampwidth(Decoder.SAMPLE_SIZE//Decoder.CHANNELS)
        self._file.setframerate(Decoder.SAMPLING_RATE)

    def write(self, data):
        self._file.writeframes(data.data)

    def cleanup(self):
        try:
            self._file.close()
        except:
            pass

class PCMVolumeTransformerFilter(AudioSink):
    def __init__(self, destination, volume=1.0):
        if not isinstance(destination, AudioSink):
            raise TypeError('expected AudioSink not {0.__class__.__name__}.'.format(destination))

        if destination.wants_opus():
            raise ClientException('AudioSink must not request Opus encoding.')

        self.destination = destination
        self.volume = volume

    @property
    def volume(self):
        """Retrieves or sets the volume as a floating point percentage (e.g. 1.0 for 100%)."""
        return self._volume

    @volume.setter
    def volume(self, value):
        self._volume = max(value, 0.0)

    def write(self, data):
        data = audioop.mul(data.data, 2, min(self._volume, 2.0))
        self.destination.write(data)

# I need some sort of filter sink with a predicate or something
# Which means I need to sort out the write() signature issue
# Also need something to indicate a sink is "done", probably
# something like raising an exception and handling that in the write loop
# Maybe should rename some of these to Filter instead of Sink

class ConditionalFilter(AudioSink):
    def __init__(self, destination, predicate):
        self.destination = destination
        self._predicate = predicate

    def write(self, data):
        if self._predicate(data):
            self.destination.write(data)

class TimedFilter(ConditionalFilter):
    def __init__(self, destination, duration):
        super().__init__(destination, self._predicate)
        self.duration = duration

    def _predicate(self, data):
        return # TODO: return elapsed < duration

class UserFilter(ConditionalFilter):
    def __init__(self, destination, user):
        super().__init__(destination, self._predicate)
        self.user = user

    def _predicate(self, data):
        return data.user == self.user

# rename 'data' to 'payload'? or 'opus'? something else?
class VoiceData:
    __slots__ = ('data', 'user', 'packet')

    def __init__(self, data, user, packet):
        self.data = data
        self.user = user
        self.packet = packet

class AudioReader(threading.Thread):
    def __init__(self, sink, client):
        threading.Thread.__init__(self)
        self.daemon = True
        self._sink = sink
        self.client = client

        self._box = nacl.secret.SecretBox(bytes(client.secret_key))
        self._decrypt_rtp = getattr(self, '_decrypt_rtp_' + client._mode)
        self._decrypt_rtcp = getattr(self, '_decrypt_rtcp_' + client._mode)

        self._connected = client._connected
        self._current_error = None
        self._buffers = Defaultdict(lambda ssrc: BufferedDecoder(ssrc, self._write_to_sink))

        self._end = threading.Event()
        self._decoder_lock = threading.Lock()

    def stop(self, *, wait=False):
        self._end.set()
        if wait:
            self.join()

    def _decrypt_rtp_xsalsa20_poly1305(self, packet):
        nonce = bytearray(24)
        nonce[:12] = packet.header
        result = self._box.decrypt(bytes(packet.data), bytes(nonce))

        if packet.extended:
            offset = packet.update_ext_headers(result)
            result = result[offset:]

        return result

    def _decrypt_rtcp_xsalsa20_poly1305(self, data):
        nonce = bytearray(24)
        nonce[:8] = data[:8]
        result = self._box.decrypt(data[8:], bytes(nonce))

        return data[:8] + result

    def _decrypt_rtp_xsalsa20_poly1305_suffix(self, packet):
        nonce = packet.data[-24:]
        voice_data = packet.data[:-24]
        result = self._box.decrypt(bytes(voice_data), bytes(nonce))

        if packet.extended:
            offset = packet.update_ext_headers(result)
            result = result[offset:]

        return result

    def _decrypt_rtcp_xsalsa20_poly1305_suffix(self, data):
        nonce = data[-24:]
        header = data[:8]
        result = self._box.decrypt(data[8:-24], nonce)

        return header + result

    def _decrypt_rtp_xsalsa20_poly1305_lite(self, packet):
        nonce = bytearray(24)
        nonce[:4] = packet.data[-4:]
        voice_data = packet.data[:-4]
        result = self._box.decrypt(bytes(voice_data), bytes(nonce))

        if packet.extended:
            offset = packet.update_ext_headers(result)
            result = result[offset:]

        return result

    def _decrypt_rtcp_xsalsa20_poly1305_lite(self, data):
        nonce = bytearray(24)
        nonce[:4] = data[-4:]
        header = data[:8]
        result = self._box.decrypt(data[8:-4], bytes(nonce))

        return header + result

    def _reset_decoders(self, *ssrcs):
        with self._decoder_lock:
            if not ssrcs:
                for decoder in self._buffers.values():
                    decoder.reset()
            else:
                for ssrc in ssrcs:
                    d = self._buffers.get(ssrc)
                    if d:
                        d.reset()

    def _ssrc_removed(self, ssrc):
        # An user has disconnected but there still may be
        # packets from them left in the buffer to read
        # For now we're just going to kill the decoder and see how that works out
        # I *think* this is the correct way to do this
        # Depending on how many leftovers I end up with I may reconsider

        with self._decoder_lock:
            decoder = self._buffers.pop(ssrc, None)

            if decoder is None:
                print(f"!!! No decoder for ssrc {ssrc} was found?")
            else:
                decoder.stop() # flush?
                # if decoder._buffer:
                    # print(f"Decoder had {len(decoder._buffer)} packets remaining")

    def _get_user(self, packet):
        return self.client._ssrcs.get(packet.ssrc)

    def _write_to_sink(self, pcm, opus, packet):
        try:
            data = opus if self._sink.wants_opus() else pcm
            user = self._get_user(packet)
            self._sink.write(VoiceData(data, user, packet))
        except:
            traceback.print_exc()
            # insert optional error handling here

    def _do_run(self):
        print("Starting socket loop")
        while not self._end.is_set():
            if not self._connected.is_set():
                self._connected.wait()

            ready, _, err = select.select([self.client.socket], [],
                                          [self.client.socket], 0.1)
            if not ready:
                if err:
                    print("Socket error")
                continue

            try:
                raw_data = self.client.socket.recv(4096)
            except socket.error as e:
                t0 = time.time()

                if e.errno == 10038:
                    continue

                print(f"Socket error in reader thread: {e} {t0}")

                with self.client._connecting:
                    timed_out = self.client._connecting.wait(20)

                if not timed_out:
                    raise
                elif self.client.is_connected():
                    print(f"Reconnected in {time.time()-t0:.4f}s")
                    continue
                else:
                    raise

            try:
                packet = None

                if not rtp.is_rtcp(raw_data):
                    packet = rtp.decode(raw_data)
                    packet.decrypted_data = self._decrypt_rtp(packet)
                else:
                    packet = rtp.decode(self._decrypt_rtcp(raw_data))
                    if not isinstance(packet, rtp.ReceiverReportPacket):
                        print(packet)
                    continue # oh right I don't have any way to send these yet

            except CryptoError:
                log.exception("CryptoError decoding packet %s", packet)
                continue

            except:
                log.exception("Error decoding packet")
                traceback.print_exc()

            else:
                # Do I hold these packets or drop them?
                if packet.ssrc not in self.client._ssrcs:
                    log.debug("Unknown user for ssrc %s", packet.ssrc)

                    # TODO
                    # As expected, this is racy with users disconnecting
                    # I'm not sure what to do about this yet
                    # The problem is this is a 2-part problem
                    # I can capture packets for processing before I get
                    # an ssrc-userid mapping and thats fine, but it also
                    # recreates the decoder when there are leftover packets
                    # after someone disconnects.
                    # The RTCP timestamp offset i've been wishing for would fix this

                self._buffers[packet.ssrc].feed_rtp(packet)

        # flush decoders?

    def run(self):
        try:
            self._do_run()
        except socket.error as e:
            self.stop()
        except Exception as e:
            traceback.print_exc()
            self._current_error = e
            self.stop()
        finally:
            for decoder in list(self._buffers.values()):
                decoder.stop()
            try:
                self._sink.cleanup()
            except:
                # Testing only
                traceback.print_exc()
